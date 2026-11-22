"""FastAPI service — Phase 2, Day 6.

Endpoints:
    POST /token   -> issue a JWT (OAuth2 password form): the shared demo account, or a named
                     account from the user store (sentinel.users), optionally narrowed with `scope`.
    GET  /me      -> the token's account and scopes.
    POST /query   -> Answer, SSE-streamed tokens; JWT-protected; rate-limited.
    POST /feedback -> a thumbs up/down (and optional comment) on one answer, by its answer_id.
    GET  /healthz -> liveness (Cloud Run's probes call it inside the container).
    GET  /health  -> the same, for checks from outside: Cloud Run's front end reserves paths
                     ending in "z" and answers /healthz with its own 404.

Production hygiene: JWT auth (FR-S3), slowapi per-IP rate limit (FR-S4, a test proves it
fires), structured per-stage-latency logging (FR-S5), Pydantic validation everywhere (FR-S6).

Guardrails (v2), applied here in the service layer so every engine gets them:
  * the question is screened first (sentinel.guardrails.input); a flagged one is refused with a
    fixed message and never reaches retrieval or the model;
  * retrieval is scoped to the document scopes in the caller's JWT (v2-R4), and quarantined
    chunks are never retrieved (v2-R1);
  * PII is redacted from retrieved passages before generation (v2-R2);
  * the generation prompt fences passages as untrusted data (sentinel.generate, v2-R1);
  * the token stream passes through the output sanitiser — untrusted links, images and PII
    never reach the client — and citations are checked at the end (v2-R3);
  * the finished answer is checked for carried-out passage instructions
    (sentinel.guardrails.answer), and a flagged answer is withheld. By default the answer is
    held until the check passes: the stream is `retrieved` -> `checking` -> `token`* (the
    answer's own tokens, or the withheld message alone) -> `done`, so a withheld answer never
    reaches the client.
    With config.answer_check_blocks off, tokens stream as generated and the `done` event's text
    replaces them; clients must show it instead.
The `done` event carries a GuardrailReport of everything that was changed or flagged; its text
is always the final answer.

The retrieval+generation pipeline is injected as a `QueryEngine` dependency so tests can swap
in an instant fake — that keeps the auth/rate-limit/contract tests hermetic (no models, no API).

Run:  python -m sentinel.serve   (or: uvicorn sentinel.serve:app)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from sentinel.cache import AnswerCache, Entry, RedisAnswerCache, context_key, redacted
from sentinel.config import IndexPaths, settings
from sentinel.generate import (
    HARDENING_REMINDER,
    HARDENING_RULE,
    SYSTEM_PROMPT,
    generate,
    is_abstention,
    parse_citations,
)
from sentinel.guardrails.answer import WITHHELD_MESSAGE, AnswerCheck, check_answer
from sentinel.guardrails.input import REFUSAL_MESSAGE, InputScreening, screen_question
from sentinel.guardrails.output import (
    OFF_TOPIC_MESSAGE,
    UNGROUNDED_MESSAGE,
    OutputReport,
    StreamSanitizer,
    check_citations,
)
from sentinel.guardrails.pii import redact, redact_chunks
from sentinel.logging_config import configure_logging, log
from sentinel.plan import Plan
from sentinel.plan import plan as plan_question
from sentinel.retrieve import Access, hybrid_retrieve_timed, multi_retrieve_timed
from sentinel.schema import (
    Account,
    Answer,
    Citation,
    FeedbackRequest,
    GuardrailReport,
    LatencyMs,
    QueryRequest,
    RetrievedChunk,
    TokenResponse,
)
from sentinel.spend import RedisDayTotal, meter, start_request_cost
from sentinel.users import User, UserStore

# --------------------------------------------------------------------------- injectable engine


@dataclass
class QueryEngine:
    """The real pipeline. Injected via Depends(get_engine) so tests can override it."""

    def retrieve(
        self, question: str, top_k: int | None, access: Access
    ) -> tuple[list[RetrievedChunk], float, float]:
        return hybrid_retrieve_timed(question, top_k, access=access)

    def plan(self, question: str) -> Plan:
        return plan_question(question)

    def embed(self, question: str):
        """The question's embedding, as retrieval computes it (the semantic cache's key)."""
        from sentinel.retrieve import QUERY_INSTRUCTION, _embedder

        return _embedder().encode([QUERY_INSTRUCTION + question], normalize_embeddings=True,
                                  convert_to_numpy=True).astype("float32")[0]

    def retrieve_many(
        self, queries: list[str], top_k: int | None, access: Access
    ) -> tuple[list[RetrievedChunk], float, float]:
        return multi_retrieve_timed(queries, top_k, access=access)

    def generate_tokens(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        return generate(question, chunks)

    def screen_input(self, question: str) -> InputScreening:
        return screen_question(question)

    def check_answer(self, question: str, answer: str) -> AnswerCheck:
        return check_answer(question, answer)


_default_engine = QueryEngine()


def get_engine() -> QueryEngine:
    return _default_engine


# --------------------------------------------------------------------------- auth (JWT)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def default_user_store() -> UserStore:
    return UserStore(demo_user=settings.demo_user, demo_password=settings.demo_password,
                     default_scopes=settings.default_scopes, users_file=settings.users_file)


@dataclass(frozen=True)
class Principal:
    """An authenticated caller: who, and which document scopes they may retrieve (v2-R4)."""

    subject: str
    scopes: frozenset[str]
    shared: bool = False


def create_access_token(user: User, scopes: frozenset[str] | None = None) -> str:
    """A signed token for `user`, carrying `scopes` (default: all the account's) and the
    account's credential tag, so it stops working if the password or scopes change."""
    scopes = user.scopes if scopes is None else scopes
    if not scopes <= user.scopes:
        raise ValueError(f"{user.username} may not be issued {sorted(scopes - user.scopes)}")
    now = datetime.now(UTC)
    claims = {"sub": user.username, "iat": now,
              "exp": now + timedelta(minutes=settings.jwt_expiry_minutes),
              "scopes": sorted(scopes), "cred": user.credential_tag(settings.jwt_secret)}
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def _verified_account(token: str, store: UserStore) -> tuple[User, frozenset[str]] | None:
    """The account a token belongs to and the scopes it grants, or None. A valid signature is
    not enough: the account must still exist and be enabled, and its credential tag must
    match, so a disabled account, a changed password or a removed scope ends old tokens."""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None
    user = store.get(payload.get("sub") or "")
    scopes = payload.get("scopes")
    if (user is None or not isinstance(scopes, list)
            or not all(isinstance(x, str) for x in scopes)
            or payload.get("cred") != user.credential_tag(settings.jwt_secret)):
        return None
    granted = frozenset(scopes) & user.scopes
    return (user, granted) if granted else None


def get_current_user(request: Request, token: str = Depends(oauth2_scheme)) -> Principal:
    """Decode + validate the bearer JWT against the user store; 401 on anything wrong (FR-S3)."""
    found = _verified_account(token, request.app.state.users)
    if found is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user, scopes = found
    return Principal(subject=user.username, scopes=scopes, shared=user.shared)


def _forwarded_for(request: Request) -> list[str]:
    entries = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",")]
    return [h for h in entries if h]


def _client_ip(request: Request) -> str:
    """The caller's address for rate limiting. Not the leftmost X-Forwarded-For entry (what
    uvicorn's proxy_headers puts in request.client): the caller writes that one, so a made-up
    header per request would get a fresh rate-limit bucket each time. The trusted proxy appends
    the address it actually saw, so count settings.trusted_proxy_hops entries from the right."""
    hops = settings.trusted_proxy_hops
    forwarded = _forwarded_for(request)
    if hops > 0 and forwarded:
        return forwarded[max(0, len(forwarded) - hops)]
    return get_remote_address(request)


def _rate_key(request: Request) -> str:
    """Rate limit bucket: a named account's own (wherever it calls from), or the caller's IP
    for the shared demo account and anything unauthenticated."""
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        found = _verified_account(auth[7:].strip(), request.app.state.users)
        if found and not found[0].shared:
            return f"user:{found[0].username}"
    return _client_ip(request)


def _bucket_tag(key: str) -> str:
    """A stand-in for a rate-limit key in the logs: the same caller always gets the same tag and
    different callers different ones, but the tag can't be turned back into an address (it is
    keyed with the JWT secret)."""
    return hmac.new(settings.jwt_secret.encode(), key.encode(), hashlib.sha256).hexdigest()[:12]


# --------------------------------------------------------------------------- SSE query stream


# How much of an answer the query log keeps: enough for feedback review, bounded per entry.
_LOGGED_ANSWER_CHARS = 4000


def _scrub(text: str) -> str:
    """Pattern PII (emails, phones, cards, credentials) out of text bound for the logs."""
    return redact(text)[0]


def default_cache() -> AnswerCache | RedisAnswerCache:
    """The configured cache backend. An unreachable Redis is logged, not fatal: the service
    starts, and lookups miss until Redis answers."""
    if settings.cache_backend == "redis":
        cache = RedisAnswerCache.from_url(settings.redis_url, settings.cache_threshold,
                                          settings.cache_ttl_seconds, settings.cache_max_entries)
        log.info("cache.backend", backend="redis", url=redacted(settings.redis_url),
                 reachable=cache.ping())
        return cache
    if settings.cache_backend != "memory":
        raise ValueError(f"cache_backend must be 'memory' or 'redis', not "
                         f"{settings.cache_backend!r}")
    return AnswerCache(settings.cache_threshold, settings.cache_ttl_seconds,
                       settings.cache_max_entries)


_index_meta: dict[tuple, str] = {}


def _index_build() -> str:
    """Identifies the index being served, so a rebuilt index never reuses old answers."""
    meta = IndexPaths(settings.index_dir.resolve()).meta
    try:
        st = meta.stat()
    except OSError:
        return "no-index"
    key = (str(meta), st.st_mtime_ns, st.st_size)
    if key not in _index_meta:
        import hashlib
        # By content: the image rebuilds the index on every deploy, and the build time it
        # records would otherwise make each deploy orphan every cached answer.
        content = json.loads(meta.read_text())
        content.pop("built_utc", None)
        _index_meta[key] = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]
    return _index_meta[key]


def _cache_context(user: Principal, req: QueryRequest) -> str:
    """Everything besides the question that decides an answer. Two requests share cached
    answers only when all of it matches: above all the scopes, so a scope never sees answers
    built from documents it can't retrieve."""
    return context_key(
        scopes=user.scopes, index=_index_build(), vector_store=settings.vector_store,
        top_k=req.top_k or settings.rerank_top_k,
        prompt=SYSTEM_PROMPT + HARDENING_RULE + HARDENING_REMINDER,
        models=(settings.embedding_model, settings.reranker_model, settings.generation_provider,
                settings.generation_model, settings.planner_model),
        retrieval=(settings.dense_top_n, settings.sparse_top_n, settings.rrf_k,
                   settings.rerank_pool_size, settings.multistep_top_k, settings.query_planning),
        guards=(settings.input_guard, settings.prompt_hardening, settings.redact_context_pii,
                settings.output_guard, settings.answer_check, settings.answer_check_blocks),
    )


def _cached_events(entry: Entry, similarity: float, answer_id: str, user: Principal,
                   req: QueryRequest, latency: dict[str, float], t_request: float,
                   cost: list[float] | None) -> Iterator[dict]:
    """Replay a cached answer: the same events a fresh answer sends, under its own answer_id."""
    p = entry.payload
    retrieved = [RetrievedChunk.model_validate(c) for c in p["retrieved"]]
    yield {"event": "retrieved", "data": json.dumps([
        {"chunk_id": c.chunk_id, "doc_id": c.chunk_id.split("#", 1)[0],
         "dense_score": c.dense_score, "sparse_score": c.sparse_score,
         "fused_score": c.fused_score, "rerank_score": c.rerank_score} for c in retrieved])}
    first_token = time.perf_counter()
    for part in p["parts"]:
        yield {"event": "token", "data": part}
    answer = Answer(
        question=req.question, text=p["text"],
        citations=[Citation.model_validate(c) for c in p["citations"]],
        retrieved=retrieved,
        latency_ms=LatencyMs(retrieve=0.0, rerank=0.0, generate=0.0, **latency),
        guardrails=GuardrailReport.model_validate(p["guardrails"]),
        answer_id=answer_id, sub_questions=p["sub_questions"],
        cached_from=entry.answer_id, cache_similarity=round(similarity, 4),
    )
    log.info(
        "query",
        answer_id=answer_id,
        question=_scrub(req.question),
        answer=p["text"][:_LOGGED_ANSWER_CHARS],
        user=user.subject,
        scopes=sorted(user.scopes),
        chunk_ids=[c.chunk_id for c in retrieved],
        cache_hit=True,
        cached_from=entry.answer_id,
        cache_similarity=round(similarity, 4),
        cache_age_s=round(time.time() - entry.created),
        latency_ms={k: round(v, 1) for k, v in latency.items()},
        first_token_ms=round((first_token - t_request) * 1000.0, 1),
        request_ms=round((time.perf_counter() - t_request) * 1000.0, 1),
        cost_usd=round(cost[0], 6) if cost is not None else None,
        n_citations=len(answer.citations),
        outcome="answered",
        guardrails_flagged=False,
    )
    yield {"event": "done", "data": answer.model_dump_json()}


def _failsafe(events: Iterator[dict]) -> Iterator[dict]:
    """Ends a stream that fails partway (a provider outage mid-answer) with an `error` event, so
    the client can say so instead of showing a cut-off answer as if it were finished."""
    try:
        yield from events
    except Exception as exc:   # noqa: BLE001 - anything that escapes is reported, then logged
        log.exception("query.failed", error=type(exc).__name__)
        yield {"event": "error", "data": json.dumps(
            {"detail": "The answer failed partway through. Try again in a moment."})}


def _query_events(engine: QueryEngine, user: Principal, req: QueryRequest,
                  cost: list[float] | None = None,
                  cache: AnswerCache | RedisAnswerCache | None = None) -> Iterator[dict]:
    """Sync generator of SSE events: retrieved -> token* -> done. Runs in a threadpool (the
    LLM stream is sync), so it never blocks the event loop. `cost` is the request's spend tally
    (spend.start_request_cost), reported in the query log."""
    answer_id = uuid.uuid4().hex   # ties a later POST /feedback to this answer's log entry
    t_request = time.perf_counter()
    first_token: list[float] = []   # when the reader got the first word of the answer
    latency: dict[str, float] = {}
    guard_s = 0.0
    if settings.input_guard:
        t = time.perf_counter()
        screening = engine.screen_input(req.question)
        guard_s += time.perf_counter() - t
        if screening.flagged:
            yield from _refusal(user, req, screening, guard_s, answer_id, cost)
            return
    # The semantic cache, consulted only after the input guard has passed the question. A
    # question with PII in it is neither looked up nor stored.
    cache_key = vector = None
    if cache is not None and settings.semantic_cache and not redact(req.question)[1]:
        t = time.perf_counter()
        vector = engine.embed(req.question)
        cache_key = _cache_context(user, req)
        hit = cache.lookup(req.question, vector, cache_key)
        latency["cache"] = (time.perf_counter() - t) * 1000.0
        if hit is not None:
            latency["guard"] = guard_s * 1000.0
            cache.served(answer_id, hit.entry)   # so a thumbs-down on this reply evicts it
            yield from _cached_events(hit.entry, hit.similarity, answer_id, user, req, latency,
                                      t_request, cost)
            return

    access = Access(scopes=user.scopes)
    plan = engine.plan(req.question) if settings.query_planning else None
    if plan is not None:
        latency["plan"] = plan.latency_ms
    if plan is not None and plan.multi:
        chunks, latency["retrieve"], latency["rerank"] = engine.retrieve_many(
            plan.queries, req.top_k, access)
    else:
        chunks, latency["retrieve"], latency["rerank"] = engine.retrieve(
            req.question, req.top_k, access)
    t = time.perf_counter()
    redactions: dict[str, int] = {}
    if settings.redact_context_pii:
        chunks, counts = redact_chunks(chunks)
        redactions = dict(counts)
    guard_s += time.perf_counter() - t

    # Up-front sources so a client can render citations context before tokens arrive.
    yield {
        "event": "retrieved",
        "data": json.dumps(
            [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.chunk_id.split("#", 1)[0],
                    "dense_score": c.dense_score,
                    "sparse_score": c.sparse_score,
                    "fused_score": c.fused_score,
                    "rerank_score": c.rerank_score,
                }
                for c in chunks
            ]
        ),
    }

    # Off-topic: the best passage barely relates to the question, so there's nothing to ground
    # an answer in. Stop before the model (and the answer check) is called.
    cutoff = settings.min_rerank_score
    scores = [c.rerank_score for c in chunks if c.rerank_score is not None]
    if cutoff is not None and scores and max(scores) < cutoff:
        latency["guard"] = guard_s * 1000.0
        yield from _off_topic(user, req, chunks, redactions, latency, answer_id, plan,
                              max(scores), t_request, cost)
        return

    report = OutputReport()
    sanitizer = StreamSanitizer(report) if settings.output_guard else None
    # Blocking: tokens are collected, not sent, until the answer check has passed.
    blocking = settings.answer_check and settings.answer_check_blocks
    parts: list[str] = []
    stream_guard_s = 0.0   # sanitising time inside the stream, reported as guard, not generate
    t0 = time.perf_counter()
    for token in engine.generate_tokens(req.question, chunks):
        if sanitizer is not None:
            t = time.perf_counter()
            token = sanitizer.feed(token)
            stream_guard_s += time.perf_counter() - t
        if token:
            parts.append(token)
            if not blocking:
                first_token or first_token.append(time.perf_counter())
                yield {"event": "token", "data": token}
    if sanitizer is not None and (tail := sanitizer.finish()):
        parts.append(tail)
        if not blocking:
            first_token or first_token.append(time.perf_counter())
            yield {"event": "token", "data": tail}
    latency["generate"] = (time.perf_counter() - t0 - stream_guard_s) * 1000.0
    guard_s += stream_guard_s

    text = "".join(parts)
    t = time.perf_counter()
    check_citations(text, chunks, report)
    abstained = is_abstention(text)
    guard_s += time.perf_counter() - t
    latency["guard"] = guard_s * 1000.0
    # An answer that cites nothing it was given isn't grounded: a reply to "hi", small talk, or
    # the model's own memory. It's replaced outright, so the answer check has nothing to read.
    ungrounded = settings.withhold_uncited and report.uncited
    if ungrounded:
        text = UNGROUNDED_MESSAGE
    verdict = None
    if settings.answer_check and not abstained and not ungrounded:
        if blocking:
            yield {"event": "checking", "data": ""}   # lets a client say why it's waiting
        t = time.perf_counter()
        verdict = engine.check_answer(req.question, text)
        latency["answer_check"] = (time.perf_counter() - t) * 1000.0
        if verdict.flagged:
            text = WITHHELD_MESSAGE
    withheld = verdict is not None and verdict.flagged
    if blocking:
        # Released as the model's own tokens, so clients can still render word by word; a
        # withheld answer goes out as the replacement alone.
        first_token or first_token.append(time.perf_counter())
        for part in [text] if withheld or ungrounded else parts:
            yield {"event": "token", "data": part}
    citations = [] if abstained or withheld or ungrounded else parse_citations(text, chunks)
    guardrails = GuardrailReport(context_redactions=redactions, **report.as_dict())
    if withheld:
        guardrails.answer_withheld = guardrails.flagged = True
        guardrails.answer_check_error = verdict.error
    if ungrounded:
        guardrails.answer_ungrounded = guardrails.flagged = True
    answer = Answer(
        question=req.question,
        text=text,
        citations=citations,
        retrieved=chunks,
        latency_ms=LatencyMs(**latency),
        guardrails=guardrails,
        answer_id=answer_id,
        sub_questions=plan.queries if plan is not None and plan.multi else [],
    )

    # Only an answer that went out clean is reused: cited, not abstained, not withheld, and
    # nothing removed or flagged by the output guard.
    stored = False
    if (cache_key is not None and citations and not abstained and not withheld
            and not guardrails.flagged and not guardrails.removed):
        cache.store(req.question, vector, cache_key, {
            "text": text, "parts": parts,
            "citations": [c.model_dump() for c in citations],
            "retrieved": [c.model_dump() for c in chunks],
            "sub_questions": answer.sub_questions,
            "guardrails": guardrails.model_dump(),
        }, answer_id)
        stored = True

    # The question and answer are logged (PII scrubbed) so feedback can be reviewed against
    # what was actually said; the answer already passed the output guard's PII redaction.
    log.info(
        "query",
        answer_id=answer_id,
        question=_scrub(req.question),
        # For a withheld answer, what was generated (and held back), so a wrong withhold can be
        # reviewed; the reader saw WITHHELD_MESSAGE.
        answer="".join(parts)[:_LOGGED_ANSWER_CHARS],
        user=user.subject,
        scopes=sorted(user.scopes),
        chunk_ids=[c.chunk_id for c in chunks],
        sub_queries=len(plan.queries) if plan is not None else None,
        cache_hit=False if cache_key is not None else None,
        cache_stored=stored,
        plan_error=plan.error if plan is not None else None,
        latency_ms={k: round(v, 1) for k, v in latency.items()},
        # End to end inside the service: the first word the reader saw, and the whole request.
        first_token_ms=round((first_token[0] - t_request) * 1000.0, 1) if first_token else None,
        request_ms=round((time.perf_counter() - t_request) * 1000.0, 1),
        cost_usd=round(cost[0], 6) if cost is not None else None,
        n_citations=len(citations),
        outcome=("withheld" if withheld else "ungrounded" if ungrounded
                 else "abstained" if abstained else "answered"),
        guardrails_flagged=guardrails.flagged or bool(redactions),
    )
    yield {"event": "done", "data": answer.model_dump_json()}


def _off_topic(user: Principal, req: QueryRequest, chunks: list[RetrievedChunk],
               redactions: dict[str, int], latency: dict[str, float], answer_id: str,
               plan: Plan | None, top_score: float, t_request: float,
               cost: list[float] | None = None):
    """The rest of the SSE stream when no passage matches the question: no generation."""
    answer = Answer(
        question=req.question,
        text=OFF_TOPIC_MESSAGE,
        retrieved=chunks,
        latency_ms=LatencyMs(generate=0.0, **latency),
        guardrails=GuardrailReport(context_redactions=redactions, off_topic=True, flagged=True),
        answer_id=answer_id,
        sub_questions=plan.queries if plan is not None and plan.multi else [],
    )
    log.info(
        "query",
        answer_id=answer_id,
        question=_scrub(req.question),
        user=user.subject,
        scopes=sorted(user.scopes),
        chunk_ids=[c.chunk_id for c in chunks],
        top_rerank_score=round(top_score, 3),
        latency_ms={k: round(v, 1) for k, v in latency.items()},
        request_ms=round((time.perf_counter() - t_request) * 1000.0, 1),
        cost_usd=round(cost[0], 6) if cost is not None else None,
        n_citations=0,
        outcome="off_topic",
        guardrails_flagged=True,
    )
    yield {"event": "token", "data": OFF_TOPIC_MESSAGE}
    yield {"event": "done", "data": answer.model_dump_json()}


def _refusal(user: Principal, req: QueryRequest, screening: InputScreening, guard_s: float,
             answer_id: str, cost: list[float] | None = None):
    """The SSE stream for a question the input guard refused: no retrieval, no generation."""
    report = GuardrailReport(input_flagged=True, input_error=screening.error, flagged=True)
    answer = Answer(
        question=req.question,
        text=REFUSAL_MESSAGE,
        latency_ms=LatencyMs(retrieve=0.0, rerank=0.0, generate=0.0, guard=guard_s * 1000.0),
        guardrails=report,
        answer_id=answer_id,
    )
    log.warning(
        "query.refused",
        answer_id=answer_id,
        question=_scrub(req.question),
        user=user.subject,
        input_scores=screening.scores,
        input_error=screening.error,
        request_ms=round(guard_s * 1000.0, 1),
        cost_usd=round(cost[0], 6) if cost is not None else None,
    )
    yield {"event": "retrieved", "data": "[]"}
    yield {"event": "token", "data": REFUSAL_MESSAGE}
    yield {"event": "done", "data": answer.model_dump_json()}


# --------------------------------------------------------------------------- app factory


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the index + local models at startup so first-query latency is honest (NFR-3).
    Only runs when the app is actually served (uvicorn / `with TestClient(...)`), so the
    hermetic tests — which use TestClient without a context manager — stay fast."""
    from sentinel.retrieve import warmup

    log.info("warmup_start")
    warmup()
    log.info("warmup_done")
    yield


def create_app(users: UserStore | None = None) -> FastAPI:
    configure_logging(json_output=True)
    app = FastAPI(
        title="Sentinel",
        description="Hybrid-retrieval RAG service with an eval gate.",
        lifespan=_lifespan,
    )
    if settings.cors_origins:   # the deployed dashboard calls /token and /query from the browser
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    app.state.users = users or default_user_store()
    app.state.cache = default_cache()
    # The daily budget's tally goes to the same Redis as the cache, so a cold start or a
    # redeploy doesn't reset what today has already spent.
    meter.use_store(RedisDayTotal.from_url(settings.redis_url)
                    if settings.cache_backend == "redis" else None)
    # Limiter is created per-app (fresh in-memory storage) so tests don't bleed rate state.
    limiter = Limiter(key_func=_rate_key)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/healthz")
    @app.get("/health")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/token", response_model=TokenResponse)
    @limiter.limit(settings.token_rate_limit, key_func=_client_ip)
    def issue_token(request: Request,
                    form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
        user = app.state.users.authenticate(form.username, form.password)
        if user is None:
            # No username in the log: a mistyped password often lands in that field.
            log.warning("token.refused")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        # OAuth2 `scope`: a caller may ask for fewer scopes than the account has, never more.
        asked = frozenset(form.scopes)
        if not asked <= user.scopes:
            log.warning("token.invalid_scope", user=user.username, asked=sorted(asked))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"invalid_scope: not granted {sorted(asked - user.scopes)}",
            )
        scopes = asked or user.scopes
        log.info("token.issued", user=user.username, scopes=sorted(scopes), shared=user.shared)
        return TokenResponse(access_token=create_access_token(user, scopes),
                             scopes=sorted(scopes),
                             expires_in=settings.jwt_expiry_minutes * 60)

    @app.get("/me", response_model=Account)
    def me(user: Principal = Depends(get_current_user)) -> Account:
        return Account(username=user.subject, scopes=sorted(user.scopes), shared=user.shared)

    @app.post("/query")
    @limiter.limit(settings.rate_limit)
    async def query(
        request: Request,
        body: QueryRequest,
        user: Principal = Depends(get_current_user),
        engine: QueryEngine = Depends(get_engine),
    ) -> EventSourceResponse:
        # Which rate-limit bucket the request landed in, and how many X-Forwarded-For entries
        # arrived: enough to check in the logs that separate callers are limited separately and
        # that a client-written entry isn't what decides the bucket.
        log.info("query.client", bucket=_bucket_tag(_rate_key(request)),
                 xff_entries=len(_forwarded_for(request)),
                 trusted_proxy_hops=settings.trusted_proxy_hops)
        # Checked before any work, so a request either runs whole or is turned away: a public
        # demo must not be able to spend the API budget faster than the owner intends.
        limit = settings.daily_spend_limit_usd
        if limit is not None and meter.spent_today() >= limit:
            log.warning("query.budget_exhausted", spent=round(meter.spent_today(), 4), limit=limit)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The demo's daily API budget is used up; it resets at 00:00 UTC.",
            )
        return EventSourceResponse(_failsafe(_query_events(engine, user, body, start_request_cost(),
                                                           request.app.state.cache)))

    # Feedback is a log event, not a database row: the service stays stateless and adds no
    # infrastructure, and scripts/feedback_candidates.py joins it to the query log by
    # answer_id. An unknown id is accepted here and simply finds no match there.
    @app.post("/feedback", status_code=status.HTTP_202_ACCEPTED)
    @limiter.limit(settings.rate_limit)
    async def feedback(
        request: Request,
        body: FeedbackRequest,
        user: Principal = Depends(get_current_user),
    ) -> dict[str, str]:
        # A reader saying an answer is wrong also stops it being served again from the cache.
        evicted = body.rating == "down" and request.app.state.cache.evict_answer(body.answer_id)
        log.info(
            "feedback",
            answer_id=body.answer_id,
            rating=body.rating,
            cache_evicted=evicted,
            comment=_scrub(body.comment) if body.comment else None,
            user=user.subject,
        )
        return {"status": "recorded"}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    # Behind a proxy (Cloud Run), take the client address from X-Forwarded-For so the per-IP
    # rate limit sees each caller rather than the proxy.
    uvicorn.run(app, host=settings.api_host, port=settings.api_port,
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
