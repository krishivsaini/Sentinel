"""FastAPI service (§11) — Phase 2, Day 6.

Endpoints:
    POST /token   -> issue a JWT for the single demo user (OAuth2 password form).
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

import json
import secrets
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from sentinel.config import settings
from sentinel.generate import generate, is_abstention, parse_citations
from sentinel.guardrails.answer import WITHHELD_MESSAGE, AnswerCheck, check_answer
from sentinel.guardrails.input import REFUSAL_MESSAGE, InputScreening, screen_question
from sentinel.guardrails.output import OutputReport, StreamSanitizer, check_citations
from sentinel.guardrails.pii import redact, redact_chunks
from sentinel.logging_config import configure_logging, log
from sentinel.retrieve import Access, hybrid_retrieve_timed
from sentinel.spend import meter
from sentinel.schema import (
    Answer,
    FeedbackRequest,
    GuardrailReport,
    LatencyMs,
    QueryRequest,
    RetrievedChunk,
    TokenResponse,
)


# --------------------------------------------------------------------------- injectable engine


@dataclass
class QueryEngine:
    """The real pipeline. Injected via Depends(get_engine) so tests can override it."""

    def retrieve(
        self, question: str, top_k: int | None, access: Access
    ) -> tuple[list[RetrievedChunk], float, float]:
        return hybrid_retrieve_timed(question, top_k, access=access)

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


@dataclass(frozen=True)
class Principal:
    """An authenticated caller: who, and which document scopes they may retrieve (v2-R4)."""

    subject: str
    scopes: frozenset[str]


def create_access_token(subject: str, scopes: tuple[str, ...] | None = None) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    claims = {"sub": subject, "exp": expire, "scopes": list(scopes or settings.default_scopes)}
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def get_current_user(token: str = Depends(oauth2_scheme)) -> Principal:
    """Decode + validate the bearer JWT; 401 on anything wrong (FR-S3). The scopes come from the
    signed token; one without a scopes claim gets the default scopes, never more."""
    credentials_error = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise credentials_error from exc
    subject = payload.get("sub")
    scopes = payload.get("scopes", list(settings.default_scopes))
    if not subject or not isinstance(scopes, list) or not all(isinstance(s, str) for s in scopes):
        raise credentials_error
    return Principal(subject=subject, scopes=frozenset(scopes))


# --------------------------------------------------------------------------- SSE query stream


# How much of an answer the query log keeps: enough for feedback review, bounded per entry.
_LOGGED_ANSWER_CHARS = 4000


def _scrub(text: str) -> str:
    """Pattern PII (emails, phones, cards, credentials) out of text bound for the logs."""
    return redact(text)[0]


def _query_events(engine: QueryEngine, user: Principal, req: QueryRequest) -> Iterator[dict]:
    """Sync generator of SSE events: retrieved -> token* -> done. Runs in a threadpool (the
    LLM stream is sync), so it never blocks the event loop."""
    answer_id = uuid.uuid4().hex   # ties a later POST /feedback to this answer's log entry
    latency: dict[str, float] = {}
    guard_s = 0.0
    if settings.input_guard:
        t = time.perf_counter()
        screening = engine.screen_input(req.question)
        guard_s += time.perf_counter() - t
        if screening.flagged:
            yield from _refusal(user, req, screening, guard_s, answer_id)
            return
    chunks, latency["retrieve"], latency["rerank"] = engine.retrieve(
        req.question, req.top_k, Access(scopes=user.scopes)
    )
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
                yield {"event": "token", "data": token}
    if sanitizer is not None and (tail := sanitizer.finish()):
        parts.append(tail)
        if not blocking:
            yield {"event": "token", "data": tail}
    latency["generate"] = (time.perf_counter() - t0 - stream_guard_s) * 1000.0
    guard_s += stream_guard_s

    text = "".join(parts)
    t = time.perf_counter()
    check_citations(text, chunks, report)
    abstained = is_abstention(text)
    guard_s += time.perf_counter() - t
    latency["guard"] = guard_s * 1000.0
    verdict = None
    if settings.answer_check and not abstained:
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
        for part in [WITHHELD_MESSAGE] if withheld else parts:
            yield {"event": "token", "data": part}
    citations = [] if abstained or withheld else parse_citations(text, chunks)
    guardrails = GuardrailReport(context_redactions=redactions, **report.as_dict())
    if withheld:
        guardrails.answer_withheld = guardrails.flagged = True
        guardrails.answer_check_error = verdict.error
    answer = Answer(
        question=req.question,
        text=text,
        citations=citations,
        retrieved=chunks,
        latency_ms=LatencyMs(**latency),
        guardrails=guardrails,
        answer_id=answer_id,
    )

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
        latency_ms={k: round(v, 1) for k, v in latency.items()},
        n_citations=len(citations),
        outcome="withheld" if withheld else "abstained" if abstained else "answered",
        guardrails_flagged=guardrails.flagged or bool(redactions),
    )
    yield {"event": "done", "data": answer.model_dump_json()}


def _refusal(user: Principal, req: QueryRequest, screening: InputScreening, guard_s: float,
             answer_id: str):
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


def create_app() -> FastAPI:
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
    # Limiter is created per-app (fresh in-memory storage) so tests don't bleed rate state.
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/healthz")
    @app.get("/health")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/token", response_model=TokenResponse)
    def issue_token(form: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
        ok_user = secrets.compare_digest(form.username, settings.demo_user)
        ok_pass = secrets.compare_digest(form.password, settings.demo_password)
        if not (ok_user and ok_pass):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return TokenResponse(access_token=create_access_token(form.username))

    @app.post("/query")
    @limiter.limit(settings.rate_limit)
    async def query(
        request: Request,
        body: QueryRequest,
        user: Principal = Depends(get_current_user),
        engine: QueryEngine = Depends(get_engine),
    ) -> EventSourceResponse:
        # Checked before any work, so a request either runs whole or is turned away: a public
        # demo must not be able to spend the API budget faster than the owner intends.
        limit = settings.daily_spend_limit_usd
        if limit is not None and meter.spent_today() >= limit:
            log.warning("query.budget_exhausted", spent=round(meter.spent_today(), 4), limit=limit)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The demo's daily API budget is used up; it resets at 00:00 UTC.",
            )
        return EventSourceResponse(_query_events(engine, user, body))

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
        log.info(
            "feedback",
            answer_id=body.answer_id,
            rating=body.rating,
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
