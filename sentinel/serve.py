"""FastAPI service (§11) — Phase 2, Day 6.

Endpoints:
    POST /token   -> issue a JWT for the single demo user (OAuth2 password form).
    POST /query   -> Answer, SSE-streamed tokens; JWT-protected; rate-limited.
    GET  /healthz -> liveness.

Production hygiene: JWT auth (FR-S3), slowapi per-IP rate limit (FR-S4, a test proves it
fires), structured per-stage-latency logging (FR-S5), Pydantic validation everywhere (FR-S6).

The retrieval+generation pipeline is injected as a `QueryEngine` dependency so tests can swap
in an instant fake — that keeps the auth/rate-limit/contract tests hermetic (no models, no API).

Run:  python -m sentinel.serve   (or: uvicorn sentinel.serve:app)
"""

from __future__ import annotations

import json
import secrets
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

from sentinel.config import settings
from sentinel.generate import generate, is_abstention, parse_citations
from sentinel.logging_config import configure_logging, log
from sentinel.retrieve import hybrid_retrieve_timed
from sentinel.schema import (
    Answer,
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
        self, question: str, top_k: int | None
    ) -> tuple[list[RetrievedChunk], float, float]:
        return hybrid_retrieve_timed(question, top_k)

    def generate_tokens(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        return generate(question, chunks)


_default_engine = QueryEngine()


def get_engine() -> QueryEngine:
    return _default_engine


# --------------------------------------------------------------------------- auth (JWT)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def create_access_token(subject: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    return jwt.encode(
        {"sub": subject, "exp": expire}, settings.jwt_secret, algorithm=settings.jwt_algorithm
    )


def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    """Decode + validate the bearer JWT; 401 on anything wrong (FR-S3)."""
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
    if not subject:
        raise credentials_error
    return subject


# --------------------------------------------------------------------------- SSE query stream


def _query_events(engine: QueryEngine, user: str, req: QueryRequest) -> Iterator[dict]:
    """Sync generator of SSE events: retrieved -> token* -> done. Runs in a threadpool (the
    LLM stream is sync), so it never blocks the event loop."""
    latency: dict[str, float] = {}
    chunks, latency["retrieve"], latency["rerank"] = engine.retrieve(req.question, req.top_k)

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

    parts: list[str] = []
    t0 = time.perf_counter()
    for token in engine.generate_tokens(req.question, chunks):
        parts.append(token)
        yield {"event": "token", "data": token}
    latency["generate"] = (time.perf_counter() - t0) * 1000.0

    text = "".join(parts)
    abstained = is_abstention(text)
    citations = [] if abstained else parse_citations(text, chunks)
    answer = Answer(
        question=req.question,
        text=text,
        citations=citations,
        retrieved=chunks,
        latency_ms=LatencyMs(**latency),
    )

    log.info(
        "query",
        user=user,
        chunk_ids=[c.chunk_id for c in chunks],
        latency_ms={k: round(v, 1) for k, v in latency.items()},
        n_citations=len(citations),
        outcome="abstained" if abstained else "answered",
    )
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
    # Limiter is created per-app (fresh in-memory storage) so tests don't bleed rate state.
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.get("/healthz")
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
        user: str = Depends(get_current_user),
        engine: QueryEngine = Depends(get_engine),
    ) -> EventSourceResponse:
        return EventSourceResponse(_query_events(engine, user, body))

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
