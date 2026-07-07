"""API contract tests (§4, §11) — Phase 2, Day 6.

Hermetic: the retrieval+generation pipeline is replaced by a fake engine via
`dependency_overrides`, so these tests exercise the *service* (auth, rate limiting, SSE
contract, per-stage latency) with no models and no Gemini calls. A fresh app per test gives
each one its own rate-limit storage.

Coverage:
  - GET /healthz -> 200
  - unauthenticated POST /query -> 401 (FR-S3)
  - bad /token credentials -> 401
  - authenticated /query streams a well-formed Answer with citations + per-stage latency
  - the rate limit fires under load (FR-S4)
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sentinel.config import settings
from sentinel.schema import RetrievedChunk
from sentinel.serve import create_app, get_engine


class FakeEngine:
    """Instant stand-in for QueryEngine: canned chunk + a short token stream, no I/O."""

    def retrieve(self, question: str, top_k: int | None) -> tuple[list[RetrievedChunk], float, float]:
        chunk = RetrievedChunk(
            chunk_id="rfc6585#0002",
            text="The 429 status code indicates too many requests.",
            dense_score=0.80,
            sparse_score=12.0,
            fused_score=0.031,
            rerank_score=3.0,
        )
        return [chunk], 1.5, 0.7

    def generate_tokens(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        yield from ["The 429 ", "status code ", "means too many requests ", "[rfc6585#0002]"]


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: FakeEngine()
    return TestClient(app)


def _auth_header(client: TestClient) -> dict[str, str]:
    r = client.post(
        "/token",
        data={"username": settings.demo_user, "password": settings.demo_password},
    )
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _parse_sse(text: str) -> list[tuple[str | None, str]]:
    events: list[tuple[str | None, str]] = []
    ev: str | None = None
    data: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.startswith("event:"):
            ev = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data.append(line[len("data:"):].lstrip())
        elif line == "":
            if ev is not None or data:
                events.append((ev, "\n".join(data)))
            ev, data = None, []
    if ev is not None or data:
        events.append((ev, "\n".join(data)))
    return events


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_unauthenticated_query_is_rejected(client: TestClient) -> None:
    r = client.post("/query", json={"question": "What is 429?"})
    assert r.status_code == 401


def test_token_rejects_bad_credentials(client: TestClient) -> None:
    r = client.post("/query", json={"question": "x"}, headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401
    r2 = client.post("/token", data={"username": settings.demo_user, "password": "wrong"})
    assert r2.status_code == 401


def test_authenticated_query_streams_wellformed_answer(client: TestClient) -> None:
    r = client.post("/query", json={"question": "What is 429?"}, headers=_auth_header(client))
    assert r.status_code == 200

    events = _parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert "retrieved" in kinds and "token" in kinds and kinds[-1] == "done"

    done_data = next(data for ev, data in events if ev == "done")
    answer = json.loads(done_data)
    assert "429" in answer["text"]
    assert [c["chunk_id"] for c in answer["citations"]] == ["rfc6585#0002"]
    # per-stage latency present and never blended into one number (NFR-3)
    assert set(answer["latency_ms"]) == {"retrieve", "rerank", "generate"}
    assert answer["retrieved"][0]["chunk_id"] == "rfc6585#0002"


def test_rate_limit_fires_under_load(client: TestClient) -> None:
    headers = _auth_header(client)
    limit = int(settings.rate_limit.split("/")[0])
    codes = [
        client.post("/query", json={"question": "q"}, headers=headers).status_code
        for _ in range(limit + 3)
    ]
    assert 200 in codes, "some requests should succeed"
    assert 429 in codes, "the rate limit must fire under load (FR-S4)"
    # once limited, it stays limited (monotonic): no 200 after the first 429
    first_429 = codes.index(429)
    assert all(c == 429 for c in codes[first_429:])
