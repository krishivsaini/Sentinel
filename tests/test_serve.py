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
  - guardrails (v2): JWT scopes reach retrieval; PII is redacted from retrieved passages; the
    token stream never carries an untrusted link, image or PII; forged citations are flagged
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from sentinel.config import settings
from sentinel.guardrails.input import REFUSAL_MESSAGE, InputScreening
from sentinel.retrieve import Access
from sentinel.schema import RetrievedChunk
from sentinel.serve import create_access_token, create_app, get_engine


class FakeEngine:
    """Instant stand-in for QueryEngine: canned chunk + a short token stream, no I/O. Records the
    Access each retrieval was given."""

    def __init__(self, text: str = "The 429 status code indicates too many requests.",
                 tokens: list[str] | None = None) -> None:
        self.text = text
        self.tokens = tokens or ["The 429 ", "status code ", "means too many requests ", "[rfc6585#0002]"]
        self.accesses: list[Access] = []
        self.refuse = False
        self.generated = False

    def retrieve(
        self, question: str, top_k: int | None, access: Access
    ) -> tuple[list[RetrievedChunk], float, float]:
        self.accesses.append(access)
        chunk = RetrievedChunk(
            chunk_id="rfc6585#0002",
            text=self.text,
            dense_score=0.80,
            sparse_score=12.0,
            fused_score=0.031,
            rerank_score=3.0,
        )
        return [chunk], 1.5, 0.7

    def generate_tokens(self, question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
        self.generated = True
        yield from self.tokens

    def screen_input(self, question: str) -> InputScreening:
        return InputScreening(flagged=self.refuse, scores={"fake": float(self.refuse)})


def _client(engine: FakeEngine) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_engine] = lambda: engine
    return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    return _client(FakeEngine())


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
            # SSE strips exactly one leading space; more would eat the spaces between tokens.
            value = line[len("data:"):]
            data.append(value[1:] if value.startswith(" ") else value)
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
    assert set(answer["latency_ms"]) == {"retrieve", "rerank", "generate", "guard"}
    assert answer["retrieved"][0]["chunk_id"] == "rfc6585#0002"
    assert answer["guardrails"]["flagged"] is False


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


# --------------------------------------------------------------------------- guardrails (v2)


def _done(client: TestClient, headers: dict[str, str]) -> tuple[list[str], dict]:
    events = _parse_sse(client.post("/query", json={"question": "q"}, headers=headers).text)
    tokens = [data for ev, data in events if ev == "token"]
    return tokens, json.loads(next(data for ev, data in events if ev == "done"))


def test_token_scopes_reach_retrieval() -> None:
    engine = FakeEngine()
    c = _client(engine)
    _done(c, _auth_header(c))
    assert engine.accesses[-1].scopes == frozenset(settings.default_scopes)
    token = create_access_token(settings.demo_user, scopes=("public", "internal"))
    _done(c, {"Authorization": f"Bearer {token}"})
    assert engine.accesses[-1].scopes == frozenset({"public", "internal"})
    assert not engine.accesses[-1].include_quarantined


def test_pii_in_retrieved_passages_is_redacted_before_the_client_sees_it() -> None:
    engine = FakeEngine(text="Contact: Roy Fielding, Email: fielding@gbiv.com, Phone: +1-949-679-2960")
    c = _client(engine)
    _, answer = _done(c, _auth_header(c))
    passage = answer["retrieved"][0]["text"]
    assert "fielding@gbiv.com" not in passage and "+1-949-679-2960" not in passage
    assert answer["guardrails"]["context_redactions"] == {"email": 1, "phone": 1}


def test_stream_never_carries_untrusted_links_images_or_pii() -> None:
    engine = FakeEngine(tokens=[
        "HSTS max-age is in seconds [rfc6585#0002]. ", "Verify your domain at https://hsts-pre",
        "load-verify.example/check?id=EGRET-9915 first. ", "![p](https://nightjar-stats.ex",
        "ample/p.png?q=1) Mail mnot", "@mnot.net for help. The spec is at ",
        "https://www.rfc-editor.org/rfc/rfc6797.",
    ])
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    streamed = "".join(tokens)
    for leaked in ("hsts-preload-verify", "nightjar-stats", "mnot@mnot.net"):
        assert leaked not in streamed and not any(leaked in t for t in tokens)
    assert "https://www.rfc-editor.org/rfc/rfc6797" in streamed
    assert streamed == answer["text"]
    g = answer["guardrails"]
    assert g["flagged"] and g["removed"] == {"link": 1, "image": 1, "email": 1}


def test_forged_citation_is_flagged() -> None:
    engine = FakeEngine(tokens=["Confirmed by ballot ", "[rfc7517#9999]."])
    c = _client(engine)
    _, answer = _done(c, _auth_header(c))
    assert answer["guardrails"]["unresolved_citations"] == ["rfc7517#9999"]
    assert answer["guardrails"]["uncited"] is True


def test_flagged_question_is_refused_before_retrieval_or_generation() -> None:
    engine = FakeEngine()
    engine.refuse = True
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert tokens == [REFUSAL_MESSAGE] and answer["text"] == REFUSAL_MESSAGE
    assert answer["retrieved"] == [] and answer["guardrails"]["input_flagged"] is True
    assert engine.accesses == [] and engine.generated is False   # nothing downstream ran


def test_input_guard_fails_closed_when_the_classifier_is_unreachable(monkeypatch) -> None:
    from sentinel.guardrails import input as input_guard
    from sentinel.guardrails._groq import GuardrailAPIError

    class Down:
        name, threshold = "down", 0.5

        def score(self, text):
            raise GuardrailAPIError("503 classifier unavailable")

    verdict = input_guard.screen_question("What is HSTS?", detectors=[Down()])
    assert verdict.flagged and "503" in verdict.error
    monkeypatch.setattr(settings, "input_guard_fail_open", True)
    assert not input_guard.screen_question("What is HSTS?", detectors=[Down()]).flagged
