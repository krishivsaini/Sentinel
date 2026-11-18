"""API contract tests — Phase 2, Day 6.

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
from sentinel.guardrails.answer import WITHHELD_MESSAGE, AnswerCheck
from sentinel.guardrails.input import REFUSAL_MESSAGE, InputScreening
from sentinel.guardrails.output import OFF_TOPIC_MESSAGE, UNGROUNDED_MESSAGE
from sentinel.retrieve import Access
from sentinel.schema import RetrievedChunk
from sentinel.serve import create_app, get_engine
from sentinel.users import UserStore, hash_password, write_users_file


class FakeEngine:
    """Instant stand-in for QueryEngine: canned chunk + a short token stream, no I/O. Records the
    Access each retrieval was given."""

    def __init__(self, text: str = "The 429 status code indicates too many requests.",
                 tokens: list[str] | None = None) -> None:
        self.text = text
        self.tokens = tokens or ["The 429 ", "status code ", "means too many requests ", "[rfc6585#0002]"]
        self.accesses: list[Access] = []
        self.refuse = False
        self.withhold = False
        self.generated = False
        self.checked: list[str] = []

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

    def check_answer(self, question: str, answer: str) -> AnswerCheck:
        self.checked.append(answer)
        return AnswerCheck(flagged=self.withhold, score=float(self.withhold))


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


@pytest.mark.parametrize("path", ["/healthz", "/health"])
def test_healthz(client: TestClient, path: str) -> None:
    r = client.get(path)
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
    assert set(answer["latency_ms"]) == {"retrieve", "rerank", "generate", "guard",
                                         "answer_check", "plan", "cache"}
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


def _sign_in_codes(client: TestClient, forwarded_for: list[str]) -> list[int]:
    return [
        client.post("/token", data={"username": "nobody", "password": "x"},
                    headers={"X-Forwarded-For": xff}).status_code
        for xff in forwarded_for
    ]


def test_a_made_up_forwarded_for_header_does_not_dodge_the_rate_limit() -> None:
    # The caller writes the left of X-Forwarded-For; the proxy appends the address it saw.
    c = _client(FakeEngine())
    limit = int(settings.token_rate_limit.split("/")[0])
    codes = _sign_in_codes(c, [f"203.0.113.{i}, 198.51.100.7" for i in range(limit + 2)])
    assert codes[:limit] == [401] * limit and codes[limit:] == [429, 429]


def test_each_real_client_address_has_its_own_bucket() -> None:
    c = _client(FakeEngine())
    limit = int(settings.token_rate_limit.split("/")[0])
    assert 429 in _sign_in_codes(c, ["198.51.100.7"] * (limit + 1))
    assert _sign_in_codes(c, ["198.51.100.8"]) == [401]     # a different caller isn't limited


def test_index_fingerprint_ignores_the_build_time(tmp_path, monkeypatch) -> None:
    from sentinel import serve

    monkeypatch.setattr(settings, "index_dir", tmp_path)
    meta = tmp_path / "meta.json"
    fingerprints = []
    for built, chunks in (("2026-11-16T05:19:24Z", 2888), ("2026-11-18T09:00:00Z", 2888),
                          ("2026-11-18T09:00:00Z", 2889)):
        meta.write_text(json.dumps({"built_utc": built, "chunk_count": chunks}))
        fingerprints.append(serve._index_build())
    assert fingerprints[0] == fingerprints[1]        # rebuilt, same content: cache kept
    assert fingerprints[1] != fingerprints[2]        # different content: cache not reused


# --------------------------------------------------------------------------- guardrails (v2)


def _done(client: TestClient, headers: dict[str, str]) -> tuple[list[str], dict]:
    events = _parse_sse(client.post("/query", json={"question": "q"}, headers=headers).text)
    tokens = [data for ev, data in events if ev == "token"]
    return tokens, json.loads(next(data for ev, data in events if ev == "done"))


def test_token_scopes_reach_retrieval(tmp_path) -> None:
    users_file = tmp_path / "users.json"
    write_users_file(users_file, [{"username": "alice", "password_hash": hash_password("pw-a"),
                                   "scopes": ["public", "internal"]}])
    engine = FakeEngine()
    app = create_app(UserStore(demo_user=settings.demo_user, demo_password=settings.demo_password,
                               default_scopes=settings.default_scopes, users_file=users_file))
    app.dependency_overrides[get_engine] = lambda: engine
    c = TestClient(app)
    _done(c, _auth_header(c))
    assert engine.accesses[-1].scopes == frozenset(settings.default_scopes)
    token = c.post("/token", data={"username": "alice", "password": "pw-a"}).json()["access_token"]
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
    tokens, answer = _done(c, _auth_header(c))
    assert answer["guardrails"]["unresolved_citations"] == ["rfc7517#9999"]
    assert answer["guardrails"]["uncited"] is True
    assert tokens == [UNGROUNDED_MESSAGE] and answer["guardrails"]["answer_ungrounded"]


def test_uncited_answer_is_withheld_before_it_reaches_the_client() -> None:
    # "hi" -> "Hello!": nothing retrieved is cited, so nothing of it is sent.
    engine = FakeEngine(tokens=["Hello", "!"])
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert tokens == [UNGROUNDED_MESSAGE] and answer["text"] == UNGROUNDED_MESSAGE
    g = answer["guardrails"]
    assert g["uncited"] and g["answer_ungrounded"] and g["flagged"]
    assert answer["citations"] == [] and engine.checked == []   # no answer check to pay for


def test_uncited_answers_pass_through_when_the_switch_is_off(monkeypatch) -> None:
    monkeypatch.setattr(settings, "withhold_uncited", False)
    engine = FakeEngine(tokens=["Hello", "!"])
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert "".join(tokens) == answer["text"] == "Hello!"
    assert answer["guardrails"]["uncited"] and not answer["guardrails"]["answer_ungrounded"]


def test_flagged_question_is_refused_before_retrieval_or_generation() -> None:
    engine = FakeEngine()
    engine.refuse = True
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert tokens == [REFUSAL_MESSAGE] and answer["text"] == REFUSAL_MESSAGE
    assert answer["retrieved"] == [] and answer["guardrails"]["input_flagged"] is True
    assert engine.accesses == [] and engine.generated is False   # nothing downstream ran


def _kinds(client: TestClient) -> list[str | None]:
    r = client.post("/query", json={"question": "q"}, headers=_auth_header(client))
    return [ev for ev, _ in _parse_sse(r.text)]


# Long enough that the output guard's held-back tail doesn't swallow the whole answer.
_LONG = [f"Sentence {i} of a longer answer about status codes. " for i in range(20)] + ["[rfc6585#0002]"]


def test_flagged_answer_never_reaches_the_client() -> None:
    engine = FakeEngine()
    engine.withhold = True
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert tokens == [WITHHELD_MESSAGE]                # nothing of the answer was sent
    assert engine.checked == ["".join(engine.tokens)]
    assert answer["text"] == WITHHELD_MESSAGE and answer["citations"] == []
    g = answer["guardrails"]
    assert g["answer_withheld"] is True and g["flagged"] is True


def test_answer_tokens_are_released_only_after_the_check() -> None:
    engine = FakeEngine(tokens=_LONG)
    c = _client(engine)
    kinds = _kinds(c)
    assert kinds[:2] == ["retrieved", "checking"] and kinds[-1] == "done"
    assert set(kinds[2:-1]) == {"token"} and len(kinds[2:-1]) > 1   # still word by word
    tokens, answer = _done(c, _auth_header(c))
    assert "".join(tokens) == answer["text"] == "".join(_LONG)
    assert answer["guardrails"]["answer_withheld"] is False


def test_streaming_mode_retracts_in_the_done_event(monkeypatch) -> None:
    monkeypatch.setattr(settings, "answer_check_blocks", False)
    engine = FakeEngine(tokens=_LONG)
    engine.withhold = True
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert len(tokens) > 1 and "".join(tokens) == "".join(_LONG)   # streamed; done replaces it
    assert answer["text"] == WITHHELD_MESSAGE and answer["guardrails"]["answer_withheld"]
    assert "checking" not in _kinds(c)


def test_without_the_answer_check_tokens_stream(monkeypatch) -> None:
    monkeypatch.setattr(settings, "answer_check", False)
    engine = FakeEngine(tokens=_LONG)
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert len(tokens) > 1 and engine.checked == []   # nothing to wait for
    assert answer["text"] == "".join(tokens)


def test_abstentions_skip_the_answer_check() -> None:
    from sentinel.generate import ABSTAIN_MESSAGE

    engine = FakeEngine(tokens=[ABSTAIN_MESSAGE])
    engine.withhold = True
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert engine.checked == [] and answer["text"] == ABSTAIN_MESSAGE
    assert tokens == [ABSTAIN_MESSAGE] and "checking" not in _kinds(c)


def test_answer_check_fails_closed_when_the_classifier_is_unreachable(monkeypatch) -> None:
    from sentinel.guardrails import answer as answer_check
    from sentinel.guardrails._groq import GuardrailAPIError

    class Down:
        threshold = 0.5

        def score(self, question, answer):
            raise GuardrailAPIError("503 classifier unavailable")

    verdict = answer_check.check_answer("What is HSTS?", "HSTS is ...", classifier=Down())
    assert verdict.flagged and "503" in verdict.error
    monkeypatch.setattr(settings, "answer_check_fail_open", True)
    assert not answer_check.check_answer("What is HSTS?", "HSTS is ...", classifier=Down()).flagged


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


# --------------------------------------------------------------------------- public deployment


def test_query_is_turned_away_whole_once_the_daily_budget_is_spent(monkeypatch) -> None:
    from sentinel.spend import meter

    monkeypatch.setattr(settings, "daily_spend_limit_usd", 0.10)
    meter.reset()
    engine = FakeEngine()
    c = _client(engine)
    headers = _auth_header(c)
    _, answer = _done(c, headers)                       # under budget: served
    assert answer["text"]
    meter.record("openai/gpt-oss-120b", 0, 200_000)     # $0.12 today
    r = c.post("/query", json={"question": "q"}, headers=headers)
    assert r.status_code == 503 and "daily API budget" in r.json()["detail"]
    assert len(engine.accesses) == 1                    # the refused request did no work
    meter.reset()


def test_spend_today_resets_at_midnight_utc(monkeypatch) -> None:
    from datetime import date

    from sentinel import spend

    spend.meter.reset()
    spend.meter.record("openai/gpt-oss-20b", 1_000_000, 0)
    assert spend.meter.spent_today() == pytest.approx(0.075)
    spend.meter._day = date(2000, 1, 1)                 # as if recorded on an earlier day
    assert spend.meter.spent_today() == 0.0
    spend.meter.record("openai/gpt-oss-20b", 1_000_000, 0)
    assert spend.meter.spent_today() == pytest.approx(0.075)
    assert spend.meter.usd == pytest.approx(0.15)       # the run total keeps counting
    spend.meter.reset()


def test_cors_allows_only_the_configured_dashboard(monkeypatch) -> None:
    monkeypatch.setattr(settings, "cors_origins", ("https://sentinel.example.app",))
    c = _client(FakeEngine())
    preflight = {"Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": "authorization,content-type"}
    ok = c.options("/query", headers={"Origin": "https://sentinel.example.app", **preflight})
    assert ok.headers.get("access-control-allow-origin") == "https://sentinel.example.app"
    other = c.options("/query", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in other.headers


# --------------------------------------------------------------------------- feedback


def _logged(capsys, event: str) -> list[dict]:
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("{")]
    return [e for e in map(json.loads, lines) if e.get("event") == event]


def test_feedback_is_logged_against_the_answer_it_rates(capsys) -> None:
    engine = FakeEngine()
    c = _client(engine)
    headers = _auth_header(c)
    _, answer = _done(c, headers)
    [query] = _logged(capsys, "query")
    assert query["answer_id"] == answer["answer_id"] and len(answer["answer_id"]) == 32
    assert query["question"] == "q" and query["answer"] == answer["text"]

    r = c.post("/feedback", headers=headers, json={
        "answer_id": answer["answer_id"], "rating": "down",
        "comment": "Wrong section. Reach me at jane.doe@gmail.com"})
    assert r.status_code == 202 and r.json() == {"status": "recorded"}
    [fb] = _logged(capsys, "feedback")
    assert fb["answer_id"] == answer["answer_id"] and fb["rating"] == "down"
    assert "jane.doe@gmail.com" not in fb["comment"] and "Wrong section" in fb["comment"]


def test_a_withheld_answer_is_logged_as_generated_for_review(capsys) -> None:
    engine = FakeEngine()
    engine.withhold = True
    c = _client(engine)
    _, answer = _done(c, _auth_header(c))
    [query] = _logged(capsys, "query")
    assert answer["text"] == WITHHELD_MESSAGE and query["outcome"] == "withheld"
    assert query["answer"] == "".join(engine.tokens)   # what was held back, not the notice


def test_a_refused_question_still_gets_an_answer_id(capsys) -> None:
    engine = FakeEngine()
    engine.refuse = True
    c = _client(engine)
    _, answer = _done(c, _auth_header(c))
    [refused] = _logged(capsys, "query.refused")
    assert refused["answer_id"] == answer["answer_id"]   # a wrong refusal can be thumbed down


@pytest.mark.parametrize("body", [
    {"answer_id": "not-an-id", "rating": "up"},
    {"answer_id": "a" * 32, "rating": "meh"},
    {"answer_id": "a" * 32, "rating": "down", "comment": "x" * 501},
])
def test_malformed_feedback_is_rejected(client, body) -> None:
    assert client.post("/feedback", headers=_auth_header(client), json=body).status_code == 422


def test_feedback_needs_a_token(client) -> None:
    r = client.post("/feedback", json={"answer_id": "a" * 32, "rating": "up"})
    assert r.status_code == 401


# --------------------------------------------------------------------------- observability


class _PricedEngine(FakeEngine):
    """Records known Groq usage while generating and checking, as the real calls do."""

    def generate_tokens(self, question, chunks):
        from sentinel.spend import meter

        meter.record("openai/gpt-oss-20b", 1_000_000, 0)            # $0.075
        yield from super().generate_tokens(question, chunks)

    def check_answer(self, question, answer):
        from sentinel.spend import meter

        meter.record("openai/gpt-oss-safeguard-20b", 0, 1_000_000)  # $0.30
        return super().check_answer(question, answer)


def test_the_query_log_carries_this_requests_cost_and_end_to_end_time(capsys) -> None:
    c = _client(_PricedEngine())
    headers = _auth_header(c)
    _done(c, headers)
    _done(c, headers)   # a second request starts its own tally, not a running total
    first, second = _logged(capsys, "query")
    for q in (first, second):
        assert q["cost_usd"] == pytest.approx(0.375)
        assert 0 <= q["first_token_ms"] <= q["request_ms"]


class OffTopicEngine(FakeEngine):
    """Retrieval finds nothing about the question: the best passage scores below the cutoff."""

    def retrieve(self, question, top_k, access):
        chunks, r, k = super().retrieve(question, top_k, access)
        return [c.model_copy(update={"rerank_score": -2.8}) for c in chunks], r, k


def test_off_topic_question_never_reaches_the_model() -> None:
    engine = OffTopicEngine()
    c = _client(engine)
    tokens, answer = _done(c, _auth_header(c))
    assert tokens == [OFF_TOPIC_MESSAGE] and answer["text"] == OFF_TOPIC_MESSAGE
    assert engine.generated is False and engine.checked == []
    assert answer["guardrails"]["off_topic"] and answer["guardrails"]["flagged"]
    assert answer["retrieved"] and answer["citations"] == []


def test_off_topic_cutoff_can_be_turned_off(monkeypatch) -> None:
    monkeypatch.setattr(settings, "min_rerank_score", None)
    engine = OffTopicEngine()
    c = _client(engine)
    _, answer = _done(c, _auth_header(c))
    assert engine.generated is True and not answer["guardrails"]["off_topic"]


class FailingEngine(FakeEngine):
    """The model provider drops the connection after the first token."""

    def generate_tokens(self, question, chunks):
        yield "The 429 "
        raise ConnectionError("provider went away")


def test_a_stream_that_fails_partway_ends_with_an_error_event(monkeypatch) -> None:
    monkeypatch.setattr(settings, "answer_check_blocks", False)
    c = _client(FailingEngine())
    r = c.post("/query", json={"question": "q"}, headers=_auth_header(c))
    events = _parse_sse(r.text)
    assert events[-1][0] == "error" and "failed partway" in json.loads(events[-1][1])["detail"]
    assert "done" not in [ev for ev, _ in events]


class _FakeRedis:
    """Just enough of redis.Redis for RedisDayTotal; `down` makes every call fail."""

    def __init__(self) -> None:
        self.data: dict[str, float] = {}
        self.down = False

    def _check(self) -> None:
        import redis
        if self.down:
            raise redis.ConnectionError("down")

    def get(self, key):
        self._check()
        return None if key not in self.data else str(self.data[key]).encode()

    def pipeline(self):
        outer = self

        class Pipe:
            ops: list = []

            def incrbyfloat(self, key, v):
                self.ops.append((key, v))

            def expire(self, key, s):
                pass

            def execute(self):
                outer._check()
                for key, v in self.ops:
                    outer.data[key] = outer.data.get(key, 0.0) + v
                self.ops.clear()

        return Pipe()


def test_daily_spend_survives_a_restart_through_the_shared_store() -> None:
    from sentinel.spend import RedisDayTotal, _Meter

    redis = _FakeRedis()
    before = _Meter()
    before.use_store(RedisDayTotal(redis))
    before.record("openai/gpt-oss-20b", 1_000_000, 0)            # $0.075
    after = _Meter()                                              # a cold start
    assert after.spent_today() == 0.0
    after.use_store(RedisDayTotal(redis))
    assert after.spent_today() == pytest.approx(0.075)            # today's spend is still known
    redis.down = True                                             # Redis unreachable:
    after.record("openai/gpt-oss-20b", 2_000_000, 0)              # recording doesn't fail,
    assert after.spent_today() == pytest.approx(0.15)             # and the process's own count is used
