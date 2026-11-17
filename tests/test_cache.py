"""The semantic cache reuses an answer only for the same question in the same context, and
never an answer that didn't go out clean.

No model calls: the service tests use a fake engine whose embeddings are fixed vectors.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from sentinel import cache as cache_mod
from sentinel.cache import AnswerCache, context_key, technical_tokens
from sentinel.config import settings
from sentinel.serve import create_app, get_engine
from sentinel.users import UserStore, hash_password, write_users_file
from tests.test_serve import FakeEngine, _auth_header, _parse_sse


@pytest.mark.parametrize("a, b, same", [
    ("What does the 308 status code mean?", "What is the meaning of the 308 status code?", True),
    ("What does the 308 status code mean?", "What does the 307 status code mean?", False),
    ("What does SETTINGS_HEADER_TABLE_SIZE control?", "What is SETTINGS_HEADER_TABLE_SIZE for?",
     True),
    ("What does the HttpOnly attribute do?", "What does the Secure attribute do?", False),
    ("What does the key word MUST NOT mean?", "What does the key word SHOULD NOT mean?", False),
    ("How does HTTP/2 compress headers?", "How are headers compressed in HTTP/2?", True),
    ("What does DoT use?", "What does DoH use?", False),
    ("What is TLS 1.3?", "What is TLS 1.2?", False),
    ("What does 'no-store' mean?", "What does the no-store directive mean?", True),
])
def test_technical_tokens(a: str, b: str, same: bool) -> None:
    assert (technical_tokens(a) == technical_tokens(b)) is same


def test_a_capital_that_starts_a_sentence_is_not_a_name() -> None:
    assert technical_tokens("Why? Because. What does Basic auth do?") == {"basic"}


def _unit(seed: int, dim: int = 8) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype("float32")
    return v / np.linalg.norm(v)


Q = "What does the 429 status code mean?"


def test_a_hit_needs_similarity_tokens_and_context() -> None:
    c = AnswerCache(threshold=0.98, ttl_seconds=60, max_entries=10)
    v = _unit(1)
    c.store(Q, v, "ctx", {"text": "a"}, "id1")
    assert c.lookup("What is the meaning of the 429 status code?", v, "ctx").similarity == \
        pytest.approx(1.0)
    assert c.lookup("What does the 431 status code mean?", v, "ctx") is None   # tokens differ
    assert c.lookup(Q, v, "other scopes") is None                               # context differs
    w = _unit(2)
    w = (w - (w @ v) * v) / np.linalg.norm(w - (w @ v) * v)                    # orthogonal to v
    near = v + 0.3 * w                                                          # cosine 0.958
    assert c.lookup(Q, near / np.linalg.norm(near), "ctx") is None             # too far


def test_entries_expire_and_the_oldest_go_first(monkeypatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(cache_mod.time, "time", lambda: now[0])
    c = AnswerCache(threshold=0.9, ttl_seconds=60, max_entries=2)
    for i in range(3):
        c.store(f"What is RFC {i}?", _unit(i), "ctx", {}, f"id{i}")
    assert len(c) == 2 and c.lookup("What is RFC 0?", _unit(0), "ctx") is None
    now[0] += 61
    assert c.lookup("What is RFC 2?", _unit(2), "ctx") is None and len(c) == 0


def test_feedback_on_any_served_copy_evicts_the_entry() -> None:
    c = AnswerCache(threshold=0.9, ttl_seconds=60, max_entries=10)
    e = c.store(Q, _unit(1), "ctx", {}, "first")
    c.served("replay", e)
    assert c.evict_answer("replay") and c.lookup(Q, _unit(1), "ctx") is None
    assert not c.evict_answer("first") and not c.evict_answer("unknown")


def test_the_context_key_separates_scopes() -> None:
    assert context_key(scopes=frozenset({"public"})) != context_key(
        scopes=frozenset({"public", "internal"}))
    assert context_key(scopes=frozenset({"a", "b"})) == context_key(scopes=frozenset({"b", "a"}))


# --------------------------------------------------------------------------- the service


class CachingEngine(FakeEngine):
    """Counts generations; embeds a question as a fixed vector of its text."""

    def __init__(self) -> None:
        super().__init__()
        self.generations = 0

    def generate_tokens(self, question, chunks):
        self.generations += 1
        yield from super().generate_tokens(question, chunks)

    def embed(self, question: str) -> np.ndarray:
        seed = int(hashlib.sha256(question.encode()).hexdigest()[:8], 16)
        return _unit(seed, 16)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "semantic_cache", True)
    users = tmp_path / "users.json"
    write_users_file(users, [{"username": "alice", "password_hash": hash_password("pw-alice-1234"),
                              "scopes": ["public", "internal"]}])
    app = create_app(users=UserStore(demo_user=settings.demo_user,
                                     demo_password=settings.demo_password,
                                     default_scopes=settings.default_scopes, users_file=users))
    engine = CachingEngine()
    app.dependency_overrides[get_engine] = lambda: engine
    return app, engine


def _ask(client, headers, question=Q) -> dict:
    r = client.post("/query", json={"question": question}, headers=headers)
    return json.loads(_parse_sse(r.text)[-1][1])


def test_a_repeated_question_is_answered_from_the_cache(app) -> None:
    from fastapi.testclient import TestClient
    app, engine = app
    client = TestClient(app)
    h = _auth_header(client)
    first = _ask(client, h)
    again = _ask(client, h)
    assert engine.generations == 1
    assert again["cached_from"] == first["answer_id"] != again["answer_id"]
    assert again["text"] == first["text"] and again["cache_similarity"] == pytest.approx(1.0)
    assert again["citations"] == first["citations"] and "cache" in again["latency_ms"]
    assert first["cached_from"] is None


def test_another_scope_never_gets_a_cached_answer(app) -> None:
    from fastapi.testclient import TestClient
    app, engine = app
    client = TestClient(app)
    _ask(client, _auth_header(client))
    tok = client.post("/token", data={"username": "alice", "password": "pw-alice-1234"}).json()
    alice = _ask(client, {"Authorization": f"Bearer {tok['access_token']}"})
    assert alice["cached_from"] is None and engine.generations == 2


def test_a_withheld_answer_is_not_cached(app) -> None:
    from fastapi.testclient import TestClient
    app, engine = app
    client = TestClient(app)
    h = _auth_header(client)
    engine.withhold = True
    _ask(client, h)
    engine.withhold = False
    assert _ask(client, h)["cached_from"] is None and engine.generations == 2


def test_a_question_with_pii_is_neither_stored_nor_looked_up(app) -> None:
    from fastapi.testclient import TestClient
    app, engine = app
    client = TestClient(app)
    h = _auth_header(client)
    q = "I'm jane.doe@acme-mail.net: what does the 429 status code mean?"  # not a doc domain
    _ask(client, h, q)
    assert _ask(client, h, q)["cached_from"] is None and len(app.state.cache) == 0


def test_a_thumbs_down_stops_the_answer_being_reused(app) -> None:
    from fastapi.testclient import TestClient
    app, engine = app
    client = TestClient(app)
    h = _auth_header(client)
    _ask(client, h)
    replay = _ask(client, h)
    r = client.post("/feedback", json={"answer_id": replay["answer_id"], "rating": "down"},
                    headers=h)
    assert r.status_code == 202
    assert _ask(client, h)["cached_from"] is None and engine.generations == 2


def test_the_input_guard_still_runs_before_the_cache(app) -> None:
    from fastapi.testclient import TestClient
    app, engine = app
    client = TestClient(app)
    h = _auth_header(client)
    _ask(client, h)
    engine.refuse = True
    refused = _ask(client, h)
    assert refused["guardrails"]["input_flagged"] and refused["cached_from"] is None
