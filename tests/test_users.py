"""Multi-user auth: the user store, scoped tokens, revocation, per-account rate limits.

Every test builds its own users file in tmp_path and its own app, so nothing touches
data/users.json or another test's rate limit state. No model or API calls (FakeEngine).
"""

from __future__ import annotations

import io
import json
import os

import pytest
from fastapi.testclient import TestClient

from sentinel.config import settings
from sentinel.serve import _rate_key, create_access_token, create_app, get_engine
from sentinel.users import (
    UserStore,
    hash_password,
    read_users_file,
    verify_password,
    write_users_file,
)
from tests.test_serve import FakeEngine, _parse_sse

ALICE_PW, BOB_PW = "alice-password-1", "bob-password-22"


def _write(path, users) -> None:
    write_users_file(path, users)
    # A later write in the same clock tick must still look changed to the store.
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))


@pytest.fixture
def users_file(tmp_path):
    path = tmp_path / "users.json"
    _write(path, [
        {"username": "alice", "password_hash": hash_password(ALICE_PW),
         "scopes": ["adversarial", "public"]},
        {"username": "bob", "password_hash": hash_password(BOB_PW), "scopes": ["public"]},
    ])
    return path


def _store(users_file) -> UserStore:
    return UserStore(demo_user=settings.demo_user, demo_password=settings.demo_password,
                     default_scopes=settings.default_scopes, users_file=users_file)


@pytest.fixture
def app(users_file):
    engine = FakeEngine()
    app = create_app(_store(users_file))
    app.dependency_overrides[get_engine] = lambda: engine
    app.state.engine = engine
    return app


def _token(client, username, password, scope=None):
    data = {"username": username, "password": password}
    if scope is not None:
        data["scope"] = scope
    return client.post("/token", data=data)


def _bearer(r) -> dict[str, str]:
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def _set(users_file, name, **changes) -> None:
    users = read_users_file(users_file)
    for u in users:
        if u["username"] == name:
            u.update(changes)
    _write(users_file, users)


# --------------------------------------------------------------------------- passwords


def test_passwords_are_stored_as_salted_scrypt_hashes() -> None:
    a, b = hash_password("same password"), hash_password("same password")
    assert a.startswith("scrypt$") and "same password" not in a
    assert a != b                                   # salted
    assert verify_password("same password", a) and verify_password("same password", b)
    assert not verify_password("same passwore", a)
    assert not verify_password("x", "plaintext") and not verify_password("x", "scrypt$bad")


@pytest.mark.parametrize("record,problem", [
    ({"username": "Alice", "password_hash": "scrypt$x", "scopes": ["public"]}, "username"),
    ({"username": "carol", "password_hash": "hunter2", "scopes": ["public"]}, "scrypt"),
    ({"username": "carol", "password_hash": "scrypt$x", "scopes": []}, "scopes"),
])
def test_malformed_users_file_is_refused(tmp_path, record, problem) -> None:
    path = tmp_path / "users.json"
    path.write_text(json.dumps({"users": [record]}))
    with pytest.raises(ValueError, match=problem):
        read_users_file(path)


def test_a_named_account_cannot_take_the_demo_accounts_name(tmp_path) -> None:
    path = tmp_path / "users.json"
    _write(path, [{"username": settings.demo_user, "password_hash": hash_password("x" * 12),
                   "scopes": ["adversarial"]}])
    with pytest.raises(ValueError, match="demo"):
        _store(path)


# --------------------------------------------------------------------------- tokens and scopes


def test_each_account_gets_its_own_scopes(app) -> None:
    c = TestClient(app)
    for name, pw, scopes in [(settings.demo_user, settings.demo_password, ["public"]),
                             ("alice", ALICE_PW, ["adversarial", "public"]),
                             ("bob", BOB_PW, ["public"])]:
        r = _token(c, name, pw)
        assert r.json()["scopes"] == scopes
        me = c.get("/me", headers=_bearer(r)).json()
        assert me == {"username": name, "scopes": scopes, "shared": name == settings.demo_user}
        c.post("/query", json={"question": "q"}, headers=_bearer(r))
        assert app.state.engine.accesses[-1].scopes == frozenset(scopes)


def test_a_token_can_ask_for_fewer_scopes_never_more(app) -> None:
    c = TestClient(app)
    narrowed = _token(c, "alice", ALICE_PW, scope="public")
    assert narrowed.json()["scopes"] == ["public"]
    c.post("/query", json={"question": "q"}, headers=_bearer(narrowed))
    assert app.state.engine.accesses[-1].scopes == frozenset({"public"})
    for name, pw in [("bob", BOB_PW), (settings.demo_user, settings.demo_password)]:
        r = _token(c, name, pw, scope="public adversarial")
        assert r.status_code == 400 and "invalid_scope" in r.json()["detail"]


def test_wrong_password_and_unknown_account_look_the_same(app) -> None:
    c = TestClient(app)
    wrong, unknown = _token(c, "alice", BOB_PW), _token(c, "mallory", ALICE_PW)
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()


def test_a_token_cannot_be_minted_beyond_the_accounts_scopes(users_file) -> None:
    bob = _store(users_file).get("bob")
    with pytest.raises(ValueError):
        create_access_token(bob, frozenset({"public", "adversarial"}))


def test_a_token_carrying_another_accounts_credential_tag_is_refused(app, users_file) -> None:
    from jose import jwt
    c = TestClient(app)
    claims = jwt.get_unverified_claims(_token(c, "bob", BOB_PW).json()["access_token"])
    claims["sub"], claims["scopes"] = "alice", ["adversarial", "public"]
    forged = jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    assert c.get("/me", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


# --------------------------------------------------------------------------- revocation


@pytest.mark.parametrize("change", [
    {"disabled": True},
    {"password_hash": hash_password("new-password-333")},
    {"scopes": ["public"]},                    # a scope taken away
])
def test_changing_the_account_ends_its_existing_tokens(app, users_file, change) -> None:
    c = TestClient(app)
    headers = _bearer(_token(c, "alice", ALICE_PW))
    assert c.get("/me", headers=headers).status_code == 200
    _set(users_file, "alice", **change)
    assert c.get("/me", headers=headers).status_code == 401
    assert c.post("/query", json={"question": "q"}, headers=headers).status_code == 401
    # bob is untouched
    assert c.get("/me", headers=_bearer(_token(c, "bob", BOB_PW))).status_code == 200


def test_a_disabled_account_cannot_sign_in(app, users_file) -> None:
    _set(users_file, "bob", disabled=True)
    assert _token(TestClient(app), "bob", BOB_PW).status_code == 401


def test_a_removed_users_file_leaves_only_the_demo_account(app, users_file) -> None:
    c = TestClient(app)
    headers = _bearer(_token(c, "alice", ALICE_PW))
    users_file.unlink()
    assert c.get("/me", headers=headers).status_code == 401
    assert _token(c, settings.demo_user, settings.demo_password).status_code == 200


# --------------------------------------------------------------------------- rate limits


def test_named_accounts_are_rate_limited_per_account_the_demo_per_ip(app) -> None:
    c = TestClient(app)

    class Req:   # just what _rate_key reads
        def __init__(self, headers):
            self.headers, self.app = headers, app
            self.client = type("C", (), {"host": "203.0.113.7"})()

    alice = _bearer(_token(c, "alice", ALICE_PW))
    demo = _bearer(_token(c, settings.demo_user, settings.demo_password))
    assert _rate_key(Req({k.lower(): v for k, v in alice.items()})) == "user:alice"
    assert _rate_key(Req({k.lower(): v for k, v in demo.items()})) == "203.0.113.7"
    assert _rate_key(Req({"authorization": "Bearer forged"})) == "203.0.113.7"


def test_sign_in_attempts_are_rate_limited(app) -> None:
    c = TestClient(app)
    limit = int(settings.token_rate_limit.split("/")[0])
    codes = [_token(c, "alice", "guess").status_code for _ in range(limit + 2)]
    assert codes[:limit] == [401] * limit and codes[limit:] == [429, 429]


def test_one_accounts_queries_do_not_use_up_anothers_limit(app) -> None:
    c = TestClient(app)
    alice, bob = _bearer(_token(c, "alice", ALICE_PW)), _bearer(_token(c, "bob", BOB_PW))
    limit = int(settings.rate_limit.split("/")[0])
    codes = [c.post("/query", json={"question": "q"}, headers=alice).status_code
             for _ in range(limit + 1)]
    assert codes[-1] == 429
    r = c.post("/query", json={"question": "q"}, headers=bob)
    assert r.status_code == 200 and _parse_sse(r.text)[-1][0] == "done"


# --------------------------------------------------------------------------- the CLI


def test_cli_manages_accounts_without_storing_passwords(tmp_path, monkeypatch, capsys) -> None:
    from scripts.users import main
    path = tmp_path / "users.json"
    monkeypatch.setattr("sys.stdin", io.StringIO("carol-password-9\n"))
    main(["--file", str(path), "add", "carol", "--scopes", "public", "--password-stdin"])
    assert "carol-password-9" not in path.read_text()
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    store = _store(path)
    assert store.authenticate("carol", "carol-password-9").scopes == frozenset({"public"})
    main(["--file", str(path), "disable", "carol"])
    assert store.authenticate("carol", "carol-password-9") is None
    main(["--file", str(path), "list"])
    assert "disabled" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO("short\n"))
    with pytest.raises(SystemExit):
        main(["--file", str(path), "passwd", "carol", "--password-stdin"])
