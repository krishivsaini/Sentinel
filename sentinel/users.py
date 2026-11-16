"""Accounts: who may sign in, and which document scopes each may retrieve.

Two kinds of account:

  * the shared demo account (config.demo_user / demo_password). Its password is public, since
    the dashboard signs in with it, so it only ever gets config.default_scopes and is rate
    limited per IP like an anonymous caller;
  * named accounts, read from a JSON file (config.users_file). Each has its own password,
    stored as a salted scrypt hash, its own scopes, and its own rate limit bucket.

    {"users": [{"username": "alice", "password_hash": "scrypt$16384$8$1$<salt>$<key>",
                "scopes": ["public"], "disabled": false}]}

The file holds hashes, never passwords; manage it with scripts/users.py. In production it is a
Secret Manager secret mounted into the container, so it isn't in the image or the repository.
The store re-reads the file when it changes, which makes disabling an account, removing a
scope or changing a password take effect on the next request; no redeploy, no database.

Tokens carry a credential tag: an HMAC (under the JWT secret) of the account's password hash
and scopes. A token whose tag no longer matches the account, because the password or the
scopes changed, is rejected, so a token can never outlive the access it was issued for.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path

# scrypt cost: ~16 MiB and a few tens of ms per hash. Enough to make offline guessing of a
# leaked file slow, cheap enough for a sign-in endpoint that is rate limited anyway.
_N, _R, _P, _DKLEN = 2**14, 8, 1, 32
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,31}$")
SCOPE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=_DKLEN)
    b64 = lambda b: base64.b64encode(b).decode()   # noqa: E731
    return f"scrypt${_N}${_R}${_P}${b64(salt)}${b64(key)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, n, r, p, salt, key = encoded.split("$")
        if algo != "scrypt":
            return False
        expected = base64.b64decode(key)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r),
                             p=int(p), dklen=len(expected))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(got, expected)


# Checked against when the username is unknown, so a wrong username costs the same time as a
# wrong password and response timing doesn't reveal which accounts exist.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


@dataclass(frozen=True)
class User:
    username: str
    scopes: frozenset[str]
    password_hash: str = field(repr=False)
    disabled: bool = False
    shared: bool = False        # the public demo account: default scopes, per-IP rate limit

    def credential_tag(self, secret: str) -> str:
        """Changes whenever the password or the scopes change; reveals neither without the
        JWT secret (and anyone holding that can mint tokens anyway)."""
        msg = "\0".join([self.username, self.password_hash, *sorted(self.scopes)]).encode()
        return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()[:22]


def validate(user: dict) -> None:
    """Raise ValueError if a record in the users file is malformed."""
    name = user.get("username", "")
    if not USERNAME_RE.match(name):
        raise ValueError(f"bad username {name!r}: lowercase letters, digits, . _ -; 2-32 chars")
    if not str(user.get("password_hash", "")).startswith("scrypt$"):
        raise ValueError(f"{name}: password_hash must be a scrypt hash (use scripts/users.py)")
    scopes = user.get("scopes")
    if not isinstance(scopes, list) or not scopes or not all(
            isinstance(s, str) and SCOPE_RE.match(s) for s in scopes):
        raise ValueError(f"{name}: scopes must be a non-empty list of scope names")


def read_users_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    users = data.get("users", [])
    seen: set[str] = set()
    for u in users:
        validate(u)
        if u["username"] in seen:
            raise ValueError(f"duplicate username {u['username']!r}")
        seen.add(u["username"])
    return users


def write_users_file(path: Path, users: list[dict]) -> None:
    for u in users:
        validate(u)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps({"users": sorted(users, key=lambda u: u["username"])}, indent=2)
                   + "\n", encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(path)


class UserStore:
    """The demo account plus the accounts in the users file, re-read when the file changes."""

    def __init__(self, *, demo_user: str | None, demo_password: str | None,
                 default_scopes: tuple[str, ...], users_file: Path | None = None) -> None:
        self._demo: User | None = None
        if demo_user and demo_password:
            # The demo password is public and lives in config, so its "hash" is derived
            # deterministically: the credential tag then survives restarts, and a token from
            # one instance stays valid on the next.
            digest = hashlib.sha256(f"demo\0{demo_user}\0{demo_password}".encode()).hexdigest()
            self._demo = User(username=demo_user, scopes=frozenset(default_scopes),
                              password_hash=f"demo${digest}", shared=True)
            self._demo_password = demo_password
        self._path = users_file
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self._named: dict[str, User] = {}
        self._reload()

    def _reload(self) -> None:
        if self._path is None:
            return
        try:
            st = self._path.stat()
        except FileNotFoundError:
            stamp = None
        else:
            stamp = (st.st_mtime_ns, st.st_size)
        if stamp == self._stamp:
            return
        with self._lock:
            if stamp == self._stamp:
                return
            named: dict[str, User] = {}
            if stamp is not None:
                for u in read_users_file(self._path):
                    if self._demo and u["username"] == self._demo.username:
                        raise ValueError(f"{u['username']!r} is the demo account's name")
                    named[u["username"]] = User(
                        username=u["username"], scopes=frozenset(u["scopes"]),
                        password_hash=u["password_hash"], disabled=bool(u.get("disabled")))
            self._named, self._stamp = named, stamp

    def get(self, username: str) -> User | None:
        """The account, if it exists and may sign in."""
        self._reload()
        if self._demo and username == self._demo.username:
            return self._demo
        user = self._named.get(username)
        return user if user and not user.disabled else None

    def authenticate(self, username: str, password: str) -> User | None:
        self._reload()
        if self._demo and username == self._demo.username:
            ok = hmac.compare_digest(password.encode(), self._demo_password.encode())
            return self._demo if ok else None
        user = self._named.get(username)
        ok = verify_password(password, user.password_hash if user else _DUMMY_HASH)
        return user if ok and user and not user.disabled else None

    def __len__(self) -> int:
        self._reload()
        return len(self._named) + bool(self._demo)
