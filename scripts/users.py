"""Manage Sentinel's named accounts (sentinel.users): a JSON file of scrypt hashes and scopes.

    python scripts/users.py add alice --scopes public            # prompts for a password
    python scripts/users.py add bob --scopes public adversarial --generate
    python scripts/users.py list
    python scripts/users.py passwd alice                          # ends alice's current tokens
    python scripts/users.py scopes alice public                   # ditto, if scopes shrink
    python scripts/users.py disable alice | enable alice | remove alice

The file is config.users_file (SENTINEL_USERS_FILE), or data/users.json, which git ignores.
Passwords are read with getpass (or one line of stdin with --password-stdin) and stored only as
hashes; --generate makes a random one and prints it once. Nothing here calls a network service.

To use the file in production, upload it as a new version of the service's users secret (see
deploy/README.md); the running service picks the change up without a redeploy.
"""

from __future__ import annotations

import argparse
import getpass
import secrets
import sys
from pathlib import Path

from sentinel.config import DATA_DIR, settings
from sentinel.users import SCOPE_RE, USERNAME_RE, hash_password, read_users_file, write_users_file

DEFAULT_FILE = DATA_DIR / "users.json"
MIN_PASSWORD = 12


def _password(args) -> str:
    if args.generate:
        pw = secrets.token_urlsafe(18)
        print(f"generated password (shown once, store it now): {pw}")
        return pw
    if args.password_stdin:
        pw = sys.stdin.readline().rstrip("\n")
    else:
        pw = getpass.getpass("password: ")
        if getpass.getpass("again: ") != pw:
            sys.exit("passwords differ")
    if len(pw) < MIN_PASSWORD:
        sys.exit(f"password too short: at least {MIN_PASSWORD} characters")
    return pw


def _scopes(names: list[str]) -> list[str]:
    bad = [s for s in names if not SCOPE_RE.match(s)]
    if bad:
        sys.exit(f"bad scope name(s): {bad}")
    return sorted(set(names))


def _find(users: list[dict], name: str) -> dict:
    for u in users:
        if u["username"] == name:
            return u
    sys.exit(f"no account {name!r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, default=settings.users_file or DEFAULT_FILE)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("add", "passwd"):
        p = sub.add_parser(name)
        p.add_argument("username")
        p.add_argument("--generate", action="store_true", help="make a random password")
        p.add_argument("--password-stdin", action="store_true", help="read it from stdin")
        if name == "add":
            p.add_argument("--scopes", nargs="+", default=list(settings.default_scopes))
    p = sub.add_parser("scopes")
    p.add_argument("username")
    p.add_argument("scopes", nargs="+")
    for name in ("disable", "enable", "remove"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("list")
    args = ap.parse_args(argv)

    users = read_users_file(args.file) if args.file.exists() else []
    if args.cmd == "list":
        if not users:
            print(f"no accounts in {args.file}")
        for u in users:
            state = "disabled" if u.get("disabled") else "active"
            print(f"{u['username']:<24} {state:<9} scopes: {' '.join(u['scopes'])}")
        return 0

    name = args.username
    if args.cmd == "add":
        if not USERNAME_RE.match(name):
            sys.exit("username: lowercase letters, digits, . _ -; 2-32 characters")
        if name == settings.demo_user:
            sys.exit(f"{name!r} is the shared demo account's name")
        if any(u["username"] == name for u in users):
            sys.exit(f"{name!r} already exists (use passwd or scopes)")
        users.append({"username": name, "password_hash": hash_password(_password(args)),
                      "scopes": _scopes(args.scopes), "disabled": False})
    elif args.cmd == "passwd":
        _find(users, name)["password_hash"] = hash_password(_password(args))
    elif args.cmd == "scopes":
        _find(users, name)["scopes"] = _scopes(args.scopes)
    elif args.cmd in ("disable", "enable"):
        _find(users, name)["disabled"] = args.cmd == "disable"
    elif args.cmd == "remove":
        users.remove(_find(users, name))
    write_users_file(args.file, users)
    print(f"{args.cmd} {name}: {args.file} now has {len(users)} account(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
