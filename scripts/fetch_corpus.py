"""Fetch the Sentinel corpus (IETF web-protocol RFCs) verbatim from the RFC Editor.

Reads data/corpus_manifest.json and downloads each RFC's plain-text into data/corpus/
as rfc<NNNN>.txt. Text is stored VERBATIM (no cleaning, no derivatives) - the IETF Trust
permits whole-RFC reproduction; ingestion does its own cleaning downstream (§8).

Idempotent (FR-I3 spirit): existing non-empty files are skipped unless --force. Every
download is verified (HTTP 200, non-empty, the file names its own RFC number) and recorded
in data/corpus/PROVENANCE.json with sha256 + byte count + fetch timestamp, so the exact
corpus that produced any index is attributable.

Stdlib only - runnable before `uv sync`.

Run:  python scripts/fetch_corpus.py            # fetch missing
      python scripts/fetch_corpus.py --force    # re-fetch everything
      python scripts/fetch_corpus.py --check     # verify local files, no network
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "corpus_manifest.json"
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
PROVENANCE_PATH = CORPUS_DIR / "PROVENANCE.json"

USER_AGENT = "Sentinel-RAG-corpus-fetcher/1.0 (+https://github.com/krishivsaini/Sentinel)"
POLITE_DELAY_S = 1.0          # be a good citizen of rfc-editor.org
CONNECT_TIMEOUT_S = 30
MIN_BYTES = 2_000             # every real RFC is far larger; guards against error pages


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> dict:
    with MANIFEST_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def rfc_url(base: str, number: int) -> str:
    return f"{base}rfc{number}.txt"


def local_path(number: int) -> Path:
    return CORPUS_DIR / f"rfc{number}.txt"


def verify_bytes(number: int, data: bytes) -> str | None:
    """Return an error string if the payload doesn't look like the right RFC, else None."""
    if len(data) < MIN_BYTES:
        return f"too small ({len(data)} bytes) - likely an error page"
    head = data[:4000].decode("utf-8", errors="replace")
    if f"RFC {number}" not in head and f"Request for Comments: {number}" not in head:
        return f"header does not name RFC {number}"
    return None


class DownloadError(RuntimeError):
    """Raised when a URL cannot be fetched cleanly."""


def download(url: str) -> bytes:
    """Fetch via curl so the system trust store is used (portable across macOS/Linux;
    Python's urllib can't see the macOS keychain, which breaks TLS verification there)."""
    try:
        proc = subprocess.run(
            ["curl", "-sSL", "--fail", "--max-time", str(CONNECT_TIMEOUT_S),
             "-A", USER_AGENT, url],
            capture_output=True, check=True, timeout=CONNECT_TIMEOUT_S + 10,
        )
    except FileNotFoundError as exc:
        raise DownloadError("curl not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise DownloadError(f"timeout after {CONNECT_TIMEOUT_S}s") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace").strip()
        raise DownloadError(f"curl exit {exc.returncode}: {stderr}") from exc
    return proc.stdout


def cmd_fetch(manifest: dict, force: bool) -> int:
    base = manifest["source_base_url"]
    rfcs = manifest["rfcs"]
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    provenance: dict[str, dict] = {}
    fetched = skipped = failed = 0

    for i, entry in enumerate(rfcs, 1):
        number = entry["number"]
        path = local_path(number)
        tag = f"[{i:>2}/{len(rfcs)}] RFC {number:<4} {entry['title'][:52]}"

        if path.exists() and path.stat().st_size >= MIN_BYTES and not force:
            data = path.read_bytes()
            provenance[f"rfc{number}.txt"] = {
                "number": number, "title": entry["title"], "cluster": entry["cluster"],
                "url": rfc_url(base, number), "bytes": len(data), "sha256": _sha256(data),
                "fetched_utc": None, "status": "skipped-exists",
            }
            print(f"{tag}  SKIP (exists, {len(data):,} B)")
            skipped += 1
            continue

        url = rfc_url(base, number)
        try:
            data = download(url)
        except DownloadError as exc:
            print(f"{tag}  FAIL download: {exc}")
            failed += 1
            continue

        err = verify_bytes(number, data)
        if err:
            print(f"{tag}  FAIL verify: {err}")
            failed += 1
            continue

        path.write_bytes(data)
        provenance[f"rfc{number}.txt"] = {
            "number": number, "title": entry["title"], "cluster": entry["cluster"],
            "url": url, "bytes": len(data), "sha256": _sha256(data),
            "fetched_utc": _now(), "status": "fetched",
        }
        print(f"{tag}  OK   ({len(data):,} B)")
        fetched += 1
        time.sleep(POLITE_DELAY_S)

    manifest["fetched_utc"] = _now()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    total_bytes = sum(p["bytes"] for p in provenance.values())
    PROVENANCE_PATH.write_text(
        json.dumps(
            {
                "corpus_name": manifest["corpus_name"],
                "generated_utc": _now(),
                "count": len(provenance),
                "total_bytes": total_bytes,
                "files": provenance,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"\nfetched={fetched}  skipped={skipped}  failed={failed}  "
        f"total={len(provenance)}/{len(rfcs)}  bytes={total_bytes:,}"
    )
    print(f"provenance -> {PROVENANCE_PATH.relative_to(REPO_ROOT)}")
    return 1 if failed else 0


def cmd_check(manifest: dict) -> int:
    """Offline integrity pass: every manifest RFC is present, non-empty, self-naming."""
    problems = 0
    for entry in manifest["rfcs"]:
        number = entry["number"]
        path = local_path(number)
        if not path.exists():
            print(f"MISSING  rfc{number}.txt")
            problems += 1
            continue
        err = verify_bytes(number, path.read_bytes())
        if err:
            print(f"BAD      rfc{number}.txt: {err}")
            problems += 1
    print("check: OK" if not problems else f"check: {problems} problem(s)")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if a file exists")
    ap.add_argument("--check", action="store_true", help="verify local files only (no network)")
    args = ap.parse_args()

    manifest = load_manifest()
    if args.check:
        return cmd_check(manifest)
    return cmd_fetch(manifest, force=args.force)


if __name__ == "__main__":
    sys.exit(main())
