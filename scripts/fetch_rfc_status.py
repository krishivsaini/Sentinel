"""Record each corpus RFC's standing — obsoleted by, updated by, status — from the RFC Editor.

RFCs are never edited after publication; they are replaced. RFC 2818 (HTTP over TLS) is still
in the corpus but obsoleted by RFC 9110, and an answer that cites it as current is wrong in a
way faithfulness can't catch: the text is quoted accurately, it just isn't the standard any
more. The relationships pointing *forward* (obsoleted_by, updated_by) aren't in an RFC's own
text — the RFC that replaced it came later — so they are taken from the RFC Editor's metadata
for each RFC and written to data/rfc_status.json, which the service reads at startup.

Stdlib only, polite (1 request/s), and every record is checked against the RFC it describes.

Run:  python scripts/fetch_rfc_status.py           # (re)write data/rfc_status.json
      python scripts/fetch_rfc_status.py --check   # validate the committed file, no network
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import POLITE_DELAY_S, DownloadError, _now, download, load_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
STATUS_PATH = REPO_ROOT / "data" / "rfc_status.json"
FIELDS = ("obsoleted_by", "updated_by", "obsoletes", "updates")


def _numbers(refs: list[str]) -> list[int]:
    """["RFC9110", "RFC0793"] -> [9110, 793]; non-RFC references (BCP/STD/FYI) are dropped."""
    return sorted(int(r[3:]) for r in refs if r.upper().startswith("RFC") and r[3:].isdigit())


def fetch_one(base: str, number: int) -> dict:
    meta = json.loads(download(f"{base}rfc{number}.json"))
    if meta.get("doc_id", "").upper() != f"RFC{number:04d}":
        raise ValueError(f"metadata names {meta.get('doc_id')!r}, expected RFC{number:04d}")
    rec = {f: _numbers(meta.get(f) or []) for f in FIELDS}
    return {"status": meta.get("status", "").upper(), **rec}


def check(records: dict, manifest: dict) -> list[str]:
    errors = []
    expected = {f"rfc{e['number']}" for e in manifest["rfcs"]}
    if set(records) != expected:
        errors.append(f"RFC set differs from the manifest: {sorted(set(records) ^ expected)}")
    for doc_id, rec in records.items():
        missing = [f for f in ("status", *FIELDS) if f not in rec]
        if missing:
            errors.append(f"{doc_id}: missing {missing}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="validate the committed file only")
    args = ap.parse_args()
    manifest = load_manifest()

    if args.check:
        errors = check(json.loads(STATUS_PATH.read_text())["rfcs"], manifest)
        print("\n".join(errors) or f"OK: {STATUS_PATH.relative_to(REPO_ROOT)}")
        return 1 if errors else 0

    records, failed = {}, 0
    for i, entry in enumerate(manifest["rfcs"]):
        if i:
            time.sleep(POLITE_DELAY_S)
        number = entry["number"]
        try:
            records[f"rfc{number}"] = rec = fetch_one(manifest["source_base_url"], number)
        except (DownloadError, ValueError, json.JSONDecodeError) as exc:
            print(f"RFC {number}: FAIL {exc}")
            failed += 1
            continue
        flags = ", ".join(f"{f} {rec[f]}" for f in ("obsoleted_by", "updated_by") if rec[f])
        print(f"RFC {number:<5} {rec['status']:<22} {flags}")
    if failed:
        print(f"{failed} failed; {STATUS_PATH.name} not written")
        return 1

    errors = check(records, manifest)
    if errors:
        print("\n".join(errors))
        return 1
    source = f"{manifest['source_base_url']}rfc<N>.json"
    payload = {"source": source, "fetched_utc": _now(), "rfcs": records}
    STATUS_PATH.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {STATUS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
