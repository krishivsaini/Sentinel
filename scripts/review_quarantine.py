"""Review the real RFC chunks injection screening quarantines, and record reviewed exceptions.

The detectors flag 3 of 2,888 genuine RFC chunks (data/guardrails/report.json). Retuning a
detector to pass them would loosen it for everything; instead a person reads the flagged text
and, if it is ordinary RFC content, records an exception in
data/guardrails/screening_exceptions.json. The exception is pinned to the exact text, its
document and the detectors that flagged it (sentinel.guardrails.injection.ScreeningException),
so it can't be reused by an edited passage or by the same words planted elsewhere.

    python scripts/review_quarantine.py                       # list flagged public chunks
    python scripts/review_quarantine.py --show rfc8446#0143   # print one in full
    python scripts/review_quarantine.py --release rfc8446#0143 --reason "TLS 1.3 C.3 checklist"

Reads the built index and cached verdicts only: no API calls. Only public-scope chunks can be
released; the adversarial corpus is never eligible. Rebuild the index afterwards
(python -m sentinel.ingest) for the change to reach retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from sentinel.guardrails.injection import (
    SCREENING_EXCEPTIONS_PATH,
    load_exceptions,
    screen_texts,
    text_key,
)
from sentinel.retrieve import _load_index

HEADER = (
    "Reviewed screening exceptions: RFC chunks a detector flagged that a person read and found "
    "to be ordinary RFC text. Each releases only that exact text (sha256), from that document, "
    "while no detector outside flagged_by objects. Written by scripts/review_quarantine.py."
)


def _flagged():
    """(chunk, screening) for every public chunk any detector flags, in index order."""
    chunks = [c for c in _load_index().id2chunk.values() if c.scope == "public"]
    screenings = screen_texts([c.text for c in chunks], offline=True)
    return [(c, s) for c, s in zip(chunks, screenings, strict=True) if s.flagged_by]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--show", metavar="CHUNK_ID", help="print a flagged chunk in full")
    ap.add_argument("--release", metavar="CHUNK_ID", help="record a reviewed exception")
    ap.add_argument("--reason", help="why the text is ordinary RFC content (with --release)")
    args = ap.parse_args()

    exceptions = load_exceptions()
    flagged = _flagged()
    by_id = {c.chunk_id: (c, s) for c, s in flagged}

    if args.show or args.release:
        cid = args.show or args.release
        if cid not in by_id:
            print(f"{cid} is not a flagged public chunk")
            return 1
        chunk, screening = by_id[cid]
        if args.show:
            print(f"{cid}  flagged by {', '.join(screening.flagged_by)}\n\n{chunk.text}")
            return 0
        if not args.reason:
            print("--release needs --reason")
            return 1
        if any(e.covers(chunk, screening) for e in exceptions):
            print(f"{cid} is already released")
            return 0
        data = (json.loads(SCREENING_EXCEPTIONS_PATH.read_text(encoding="utf-8"))
                if SCREENING_EXCEPTIONS_PATH.exists() else {"_about": HEADER, "exceptions": []})
        data["exceptions"].append({
            "chunk_id": cid,
            "doc_id": chunk.doc_id,
            "sha256": text_key(chunk.text),
            "flagged_by": sorted(screening.flagged_by),
            "reason": args.reason,
            "reviewed_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        })
        SCREENING_EXCEPTIONS_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"released {cid}; rebuild the index for retrieval to see it")
        return 0

    print(f"{len(flagged)} public chunks flagged by screening:")
    for c, s in flagged:
        state = "released" if any(e.covers(c, s) for e in exceptions) else "QUARANTINED"
        short = ", ".join(d.split(":")[0] + ":" + d.split(":")[-1] for d in s.flagged_by)
        print(f"  {c.chunk_id:16} {state:12} {short}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
