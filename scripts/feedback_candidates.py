"""Turn readers' thumbs-down into eval candidates: join POST /feedback to the answers it rates.

The service logs every answer (`query` / `query.refused`, PII scrubbed) and every rating
(`feedback`) as structured events, keyed by answer_id; nothing else stores state. This script
reads those events, either from Cloud Logging or from a saved log file, and writes one record per
rated answer to data/feedback/candidates.jsonl:

    python scripts/feedback_candidates.py --gcloud --project <gcp-project> --days 30
    python scripts/feedback_candidates.py --log-file service.log      # local uvicorn output

A thumbs-down is a lead, not a label. Each record starts as review "unreviewed"; a person reads
it and, if the answer really was wrong, writes a reference answer and the relevant passages into
data/ground_truth.jsonl (by hand, so the eval stays human-verified). Re-running keeps review
states already written. The file holds readers' questions, so it is git-ignored: only what a
person promotes into the ground truth gets committed.

Reads logs only: no model calls, no cost (Cloud Logging reads are free at this volume).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sentinel.config import DATA_DIR
from sentinel.ops import ANSWER_EVENTS, read_cloud_logging, read_log_file

OUT = DATA_DIR / "feedback" / "candidates.jsonl"


def join(events: list[dict]) -> list[dict]:
    """One record per rated answer (a reader's latest rating wins); feedback whose answer_id
    matches no logged answer is dropped."""
    answers = {e["answer_id"]: e for e in events
               if e.get("event") in ANSWER_EVENTS and e.get("answer_id")}
    ratings = sorted((e for e in events if e.get("event") == "feedback"),
                     key=lambda e: e.get("timestamp", ""))
    latest = {e["answer_id"]: e for e in ratings if e.get("answer_id") in answers}
    out = []
    for aid, fb in latest.items():
        a = answers[aid]
        out.append({
            "answer_id": aid,
            "rated_utc": fb.get("timestamp"),
            "rating": fb["rating"],
            "comment": fb.get("comment"),
            "question": a.get("question"),
            "answer": a.get("answer"),
            "outcome": "refused" if a["event"] == "query.refused" else a.get("outcome"),
            "chunk_ids": a.get("chunk_ids", []),
            "review": "unreviewed",
        })
    return sorted(out, key=lambda r: r["rated_utc"] or "")


def merge(new: list[dict], path: Path = OUT) -> list[dict]:
    """Add new records to the file, keeping the review state of ones already there."""
    old = {}
    if path.exists():
        old = {r["answer_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}
    for r in new:
        if r["answer_id"] in old:
            r["review"] = old[r["answer_id"]].get("review", "unreviewed")
        old[r["answer_id"]] = r
    merged = sorted(old.values(), key=lambda r: r["rated_utc"] or "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in merged))
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--gcloud", action="store_true", help="read the Cloud Run service's logs")
    src.add_argument("--log-file", type=Path, help="read JSON log lines from a file")
    ap.add_argument("--project", help="GCP project id (with --gcloud)")
    ap.add_argument("--days", type=int, default=30, help="how far back to read (with --gcloud)")
    args = ap.parse_args()
    if args.gcloud and not args.project:
        ap.error("--gcloud needs --project")

    events = (read_cloud_logging(args.project, args.days, ("query", "query.refused", "feedback"))
              if args.gcloud else read_log_file(args.log_file))
    records = merge(join(events))
    tally = Counter((r["outcome"], r["rating"]) for r in records)
    print(f"{len(records)} rated answers in {OUT.relative_to(DATA_DIR.parent)}")
    for outcome in ("answered", "abstained", "withheld", "refused"):
        up, down = tally[(outcome, "up")], tally[(outcome, "down")]
        if up or down:
            print(f"  {outcome:10} up {up:3}  down {down:3}")
    hints = {"answered": "wrong or badly grounded answer?",
             "abstained": "retrieval miss: the corpus may hold the answer",
             "withheld": "answer check false positive?",
             "refused": "input guard false positive?"}
    todo = [r for r in records if r["rating"] == "down" and r["review"] == "unreviewed"]
    for r in todo:
        question = (r["question"] or "")[:70]
        print(f"  - [{r['outcome']}] {question!r}  -> {hints.get(r['outcome'], '')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
