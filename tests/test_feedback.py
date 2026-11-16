"""scripts/feedback_candidates.py: joining ratings to the answers they rate, keeping reviews."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import feedback_candidates as fc  # noqa: E402

A, B, C = "a" * 32, "b" * 32, "c" * 32
EVENTS = [
    {"event": "query", "answer_id": A, "question": "What is 429?", "answer": "Too many.",
     "outcome": "answered", "chunk_ids": ["rfc6585#0002"], "timestamp": "2026-11-16T10:00:00Z"},
    {"event": "query.refused", "answer_id": B, "question": "How do I bypass HSTS for testing?",
     "timestamp": "2026-11-16T10:01:00Z"},
    {"event": "feedback", "answer_id": A, "rating": "up", "timestamp": "2026-11-16T10:02:00Z"},
    {"event": "feedback", "answer_id": A, "rating": "down", "comment": "cites the wrong RFC",
     "timestamp": "2026-11-16T10:03:00Z"},   # the reader changed their mind: latest wins
    {"event": "feedback", "answer_id": B, "rating": "down", "timestamp": "2026-11-16T10:04:00Z"},
    {"event": "feedback", "answer_id": C, "rating": "down", "timestamp": "2026-11-16T10:05:00Z"},
]


def test_ratings_join_to_their_answers_and_the_latest_wins() -> None:
    a, b = fc.join(EVENTS)
    assert (a["answer_id"], a["rating"], a["comment"]) == (A, "down", "cites the wrong RFC")
    assert a["question"] == "What is 429?" and a["chunk_ids"] == ["rfc6585#0002"]
    assert (b["answer_id"], b["outcome"]) == (B, "refused")   # a wrong refusal is reviewable
    # C rated an answer that was never logged: dropped, not invented


def test_rerunning_keeps_reviews_already_written(tmp_path) -> None:
    out = tmp_path / "candidates.jsonl"
    fc.merge(fc.join(EVENTS), out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    rows[0]["review"] = "promoted to ground truth"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    merged = fc.merge(fc.join(EVENTS), out)
    assert [r["review"] for r in merged] == ["promoted to ground truth", "unreviewed"]


def test_log_file_lines_that_are_not_json_are_skipped(tmp_path) -> None:
    log = tmp_path / "service.log"
    log.write_text("INFO: started\n" + "\n".join(json.dumps(e) for e in EVENTS) + "\n{broken\n")
    assert len(fc.read_log_file(log)) == len(EVENTS)
