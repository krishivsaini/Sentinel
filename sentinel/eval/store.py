"""Eval persistence + dashboard export (§12) — Phase 3, Day 9.

Two responsibilities:

1. **Checkpointing (resumability).** Every scored item is written to SQLite
   (`config.EVAL_DB_PATH`) immediately, keyed by (git_sha, question). A run that dies mid-way
   to a free-tier 429 storm resumes from the DB instead of restarting and re-burning quota
   (NFR-5) — `load_scored_items()` tells `run_eval` which questions are already done.

2. **Export.** `save()` writes the run summary row and dumps the full `EvalResult` as one JSON
   file per run into `dashboard/data/` for the read-only dashboard (§14) — the dashboard
   consumes JSON, never a live DB connection (FR-D4).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sentinel.config import DASHBOARD_DATA_DIR, EVAL_DB_PATH
from sentinel.schema import EvalItemResult, EvalResult, RetrievedChunk

_ITEMS_DDL = """
CREATE TABLE IF NOT EXISTS eval_items (
    git_sha          TEXT NOT NULL,
    question         TEXT NOT NULL,
    run_id           TEXT NOT NULL,
    faithfulness     REAL NOT NULL,
    answer_relevance REAL NOT NULL,
    context_recall   REAL NOT NULL,
    attribution      TEXT,
    generated_answer TEXT NOT NULL,
    retrieved_json   TEXT NOT NULL,
    ts               TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (git_sha, question)
)
"""

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id                TEXT PRIMARY KEY,
    git_sha               TEXT NOT NULL,
    timestamp             TEXT NOT NULL,
    mean_faithfulness     REAL NOT NULL,
    mean_answer_relevance REAL NOT NULL,
    mean_context_recall   REAL NOT NULL,
    retrieval_fail        INTEGER NOT NULL,
    generation_fail       INTEGER NOT NULL,
    json_path             TEXT NOT NULL
)
"""


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    # Look up EVAL_DB_PATH at call time (not as a default) so tests can monkeypatch it.
    conn = sqlite3.connect(db_path or EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(_ITEMS_DDL)
    conn.execute(_RUNS_DDL)
    return conn


# --------------------------------------------------------------------------- checkpointing


def upsert_item(git_sha: str, run_id: str, item: EvalItemResult) -> None:
    """Persist one scored item immediately (the resume checkpoint). Idempotent: re-scoring the
    same (git_sha, question) overwrites, so a resumed run's fresh scores win."""
    retrieved_json = "[" + ",".join(c.model_dump_json() for c in item.retrieved) + "]"
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO eval_items
                (git_sha, question, run_id, faithfulness, answer_relevance, context_recall,
                 attribution, generated_answer, retrieved_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(git_sha, question) DO UPDATE SET
                run_id=excluded.run_id,
                faithfulness=excluded.faithfulness,
                answer_relevance=excluded.answer_relevance,
                context_recall=excluded.context_recall,
                attribution=excluded.attribution,
                generated_answer=excluded.generated_answer,
                retrieved_json=excluded.retrieved_json,
                ts=datetime('now')
            """,
            (
                git_sha,
                item.question,
                run_id,
                item.faithfulness,
                item.answer_relevance,
                item.context_recall,
                item.attribution,
                item.generated_answer,
                retrieved_json,
            ),
        )


def load_scored_items(git_sha: str) -> dict[str, EvalItemResult]:
    """Return already-scored items for this git_sha, keyed by question, so `run_eval` can skip
    them on a resume. Empty dict when nothing has been scored for this SHA yet."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM eval_items WHERE git_sha = ?", (git_sha,)
        ).fetchall()
    out: dict[str, EvalItemResult] = {}
    for r in rows:
        out[r["question"]] = EvalItemResult(
            question=r["question"],
            faithfulness=r["faithfulness"],
            answer_relevance=r["answer_relevance"],
            context_recall=r["context_recall"],
            attribution=r["attribution"],
            generated_answer=r["generated_answer"],
            retrieved=[
                RetrievedChunk.model_validate(c)
                for c in json.loads(r["retrieved_json"])
            ],
        )
    return out


# --------------------------------------------------------------------------- run summary + export


def save(result: EvalResult) -> Path:
    """Write the run summary row and export the full EvalResult as JSON for the dashboard.
    Returns the JSON path. Per-item rows are already checkpointed via upsert_item during the run;
    this backfills any not yet written (e.g. reused-from-a-prior-SHA items) and records the run."""
    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DASHBOARD_DATA_DIR / f"eval_{result.run_id}.json"
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    # Also refresh a stable "latest" pointer the dashboard/README can link without a run_id.
    (DASHBOARD_DATA_DIR / "latest.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8"
    )

    with _connect() as conn:
        for item in result.per_item:
            _upsert_item_conn(conn, result.git_sha, result.run_id, item)
        conn.execute(
            """
            INSERT INTO eval_runs
                (run_id, git_sha, timestamp, mean_faithfulness, mean_answer_relevance,
                 mean_context_recall, retrieval_fail, generation_fail, json_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                git_sha=excluded.git_sha,
                timestamp=excluded.timestamp,
                mean_faithfulness=excluded.mean_faithfulness,
                mean_answer_relevance=excluded.mean_answer_relevance,
                mean_context_recall=excluded.mean_context_recall,
                retrieval_fail=excluded.retrieval_fail,
                generation_fail=excluded.generation_fail,
                json_path=excluded.json_path
            """,
            (
                result.run_id,
                result.git_sha,
                result.timestamp.isoformat(),
                result.means.faithfulness,
                result.means.answer_relevance,
                result.means.context_recall,
                result.attribution_counts.retrieval_fail,
                result.attribution_counts.generation_fail,
                str(json_path),
            ),
        )
    return json_path


def _upsert_item_conn(
    conn: sqlite3.Connection, git_sha: str, run_id: str, item: EvalItemResult
) -> None:
    """upsert_item body reusing an open connection (used inside save's transaction)."""
    retrieved_json = "[" + ",".join(c.model_dump_json() for c in item.retrieved) + "]"
    conn.execute(
        """
        INSERT INTO eval_items
            (git_sha, question, run_id, faithfulness, answer_relevance, context_recall,
             attribution, generated_answer, retrieved_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(git_sha, question) DO UPDATE SET
            run_id=excluded.run_id,
            faithfulness=excluded.faithfulness,
            answer_relevance=excluded.answer_relevance,
            context_recall=excluded.context_recall,
            attribution=excluded.attribution,
            generated_answer=excluded.generated_answer,
            retrieved_json=excluded.retrieved_json,
            ts=datetime('now')
        """,
        (
            git_sha,
            item.question,
            run_id,
            item.faithfulness,
            item.answer_relevance,
            item.context_recall,
            item.attribution,
            item.generated_answer,
            retrieved_json,
        ),
    )
