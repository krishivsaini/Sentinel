"""Export eval run history to JSON for the dashboard (§14, FR-D4).

`store.save()` already writes per-run JSON + `latest.json`. The dashboard's trend + runs views
also need the *history* across runs, which lives in the `eval_runs` SQLite table. This writes that
history to `dashboard/data/history.json` (newest last) so the dashboard stays a pure JSON consumer
— no live DB connection. Run after an eval, or any time, to refresh the trend.

    python scripts/export_dashboard_data.py
"""

from __future__ import annotations

import json
import sqlite3

from sentinel.config import DASHBOARD_DATA_DIR, EVAL_DB_PATH, settings


def main() -> None:
    if not EVAL_DB_PATH.exists():
        raise SystemExit(f"{EVAL_DB_PATH} not found — run an eval first.")
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    runs = conn.execute(
        "SELECT * FROM eval_runs ORDER BY timestamp ASC"
    ).fetchall()

    history = []
    for r in runs:
        # Count items from the run's own JSON export (eval_items' run_id gets reassigned across
        # resumes, so a DB count-by-run_id undercounts a superseded run).
        run_json = DASHBOARD_DATA_DIR / f"eval_{r['run_id']}.json"
        if run_json.exists():
            items = len(json.loads(run_json.read_text())["per_item"])
        else:
            items = conn.execute(
                "SELECT COUNT(*) FROM eval_items WHERE git_sha = ? AND run_id = ?",
                (r["git_sha"], r["run_id"]),
            ).fetchone()[0]
        faith = r["mean_faithfulness"]
        history.append(
            {
                "run_id": r["run_id"],
                "git_sha": r["git_sha"],
                "timestamp": r["timestamp"],
                "faithfulness": round(faith, 4),
                "answer_relevance": round(r["mean_answer_relevance"], 4),
                "context_recall": round(r["mean_context_recall"], 4),
                "retrieval_fail": r["retrieval_fail"],
                "generation_fail": r["generation_fail"],
                "items": items,
                "gate_pass": faith >= settings.faithfulness_threshold
                and items >= settings.ci_min_items,
            }
        )
    conn.close()

    DASHBOARD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DASHBOARD_DATA_DIR / "history.json"
    out.write_text(
        json.dumps(
            {
                "faithfulness_threshold": settings.faithfulness_threshold,
                "judge_model": settings.judge_model,
                "generation_model": settings.generation_model,
                "runs": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {len(history)} runs to {out}")


if __name__ == "__main__":
    main()
