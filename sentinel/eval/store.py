"""Eval persistence + dashboard export (§12) — Phase 3, Day 9.

Writes each EvalResult to SQLite (config.EVAL_DB_PATH) keyed by git SHA + run_id, and
exports one JSON file per run into dashboard/data/ for the read-only dashboard (§14).
The dashboard consumes the JSON export — no live DB connection (FR-D4).
"""

from __future__ import annotations

from sentinel.schema import EvalResult


def save(result: EvalResult) -> None:
    # TODO(Phase 3, Day 9): write to SQLite + export JSON snapshot for the dashboard.
    raise NotImplementedError("store.py is a Phase 3 (Day 9) stub — not yet implemented.")
