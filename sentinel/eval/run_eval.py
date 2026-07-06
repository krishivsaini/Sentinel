"""Ragas evaluation pipeline (§12) — the headline — Phase 3, Day 9.

Runs the full retrieve -> rerank -> generate pipeline over every item in ground_truth.jsonl,
scores each with Ragas (faithfulness, answer relevance, context recall), runs failure
attribution, writes an EvalResult to SQLite tagged with the git SHA + exports JSON, and
prints a summary table with the mean faithfulness (the CI gate reads this).

Judge calls are wrapped in tenacity backoff (config.judge_max_retries) so free-tier 429s
retry rather than crash the run (NFR-5).

Run:  python -m sentinel.eval.run_eval [--subset N]   (--subset drives the CI gate, §13)
"""

from __future__ import annotations


def main() -> None:
    # TODO(Phase 3, Day 9): Ragas scoring over ground truth + attribution + store + summary.
    raise NotImplementedError("run_eval.py is a Phase 3 (Day 9) stub — not yet implemented.")


if __name__ == "__main__":
    main()
