"""Failure attribution (§12, FR-E3) — Phase 3, Day 10.

For each low-scoring item (< config.low_score_threshold), classify:
  - context_recall < config.retrieval_fail_recall_threshold  => RETRIEVAL failure
    (the relevant context was never retrieved)
  - context recalled but faithfulness/relevance still low     => GENERATION failure
    (right docs retrieved, bad answer)

This single distinction is what separates serious RAG work from prompt-tweaking. Counts of
each feed the dashboard's attribution view (the README screenshot, §14).
"""

from __future__ import annotations

from sentinel.schema import EvalItemResult


def classify(item: EvalItemResult) -> str:
    """Return "pass" | "retrieval_fail" | "generation_fail"."""
    # TODO(Phase 3, Day 10): implement the recall-threshold classifier.
    raise NotImplementedError("attribution.py is a Phase 3 (Day 10) stub — not yet implemented.")
