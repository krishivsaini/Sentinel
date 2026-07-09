"""Failure attribution (§12, FR-E3) — the classifier used by Day 9's run, calibrated in Day 10.

For each item, decide whether a weak result is the retriever's fault or the generator's:
  - context_recall < config.retrieval_fail_recall_threshold  => RETRIEVAL failure
    (the relevant context was never retrieved — nothing downstream could fix it)
  - context recalled but faithfulness/relevance still low     => GENERATION failure
    (the right docs were retrieved; the answer is still wrong/unfaithful)
  - otherwise                                                 => pass

An item only *fails* at all if any metric dips below config.low_score_threshold; a fully
healthy item is a "pass" regardless of recall. This single distinction — retrieval vs
generation — is what separates real RAG work from prompt-tweaking, and its counts feed the
dashboard's attribution view (the README screenshot, §14).
"""

from __future__ import annotations

from sentinel.config import settings
from sentinel.schema import EvalItemResult


def classify(item: EvalItemResult) -> str:
    """Return "pass" | "retrieval_fail" | "generation_fail" for one scored item.

    Two independent signals, checked in priority order:
      1. retrieval missed  — context_recall < retrieval_fail_recall_threshold: the relevant
         context was never retrieved, so nothing downstream could have fixed it. This is a
         RETRIEVAL failure whatever the answer looked like (a good-looking answer over missing
         context isn't grounded — it's luck or hallucination).
      2. answer weak        — faithfulness or answer_relevance < low_score_threshold while the
         context WAS recalled: the right docs were there and the answer is still wrong. A
         GENERATION failure.
    Anything else passes. (Recall alone, above the retrieval cutoff, does not fail an item — the
    generic low_score_threshold governs only answer quality.)
    """
    if item.context_recall < settings.retrieval_fail_recall_threshold:
        return "retrieval_fail"
    if (
        item.faithfulness < settings.low_score_threshold
        or item.answer_relevance < settings.low_score_threshold
    ):
        return "generation_fail"
    return "pass"
