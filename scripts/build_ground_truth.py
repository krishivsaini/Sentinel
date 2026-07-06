"""Ground-truth draft generator (§7, §15 day 8) — Phase 3, Day 8.

LLM-drafts {question, reference_answer, relevant_contexts} triples from the corpus, to be
written to data/ground_truth.jsonl. CRITICAL: every drafted item MUST then be hand-corrected
(FR-GT2). Fully synthetic gold tests LLM-vs-LLM agreement, not retrieval quality — this
script only produces a *draft* for human correction.

Target: 80–120 hand-corrected triples, including adversarial version-specific questions that
exercise BM25 (e.g. "RFC 7231 vs RFC 9110" on the same status code).

Run:  python -m scripts.build_ground_truth
"""

from __future__ import annotations


def main() -> None:
    # TODO(Phase 3, Day 8): draft triples from the corpus for hand-correction.
    raise NotImplementedError("build_ground_truth.py is a Phase 3 (Day 8) stub.")


if __name__ == "__main__":
    main()
