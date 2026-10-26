"""Comparing rerankers on the ground truth: LLM-free ranking metrics and their uncertainty.

With 48 questions a difference of one or two questions moves a mean noticeably, so a comparison
reports, besides the means: a paired bootstrap 95% interval of the difference (resampling
questions, keeping each question's pair of scores together), and how many questions got
better, worse or stayed the same — by the rank of the first relevant passage.
"""

from __future__ import annotations

import math

import numpy as np

METRICS = ("mrr@10", "ndcg@10", "recall@5", "hit@5")


def rank_metrics(ranked: list[str], relevant: set[str]) -> dict[str, float]:
    """MRR@10, nDCG@10 (binary relevance), Recall@5 and Hit@5 of one ranking."""
    first = next((i for i, c in enumerate(ranked[:10], 1) if c in relevant), None)
    dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(ranked[:10], 1) if c in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), 10) + 1))
    top5 = set(ranked[:5])
    return {
        "mrr@10": 1.0 / first if first else 0.0,
        "ndcg@10": dcg / idcg if idcg else 0.0,
        "recall@5": len(relevant & top5) / len(relevant),
        "hit@5": float(bool(relevant & top5)),
    }


def first_relevant_rank(ranked: list[str], relevant: set[str]) -> int | None:
    return next((i for i, c in enumerate(ranked, 1) if c in relevant), None)


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> dict:
    """Mean of b - a with a 95% percentile interval, resampling questions with replacement."""
    diff = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = diff[rng.integers(0, len(diff), size=(n, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": round(float(diff.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def wins_losses(base_ranks: list[int | None], new_ranks: list[int | None]) -> dict[str, int]:
    """Per question: did the first relevant passage move up, down, or stay (None = not found)?"""
    def key(r):
        return math.inf if r is None else r
    out = {"better": 0, "worse": 0, "same": 0}
    for a, b in zip(base_ranks, new_ranks, strict=True):
        out["better" if key(b) < key(a) else "worse" if key(b) > key(a) else "same"] += 1
    return out
