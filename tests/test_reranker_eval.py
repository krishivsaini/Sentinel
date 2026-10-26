"""The ranking metrics and the paired comparison the reranker verdict rests on."""

from __future__ import annotations

import math

import numpy as np
import pytest

from sentinel.finetune.evaluate import paired_bootstrap, rank_metrics, wins_losses


def test_rank_metrics_by_hand() -> None:
    m = rank_metrics(["a", "x", "b", "y", "z", "c"], {"b", "c"})
    assert m["mrr@10"] == pytest.approx(1 / 3)                       # first relevant at rank 3
    dcg = 1 / math.log2(4) + 1 / math.log2(7)                         # relevant at 3 and 6
    assert m["ndcg@10"] == pytest.approx(dcg / (1 + 1 / math.log2(3)))
    assert m["recall@5"] == 0.5 and m["hit@5"] == 1.0


def test_nothing_relevant_in_the_top_ten_scores_zero() -> None:
    m = rank_metrics([f"n{i}" for i in range(12)] + ["g"], {"g"})
    assert m == {"mrr@10": 0.0, "ndcg@10": 0.0, "recall@5": 0.0, "hit@5": 0.0}


def test_identical_rankers_differ_by_exactly_zero() -> None:
    a = np.array([1.0, 0.5, 0.0, 1 / 3])
    assert paired_bootstrap(a, a.copy(), n=500) == {"diff": 0.0, "ci95": [0.0, 0.0]}


def test_the_interval_brackets_a_consistent_gain_and_is_seeded() -> None:
    rng = np.random.default_rng(1)
    a = rng.random(48)
    b = a + 0.05 + rng.normal(0, 0.01, 48)
    r = paired_bootstrap(a, b, n=2000, seed=3)
    assert 0.0 < r["ci95"][0] <= r["diff"] <= r["ci95"][1]
    assert r == paired_bootstrap(a, b, n=2000, seed=3)


def test_wins_and_losses_treat_not_found_as_worst() -> None:
    assert wins_losses([1, 3, None, 2], [1, 2, 4, None]) == {"better": 2, "worse": 1, "same": 1}
