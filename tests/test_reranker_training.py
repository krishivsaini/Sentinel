"""The hand-written training loop: that it learns, and that its output is a drop-in reranker.

Uses the real base cross-encoder (cached locally and in CI); every test runs on CPU in seconds.
"""

from __future__ import annotations

import math

import pytest
import torch

from sentinel.finetune.train import (
    BASE_MODEL,
    BASE_REVISION,
    Collate,
    GroupDataset,
    exclude_gold,
    grouped_softmax_loss,
    linear_warmup_decay,
    param_groups,
)

CHUNKS = {
    "rfc1#0001": "The 429 (Too Many Requests) status code indicates that the user has sent too "
                 "many requests in a given amount of time.",
    "rfc1#0002": "The Secure attribute limits the scope of a cookie to secure channels.",
    "rfc1#0003": "QUIC endpoints use connection IDs to route packets after migration.",
    "rfc1#0004": "A stateless reset is sent when a packet cannot be associated with a connection.",
    "rfc1#0005": "Base64url encoding uses the URL- and filename-safe alphabet without padding.",
}
EXAMPLES = [
    # positives deliberately chosen against what the base model would rank first: the loop has
    # to move the scores, not just keep them
    {"query": "what is a stateless reset", "positive": "rfc1#0002",
     "negatives": ["rfc1#0004", "rfc1#0001", "rfc1#0003"]},
    {"query": "what does status 429 mean", "positive": "rfc1#0005",
     "negatives": ["rfc1#0001", "rfc1#0002", "rfc1#0004"]},
]


@pytest.fixture(scope="module")
def base():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    return tok, lambda: AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION)


def _batch(tok):
    ds = GroupDataset(EXAMPLES, CHUNKS, negatives=3, seed=0)
    return Collate(tok, max_length=64)([ds[0], ds[1]])


def test_the_loop_overfits_a_tiny_fixed_batch(base) -> None:
    """Proves the loop learns: on one fixed batch, a few AdamW steps drive the grouped loss down
    even though the 'positives' are the wrong passages for the base model."""
    tok, make = base
    torch.manual_seed(0)
    model = make().train()
    batch = _batch(tok)
    group = batch.pop("group_size")
    opt = torch.optim.AdamW(param_groups(model, 0.01), lr=1e-4)
    losses = []
    for _ in range(25):
        loss = grouped_softmax_loss(model(**batch).logits.squeeze(-1), group)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[0] > 1.0                      # the base model gets these wrong
    assert losses[-1] < 0.1 * losses[0], losses


def test_a_saved_checkpoint_is_a_drop_in_cross_encoder(base, tmp_path) -> None:
    """What the loop saves loads with sentence-transformers' CrossEncoder — the class the
    service uses — and scores exactly as the raw model did."""
    from sentence_transformers import CrossEncoder

    tok, make = base
    model = make().eval()
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    pairs = [[e["query"], CHUNKS[c]] for e in EXAMPLES for c in [e["positive"], *e["negatives"]]]
    ce = CrossEncoder(str(tmp_path), device="cpu")
    with torch.no_grad():
        enc = tok([q for q, _ in pairs], [p for _, p in pairs], truncation=True, max_length=512,
                  padding=True, return_tensors="pt")
        raw = model(**enc).logits.squeeze(-1)
    assert torch.allclose(torch.tensor(ce.predict(pairs)), raw, atol=1e-4)


def test_grouped_loss_is_cross_entropy_with_the_positive_first() -> None:
    scores = torch.tensor([3.0, 1.0, 0.0, 2.0, 2.0, 2.0])     # two groups of 3
    loss = grouped_softmax_loss(scores, 3)
    by_hand = (-(3 - math.log(math.exp(3) + math.exp(1) + 1))   # group 1, positive 3.0
               - (2 - math.log(3 * math.exp(2)))) / 2           # group 2, a three-way tie
    assert loss.item() == pytest.approx(by_hand, rel=1e-6)


def test_negatives_are_resampled_each_epoch_but_reproducibly() -> None:
    ex = [{"query": "q", "positive": "rfc1#0001", "negatives": list(CHUNKS)[1:]}]
    ds = GroupDataset(ex, CHUNKS, negatives=2, seed=7)
    epochs = []
    for e in range(4):
        ds.set_epoch(e)
        epochs.append(tuple(ds[0]["passages"]))
    assert all(p[0] == CHUNKS["rfc1#0001"] for p in epochs)     # positive always first
    assert len(set(epochs)) > 1                                # negatives vary across epochs
    ds.set_epoch(2)
    assert tuple(ds[0]["passages"]) == epochs[2]               # ...and repeat exactly


def test_learning_rate_warms_up_then_decays_to_zero() -> None:
    f = linear_warmup_decay(warmup_steps=10, total_steps=100)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(55) == pytest.approx(0.5) and f(100) == 0.0


def test_no_weight_decay_on_biases_or_layer_norms(base) -> None:
    _, make = base
    decay, no_decay = param_groups(make(), 0.01)
    assert all(p.ndim > 1 for p in decay["params"])
    assert all(p.ndim == 1 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0


def test_gold_exclusion_removes_questions_and_negatives() -> None:
    kept, stats = exclude_gold(EXAMPLES, gold={"rfc1#0005", "rfc1#0004"})
    assert [e["query"] for e in kept] == ["what is a stateless reset"]
    assert "rfc1#0004" not in kept[0]["negatives"]
    assert stats == {"questions_dropped": 1, "negatives_dropped": 1}


# --------------------------------------------------------------------------- the switch


def test_the_reranker_switch_selects_the_model(monkeypatch, tmp_path) -> None:
    from sentinel import retrieve
    from sentinel.config import settings

    assert settings.reranker == "baseline"                     # the default is unchanged
    assert retrieve.reranker_name() == settings.reranker_model
    monkeypatch.setattr(settings, "reranker", "finetuned")
    monkeypatch.setattr(retrieve, "REPO_ROOT", tmp_path)
    (tmp_path / "runs" / "main" / "best").mkdir(parents=True)
    monkeypatch.setattr(settings, "finetuned_reranker", "runs/main/best")
    assert retrieve.reranker_name() == str(tmp_path / "runs" / "main" / "best")
    monkeypatch.setattr(settings, "finetuned_reranker", "someone/rfc-reranker")   # a hub id
    assert retrieve.reranker_name() == "someone/rfc-reranker"


def test_an_unknown_reranker_choice_is_rejected() -> None:
    from pydantic import ValidationError

    from sentinel.config import Settings

    with pytest.raises(ValidationError):
        Settings(reranker="best-one")
