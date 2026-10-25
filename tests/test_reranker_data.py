"""Reranker training data: the filtering rules, and the leakage guarantee on the real data.

The rules are tested hermetically. The leakage test reads the committed training set
(data/reranker/, built by scripts/reranker_data.py) and the ground truth: no ground-truth
question — or anything too close to one — may appear in training. It skips until the set exists.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

from sentinel.config import DATA_DIR
from sentinel.finetune.data import (
    boilerplate_kind,
    clean_queries,
    is_validation_doc,
    leaked,
    make_windows,
    normalize_query,
    select_negatives,
)

RERANKER_DATA = DATA_DIR / "reranker"


def words(text: str) -> int:   # a stand-in tokenizer: one token per word
    return len(text.split())


# --------------------------------------------------------------------------- boilerplate


@pytest.mark.parametrize("paragraph, kind", [
    ("   1.  Introduction  . . . . . . . . . . . . .   4\n   2.  Links . . . . . . . . .   6", "toc"),
    ("   [RFC2119]  Bradner, S., \"Key words\", BCP 14, RFC 2119,\n"
     "              DOI 10.17487/RFC2119, March 1997.", "reference"),
    ("   Julian Reschke\n   greenbytes GmbH\n   Email: julian@example.de", "contact"),
    ("   Copyright (c) 2022 IETF Trust and the persons identified as the\n   document authors.",
     "licence"),
    ("Internet Engineering Task Force (IETF)               R. Fielding, Ed.\n"
     "Request for Comments: 9110                                     Adobe", "header"),
    ("      +---+---+---+---+---+---+---+---+\n      | 0 | 1 | H |   Name Length (5+)  |\n"
     "      +---+---+---+-------------------+", "diagram"),
])
def test_boilerplate_is_recognised(paragraph, kind) -> None:
    assert boilerplate_kind(paragraph) == kind


@pytest.mark.parametrize("paragraph", [
    "   The 429 (Too Many Requests) status code indicates that the user has sent too many\n"
    "   requests in a given amount of time.",
    '      sub-delims  = "!" / "$" / "&" / "\'" / "(" / ")"\n                  / "*" / "+" / "," / ";" / "="',
    '     pct-encoded = "%" HEXDIG HEXDIG',
    "   See [RFC9110] for the definition of a representation.",      # cites, isn't an entry
])
def test_content_is_kept(paragraph) -> None:
    assert boilerplate_kind(paragraph) is None


# --------------------------------------------------------------------------- windows


def test_windows_cover_the_whole_chunk_within_the_limit() -> None:
    paras = [" ".join(f"w{p}x{i}" for i in range(40)) for p in range(5)]   # 5 x 40 words
    text = "\n\n".join(paras)
    windows, dropped = make_windows(text, words, max_tokens=100)
    assert all(words(w) <= 100 for w in windows)
    assert " ".join(windows).split() == text.split()      # nothing lost, order kept
    assert not dropped


def test_windows_drop_boilerplate_and_stubs() -> None:
    text = ("   1.  Intro . . . . . . . . . . . 4\n\n"
            + " ".join(["The server sends a reset when the segment is unacceptable."] * 5)
            + "\n\n4.2.  Heading")
    windows, dropped = make_windows(text, words, max_tokens=300)
    assert dropped == {"toc": 1}
    assert len(windows) == 1 and "Intro" not in windows[0]


def test_an_overlong_paragraph_is_split_at_sentences_then_words() -> None:
    long_sentence = " ".join(["word"] * 250)
    text = "First sentence here. " * 20 + long_sentence
    windows, _ = make_windows(text, words, max_tokens=60)
    assert windows and all(words(w) <= 60 for w in windows)
    assert sum(words(w) for w in windows) == words(text)   # the 10-word tail isn't dropped


def test_a_heading_joins_the_paragraph_after_it() -> None:
    heading = "4.2.  Stateless Reset"
    body = " ".join(["The endpoint sends a stateless reset token."] * 14)   # 98 words
    windows, _ = make_windows(f"{heading}\n\n{body}", words, max_tokens=98)
    assert len(windows) == 1 and windows[0].startswith("4.2. Stateless Reset")


# --------------------------------------------------------------------------- queries


def test_queries_are_cleaned_deduplicated_and_ambiguous_ones_dropped() -> None:
    kept, stats = clean_queries({
        "rfc1#0001": ["what is a token", "What is a token?", "ok", "what does http 429 mean"],
        "rfc2#0002": ["what does HTTP 429 mean", "how is a cookie scoped"],
    })
    assert kept == {"rfc1#0001": ["what is a token"], "rfc2#0002": ["how is a cookie scoped"]}
    assert stats == {"raw": 6, "bad_length": 1, "duplicate": 1, "ambiguous": 2}


def test_leakage_is_caught_by_text_or_similarity() -> None:
    gt = {normalize_query("What does the HTTP 429 status code mean?")}
    assert leaked("what does the http 429 status code mean", 0.10, gt, 0.85)   # same text
    assert leaked("meaning of status 429", 0.91, gt, 0.85)                    # too similar
    assert not leaked("how are cookies scoped", 0.40, gt, 0.85)


# --------------------------------------------------------------------------- negatives, split


def test_negatives_exclude_the_positive_its_neighbours_and_near_duplicates() -> None:
    pool = ["rfc1#0004", "rfc1#0005", "rfc1#0006", "rfc1#0009", "rfc2#0005", "rfc3#0001"]
    negs, stats = select_negatives("rfc1#0005", pool, lambda a, b: b == "rfc3#0001", limit=10)
    assert negs == ["rfc1#0009", "rfc2#0005"]
    assert stats == {"neighbour": 2, "near_duplicate": 1}


def test_negatives_respect_the_limit_in_pool_order() -> None:
    pool = [f"rfc9#{i:04d}" for i in range(20)]
    negs, _ = select_negatives("rfc1#0000", pool, lambda a, b: False, limit=3)
    assert negs == ["rfc9#0000", "rfc9#0001", "rfc9#0002"]


def test_the_split_is_by_document_and_stable() -> None:
    docs = [f"rfc{n}" for n in range(1000)]
    val = [d for d in docs if is_validation_doc(d, 0.12)]
    assert 80 < len(val) < 160                                   # ~12%
    assert val == [d for d in docs if is_validation_doc(d, 0.12)]   # deterministic


# --------------------------------------------------------------------------- the real data


def _load(name: str) -> list[dict]:
    with gzip.open(RERANKER_DATA / name, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@pytest.mark.skipif(not (RERANKER_DATA / "train.jsonl.gz").exists(),
                    reason="training data not built yet (scripts/reranker_data.py build)")
def test_no_ground_truth_question_leaks_into_training() -> None:
    """The ground truth is evaluation-only. Re-checks the committed data from scratch: no
    training or validation query matches a ground-truth question as text, or comes within the
    leakage threshold the data was built with."""
    from sentinel.retrieve import _embedder

    stats = json.loads((RERANKER_DATA / "stats.json").read_text())
    threshold = stats["leakage"]["threshold"]
    gt = [json.loads(line)["question"] for line in (DATA_DIR / "ground_truth.jsonl").open()]
    rows = _load("train.jsonl.gz") + _load("val.jsonl.gz")
    gt_norm = {normalize_query(q) for q in gt}
    assert not [r["query"] for r in rows if normalize_query(r["query"]) in gt_norm]
    emb = _embedder()
    g = emb.encode(gt, normalize_embeddings=True)
    s = emb.encode([r["query"] for r in rows], normalize_embeddings=True, batch_size=256)
    worst = float(np.max(s @ g.T))
    assert worst < threshold, f"a training query is {worst:.3f} similar to a ground-truth question"


@pytest.mark.skipif(not (RERANKER_DATA / "train.jsonl.gz").exists(),
                    reason="training data not built yet")
def test_training_and_validation_share_no_rfc() -> None:
    train_docs = {r["doc_id"] for r in _load("train.jsonl.gz")}
    val_docs = {r["doc_id"] for r in _load("val.jsonl.gz")}
    assert train_docs and val_docs and not train_docs & val_docs
