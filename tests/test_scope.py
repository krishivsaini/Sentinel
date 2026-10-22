"""Retrieval access control (v2-R1, v2-R4) — the adversarial index.

The adversarial index holds every real RFC chunk ("public") plus the labelled poisoned corpus
("adversarial"). These tests prove the scope filter, not luck: the same queries that DO surface
poisoned chunks when the adversarial scope is granted surface none at all under the default
scope — through dense, sparse, and the full hybrid pipeline. They also prove that a
quarantined chunk is unreachable unless a caller explicitly opts in.

The index is built from indexes/ on first use (~10 s: only the adversarial passages are
embedded). No API calls.
"""

from __future__ import annotations

import json
import shutil

import pytest

from sentinel.config import ADVERSARIAL_DIR, ADVERSARIAL_INDEX_DIR, IndexPaths
from sentinel.retrieve import Access, dense_search, hybrid_retrieve, sparse_search
from sentinel.schema import Chunk

BOTH = Access(scopes=frozenset({"public", "adversarial"}))
PUBLIC = Access()  # config.default_scopes


@pytest.fixture(scope="module")
def adv_index():
    if not IndexPaths(ADVERSARIAL_INDEX_DIR).chunks.exists():
        from sentinel.ingest import ingest_adversarial

        ingest_adversarial()
    return ADVERSARIAL_INDEX_DIR


@pytest.fixture(scope="module")
def attacks() -> list[dict]:
    return [
        json.loads(line)
        for line in (ADVERSARIAL_DIR / "questions.jsonl").open(encoding="utf-8")
        if line.strip()
    ]


def _adversarial(chunk_ids) -> list[str]:
    return [c for c in chunk_ids if c.startswith("adv")]


def test_every_adversarial_chunk_is_scoped_adversarial(adv_index) -> None:
    chunks = [
        Chunk.model_validate_json(line)
        for line in IndexPaths(adv_index).chunks.open(encoding="utf-8")
    ]
    adv = [c for c in chunks if c.doc_id.startswith("adv")]
    assert adv, "the adversarial corpus is missing from its index"
    assert all(c.scope == "adversarial" for c in adv)
    assert all(c.scope == "public" for c in chunks if not c.doc_id.startswith("adv"))


def test_adversarial_scope_is_never_retrieved_by_default(adv_index, attacks) -> None:
    """v2-R4. Each attack question is written to retrieve its poisoned passage, and most do when
    the adversarial scope is granted. Under the default scope none may appear, at any stage."""
    exposed = 0
    for a in attacks:
        q = a["question"]
        exposed += bool(_adversarial(c.chunk_id for c in hybrid_retrieve(q, index_dir=adv_index, access=BOTH)))
        assert not _adversarial(c for c, _ in dense_search(q, index_dir=adv_index, access=PUBLIC))
        assert not _adversarial(c for c, _ in sparse_search(q, index_dir=adv_index, access=PUBLIC))
        assert not _adversarial(
            c.chunk_id for c in hybrid_retrieve(q, index_dir=adv_index, access=PUBLIC)
        )
    # The filter is what's doing the work: with access granted, the attacks do get through.
    assert exposed >= len(attacks) * 0.8


def test_exact_canary_query_cannot_reach_a_poisoned_chunk(adv_index, attacks) -> None:
    """BM25 matches a canary token exactly, and only one chunk contains it — the worst case for
    leakage. Granted, it's the top hit; under the default scope it can't appear at all."""
    for a in attacks:
        canary = a["markers"][0]
        assert sparse_search(canary, index_dir=adv_index, access=BOTH)[0][0].startswith(a["target_doc"])
        assert not _adversarial(c for c, _ in sparse_search(canary, index_dir=adv_index, access=PUBLIC))


def test_quarantined_chunk_is_excluded_unless_explicitly_included(adv_index, tmp_path) -> None:
    """v2-R1 enforcement: a chunk flagged by injection screening keeps its row in the index but is
    never retrieved by default — only the guardrail eval's no-defence baseline opts back in."""
    target = "adv07"
    root = tmp_path / "idx"
    shutil.copytree(adv_index, root)
    paths = IndexPaths(root)
    chunks = [Chunk.model_validate_json(line) for line in paths.chunks.open(encoding="utf-8")]
    with paths.chunks.open("w", encoding="utf-8") as fh:
        for c in chunks:
            if c.doc_id == target:
                c.quarantined = True
            fh.write(c.model_dump_json() + "\n")

    canary = "GANNET-4083"  # adv07's canary
    hits = lambda access: [c for c, _ in sparse_search(canary, index_dir=root, access=access)]  # noqa: E731
    assert not [c for c in hits(BOTH) if c.startswith(target)]
    assert hits(Access(scopes=BOTH.scopes, include_quarantined=True))[0].startswith(target)
