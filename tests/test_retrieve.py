"""Retrieval tests (§4) — Phase 1, Days 3-4.

Coverage:
  - dense and BM25 indexes share identical, aligned chunk IDs (guards the silent
    fusion-corruption bug — the #1 risk in the register)
  - hybrid_retrieve returns top-K carrying all four scores (FR-R5)
  - FR-R7: a natural-language query whose gold chunk BM25 ranks #1 but pure dense misses,
    which the hybrid path then recovers — the concrete justification for hybrid retrieval
  - the LangChain EnsembleRetriever orchestration is wired and returns candidates

Requires built indexes (`python -m sentinel.ingest`). Models load once per process (lru_cache).

FR-R6 (rerank measurably improves context recall vs fusion-only) is asserted in Phase 3 once
the ground-truth set exists; it needs labeled relevance, not available here.
"""

from __future__ import annotations

import pytest

from sentinel.config import settings
from sentinel.retrieve import (
    _load_index,
    build_ensemble_retriever,
    dense_search,
    hybrid_retrieve,
    sparse_search,
)

# The locked FR-R7 example (see docs / README): a real question containing an exact protocol
# identifier. BM25 pins the defining chunk at rank 1; pure dense scatters it far down.
FR_R7_QUERY = "What does the invalid_grant error mean?"
FR_R7_DOC = "rfc6749"          # OAuth 2.0 Authorization Framework
FR_R7_LITERAL = "invalid_grant"


def _ids(ranked: list[tuple[str, float]]) -> list[str]:
    return [cid for cid, _ in ranked]


def test_bm25_and_dense_share_chunk_ids() -> None:
    """FR-I2: FAISS rows, BM25 docs, and the canonical store are the same aligned IDs."""
    idx = _load_index()
    assert idx.faiss.ntotal == len(idx.chunks) == len(idx.chunk_ids) > 0
    assert len(set(idx.chunk_ids)) == len(idx.chunk_ids), "duplicate chunk_id"
    # _load_index() already raises if BM25 order != canonical order; assert the invariant too.
    assert [c.chunk_id for c in idx.chunks] == idx.chunk_ids


def test_hybrid_returns_topk_with_all_four_scores() -> None:
    """FR-R4/FR-R5: top-K reranked results, each inspectable via fused + rerank scores."""
    results = hybrid_retrieve(FR_R7_QUERY)
    idx = _load_index()

    assert len(results) == settings.rerank_top_k
    # fused + rerank are set for every survivor; dense/sparse may be None (came from only one list)
    for rc in results:
        assert rc.chunk_id in idx.id2chunk
        assert rc.text
        assert rc.fused_score is not None
        assert rc.rerank_score is not None
    # final ordering is by the cross-encoder
    rerank_scores = [rc.rerank_score for rc in results]
    assert rerank_scores == sorted(rerank_scores, reverse=True)


def test_hybrid_recovers_exact_match_query_missed_by_dense() -> None:
    """FR-R7: the hybrid-justifying case. Gold chunk is BM25 top-3 but absent from dense top-5;
    the full hybrid pipeline recovers it into the final top-K."""
    idx = _load_index()
    gold = {c.chunk_id for c in idx.chunks
            if c.doc_id == FR_R7_DOC and FR_R7_LITERAL in c.text}
    assert gold, "expected a gold chunk containing the exact literal"

    sparse_top3 = _ids(sparse_search(FR_R7_QUERY))[:3]
    dense_top5 = _ids(dense_search(FR_R7_QUERY))[:5]
    hybrid_ids = {rc.chunk_id for rc in hybrid_retrieve(FR_R7_QUERY)}

    assert gold & set(sparse_top3), "BM25 should rank the exact-match gold chunk near the top"
    assert not (gold & set(dense_top5)), "pure dense should miss it from its top-5"
    assert gold & hybrid_ids, "hybrid RRF + rerank should recover the gold chunk into top-K"


def test_ensemble_retriever_orchestration() -> None:
    """The deliberate LangChain use: EnsembleRetriever fuses the dense + sparse BaseRetrievers
    and returns chunk-tagged Documents."""
    docs = build_ensemble_retriever().invoke(FR_R7_QUERY)
    assert docs, "ensemble should return candidates"
    idx = _load_index()
    assert all(d.metadata.get("chunk_id") in idx.id2chunk for d in docs)


@pytest.mark.skip(reason="FR-R6 (rerank improves context recall) needs ground truth — Phase 3.")
def test_rerank_improves_context_recall_vs_fusion_only() -> None: ...
