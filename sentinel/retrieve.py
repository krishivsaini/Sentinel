"""Retrieval layer (§9) — engineering-signal #1 — Phase 1, Days 3–4.

1. Dense retrieval: FAISS top-N (config.dense_top_n).
2. Sparse retrieval: BM25 top-N (config.sparse_top_n) over the same chunk set.
3. Fusion: Reciprocal Rank Fusion (RRF, config.rrf_k) into one candidate set.
4. Rerank: cross-encoder re-scores fused top-N -> top-K (config.rerank_top_k).

Returns RetrievedChunk objects carrying all four scores (dense/sparse/fused/rerank) so
eval and the dashboard can inspect why a chunk was selected (FR-R5). LangChain's
EnsembleRetriever may orchestrate the BM25+dense combination; RRF/scores are exposed
explicitly so the four scores survive fusion.

Run:  python -m sentinel.retrieve "some query"   (dev harness)
"""

from __future__ import annotations

from sentinel.schema import RetrievedChunk


def retrieve(question: str) -> list[RetrievedChunk]:
    # TODO(Phase 1, Days 3–4): dense + sparse -> RRF -> cross-encoder rerank.
    raise NotImplementedError("retrieve.py is a Phase 1 (Days 3–4) stub — not yet implemented.")
