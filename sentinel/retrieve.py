"""Retrieval layer (§9) — engineering-signal #1 — Phase 1, Days 3-4.

Pipeline:
  1. Dense retrieval  — FAISS top-N by bge cosine similarity (config.dense_top_n).
  2. Sparse retrieval — BM25 top-N over the same chunk set (config.sparse_top_n).
  3. Fusion          — explicit Reciprocal Rank Fusion (RRF, config.rrf_k) into one set.
  4. Rerank          — cross-encoder re-scores the fused pool down to top-K (config.rerank_top_k).

Every returned RetrievedChunk carries all four scores (dense/sparse/fused/rerank) so eval and
the dashboard can inspect *why* a chunk was selected (FR-R5).

Deliberate LangChain use: the dense and sparse stages are LangChain `BaseRetriever`
subclasses, and `build_ensemble_retriever()` composes them with a real `EnsembleRetriever`
(the framework's orchestration primitive, exercised in tests). The scored path here computes
RRF explicitly because EnsembleRetriever fuses internally and does NOT surface the per-retriever
ranks/scores that FR-R5 requires — so we keep the LangChain orchestration *and* the four scores.

Run:  python -m sentinel.retrieve "your query here"
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from sentinel.config import BM25_PATH, CHUNKS_PATH, FAISS_DIR, settings
from sentinel.ingest import bm25_tokenize
from sentinel.schema import Chunk, RetrievedChunk

# bge-*-en-v1.5 wants this instruction on the QUERY side only (passages were embedded plain).
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Fused candidates handed to the cross-encoder before it trims to top-K (the "N" in 50 -> 5).
RERANK_POOL_SIZE = max(settings.dense_top_n, settings.sparse_top_n)


# --------------------------------------------------------------------------- loaded state


@dataclass
class _Index:
    chunks: list[Chunk]
    id2chunk: dict[str, Chunk]
    chunk_ids: list[str]      # canonical order; FAISS row i and BM25 doc i both map here
    faiss: object             # faiss.IndexFlatIP
    bm25: object              # rank_bm25.BM25Okapi


@lru_cache(maxsize=1)
def _load_index() -> _Index:
    import faiss

    chunks = [Chunk.model_validate_json(line) for line in CHUNKS_PATH.open(encoding="utf-8")]
    with BM25_PATH.open("rb") as fh:
        bm = pickle.load(fh)
    idx = _Index(
        chunks=chunks,
        id2chunk={c.chunk_id: c for c in chunks},
        chunk_ids=[c.chunk_id for c in chunks],
        faiss=faiss.read_index(str(FAISS_DIR / "index.faiss")),
        bm25=bm["bm25"],
    )
    # FR-I2 guard: the persisted BM25 order must match the canonical chunk order, or fusion
    # (which aligns the two by row) would silently corrupt.
    if bm["chunk_ids"] != idx.chunk_ids:
        raise RuntimeError("BM25 chunk order != canonical chunk order — re-run ingestion.")
    return idx


@lru_cache(maxsize=1)
def _embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(settings.embedding_model)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model)


def warmup() -> None:
    """Eagerly load the index + both local models AND run one dummy end-to-end retrieval so the
    FIRST real query isn't penalized by one-time model loading or first-inference kernel warmup
    — otherwise that cost lands inside the timed retrieve/rerank stages and makes per-stage
    latency dishonest (NFR-3). Called from the service lifespan."""
    _load_index()
    hybrid_retrieve_timed("warmup", top_k=1)


# --------------------------------------------------------------------------- base searches


def dense_search(query: str, top_n: int | None = None) -> list[tuple[str, float]]:
    """FAISS top-N by cosine similarity. Returns (chunk_id, dense_score) best-first."""
    top_n = top_n or settings.dense_top_n
    idx = _load_index()
    vec = _embedder().encode(
        [QUERY_INSTRUCTION + query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    sims, rows = idx.faiss.search(vec, top_n)
    return [(idx.chunk_ids[r], float(s)) for s, r in zip(sims[0], rows[0]) if r != -1]


def sparse_search(query: str, top_n: int | None = None) -> list[tuple[str, float]]:
    """BM25 top-N over the same chunks. Returns (chunk_id, sparse_score) best-first."""
    top_n = top_n or settings.sparse_top_n
    idx = _load_index()
    scores = idx.bm25.get_scores(bm25_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [(idx.chunk_ids[i], float(scores[i])) for i in ranked]


# --------------------------------------------------------------------------- LangChain retrievers


class _ScoredRetriever(BaseRetriever):
    """Base: runs one search fn and packs (chunk_id, score) into Document metadata."""

    top_n: int
    score_key: str

    def _search(self, query: str) -> list[tuple[str, float]]:  # overridden
        raise NotImplementedError

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        idx = _load_index()
        return [
            Document(
                page_content=idx.id2chunk[cid].text,
                metadata={"chunk_id": cid, self.score_key: score},
            )
            for cid, score in self._search(query)
        ]


class DenseRetriever(_ScoredRetriever):
    top_n: int = settings.dense_top_n
    score_key: str = "dense_score"

    def _search(self, query: str) -> list[tuple[str, float]]:
        return dense_search(query, self.top_n)


class SparseRetriever(_ScoredRetriever):
    top_n: int = settings.sparse_top_n
    score_key: str = "sparse_score"

    def _search(self, query: str) -> list[tuple[str, float]]:
        return sparse_search(query, self.top_n)


def build_ensemble_retriever(weights: tuple[float, float] = (0.5, 0.5)):
    """The deliberate LangChain orchestration: EnsembleRetriever over the dense + sparse
    BaseRetrievers. Used in tests to show the framework fuses the same candidate set; the
    scored production path uses explicit RRF below to preserve per-stage scores (FR-R5)."""
    from langchain_classic.retrievers import EnsembleRetriever

    return EnsembleRetriever(
        retrievers=[DenseRetriever(), SparseRetriever()], weights=list(weights)
    )


# --------------------------------------------------------------------------- fusion + rerank


def rrf_fuse(ranked_lists: list[list[str]], k: int | None = None) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(id) = sum_l 1/(k + rank_l(id)), ranks 1-based.
    Parameter-light and robust to the two retrievers' incomparable score scales (§9.3)."""
    k = k if k is not None else settings.rrf_k
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, cid in enumerate(ranked, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


def hybrid_retrieve_timed(
    question: str, top_k: int | None = None
) -> tuple[list[RetrievedChunk], float, float]:
    """Full pipeline, returning (results, retrieve_ms, rerank_ms) so the service can report
    per-stage latency in isolation (never blended, NFR-3). 'retrieve' covers dense + sparse +
    RRF; 'rerank' covers the cross-encoder pass."""
    top_k = top_k or settings.rerank_top_k
    idx = _load_index()

    t0 = time.perf_counter()
    dense = dense_search(question)
    sparse = sparse_search(question)
    dense_map = dict(dense)
    sparse_map = dict(sparse)
    fused_map = rrf_fuse([[c for c, _ in dense], [c for c, _ in sparse]])
    pool = sorted(fused_map, key=lambda c: fused_map[c], reverse=True)[:RERANK_POOL_SIZE]
    retrieve_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    rerank_scores = _reranker().predict([[question, idx.id2chunk[c].text] for c in pool])
    reranked = sorted(zip(pool, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]
    rerank_ms = (time.perf_counter() - t1) * 1000.0

    results = [
        RetrievedChunk(
            chunk_id=cid,
            text=idx.id2chunk[cid].text,
            dense_score=dense_map.get(cid),
            sparse_score=sparse_map.get(cid),
            fused_score=fused_map.get(cid),
            rerank_score=float(score),
        )
        for cid, score in reranked
    ]
    return results, retrieve_ms, rerank_ms


def hybrid_retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Full pipeline: dense + sparse -> RRF -> cross-encoder rerank -> top-K, all four scores."""
    results, _, _ = hybrid_retrieve_timed(question, top_k)
    return results


def retrieve(question: str) -> list[RetrievedChunk]:
    """Public entry point used by generation/eval/service."""
    return hybrid_retrieve(question)


# --------------------------------------------------------------------------- dev harness


def _print_run(question: str) -> None:
    idx = _load_index()
    print(f"\nquery: {question!r}\n")
    print("  dense top-5:")
    for cid, s in dense_search(question)[:5]:
        print(f"    {s:6.3f}  {cid:<14} {idx.id2chunk[cid].text[:60].strip()!r}")
    print("  sparse top-5:")
    for cid, s in sparse_search(question)[:5]:
        print(f"    {s:6.3f}  {cid:<14} {idx.id2chunk[cid].text[:60].strip()!r}")
    print(f"  hybrid (RRF + rerank) top-{settings.rerank_top_k}:")
    for rc in hybrid_retrieve(question):
        d = f"{rc.dense_score:.3f}" if rc.dense_score is not None else "  -  "
        sp = f"{rc.sparse_score:.2f}" if rc.sparse_score is not None else "  -  "
        print(f"    rerank={rc.rerank_score:6.3f} fused={rc.fused_score:.4f} "
              f"dense={d} sparse={sp}  {rc.chunk_id:<14} {rc.text[:50].strip()!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid retrieval dev harness.")
    ap.add_argument("query", help="the question to retrieve for")
    ap.add_argument("--json", action="store_true", help="emit the top-K RetrievedChunks as JSON")
    args = ap.parse_args()
    if args.json:
        print(json.dumps([rc.model_dump() for rc in hybrid_retrieve(args.query)], indent=2))
    else:
        _print_run(args.query)


if __name__ == "__main__":
    main()
