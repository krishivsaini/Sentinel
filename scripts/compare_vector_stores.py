"""FAISS vs pgvector: does moving dense search to Postgres change what retrieval returns?

FAISS's IndexFlatIP is exact. pgvector serves the same query from an HNSW index, which is
approximate, so this measures the difference over every ground-truth question, local and
judge-free (no API calls):

  * dense agreement: overlap of the two dense top-N lists (config.dense_top_n), how often the
    top 5 are identical, and the largest score difference on a shared chunk;
  * end to end: hit@K, recall@K and MRR of the full hybrid pipeline (dense + BM25 -> RRF ->
    cross-encoder) built on each store's dense list. BM25 and the reranker are shared, so the
    two differ only through dense search;
  * filtered: the same agreement on indexes_adversarial for the 50 attack questions, each aimed
    at an adversarial document, under two access modes: everything (the adversarial scope and
    quarantined chunks included), and the default, where the filters reject exactly the chunks
    those questions pull in, so pgvector's iterative scan has to read past them. Under the
    default it also counts any returned chunk the caller may not see (must be 0);
  * latency of the dense search alone (the query vector is computed once, outside the timing).

Needs both stores: the index directories, and their collections synced into Postgres
(python -m sentinel.vectorstore sync [--index-dir ...]).

    python scripts/compare_vector_stores.py      # writes data/vector_store_comparison.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
from retrieval_ablation import rank_metrics  # noqa: E402

from sentinel import vectorstore  # noqa: E402
from sentinel.config import (  # noqa: E402
    ADVERSARIAL_DIR,
    ADVERSARIAL_INDEX_DIR,
    DATA_DIR,
    INDEXES_DIR,
    settings,
)
from sentinel.eval.run_eval import load_ground_truth  # noqa: E402
from sentinel.retrieve import (  # noqa: E402
    DEFAULT_ACCESS,
    QUERY_INSTRUCTION,
    RERANK_POOL_SIZE,
    Access,
    _embedder,
    _load_index,
    _reranker,
    rrf_fuse,
    sparse_search,
)

OUT_PATH = DATA_DIR / "vector_store_comparison.json"


def faiss_search(idx, vec, top_n: int, access: Access) -> list[tuple[str, float]]:
    import faiss
    allowed = access.rows(idx)
    params = None if allowed is None else faiss.SearchParameters(
        sel=faiss.IDSelectorBatch(allowed))
    sims, rows = idx.faiss.search(vec[None, :], top_n, params=params)
    return [(idx.chunk_ids[r], float(s))
            for s, r in zip(sims[0], rows[0], strict=True) if r != -1]


def pg_search(root: Path, vec, top_n: int, access: Access) -> list[tuple[str, float]]:
    return vectorstore.dense_search(root, vec, top_n,
                                    access.scopes or frozenset(settings.default_scopes),
                                    access.include_quarantined)


def agreement(a: list[tuple[str, float]], b: list[tuple[str, float]]) -> dict:
    ids_a, ids_b = [c for c, _ in a], [c for c, _ in b]
    sb = dict(b)
    return {
        "overlap": len(set(ids_a) & set(ids_b)) / max(len(ids_a), 1),
        "same_top5": ids_a[:5] == ids_b[:5],
        "same_order": ids_a == ids_b,
        "max_score_diff": max((abs(s - sb[c]) for c, s in a if c in sb), default=0.0),
        "len_pg": len(ids_b),
    }


def p(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 3)


def compare(root: Path, questions: list[str], relevant: list[set[str]] | None,
            access: Access = DEFAULT_ACCESS) -> dict:
    idx = _load_index(root)
    top_n, k = settings.dense_top_n, settings.rerank_top_k
    vecs = _embedder().encode([QUERY_INSTRUCTION + q for q in questions],
                              normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    allowed = access.rows(idx)
    allowed_ids = set(idx.chunk_ids) if allowed is None else {idx.chunk_ids[i] for i in allowed}
    # Latency in its own loop: interleaved with the cross-encoder, both stores time the CPU
    # contention and cache eviction it causes. Round 1 warms caches; round 2 is reported.
    for _ in range(2):
        t_faiss, t_pg = [], []
        for vec in vecs:
            t0 = time.perf_counter()
            faiss_search(idx, vec, top_n, access)
            t1 = time.perf_counter()
            pg_search(root, vec, top_n, access)
            t_faiss.append((t1 - t0) * 1000)
            t_pg.append((time.perf_counter() - t1) * 1000)
    rows = []
    for qi, (question, vec) in enumerate(zip(questions, vecs, strict=True)):
        fa = faiss_search(idx, vec, top_n, access)
        pg = pg_search(root, vec, top_n, access)
        row = {"question": question, **agreement(fa, pg),
               "not_allowed": sum(c not in allowed_ids for c, _ in pg)}
        if relevant is not None:
            sparse = [c for c, _ in sparse_search(question, index_dir=root, access=access)]
            pools = {}
            for name, dense in (("faiss", fa), ("pgvector", pg)):
                fused = rrf_fuse([[c for c, _ in dense], sparse])
                ranked = sorted(fused, key=lambda c: fused[c], reverse=True)
                pools[name] = ranked[:RERANK_POOL_SIZE]
            union = sorted(set(pools["faiss"]) | set(pools["pgvector"]))
            scores = dict(zip(union, _reranker().predict(
                [[question, idx.id2chunk[c].text] for c in union]), strict=True))
            for name, pool in pools.items():
                top = sorted(pool, key=lambda c: scores[c], reverse=True)[:k]
                row[f"hybrid_{name}"] = rank_metrics(top, relevant[qi])
            row["same_hybrid_top_k"] = (
                sorted(pools["faiss"], key=lambda c: scores[c], reverse=True)[:k]
                == sorted(pools["pgvector"], key=lambda c: scores[c], reverse=True)[:k])
        rows.append(row)
        print(f"  {root.name}: {qi + 1}/{len(questions)}", end="\r", flush=True)
    print()

    summary = {
        "questions": len(rows),
        "dense_top_n": top_n,
        "ef_search": settings.pgvector_ef_search,
        "mean_overlap": round(statistics.mean(r["overlap"] for r in rows), 4),
        "min_overlap": round(min(r["overlap"] for r in rows), 4),
        "same_top5": sum(r["same_top5"] for r in rows),
        "same_order": sum(r["same_order"] for r in rows),
        "short_results": sum(r["len_pg"] < top_n for r in rows),
        "not_allowed_returned": sum(r["not_allowed"] for r in rows),
        "max_score_diff": float(f"{max(r['max_score_diff'] for r in rows):.2e}"),
        "dense_ms": {"faiss_p50": p(t_faiss, 50), "faiss_p95": p(t_faiss, 95),
                     "pgvector_p50": p(t_pg, 50), "pgvector_p95": p(t_pg, 95)},
    }
    if relevant is not None:
        summary["same_hybrid_top_k"] = sum(r["same_hybrid_top_k"] for r in rows)
        for name in ("faiss", "pgvector"):
            summary[f"hybrid_{name}"] = {
                m: round(statistics.mean(r[f"hybrid_{name}"][m] for r in rows), 3)
                for m in ("hit", "recall", "rr")}
    return {"summary": summary, "per_question": rows}


def main() -> None:
    if settings.vector_store != "faiss":
        sys.exit("run with SENTINEL_VECTOR_STORE=faiss (the default): FAISS is the reference")
    gt = load_ground_truth()
    idx = _load_index(INDEXES_DIR)
    text2id = {c.text: cid for cid, c in idx.id2chunk.items()}
    questions = [g.question for g in gt]
    relevant = [{text2id[t] for t in g.relevant_contexts} for g in gt]
    _reranker().predict([["warmup", "warmup"]])

    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "pgvector": vectorstore.status(),
              "corpus": compare(INDEXES_DIR, questions, relevant)}
    if ADVERSARIAL_INDEX_DIR.exists():
        attacks = [json.loads(line)["question"]
                   for line in (ADVERSARIAL_DIR / "questions.jsonl").open(encoding="utf-8")]
        everything = Access(scopes=frozenset({"public", "adversarial"}), include_quarantined=True)
        report["attacks_unfiltered"] = compare(ADVERSARIAL_INDEX_DIR, attacks, None, everything)
        report["attacks_filtered"] = compare(ADVERSARIAL_INDEX_DIR, attacks, None, Access())
    OUT_PATH.write_text(json.dumps(report, indent=1) + "\n")
    for name in ("corpus", "attacks_unfiltered", "attacks_filtered"):
        if name in report:
            print(name, json.dumps(report[name]["summary"], indent=1))
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
