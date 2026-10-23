"""Retrieval ablation: does each stage of the hybrid pipeline earn its place? (§16.1)

Scores the top-K (config.rerank_top_k) passages each retrieval configuration would hand the
generator, over every ground-truth question:

  dense     FAISS bge-small only
  bm25      BM25 only
  fusion    dense + BM25 -> RRF, no rerank
  hybrid    dense + BM25 -> RRF -> cross-encoder rerank   (what ships)

Two measurements, both on identical questions:

  * Deterministic, judge-free: every relevant_context in the ground truth is the exact text of an
    indexed chunk, so a retrieved chunk is relevant iff its text is one of them. Reports hit@K
    (any relevant chunk in the top K), recall@K (share of the relevant chunks retrieved) and MRR.
  * Ragas context recall (the eval's own metric, LLM-judged) for fusion vs hybrid — the
    before/after the reranker is measured by. Skip with --no-judge.

It also lists the questions BM25 answers in its top K that dense misses: the concrete case for
hybrid retrieval.

    python scripts/retrieval_ablation.py              # writes data/retrieval_ablation.json
    python scripts/retrieval_ablation.py --no-judge   # deterministic metrics only, no API
    python scripts/retrieval_ablation.py --pool-sweep # rerank pool size vs quality and latency

The pool sweep asks how many fused candidates the cross-encoder needs to score. Its cost is linear
in the pool, and it dominates query latency. The cross-encoder scores each (question, passage)
pair independently, so scoring the top 50 once gives the exact top-K for every smaller pool;
latency is then timed separately per pool size. Judge-free and local: no API calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

from sentinel.config import DATA_DIR, settings  # noqa: E402
from sentinel.eval.run_eval import load_ground_truth  # noqa: E402
from sentinel.retrieve import (  # noqa: E402
    RERANK_POOL_SIZE,
    _load_index,
    _reranker,
    dense_search,
    rrf_fuse,
    sparse_search,
    warmup,
)

OUT_PATH = DATA_DIR / "retrieval_ablation.json"
SWEEP_PATH = DATA_DIR / "rerank_pool_sweep.json"
POOL_SIZES = (5, 10, 15, 20, 25, 30, 40, 50)
CONFIGS = ("dense", "bm25", "fusion", "hybrid")


def top_k_ids(question: str, k: int) -> dict[str, list[str]]:
    """The top-k chunk ids for every configuration, from ONE dense + ONE sparse search, so the
    configs differ only in the stages they apply."""
    dense = [c for c, _ in dense_search(question)]
    sparse = [c for c, _ in sparse_search(question)]
    fused = rrf_fuse([dense, sparse])
    pool = sorted(fused, key=lambda c: fused[c], reverse=True)[:RERANK_POOL_SIZE]
    idx = _load_index()
    scores = _reranker().predict([[question, idx.id2chunk[c].text] for c in pool])
    reranked = [c for c, _ in sorted(zip(pool, scores), key=lambda x: x[1], reverse=True)]
    return {"dense": dense[:k], "bm25": sparse[:k], "fusion": pool[:k], "hybrid": reranked[:k]}


def rank_metrics(retrieved: list[str], relevant: set[str]) -> dict[str, float]:
    first = next((i for i, c in enumerate(retrieved, 1) if c in relevant), None)
    return {
        "hit": 1.0 if first else 0.0,
        "recall": len(relevant & set(retrieved)) / len(relevant),
        "rr": 1.0 / first if first else 0.0,
    }


async def judge_recall(items: list[dict]) -> None:
    """Add Ragas context recall for fusion and hybrid to each item, in place."""
    import sentinel.eval  # noqa: F401  (installs the ragas <-> langchain-community shim)
    from ragas import SingleTurnSample

    from sentinel.eval.run_eval import _build_metrics, _coerce, _score_async

    _, _, context_recall = _build_metrics()
    idx = _load_index()
    for n, it in enumerate(items, 1):
        for cfg in ("fusion", "hybrid"):
            sample = SingleTurnSample(
                user_input=it["question"],
                retrieved_contexts=[idx.id2chunk[c].text for c in it["top_k"][cfg]],
                reference=it["reference_answer"],
            )
            it["context_recall"][cfg] = _coerce(await _score_async(context_recall, sample))
        print(f"  judged {n}/{len(items)}", end="\r", flush=True)
    print()


def pool_sweep(timing_repeats: int = 3) -> None:
    """Quality (hit/recall/MRR@K) and rerank latency for each pool size in POOL_SIZES."""
    import time

    k = settings.rerank_top_k
    warmup()
    idx = _load_index()
    text2id = {c.text: cid for cid, c in idx.id2chunk.items()}
    reranker = _reranker()
    biggest = max(POOL_SIZES)

    per_q = []
    for gt in load_ground_truth():
        relevant = {text2id[t] for t in gt.relevant_contexts}
        dense = [c for c, _ in dense_search(gt.question)]
        sparse = [c for c, _ in sparse_search(gt.question)]
        fused = rrf_fuse([dense, sparse])
        pool = sorted(fused, key=lambda c: fused[c], reverse=True)[:biggest]
        pairs = [[gt.question, idx.id2chunk[c].text] for c in pool]
        scores = list(reranker.predict(pairs))
        per_q.append((gt.question, relevant, pool, scores, pairs))

    rows = []
    for size in POOL_SIZES:
        metrics, times = [], []
        for _, relevant, pool, scores, pairs in per_q:
            top = [c for c, _ in sorted(zip(pool[:size], scores[:size]), key=lambda x: x[1], reverse=True)][:k]
            metrics.append(rank_metrics(top, relevant))
            for _ in range(timing_repeats):
                t0 = time.perf_counter()
                reranker.predict(pairs[:size])
                times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        row = {
            "pool_size": size,
            "hit": round(statistics.mean(m["hit"] for m in metrics), 3),
            "recall": round(statistics.mean(m["recall"] for m in metrics), 3),
            "mrr": round(statistics.mean(m["rr"] for m in metrics), 3),
            "rerank_p50_ms": round(times[len(times) // 2], 1),
            "rerank_p95_ms": round(times[int(len(times) * 0.95)], 1),
        }
        rows.append(row)
        print(f"  pool {size:>3}  hit@{k} {row['hit']:.3f}  recall@{k} {row['recall']:.3f}  "
              f"MRR {row['mrr']:.3f}  rerank p50 {row['rerank_p50_ms']:7.1f} ms  "
              f"p95 {row['rerank_p95_ms']:7.1f} ms", flush=True)
    SWEEP_PATH.write_text(json.dumps({
        "k": k, "n": len(per_q), "device": str(settings.torch_device), "timing_repeats": timing_repeats,
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {SWEEP_PATH}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-judged context recall")
    ap.add_argument("--pool-sweep", action="store_true", help="rerank pool size vs quality/latency")
    args = ap.parse_args()
    if args.pool_sweep:
        pool_sweep()
        return

    k = settings.rerank_top_k
    warmup()
    idx = _load_index()
    text2id = {c.text: cid for cid, c in idx.id2chunk.items()}

    items = []
    for gt in load_ground_truth():
        relevant = {text2id[t] for t in gt.relevant_contexts}
        top = top_k_ids(gt.question, k)
        items.append({
            "question": gt.question,
            "reference_answer": gt.reference_answer,
            "relevant": sorted(relevant),
            "top_k": top,
            "metrics": {cfg: rank_metrics(top[cfg], relevant) for cfg in CONFIGS},
            "context_recall": {},
        })

    if not args.no_judge:
        asyncio.run(judge_recall(items))

    summary = {
        cfg: {
            m: round(statistics.mean(it["metrics"][cfg][m] for it in items), 3)
            for m in ("hit", "recall", "rr")
        }
        for cfg in CONFIGS
    }
    for cfg in ("fusion", "hybrid"):
        if not args.no_judge:
            summary[cfg]["context_recall"] = round(
                statistics.mean(it["context_recall"][cfg] for it in items), 3
            )

    bm25_only = [
        {"question": it["question"], "relevant": it["relevant"], "bm25_top_k": it["top_k"]["bm25"]}
        for it in items
        if it["metrics"]["bm25"]["hit"] and not it["metrics"]["dense"]["hit"]
    ]
    rerank_rescues = [
        it["question"] for it in items
        if it["metrics"]["hybrid"]["hit"] and not it["metrics"]["fusion"]["hit"]
    ]
    rerank_losses = [
        it["question"] for it in items
        if it["metrics"]["fusion"]["hit"] and not it["metrics"]["hybrid"]["hit"]
    ]

    OUT_PATH.write_text(json.dumps({
        "k": k, "n": len(items), "summary": summary,
        "bm25_hit_dense_miss": bm25_only,
        "rerank_rescues": rerank_rescues, "rerank_losses": rerank_losses,
        "per_item": items,
    }, indent=2), encoding="utf-8")

    print(f"\n  RETRIEVAL ABLATION — top-{k}, {len(items)} ground-truth questions\n")
    print(f"  {'config':8} {'hit@k':>6} {'recall@k':>9} {'MRR':>6} {'ctx-recall (judge)':>19}")
    print("  " + "-" * 52)
    for cfg in CONFIGS:
        s = summary[cfg]
        cr = f"{s['context_recall']:.3f}" if "context_recall" in s else "—"
        print(f"  {cfg:8} {s['hit']:6.3f} {s['recall']:9.3f} {s['rr']:6.3f} {cr:>19}")
    print(f"\n  BM25 hit, dense missed: {len(bm25_only)}")
    for b in bm25_only:
        print(f"    - {b['question']}")
    print(f"  rerank rescued (fusion missed, hybrid hit): {len(rerank_rescues)}")
    print(f"  rerank lost    (fusion hit, hybrid missed): {len(rerank_losses)}")
    print(f"\n  wrote {OUT_PATH}\n")


if __name__ == "__main__":
    main()
