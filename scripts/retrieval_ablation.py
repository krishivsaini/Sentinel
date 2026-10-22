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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-judged context recall")
    args = ap.parse_args()

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
