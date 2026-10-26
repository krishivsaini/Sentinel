"""Baseline vs fine-tuned reranker on the ground truth — identical pipeline, no LLM calls.

    python scripts/rerank_eval.py                                   # baseline vs runs/main, runs/nogold
    python scripts/rerank_eval.py --reranker main=runs/main/best    # any name=path (or model id)

For every ground-truth question the first-stage candidates are computed ONCE (dense + BM25 ->
RRF, the service's own functions) and every reranker scores the same candidates; the relevant
passages are the ground truth's, matched by exact text. Reported per reranker:

  * MRR@10, nDCG@10, Recall@5, Hit@5 at the service's pool size (50), against the baseline with a
    paired bootstrap 95% interval and per-question better / worse / same;
  * the same metrics at smaller pools (10-50) — scores are per pair, so a smaller pool is the top
    of the fused list re-scored, no extra model calls;
  * rerank latency per question at pool 50 and 20, timed separately on this machine.

Writes data/reranker/eval.json. The ceiling — the share of relevant passages anywhere in the pool —
bounds what any reranker can reach.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sentinel.config import DATA_DIR, settings  # noqa: E402
from sentinel.finetune.evaluate import (  # noqa: E402
    METRICS,
    first_relevant_rank,
    paired_bootstrap,
    rank_metrics,
    wins_losses,
)
from sentinel.finetune.train import BASE_REVISION  # noqa: E402

POOLS = (10, 20, 30, 40, 50)
DEFAULT_RERANKERS = ("main=runs/main/best", "nogold=runs/nogold/best")


def load_reranker(spec: str, device: str):
    from sentence_transformers import CrossEncoder

    if spec == "baseline":
        return CrossEncoder(settings.reranker_model, revision=BASE_REVISION, device=device)
    return CrossEncoder(spec, device=device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reranker", action="append", default=None, metavar="NAME=PATH",
                    help="a reranker to compare with the baseline (repeatable)")
    ap.add_argument("--bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--timing-repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "reranker" / "eval.json")
    args = ap.parse_args()

    from sentinel.retrieve import _load_index, dense_search, rrf_fuse, sparse_search

    specs = {"baseline": "baseline"}
    for item in args.reranker or DEFAULT_RERANKERS:
        name, _, path = item.partition("=")
        if Path(path).exists():
            specs[name] = path
        else:
            print(f"skipping {name}: {path} not found")

    idx = _load_index()
    text2id = {c.text: cid for cid, c in idx.id2chunk.items()}
    gt = [json.loads(line) for line in (DATA_DIR / "ground_truth.jsonl").open(encoding="utf-8")]
    items = []
    for g in gt:
        dense = [c for c, _ in dense_search(g["question"])]
        sparse = [c for c, _ in sparse_search(g["question"])]
        fused = rrf_fuse([dense, sparse])
        pool = sorted(fused, key=fused.get, reverse=True)[: max(POOLS)]
        items.append({"question": g["question"], "pool": pool,
                      "relevant": {text2id[t] for t in g["relevant_contexts"]}})
    ceiling = {p: statistics.mean(float(bool(it["relevant"] & set(it["pool"][:p]))) for it in items)
               for p in POOLS}

    results: dict = {"questions": len(items), "pool_sizes": list(POOLS), "device": args.device,
                     "ceiling_hit": {str(p): round(v, 4) for p, v in ceiling.items()},
                     "rerankers": {}}
    per_q: dict[str, dict] = {}
    for name, spec in specs.items():
        model = load_reranker(spec, args.device)
        model.predict([["warm up", "the model"]])
        scores = [model.predict([[it["question"], idx.id2chunk[c].text] for c in it["pool"]])
                  for it in items]
        by_pool = {}
        for p in POOLS:
            ranked = [[c for c, _ in sorted(zip(it["pool"][:p], s[:p], strict=True),
                                            key=lambda x: -x[1])] for it, s in zip(items, scores, strict=True)]
            m = [rank_metrics(r, it["relevant"]) for r, it in zip(ranked, items, strict=True)]
            by_pool[str(p)] = {k: round(statistics.mean(x[k] for x in m), 4) for k in METRICS}
            if p == max(POOLS):
                per_q[name] = {
                    "metrics": m,
                    "first_rank": [first_relevant_rank(r, it["relevant"]) for r, it in zip(ranked, items, strict=True)],
                }
        timing = {}
        for p in (50, 20):
            runs = []
            for _ in range(args.timing_repeats):
                for it in items:
                    t = time.perf_counter()
                    model.predict([[it["question"], idx.id2chunk[c].text] for c in it["pool"][:p]])
                    runs.append((time.perf_counter() - t) * 1000)
            timing[str(p)] = {"p50_ms": round(float(np.percentile(runs, 50)), 1),
                              "p95_ms": round(float(np.percentile(runs, 95)), 1)}
        results["rerankers"][name] = {"model": spec, "by_pool": by_pool, "latency": timing}
        print(f"{name:10} pool 50: " + "  ".join(f"{k} {by_pool['50'][k]:.3f}" for k in METRICS)
              + f"   rerank p50 {timing['50']['p50_ms']} ms (pool 20: {timing['20']['p50_ms']} ms)",
              flush=True)

    base = per_q["baseline"]
    for name in per_q:
        if name == "baseline":
            continue
        new = per_q[name]
        cmp = {k: paired_bootstrap(np.array([m[k] for m in base["metrics"]]),
                                   np.array([m[k] for m in new["metrics"]]), args.bootstrap, args.seed)
               for k in METRICS}
        cmp["questions"] = wins_losses(base["first_rank"], new["first_rank"])
        cmp["changed"] = [
            {"question": it["question"], "baseline_rank": a, "rank": b}
            for it, a, b in zip(items, base["first_rank"], new["first_rank"], strict=True) if a != b
        ]
        results["rerankers"][name]["vs_baseline"] = cmp
        print(f"  {name} vs baseline: " + "  ".join(
            f"{k} {cmp[k]['diff']:+.3f} [{cmp[k]['ci95'][0]:+.3f}, {cmp[k]['ci95'][1]:+.3f}]" for k in METRICS)
            + f"   questions {cmp['questions']}")

    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
