"""Training data for the reranker fine-tune, in two steps around query generation.

    python scripts/reranker_data.py export     # -> data/reranker/windows.jsonl.gz
    # generate questions for every window (scripts/generate_queries.py, on a GPU: see
    # notebooks/reranker_querygen.ipynb) -> data/reranker/queries.jsonl.gz
    python scripts/reranker_data.py build      # -> train / val / chunks .jsonl.gz + stats.json

export cuts every retrievable chunk into the windows doc2query reads (sentinel.finetune.data),
measured with doc2query's own tokenizer. build cleans the generated questions, drops any that
leak the ground truth, keeps only those whose source chunk the first-stage retriever puts in the
reranker's candidate pool (a round-trip check that also matches training to inference), mines
hard negatives from that same pool, and splits by RFC. Everything is local: no API calls.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from sentinel.config import DATA_DIR, settings  # noqa: E402
from sentinel.finetune.data import (  # noqa: E402
    WINDOW_TOKENS,
    chunk_ordinal,
    clean_queries,
    is_validation_doc,
    leaked,
    make_windows,
    normalize_query,
    select_negatives,
)

OUT = DATA_DIR / "reranker"
QUERYGEN_MODEL = "doc2query/msmarco-t5-base-v1"
QUERYGEN_REVISION = "e673dca0dff1f19fda73ac62420eedf0219e692b"


def write_jsonl_gz(path: Path, rows) -> int:
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl_gz(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _retrievable_chunks():
    """Chunks the service can return: public, not quarantined — the reranker's whole world."""
    from sentinel.retrieve import _load_index

    idx = _load_index()
    return idx, [c for c in idx.chunks if c.scope == "public" and not c.quarantined]


# --------------------------------------------------------------------------- export


def export() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(QUERYGEN_MODEL, revision=QUERYGEN_REVISION)

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=True)["input_ids"])

    _, chunks = _retrievable_chunks()
    rows, dropped, per_chunk, lengths = [], Counter(), [], []
    for c in chunks:
        windows, why = make_windows(c.text, count)
        dropped.update(why)
        per_chunk.append(len(windows))
        for k, w in enumerate(windows):
            rows.append({"window_id": f"{c.chunk_id}/w{k}", "chunk_id": c.chunk_id, "text": w})
            lengths.append(count(w))
    OUT.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(OUT / "windows.jsonl.gz", rows)
    stats = {
        "chunks": len(chunks),
        "chunks_with_windows": sum(n > 0 for n in per_chunk),
        "windows": len(rows),
        "windows_per_chunk": dict(sorted(Counter(per_chunk).items())),
        "window_tokens": {"target": WINDOW_TOKENS, "median": statistics.median(lengths),
                          "max": max(lengths), "model_input_limit": 320,
                          "over_limit": sum(n > 320 for n in lengths)},
        "boilerplate_paragraphs_dropped": dict(dropped.most_common()),
        "tokenizer": f"{QUERYGEN_MODEL}@{QUERYGEN_REVISION[:7]}",
    }
    (OUT / "windows.stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


# --------------------------------------------------------------------------- build


def build(queries_path: Path, leak_threshold: float, near_dup: float, negatives: int,
          val_fraction: float, out: Path = OUT) -> None:
    from sentinel.retrieve import (
        RERANK_POOL_SIZE,
        _embedder,
        dense_search,
        rrf_fuse,
        sparse_search,
    )

    t0 = time.time()
    idx, chunks = _retrievable_chunks()
    allowed = {c.chunk_id for c in chunks}
    stats: dict = {"queries_file": queries_path.name}

    # 1. clean -----------------------------------------------------------------------------
    by_chunk: dict[str, list[str]] = defaultdict(list)
    for row in read_jsonl_gz(queries_path):
        if row["chunk_id"] in allowed:
            by_chunk[row["chunk_id"]].extend(row["queries"])
    by_chunk, stats["cleaning"] = clean_queries(by_chunk)
    pairs = [(q, cid) for cid, qs in by_chunk.items() for q in qs]
    print(f"[clean] {stats['cleaning']} -> {len(pairs)} queries", flush=True)

    # 2. leakage vs the ground truth (evaluation only, never trained on) --------------------
    gt = [json.loads(line) for line in (DATA_DIR / "ground_truth.jsonl").open(encoding="utf-8")]
    gt_questions = [g["question"] for g in gt]
    emb = _embedder()
    gq = emb.encode(gt_questions, normalize_embeddings=True, batch_size=64)
    sq = emb.encode([q for q, _ in pairs], normalize_embeddings=True, batch_size=256,
                    show_progress_bar=False)
    sims = sq @ gq.T
    best, best_gt = sims.max(axis=1), sims.argmax(axis=1)
    gt_norm = {normalize_query(q) for q in gt_questions}
    keep = [not leaked(q, float(s), gt_norm, leak_threshold) for (q, _), s in zip(pairs, best, strict=True)]
    removed = [(float(best[i]), pairs[i][0], gt_questions[best_gt[i]])
               for i in range(len(pairs)) if not keep[i]]
    stats["leakage"] = {
        "threshold": leak_threshold,
        "embedding": settings.embedding_model,
        "removed": len(removed),
        "max_similarity_histogram": {
            f">={t}": int((best >= t).sum()) for t in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
        },
        "removed_examples": [
            {"similarity": round(s, 3), "synthetic": q, "ground_truth": g}
            for s, q, g in sorted(removed, reverse=True)[:5] + sorted(removed)[:5]
        ],
        "closest_kept": [
            {"similarity": round(float(best[i]), 3), "synthetic": pairs[i][0],
             "ground_truth": gt_questions[best_gt[i]]}
            for i in np.argsort(-best) if keep[i]
        ][:5],
    }
    pairs = [p for p, k in zip(pairs, keep, strict=True) if k]
    print(f"[leakage] removed {len(removed)} at cosine >= {leak_threshold} -> {len(pairs)}", flush=True)

    # 3. round trip + hard negatives, from the reranker's own candidate pool -----------------
    text2id = {c.text: c.chunk_id for c in idx.chunks}
    gold = {text2id[t] for g in gt for t in g["relevant_contexts"]}
    vectors = idx.faiss.reconstruct_n(0, idx.faiss.ntotal)
    row_of = {cid: i for i, cid in enumerate(idx.chunk_ids)}

    def near_duplicate(a: str, b: str) -> bool:
        return float(vectors[row_of[a]] @ vectors[row_of[b]]) >= near_dup

    records, neg_stats, ranks, not_in_pool = [], Counter(), [], 0
    for n, (q, positive) in enumerate(pairs, 1):
        dense = [c for c, _ in dense_search(q)]
        sparse = [c for c, _ in sparse_search(q)]
        fused = rrf_fuse([dense, sparse])
        pool = sorted(fused, key=fused.get, reverse=True)[:RERANK_POOL_SIZE]
        if positive not in pool:
            not_in_pool += 1
            continue
        negs, why = select_negatives(positive, pool, near_duplicate, negatives)
        neg_stats.update(why)
        ranks.append(pool.index(positive) + 1)
        doc, _ = chunk_ordinal(positive)
        records.append({
            "qid": f"q{n:05d}", "query": q, "positive": positive, "negatives": negs,
            "doc_id": doc, "positive_is_gold": positive in gold,
            "positive_pool_rank": pool.index(positive) + 1,
        })
        if n % 1000 == 0:
            print(f"  mined {n}/{len(pairs)}", flush=True)
    stats["round_trip"] = {"kept": len(records), "positive_not_in_pool": not_in_pool,
                           "pool_size": RERANK_POOL_SIZE}
    stats["negatives"] = {
        "per_query_max": negatives,
        "per_query": {"mean": round(statistics.mean(len(r["negatives"]) for r in records), 2),
                      "min": min(len(r["negatives"]) for r in records)},
        "excluded": dict(neg_stats),
        "near_duplicate_cosine": near_dup,
    }
    stats["positive_pool_rank"] = {"median": statistics.median(ranks),
                                   "top1_share": round(sum(r == 1 for r in ranks) / len(ranks), 3)}

    # 4. split by RFC ---------------------------------------------------------------------------
    train = [r for r in records if not is_validation_doc(r["doc_id"], val_fraction)]
    val = [r for r in records if is_validation_doc(r["doc_id"], val_fraction)]
    stats["split"] = {
        "by": "RFC (doc_id)",
        "train": {"queries": len(train), "rfcs": len({r["doc_id"] for r in train}),
                  "gold_positives": sum(r["positive_is_gold"] for r in train)},
        "val": {"queries": len(val), "rfcs": len({r["doc_id"] for r in val}),
                "docs": sorted({r["doc_id"] for r in val})},
    }
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(out / "train.jsonl.gz", train)
    write_jsonl_gz(out / "val.jsonl.gz", val)
    write_jsonl_gz(out / "chunks.jsonl.gz",
                   ({"chunk_id": c.chunk_id, "text": c.text} for c in chunks))
    stats["chunks"] = len(chunks)
    stats["gold_chunks"] = len(gold)
    stats["seconds"] = round(time.time() - t0, 1)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "leakage"}, indent=2))
    print(json.dumps(stats["leakage"], indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("export", help="cut chunks into doc2query windows")
    b = sub.add_parser("build", help="clean, filter, mine negatives, split")
    b.add_argument("--queries", type=Path, default=OUT / "queries.jsonl.gz")
    b.add_argument("--leak-threshold", type=float, default=0.85,
                   help="drop a synthetic query whose cosine similarity to any ground-truth "
                        "question reaches this")
    b.add_argument("--near-duplicate", type=float, default=0.95,
                   help="a candidate this similar (cosine) to the positive is not a negative")
    b.add_argument("--negatives", type=int, default=30, help="hard negatives kept per query")
    b.add_argument("--val-fraction", type=float, default=0.12, help="share of RFCs held out")
    b.add_argument("--out", type=Path, default=OUT, help="where train/val/chunks/stats go")
    args = ap.parse_args()
    if args.cmd == "export":
        export()
    else:
        build(args.queries, args.leak_threshold, args.near_duplicate, args.negatives,
              args.val_fraction, args.out)


if __name__ == "__main__":
    main()
