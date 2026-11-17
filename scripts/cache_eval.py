"""Semantic cache: at what similarity can one question safely reuse another's answer?

A false hit serves a confident, cited answer to a different question, which is worse than
no cache at all. So the threshold is set from measurement, local and free (embeddings only, no
API calls), on three sets:

  * data/cache_pairs.jsonl: for 32 ground-truth questions, a paraphrase that should hit, and a
    near-miss that must not: one discriminating thing changed (308 -> 307, the server's
    certificate -> the client's, DoT -> DoH). Several near-misses differ only in ordinary words,
    which the token check can't see; only the threshold stops those. Plus 3 casual lower-case
    phrasings, where a miss is acceptable.
  * every pair of distinct ground-truth questions (1,128 pairs), all of which must miss.

For each threshold it reports paraphrase hits and false hits, with and without the exact-token
check (sentinel.cache.technical_tokens). The chosen threshold clears the most similar negative
that the token check lets through by at least MARGIN, rounded up to the next 0.01.

    python scripts/cache_eval.py        # writes data/cache_eval.json
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from sentinel.cache import technical_tokens  # noqa: E402
from sentinel.config import DATA_DIR  # noqa: E402
from sentinel.eval.run_eval import load_ground_truth  # noqa: E402
from sentinel.retrieve import QUERY_INSTRUCTION, _embedder  # noqa: E402

PAIRS_PATH = DATA_DIR / "cache_pairs.jsonl"
OUT_PATH = DATA_DIR / "cache_eval.json"
THRESHOLDS = [round(0.80 + 0.01 * i, 2) for i in range(20)]
MARGIN = 0.01     # above the most similar negative the token check can't stop


def embed(texts: list[str]) -> np.ndarray:
    return _embedder().encode([QUERY_INSTRUCTION + t for t in texts], normalize_embeddings=True,
                              convert_to_numpy=True).astype("float32")


def main() -> None:
    pairs = [json.loads(line) for line in PAIRS_PATH.open(encoding="utf-8")]
    texts = sorted({p["cached"] for p in pairs} | {p["asked"] for p in pairs})
    vec = dict(zip(texts, embed(texts), strict=True))
    for p in pairs:
        p["similarity"] = round(float(vec[p["cached"]] @ vec[p["asked"]]), 4)
        p["same_tokens"] = technical_tokens(p["cached"]) == technical_tokens(p["asked"])

    gt = [g.question for g in load_ground_truth()]
    gv = dict(zip(gt, embed(gt), strict=True))
    cross = [{"a": a, "b": b, "similarity": round(float(gv[a] @ gv[b]), 4),
              "same_tokens": technical_tokens(a) == technical_tokens(b)}
             for a, b in itertools.combinations(gt, 2)]

    para = [p for p in pairs if p["expect"] == "hit" and p["why"] == "paraphrase"]
    casual = [p for p in pairs if p["why"] == "casual lower-case"]
    near = [p for p in pairs if p["expect"] == "miss"]

    def hits(rows, t, guard):
        return sum(r["similarity"] >= t and (r["same_tokens"] or not guard) for r in rows)

    sweep = []
    for t in THRESHOLDS:
        row = {"threshold": t}
        for guard in (True, False):
            k = "with_tokens" if guard else "embedding_only"
            row[k] = {"paraphrase_hits": hits(para, t, guard),
                      "casual_hits": hits(casual, t, guard),
                      "near_miss_false_hits": hits(near, t, guard),
                      "ground_truth_false_hits": hits(cross, t, guard)}
        sweep.append(row)

    def safe(row, k):
        return row[k]["near_miss_false_hits"] == 0 and row[k]["ground_truth_false_hits"] == 0

    lowest = {k: next((r["threshold"] for r in sweep if safe(r, k)), None)
              for k in ("with_tokens", "embedding_only")}
    negatives_passing_tokens = [r["similarity"] for r in near + cross if r["same_tokens"]]
    chosen = min(0.99, float(np.ceil((max(negatives_passing_tokens) + MARGIN) * 100) / 100))
    at = lambda t, k: next(r[k] for r in sweep if r["threshold"] == t)  # noqa: E731
    report = {
        "sets": {"paraphrases": len(para), "casual": len(casual), "near_misses": len(near),
                 "ground_truth_pairs": len(cross)},
        "near_misses_stopped_by_tokens": sum(not r["same_tokens"] for r in near),
        "paraphrases_with_same_tokens": sum(r["same_tokens"] for r in para),
        "highest_negative_similarity_with_same_tokens": max(negatives_passing_tokens),
        "lowest_paraphrase_similarity": min(r["similarity"] for r in para),
        "lowest_safe_threshold": lowest,
        "margin": MARGIN,
        "chosen_threshold": chosen,
        "at_chosen": {"with_tokens": at(chosen, "with_tokens")} if chosen else None,
        "embedding_only_at_chosen": at(chosen, "embedding_only") if chosen else None,
        "sweep": sweep,
        "pairs": pairs,
        "closest_ground_truth_pairs": sorted(cross, key=lambda r: -r["similarity"])[:10],
    }
    OUT_PATH.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("sweep", "pairs", "closest_ground_truth_pairs")}, indent=1))
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
