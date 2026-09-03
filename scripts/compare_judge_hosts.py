"""Compare the SAME judge model served by two different providers, on IDENTICAL inputs.

Why this exists: moving the judge from Groq to NVIDIA keeps the same weights (openai/gpt-oss-120b),
so the calibration (r = +0.90 vs hand labels) *should* carry over. But hosts can differ in
quantisation, sampling, and default params, any of which can shift judgments — and comparing two
eval runs cannot answer it, because generation is not bitwise deterministic (only ~half of
answers repeat exactly at temperature 0), which confounds host effect with answer drift.

So this pins the inputs: it replays stored (question, answer, retrieved contexts) triples from a
completed run and scores each through both providers, reporting per-item deltas. A small mean
delta means the calibration transfers and no re-labelling is needed.

    python scripts/compare_judge_hosts.py --model openai/gpt-oss-120b -n 5
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
warnings.filterwarnings("ignore")

import sentinel.eval  # noqa: E402  (installs the ragas <-> langchain-community shim)
from ragas import SingleTurnSample  # noqa: E402
from ragas.embeddings.base import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms.base import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from sentinel.config import settings  # noqa: E402
from sentinel.eval import store  # noqa: E402
from sentinel.llm import chat_model  # noqa: E402


def load_samples(git_sha: str | None, n: int) -> list[SingleTurnSample]:
    """Pull stored items straight from the checkpoint table so both hosts judge byte-identical
    inputs (same question, same generated answer, same retrieved contexts)."""
    import sqlite3

    conn = sqlite3.connect(store.EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    if git_sha is None:
        row = conn.execute(
            "SELECT git_sha FROM eval_items GROUP BY git_sha ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
        if not row:
            sys.exit("eval_items is empty — run an eval first.")
        git_sha = row["git_sha"]
    rows = conn.execute(
        "SELECT * FROM eval_items WHERE git_sha = ? ORDER BY question", (git_sha,)
    ).fetchall()
    conn.close()
    step = max(1, len(rows) // n)
    picked = rows[::step][:n]
    print(f"replaying {len(picked)} stored items from {git_sha}")
    import json as _json

    return [
        SingleTurnSample(
            user_input=r["question"],
            response=r["generated_answer"],
            retrieved_contexts=[c["text"] for c in _json.loads(r["retrieved_json"])],
            reference=r["generated_answer"],
        )
        for r in picked
    ]


def build_metrics(provider: str, model: str):
    from langchain_huggingface import HuggingFaceEmbeddings

    rc = RunConfig(timeout=180, max_retries=1, max_workers=1)
    llm = LangchainLLMWrapper(
        chat_model(provider, model, temperature=0.0, max_retries=1), run_config=rc, bypass_n=True
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model), run_config=rc
    )
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=emb),
        "context_recall": LLMContextRecall(llm=llm),
    }


async def score_all(metrics, samples) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {k: [] for k in metrics}
    for s in samples:
        for name, m in metrics.items():
            try:
                out[name].append(float(await m.single_turn_ascore(s)))
            except Exception:
                out[name].append(float("nan"))
    return out


async def main_async(model: str, hosts: list[str], n: int, git_sha: str | None) -> None:
    samples = load_samples(git_sha, n)
    results = {}
    for h in hosts:
        print(f"  scoring via {h} …", flush=True)
        results[h] = await score_all(build_metrics(h, model), samples)

    a, b = hosts
    print("\n" + "=" * 72)
    print(f"  JUDGE HOST COMPARISON — {model}   ({len(samples)} identical inputs)")
    print("=" * 72)
    print(f"  {'metric':18} {a:>12} {b:>12} {'mean|Δ|':>9} {'max|Δ|':>8}")
    print("  " + "-" * 62)
    for metric in results[a]:
        va, vb = results[a][metric], results[b][metric]
        deltas = [abs(x - y) for x, y in zip(va, vb) if not (math.isnan(x) or math.isnan(y))]
        ma = statistics.mean([v for v in va if not math.isnan(v)] or [float("nan")])
        mb = statistics.mean([v for v in vb if not math.isnan(v)] or [float("nan")])
        print(
            f"  {metric:18} {ma:12.3f} {mb:12.3f} "
            f"{(statistics.mean(deltas) if deltas else float('nan')):9.3f} "
            f"{(max(deltas) if deltas else float('nan')):8.2f}"
        )
    faith_d = [
        abs(x - y)
        for x, y in zip(results[a]["faithfulness"], results[b]["faithfulness"])
        if not (math.isnan(x) or math.isnan(y))
    ]
    md = statistics.mean(faith_d) if faith_d else float("nan")
    verdict = "TRANSFERS" if md <= 0.10 else ("BORDERLINE" if md <= 0.20 else "DOES NOT TRANSFER")
    print(f"\n  faithfulness calibration across hosts: {verdict}  (mean |Δ| = {md:.3f})")
    print("=" * 72 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare one judge model across two providers.")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--hosts", nargs=2, default=["groq", "nvidia"])
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--git-sha", default=None, help="replay items from this run (default: largest)")
    args = ap.parse_args()
    asyncio.run(main_async(args.model, args.hosts, args.n, args.git_sha))


if __name__ == "__main__":
    main()
