"""Probe a candidate judge model for Ragas structured-output reliability.

The open question when switching judges is NOT cost — it's whether the model reliably returns
output Ragas can parse. `gpt-oss-120b` only does so with `reasoning_effort="low"` (at default
effort it returns reasoning-polluted output and Ragas yields NaN). A larger judge is *hypothesised*
to work at full effort; this measures that instead of assuming it.

For N ground-truth items it runs the three judged metrics and reports, per metric:
  ok / NaN / error counts, plus the scores — so a judge that "works" but returns garbage is
  visible, not just one that crashes.

    python scripts/probe_judge.py --provider nvidia --model nvidia/llama-3.1-nemotron-ultra-253b-v1
    python scripts/probe_judge.py --provider groq --model openai/gpt-oss-120b   # baseline

Run this BEFORE changing config defaults or unwinding the throttling constants.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")

import sentinel.eval  # noqa: E402  (installs the ragas<->langchain-community shim)
from ragas import SingleTurnSample  # noqa: E402
from ragas.embeddings.base import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms.base import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from sentinel.config import DASHBOARD_DATA_DIR, settings  # noqa: E402
from sentinel.llm import accepts_temperature, chat_model  # noqa: E402


def load_samples(n: int) -> list[SingleTurnSample]:
    """Reuse answers from the last exported eval run, so the judge is scored on real generated
    answers + their real retrieved contexts (no new generation calls, no extra quota)."""
    path = DASHBOARD_DATA_DIR / "latest.json"
    if not path.exists():
        sys.exit(f"{path} not found — run an eval first (it supplies the answers to judge).")
    items = json.loads(path.read_text())["per_item"]
    step = max(1, len(items) // n)
    picked = items[::step][:n]
    return [
        SingleTurnSample(
            user_input=it["question"],
            response=it["generated_answer"],
            retrieved_contexts=[c["text"] for c in it["retrieved"]],
            reference=it["generated_answer"],  # recall needs a reference; the answer stands in
        )
        for it in picked
    ]


async def main_async(provider: str, model: str, n: int, pause: float) -> int:
    from langchain_huggingface import HuggingFaceEmbeddings

    rc = RunConfig(timeout=180, max_retries=1, max_workers=1)
    judge = chat_model(provider, model, temperature=0.0, max_retries=1)
    llm = LangchainLLMWrapper(
        judge, run_config=rc, bypass_n=True,
        bypass_temperature=not accepts_temperature(provider, model),
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model), run_config=rc
    )
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=emb),
        "context_recall": LLMContextRecall(llm=llm),
    }

    samples = load_samples(n)
    tally = {k: {"ok": 0, "nan": 0, "err": 0, "scores": []} for k in metrics}
    print(f"\nprobing judge: {provider} / {model}   ({len(samples)} items)\n")

    for i, s in enumerate(samples, 1):
        line = [f"  [{i}/{len(samples)}]"]
        for name, metric in metrics.items():
            try:
                v = float(await metric.single_turn_ascore(s))
                if math.isnan(v):
                    tally[name]["nan"] += 1
                    line.append(f"{name[:5]}=NaN")
                else:
                    tally[name]["ok"] += 1
                    tally[name]["scores"].append(v)
                    line.append(f"{name[:5]}={v:.2f}")
            except Exception as e:
                tally[name]["err"] += 1
                line.append(f"{name[:5]}=ERR({type(e).__name__})")
            await asyncio.sleep(pause)
        print("  ".join(line), flush=True)

    print("\n" + "=" * 66)
    print(f"  JUDGE PROBE — {provider} / {model}")
    print("=" * 66)
    total_bad = 0
    for name, t in tally.items():
        n_all = t["ok"] + t["nan"] + t["err"]
        mean = sum(t["scores"]) / len(t["scores"]) if t["scores"] else float("nan")
        total_bad += t["nan"] + t["err"]
        print(
            f"  {name:17} ok={t['ok']:2}/{n_all:2}  NaN={t['nan']:2}  err={t['err']:2}  "
            f"mean={mean:.3f}"
        )
    verdict = "USABLE" if total_bad == 0 else ("DEGRADED" if total_bad <= 2 else "UNUSABLE")
    print(f"\n  parse reliability: {verdict}  ({total_bad} NaN/error across all metrics)")
    print("=" * 66 + "\n")
    return 0 if verdict != "UNUSABLE" else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Probe a judge model's Ragas parse reliability.")
    ap.add_argument("--provider", default="nvidia")
    ap.add_argument("--model", required=True)
    ap.add_argument("-n", type=int, default=5, help="items to judge")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between judge calls")
    args = ap.parse_args()
    t0 = time.perf_counter()
    code = asyncio.run(main_async(args.provider, args.model, args.n, args.pause))
    print(f"  elapsed {time.perf_counter() - t0:.0f}s")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
