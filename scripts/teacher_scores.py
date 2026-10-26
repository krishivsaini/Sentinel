"""Score every training candidate with a stronger reranker — the teacher for v2.

v1 showed the synthetic labels are noisy: the passage a question was generated from is often not
its best answer, and a mined "negative" often is. A larger cross-encoder scores every (question,
candidate) pair in data/reranker/{train,val}.jsonl.gz; scripts/train_reranker.py then uses those
scores to drop questions and negatives the teacher disagrees with, and as soft targets
(distillation), so the small student learns from the teacher's judgement as well as the labels.

Standalone (torch + transformers) so it runs as-is on a Colab/Kaggle GPU:

    python scripts/teacher_scores.py --out data/reranker/teacher.jsonl.gz

Writes one line per question: {"qid", "scores": [positive, negative_1, ...]}, in the order the
candidates appear in the data files. Questions are scored in blocks, each block length-sorted to
waste little on padding and appended as it finishes, so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer

TEACHER = "BAAI/bge-reranker-base"                       # XLM-RoBERTa, ~278M parameters, MIT
TEACHER_REVISION = "2cfc18c9415c912f9d8155881c133215df768a70"


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/reranker"))
    ap.add_argument("--out", type=Path, default=Path("data/reranker/teacher.jsonl.gz"))
    ap.add_argument("--model", default=TEACHER)
    ap.add_argument("--revision", default=TEACHER_REVISION)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-pairs", type=int, default=128)
    ap.add_argument("--block", type=int, default=256, help="questions per resumable block")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions per split")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else (
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    amp = device == "cuda"
    chunk_text = {r["chunk_id"]: r["text"] for r in read_jsonl(args.data_dir / "chunks.jsonl.gz")}
    questions = []
    for split in ("train", "val"):
        rows = read_jsonl(args.data_dir / f"{split}.jsonl.gz")
        questions += rows[: args.limit] if args.limit else rows
    done = {r["qid"] for r in read_jsonl(args.out)} if args.out.exists() else set()
    todo = [q for q in questions if q["qid"] not in done]

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.revision).to(device).eval()
    print(f"{len(questions)} questions, {len(done)} already scored, {len(todo)} to go on {device}",
          flush=True)

    t0, pairs_done = time.time(), 0
    with gzip.open(args.out, "at", encoding="utf-8") as out, torch.inference_mode():
        for b in range(0, len(todo), args.block):
            block = todo[b: b + args.block]
            pairs, owner = [], []
            for qi, q in enumerate(block):
                for c in [q["positive"], *q["negatives"]]:
                    pairs.append((q["query"], chunk_text[c]))
                    owner.append(qi)
            order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
            scores = [0.0] * len(pairs)
            for s in range(0, len(order), args.batch_pairs):
                idx = order[s: s + args.batch_pairs]
                enc = tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation=True,
                          max_length=args.max_length, padding=True, return_tensors="pt").to(device)
                with torch.autocast("cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
                    logits = model(**enc).logits.view(-1).float().cpu().tolist()
                for i, v in zip(idx, logits, strict=True):
                    scores[i] = v
            per_q: list[list[float]] = [[] for _ in block]
            for i, qi in enumerate(owner):
                per_q[qi].append(round(scores[i], 4))
            for q, s in zip(block, per_q, strict=True):
                out.write(json.dumps({"qid": q["qid"], "scores": s}) + "\n")
            out.flush()
            pairs_done += len(pairs)
            rate = pairs_done / (time.time() - t0)
            print(f"  {min(b + args.block, len(todo))}/{len(todo)} questions  ({rate:.0f} pairs/s)",
                  flush=True)

    meta = {"model": args.model, "revision": args.revision, "max_length": args.max_length,
            "device": device, "amp": amp, "questions": len(questions),
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "seconds_this_run": round(time.time() - t0, 1)}
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
