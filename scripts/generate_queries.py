"""Write synthetic questions for every window with doc2query (T5), on a GPU or locally.

Standalone on purpose — it needs only torch and transformers, so it runs as-is on Colab or Kaggle
(notebooks/reranker_querygen.ipynb) without installing the project:

    python scripts/generate_queries.py \
        --windows data/reranker/windows.jsonl.gz --out data/reranker/queries.jsonl.gz

Sampling follows the model card (top-p 0.95, inputs truncated to 320 word pieces, outputs up to
64). It is reproducible: the model revision is pinned, and every batch reseeds from --seed and
its own index, so a run resumed after a disconnect writes the same questions as an unbroken one.
Results are appended per batch; re-running skips windows already written.
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "doc2query/msmarco-t5-base-v1"
REVISION = "e673dca0dff1f19fda73ac62420eedf0219e692b"


def _open(path: Path, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if path.suffix == ".gz" else path.open(mode, encoding="utf-8")


def _device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--revision", default=REVISION, help="pin the model's exact weights")
    ap.add_argument("--per-window", type=int, default=4, help="questions sampled per window")
    ap.add_argument("--batch-size", type=int, default=32, help="windows per generate() call")
    ap.add_argument("--max-input", type=int, default=320)
    ap.add_argument("--max-output", type=int, default=64)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    ap.add_argument("--limit", type=int, default=0, help="only the first N windows (smoke test)")
    args = ap.parse_args()

    with _open(args.windows, "r") as fh:
        windows = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        windows = windows[: args.limit]
    done = set()
    if args.out.exists():
        with _open(args.out, "r") as fh:
            done = {json.loads(line)["window_id"] for line in fh if line.strip()}

    device = _device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # fp32 throughout: T5 v1.1 checkpoints overflow in fp16, and a T4 has no bf16.
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.revision).to(device).eval()

    batches = [windows[i: i + args.batch_size] for i in range(0, len(windows), args.batch_size)]
    todo = [(b, batch) for b, batch in enumerate(batches) if any(w["window_id"] not in done for w in batch)]
    print(f"{len(windows)} windows, {len(done)} already written, {len(todo)} batches to run on {device}",
          flush=True)
    t0 = time.time()
    with _open(args.out, "a") as out, torch.inference_mode():
        for n, (b, batch) in enumerate(todo, 1):
            torch.manual_seed(args.seed + b)   # per batch: resuming reproduces the same sample
            enc = tok([w["text"] for w in batch], max_length=args.max_input, truncation=True,
                      padding=True, return_tensors="pt").to(device)
            gen = model.generate(**enc, max_length=args.max_output, do_sample=True,
                                 top_p=args.top_p, num_return_sequences=args.per_window)
            texts = tok.batch_decode(gen, skip_special_tokens=True)
            for i, w in enumerate(batch):
                if w["window_id"] in done:
                    continue
                qs = texts[i * args.per_window: (i + 1) * args.per_window]
                out.write(json.dumps({"window_id": w["window_id"], "chunk_id": w["chunk_id"],
                                      "queries": qs}, ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"  batch {n}/{len(todo)}  ({rate:.2f} batches/s)", flush=True)

    meta = {
        "model": args.model, "revision": args.revision, "per_window": args.per_window,
        "batch_size": args.batch_size, "max_input": args.max_input, "max_output": args.max_output,
        "top_p": args.top_p, "seed": args.seed, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "windows": len(windows),
        "seconds_this_run": round(time.time() - t0, 1),
    }
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
