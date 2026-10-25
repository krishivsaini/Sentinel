"""Fine-tune the reranker: the command line for sentinel/finetune/train.py.

    python scripts/train_reranker.py --smoke                  # tiny CPU run, < 2 minutes
    python scripts/train_reranker.py --out-dir runs/reranker  # the full run (GPU recommended)
    python scripts/train_reranker.py --exclude-gold ...       # no ground-truth passage anywhere

Every hyperparameter is a flag and is written, with the library versions and device, to the
first line of <out-dir>/log.jsonl. The best checkpoint (by validation MRR@10, starting from the
base model itself) is saved to <out-dir>/best and loads directly with sentence-transformers'
CrossEncoder. Weights are not committed (runs/ is gitignored).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel.finetune.train import TrainConfig, train  # noqa: E402

SMOKE = dict(limit_train=48, limit_val=24, negatives=3, batch_size=8, max_length=128, epochs=1,
             device="cpu", log_every=1, eval_batch_pairs=64, out_dir="runs/reranker-smoke")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true", help="tiny CPU run to check the loop works")
    for f in fields(TrainConfig):
        if f.name == "notes":
            continue
        flag = "--" + f.name.replace("_", "-")
        if f.type == "bool":
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=None)
        else:
            ap.add_argument(flag, type={"int": int, "float": float}.get(f.type, str), default=None)
    args = ap.parse_args()
    overrides = SMOKE.copy() if args.smoke else {}
    overrides.update({k: v for k, v in vars(args).items() if v is not None and k != "smoke"})
    train(TrainConfig(**overrides))


if __name__ == "__main__":
    main()
