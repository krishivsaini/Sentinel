"""Choose which of the base model's layers a smaller student keeps — on validation data only.

    python scripts/pick_student_layers.py --teacher-path data/reranker/teacher-l6.jsonl.gz --keep 2

Every way of keeping `--keep` of the base model's encoder layers (15 for 2 of 6) is built with
keep_layers() and, untrained, scored on held-out synthetic questions by agreement with the
teacher (teacher_mrr@10). The best is the starting point for distillation. Only the validation
split is read; the ground truth plays no part in the choice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel.finetune.train import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    TrainConfig,
    apply_teacher,
    evaluate_candidates,
    keep_layers,
    pick_device,
    read_jsonl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/reranker"))
    ap.add_argument("--teacher-path", type=Path, required=True)
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--questions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=Path("runs/student-layers.json"))
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = pick_device(args.device)
    amp = device == "cuda"
    cfg = TrainConfig()
    chunk_text = {r["chunk_id"]: r["text"] for r in read_jsonl(args.data_dir / "chunks.jsonl.gz")}
    teacher = {r["qid"]: r["scores"] for r in read_jsonl(args.teacher_path)}
    val, _ = apply_teacher(read_jsonl(args.data_dir / "val.jsonl.gz"), teacher, max_rank=0)
    val = random.Random(args.seed).sample(val, min(args.questions, len(val)))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)

    def load():
        return AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, revision=BASE_REVISION)

    n_layers = load().config.num_hidden_layers
    results = []
    for layers in itertools.combinations(range(n_layers), args.keep):
        model = keep_layers(load(), list(layers)).to(device)
        m = evaluate_candidates(model, tok, val, chunk_text, device, cfg, amp)
        results.append({"layers": list(layers), **{k: round(v, 4) for k, v in m.items()}})
        print(json.dumps(results[-1]), flush=True)
    results.sort(key=lambda r: -r["teacher_mrr@10"])
    out = {"keep": args.keep, "questions": len(val), "best": results[0]["layers"], "all": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("best:", ",".join(map(str, results[0]["layers"])))


if __name__ == "__main__":
    main()
