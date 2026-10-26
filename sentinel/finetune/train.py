"""Fine-tune the cross-encoder reranker with a hand-written PyTorch training loop.

No Trainer, no .fit(): every step is here — the dataset and its per-epoch negative sampling,
the tokenizing collate function, the forward pass, the loss, backward, AdamW, the warmup/decay
schedule, gradient clipping, mixed precision, validation, early stopping and checkpointing.

Loss: grouped softmax cross-entropy. Each training example is one question with its positive
passage and N hard negatives, all scored by the cross-encoder; a softmax over those N + 1 scores
should put its mass on the positive (target index 0). Why this rather than pointwise binary
cross-entropy:
  * it optimises what the reranker is used for — the order of candidates for one question —
    rather than each pair's absolute probability, which a ranking never reads;
  * every negative is contrasted with its own question's positive, so the hardest negatives get
    the most gradient automatically, instead of being diluted among easy pairs;
  * pointwise BCE with 1 positive : N negatives is a 1 : N class imbalance that needs
    re-weighting, and on the RFCs some mined "negatives" are partly relevant: forcing them to
    an absolute 0 is a stronger (wrong) claim than ranking them below the positive.

v2 — learning from a teacher. v1 trained on the synthetic labels alone and got worse on real
questions: the passage a question was generated from is often not its best answer, and a mined
"negative" often is. With teacher scores from a stronger cross-encoder (scripts/teacher_scores.py):
  * denoising — a question is kept only if the teacher ranks its labelled positive in its top
    `teacher_max_rank`, and a negative the teacher scores at or above the positive is dropped as
    a likely correct answer (RocketQA- / InPars-style filtering);
  * distillation — the loss adds KL(teacher || student) over each group's softmax, weighted by
    `distill_weight`, so the student learns the teacher's graded judgement of every candidate
    rather than only "the positive beats the rest", and stays anchored to sensible relevance.
The student keeps its size, so serving cost does not change.

Inputs are tokenized exactly as CrossEncoder.predict does at inference (question, passage;
truncation to 512 tokens) — checked to reproduce its scores to 2e-6 — so the model is trained
on what it will see. Only torch, transformers and numpy are needed, so this also runs on a
plain Colab/Kaggle GPU: see notebooks/reranker_train.ipynb and scripts/train_reranker.py.
"""

from __future__ import annotations

import gzip
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BASE_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"   # the weights the service runs today


@dataclass
class TrainConfig:
    data_dir: str = "data/reranker"
    out_dir: str = "runs/reranker"
    base_model: str = BASE_MODEL
    base_revision: str = BASE_REVISION
    negatives: int = 7               # hard negatives per question per step (group = 1 + this)
    batch_size: int = 16             # questions per step, so batch_size * (1 + negatives) pairs
    max_length: int = 512            # as at inference
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup: float = 0.1              # share of all steps spent warming up the learning rate
    epochs: int = 3
    clip: float = 1.0                # max gradient norm
    patience: int = 1                # evaluations without improvement before stopping
    eval_every: int = 0              # steps between validations; 0 = once per epoch
    exclude_gold: bool = False       # drop ground-truth passages from training entirely
    amp: bool = True                 # fp16 autocast, CUDA only
    seed: int = 13
    device: str = "auto"
    num_workers: int = -1            # -1: 2 on CUDA, 0 elsewhere
    log_every: int = 50
    limit_train: int = 0             # 0 = all; small numbers for smoke runs
    limit_val: int = 0
    eval_batch_pairs: int = 256
    gt_path: str = "data/ground_truth.jsonl"
    teacher_path: str = ""           # teacher scores (v2); empty = labels only (v1)
    teacher_max_rank: int = 3        # keep a question only if the teacher ranks its positive <= this
    distill_weight: float = 0.5      # loss = (1 - w) * label cross-entropy + w * KL to the teacher
    distill_temperature: float = 1.0
    notes: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- setup


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gold_chunk_ids(gt_path: Path, chunk_text: dict[str, str]) -> set[str]:
    """The chunks the ground truth marks relevant (matched by exact text, as the eval does)."""
    text2id = {t: cid for cid, t in chunk_text.items()}
    return {text2id[t] for g in read_jsonl(gt_path) for t in g["relevant_contexts"] if t in text2id}


def exclude_gold(examples: list[dict], gold: set[str]) -> tuple[list[dict], dict]:
    """Remove every trace of the ground-truth passages: questions whose answer is one, and those
    passages as negatives — the strictest no-leakage setting."""
    kept, dropped_q, dropped_n = [], 0, 0
    for ex in examples:
        if ex["positive"] in gold:
            dropped_q += 1
            continue
        negs = [n for n in ex["negatives"] if n not in gold]
        dropped_n += len(ex["negatives"]) - len(negs)
        kept.append({**ex, "negatives": negs})
    return kept, {"questions_dropped": dropped_q, "negatives_dropped": dropped_n}


def apply_teacher(examples: list[dict], teacher: dict[str, list[float]], max_rank: int
                  ) -> tuple[list[dict], dict]:
    """Attach the teacher's score to every candidate and drop what the teacher contradicts:
    questions whose positive it ranks below `max_rank`, and negatives it scores at or above the
    positive (likely correct answers mislabelled as negatives)."""
    kept = []
    stats = {"questions_in": len(examples), "no_teacher_scores": 0, "positive_ranked_low": 0,
             "negatives_in": 0, "negatives_above_positive": 0}
    for ex in examples:
        scores = teacher.get(ex["qid"])
        if scores is None:
            stats["no_teacher_scores"] += 1
            continue
        by_id = dict(zip([ex["positive"], *ex["negatives"]], scores, strict=True))
        pos = by_id[ex["positive"]]
        if 1 + sum(by_id[n] > pos for n in ex["negatives"]) > max_rank:
            stats["positive_ranked_low"] += 1
            continue
        negs = [n for n in ex["negatives"] if by_id[n] < pos]
        stats["negatives_in"] += len(ex["negatives"])
        stats["negatives_above_positive"] += len(ex["negatives"]) - len(negs)
        kept.append({**ex, "negatives": negs, "teacher": by_id})
    stats["questions_kept"] = len(kept)
    return kept, stats


# --------------------------------------------------------------------------- data


class GroupDataset(Dataset):
    """One item = one question, its positive and `negatives` hard negatives. The negatives are
    re-sampled from the mined list every epoch (deterministically from seed, epoch and index),
    so later epochs see different hard negatives rather than memorising one set."""

    def __init__(self, examples: list[dict], chunk_text: dict[str, str], negatives: int, seed: int):
        self.examples = [e for e in examples if len(e["negatives"]) >= negatives]
        self.chunk_text = chunk_text
        self.negatives = negatives
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        ex = self.examples[i]
        rng = random.Random(f"{self.seed}:{self.epoch}:{i}")
        negs = rng.sample(ex["negatives"], self.negatives)
        item = {
            "query": ex["query"],
            "passages": [self.chunk_text[ex["positive"]]] + [self.chunk_text[n] for n in negs],
        }
        if "teacher" in ex:
            item["teacher"] = [ex["teacher"][c] for c in [ex["positive"], *negs]]
        return item


class Collate:
    """Tokenizes a batch of groups into one flat batch of (question, passage) pairs, positive
    first in every group, padded to the batch's longest pair."""

    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict:
        queries = [item["query"] for item in batch for _ in item["passages"]]
        passages = [p for item in batch for p in item["passages"]]
        enc = self.tokenizer(queries, passages, truncation=True, max_length=self.max_length,
                             padding=True, return_tensors="pt")
        out = {**enc, "group_size": len(batch[0]["passages"])}
        if "teacher" in batch[0]:
            out["teacher"] = torch.tensor([t for item in batch for t in item["teacher"]])
        return out


def grouped_softmax_loss(scores: torch.Tensor, group_size: int) -> torch.Tensor:
    """Cross-entropy over each group's scores, the positive (index 0) as the target."""
    logits = scores.view(-1, group_size).float()
    target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return torch.nn.functional.cross_entropy(logits, target)


def distillation_loss(scores: torch.Tensor, teacher: torch.Tensor, group_size: int,
                      temperature: float = 1.0) -> torch.Tensor:
    """KL(teacher || student) between each group's softmax distributions (listwise distillation),
    scaled by T^2 so its gradients keep their size as the temperature changes."""
    s = torch.log_softmax(scores.view(-1, group_size).float() / temperature, dim=-1)
    t = torch.softmax(teacher.view(-1, group_size).float() / temperature, dim=-1)
    return torch.nn.functional.kl_div(s, t, reduction="batchmean") * temperature ** 2


def linear_warmup_decay(warmup_steps: int, total_steps: int):
    """Learning-rate multiplier: 0 -> 1 linearly over the warmup, then linearly back to 0."""
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return factor


def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """Weight decay on weight matrices only — not on biases or LayerNorm, by convention."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim == 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# --------------------------------------------------------------------------- evaluation


@torch.inference_mode()
def score_pairs(model, tokenizer, pairs: list[tuple[str, str]], device: str, max_length: int,
                batch_pairs: int, amp: bool) -> np.ndarray:
    """Scores for (question, passage) pairs, batched by length to waste little on padding."""
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
    out = np.empty(len(pairs), dtype=np.float32)
    model.eval()
    for s in range(0, len(order), batch_pairs):
        idx = order[s: s + batch_pairs]
        enc = tokenizer([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation=True,
                        max_length=max_length, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda" if amp else "cpu", dtype=torch.float16, enabled=amp):
            logits = model(**enc).logits.squeeze(-1)
        out[idx] = logits.float().cpu().numpy()
    return out


def evaluate_candidates(model, tokenizer, examples: list[dict], chunk_text: dict[str, str],
                        device: str, cfg: TrainConfig, amp: bool) -> dict:
    """Validation: rerank each question's full mined candidate set (positive + all its hard
    negatives) and report MRR@10, recall@1 and nDCG@10 of the positive. Fixed candidates, so the
    numbers are comparable across evaluations."""
    pairs, spans = [], []
    for ex in examples:
        cands = [ex["positive"]] + ex["negatives"]
        spans.append((len(pairs), len(cands)))
        pairs.extend((ex["query"], chunk_text[c]) for c in cands)
    scores = score_pairs(model, tokenizer, pairs, device, cfg.max_length, cfg.eval_batch_pairs, amp)
    mrr, r1, ndcg = [], [], []
    for start, n in spans:
        s = scores[start: start + n]
        rank = 1 + int((s[1:] > s[0]).sum())          # rank of the positive (index 0)
        mrr.append(1.0 / rank if rank <= 10 else 0.0)
        r1.append(float(rank == 1))
        ndcg.append(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0)
    return {"mrr@10": float(np.mean(mrr)), "recall@1": float(np.mean(r1)),
            "ndcg@10": float(np.mean(ndcg)), "questions": len(examples)}


# --------------------------------------------------------------------------- training


def train(cfg: TrainConfig) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import transformers

    t_start = time.time()
    seed_everything(cfg.seed)
    device = pick_device(cfg.device)
    use_amp = cfg.amp and device == "cuda"
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    log_path.write_text("")

    def log(event: str, **fields) -> None:
        row = {"event": event, "t": round(time.time() - t_start, 1), **fields}
        with log_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    data = Path(cfg.data_dir)
    chunk_text = {r["chunk_id"]: r["text"] for r in read_jsonl(data / "chunks.jsonl.gz")}
    train_ex = read_jsonl(data / "train.jsonl.gz")
    val_ex = read_jsonl(data / "val.jsonl.gz")
    data_notes = {}
    if cfg.teacher_path:
        teacher = {r["qid"]: r["scores"] for r in read_jsonl(Path(cfg.teacher_path))}
        train_ex, data_notes["train_teacher_filter"] = apply_teacher(train_ex, teacher, cfg.teacher_max_rank)
        val_ex, data_notes["val_teacher_filter"] = apply_teacher(val_ex, teacher, cfg.teacher_max_rank)
    if cfg.exclude_gold:
        gold = gold_chunk_ids(Path(cfg.gt_path), chunk_text)
        train_ex, data_notes["train_gold_exclusion"] = exclude_gold(train_ex, gold)
        val_ex, data_notes["val_gold_exclusion"] = exclude_gold(val_ex, gold)
    if cfg.limit_train:
        train_ex = random.Random(cfg.seed).sample(train_ex, min(cfg.limit_train, len(train_ex)))
    if cfg.limit_val:
        val_ex = random.Random(cfg.seed + 1).sample(val_ex, min(cfg.limit_val, len(val_ex)))

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, revision=cfg.base_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, revision=cfg.base_revision).to(device)

    dataset = GroupDataset(train_ex, chunk_text, cfg.negatives, cfg.seed)
    workers = cfg.num_workers if cfg.num_workers >= 0 else (2 if device == "cuda" else 0)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                        collate_fn=Collate(tokenizer, cfg.max_length), num_workers=workers,
                        generator=torch.Generator().manual_seed(cfg.seed),
                        pin_memory=device == "cuda")
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, linear_warmup_decay(int(cfg.warmup * total_steps), total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log("config", config=asdict(cfg), device=device, amp=use_amp, workers=workers,
        train_questions=len(dataset), val_questions=len(val_ex), steps_per_epoch=steps_per_epoch,
        total_steps=total_steps, data=data_notes, torch=torch.__version__,
        transformers=transformers.__version__, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else None)

    def validate(step: int, epoch: float) -> dict:
        m = evaluate_candidates(model, tokenizer, val_ex, chunk_text, device, cfg, use_amp)
        log("val", step=step, epoch=round(epoch, 3), **m)
        return m

    best = validate(0, 0.0)               # the base model, before any training
    best_step, stale, step, stopped_early = 0, 0, 0, False
    model.save_pretrained(out / "best")
    tokenizer.save_pretrained(out / "best")

    for epoch in range(cfg.epochs):
        dataset.set_epoch(epoch)
        model.train()
        for batch in loader:
            group = batch.pop("group_size")
            teacher = batch.pop("teacher", None)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp):
                scores = model(**batch).logits.squeeze(-1)
                ce = grouped_softmax_loss(scores, group)
                if teacher is None:
                    kd, loss = None, ce
                else:
                    kd = distillation_loss(scores, teacher.to(device), group, cfg.distill_temperature)
                    loss = (1 - cfg.distill_weight) * ce + cfg.distill_weight * kd
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)            # clip the true gradients, not scaled ones
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            if step % cfg.log_every == 0 or step == 1:
                parts = {} if kd is None else {"ce": round(ce.item(), 5), "kd": round(kd.item(), 5)}
                log("train", step=step, epoch=round(step / steps_per_epoch, 3),
                    loss=round(loss.item(), 5), **parts, lr=scheduler.get_last_lr()[0],
                    grad_norm=round(float(grad_norm), 4))
            if cfg.eval_every and step % cfg.eval_every == 0 and step % steps_per_epoch:
                m = validate(step, step / steps_per_epoch)
                if m["mrr@10"] > best["mrr@10"]:
                    best, best_step, stale = m, step, 0
                    model.save_pretrained(out / "best")
                    tokenizer.save_pretrained(out / "best")
                model.train()
        m = validate(step, epoch + 1)
        if m["mrr@10"] > best["mrr@10"]:
            best, best_step, stale = m, step, 0
            model.save_pretrained(out / "best")
            tokenizer.save_pretrained(out / "best")
        else:
            stale += 1
            if stale > cfg.patience - 1 and epoch < cfg.epochs - 1:
                stopped_early = True
                log("early_stop", step=step, epoch=epoch + 1, best_step=best_step)
                break

    summary = {"best_step": best_step, "best_val": best, "steps": step,
               "stopped_early": stopped_early, "seconds": round(time.time() - t_start, 1),
               "device": device, "config": asdict(cfg)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "best" / "sentinel_training.json").write_text(json.dumps(summary, indent=2) + "\n")
    log("done", **{k: v for k, v in summary.items() if k != "config"})
    return summary
