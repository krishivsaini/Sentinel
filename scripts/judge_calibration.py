"""Judge calibration (§12, FR-E6) — Phase 3, Day 10.

Is the Ragas faithfulness *judge* trustworthy? An eval gate is only as credible as the judge
behind it, so we don't take the judge on faith: we hand-score a spread of generated answers for
faithfulness (0 = unfaithful/hallucinated, 1 = fully grounded) and correlate our human labels
against the judge's scores. A calibrated modest correlation beats an uncalibrated pretty number
— we report whatever we find, honestly, in data/ground_truth_audit.md.

Flow:
  1. Run the eval so answers + judge scores exist:   python -m sentinel.eval.run_eval
  2. Dump a spread of items to review:                python scripts/judge_calibration.py --dump 20
  3. Read each item in the review file; fill HAND_SCORES below with your own faithfulness label.
  4. Compute + print the calibration stats:          python scripts/judge_calibration.py

The review file is regenerable and gitignored; HAND_SCORES (this file) is the durable, auditable
record of the human labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sentinel.config import DATA_DIR, EVAL_DB_PATH  # noqa: E402

REVIEW_PATH = DATA_DIR / "judge_calibration.review.jsonl"

# ---------------------------------------------------------------------------------------------
# HUMAN LABELS. Key = qkey(question) (first 8 hex of sha1). Value = faithfulness in [0, 1]:
#   1.0  every claim in the answer is supported by the retrieved contexts (or a correct abstention)
#   0.5  partially grounded / one unsupported or overstated claim
#   0.0  a material claim is unsupported by the contexts (hallucination)
# Filled by reading data/judge_calibration.review.jsonl after an eval run (step 3 above).
# Each label is the author's faithfulness judgment of the *generated answer vs. the retrieved
# contexts it was given* (grounding, not mere truth) — verified chunk-by-chunk. Notable cases:
#   34fe7013 QUIC loss: answer states the kTimeThreshold/kGranularity formula, but the cited
#            chunk is about packet reordering — the formula isn't in the retrieved text.  -> 0.5
#   c5d31ac8 MSL: "2 minutes" cites the ISN chunk (rfc9293#0095); MSL def wasn't retrieved. -> 0.5
#   563701c0 invalid_grant: "Section 5.2" cites the §11.4 registry chunk; never defines it.  -> 0.5
#   17926543 SETTINGS_HEADER_TABLE_SIZE: claim is verbatim in CTX1 -> fully grounded 1.0 (judge 0.5).
HAND_SCORES: dict[str, float] = {
    "7dc0e4be": 1.0,   # RFC 8174 uppercase rule — grounded in rfc8174#0000
    "7ee78eef": 1.0,   # cache freshness (lifetime vs age) — grounded in the §4.2 chunks
    "34fe7013": 0.5,   # QUIC loss formula NOT in retrieved context (judge said 1.0)
    "8918f788": 1.0,   # URI Template simple string expansion — grounded
    "831f7b29": 1.0,   # ACME http-01 — grounded in rfc8555#0067/0068
    "1fb62edf": 0.75,  # bearer 3 methods — URI-query method not in retrieved context
    "1d37c635": 1.0,   # WebSocket 101 — grounded in rfc6455#0012
    "c95f1d2d": 1.0,   # JWT registered claims ("include") — grounded
    "908c3495": 1.0,   # 308 Permanent Redirect — grounded
    "de139471": 1.0,   # 429 Too Many Requests — grounded
    "17926543": 1.0,   # SETTINGS_HEADER_TABLE_SIZE verbatim in CTX1 (judge said 0.5)
    "563701c0": 0.5,   # invalid_grant miscited + undefined (judge said 1.0)
    "57039f91": 1.0,   # HSTS max-age — grounded in rfc6797#0021
    "e6955712": 0.75,  # QUIC connection ID — primary citation is the ack-eliciting chunk
    "6db1d152": 1.0,   # PATCH method + idempotency — grounded
    "c5d31ac8": 0.5,   # MSL "2 minutes" not in retrieved context (judge said 1.0)
    "ebc3da16": 0.75,  # TCP receive window — cited chunk is urgent-data; weak grounding
    "ceb616d9": 1.0,   # HPACK dynamic table — grounded in rfc7541 dynamic-table chunks
    "b2650475": 1.0,   # MUST/SHOULD grounded + honestly flags MAY as absent (judge 0.8)
    "a0abd31b": 1.0,   # QPACK separate streams — grounded in rfc9114#0006 + rfc9204#0005
}


def qkey(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]


def _load_items() -> list[dict]:
    """Load scored items for the git_sha that has the most rows (the latest full-ish run)."""
    if not EVAL_DB_PATH.exists():
        sys.exit(f"{EVAL_DB_PATH} not found — run `python -m sentinel.eval.run_eval` first.")
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    top = conn.execute(
        "SELECT git_sha, COUNT(*) n FROM eval_items GROUP BY git_sha ORDER BY n DESC LIMIT 1"
    ).fetchone()
    if not top:
        sys.exit("eval_items is empty — run the eval first.")
    rows = conn.execute(
        "SELECT question, faithfulness, generated_answer, retrieved_json "
        "FROM eval_items WHERE git_sha = ? ORDER BY question",
        (top["git_sha"],),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        contexts = [c["text"] for c in json.loads(r["retrieved_json"])]
        out.append(
            {
                "qkey": qkey(r["question"]),
                "question": r["question"],
                "ragas_faithfulness": r["faithfulness"],
                "generated_answer": r["generated_answer"],
                "contexts": contexts,
            }
        )
    return out


def _select(items: list[dict], n: int) -> list[dict]:
    """Deterministic, evenly-spaced spread (same policy as the eval subset)."""
    if n >= len(items):
        return items
    if n <= 1:
        return items[:1]
    idxs = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idxs]


def dump(n: int) -> None:
    items = _select(_load_items(), n)
    with REVIEW_PATH.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {REVIEW_PATH}")
    print("Review each, then add its qkey -> your 0/0.5/1 faithfulness label to HAND_SCORES.")


def _pearson(x, y) -> float:
    from scipy.stats import pearsonr

    return float(pearsonr(x, y)[0])


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y)[0])


def _cohen_kappa(a, b) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b))


def compute() -> None:
    items = {it["qkey"]: it for it in _load_items()}
    missing = [k for k in items if k not in HAND_SCORES]
    paired = [(k, HAND_SCORES[k], items[k]["ragas_faithfulness"]) for k in HAND_SCORES if k in items]
    if not paired:
        sys.exit("No overlap between HAND_SCORES and scored items. Run --dump and fill HAND_SCORES.")

    hand = [p[1] for p in paired]
    ragas = [p[2] for p in paired]
    n = len(paired)

    print(f"\n=== Judge calibration — {n} hand-scored items (of {len(items)} scored) ===")
    print(f"  {'qkey':8} {'hand':>5} {'ragas':>6}  question")
    print("  " + "-" * 60)
    for k, h, r in paired:
        print(f"  {k:8} {h:5.2f} {r:6.2f}  {items[k]['question'][:38]}")
    print("  " + "-" * 60)

    mae = sum(abs(h - r) for h, r in zip(hand, ragas)) / n
    # binary agreement: does each rater call the answer "faithful" (>= 0.5)?
    hb = [1 if h >= 0.5 else 0 for h in hand]
    rb = [1 if r >= 0.5 else 0 for r in ragas]
    agree = sum(a == b for a, b in zip(hb, rb)) / n

    print(f"  n              : {n}")
    print(f"  Pearson r      : {_pearson(hand, ragas):+.3f}" if len(set(ragas)) > 1 else "  Pearson r      : n/a (ragas constant)")
    print(f"  Spearman rho   : {_spearman(hand, ragas):+.3f}" if len(set(ragas)) > 1 else "  Spearman rho   : n/a")
    print(f"  MAE            : {mae:.3f}")
    print(f"  binary agree   : {agree:.1%}  (faithful vs not, threshold 0.5)")
    try:
        print(f"  Cohen's kappa  : {_cohen_kappa(hb, rb):+.3f}")
    except Exception as e:
        print(f"  Cohen's kappa  : n/a ({e})")
    if missing:
        print(f"\n  ({len(missing)} scored items not yet hand-labelled)")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the Ragas faithfulness judge against hand scores.")
    ap.add_argument("--dump", type=int, metavar="N", help="write N spread items to the review file")
    args = ap.parse_args()
    if args.dump:
        dump(args.dump)
    else:
        compute()


if __name__ == "__main__":
    main()
