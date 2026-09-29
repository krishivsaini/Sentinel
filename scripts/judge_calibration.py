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

Calibrating a NEW judge (after a model or provider swap):
    python scripts/judge_calibration.py --rejudge openai gpt-6-luna --repeat 2
  This re-scores the exact stored answers the hand labels were written against (run LABEL_SHA),
  so answer drift can't confound the comparison, prints old judge vs new judge side by side, and
  with --repeat N measures run-to-run repeatability (needed for judges that can't be pinned to
  temperature 0).

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
# The eval run whose generated answers HAND_SCORES were written against. Judge scores stored for
# this run are from the original calibrated judge (groq / gpt-oss-120b) — the baseline.
LABEL_SHA = "6e91cd1"

# BLIND labels. HAND_SCORES were written with the judge's score visible beside each answer, which
# can anchor the labeller toward that judge and flatter it in the comparison. BLIND_SCORES were
# written from `--dump-blind`, which shows only question, answer and retrieved passages — no score
# from any judge. They cover the 30 answers in run BLIND_SHA that HAND_SCORES doesn't, so the two
# sets are disjoint. Judge scores stored for BLIND_SHA come from the NVIDIA-hosted gpt-oss-120b.
BLIND_SHA = "d0c59e4"
BLIND_PATH = DATA_DIR / "judge_calibration.blind.jsonl"
BLIND_SCORES: dict[str, float] = {
    "29efbcf9": 0.5,   # DoH — "HTTP exchange" grounded; POST/GET + body placement not in ctx
    "d63ba528": 1.0,   # Basic auth — grounded (rfc7617#0005, #0008)
    "900a5a66": 1.0,   # PKCE — abstention
    "34fe7013": 0.75,  # QUIC loss — formula/constants grounded; "packet numbers not encrypted"
                       #   contradicts rfc9002#0013
    "c950afbb": 0.75,  # SNI — server_name grounded; "server must use it to pick target" unsupported
    "86f13d09": 0.75,  # cookie domain-match — rules verbatim; "§5.3" mislabels the storage model
    "7602d00c": 1.0,   # base64url — alphabet (#0081) + no padding (#0006) grounded
    "831f7b29": 1.0,   # ACME http-01 — grounded (rfc8555#0068)
    "5efcd667": 0.5,   # TCP 3WHS — says A goes SYN-SENT -> SYN-RECEIVED (Fig. 6: -> ESTABLISHED);
                       #   "handshake mandatory for all TCP" unsupported
    "ad0651aa": 1.0,   # JWS Compact — grounded (rfc7515#0023)
    "1fb62edf": 1.0,   # bearer token — three methods + conditions grounded
    "c95f1d2d": 1.0,   # JWT registered claims — abstention (answer was in ctx: incomplete, not
                       #   unfaithful)
    "609324bd": 0.5,   # A record — "host address" loosely grounded; IPv4 not in ctx
    "48f4b791": 1.0,   # OCSP — status values, validity interval, signer grounded
    "95a78b21": 1.0,   # 431 — verbatim
    "64750d1c": 1.0,   # NXDOMAIN — name error + negative caching grounded
    "eb0879da": 1.0,   # JWS alg — grounded (rfc7515#0011)
    "563701c0": 1.0,   # invalid_grant — abstention
    "57039f91": 1.0,   # HSTS max-age — verbatim
    "cb106ee1": 0.75,  # X.509 Subject — subject/Name/binding grounded; "distinguished name" not
    "e6955712": 1.0,   # QUIC connection ID — grounded (rfc9000#0030, #0010, #0174)
    "fc80e9a9": 0.5,   # RFC 3339 — ISO 8601 profile grounded; the actual format asked for is not
    "c5d31ac8": 1.0,   # MSL — 2 min (#0095) + drain rationale (#0028) grounded
    "ebc3da16": 1.0,   # TCP receive window — grounded (rfc9293#0054)
    "cb78b83c": 1.0,   # ALPN — grounded (rfc7301#0002, #0004, #0005)
    "ceb616d9": 1.0,   # HPACK dynamic table — grounded (rfc7541#0006, rfc9113#0014)
    "f3f23941": 1.0,   # Digest nonce — grounded (rfc7616#0007, #0028, #0029)
    "703183ff": 1.0,   # JSON structured types — verbatim
    "2e99d95d": 1.0,   # DoT port 853 — verbatim
    "0781ec61": 1.0,   # TLS 1.3 Certificate message — grounded (rfc8446#0025, #0064)
}

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
# Refreshed against the SHIPPED judge (gpt-oss-120b) on its own generated answers (18-item
# spread from the run). Each label = author's faithfulness judgment of the answer's claims vs the
# retrieved contexts it was given (grounding, not truth), verified chunk-by-chunk.
HAND_SCORES: dict[str, float] = {
    "7dc0e4be": 1.0,   # RFC 8174 uppercase rule — grounded
    "7ee78eef": 1.0,   # cache freshness (lifetime vs age) — grounded
    "8918f788": 1.0,   # URI Template simple string expansion — grounded
    "5644d0fb": 1.0,   # Sec-WebSocket-Accept steps — grounded (GUID concat in ctx)
    "1d37c635": 1.0,   # WebSocket 101 — grounded
    "605c1219": 1.0,   # URI generic-syntax 5 components — grounded
    "908c3495": 0.75,  # 308 — core grounded, over-elaborates (judge 0.67)
    "de139471": 1.0,   # 429 Too Many Requests — grounded
    "17926543": 1.0,   # SETTINGS_HEADER_TABLE_SIZE — grounded
    "36dba029": 1.0,   # cookie Secure attribute — grounded
    "97ae64c7": 0.5,   # MUST NOT — cites rfc2119#0000 but retrieved ctx are other RFCs (judge 0.75)
    "06b1a8a7": 1.0,   # QUIC stateless reset — grounded
    "6db1d152": 1.0,   # PATCH method — grounded
    "3b3680d9": 1.0,   # cookie HttpOnly — grounded
    "8bc6d836": 0.5,   # Sec-WebSocket-Key purpose — weak grounding in retrieved ctx (judge 0.33)
    "b2650475": 0.6,   # MUST/SHOULD/MAY — definitions only partly in retrieved chunk (judge 0.40)
    "04fa2607": 0.9,   # percent-encoding — grounded (judge 0.75)
    "a0abd31b": 1.0,   # QPACK separate streams — grounded
}


def qkey(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]


def _load_items(git_sha: str | None = None) -> list[dict]:
    """Load scored items for `git_sha`, or by default the run with the most rows."""
    if not EVAL_DB_PATH.exists():
        sys.exit(f"{EVAL_DB_PATH} not found — run `python -m sentinel.eval.run_eval` first.")
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    if git_sha is None:
        top = conn.execute(
            "SELECT git_sha, COUNT(*) n FROM eval_items GROUP BY git_sha ORDER BY n DESC LIMIT 1"
        ).fetchone()
        if not top:
            sys.exit("eval_items is empty — run the eval first.")
        git_sha = top["git_sha"]
    rows = conn.execute(
        "SELECT question, faithfulness, generated_answer, retrieved_json "
        "FROM eval_items WHERE git_sha = ? ORDER BY question",
        (git_sha,),
    ).fetchall()
    if not rows:
        sys.exit(f"no stored items for run {git_sha}")
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


def _stats(hand: list[float], judge: list[float]) -> str:
    """Pearson / Spearman / MAE over pairs where the judge produced a score."""
    import math

    pairs = [(h, j) for h, j in zip(hand, judge) if not math.isnan(j)]
    if len(pairs) < 3 or len({j for _, j in pairs}) < 2:
        return f"n={len(pairs)}  (too few / constant scores for correlation)"
    h, j = zip(*pairs)
    mae = sum(abs(a - b) for a, b in pairs) / len(pairs)
    return f"n={len(pairs)}  r={_pearson(h, j):+.2f}  rho={_spearman(h, j):+.2f}  MAE={mae:.3f}"


def dump_blind() -> None:
    """Write the answers HAND_SCORES doesn't cover, with NO judge score of any kind — only what a
    labeller needs: question, answer, and each retrieved passage with its chunk id."""
    conn = sqlite3.connect(EVAL_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT question, generated_answer, retrieved_json FROM eval_items "
        "WHERE git_sha = ? ORDER BY question",
        (BLIND_SHA,),
    ).fetchall()
    conn.close()
    n = 0
    with BLIND_PATH.open("w", encoding="utf-8") as fh:
        for r in rows:
            k = qkey(r["question"])
            if k in HAND_SCORES:
                continue
            passages = [
                {"chunk_id": c["chunk_id"], "text": c["text"]}
                for c in json.loads(r["retrieved_json"])
            ]
            fh.write(json.dumps({"qkey": k, "question": r["question"],
                                 "answer": r["generated_answer"], "passages": passages},
                                ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} blind items (no judge scores) to {BLIND_PATH}")


def rejudge(provider: str, model: str, repeat: int, labels: str = "original") -> None:
    """Re-score labelled answers with a candidate judge on pinned inputs, and compare against the
    labels and the judge scores stored for that run. labels: "original" | "blind"."""
    import asyncio
    import math
    import warnings

    warnings.filterwarnings("ignore")
    import sentinel.eval  # noqa: F401  (installs the ragas <-> langchain-community shim)
    from ragas import SingleTurnSample
    from ragas.llms.base import LangchainLLMWrapper
    from ragas.metrics import Faithfulness
    from ragas.run_config import RunConfig

    from sentinel.eval.run_eval import _coerce_faithfulness, _score_async
    from sentinel.generate import is_abstention
    from sentinel.llm import accepts_temperature, chat_model

    if labels == "blind":
        label_set, sha, stored_judge = BLIND_SCORES, BLIND_SHA, "nvidia/gpt-oss-120b"
        if not label_set:
            sys.exit("BLIND_SCORES is empty — run --dump-blind and label the items first.")
    else:
        label_set, sha, stored_judge = HAND_SCORES, LABEL_SHA, "groq/gpt-oss-120b"
    items = {it["qkey"]: it for it in _load_items(sha)}
    keys = [k for k in label_set if k in items]
    llm = LangchainLLMWrapper(
        chat_model(provider, model, temperature=0.0, max_retries=2),
        run_config=RunConfig(timeout=180, max_retries=1, max_workers=1),
        bypass_n=True,
        bypass_temperature=not accepts_temperature(provider, model),
    )
    metric = Faithfulness(llm=llm)

    async def score_all() -> list[dict[str, float]]:
        # All repeats share ONE event loop — a second asyncio.run() would orphan the client's
        # cached async transport ("Event loop is closed"), the failure run_eval also avoids.
        runs = []
        for rep in range(repeat):
            out = {}
            for i, k in enumerate(keys, 1):
                it = items[k]
                sample = SingleTurnSample(
                    user_input=it["question"],
                    response=it["generated_answer"],
                    retrieved_contexts=it["contexts"],
                )
                try:
                    # run_eval's backoff honours the server's retry-after hint, so a rate-limited
                    # free-tier judge (Groq) waits out its window instead of recording a NaN.
                    raw = await _score_async(metric, sample)
                except Exception as e:
                    print(f"    [{rep + 1}] {k} error: {type(e).__name__}", flush=True)
                    raw = float("nan")
                # Same coercion run_eval applies, so new scores are comparable to stored ones.
                out[k] = (
                    float("nan")
                    if math.isnan(raw) and not is_abstention(it["generated_answer"])
                    else _coerce_faithfulness(raw, is_abstention(it["generated_answer"]))
                )
                print(f"    run {rep + 1}/{repeat}  item {i}/{len(keys)}", end="\r", flush=True)
            runs.append(out)
        print()
        return runs

    print(f"\nre-judging {len(keys)} {labels}-labelled answers from run {sha} "
          f"with {provider} / {model}  (x{repeat})")
    runs = asyncio.run(score_all())

    hand = [label_set[k] for k in keys]
    base = [items[k]["ragas_faithfulness"] for k in keys]
    hdr = "".join(f"  new#{i + 1}" for i in range(repeat))
    print(f"\n  {'qkey':8} {'hand':>5} {'old':>5}{hdr}  question")
    print("  " + "-" * (40 + 7 * repeat))
    for k, h, b in zip(keys, hand, base):
        cells = "".join(f" {r[k]:6.2f}" for r in runs)
        print(f"  {k:8} {h:5.2f} {b:5.2f}{cells}  {items[k]['question'][:34]}")
    print("  " + "-" * (40 + 7 * repeat))
    print(f"  stored judge ({stored_judge}) vs hand : {_stats(hand, base)}")
    for i, r in enumerate(runs, 1):
        print(f"  new judge run {i} ({model}) vs hand{' ' * max(0, 14 - len(model))}: "
              f"{_stats(hand, [r[k] for k in keys])}")
    # Abstentions are coerced to 1.0 for every judge and labelled 1.0, so they agree for free and
    # flatter all judges equally. Re-state the comparison on the answers that actually make claims.
    answered = [i for i, k in enumerate(keys) if not is_abstention(items[k]["generated_answer"])]
    if len(answered) < len(keys):
        h_a = [hand[i] for i in answered]
        print(f"  excluding {len(keys) - len(answered)} abstentions:")
        print(f"    stored judge vs hand : {_stats(h_a, [base[i] for i in answered])}")
        for n, r in enumerate(runs, 1):
            print(f"    new run {n} vs hand    : {_stats(h_a, [r[keys[i]] for i in answered])}")
    if repeat > 1:
        deltas = [
            abs(runs[0][k] - runs[1][k])
            for k in keys
            if not (math.isnan(runs[0][k]) or math.isnan(runs[1][k]))
        ]
        if deltas:
            changed = sum(d > 0 for d in deltas)
            print(f"  repeatability (run 1 vs 2): mean |Δ| = {sum(deltas) / len(deltas):.3f}, "
                  f"max {max(deltas):.2f}, {changed}/{len(deltas)} items changed")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate the Ragas faithfulness judge against hand scores.")
    ap.add_argument("--dump", type=int, metavar="N", help="write N spread items to the review file")
    ap.add_argument("--rejudge", nargs=2, metavar=("PROVIDER", "MODEL"),
                    help="re-score the hand-labelled answers with a candidate judge")
    ap.add_argument("--repeat", type=int, default=1, help="with --rejudge: runs, for repeatability")
    ap.add_argument("--dump-blind", action="store_true",
                    help="write the unlabelled answers with no judge scores, for blind labelling")
    ap.add_argument("--labels", choices=["original", "blind"], default="original",
                    help="with --rejudge: which label set to calibrate against")
    args = ap.parse_args()
    if args.dump:
        dump(args.dump)
    elif args.dump_blind:
        dump_blind()
    elif args.rejudge:
        rejudge(args.rejudge[0], args.rejudge[1], args.repeat, args.labels)
    else:
        compute()


if __name__ == "__main__":
    main()
