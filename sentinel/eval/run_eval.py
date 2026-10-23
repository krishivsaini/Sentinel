"""Ragas evaluation pipeline (§12) — the headline — Phase 3, Day 9.

Runs the full retrieve -> rerank -> generate pipeline over every item in ground_truth.jsonl,
scores each with Ragas (faithfulness, answer relevance, context recall), attributes failures
(retrieval vs generation), writes an EvalResult to SQLite tagged with the git SHA + exports
JSON for the dashboard, and prints a summary table with the mean faithfulness (the CI gate
reads this).

Design for the free tier (NFR-5):
  * **Per-item, not batch.** ragas' batch `evaluate()` fans out concurrent judge calls that
    trip the 10-15 RPM free tier instantly and can't be resumed. We score item-by-item so each
    call is controlled, backed off, and checkpointed.
  * **Checkpointed + resumable.** Every scored item is written to SQLite keyed by (git_sha,
    question). A run killed by a 429 storm resumes from the DB instead of re-burning quota.
  * **tenacity backoff** around each judge call; the judge LLM's own SDK retries are disabled
    so a 429 isn't amplified into N real hits (the daily-quota trap).
  * **₹0.** Judge = config.judge_provider/model (Groq's llama-3.3-70b free tier, ~1,000 req/day).
    Answer-relevancy embeddings = the same local bge model as retrieval — no OpenAI dependency.

Run:  python -m sentinel.eval.run_eval [--subset N | --ci]   (--ci drives the gate, §13)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import subprocess
import uuid
import warnings
from datetime import datetime, timezone
from functools import partial

# ragas 0.4.x emits DeprecationWarnings nudging toward its v1.0 API: metric classes move to
# `ragas.metrics.collections`, and the Langchain LLM/embeddings wrappers toward `llm_factory`.
# We deliberately keep the (still-supported, <0.5) `ragas.metrics` + Langchain-wrapper path — it
# is the documented, verified way to drive Gemini + local embeddings; the factory path for Gemini
# is thinner. Silence the accurate-but-noisy notices so this headline tool prints clean output.
for _pat in (
    r".*ragas\.metrics.*",
    r"Langchain(LLM|Embeddings)Wrapper is deprecated.*",
):
    warnings.filterwarnings("ignore", message=_pat, category=DeprecationWarning)

# Importing this package first runs sentinel/eval/__init__.py, which installs the
# langchain-community <-> ragas compatibility shim BEFORE ragas is imported below.
from ragas import SingleTurnSample
from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy
from ragas.run_config import RunConfig

import tenacity

from sentinel.config import DATA_DIR, GROUND_TRUTH_PATH, REPO_ROOT, settings
from sentinel.eval import attribution, store
from sentinel.generate import collect_answer, is_abstention
from sentinel.guardrails.output import guard_answer
from sentinel.guardrails.pii import redact_chunks
from sentinel.logging_config import configure_logging, log
from sentinel.retrieve import hybrid_retrieve, warmup
from sentinel.retry import (
    backoff_wait,
    is_daily_quota,
    is_transient,
    log_retry,
    server_retry_delay,
)
from sentinel.schema import (
    AttributionCounts,
    EvalItemResult,
    EvalMeans,
    EvalResult,
    GroundTruthItem,
)

# Main's per-question faithfulness on the CI subset, for the relative gate (write_baseline).
BASELINE_PATH = DATA_DIR / "faithfulness_baseline.json"

# The retry policy is shared with generation and the guardrail classifiers (sentinel.retry);
# these aliases keep the eval's long-standing private names.
_is_transient = is_transient
_is_daily_quota = is_daily_quota
_server_retry_delay = server_retry_delay
_gemini_wait = backoff_wait
_log_retry = log_retry

# A judge occasionally emits structured output ragas can't parse (esp. smaller models). It's not
# a rate-limit, but a fresh call is often well-formed (LLMs are stochastic), so it's worth a
# retry; if it still fails the *item* is skipped rather than crashing the whole run.
_PARSE_SIGNATURES = (
    "failed to parse",
    "outputparser",
    "output_parsing",
    "validation error",
    "nonetype",          # a reasoning model occasionally returns None content; a re-call fixes it
)

# --------------------------------------------------------------------------- helpers


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:  # not a git checkout / git unavailable
        return "nogit"


def load_ground_truth() -> list[GroundTruthItem]:
    if not GROUND_TRUTH_PATH.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH_PATH} not found — build the ground truth first (Day 8)."
        )
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as fh:
        return [GroundTruthItem.model_validate_json(line) for line in fh if line.strip()]


def _subset(items: list[GroundTruthItem], n: int | None) -> list[GroundTruthItem]:
    """Deterministic, evenly-spaced subset. Ground truth is cluster-ordered (3/cluster), so
    even spacing preserves topical spread — a representative slice for the CI gate, not a
    random or head-biased one."""
    if n is None or n >= len(items):
        return items
    if n <= 1:
        return items[:1]
    idxs = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idxs]


def _build_metrics() -> tuple[Faithfulness, ResponseRelevancy, LLMContextRecall]:
    """Wire the three Ragas metrics to the judge (config.judge_provider/model — Groq by default)
    + local embeddings. SDK-level retries on the judge are disabled (max_retries=0): we own retry
    via tenacity so a 429 is one real hit, not N (the RPD trap)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    from sentinel.llm import accepts_temperature, chat_model

    rc = RunConfig(
        timeout=int(settings.judge_call_timeout),
        max_retries=1,      # our tenacity layer is the real retry budget
        max_workers=1,      # sequential — stay under the free-tier RPM
    )
    judge = chat_model(
        settings.judge_provider, settings.judge_model, temperature=0.0, max_retries=0
    )
    # bypass_n=True: some metrics (answer relevancy) ask the judge for n>1 completions in one
    # call; Groq models reject n>1, so the wrapper issues n separate single-completion calls.
    # bypass_temperature: Ragas otherwise sets temperature=0.01 per call, which OpenAI reasoning
    # models reject outright (see sentinel.llm.accepts_temperature).
    llm = LangchainLLMWrapper(
        judge,
        run_config=rc,
        bypass_n=True,
        bypass_temperature=not accepts_temperature(settings.judge_provider, settings.judge_model),
    )
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": settings.torch_device} if settings.torch_device else {},
        ),
        run_config=rc,
    )
    return (
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextRecall(llm=llm),
    )


def _is_parse_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(t in s for t in _PARSE_SIGNATURES)


def _is_retryable(exc: BaseException) -> bool:
    """Retry rate-limit/timeout bursts AND transient judge parse failures (a re-call often parses)."""
    return _is_transient(exc) or (_is_parse_error(exc) and not _is_daily_quota(exc))


def _with_backoff(fn, *args, **kwargs):
    """Call fn with backoff on *transient* failures (429s / quota / timeouts / transient 5xx),
    waiting out the server-requested delay. Non-transient errors re-raise immediately so we don't
    spin on a real bug. Used for BOTH the judged-metric calls and the generation call — any Gemini
    step can be throttled, and a throttle should back off (and, failing that, leave the run
    resumable from its checkpoint), never crash mid-item."""
    retryer = tenacity.Retrying(
        wait=_gemini_wait,
        stop=tenacity.stop_after_attempt(settings.judge_max_retries),
        retry=tenacity.retry_if_exception(_is_transient),
        before_sleep=_log_retry,
        reraise=True,
    )
    return retryer(fn, *args, **kwargs)


async def _score_async(metric, sample: SingleTurnSample) -> float:
    """One judged metric, scored on the *async* API with async backoff. Returns the raw score
    (may be NaN — ragas returns NaN when there's nothing to judge, e.g. a claim-free answer);
    callers coerce.

    We use `single_turn_ascore` (not the sync `single_turn_score`) deliberately: the sync path
    spins a fresh event loop per call, which orphans a chat client's cached async transport
    (ChatGroq raises "Event loop is closed"). Driving every judge call from the one persistent
    loop in `run()` keeps that client bound to a live loop for the whole run."""
    retryer = tenacity.AsyncRetrying(
        wait=_gemini_wait,
        stop=tenacity.stop_after_attempt(settings.judge_max_retries),
        retry=tenacity.retry_if_exception(_is_retryable),
        before_sleep=_log_retry,
        reraise=True,
    )
    return float(await retryer(metric.single_turn_ascore, sample))


def _coerce_faithfulness(raw: float, abstained: bool) -> float:
    """An abstention makes no factual claim about the domain, so it cannot be *unfaithful* —
    it is vacuously faithful (1.0), regardless of any meta-claim ragas extracts from the refusal
    text. Penalizing a correct 'I don't have enough information' would make the headline gate
    reward hallucination over honest refusal. (An abstention on an *answerable* question is still
    caught — as low answer_relevance -> generation_fail — just not miscounted as a hallucination.)
    A NaN on a substantive answer is treated conservatively as 0.0."""
    if abstained:
        return 1.0
    return 0.0 if math.isnan(raw) else raw


def _coerce(raw: float) -> float:
    return 0.0 if (raw is None or math.isnan(raw)) else raw


# --------------------------------------------------------------------------- run


def run(subset_n: int | None = None, resume: bool = True, gate_only: bool = False) -> EvalResult:
    """gate_only scores faithfulness alone — the one metric the CI gate reads. Answer relevance
    and context recall are ~2/3 of the judge's token spend, and the free-tier judge is capped at
    8K tokens/min, so a full-metric CI run throttles into timeouts. A gate-only run is never
    checkpointed or exported: its NaN relevance/recall must not reach a resumed full eval or the
    dashboard."""
    configure_logging()
    git_sha = _git_sha()
    run_id = uuid.uuid4().hex[:12]
    items = _subset(load_ground_truth(), subset_n)

    log.info(
        "eval.start", run_id=run_id, git_sha=git_sha, n=len(items), resume=resume,
        gate_only=gate_only,
    )
    warmup()  # load retrieval index + local models once, before the timed loop

    # The whole scoring loop runs in ONE event loop (see _score_async) so the judge client's
    # async transport stays bound to a live loop for the entire run.
    per_item = asyncio.run(_score_items(items, git_sha, run_id, resume, gate_only))

    result = _aggregate(run_id, git_sha, per_item)
    json_path = None if gate_only else store.save(result)
    _print_summary(result, json_path)
    return result


async def _score_one(
    gt: GroundTruthItem, faithfulness, answer_relevancy, context_recall, gate_only: bool = False
):
    """Retrieve -> generate -> score one item, through the same guardrails the service applies
    (context PII redaction, hardened prompt, output sanitising), so the gate measures what ships.
    Returns (answer, abstained, retrieved, faith, arel, crec); arel/crec are NaN when gate_only.
    Wrapped in a wall-clock timeout by the caller so a hang can't stall the run."""
    retrieved = hybrid_retrieve(gt.question)
    if settings.redact_context_pii:
        retrieved, _ = redact_chunks(retrieved)
    # Generation is sync (streams via the sync client); run it off the loop in a worker thread.
    # retry=False so _with_backoff owns the retry budget (no SDK amplification).
    answer, _citations, _gen_ms = await asyncio.to_thread(
        _with_backoff, partial(collect_answer, gt.question, retrieved, retry=False)
    )
    if settings.output_guard:
        answer, _ = guard_answer(answer, retrieved)
    abstained = is_abstention(answer)
    sample = SingleTurnSample(
        user_input=gt.question,
        response=answer,
        retrieved_contexts=[c.text for c in retrieved],
        reference=gt.reference_answer,
    )
    faith = _coerce_faithfulness(await _score_async(faithfulness, sample), abstained)
    if gate_only:
        return answer, abstained, retrieved, faith, math.nan, math.nan
    arel = _coerce(await _score_async(answer_relevancy, sample))
    crec = _coerce(await _score_async(context_recall, sample))
    return answer, abstained, retrieved, faith, arel, crec


async def _score_items(
    items: list[GroundTruthItem], git_sha: str, run_id: str, resume: bool, gate_only: bool = False
) -> list[EvalItemResult]:
    faithfulness, answer_relevancy, context_recall = _build_metrics()
    done = store.load_scored_items(git_sha) if resume and not gate_only else {}
    per_item: list[EvalItemResult] = []

    for i, gt in enumerate(items, start=1):
        if resume and gt.question in done:
            per_item.append(done[gt.question])
            log.info("eval.item.skip", i=i, of=len(items), question=gt.question[:70])
            continue

        try:
            # A whole-item wall-clock ceiling: a hung HTTP call (server stops streaming without
            # closing) has no other timeout on the generation path and would otherwise stall the
            # entire run indefinitely. wait_for turns that into a skip -> resume retries the item.
            answer, abstained, retrieved, faith, arel, crec = await asyncio.wait_for(
                _score_one(gt, faithfulness, answer_relevancy, context_recall, gate_only),
                timeout=settings.eval_item_timeout,
            )
        except Exception as e:
            if _is_daily_quota(e):
                # Every later item would hit the same wall; stop now with what's scored. The
                # CI gate then fails on coverage in minutes rather than timing out item by item.
                log.error("eval.quota_exhausted", i=i, scored=len(per_item), error=str(e)[:240])
                break
            # One item's unrecoverable failure (a never-parseable judge, a hang) must not sink the
            # run. Skip it — it is NOT checkpointed, so a later resume retries it (a fresh
            # stochastic call often succeeds). repr: a wait_for timeout's str() is empty.
            log.warning("eval.item.error", i=i, question=gt.question[:70], error=repr(e)[:180])
            continue

        item = EvalItemResult(
            question=gt.question,
            faithfulness=faith,
            answer_relevance=arel,
            context_recall=crec,
            generated_answer=answer,
            retrieved=retrieved,
        )
        # Attribution needs context recall, which a gate-only run doesn't score.
        if not gate_only:
            item.attribution = attribution.classify(item)
            store.upsert_item(git_sha, run_id, item)  # checkpoint before the next item
        per_item.append(item)

        log.info(
            "eval.item",
            i=i,
            of=len(items),
            faithfulness=round(faith, 3),
            answer_relevance=round(arel, 3),
            context_recall=round(crec, 3),
            attribution=item.attribution,
            abstained=abstained,
        )
        if i < len(items):
            await asyncio.sleep(settings.eval_item_pause_seconds)  # proactive RPM pacing

    return per_item


def _aggregate(run_id: str, git_sha: str, per_item: list[EvalItemResult]) -> EvalResult:
    n = max(len(per_item), 1)
    means = EvalMeans(
        faithfulness=sum(x.faithfulness for x in per_item) / n,
        answer_relevance=sum(x.answer_relevance for x in per_item) / n,
        context_recall=sum(x.context_recall for x in per_item) / n,
    )
    counts = AttributionCounts(
        retrieval_fail=sum(x.attribution == "retrieval_fail" for x in per_item),
        generation_fail=sum(x.attribution == "generation_fail" for x in per_item),
    )
    return EvalResult(
        run_id=run_id,
        git_sha=git_sha,
        timestamp=datetime.now(timezone.utc),
        per_item=per_item,
        means=means,
        attribution_counts=counts,
    )


# --------------------------------------------------------------------------- summary


def _print_summary(result: EvalResult, json_path) -> None:
    print("\n" + "=" * 78)
    print(f"  EVAL RUN {result.run_id}  @ {result.git_sha}  ({len(result.per_item)} items)")
    print("=" * 78)
    print(f"  {'faith':>6} {'a-rel':>6} {'c-rec':>6}  attribution   question")
    print("  " + "-" * 74)
    for x in result.per_item:
        print(
            f"  {_fmt(x.faithfulness)} {_fmt(x.answer_relevance)} {_fmt(x.context_recall)}  "
            f"{(x.attribution or ''):13} {x.question[:34]}"
        )
    print("  " + "-" * 74)
    m = result.means
    print(f"  {_fmt(m.faithfulness)} {_fmt(m.answer_relevance)} {_fmt(m.context_recall)}   MEANS")
    print()

    gate = settings.faithfulness_threshold
    verdict = "PASS" if m.faithfulness >= gate else "FAIL"
    print(f"  MEAN FAITHFULNESS: {m.faithfulness:.3f}   (gate >= {gate:.2f} -> {verdict})")
    print(
        f"  attribution: {result.attribution_counts.retrieval_fail} retrieval_fail, "
        f"{result.attribution_counts.generation_fail} generation_fail"
    )
    print(f"  exported: {json_path or '(gate-only run — not exported)'}")
    print("=" * 78 + "\n")


def _fmt(v: float) -> str:
    """6-wide score cell; a metric a gate-only run didn't score shows as a dash, not 'nan'."""
    return f"{'—':>6}" if math.isnan(v) else f"{v:6.2f}"


def _baseline_check(result: EvalResult) -> dict | None:
    """Compare this run with main's committed per-question baseline. None when there is no
    baseline; {"applicable": False, ...} when too few of its questions are in this run (the
    subset changed) — the gate then falls back to the absolute threshold alone."""
    if not BASELINE_PATH.exists():
        return None
    base = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    per_q = {q: statistics.mean(v) for q, v in base["per_question"].items()}
    shared = [x for x in result.per_item if x.question in per_q]
    if len(shared) < settings.ci_min_items:
        return {"applicable": False, "shared": len(shared), "baseline_sha": base["git_sha"]}
    base_mean = statistics.mean(per_q[x.question] for x in shared)
    run_mean = statistics.mean(x.faithfulness for x in shared)
    margin = max(
        settings.faithfulness_regression_margin,
        settings.faithfulness_regression_sd_k * base["run_mean_sd"],
    )
    deltas = sorted((x.faithfulness - per_q[x.question], x.question) for x in shared)
    return {
        "applicable": True,
        "baseline_sha": base["git_sha"],
        "baseline_runs": base["runs"],
        "shared": len(shared),
        "baseline_mean": round(base_mean, 4),
        "run_mean": round(run_mean, 4),
        "drop": round(base_mean - run_mean, 4),
        "margin": round(margin, 4),
        "passed": base_mean - run_mean <= margin,
        "worst": [(round(d, 3), q) for d, q in deltas[:5] if d < 0],
    }


def _ci_gate(result: EvalResult) -> bool:
    """The CI faithfulness gate (§13, FR-CI2/3). Returns True to pass, False to fail. Fails on
    insufficient coverage (too few items scored to trust the mean — throttle/skips), on mean
    faithfulness below the absolute threshold, OR on a drop below main's committed baseline larger
    than the noise-calibrated margin (_baseline_check). Writes a Markdown summary to
    $GITHUB_STEP_SUMMARY when run in GitHub Actions so it surfaces in the PR's checks (FR-CI3)."""
    m = result.means
    n = len(result.per_item)
    gate = settings.faithfulness_threshold
    enough = n >= settings.ci_min_items
    rel = _baseline_check(result) if enough else None
    rel_ok = rel is None or not rel["applicable"] or rel["passed"]
    passed = enough and m.faithfulness >= gate and rel_ok

    if not enough:
        reason = f"insufficient coverage — only {n}/{settings.ci_min_items} items scored"
    elif m.faithfulness < gate:
        reason = f"mean faithfulness {m.faithfulness:.3f} < {gate:.2f}"
    elif not rel_ok:
        reason = (f"mean faithfulness {rel['run_mean']:.3f} is {rel['drop']:.3f} below main's "
                  f"baseline {rel['baseline_mean']:.3f} (allowed drop {rel['margin']:.3f})")
    else:
        reason = f"mean faithfulness {m.faithfulness:.3f} >= {gate:.2f}"

    if rel is None:
        rel_rows = ["| vs main's baseline | no baseline committed — absolute gate only |"]
    elif not rel["applicable"]:
        rel_rows = [f"| vs main's baseline | only {rel['shared']} questions in common — "
                    "absolute gate only |"]
    else:
        rel_rows = [
            f"| vs main's baseline | {rel['run_mean']:.3f} vs {rel['baseline_mean']:.3f}: "
            f"drop {rel['drop']:+.3f}, allowed {rel['margin']:.3f} "
            f"({rel['baseline_runs']} runs @ `{rel['baseline_sha']}`) |",
        ]
    worst = [] if not rel or not rel.get("worst") else [
        "", "Largest per-question drops vs the baseline:", "",
        *[f"- {d:+.3f} — {q}" for d, q in rel["worst"]],
    ]

    md = "\n".join(
        [
            f"## Faithfulness gate — {'PASS ✅' if passed else 'FAIL ❌'}",
            "",
            f"**{reason}**",
            "",
            "| metric | value |",
            "|---|---|",
            f"| mean faithfulness | **{m.faithfulness:.3f}** (gate >= {gate:.2f}) |",
            *rel_rows,
            f"| items scored | {n} of {settings.ci_eval_subset_size} "
            f"(min {settings.ci_min_items}) |",
            *worst,
            "",
            "_Answer relevance, context recall and failure attribution are not scored in CI —"
            " the gate reads faithfulness only, and scoring all three would triple the judge's"
            " token spend per PR. The full eval runs locally and feeds the dashboard._",
            "",
            f"_judge `{settings.judge_model}` · generation `{settings.generation_model}` · "
            f"git `{result.git_sha}`_",
        ]
    )
    print("\n" + md + "\n")
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    return passed


def write_baseline(runs: int) -> None:
    """Score the CI subset `runs` times on the current code (faithfulness only) and commit-ready
    write main's per-question baseline for the relative gate. Run on main whenever its answer
    quality changes on purpose, and commit the file with that change."""
    per_q: dict[str, list[float]] = {}
    run_means = []
    for k in range(runs):
        res = run(subset_n=settings.ci_eval_subset_size, resume=False, gate_only=True)
        run_means.append(res.means.faithfulness)
        for x in res.per_item:
            per_q.setdefault(x.question, []).append(x.faithfulness)
        log.info("baseline.run", run=k + 1, of=runs, mean=round(res.means.faithfulness, 4))
    BASELINE_PATH.write_text(json.dumps({
        "git_sha": _git_sha(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "runs": runs,
        "subset_size": settings.ci_eval_subset_size,
        "judge_model": settings.judge_model,
        "generation_model": settings.generation_model,
        "run_means": [round(v, 4) for v in run_means],
        "run_mean_sd": round(statistics.stdev(run_means), 4) if runs > 1 else 0.0,
        "per_question": {q: [round(v, 4) for v in vs] for q, vs in per_q.items()},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {BASELINE_PATH}: run means {[round(v, 3) for v in run_means]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Ragas eval over the ground-truth set.")
    ap.add_argument("--subset", type=int, default=None, help="evaluate only N representative items")
    ap.add_argument(
        "--ci",
        action="store_true",
        help=f"CI gate mode: faithfulness only, over config.ci_eval_subset_size "
        f"({settings.ci_eval_subset_size}) items; never checkpointed or exported",
    )
    ap.add_argument("--no-resume", action="store_true", help="ignore checkpoints; re-score all")
    ap.add_argument("--write-baseline", type=int, metavar="RUNS", default=0,
                    help="score the CI subset RUNS times and write data/faithfulness_baseline.json")
    args = ap.parse_args()
    if args.write_baseline:
        write_baseline(args.write_baseline)
        return

    subset_n = settings.ci_eval_subset_size if args.ci else args.subset
    result = run(subset_n=subset_n, resume=not args.no_resume, gate_only=args.ci)

    if args.ci and not _ci_gate(result):
        raise SystemExit(1)  # the gate: a red build below threshold / on thin coverage (§13)


if __name__ == "__main__":
    main()
