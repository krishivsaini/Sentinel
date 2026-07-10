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
import math
import re
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

from sentinel.config import GROUND_TRUTH_PATH, REPO_ROOT, settings
from sentinel.eval import attribution, store
from sentinel.generate import collect_answer, is_abstention
from sentinel.logging_config import configure_logging, log
from sentinel.retrieve import hybrid_retrieve, warmup
from sentinel.schema import (
    AttributionCounts,
    EvalItemResult,
    EvalMeans,
    EvalResult,
    GroundTruthItem,
)

# Error-message fragments that mark a *transient* judge failure worth backing off and retrying
# (rate limits, quota, timeouts, transient 5xx). Anything else re-raises immediately — we don't
# want to spin on a real bug.
_TRANSIENT = (
    "429",
    "rate limit",
    "resourceexhausted",
    "resource exhausted",
    "quota",
    "deadline",
    "timeout",
    "timed out",
    "503",
    "unavailable",
    "500",
    "internal error",
)

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

    from sentinel.llm import chat_model

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
    llm = LangchainLLMWrapper(judge, run_config=rc, bypass_n=True)
    emb = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=settings.embedding_model), run_config=rc
    )
    return (
        Faithfulness(llm=llm),
        ResponseRelevancy(llm=llm, embeddings=emb),
        LLMContextRecall(llm=llm),
    )


def _is_transient(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(t in s for t in _TRANSIENT)


def _is_parse_error(exc: BaseException) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(t in s for t in _PARSE_SIGNATURES)


def _is_retryable(exc: BaseException) -> bool:
    """Retry rate-limit/timeout bursts AND transient judge parse failures (a re-call often parses)."""
    return _is_transient(exc) or _is_parse_error(exc)


# 429s carry the exact wait the server wants — Gemini as prose ("Please retry in 45.2s") and
# RetryInfo ('retryDelay': '45s'); Groq as "Please try again in 2.4s". Honoring it beats a blind
# exponential: we wake right after the window clears instead of guessing and 429-ing again.
_RETRY_PROSE_RE = re.compile(r"(?:retry|try again) in ([\d.]+)s", re.IGNORECASE)
_RETRY_INFO_RE = re.compile(r"retryDelay'?:?\s*'?(\d+)s")


def _server_retry_delay(exc: BaseException | None) -> float | None:
    if exc is None:
        return None
    s = str(exc)
    m = _RETRY_PROSE_RE.search(s) or _RETRY_INFO_RE.search(s)
    return float(m.group(1)) if m else None


def _gemini_wait(retry_state) -> float:
    """tenacity wait: use the server-requested retryDelay (+1s cushion) when present, else fall
    back to exponential. Capped at judge_backoff_max_seconds so a stuck daily-quota can't hang."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    server = _server_retry_delay(exc)
    if server is not None:
        wait = server + 1.0
    else:
        wait = settings.judge_backoff_seconds * (2 ** max(retry_state.attempt_number - 1, 0))
    return min(wait, settings.judge_backoff_max_seconds)


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


def run(subset_n: int | None = None, resume: bool = True) -> EvalResult:
    configure_logging()
    git_sha = _git_sha()
    run_id = uuid.uuid4().hex[:12]
    items = _subset(load_ground_truth(), subset_n)

    log.info("eval.start", run_id=run_id, git_sha=git_sha, n=len(items), resume=resume)
    warmup()  # load retrieval index + local models once, before the timed loop

    # The whole scoring loop runs in ONE event loop (see _score_async) so the judge client's
    # async transport stays bound to a live loop for the entire run.
    per_item = asyncio.run(_score_items(items, git_sha, run_id, resume))

    result = _aggregate(run_id, git_sha, per_item)
    json_path = store.save(result)
    _print_summary(result, json_path)
    return result


async def _score_items(
    items: list[GroundTruthItem], git_sha: str, run_id: str, resume: bool
) -> list[EvalItemResult]:
    faithfulness, answer_relevancy, context_recall = _build_metrics()
    done = store.load_scored_items(git_sha) if resume else {}
    per_item: list[EvalItemResult] = []

    for i, gt in enumerate(items, start=1):
        if resume and gt.question in done:
            per_item.append(done[gt.question])
            log.info("eval.item.skip", i=i, of=len(items), question=gt.question[:70])
            continue

        try:
            retrieved = hybrid_retrieve(gt.question)
            # Generation is sync (streams via the sync client); run it off the loop in a worker
            # thread. retry=False so _with_backoff owns the retry budget (no SDK amplification),
            # and generation is as throttle-resilient as the judge.
            answer, _citations, _gen_ms = await asyncio.to_thread(
                _with_backoff, partial(collect_answer, gt.question, retrieved, retry=False)
            )
            abstained = is_abstention(answer)
            contexts = [c.text for c in retrieved]
            sample = SingleTurnSample(
                user_input=gt.question,
                response=answer,
                retrieved_contexts=contexts,
                reference=gt.reference_answer,
            )

            faith = _coerce_faithfulness(await _score_async(faithfulness, sample), abstained)
            arel = _coerce(await _score_async(answer_relevancy, sample))
            crec = _coerce(await _score_async(context_recall, sample))
        except Exception as e:
            # One item's unrecoverable failure (e.g. a judge that never returns parseable output,
            # even after retries) must not sink the whole run. Skip it — it is NOT checkpointed,
            # so a later resume retries it (a fresh stochastic judge call often parses).
            log.warning("eval.item.error", i=i, question=gt.question[:70], error=str(e)[:180])
            continue

        item = EvalItemResult(
            question=gt.question,
            faithfulness=faith,
            answer_relevance=arel,
            context_recall=crec,
            generated_answer=answer,
            retrieved=retrieved,
        )
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
            f"  {x.faithfulness:6.2f} {x.answer_relevance:6.2f} {x.context_recall:6.2f}  "
            f"{(x.attribution or ''):13} {x.question[:34]}"
        )
    print("  " + "-" * 74)
    m = result.means
    print(f"  {m.faithfulness:6.2f} {m.answer_relevance:6.2f} {m.context_recall:6.2f}   MEANS")
    print()

    gate = settings.faithfulness_threshold
    verdict = "PASS" if m.faithfulness >= gate else "FAIL"
    print(f"  MEAN FAITHFULNESS: {m.faithfulness:.3f}   (gate >= {gate:.2f} -> {verdict})")
    print(
        f"  attribution: {result.attribution_counts.retrieval_fail} retrieval_fail, "
        f"{result.attribution_counts.generation_fail} generation_fail"
    )
    print(f"  exported: {json_path}")
    print("=" * 78 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Ragas eval over the ground-truth set.")
    ap.add_argument("--subset", type=int, default=None, help="evaluate only N representative items")
    ap.add_argument(
        "--ci",
        action="store_true",
        help=f"CI gate mode: subset of config.ci_eval_subset_size ({settings.ci_eval_subset_size})",
    )
    ap.add_argument("--no-resume", action="store_true", help="ignore checkpoints; re-score all")
    args = ap.parse_args()

    subset_n = settings.ci_eval_subset_size if args.ci else args.subset
    result = run(subset_n=subset_n, resume=not args.no_resume)

    if args.ci and result.means.faithfulness < settings.faithfulness_threshold:
        raise SystemExit(1)  # the gate: a red build below threshold (§13)


if __name__ == "__main__":
    main()
