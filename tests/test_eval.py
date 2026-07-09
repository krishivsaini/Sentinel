"""Eval-system tests (§12) — Phase 3, Day 9.

Hermetic tests (no API) cover the load-bearing, easy-to-regress logic of the headline eval:
  * failure attribution (retrieval vs generation) at the threshold boundaries,
  * SQLite checkpointing + resume (the free-tier survival mechanism),
  * the coercion + retry-delay-backoff helpers that keep a run honest and unstuck.

A live end-to-end run is opt-in (SENTINEL_RUN_LLM_TESTS=1) so pytest/CI stay cheap, deterministic,
and don't burn the Gemini free-tier quota.
"""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone

import pytest

from sentinel.config import settings
from sentinel.eval import attribution, run_eval, store
from sentinel.schema import (
    AttributionCounts,
    EvalItemResult,
    EvalMeans,
    EvalResult,
    RetrievedChunk,
)

_RUN_LLM = os.getenv("SENTINEL_RUN_LLM_TESTS") == "1"


def _item(
    question: str = "q?",
    faith: float = 1.0,
    arel: float = 1.0,
    crec: float = 1.0,
) -> EvalItemResult:
    return EvalItemResult(
        question=question,
        faithfulness=faith,
        answer_relevance=arel,
        context_recall=crec,
        generated_answer="an answer",
        retrieved=[RetrievedChunk(chunk_id="rfc1035#0021", text="body", rerank_score=6.7)],
    )


# --------------------------------------------------------------------------- attribution


def test_attribution_pass_when_all_healthy() -> None:
    assert attribution.classify(_item(faith=0.95, arel=0.9, crec=0.9)) == "pass"


def test_attribution_retrieval_fail_on_low_recall() -> None:
    """A failing item whose recall is below the retrieval threshold => the retriever's fault."""
    it = _item(faith=0.2, arel=0.3, crec=settings.retrieval_fail_recall_threshold - 0.01)
    assert attribution.classify(it) == "retrieval_fail"


def test_attribution_generation_fail_when_recalled_but_unfaithful() -> None:
    """Right docs retrieved (recall high) but the answer is unfaithful => the generator's fault."""
    it = _item(faith=0.1, arel=0.9, crec=1.0)
    assert attribution.classify(it) == "generation_fail"


def test_attribution_low_recall_is_retrieval_fail_even_with_good_answer() -> None:
    """A strong-looking answer over context that was never retrieved isn't grounded success —
    it's a retrieval failure (the grounding was luck), and recall takes priority."""
    it = _item(faith=0.95, arel=0.9, crec=0.1)
    assert attribution.classify(it) == "retrieval_fail"


# --------------------------------------------------------------------------- store: checkpoint + resume


@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Point the store at a throwaway DB + dashboard dir."""
    monkeypatch.setattr(store, "EVAL_DB_PATH", tmp_path / "eval.db")
    monkeypatch.setattr(store, "DASHBOARD_DATA_DIR", tmp_path / "dash")
    return tmp_path


def test_upsert_then_load_roundtrips(tmp_store) -> None:
    sha = "abc1234"
    it = _item(question="what is a CNAME?", faith=0.5, arel=0.6, crec=0.4)
    it.attribution = attribution.classify(it)
    store.upsert_item(sha, "run1", it)

    loaded = store.load_scored_items(sha)
    assert set(loaded) == {"what is a CNAME?"}
    got = loaded["what is a CNAME?"]
    assert (got.faithfulness, got.answer_relevance, got.context_recall) == (0.5, 0.6, 0.4)
    assert got.attribution == it.attribution
    # retrieved chunks (with their scores) survive the JSON round-trip
    assert got.retrieved[0].chunk_id == "rfc1035#0021"
    assert got.retrieved[0].rerank_score == 6.7


def test_upsert_is_idempotent_and_overwrites(tmp_store) -> None:
    """Re-scoring the same (git_sha, question) overwrites — a resumed run's fresh score wins,
    and the checkpoint never duplicates a row."""
    sha = "abc1234"
    store.upsert_item(sha, "run1", _item(question="q", faith=0.0))
    store.upsert_item(sha, "run2", _item(question="q", faith=1.0))
    loaded = store.load_scored_items(sha)
    assert len(loaded) == 1
    assert loaded["q"].faithfulness == 1.0


def test_load_scored_items_scoped_by_git_sha(tmp_store) -> None:
    """Checkpoints are per-SHA: a different commit starts with an empty slate (correctly forces
    a re-eval when the code/index changed)."""
    store.upsert_item("sha_a", "r", _item(question="q"))
    assert store.load_scored_items("sha_a")
    assert store.load_scored_items("sha_b") == {}


def test_save_writes_run_row_and_json_export(tmp_store) -> None:
    result = EvalResult(
        run_id="run_xyz",
        git_sha="deadbee",
        timestamp=datetime.now(timezone.utc),
        per_item=[_item(question="q1"), _item(question="q2", faith=0.1, arel=0.2, crec=0.1)],
        means=EvalMeans(faithfulness=0.55, answer_relevance=0.6, context_recall=0.55),
        attribution_counts=AttributionCounts(retrieval_fail=1, generation_fail=0),
    )
    result.per_item[1].attribution = "retrieval_fail"

    json_path = store.save(result)
    # per-run JSON + stable latest.json both written and re-validate as EvalResult
    assert json_path.exists() and json_path.name == "eval_run_xyz.json"
    latest = tmp_store / "dash" / "latest.json"
    assert latest.exists()
    EvalResult.model_validate_json(latest.read_text())
    # both items were persisted to the checkpoint table by save()
    assert set(store.load_scored_items("deadbee")) == {"q1", "q2"}


# --------------------------------------------------------------------------- run_eval helpers


def test_faithfulness_coercion_abstention_is_vacuously_faithful() -> None:
    # An abstention cannot be a hallucination, whatever ragas extracts from the refusal text.
    assert run_eval._coerce_faithfulness(0.0, abstained=True) == 1.0
    assert run_eval._coerce_faithfulness(math.nan, abstained=True) == 1.0
    # A substantive answer with no verifiable claim is conservatively 0.0.
    assert run_eval._coerce_faithfulness(math.nan, abstained=False) == 0.0
    assert run_eval._coerce_faithfulness(0.83, abstained=False) == 0.83


def test_metric_coercion_nan_to_zero() -> None:
    assert run_eval._coerce(math.nan) == 0.0
    assert run_eval._coerce(0.9) == 0.9


def test_is_transient_matches_rate_limit_signatures() -> None:
    assert run_eval._is_transient(Exception("429 RESOURCE_EXHAUSTED quota"))
    assert run_eval._is_transient(Exception("deadline exceeded / timeout"))
    assert not run_eval._is_transient(ValueError("bad prompt template"))


def test_server_retry_delay_parses_all_forms() -> None:
    assert run_eval._server_retry_delay(Exception("Please retry in 45.26s.")) == 45.26  # Gemini prose
    assert run_eval._server_retry_delay(Exception("Please try again in 2.4s")) == 2.4    # Groq prose
    assert run_eval._server_retry_delay(Exception("'retryDelay': '21s'")) == 21.0        # RetryInfo
    assert run_eval._server_retry_delay(Exception("no delay here")) is None


class _FakeRetryState:
    """Minimal tenacity RetryState stand-in for _gemini_wait."""

    def __init__(self, exc: Exception | None, attempt: int) -> None:
        self.attempt_number = attempt
        self.outcome = type("O", (), {"exception": lambda self: exc})()


def test_gemini_wait_honors_server_delay_with_cushion() -> None:
    wait = run_eval._gemini_wait(_FakeRetryState(Exception("retry in 20s"), attempt=1))
    assert wait == pytest.approx(21.0)  # server delay + 1s cushion


def test_gemini_wait_caps_long_delays() -> None:
    wait = run_eval._gemini_wait(_FakeRetryState(Exception("retry in 999s"), attempt=1))
    assert wait == settings.judge_backoff_max_seconds


def test_gemini_wait_falls_back_to_exponential() -> None:
    wait = run_eval._gemini_wait(_FakeRetryState(Exception("opaque error"), attempt=3))
    assert wait == pytest.approx(settings.judge_backoff_seconds * 4)  # 2^(3-1)


# --------------------------------------------------------------------------- subset selection


def _gt(n: int):
    from sentinel.schema import GroundTruthItem

    return [GroundTruthItem(question=f"q{i}", reference_answer="a") for i in range(n)]


def test_subset_none_returns_all() -> None:
    items = _gt(10)
    assert run_eval._subset(items, None) is items
    assert run_eval._subset(items, 99) is items  # n >= len -> all


def test_subset_is_deterministic_and_spread() -> None:
    items = _gt(48)
    a = run_eval._subset(items, 5)
    b = run_eval._subset(items, 5)
    assert [x.question for x in a] == [x.question for x in b]  # deterministic
    assert a[0].question == "q0" and a[-1].question == "q47"    # spans the full range
    assert len(a) == len(set(x.question for x in a))            # no duplicates


# --------------------------------------------------------------------------- opt-in live run


@pytest.mark.skipif(not _RUN_LLM, reason="set SENTINEL_RUN_LLM_TESTS=1 for a live Gemini run")
def test_live_run_single_item_produces_metrics() -> None:
    result = run_eval.run(subset_n=1, resume=False)
    assert len(result.per_item) == 1
    it = result.per_item[0]
    assert 0.0 <= it.faithfulness <= 1.0
    assert it.attribution in {"pass", "retrieval_fail", "generation_fail"}
