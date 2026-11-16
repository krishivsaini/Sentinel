"""sentinel.ops: the operations report built from the service's own log events."""

from __future__ import annotations

import json

import pytest

from sentinel.ops import latency_summary, summarise


def _query(aid, outcome, request_ms, cost, day="2026-11-16", **lat):
    return {"event": "query", "answer_id": aid, "outcome": outcome, "question": f"secret {aid}?",
            "answer": "text", "latency_ms": {"retrieve": 10, "rerank": 900, **lat},
            "first_token_ms": request_ms - 5, "request_ms": request_ms, "cost_usd": cost,
            "timestamp": f"{day}T10:00:00Z"}


EVENTS = [
    _query("a" * 32, "answered", 1000, 0.002),
    _query("b" * 32, "abstained", 2000, 0.001),
    _query("c" * 32, "withheld", 3000, 0.003, day="2026-11-17"),
    {"event": "query.refused", "answer_id": "d" * 32, "question": "secret d?", "request_ms": 300,
     "cost_usd": 0.0005, "timestamp": "2026-11-17T11:00:00Z"},
    {"event": "query.budget_exhausted", "timestamp": "2026-11-17T12:00:00Z"},
    {"event": "feedback", "answer_id": "c" * 32, "rating": "up",
     "timestamp": "2026-11-17T12:01:00Z"},
    {"event": "feedback", "answer_id": "c" * 32, "rating": "down", "comment": "secret comment",
     "timestamp": "2026-11-17T12:02:00Z"},
    {"event": "feedback", "answer_id": "d" * 32, "rating": "down",
     "timestamp": "2026-11-17T12:03:00Z"},
]


def test_outcomes_cost_and_feedback_are_counted() -> None:
    r = summarise(EVENTS, "test")
    assert r["requests"] == 5
    assert {k: v["n"] for k, v in r["outcomes"].items()} == {
        "abstained": 1, "answered": 1, "budget_exhausted": 1, "refused": 1, "withheld": 1}
    assert r["cost_usd"]["total"] == pytest.approx(0.0065)
    assert r["cost_usd"]["requests_with_cost"] == 4
    # c's reader changed their mind (latest wins); d thumbed down a refusal
    assert r["feedback"] == {"up": 0, "down": 2,
                             "by_outcome": {"refused": {"down": 1}, "withheld": {"down": 1}}}
    assert [d["date"] for d in r["daily"]] == ["2026-11-16", "2026-11-17"]
    assert r["daily"][1]["requests"] == 3


def test_latency_is_per_stage_and_end_to_end() -> None:
    stages = {s["stage"]: s for s in summarise(EVENTS, "test")["latency"]}
    assert list(stages) == ["retrieve", "rerank", "first_token", "request"]
    assert stages["request"]["n"] == 3 and stages["request"]["p50_ms"] == 2000
    assert stages["rerank"]["p95_ms"] == 900


def test_the_report_holds_no_reader_text() -> None:
    text = json.dumps(summarise(EVENTS, "test"))
    assert "secret" not in text


def test_percentiles_interpolate_like_the_latency_benchmark() -> None:
    s = latency_summary("x", [100, 200, 300, 400])
    assert (s["p50_ms"], s["p95_ms"], s["max_ms"]) == (250.0, 385.0, 400.0)
    assert latency_summary("x", [])["n"] == 0
