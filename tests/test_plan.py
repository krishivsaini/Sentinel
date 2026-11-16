"""Multi-step questions: the planner's output is parsed defensively, split questions are retrieved
part by part on a shared rerank budget, and the service uses a plan only when it splits.

No model or network calls: the planner's reply is given as text or faked.
"""

from __future__ import annotations

import json

import pytest

from sentinel import plan as planner
from sentinel import retrieve
from sentinel.config import settings
from sentinel.eval import run_multistep
from sentinel.plan import Plan, parse_plan
from sentinel.schema import RetrievedChunk

Q = "What port does DoT use, and which media type does DoH use?"


def test_a_split_question_becomes_standalone_queries() -> None:
    p = parse_plan(Q, '{"queries": ["DNS over TLS default port", " DoH  media type "]}')
    assert p.queries == ["DNS over TLS default port", "DoH media type"] and p.multi and not p.error


def test_an_unsplit_question_is_searched_exactly_as_asked() -> None:
    p = parse_plan(Q, '{"queries": ["a rewording of the question"]}')
    assert p.queries == [Q] and not p.multi and p.error is None


@pytest.mark.parametrize("raw, error", [
    ("I think you should search for ports.", "unparseable"),
    ('{"queries": []}', "unparseable"),
    ('{"queries": "DoT port"}', "unparseable"),
    ('{"queries": ["a", 7]}', "unparseable"),
    ('{"queries": ["a", "b", "c", "d"]}', "out_of_bounds"),
    ('{"queries": ["a", "' + "x" * 400 + '"]}', "out_of_bounds"),
])
def test_unusable_plans_fall_back_to_the_question(raw: str, error: str) -> None:
    p = parse_plan(Q, raw)
    assert p.queries == [Q] and p.error == error


def test_duplicates_collapse_and_json_inside_prose_is_found() -> None:
    p = parse_plan(Q, 'Sure: {"queries": ["DoT port", "dot  port", "DoH media type"]} done')
    assert p.queries == ["DoT port", "DoH media type"]


def test_a_planner_failure_fails_open(monkeypatch) -> None:
    class Down:
        def invoke(self, _):
            raise TimeoutError("planner unreachable")
    monkeypatch.setattr(planner, "_planner", lambda: Down())
    p = planner.plan(Q)
    assert p.queries == [Q] and p.error == "TimeoutError"


def _chunk(cid: str) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, text=cid, rerank_score=1.0)


def test_split_retrieval_shares_the_rerank_pool_and_merges_round_robin(monkeypatch) -> None:
    calls = []
    ranked = {"a": ["a1", "shared", "a3", "a4"], "b": ["shared", "b2", "b3", "b4"],
              "c": ["c1", "c2", "c3", "c4"]}

    def fake(q, top_k, *, index_dir=None, access=None, pool_size=None):
        calls.append((q, top_k, pool_size))
        return [_chunk(c) for c in ranked[q][:top_k]], 2.0, 10.0

    monkeypatch.setattr(retrieve, "hybrid_retrieve_timed", fake)
    got, r_ms, k_ms = retrieve.multi_retrieve_timed(["a", "b", "c"], 6, pool_size=50)
    assert [q for q, _, _ in calls] == ["a", "b", "c"]
    assert {p for _, _, p in calls} == {17}                    # ceil(50 / 3): one budget, shared
    # Rank 1 of each part, then rank 2 of each, skipping the chunk part b already contributed.
    assert [c.chunk_id for c in got] == ["a1", "shared", "c1", "b2", "c2", "a3"]
    assert (r_ms, k_ms) == (6.0, 30.0)


def test_one_query_is_the_ordinary_pipeline(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(retrieve, "hybrid_retrieve_timed",
                        lambda q, top_k, **kw: seen.append((q, top_k, kw)) or ([], 0.0, 0.0))
    retrieve.multi_retrieve_timed([Q])
    assert seen == [(Q, None, {"index_dir": None, "access": retrieve.DEFAULT_ACCESS,
                               "pool_size": None})]


def test_the_evals_retrieve_the_way_the_service_does(monkeypatch) -> None:
    single, multi = [], []
    monkeypatch.setattr(retrieve, "hybrid_retrieve",
                        lambda q, **kw: single.append(q) or [_chunk("s")])
    monkeypatch.setattr(retrieve, "multi_retrieve_timed",
                        lambda qs, **kw: multi.append(qs) or ([_chunk("m")], 0.0, 0.0))
    monkeypatch.setattr(planner, "plan", lambda q: Plan(q, ["part one", "part two"]))
    assert planner.planned_retrieve(Q)[0].chunk_id == "s" and single == [Q]   # planning off
    monkeypatch.setattr(settings, "query_planning", True)
    assert planner.planned_retrieve(Q)[0].chunk_id == "m"
    assert multi == [["part one", "part two"]]


# --------------------------------------------------------------------------- the service

from tests.test_serve import FakeEngine, _auth_header, _client, _parse_sse  # noqa: E402


class PlanningEngine(FakeEngine):
    def __init__(self, queries: list[str]) -> None:
        super().__init__()
        self.queries, self.many = queries, []

    def plan(self, question: str) -> Plan:
        return Plan(question, self.queries, latency_ms=12.5)

    def retrieve_many(self, queries, top_k, access):
        self.many.append(queries)
        return self.retrieve(queries[0], top_k, access)


def _done(engine, monkeypatch) -> dict:
    monkeypatch.setattr(settings, "query_planning", True)
    client = _client(engine)
    r = client.post("/query", json={"question": Q}, headers=_auth_header(client))
    return json.loads(_parse_sse(r.text)[-1][1])


def test_the_service_retrieves_each_part_of_a_split_question(monkeypatch) -> None:
    engine = PlanningEngine(["DoT port", "DoH media type"])
    done = _done(engine, monkeypatch)
    assert engine.many == [["DoT port", "DoH media type"]]
    assert done["sub_questions"] == ["DoT port", "DoH media type"]
    assert done["latency_ms"]["plan"] == 12.5


def test_an_unsplit_question_takes_the_ordinary_path(monkeypatch) -> None:
    engine = PlanningEngine([Q])
    done = _done(engine, monkeypatch)
    assert engine.many == [] and len(engine.accesses) == 1 and done["sub_questions"] == []


# --------------------------------------------------------------------------- the eval set


def test_every_part_of_the_eval_set_has_its_evidence_in_the_index() -> None:
    items = run_multistep.load_items()
    assert len(items) == 16 and all(len(it["parts"]) >= 2 for it in items)


def test_moved_evidence_fails_loudly(tmp_path, monkeypatch) -> None:
    item = json.loads(run_multistep.QUESTIONS_PATH.open().readline())
    item["parts"][0]["evidence"] = "text that is not in any chunk"
    bad = tmp_path / "q.jsonl"
    bad.write_text(json.dumps(item) + "\n")
    monkeypatch.setattr(run_multistep, "QUESTIONS_PATH", bad)
    with pytest.raises(ValueError, match="evidence not in its gold chunks"):
        run_multistep.load_items()


def test_coverage_counts_a_part_once_any_gold_chunk_is_present() -> None:
    parts = [{"relevant": ["x#1", "x#2"]}, {"relevant": ["y#1"]}]
    assert run_multistep.coverage(["x#2", "z#9"], parts) == [True, False]
