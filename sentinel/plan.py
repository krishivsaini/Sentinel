"""Multi-step questions: split a question that asks about several things into search queries.

One search finds passages for one topic. "How does HTTP/3 map request streams, and how does
QUIC close a stream early?" needs passages from two RFCs, and a single query's top 5 tends to
fill up from whichever half matches best. The planner is one small model call: it reads the
question and returns 1 to MAX_QUERIES standalone search queries. Retrieval then searches and
reranks for each query (retrieve.multi_retrieve_timed) and merges the results, and the answer is
generated from the merged passages against the ORIGINAL question, with the usual rules: answer
only from the passages, cite every claim, abstain when they don't cover it.

Design choices:
  * A question the planner doesn't split takes exactly the single-question path: same query,
    same pool, same top 5. Planning can only change multi-part questions.
  * It fails open. A planner error, malformed output or too many queries falls back to the
    question as asked, which is what the service did before planning existed. The planner's
    queries only steer retrieval inside the caller's scopes; they never reach the generator as
    instructions, so a manipulated plan can at worst retrieve worse passages.
  * It runs after the input guard, so it only ever sees a question that passed screening.
    The injection gate was run with planning on and passed (0 / 50 poisoned-document attacks,
    0 / 35 attacks in the question with the input guard).
  * The rerank budget is shared, not multiplied: each of n queries reranks a pool of
    ceil(pool / n) candidates (data/rerank_pool_sweep.json: a pool of 20 keeps hit@5 and nearly
    all recall@5), so the cross-encoder, the dominant cost, scores about as many pairs as for one
    question.

Cost: one gpt-oss-20b call per question (~$0.0001 at Groq's prices), metered like every call.

Run:  python -m sentinel.plan "How do HTTP/2 and HTTP/3 each prioritise streams?"
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage, SystemMessage

from sentinel.config import settings

MAX_QUERIES = 3
MAX_QUERY_CHARS = 300

PLANNER_PROMPT = f"""You plan searches over a collection of IETF RFCs (HTTP, TLS, QUIC, TCP, \
DNS, URIs, cookies, OAuth, JOSE and related standards). You do not answer questions.

Read the user's question and decide whether one search can find what it needs.
- If the question asks about ONE thing, even in detail, return it unchanged as a single query.
- If it asks about SEVERAL distinct things, such as two protocols to compare, two mechanisms, \
or a fact and a follow-up about a different part of a standard, return one standalone search \
query per thing, at most {MAX_QUERIES}. Each query must make sense on its own: repeat the \
protocol or RFC name instead of writing "it" or "the second one".
- Keep the exact technical terms from the question: header names, status codes, error codes, \
frame types, RFC numbers.
- Never add topics the question doesn't ask about.

Reply with JSON only, no other text: {{"queries": ["...", "..."]}}"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Plan:
    """What retrieval should search for. `queries == [question]` means a single search."""

    question: str
    queries: list[str]
    error: str | None = None        # why the planner's output wasn't used, if it wasn't
    latency_ms: float = 0.0
    raw: str = field(default="", repr=False)

    @property
    def multi(self) -> bool:
        return len(self.queries) > 1


def parse_plan(question: str, raw: str) -> Plan:
    """The queries in the planner's reply, or the question itself when the reply isn't usable."""
    match = _JSON_RE.search(raw or "")
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    queries = data.get("queries") if isinstance(data, dict) else None
    if not isinstance(queries, list) or not queries:
        return Plan(question, [question], error="unparseable", raw=raw)
    cleaned, seen = [], set()
    for q in queries:
        if not isinstance(q, str):
            return Plan(question, [question], error="unparseable", raw=raw)
        q = " ".join(q.split())
        if q and q.lower() not in seen:
            seen.add(q.lower())
            cleaned.append(q)
    if not cleaned or len(cleaned) > MAX_QUERIES or any(len(q) > MAX_QUERY_CHARS for q in cleaned):
        return Plan(question, [question], error="out_of_bounds", raw=raw)
    if len(cleaned) == 1:
        # Not split: search with the question exactly as asked, never a rewording of it.
        return Plan(question, [question], raw=raw)
    return Plan(question, cleaned, raw=raw)


def _planner():
    from sentinel.llm import chat_model

    return chat_model(settings.generation_provider, settings.planner_model, temperature=0.0,
                      max_retries=settings.judge_max_retries)


def plan(question: str) -> Plan:
    """Ask the planner how to search for `question`. Never raises: on any failure the plan is
    the question itself (a single search), with the reason in `error`."""
    t0 = time.perf_counter()
    try:
        reply = _planner().invoke([SystemMessage(content=PLANNER_PROMPT),
                                   HumanMessage(content=question)])
        raw = reply.content if isinstance(reply.content, str) else str(reply.content)
        result = parse_plan(question, raw)
    except Exception as exc:  # noqa: BLE001 - fail open to the single-search path
        result = Plan(question, [question], error=f"{type(exc).__name__}")
    result.latency_ms = (time.perf_counter() - t0) * 1000.0
    return result


def planned_retrieve(question: str, *, index_dir=None, access=None) -> list:
    """The passages the service would hand the generator for `question`: planned when
    config.query_planning is on, otherwise the single search. The evals call this so they
    measure what ships."""
    from sentinel.retrieve import DEFAULT_ACCESS, hybrid_retrieve, multi_retrieve_timed

    access = access or DEFAULT_ACCESS
    if not settings.query_planning:
        return hybrid_retrieve(question, index_dir=index_dir, access=access)
    chunks, _, _ = multi_retrieve_timed(plan(question).queries, index_dir=index_dir,
                                        access=access)
    return chunks


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("question")
    args = ap.parse_args()
    p = plan(args.question)
    print(json.dumps({"queries": p.queries, "error": p.error,
                      "latency_ms": round(p.latency_ms)}, indent=1))


if __name__ == "__main__":
    main()
