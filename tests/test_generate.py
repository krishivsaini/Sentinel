"""Generation tests (§10) — Phase 2, Day 5.

Hermetic tests (no API) cover the faithfulness-critical pure logic: citation parsing with
hallucinated-ID filtering, abstention detection, and grounded-prompt assembly. A live
end-to-end test is opt-in (SENTINEL_RUN_LLM_TESTS=1) so pytest/CI stay cheap and deterministic
and don't burn the Gemini free-tier quota.
"""

from __future__ import annotations

import os

import pytest

from sentinel.generate import (
    ABSTAIN_MESSAGE,
    build_messages,
    is_abstention,
    parse_citations,
)
from sentinel.schema import RetrievedChunk

_RUN_LLM = os.getenv("SENTINEL_RUN_LLM_TESTS") == "1"


def _chunks(*ids: str) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk_id=cid, text=f"body of {cid}") for cid in ids]


def test_parse_citations_drops_hallucinated_ids() -> None:
    """A cited ID that wasn't in the retrieved set can never become a citation (faithfulness)."""
    chunks = _chunks("rfc6585#0002")
    cites = parse_citations("429 means too many requests [rfc6585#0002] [rfc9999#0001].", chunks)
    assert [c.chunk_id for c in cites] == ["rfc6585#0002"]
    assert cites[0].doc_id == "rfc6585"


def test_parse_citations_dedups_in_first_seen_order() -> None:
    chunks = _chunks("rfc9110#0007", "rfc6749#0043")
    text = "a [rfc6749#0043] b [rfc9110#0007] c [rfc6749#0043]"
    assert [c.chunk_id for c in parse_citations(text, chunks)] == ["rfc6749#0043", "rfc9110#0007"]


def test_parse_citations_empty_when_none_cited() -> None:
    assert parse_citations("no markers here", _chunks("rfc6585#0002")) == []


def test_is_abstention() -> None:
    assert is_abstention(ABSTAIN_MESSAGE)
    assert is_abstention("  " + ABSTAIN_MESSAGE.upper() + "  ")
    assert not is_abstention("The 429 status code means too many requests [rfc6585#0002].")


def test_build_messages_grounds_on_context_and_ids() -> None:
    chunks = _chunks("rfc9110#0007")
    system, human = build_messages("What is 429?", chunks)
    assert "ONLY" in system.content and ABSTAIN_MESSAGE in system.content
    assert "rfc9110#0007" in human.content       # the citable ID is in the context block
    assert "body of rfc9110#0007" in human.content
    assert "What is 429?" in human.content


@pytest.mark.skipif(not _RUN_LLM, reason="set SENTINEL_RUN_LLM_TESTS=1 to run live Gemini tests")
def test_live_grounded_answer_is_cited_and_grounded() -> None:
    from sentinel.generate import collect_answer
    from sentinel.retrieve import retrieve

    q = "Which HTTP status code indicates the client has sent too many requests?"
    chunks = retrieve(q)
    text, cites, latency_ms = collect_answer(q, chunks)

    assert not is_abstention(text)
    assert cites, "a grounded answer over relevant context should cite at least one chunk"
    valid = {c.chunk_id for c in chunks}
    assert all(c.chunk_id in valid for c in cites), "no citation outside the retrieved set"
    assert latency_ms > 0


@pytest.mark.skipif(not _RUN_LLM, reason="set SENTINEL_RUN_LLM_TESTS=1 to run live Gemini tests")
def test_live_abstains_on_out_of_corpus_question() -> None:
    from sentinel.generate import collect_answer
    from sentinel.retrieve import retrieve

    q = "What is the average annual rainfall in the Amazon rainforest?"
    text, cites, _ = collect_answer(q, retrieve(q))
    assert is_abstention(text)
    assert cites == []
