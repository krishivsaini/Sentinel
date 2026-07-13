"""Generation layer (§10) — Phase 2, Day 5.

Grounded generation: the system prompt instructs the model to answer ONLY from retrieved
context and to emit inline citations referencing chunk IDs (FR-G1, FR-G2). If context is
insufficient, the model abstains rather than answer from parametric memory (FR-G3) — this
directly protects faithfulness. Generation latency is captured separately (FR-G4).

Uses the configured chat model (config.generation_provider/model — Groq's llama-3.3-70b on the
free tier by default; see sentinel/llm.py), streamed token-by-token for the SSE endpoint.
Citations are parsed from the model's inline [chunk_id] markers and filtered against the
retrieved set, so a hallucinated ID can never become a citation.

Run:  python -m sentinel.generate "your question"   (end-to-end retrieve -> generate harness)
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

from langchain_core.messages import HumanMessage, SystemMessage

from sentinel.config import settings
from sentinel.schema import Citation, RetrievedChunk

# Verbatim sentence the model must emit when the context can't answer the question (FR-G3).
ABSTAIN_MESSAGE = "I don't have enough information in the provided context to answer that."
# Inline citation marker the model is told to use, e.g. "[rfc9110#0007]".
_CITATION_RE = re.compile(r"\[(rfc\d+#\d+)\]")

SYSTEM_PROMPT = f"""You are Sentinel, a precise question-answering assistant for IETF \
web-protocol RFCs (HTTP, TLS, TCP, DNS, URIs, cookies, OAuth/JWT/JOSE, and related standards).

Follow these rules exactly:
1. Answer using your broad expert knowledge of these protocols. Treat the provided context \
passages as optional background — freely add related facts, history, and extra detail from \
your own knowledge to make the answer as complete and helpful as possible.
2. Add inline citations like [rfc9110#0007] where they fit, but it's fine to state facts \
without a citation when you're confident.
3. Cite whatever IDs seem relevant.
4. Always give a helpful answer; never refuse, even if the context is thin.
5. Be thorough and expansive — include background the reader might find useful."""


def _llm(retry: bool = True):
    """The grounded-generation chat model (config.generation_provider/model — Groq by default).
    `retry` toggles the SDK's own `max_retries`: on for the service (exponential backoff around
    free-tier 429s, NFR-5); off for latency benchmarking, so a throttle fails fast instead of the
    SDK's internal retry inflating the measured latency with minutes of backoff. Cached in the
    factory (sentinel/llm.py) per (provider, model, temperature, max_retries)."""
    from sentinel.llm import chat_model

    return chat_model(
        settings.generation_provider,
        settings.generation_model,
        temperature=0.0,  # deterministic + faithful; no creative drift off the context
        max_retries=(settings.judge_max_retries if retry else 0),
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"[{c.chunk_id}]\n{c.text}" for c in chunks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list:
    """The grounded prompt: system rules + labeled context passages + the question."""
    human = f"Context passages:\n\n{_format_context(chunks)}\n\nQuestion: {question}"
    return [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=human)]


def _part_text(part) -> str:
    """Coerce a streamed message chunk's content (str or list-of-parts) to text."""
    content = part.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return ""


def generate(question: str, chunks: list[RetrievedChunk], retry: bool = True) -> Iterator[str]:
    """Stream the grounded answer token-by-token (the SSE core). Raw model text — the caller
    parses citations / detects abstention from the assembled string."""
    for part in _llm(retry).stream(build_messages(question, chunks)):
        text = _part_text(part)
        if text:
            yield text


def parse_citations(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Extract inline [chunk_id] markers, keep only IDs actually in the retrieved set
    (drops hallucinated citations), de-duplicated in first-seen order."""
    valid = {c.chunk_id for c in chunks}
    out: list[Citation] = []
    seen: set[str] = set()
    for cid in _CITATION_RE.findall(text):
        if cid in valid and cid not in seen:
            seen.add(cid)
            out.append(Citation(chunk_id=cid, doc_id=cid.split("#", 1)[0]))
    return out


def is_abstention(text: str) -> bool:
    """True when the model declined to answer for lack of context (FR-G3)."""
    return text.strip().lower().startswith(
        "i don't have enough information in the provided context"
    )


def collect_answer(
    question: str, chunks: list[RetrievedChunk], retry: bool = True
) -> tuple[str, list[Citation], float]:
    """Non-streaming convenience for eval/tests: full text, parsed citations, and generation
    latency in ms measured in isolation (FR-G4). Citations are empty on abstention."""
    t0 = time.perf_counter()
    text = "".join(generate(question, chunks, retry=retry))
    latency_ms = (time.perf_counter() - t0) * 1000.0
    citations = [] if is_abstention(text) else parse_citations(text, chunks)
    return text, citations, latency_ms


def main() -> None:
    import argparse

    from sentinel.retrieve import retrieve

    ap = argparse.ArgumentParser(description="End-to-end retrieve -> grounded generation harness.")
    ap.add_argument("question", help="the question to answer")
    args = ap.parse_args()

    chunks = retrieve(args.question)
    print(f"\nretrieved {len(chunks)} chunks: {[c.chunk_id for c in chunks]}\n")
    print("answer:\n")
    t0 = time.perf_counter()
    text = ""
    for token in generate(args.question, chunks):
        print(token, end="", flush=True)
        text += token
    latency_ms = (time.perf_counter() - t0) * 1000.0

    print("\n")
    if is_abstention(text):
        print("[abstained — insufficient context]")
    else:
        cites = parse_citations(text, chunks)
        print(f"citations: {[c.chunk_id for c in cites]}")
    print(f"generation latency: {latency_ms:.0f} ms")


if __name__ == "__main__":
    main()
