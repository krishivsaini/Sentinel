"""Generation layer (§10) — Phase 2, Day 5.

Grounded generation: the system prompt instructs the model to answer ONLY from retrieved
context and to emit inline citations referencing chunk IDs (FR-G1, FR-G2). If context is
insufficient, the model must say so rather than answer from parametric memory (FR-G3) —
this directly protects faithfulness. Generation latency is captured separately (FR-G4).

Uses Gemini (config.generation_model) via langchain-google-genai; streamed token-by-token
for the SSE endpoint.
"""

from __future__ import annotations

from collections.abc import Iterator

from sentinel.schema import RetrievedChunk


def generate(question: str, chunks: list[RetrievedChunk]) -> Iterator[str]:
    # TODO(Phase 2, Day 5): grounded, cited, abstain-on-insufficient-context generation.
    raise NotImplementedError("generate.py is a Phase 2 (Day 5) stub — not yet implemented.")
