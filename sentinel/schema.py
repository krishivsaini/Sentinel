"""Pydantic data contracts for the whole pipeline (§7).

Defined before any pipeline code so every stage speaks the same types:
ingest -> Chunk, retrieve -> RetrievedChunk, generate -> Answer, eval -> EvalResult.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- Corpus / ingest


class SourceDoc(BaseModel):
    """A raw source document before chunking (e.g. one RFC)."""

    doc_id: str
    title: str
    source_uri: str
    text: str


class Chunk(BaseModel):
    """A unit of source text indexed for retrieval. chunk_id is shared by the FAISS and
    BM25 indexes so fusion can align them (§8, FR-I2)."""

    chunk_id: str
    doc_id: str
    text: str
    token_count: int
    ordinal: int  # position of this chunk within its source doc


# --------------------------------------------------------------------------- Retrieval


class RetrievedChunk(BaseModel):
    """A candidate chunk carrying every score that selected it, so eval and the dashboard
    can inspect *why* it was chosen (§9, FR-R5). Scores from a stage a chunk didn't reach
    are None (e.g. dense_score None => chunk came only from BM25)."""

    chunk_id: str
    text: str
    dense_score: float | None = None
    sparse_score: float | None = None
    fused_score: float | None = None   # RRF
    rerank_score: float | None = None  # cross-encoder


# --------------------------------------------------------------------------- Generation


class Citation(BaseModel):
    """A grounding pointer from the answer back to a retrieved chunk (§10)."""

    chunk_id: str
    doc_id: str
    span: tuple[int, int] | None = None  # char offsets into the chunk text, if resolvable


class LatencyMs(BaseModel):
    """Per-stage latency, captured in isolation so p95 is reportable per stage (§1.2.6)."""

    retrieve: float
    rerank: float
    generate: float


class Answer(BaseModel):
    """The core value unit returned by /query."""

    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    latency_ms: LatencyMs


# --------------------------------------------------------------------------- API I/O


class QueryRequest(BaseModel):
    """Input to POST /query."""

    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = None  # optional override of config.rerank_top_k


class TokenResponse(BaseModel):
    """JWT issued by the token endpoint."""

    access_token: str
    token_type: str = "bearer"


# --------------------------------------------------------------------------- Ground truth / eval


class GroundTruthItem(BaseModel):
    """A hand-corrected evaluation triple (§15 day 8, FR-GT). Never fully synthetic."""

    question: str
    reference_answer: str
    relevant_contexts: list[str] = Field(default_factory=list)


class EvalMeans(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_recall: float


class AttributionCounts(BaseModel):
    """Failure attribution tallies for a run (§12, FR-E3)."""

    retrieval_fail: int = 0
    generation_fail: int = 0


class EvalItemResult(BaseModel):
    """Per-question eval detail — powers the dashboard drill-down (§14)."""

    question: str
    faithfulness: float
    answer_relevance: float
    context_recall: float
    # "pass" | "retrieval_fail" | "generation_fail" — None until attribution runs.
    attribution: str | None = None
    generated_answer: str
    retrieved: list[RetrievedChunk] = Field(default_factory=list)


class EvalResult(BaseModel):
    """One eval run, keyed by git SHA; persisted to SQLite and exported to JSON (§12)."""

    run_id: str
    git_sha: str
    timestamp: datetime
    per_item: list[EvalItemResult] = Field(default_factory=list)
    means: EvalMeans
    attribution_counts: AttributionCounts
