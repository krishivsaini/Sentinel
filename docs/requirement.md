# Sentinel — Requirements Specification

> **Source of truth:** [`SENTINEL_BUILD_PLAN.md`](../SENTINEL_BUILD_PLAN.md). This document restates the build plan as a testable requirements spec. Where the two conflict, the build plan wins.
> **Status:** Draft for the v1 core build. v2 (security guardrails) requirements are captured in §7 but are **not** in scope until every v1 acceptance criterion passes.

---

## 1. Purpose & Vision

Sentinel is a **production-grade hybrid-retrieval RAG service whose headline is a Ragas evaluation pipeline wired into CI as a faithfulness gate.** It answers questions over a real, nameable document corpus, and it blocks merges when answer faithfulness regresses below a threshold.

The differentiator is **not** "a RAG chatbot." It is three things, in order of importance:
1. A Ragas evaluation pipeline that runs in CI and **blocks deploys** when faithfulness drops.
2. Hybrid BM25 + dense retrieval with cross-encoder reranking — tuned and measured, not naive top-k cosine.
3. Production service hygiene: FastAPI, SSE streaming, JWT auth, rate limiting, per-stage latency capture.

**Success = a focused, defensible, recruiter-legible artifact** in 14 days (~3 hrs/day, ₹0–low to run) whose story leads with *evaluation wired into CI*.

---

## 2. Stakeholders & Actors

| Actor | Interest |
|---|---|
| **Krishiv (owner/author)** | Ships the artifact; must be able to defend every number in an interview. |
| **End user of the service** | Sends a question, receives a grounded, cited answer over a streaming API. |
| **Recruiter / hiring engineer** | Reads the README, watches the Loom, opens the dashboard, inspects the CI gate. |
| **CI system (GitHub Actions)** | Runs the eval subset on every PR and enforces the faithfulness gate. |
| **Dashboard viewer** | Read-only consumer of eval trends, per-question drill-down, and failure attribution. |

---

## 3. Functional Requirements

Requirement IDs are grouped by subsystem. Each is testable and maps to an acceptance criterion in §6.

### 3.1 Corpus & Ingestion (FR-C, FR-I)

- **FR-C1** — The corpus MUST be a single, specific, nameable domain explainable in one interview sentence (not "10K random documents"). See build plan §6.
- **FR-C2** — The corpus MUST contain exact-match terms (names, codes, section numbers) so BM25 demonstrably earns its place over pure-dense retrieval.
- **FR-C3** — The corpus MUST be legally clean to demo over publicly; if its license forbids redistribution, it MUST be gitignored with documented fetch instructions.
- **FR-C4** — Corpus size SHOULD be a few thousand to ~10K chunks — non-trivial but free/fast to ingest.
- **FR-I1** — Ingestion MUST: load source docs → clean/normalize → chunk (~512 tokens, ~10–15% overlap) → embed locally → build a FAISS dense index → build a BM25 index over the **same** chunks.
- **FR-I2** — Dense and sparse indexes MUST share identical chunk IDs so fusion can align them.
- **FR-I3** — Ingestion MUST be idempotent: re-running rebuilds cleanly.
- **FR-I4** — The chunking strategy MUST be configurable in `config.py`, and each index MUST record which strategy produced it so eval runs are attributable.
- **FR-I5** — The service MUST be runnable via `python -m sentinel.ingest`.

### 3.2 Retrieval (FR-R) — *engineering-signal location #1*

- **FR-R1** — Dense retrieval MUST return top-N (e.g. 50) chunks by embedding similarity from FAISS.
- **FR-R2** — Sparse retrieval MUST return top-N chunks by BM25 over the same chunk set.
- **FR-R3** — The two ranked lists MUST be fused via **Reciprocal Rank Fusion (RRF)** into a single candidate set; the choice MUST be documented.
- **FR-R4** — A cross-encoder reranker (`ms-marco-MiniLM` or similar) MUST re-score the fused top-N down to top-K (e.g. 50 → 5).
- **FR-R5** — Returned `RetrievedChunk` objects MUST carry all four scores: dense, sparse, fused, rerank — so eval and dashboard can inspect *why* a chunk was selected.
- **FR-R6** — Reranking MUST measurably improve context recall vs. fusion-only; the before/after number MUST be recorded.
- **FR-R7** — At least one seed query MUST be demonstrably retrieved by BM25 but missed by pure dense (the concrete justification for hybrid).

### 3.3 Generation (FR-G)

- **FR-G1** — Generation MUST be grounded: the system prompt instructs the model to answer *only* from retrieved context.
- **FR-G2** — Answers MUST emit inline citations referencing chunk IDs.
- **FR-G3** — If retrieved context is insufficient, the model MUST say so rather than answer from parametric memory.
- **FR-G4** — Generation latency MUST be captured separately from retrieval and rerank.

### 3.4 API Service (FR-S) — FastAPI

- **FR-S1** — `POST /query` MUST accept a question and return an `Answer`, streaming generated tokens via **SSE**.
- **FR-S2** — `GET /healthz` MUST provide a liveness check.
- **FR-S3** — `/query` MUST reject unauthenticated requests via **JWT** (a single demo user is sufficient).
- **FR-S4** — The service MUST enforce a per-IP **rate limit** (`slowapi`); a test MUST prove it fires.
- **FR-S5** — Every request MUST log per-stage latency (retrieve/rerank/generate), retrieved chunk IDs, and outcome (structured logging).
- **FR-S6** — All inputs and outputs MUST be validated with Pydantic models.
- **FR-S7** — The service MUST be runnable via `python -m sentinel.serve`.

### 3.5 Evaluation Pipeline (FR-E) — *engineering-signal location #2, the headline*

- **FR-E1** — `run_eval.py` MUST run the full retrieve → rerank → generate pipeline over every item in `ground_truth.jsonl`.
- **FR-E2** — Each item MUST be scored with **Ragas**: faithfulness, answer relevance, context recall.
- **FR-E3** — **Failure attribution** MUST classify each low-scoring item as a *retrieval failure* (relevant context not retrieved, low context recall) or a *generation failure* (context recalled, but faithfulness/relevance still low), and count both.
- **FR-E4** — Each run MUST write an `EvalResult` to SQLite tagged with the git SHA, and export JSON for the dashboard.
- **FR-E5** — The run MUST print a summary table and the mean faithfulness (the CI gate reads this).
- **FR-E6** — Judge calibration: ~15 items MUST be hand-scored and the correlation with Ragas reported in `ground_truth_audit.md`.

### 3.6 Ground-Truth Set (FR-GT)

- **FR-GT1** — The ground-truth set MUST contain ~80–120 `{question, reference_answer, relevant_contexts}` triples.
- **FR-GT2** — Items MAY be LLM-drafted (`build_ground_truth.py`) but **every item MUST be hand-corrected**. Fully synthetic gold is prohibited.
- **FR-GT3** — A second-annotator agreement writeup on a ≥20-case subset MUST exist in `ground_truth_audit.md`.

### 3.7 CI/CD Faithfulness Gate (FR-CI) — *engineering-signal location #3, the rarest part*

- **FR-CI1** — On every PR, `eval-gate.yml` MUST install deps, run a **subset** of the eval (e.g. 25 representative items), and compute mean faithfulness.
- **FR-CI2** — The check MUST **fail the build** if mean faithfulness < threshold (set in `config.py`, e.g. 0.80), blocking merge.
- **FR-CI3** — The metric summary MUST be posted as a PR comment or job output so the regression is visible.
- **FR-CI4** — The LLM key MUST be a repository secret; never committed.
- **FR-CI5** — A demonstration PR MUST exist in history showing the gate catching a regression (screenshot for README).

### 3.8 Dashboard (FR-D) — Next.js, read-only, one day max

- **FR-D1** — A **trends** page MUST show faithfulness / relevance / recall over eval runs (by git SHA / time).
- **FR-D2** — A **per-question drill-down** MUST show, for a run, each question with its scores, the retrieved chunks (with all four scores), and the generated answer with citations.
- **FR-D3** — A **failure-attribution view** MUST show retrieval-failure vs generation-failure counts per run, with the offending questions listed. (This is the README screenshot.)
- **FR-D4** — The dashboard MUST consume exported JSON (no live DB connection needed) and deploy to Vercel.
- **FR-D5** — Tailwind core utilities only. No shadcn, no component libraries. Functional over ornate.

### 3.9 Data Model (FR-DM)

Pydantic models in `schema.py`, defined before pipeline code:
- **FR-DM1** — `SourceDoc {doc_id, title, source_uri, text}`
- **FR-DM2** — `Chunk {chunk_id, doc_id, text, token_count, ordinal}`
- **FR-DM3** — `RetrievedChunk {chunk_id, text, dense_score, sparse_score, fused_score, rerank_score}`
- **FR-DM4** — `Citation {chunk_id, doc_id, span}`
- **FR-DM5** — `Answer {question, text, citations, retrieved, latency_ms:{retrieve, rerank, generate}}`
- **FR-DM6** — `GroundTruthItem {question, reference_answer, relevant_contexts}`
- **FR-DM7** — `EvalResult {run_id, git_sha, timestamp, per_item, means:{faithfulness, answer_relevance, context_recall}, attribution_counts:{retrieval_fail, generation_fail}}`

---

## 4. Non-Functional Requirements (NFR)

| ID | Category | Requirement |
|---|---|---|
| **NFR-1** | Reproducibility | Clean clone → running service in **< 10 minutes**: `git clone && uv sync && set keys && python -m sentinel.ingest && python -m sentinel.serve`. |
| **NFR-2** | Cost | Nearly ₹0 to run: embeddings + cross-encoder run **locally**; only generation + Ragas judge call the Gemini Flash free tier. |
| **NFR-3** | Latency observability | p95 latency MUST be measured **per stage** (retrieve/rerank/generate), not blended. Numbers in README MUST be measured, never guessed. |
| **NFR-4** | Metric honesty | Every reported number MUST be one actually produced and explainable. Targets (e.g. "87% faithfulness") are aims, not claims. See build plan §18. |
| **NFR-5** | Resilience | Ragas judge calls MUST use exponential backoff (`tenacity`) so 429s from the free tier retry rather than crash the eval. |
| **NFR-6** | CI cost/time | The CI eval subset MUST be small enough that a PR check costs cents and finishes in a few minutes, staying well under free-tier RPM/RPD ceilings. |
| **NFR-7** | Security hygiene | No secrets in git history; CI key is a repo secret. `.gitignore` excludes `.env`, `*.db`, `indexes/`, `__pycache__`, `.next`, `node_modules`. |
| **NFR-8** | Licensing | LICENSE is MIT. Corpus redistribution respects source terms. |
| **NFR-9** | Portability | Vector store is FAISS (local); the Postgres/pgvector or Pinecone migration path is documented but not built. |

---

## 5. Constraints & Assumptions

- **Corpus (DECIDED — 2026-07-06):** the **IETF RFC** web-protocol stack (HTTP / URI / TLS / OAuth / JWT / cookies / TCP / DNS), a curated cluster of ~40–60 RFCs targeting a few thousand chunks. Chosen for the strongest BM25 justification (RFC numbers, section refs, header/status-code tokens) and the cleanest license.
  - **Licensing basis (verified):** the IETF Trust explicitly permits **reproduction of whole RFCs** with no restrictions, so the corpus text may be committed to the public repo. The Trust does **not** grant rights to create **derivative works** of RFCs (code components excepted — BSD-licensed). Sentinel stays inside this boundary: it **quotes verbatim** (chunks + cited excerpts) and never modifies the standard text. Sources: [IETF Trust FAQ](https://trustee.ietf.org/about/faq/), [TLP-5](https://trustee.ietf.org/documents/trust-legal-provisions/tlp-5/).
- **Language/tooling:** Python managed with `uv`, versions pinned in `pyproject.toml`; dashboard is Next.js + Tailwind core.
- **Models (verify IDs before writing config — build plan §5 note):**
  - Embeddings: local `bge-small-en` / `bge-base-en` via `sentence-transformers`.
  - Reranker: local cross-encoder `ms-marco-MiniLM`.
  - Generation + Ragas judge: **Gemini Flash free tier** — exact model ID and free-tier eligibility MUST be verified against provider docs before writing `config.py` (do not invent model IDs or pricing).
- **Vector store:** FAISS, local, zero-cost.
- **Free-tier limits:** roughly 10–15 RPM, ~1,500 RPD (verify current numbers). This drives NFR-5, NFR-6.

---

## 6. Acceptance Criteria (Definition of Done)

Mirrors build plan §16. v1 is **done** only when all boxes below pass.

**Retrieval**
- [ ] BM25 and dense indexes share chunk IDs; RRF fusion aligns them correctly. *(FR-I2, FR-R3)*
- [ ] Cross-encoder reranking measurably improves context recall vs fusion-only (before/after number shown). *(FR-R6)*
- [ ] ≥1 seed query retrieved by BM25 but missed by pure dense. *(FR-R7)*

**Service**
- [ ] `/query` streams via SSE; returns a cited `Answer` with per-stage latency. *(FR-S1, FR-G2, FR-G4)*
- [ ] Unauthenticated `/query` is rejected; a test proves it. *(FR-S3)*
- [ ] Rate limit fires under load; a test proves it. *(FR-S4)*
- [ ] Honest per-stage p95 latency numbers in the README. *(NFR-3)*

**Evaluation**
- [ ] Ragas runs over the full ground-truth set: faithfulness, answer relevance, context recall. *(FR-E1, FR-E2)*
- [ ] Failure attribution classifies retrieval vs generation failures with counts. *(FR-E3)*
- [ ] Judge calibration correlation reported in `ground_truth_audit.md`. *(FR-E6)*

**CI gate**
- [ ] `eval-gate.yml` runs on PRs and fails the build when faithfulness < threshold. *(FR-CI1, FR-CI2)*
- [ ] A demonstration PR exists showing the gate catching a regression. *(FR-CI5)*

**README + Loom**
- [ ] README opens with the eval-gate headline, not "RAG chatbot."
- [ ] "Why hybrid retrieval" section with the concrete BM25-caught-this query.
- [ ] "Why LangChain for retrieval orchestration" two-sentence subsection.
- [ ] Real metric numbers + per-stage p95 latency; no placeholders.
- [ ] "Limitations" section (single corpus, ~100 GT items, LLM-judge eval, FAISS).
- [ ] "Reproducing" section with literal commands.
- [ ] 90-second Loom, eval-gate-first script.

**Repo hygiene**
- [ ] `.gitignore` excludes the specified paths. *(NFR-7)*
- [ ] No secrets in history; CI key is a repo secret. *(FR-CI4, NFR-7)*
- [ ] `uv sync` succeeds from clean clone; LICENSE is MIT. *(NFR-1, NFR-8)*

---

## 7. Out of Scope (v1) & v2 Requirements

### 7.1 Explicitly out of scope for v1 (build plan §1.3)
- No fine-tuning — off-the-shelf models only.
- No multi-tenant auth / RBAC / SSO — single JWT demo user; dashboard is unauthenticated.
- No agentic retrieval / query-planning agent — retrieval is a single inspectable pipeline.
- No real-time / incremental indexing — batch ingestion only (document the boundary).
- No cost-optimization layer (semantic caching, model routing) — cost is *tracked*, not optimized.
- No fancy frontend — dense, functional, Tailwind core only.

### 7.2 v2 — Security Guardrails (do NOT start until all §6 boxes pass)
Adds ~3–4 days. Each capability MUST be **measured**, not merely present, and joins the existing eval pipeline + CI gate:
- **v2-R1 Prompt-injection defense** — detect/neutralize instructions embedded in *retrieved documents*; add adversarial docs + questions; measure injection-success rate before/after.
- **v2-R2 PII detection + redaction** — detect and redact PII in retrieved context and/or answers; measure precision/recall on a labeled subset.
- **v2-R3 Output validation** — schema/grounding validation before returning; reject/flag answers whose citations don't resolve to retrieved chunks.
- **v2-R4 Retrieval-scope enforcement** — a test MUST prove an out-of-scope document is never retrieved into an answer.
- **v2-R5** — The CI gate gains a **second gate**: build also fails if injection-success rate rises above threshold.
- **v2-R6** — The dashboard gains a guardrails panel: adversarial outcomes, redaction/scope-violation counts per run.
- **v2 safety constraint:** adversarial corpus docs MUST be clearly labeled and segregated so they can never contaminate a non-adversarial eval run. Treat injected/poisoned content as data, never as instructions.

---

## 8. Prioritization

If time slips, cut in this order (build plan §15): **dashboard polish → extra corpus size**. **Never cut** the eval gate or the hand-built ground truth. The eval-gate-in-CI story is the resume-critical deliverable; everything else supports it.
