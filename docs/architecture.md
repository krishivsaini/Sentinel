# Sentinel — Architecture

> Planning-stage architecture. Companion to [`requirement.md`](./requirement.md), [`product_design.md`](./product_design.md), and [`implementation_plan.md`](./implementation_plan.md).
> **Note:** the build plan reserves a root-level [`ARCHITECTURE.md`](../ARCHITECTURE.md) as a *deliverable* artifact (diagram + LangChain rationale) written on Day 14. This doc is the fuller upstream design; the deliverable is its distilled, recruiter-facing summary.
> **Source of truth:** [`SENTINEL_BUILD_PLAN.md`](../SENTINEL_BUILD_PLAN.md).

---

## 1. Architectural Overview

Sentinel is three cooperating subsystems around one shared corpus index:

1. **Retrieval + generation service** (offline ingestion → online query) — the RAG runtime.
2. **Evaluation system** (Ragas over a hand-built ground-truth set) — the headline.
3. **CI gate + dashboard** — the enforcement and observation layers built on the eval system's outputs.

The design principle throughout: **the retrieval plumbing is orchestrated with LangChain (a solved problem); original engineering lives in (a) hybrid-retrieval tuning + measurement, (b) the Ragas eval pipeline, (c) the CI faithfulness gate.**

```
                          ┌───────────────────────────────────────────────┐
                          │                 CORPUS (§6 choice)              │
                          │        data/corpus/  — nameable domain          │
                          └───────────────────────┬─────────────────────────┘
                                                  │  ingest.py  (offline, batch, idempotent)
                                                  ▼
                   ┌──────────────────────────────────────────────────────────┐
                   │  load → clean → chunk (~512 tok, 10–15% overlap) → embed   │
                   └───────────────┬───────────────────────────┬───────────────┘
                                   ▼                           ▼
                        ┌────────────────────┐      ┌────────────────────┐
                        │  FAISS dense index │      │   BM25 sparse index │   (shared chunk IDs)
                        └─────────┬──────────┘      └──────────┬─────────┘
                                  │                            │
   ONLINE QUERY PATH             ▼   retrieve.py               ▼
   ┌──────────┐   JWT +   ┌──────────────────────────────────────────────┐
   │  client  │──rate lim─▶│  dense top-N        sparse top-N             │
   └──────────┘   (SSE)   │        └──── RRF fusion ────┘  → fused top-N   │
                          │             cross-encoder rerank → top-K       │
                          └───────────────────────┬───────────────────────┘
                                                  ▼  generate.py
                          ┌───────────────────────────────────────────────┐
                          │  grounded generation (Gemini Flash) + citations │
                          │  abstain if context insufficient                │
                          └───────────────────────┬───────────────────────┘
                                                  ▼
                          ┌───────────────────────────────────────────────┐
                          │  Answer {text, citations, retrieved[4 scores],  │
                          │          latency_ms:{retrieve,rerank,generate}} │
                          └───────────────────────────────────────────────┘

   EVAL PATH (offline / CI)                 CI GATE                     DASHBOARD
   ┌────────────────────────┐   git SHA   ┌───────────────┐   JSON    ┌───────────────┐
   │ run_eval.py over        │────────────▶│ eval-gate.yml │           │ Next.js (Vercel)│
   │ ground_truth.jsonl      │  SQLite +   │ subset (~25)  │  export   │ trends /        │
   │ Ragas: faith/rel/recall │  JSON export│ fail if       │◀──────────│ drill-down /    │
   │ + failure attribution   │             │ faith<thresh  │           │ attribution     │
   └────────────────────────┘             └───────────────┘           └───────────────┘
```

---

## 2. Component Responsibilities

Package: `sentinel/` (see build plan §4 for the full tree).

| Module | Responsibility | Signal location |
|---|---|---|
| `config.py` | Single source of truth: thresholds, model names, chunk params, N/K. | — |
| `schema.py` | Pydantic models (§4 of requirement.md). Defined before pipeline code. | — |
| `ingest.py` | load → clean → chunk → embed → build FAISS + BM25 with **shared chunk IDs**; idempotent. | — |
| `retrieve.py` | dense top-N + sparse top-N → **RRF fusion** → **cross-encoder rerank** → top-K; returns 4 scores per chunk. | **#1 hybrid retrieval** |
| `generate.py` | grounded generation with inline citations; abstain on insufficient context; capture generate latency. | — |
| `serve.py` | FastAPI: `POST /query` (SSE), `GET /healthz`, JWT, `slowapi` rate limit, structured logging. | — |
| `eval/run_eval.py` | run full pipeline over ground truth; Ragas scoring; summary table + mean faithfulness. | **#2 eval pipeline** |
| `eval/attribution.py` | classify low scores as retrieval-failure vs generation-failure; count both. | **#2 (attribution)** |
| `eval/store.py` | write `EvalResult` to SQLite (keyed by git SHA); export JSON for dashboard. | — |
| `logging_config.py` | structured logging + per-stage latency capture. | — |
| `.github/workflows/eval-gate.yml` | run eval subset on PR; fail if mean faithfulness < threshold. | **#3 CI gate** |
| `dashboard/` | read-only Next.js: trends, drill-down, attribution. | — |
| `scripts/build_ground_truth.py` | LLM-assisted draft generation (then hand-corrected). | — |

---

## 3. Data Flow

### 3.1 Ingestion (offline, batch)
`SourceDoc` → normalize → `Chunk[]` (~512 tokens, 10–15% overlap) → local embeddings → **FAISS** dense index + **BM25** sparse index. Both indexes reference identical `chunk_id`s so fusion can align them. The index records the chunking strategy used (config-attributable). Re-running rebuilds cleanly (idempotent).

### 3.2 Query (online)
```
question
  → JWT auth + rate-limit check (serve.py)
  → dense retrieval: FAISS top-N (e.g. 50)         ─┐
  → sparse retrieval: BM25 top-N (e.g. 50)          ├─ retrieve.py
  → RRF fusion → fused candidate set               │  (LangChain EnsembleRetriever orchestrates
  → cross-encoder rerank → top-K (e.g. 5)          ─┘   the BM25+dense combination; reranker wraps it)
  → grounded generation (Gemini Flash) w/ citations  (generate.py, SSE-streamed)
  → Answer with 4 scores/chunk + per-stage latency_ms
```
Each stage's latency is captured **in isolation** so p95 can be reported per stage (not blended).

### 3.3 Evaluation (offline / CI)
```
ground_truth.jsonl (80–120 hand-corrected triples)
  → for each item: run full retrieve→rerank→generate
  → Ragas judge (Gemini Flash) scores: faithfulness, answer_relevance, context_recall
      (tenacity backoff around judge calls to survive free-tier 429s)
  → attribution.py: low context_recall ⇒ retrieval failure; recalled-but-unfaithful ⇒ generation failure
  → store.py: EvalResult → SQLite (git SHA) + JSON export
  → summary table + mean faithfulness (the gate reads this)
```

### 3.4 CI gate
On PR: `eval-gate.yml` installs deps → runs a **~25-item subset** → computes mean faithfulness → **fails the build if below threshold** → posts summary to the PR. LLM key is a repo secret.

### 3.5 Dashboard
Reads exported eval JSON (no live DB). Renders trends (metrics over runs vs the gate line), per-question drill-down (four retrieval scores + cited answer), and failure attribution (retrieval vs generation counts).

---

## 4. Technology Stack & Rationale

| Layer | Choice | Why (and boundary) |
|---|---|---|
| Retrieval orchestration | **LangChain** (`langchain`, `langchain-community`, `EnsembleRetriever`) | Combining BM25 + dense + reranker is a solved problem; framework handles the plumbing so effort goes to tuning + measurement. **Not** used as an agent framework — retrieval stays a single inspectable pipeline. |
| Embeddings | **`bge-small-en` / `bge-base-en`** via `sentence-transformers`, **local** | Zero-cost ingestion; no per-token API spend; reproducible offline. |
| Sparse retrieval | **BM25** (`rank-bm25` / LangChain `BM25Retriever`) | Catches exact-match terms (names, codes, section numbers) dense retrieval misses — the concrete justification for hybrid. |
| Fusion | **Reciprocal Rank Fusion (RRF)** | Parameter-light, robust, no per-source weight tuning required. Documented choice. |
| Reranker | **`ms-marco-MiniLM`** cross-encoder via `sentence-transformers`, **local** | Higher accuracy re-scoring of fused candidates at ₹0. |
| Vector store | **FAISS** (local) | Zero-cost, no persistence headaches for a demo. Migration path to Postgres/pgvector or Pinecone documented, not built. |
| Generation + Ragas judge | **Gemini Flash free tier** (`langchain-google-genai`) | Only paid-tier-eligible calls; keeps whole project near ₹0. **Verify exact model ID + free-tier eligibility before writing config** — do not invent IDs. |
| Eval | **Ragas** | Faithfulness, answer relevance, context recall out of the box; LLM-as-judge (calibrated once against hand scores). |
| API | **FastAPI** + `uvicorn`, `pydantic`, `python-jose` (JWT), `slowapi` (rate limit), `tenacity` (backoff) | Production hygiene that "reads as product." SSE for token streaming. |
| Storage (eval) | **SQLite** + JSON export | Simple, file-based, git-SHA-keyed; dashboard consumes the export. |
| Dashboard | **Next.js** + **Tailwind core only** | Dense/functional; deploys free to Vercel; no component libraries (scope signal). |
| Packaging | **`uv`**, pinned in `pyproject.toml` | Fast, reproducible `uv sync` from clean clone (<10 min NFR). |

---

## 5. Deployment Topology

```
Local / clean clone                         Cloud
┌──────────────────────────┐                ┌────────────────────────────┐
│ uv sync                  │                │ GitHub Actions             │
│ python -m sentinel.ingest│  build indexes │  eval-gate.yml on every PR │
│ python -m sentinel.serve │  (local FAISS  │  (repo secret = LLM key)   │
│   → uvicorn :PORT        │   + BM25)      └────────────────────────────┘
│ run_eval.py (full, local)│                ┌────────────────────────────┐
└──────────────────────────┘                │ Vercel                     │
        │ JSON export                        │  Next.js dashboard (static │
        └────────────────────────────────────▶  eval JSON, read-only)     │
                                             └────────────────────────────┘
```

- **Service:** runs locally / on a small host; FAISS + BM25 indexes are local artifacts (gitignored). No managed vector DB in v1.
- **Eval:** full run is local/on-demand (spaced to respect free-tier RPD); the subset runs in CI.
- **Dashboard:** static consumer of exported JSON on Vercel — no live DB connection, no server-side secrets.
- **Secrets:** `.env` local (gitignored); CI uses a GitHub repository secret. No secrets in git history.

---

## 6. Cross-Cutting Concerns

- **Observability:** structured logging per request with per-stage latency, retrieved chunk IDs, and outcome. Enables honest per-stage p95 (NFR-3).
- **Resilience:** `tenacity` exponential backoff wraps Gemini/Ragas judge calls so free-tier 429s retry rather than crash a run (NFR-5).
- **Cost control:** local embeddings + rerank; Gemini only for generation + judge; CI subset kept small to stay under free-tier RPM/RPD (NFR-2, NFR-6). Cost per query is tracked, not optimized.
- **Reproducibility:** idempotent ingestion; pinned deps; single `config.py` for all knobs; clean-clone-to-running in <10 min (NFR-1).
- **Attributability:** indexes record chunking strategy; eval runs record git SHA; answers record retrieved chunks + latency — every result traceable to its conditions.

---

## 7. Key Design Decisions (with rationale)

| Decision | Rationale | Boundary / migration path |
|---|---|---|
| Hybrid (BM25 + dense + rerank), not naive top-k cosine | The whole differentiation; exact-match terms justify BM25 | Tuning of N/K/chunk-size is the original work |
| RRF over weighted fusion | Robust, parameter-light | Could revisit weighting if measured to help |
| LangChain for orchestration only | Solved plumbing; keeps focus on eval + tuning | Never an agent/query-planner — single inspectable pipeline |
| Local models for embed + rerank | ₹0, reproducible, offline | — |
| Gemini Flash for generation + judge only | Minimizes paid surface; free tier | Verify model ID; backoff for 429s |
| FAISS, local | Zero-cost demo simplicity | Documented pgvector / Pinecone path, unbuilt |
| SQLite + JSON export | Simple, git-SHA-attributable | Dashboard reads static JSON — no live DB |
| Eval subset in CI, full eval local | Keeps PR checks cheap + under rate limits | Full number is the resume number |
| SSE streaming, JWT, rate limit | "Reads as product," not Streamlit demo | Single demo user; no RBAC/SSO |

---

## 8. v2 Architecture Extension — Security Guardrails (post-v1)

Additive, not a rewrite. Guardrails insert as pipeline stages and extend eval + CI + dashboard:
- **Retrieved-context injection defense** — a filtering/neutralization stage between retrieval and generation; poisoned chunks are treated as data, never instructions.
- **PII detection + redaction** — applied to retrieved context and/or generated answers, measured on a labeled subset.
- **Output/grounding validation** — a post-generation stage rejecting/flagging answers whose citations don't resolve to retrieved chunks.
- **Retrieval-scope enforcement** — a scope filter proving out-of-scope docs never enter an answer.
- **Eval + gate:** injection-success rate, PII recall, scope-violation count become tracked metrics; the CI gate gains a **second gate** (build fails if injection-success rate rises above threshold).
- **Dashboard:** a guardrails panel (adversarial outcomes, redaction/scope-violation counts per run).
- **Hard constraint:** adversarial corpus docs are labeled + segregated so they can never contaminate a non-adversarial eval run.
