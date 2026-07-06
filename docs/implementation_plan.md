# Sentinel — Implementation Plan

> Companion to [`requirement.md`](./requirement.md), [`product_design.md`](./product_design.md), and [`architecture.md`](./architecture.md).
> **Source of truth:** [`SENTINEL_BUILD_PLAN.md`](../SENTINEL_BUILD_PLAN.md). This plan sequences the build into phases and maps them to the build plan's 14-day schedule.
> **Timeline:** 14 days × ~3 hrs/day. **Golden rule for cuts:** if time slips, cut dashboard polish and extra corpus size — **never** the eval gate or the hand-built ground truth.

---

## 1. Phasing Overview

The build plan's 14 days group into six phases. Each phase has a hard exit gate; do not advance until it passes.

| Phase | Days | Theme | Exit gate |
|---|---|---|---|
| **0 — Foundation** | 1 | Corpus decision, scaffold, contracts | Corpus chosen + licensed; repo skeleton pushed; `config.py` + `schema.py` exist |
| **1 — Retrieval core** | 2–4 | Ingestion + hybrid retrieval | Hybrid retrieval returns reranked top-K; `test_retrieve.py` green |
| **2 — Generation + service** | 5–7 | Grounded answers + FastAPI hygiene | `/query` SSE cited answers; auth + rate-limit tests green; honest per-stage latency |
| **3 — Eval system** | 8–10 | Ground truth + Ragas + attribution | Full Ragas metrics; attribution counts; judge calibration reported |
| **4 — The gate** | 11 | CI faithfulness gate | `eval-gate.yml` blocks a deliberately-bad PR |
| **5 — Surface + ship** | 12–14 | Dashboard, deploy, narrative | Public dashboard URL; clean-clone <10 min; all §19 artifacts exist |

---

## 2. Phase 0 — Foundation (Day 1)

**Goal:** make every load-bearing decision before writing pipeline code.

**Tasks**
1. **Corpus choice + licensing** — ✅ **DONE (2026-07-06): IETF RFC web-protocol stack, 48 curated RFCs** fetched verbatim into `data/corpus/`. Exact list + titles + thematic clusters are pinned in [`data/corpus_manifest.json`](../data/corpus_manifest.json); `scripts/fetch_corpus.py` downloads idempotently from rfc-editor.org and records per-file sha256/bytes in `data/corpus/PROVENANCE.json`. Coverage: HTTP core 9110–9114, HTTP compression 7541/9204, HTTP extensions (5789/6585/6797/8288/8615), URIs 3986/6570, cookies 6265, WebSocket 6455, QUIC 9000–9002, TCP 9293, TLS 8446/7301/6066/2818, PKI 5280/6960/8555, OAuth 6749/6750/7636/8628/9068, HTTP auth 7617/7616, JOSE/JWT 7515–7519, encodings 8259/4648/3339, DNS 1034/1035/8484/7858, keywords 2119/8174. Obsoleted HTTP RFCs (2616/7231) deliberately excluded so ground truth stays unambiguous against a single current spec. Size ~5.3 MB / ~2.0–2.4K chunks — inside the FR-C4 window. Licensing verified: IETF Trust permits whole-RFC reproduction; no derivative works — pipeline quotes verbatim only (see [`requirement.md` §5](./requirement.md)).
2. Scaffold the repo tree (build plan §4): `sentinel/`, `data/`, `dashboard/`, `tests/`, `scripts/`, `.github/workflows/`.
3. `pyproject.toml` via `uv` with pinned deps. **Verify current package names + exact Gemini Flash model ID against provider docs first** (build plan §5 note) — do not invent IDs.
4. `.gitignore` (`.env`, `*.db`, `indexes/`, `__pycache__`, `.next`, `node_modules`, `dashboard/data/*.json`), `.env.example`, MIT `LICENSE` (already present).
5. `config.py` — thresholds (faithfulness gate e.g. 0.80), model names, chunk params, N/K. Single source of truth.
6. `schema.py` — all Pydantic models (requirement.md §3.9 / FR-DM).

**Exit gate:** corpus justified in one sentence; repo skeleton pushed; `config.py` + `schema.py` compile.

**Decision resolved:** corpus = IETF RFC web-protocol stack, 48 RFCs, fetched + verified + provenance-recorded (licensing verified). Phase 0 corpus work is complete; next up is Day 2 ingestion.

---

## 3. Phase 1 — Retrieval Core (Days 2–4)

**Goal:** hybrid retrieval that measurably beats naive vector search. *(Engineering-signal #1.)*

**Day 2 — Ingestion (`ingest.py`)**
- load → clean/normalize → chunk (~512 tok, 10–15% overlap) → local embeddings → FAISS dense index → BM25 index over the **same** chunks.
- Enforce **shared chunk IDs** across both indexes (FR-I2). Make ingestion idempotent (FR-I3). Record chunking strategy on the index (FR-I4).
- **Exit:** `python -m sentinel.ingest` builds both indexes cleanly.

**Day 3 — Independent retrieval + seed tests (`retrieve.py`, `test_retrieve.py`)**
- Dense top-N (FAISS) and sparse top-N (BM25) working independently.
- Seed-query tests: retrieval returns expected chunks.
- **Capture the hybrid-justifying query now:** find a seed query BM25 retrieves but pure dense misses (FR-R7) — this becomes a README artifact.
- **Exit:** `test_retrieve.py` green on basics.

**Day 4 — Fusion + rerank + tuning**
- RRF fusion over the two ranked lists (FR-R3). Cross-encoder rerank fused top-N → top-K (FR-R4). Use LangChain `EnsembleRetriever` for the BM25+dense orchestration; wrap the reranker.
- Return `RetrievedChunk` with all four scores (FR-R5).
- **Tune** N, K, fusion, chunk size against the eval set direction; **measure rerank improvement to context recall vs fusion-only** and record the before/after number (FR-R6).
- **Exit:** hybrid retrieval returns reranked top-K; before/after recall number recorded.

**Risks:** shared-chunk-ID misalignment between indexes (the most common silent bug); reranker latency. Mitigate by asserting ID alignment in a test.

---

## 4. Phase 2 — Generation + Service (Days 5–7)

**Goal:** grounded, cited answers served with production hygiene.

**Day 5 — Generation (`generate.py`)**
- Grounded system prompt: answer only from retrieved context; emit inline citations by chunk ID (FR-G1, FR-G2).
- Abstain when context is insufficient (FR-G3). Capture generation latency separately (FR-G4).
- **Exit:** end-to-end cited answers (no API yet).

**Day 6 — FastAPI service (`serve.py`, `test_serve.py`)**
- `POST /query` with **SSE** token streaming; `GET /healthz`.
- **JWT** auth (single demo user) — reject unauthenticated `/query`; `slowapi` per-IP rate limit — test proves it fires.
- Structured logging (`logging_config.py`): per-stage latency, retrieved chunk IDs, outcome. Pydantic validation everywhere.
- **Exit:** service runs; auth + rate-limit tests green.

**Day 7 — Honest latency**
- Fire a few hundred queries to get real **per-stage p95** (retrieve/rerank/generate). Record measured numbers — no guesses (NFR-3).
- **Exit:** honest per-stage latency numbers recorded.

**Risks:** SSE + Pydantic response shape friction; measuring latency per stage rather than blended (must instrument each stage).

---

## 5. Phase 3 — Eval System (Days 8–10)

**Goal:** the headline — a credible, calibrated Ragas pipeline with failure attribution.

**Day 8 — Ground truth (`scripts/build_ground_truth.py`, `data/ground_truth.jsonl`)**
- LLM-draft 80–120 `{question, reference_answer, relevant_contexts}` triples, **then hand-correct every item** (FR-GT2 — fully synthetic gold is prohibited).
- **Exit:** 80–120 hand-corrected triples committed.

**Day 9 — Ragas pipeline (`eval/run_eval.py`, `eval/store.py`)**
- Run full retrieve→rerank→generate over ground truth; score faithfulness, answer relevance, context recall.
- Wrap judge calls in `tenacity` backoff (NFR-5 — survive free-tier 429s).
- Write `EvalResult` to SQLite keyed by git SHA; export JSON. Print summary table + mean faithfulness (FR-E4, FR-E5).
- **Exit:** `run_eval.py` produces real metrics.

**Day 10 — Attribution + judge calibration (`eval/attribution.py`, `data/ground_truth_audit.md`)**
- Attribution: low context_recall ⇒ retrieval failure; recalled-but-unfaithful ⇒ generation failure; count both (FR-E3).
- Hand-score ~15 items; report correlation with Ragas in `ground_truth_audit.md` (FR-E6). Add the ≥20-case second-annotator agreement writeup (FR-GT3).
- **Exit:** `attribution.py` works; `ground_truth_audit.md` written.

**Risks:** free-tier rate limits during a full run (space it out; backoff is mandatory); poor judge correlation (surface it honestly — a calibrated modest number beats an uncalibrated good one).

---

## 6. Phase 4 — The Gate (Day 11)

**Goal:** the rarest signal — eval wired into CI, blocking merges.

**Tasks**
- `.github/workflows/eval-gate.yml`: on PR → install deps → run a **~25-item representative subset** → compute mean faithfulness → **fail the build if < threshold** (FR-CI1, FR-CI2). Keep it fast/cheap, under free-tier limits (NFR-6).
- Post the metric summary as a PR comment / job output (FR-CI3). LLM key as a **repository secret** (FR-CI4).
- **Prove it:** open a demonstration PR with a deliberately-bad change that turns the gate red; capture the screenshot; preserve the PR in history (FR-CI5).
- **Exit:** the gate blocks a bad PR; demonstration PR + screenshot exist.

**Risk:** CI runtime/cost creep — keep the subset small; the full eval stays local.

---

## 7. Phase 5 — Surface + Ship (Days 12–14)

**Goal:** dashboard, deploy, and the narrative that makes the project legible.

**Day 12 — Dashboard (`dashboard/`)**
- Next.js + **Tailwind core only** (no shadcn/component libs). Reads exported eval JSON.
- Three views: **trends** (metrics over runs vs the gate line), **per-question drill-down** (four retrieval scores + cited answer), **failure attribution** (retrieval vs generation counts, offending questions). The attribution view is the README screenshot.
- **Exit:** dashboard renders real eval JSON.

**Day 13 — Deploy + reproducibility**
- Deploy dashboard to Vercel (static JSON consumer). Serve/deploy the API. Run a **clean-clone reproducibility pass**: `git clone && uv sync && set keys && ingest && serve` in **< 10 min** (NFR-1).
- **Exit:** public dashboard URL; clean-clone run < 10 min.

**Day 14 — Narrative + artifacts**
- README (eval-gate-first hero; "why hybrid" with the concrete BM25 query; "why LangChain" two-sentence subsection; real per-stage p95 + true corpus size + measured Ragas means; limitations as scope choices; literal reproducing commands).
- Root `ARCHITECTURE.md` deliverable (diagram + LangChain rationale — distilled from [`architecture.md`](./architecture.md)).
- 90-second Loom (eval-gate-first script: gate catching a regression → dashboard attribution). Resume bullet with real numbers.
- **Exit:** all build plan §19 artifacts exist and are public.

---

## 8. Dependency Graph (what blocks what)

```
Corpus choice ──▶ ingest ──▶ retrieve ──▶ generate ──▶ serve
                     │           │            │
                     │           └────────────┴──▶ run_eval ──▶ attribution ──▶ eval-gate.yml
                     │                                  │
ground_truth ────────┴──────────────────────────────────┤
(needs corpus)                                           └──▶ JSON export ──▶ dashboard ──▶ deploy
```
- **Corpus** blocks everything. **Ground truth** blocks eval. **Eval** blocks the gate and the dashboard. **The gate** is the deliverable everything else protects.

---

## 9. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| Corpus lacks strong exact-match terms | Hybrid can't be justified | Choose corpus for FR-C2 explicitly on Day 1; validate with the BM25-caught query on Day 3 |
| Shared chunk-ID misalignment | Silent fusion corruption | Assert ID alignment in a test during Phase 1 |
| Free-tier 429s crash eval | Broken headline | `tenacity` backoff (mandatory); space full runs; small CI subset |
| Uncalibrated / weak judge | Eval not credible | Hand-score 15, report correlation honestly (Day 10) |
| Synthetic ground truth slips in | Tests LLM-vs-LLM, not retrieval | Hand-correct **every** item (Day 8) |
| Blended latency reported | Indefensible in interview | Instrument each stage; report per-stage p95 (Day 7) |
| Wrong/invented model ID | Build breaks / dishonest claim | Verify against provider docs before config (Day 1) |
| Scope creep (agents/fine-tune/cache) | Dilutes the focused story | Hold the §1.3 scope boundary; say no visibly |
| Time slip | Missed ship | Cut dashboard polish + corpus size first; never the gate or ground truth |

---

## 10. Definition of Done (v1)

All acceptance criteria in [`requirement.md` §6](./requirement.md) pass, and all 10 output artifacts in build plan §19 exist and are public. Only then consider the **v2 security-guardrail extension** (requirement.md §7.2) — do not begin it until every v1 box is checked.

---

## 11. Immediate Next Actions

1. ✅ **Corpus done** — IETF RFC web-protocol stack (48 RFCs) finalized, fetched, verified, provenance-recorded. Pinned in `data/corpus_manifest.json`.
2. ✅ Repo scaffold, `config.py`, `schema.py`, `pyproject.toml` — committed (979d5c3).
3. **Next: Phase 1, Day 2 — ingestion** (`sentinel/ingest.py`): load the 48 RFCs → clean/normalize → chunk → embed → FAISS + BM25 over shared chunk IDs.

> These docs (`requirement.md`, `product_design.md`, `architecture.md`, `implementation_plan.md`) complete the planning layer. The next commit should be the Phase 0 scaffold — but the corpus decision comes first.
