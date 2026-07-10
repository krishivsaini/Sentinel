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

> **Status (2026-07-07):** Days 2 & 3 done, and Day 4's *core* pulled forward — the full
> dense→sparse→RRF→cross-encoder pipeline is built and tested. Ingestion: 48 RFCs → 2,888
> chunks (bge-small, 384-d), FAISS + BM25 with asserted shared IDs. `test_retrieve.py`: 4
> passing (alignment, four-scores, FR-R7, EnsembleRetriever orchestration), 1 skipped (FR-R6).
> **FR-R7 artifact captured** (for the README): query *"What does the invalid_grant error
> mean?"* → gold chunk `rfc6749#0043` is **BM25 rank 1** but **pure-dense rank 24**; hybrid
> RRF+rerank recovers it into the final top-5. **Still open (needs the eval set, Phase 3):**
> tuning N/K/fusion/chunk-size and the FR-R6 rerank-vs-fusion recall number.

**Day 2 — Ingestion (`ingest.py`)** — ✅ done
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

> **Status (2026-07-08):** Day 5 done. `generate.py` streams grounded answers over
> `gemini-3.5-flash` (temperature 0) via langchain-google-genai, emits inline `[chunk_id]`
> citations parsed + filtered against the retrieved set (hallucinated IDs can't become
> citations), and abstains with a fixed sentence when context is insufficient. Verified
> live end-to-end: the 429 question → grounded answer quoting `429`/`Retry-After`/`MUST`
> with citation `[rfc6585#0002]`; an out-of-corpus question → clean abstention. Note:
> Gemini-3 returns **structured content blocks**, not plain strings — handled in `_part_text`.
> `test_generate.py`: 5 hermetic (always run) + 2 live (opt-in via `SENTINEL_RUN_LLM_TESTS=1`,
> both passing). Retry/backoff for 429s via LangChain `.with_retry()`.

**Day 5 — Generation (`generate.py`)** — ✅ done
- Grounded system prompt: answer only from retrieved context; emit inline citations by chunk ID (FR-G1, FR-G2).
- Abstain when context is insufficient (FR-G3). Capture generation latency separately (FR-G4).
- **Exit:** end-to-end cited answers.

**Day 6 — FastAPI service (`serve.py`, `test_serve.py`)** — ✅ done
- `POST /query` with **SSE** token streaming (`retrieved` → `token`* → `done` events; the `done` event carries the full `Answer` — text, citations, retrieved set with all four scores, per-stage `latency_ms`); `GET /healthz`; `POST /token` (JWT).
- **JWT** auth (single demo user) — unauthenticated `/query` → 401; `slowapi` per-IP rate limit — a test proves it fires (limiter created per-app so tests don't bleed state).
- Structured logging (`logging_config.py`, structlog JSON): per-stage latency, retrieved chunk IDs, outcome. Pydantic validation everywhere. Pipeline injected as a `QueryEngine` dependency so `test_serve.py` (5 passing) is hermetic — no models/API.
- **Startup warmup (lifespan):** loads the index + both local models *and runs one dummy retrieval*, so first-query latency is honest (cold-start model load + first-inference kernel warmup would otherwise land inside the timed stages). Lifespan doesn't fire under `TestClient(app)` without `with`, keeping tests fast.
- **Exit:** service runs; auth + rate-limit tests green. Verified live over HTTP (token → SSE query, grounded + cited).

**Day 7 — Honest latency** — ✅ done
- `scripts/bench_latency.py` measures each stage in isolation (never blended, NFR-3) after a warmup; writes `data/latency_report.json` (committable artifact) with p50/p95/p99 + platform/device/config.
- **Measured (macOS, MPS, 2026-07-08):**

  | stage | n | p50 | **p95** | p99 |
  |---|---|---|---|---|
  | retrieve (dense+sparse+RRF) | 200 | 61 ms | **379 ms** | 518 ms |
  | rerank (cross-encoder) | 200 | 878 ms | **1928 ms** | 3624 ms |
  | generate (Gemini 3.5 Flash) | 6 | 5951 ms | **7192 ms** | 7261 ms |

- **Honesty notes:** retrieve/rerank are local → full 200-sample p95. **Rerank is the dominant local cost** (lever: shrink the rerank pool). Generation p95 is over **n=6 clean, un-throttled single calls** — the Gemini free tier throttled heavily during benchmarking (the google-genai SDK's *transport-level* retries, not just LangChain's, inflate a throttled call to minutes; `max_retries=0` doesn't fully disable them). Throttled calls are excluded as a free-tier artifact, not a system property; the benchmark now guards each generate call with a hard timeout + per-sample skip so this can't pollute future runs.
- **Exit:** honest per-stage p95 recorded in `data/latency_report.json` + here.

**Risks:** SSE + Pydantic response shape friction; measuring latency per stage rather than blended (must instrument each stage).

---

## 5. Phase 3 — Eval System (Days 8–10)

**Goal:** the headline — a credible, calibrated Ragas pipeline with failure attribution.

> **Status (2026-07-09):** Day 8 batch done — **48 hand-authored/verified triples** in
> `data/ground_truth.jsonl` (balanced 3/cluster across all 16 clusters). Free-tier throttling
> made bulk LLM drafting unreliable (19/48, then 0), so — per the production-reliability call —
> ground truth is built **LLM-free**: `build_ground_truth.py --assist` surfaces candidate chunks
> via local retrieval, and answers were authored/verified against the actual RFC text
> (`scripts/assemble_ground_truth.py` holds the authored answers + gold chunk_ids; 17 verified
> Gemini drafts reused, 31 hand-authored). Every `relevant_contexts` entry is a real corpus
> passage; `[chunk_id]` markers stripped; faithfulness spot-checked. **To reach the §16 target
> of 80–120:** append seeds to `data/gt_seed_questions.jsonl` and re-run the same LLM-free flow.

> **Status (2026-07-09) — Day 9 done:** the Ragas pipeline runs end-to-end and produces real
> metrics (`sentinel/eval/run_eval.py`, `store.py`, `attribution.py`; 31 hermetic tests green).
> **Model IDs re-verified against the live API** (no invention): `gemini-3.5-flash` free tier is
> only **~20 requests/day** (would throttle both the eval *and* the product endpoint) and
> `gemini-2.0-flash` is no longer free — both dropped. Final picks: **generation =
> `gemini-2.5-flash`**, **judge = `gemini-2.5-flash-lite`** (different models on purpose:
> separate free-tier quota pools + the judge never grades its own model's output). Free-tier
> survival is real, not hoped-for: per-item (not batch `evaluate()`) scoring, **backoff that
> honors the server's `retryDelay`** (waits out the 10-RPM window instead of guessing), and
> **SQLite checkpointing keyed by git SHA** so a 429-killed run *resumes* instead of re-burning
> quota. A ragas↔langchain-community-0.4 import incompatibility (`ChatVertexAI`) is handled by a
> targeted shim in `sentinel/eval/__init__.py`. Abstentions score faithfulness **1.0** (a refusal
> can't hallucinate; its failure to answer is caught as low answer-relevance → `generation_fail`).

**Day 8 — Ground truth (`scripts/build_ground_truth.py`, `data/ground_truth.jsonl`)** — ✅ batch (48/80–120)
- LLM-draft 80–120 `{question, reference_answer, relevant_contexts}` triples, **then hand-correct every item** (FR-GT2 — fully synthetic gold is prohibited). *(Built LLM-free from local retrieval instead — see status above.)*
- **Exit:** hand-corrected triples committed (48 now; extend to 80–120 by appending seeds).

**Day 9 — Ragas pipeline (`eval/run_eval.py`, `eval/store.py`)** — ✅ done
- Run full retrieve→rerank→generate over ground truth; score faithfulness, answer relevance, context recall. *(Done — per-item loop, not batch `evaluate()`, so it's controllable + resumable.)*
- Wrap judge calls in backoff (NFR-5 — survive free-tier 429s). *(Done — retry-delay-aware backoff around **both** judge and generation calls.)*
- Write `EvalResult` to SQLite keyed by git SHA; export JSON. Print summary table + mean faithfulness (FR-E4, FR-E5). *(Done — `store.py`; per-item checkpoint + `dashboard/data/eval_<run>.json` + `latest.json`.)*
- **Exit:** `run_eval.py` produces real metrics. ✅ `python -m sentinel.eval.run_eval [--subset N | --ci]`.

**Day 10 — Attribution + judge calibration (`eval/attribution.py`, `data/ground_truth_audit.md`)** — ✅ done
- Attribution: low context_recall ⇒ retrieval failure; recalled-but-unfaithful ⇒ generation failure; count both (FR-E3). *(Done — `attribution.py`, exercised in `test_eval.py`.)*
- Hand-score ~20 items; report correlation with Ragas in `ground_truth_audit.md` (FR-E6). *(Done — Pearson r ≈ −0.10; the judge is lenient (credits topically-plausible claims), so the headline faithfulness is an upper bound. `scripts/judge_calibration.py`.)*
- Second-annotator agreement writeup (FR-GT3). *(Done — 24-item second pass; 22 confirmed, 2 gold-context defects fixed; honest single-annotator caveat.)*
- **Exit:** `attribution.py` works; `ground_truth_audit.md` written. ✅

> **Status (2026-07-10) — Phase 3 done.** Eval runs end-to-end on Groq's free tier via a
> provider-agnostic factory (`sentinel/llm.py`): generation `gpt-oss-20b`, judge `gpt-oss-120b`,
> **both non-deprecated**. The LLM-backend journey is documented for honesty: Gemini free is
> ~20 req/day (dropped); Groq's Scout judge works but deprecates 2026-07-17 (dropped); the
> non-deprecated GPT-OSS models are *reasoning* models that at default effort return empty answers
> (generation) or Ragas-breaking output (judge) — fixed with `reasoning_effort="low"`, which makes
> both reliable. Ground truth = 48 hand-authored triples (second-pass audited); judge calibrated
> and reported honestly (r ≈ −0.10 — the judge is lenient, so faithfulness is an upper bound). A
> paid `gpt-4o-mini` judge remains a one-line swap for anyone wanting a stronger/faster judge.

**Risks (borne out):** free-tier rate limits forced heavy pacing + retry-delay-aware backoff; the
judge correlation is weak and is surfaced honestly rather than hidden — a calibrated modest number
beats an uncalibrated good one.

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
