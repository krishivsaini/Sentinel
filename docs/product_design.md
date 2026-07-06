# Sentinel — Product Design

> Companion to [`requirement.md`](./requirement.md) and [`architecture.md`](./architecture.md). This doc covers *what the product is, who it's for, and how each surface should feel* — the product/UX layer above the engineering spec.
> **Source of truth:** [`SENTINEL_BUILD_PLAN.md`](../SENTINEL_BUILD_PLAN.md).

---

## 1. Product Thesis

Most fresher RAG portfolios ship a chatbot. Sentinel ships an **evaluation system that happens to have a RAG service attached to it.** The product is designed backwards from a single sentence a recruiter should be able to repeat:

> *"Every pull request runs a Ragas evaluation over a hand-built ground-truth set; merges are blocked when faithfulness regresses below threshold."*

Every product surface — the API, the CI gate, the dashboard, the README — exists to make that sentence **true, visible, and defensible.** When a design choice trades "impressive-looking" against "defensible in an interview," defensibility wins.

**Positioning line (README hero):**
> Sentinel: a production RAG service with an evaluation gate in CI. Hybrid BM25 + dense retrieval with cross-encoder reranking, served over FastAPI with SSE streaming, JWT auth, and rate limiting. Every PR runs a Ragas evaluation over a hand-built ground-truth set; merges are **blocked when faithfulness regresses below threshold**. The dashboard attributes every low-scoring answer to either a retrieval failure or a generation failure.

---

## 2. Target Users & Their Jobs

| Persona | Primary job-to-be-done | What they touch | What convinces them |
|---|---|---|---|
| **Recruiter / hiring engineer** | Decide in 90 seconds whether this candidate is serious | README hero → Loom → live dashboard → Actions tab | The CI gate catching a real regression; real (not round) numbers |
| **Interviewer (deep dive)** | Probe whether the numbers are real and the choices are understood | ARCHITECTURE.md, `ground_truth_audit.md`, retrieval code, the demonstration PR | Per-stage latency, judge calibration correlation, the "BM25 caught this" query |
| **Service consumer** | Ask a question, get a grounded, cited answer | `POST /query` (SSE) | Streaming tokens, inline citations, "insufficient context" honesty |
| **Dashboard viewer** | Understand eval health over time and *why* answers fail | Trends, drill-down, attribution views | Failure attribution — retrieval vs generation, with the offending questions |
| **Owner (Krishiv)** | Ship a defensible artifact and speak to every number | Everything | The whole thing reproduces in <10 min from a clean clone |

**Design implication:** the recruiter and interviewer are the *real* primary users. The service consumer is a means, not the end. This is why the eval and dashboard get design attention while the service stays deliberately lean (single demo user, no chat UI).

---

## 3. Product Surfaces

Sentinel has four user-facing surfaces plus one narrative surface. Each has a distinct design goal.

### 3.1 The API Service — *"reads as product, not student demo"*
- **Design goal:** production hygiene legible at a glance. The moment someone reads the endpoint list they should think "someone who has run a service in production built this."
- **Shape:** `POST /query` (JWT-protected, SSE-streamed, rate-limited) and `GET /healthz`. Nothing more.
- **Answer object as the product's core value unit:** every response carries `text`, `citations` (chunk IDs), the full `retrieved` set with all four scores, and `latency_ms` split into `{retrieve, rerank, generate}`. The answer is *inspectable*, not just readable.
- **Honesty as a feature:** when context is insufficient, the answer says so. "I don't know from the provided context" is a first-class product behavior, not a failure — it's what protects faithfulness.
- **Deliberately absent:** no chat memory, no conversation UI, no user management. Those would dilute the retrieval story.

### 3.2 The Evaluation Pipeline — *the headline product*
- **Design goal:** make evaluation feel like a **first-class engineering system**, not a script. It has a schema (`EvalResult`), persistent storage (SQLite keyed by git SHA), and an export contract (JSON for the dashboard).
- **Two audiences, two run modes:**
  - **Full eval** — runs over all 80–120 ground-truth items, locally / on-demand, spaced out to respect rate limits. This is the number that goes on the resume.
  - **CI subset** — ~25 representative items on every PR, fast and cheap, powering the gate.
- **Failure attribution is the signature feature.** A low score is never just "bad" — it's classified as a *retrieval failure* (relevant context wasn't retrieved) or a *generation failure* (context was there, answer was still wrong). This single distinction is the product's intellectual differentiator.
- **Judge honesty as a designed artifact:** the pipeline ships with `ground_truth_audit.md` reporting the correlation between hand scores and Ragas scores. An uncalibrated judge is treated as an unfinished product.

### 3.3 The CI Faithfulness Gate — *the rarest surface*
- **Design goal:** turn "we care about quality" into an **enforced, visible mechanism.** The gate is a product feature, not infrastructure trivia.
- **User-visible behavior:** on a PR, a green check with a metric summary comment ("faithfulness 0.83 ≥ 0.80 ✓"), or a red X that blocks merge on regression.
- **The demonstration PR** is a designed artifact: a deliberately-bad change that turns the gate red, preserved in history as the README's proof screenshot. This is the single most persuasive object in the whole project.

### 3.4 The Dashboard — *dense, functional, one day*
- **Design goal:** an analyst's read-only console, not a marketing page. Information density over ornament. Tailwind core only.
- **Design principle:** every pixel earns its place by helping someone answer "is eval health improving, and when it isn't, *why*?"
- **Three views** (see §4 for IA).
- **Non-goals:** no animations, no charts library flourish, no auth, no live DB. It reads a static JSON export and renders. Simplicity here is a *scope signal*, not a limitation to apologize for.

### 3.5 The Narrative Surface — README + Loom + ARCHITECTURE.md
- **Design goal:** control the reading order so the eval-gate story lands first and the "RAG chatbot" framing never gets a chance to form.
- **README reading path:** hero sentence → live dashboard link → 90s Loom → architecture → "why hybrid" (with the concrete BM25 query) → "why LangChain" → real results (per-stage p95, true corpus size, measured Ragas means) → limitations (framed as scope choices) → reproducing (literal commands).
- **Loom script (90s):** open on the CI gate catching a regression, *then* show the dashboard attribution view. The chatbot is shown last, briefly, if at all.

---

## 4. Dashboard Information Architecture

Three views, navigable from a single top nav. Data source: exported eval JSON, one file per run (keyed by git SHA + timestamp).

### 4.1 Trends (landing view)
- **Purpose:** answer "is quality improving or regressing over time?" at a glance.
- **Content:** three metric lines — faithfulness, answer relevance, context recall — plotted over eval runs (x-axis = git SHA / time). A horizontal threshold line on faithfulness (the gate line, e.g. 0.80) so a regression below the gate is instantly visible.
- **Interaction:** click a run point → drill-down for that run.

### 4.2 Per-Question Drill-Down
- **Purpose:** answer "for this run, which questions did well/poorly, and why?"
- **Content per question:** the question, its three Ragas scores, the full retrieved set (each chunk with **all four scores** — dense, sparse, fused, rerank), and the generated answer with inline citations resolving to chunk IDs.
- **Design value:** exposes the retrieval internals a reviewer would otherwise have to take on faith. Seeing dense vs sparse vs rerank scores side by side *is* the hybrid-retrieval story made visual.

### 4.3 Failure Attribution (the README screenshot)
- **Purpose:** answer "when answers fail, is it retrieval or generation?"
- **Content:** per-run counts of retrieval-failures vs generation-failures, with the offending questions listed under each bucket, linking back to their drill-down.
- **Design value:** this is the view that separates Sentinel from prompt-tweaking projects. It gets the most design care of the three; it's the frame captured for the README.

---

## 5. Key Product Behaviors & Rules

- **Grounded-or-abstain:** the model answers only from retrieved context and abstains when context is insufficient. This is a product promise, enforced in the generation prompt and checked by faithfulness.
- **Everything is attributable:** an index records its chunking strategy; an eval run records its git SHA; an answer records its retrieved chunks and per-stage latency. Any result can be traced to the conditions that produced it.
- **Measured, not claimed:** no placeholder numbers anywhere user-facing. If faithfulness is 0.79, the product says 0.79. Honest numbers are a designed trust signal (build plan §18).
- **Cost is visible, not optimized:** cost per query is tracked and surfaced; it is deliberately *not* optimized (that's a separate project). Tracking-without-optimizing is itself a stated scope decision.

---

## 6. Experience Principles

1. **Lead with the gate.** Every entry point (README, Loom, dashboard nav) surfaces the eval-gate story before the chatbot.
2. **Inspectable over impressive.** Prefer surfacing internals (four retrieval scores, per-stage latency, attribution) over polish.
3. **Honest by construction.** Abstention, real numbers, and judge calibration are features, not caveats.
4. **Dense over decorative.** Especially the dashboard — functional clarity beats visual flourish; low ornament is a scope signal.
5. **Sharp scope boundary.** Say no to agents, fine-tuning, chat memory, and cost optimization *visibly* — a clean "what this is not" is part of the product.

---

## 7. Success Metrics (Product-Level)

| Signal | Target / definition |
|---|---|
| Recruiter comprehension | The eval-gate sentence is understood from the README hero + Loom alone. |
| Reproducibility | A stranger clones and runs the service in **< 10 minutes**. |
| Defensibility | Every number on the README is reconstructable by the owner on a whiteboard. |
| Proof of the gate | A real demonstration PR shows the gate turning red on a regression. |
| Attribution clarity | A viewer can name, from the dashboard, whether the last run's failures were retrieval or generation. |

---

## 8. v2 Product Extension — Security Guardrails (post-v1 only)

If runway remains after every v1 acceptance criterion passes, the guardrail layer extends the *same* eval-and-gate story rather than opening a new one:
- **New product promise:** *"Every PR runs both a faithfulness gate and an adversarial-safety gate; merges are blocked when either regresses."*
- **New surfaces:** a guardrails panel on the dashboard (adversarial-case outcomes, redaction and scope-violation counts per run) and a second CI gate on injection-success rate.
- **Capabilities:** prompt-injection defense (against poisoned retrieved chunks), PII detection + redaction, output/grounding validation, retrieval-scope enforcement — each **measured**, joining the existing metrics.
- **Product guardrail:** adversarial corpus content is labeled and segregated so it can never contaminate a normal eval run. This scope discipline is itself part of the product story. Do not begin until v1 is a finished, defensible artifact on its own.
