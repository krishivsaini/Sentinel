# Ground-Truth & Judge Audit

> Companion artifact to the eval system (§12). Documents (1) how the evaluation ground truth
> was built, (2) a second-pass verification of its correctness (FR-GT3), and (3) calibration of
> the Ragas faithfulness *judge* against human labels (FR-E6). The guiding principle: **a
> calibrated, honestly-reported number beats an impressive uncalibrated one.**

---

## 1. Ground-truth provenance & construction (FR-GT1/2)

**What it is.** `data/ground_truth.jsonl` — **48** hand-authored `{question, reference_answer,
relevant_contexts}` triples over the 48-RFC web-protocol corpus, balanced **3 per thematic
cluster across all 16 clusters** (keywords, HTTP core/compression/extensions, URIs, cookies,
WebSocket, QUIC, TCP, TLS, PKI, OAuth, HTTP-auth, JOSE/JWT, encodings, DNS).

**How it was built — without a hosted LLM in the authoring loop.** Bulk LLM drafting was
abandoned after free-tier throttling made it non-reproducible (19/48 drafted, then 0). Instead,
per the project's production-reliability principle, the gold is a **dependency-free, reproducible
artifact**:

1. `scripts/build_ground_truth.py --assist` runs the **local** hybrid retriever (no API) over a
   seed question and writes the candidate chunks for review.
2. Each `reference_answer` was **authored and verified against the actual RFC text**;
   `relevant_contexts` are the real corpus passages (resolved by `chunk_id` from the index) that
   support the answer. `scripts/assemble_ground_truth.py` holds the authored answers + gold
   `chunk_id`s and resolves them to text.
3. Provenance: **31** items hand-authored from the RFC text; **17** reused from earlier Gemini
   drafts *after* each was checked against its cited chunk. Inline `[chunk_id]` markers are
   stripped from the stored answers.

No answer is fully synthetic (FR-GT2): every one was read against, and traced to, primary source
text. Average gold contexts per item: ~1.6.

---

## 2. Second-pass verification (FR-GT3)

A structured second pass independently re-checked a **24-item sample** (stride-2 over the file,
so it spans all 16 clusters): for each item, does the `reference_answer` actually follow from its
`relevant_contexts`, and is it accurate per the RFC?

**Result: 22 / 24 confirmed grounded on the first pass; 2 gold-context defects found and fixed.**

| Item | Finding | Fix |
|---|---|---|
| "purpose of the TCP receive window" | Gold `ctx0` was the **urgent-pointer** chunk (irrelevant); only the zero-window behaviour was supported. | Replaced `ctx0` with `rfc9293#0009` (the *Window field* definition — "the number of data octets … the sender of this segment is willing to accept"). |
| "What does an A resource record provide" | Gold context was an **off-by-one SOA chunk** (`rfc1035#0020`, SOA `MINIMUM`) that never mentions A records. | Replaced with `rfc1035#0021`, which carries the A-record RDATA ("ADDRESS A 32 bit Internet address … multiple A records … no additional section processing"). |

Both are context-selection defects (the *answers* were correct); both were corrected in
`data/ground_truth.jsonl`. Agreement between the original authoring and the independent second
pass was **22/24 (92%)** before correction, **24/24 after**.

*Honest limitation:* this is a single-author project, so the "second pass" is a structured
re-review by the same author rather than a fully independent second annotator. The process (blind
re-derivation of support from primary text, tabulated disagreements, corrections applied) is the
credible part; a second human annotator would strengthen inter-rater reliability further.

> Note on how the eval uses this: faithfulness/answer-relevance and `LLMContextRecall` score the
> pipeline's **retrieved** contexts and generated answer against the **reference answer** — the
> gold `relevant_contexts` field documents support but is not itself fed to the judge, so these
> fixes improve the artifact without moving the reported metrics.

---

## 3. Judge calibration (FR-E6)

An eval gate is only as trustworthy as its judge, so we don't take the Ragas faithfulness judge on
faith. We hand-scored an **18-item spread** of the shipped judge's own run (judge `gpt-oss-120b`)
for faithfulness — checking each answer's claims **against the retrieved contexts it was actually
given** (grounding, not mere truth), verified chunk-by-chunk — and compared to the judge's scores
(`scripts/judge_calibration.py` holds the labels + computes the stats).

**Result (n = 18, judge `gpt-oss-120b`):**

| Metric | Value |
|---|---|
| **Pearson r** (hand vs. judge) | **+0.90** |
| **Spearman ρ** | **+0.98** |
| Mean absolute error | 0.047 |
| Agreement "faithful" (≥ 0.5) | 16/18 |

**Reading it honestly.** The judge is **well-calibrated**: it agrees with a careful human not just
in aggregate but on the *discriminating* cases — it correctly docked the answers whose claim was
only weakly grounded in the retrieved context, e.g.:

- *Sec-WebSocket-Key purpose* — the retrieved chunks were about the Host header / extensions, not
  the key's role; judge **0.33**, hand 0.5.
- *MUST / SHOULD / MAY* — the definitions were only partly present in the retrieved chunk; judge
  **0.40**, hand 0.6.
- *308 Permanent Redirect* — core is grounded but the answer over-elaborates; judge 0.67, hand 0.75.

…while scoring the cleanly-grounded answers ~1.0 in step with the human. Cohen's κ is uninformative
here (at the 0.5 threshold almost every item is "faithful", so the binary labels have near-zero
variance) — the continuous correlation is the meaningful signal.

**Context — an earlier judge was *not* calibrated.** The first Groq run used `llama-4-scout-17b`
(since deprecated), which scored a near-zero correlation (r ≈ −0.10): it credited
topically-plausible claims even when the fact wasn't retrieved. Moving to `gpt-oss-120b` (forced by
Scout's deprecation, enabled by `reasoning_effort="low"`) is what produced a judge that actually
tracks grounding — a reminder that the calibration step is load-bearing, not ceremony.

> Provenance note: hand-scores + judge scores are on the shipped `gpt-oss-120b` run. The headline
> means below come from the same run and are re-exported to `dashboard/data/` on every run.

### Headline means (this run)

Free-tier per-minute token limits + connection drops under sustained load capped a single run at
**22 of 48** items (documented in §5 of the plan; a paid key or repeated resume closes the gap).
Over those 22:

| Metric | Mean | Gate |
|---|---|---|
| **Faithfulness** | **0.905** | ≥ 0.80 → **PASS** |
| Answer relevance | 0.939 | — |
| Context recall | 0.951 | — |
| Attribution | 20 pass / 2 generation_fail / 0 retrieval_fail | — |

Faithfulness held **0.85–0.92 across every sample size** (8 → 12 → 18 → 22), so the figure is
stable, not an artifact of which items happened to complete.
