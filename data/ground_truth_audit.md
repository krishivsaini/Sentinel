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

An eval gate is only as trustworthy as its judge, so we don't take the Ragas faithfulness judge
(`llama-4-scout-17b`) on faith. We hand-scored a **20-item spread** of generated answers for
faithfulness — checking each answer's claims **against the retrieved contexts it was actually
given** (grounding, not mere truth), verified chunk-by-chunk — and compared to the judge's scores
(`scripts/judge_calibration.py` holds the labels + computes the stats).

**Result (n = 20):**

| Metric | Value |
|---|---|
| Pearson r (hand vs. judge) | **−0.10** |
| Spearman ρ | −0.02 |
| Mean absolute error | 0.135 |
| Agreement "faithful" (≥ 0.5) | 20/20 |

**Reading it honestly.** The MAE is small only because both raters cluster near 1.0; the
**near-zero correlation is the real signal** — on every *discriminating* case the judge and a
careful human diverge. The judge produced clear **false positives**, scoring 1.0 on answers whose
key claim was absent from the retrieved context:

- *QUIC loss detection* — answer states the `kTimeThreshold/kGranularity` formula, but the cited
  chunk is about packet **reordering**; the formula was never retrieved. (hand 0.5, judge 1.0)
- *TCP MSL* — "2 minutes" cites the **ISN** chunk; the MSL definition wasn't retrieved. (0.5 vs 1.0)
- *OAuth `invalid_grant`* — cites the §11.4 **registry** chunk for a "Section 5.2" claim and never
  defines the error. (0.5 vs 1.0)

…and one **false negative** (`SETTINGS_HEADER_TABLE_SIZE`, verbatim-grounded in the context, judge
0.5 / hand 1.0). A spot-check with a stronger model (`gpt-oss-120b`) reproduced the same leniency
on the QUIC case, so this is **not Scout-specific**: LLM faithfulness judges credit
*topically-plausible* claims even when the specific fact isn't in the retrieved text.

**What this means for the headline number.** The reported mean faithfulness should be read as an
**upper bound** on true context-grounding — the judge is lenient, so real faithfulness is somewhat
lower than the score implies. This is exactly why the number is reported *with* its calibration
rather than as gospel. The honest fix is a stronger, calibrated judge (the Ragas-native paid
`gpt-4o-mini` is a one-line swap via `sentinel/llm.py`); the free-tier judge is a documented
constraint, not a hidden one.

> Provenance note: the 20 hand-scored answers come from a Groq eval run; the *judge* under test
> (Scout) is the shipped one, so the calibration characterizes the judge that produces the gate's
> numbers. The headline means (faithfulness / answer-relevance / context-recall over all 48 items)
> are recorded in the exported run JSON under `dashboard/data/`.
