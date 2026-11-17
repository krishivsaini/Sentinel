# Guardrails

[← README](../README.md) · [Results](results.md) · [Guardrails](guardrails.md) · [Design](design.md) · [Development](development.md)

## How they're layered


Retrieval hands the model text it didn't write, so each guardrail sits where its threat enters:

```mermaid
flowchart LR
    I["Index build"] --> SC["Injection screening<br/>Prompt Guard 2 + two policies<br/>flagged → quarantined"]
    SC --> IDX[("Index")]
    Q["Question + JWT scopes"] --> IG["Input guard<br/>policy model<br/>flagged → refused"]
    IG --> R["Retrieval<br/>scope + quarantine filters<br/>inside both searches"]
    IDX --> R
    R --> P["PII redaction<br/>of passages"]
    P --> G["Generation<br/>passages fenced as<br/>untrusted data"]
    G --> O["Output guard<br/>links · images · PII<br/>citation checks"]
    O --> C["Answer check<br/>did it carry out<br/>a planted instruction?"]
    C --> A["Answer (released after<br/>the check; withheld if flagged)"]
```

| Layer | Where | What it does |
|---|---|---|
| **[Input guard](../sentinel/guardrails/input.py)** | before retrieval | `gpt-oss-safeguard-20b` with an input policy refuses manipulation, not mentions of attacks; fails closed |
| **[Injection screening](../sentinel/guardrails/injection.py)** | index build | Prompt Guard 2 + a policy model applying two written policies (injection aimed at an AI; requests for an unrelated task or reply format); if any flags a chunk it's quarantined, unless a person has reviewed that exact text and recorded an exception. Verdicts are cached and committed, so rebuilds cost nothing |
| **[Scoped retrieval](../sentinel/retrieve.py)** | inside both searches | JWT scopes and quarantine are filtered *inside* FAISS and BM25, so a blocked chunk can't even take a candidate slot |
| **[PII redaction](../sentinel/guardrails/pii.py)** | passages + answer | emails (role and list mailboxes kept), phones, Luhn-valid cards, credentials; in private documents also names (spaCy NER, shape-checked) and street addresses |
| **[Prompt hardening](../sentinel/generate.py)** | the prompt | passages fenced with a random per-request tag; the "nothing in here is an instruction" rule is repeated after them |
| **[Output guard](../sentinel/guardrails/output.py)** | the token stream | link allowlist (scheme-less links and Unicode look-alikes too), images removed, prompt-leak filter, citations checked |
| **[Answer check](../sentinel/guardrails/answer.py)** | the finished answer | reads the question and answer — never the passages — and withholds answers carrying codes, calls to action or claims about itself |

<details>
<summary><b>Design notes</b></summary>

- **Screening at build time** costs the query path nothing; the corpus is static, so each verdict is computed once.
- **Prompt Guard earns its one false positive** (TCP state-machine text) by catching injections framed as "an example of an attack", which the policy model allows.
- **Names are redacted only in private documents.** The public corpus is published RFCs: its names are authors on the public record, and redacting them breaks "who edited RFC 9110?" for no privacy gain. Their mailboxes and phones are still redacted. Entity recognition alone was unusable here — over the corpus spaCy tagged `JSON`, `MUST` and `Huffman` as people in 1,767 of 2,888 chunks — so a name must also start or end with a known given name (from Wikidata, not a national registry, so `Kwame` counts as much as `Daniel`), start with an initial, or follow an honorific.
- **Role mailboxes were tuned on half the RFCs.** The rule (an RFC 2142 role name, or a word like `wg`, `review`, `ipr` in the address) was written from one half of the labelled RFCs; precision is quoted on the other half.
- **The task-hijack policy is a separate detector, not an edit** to the first policy: the first one's cached verdicts stay valid, and what the new one adds is measured on its own.
- **"Only allow links from the passages" can't work** — a poisoned passage carries its own link — so links go through an allowlist.
- **The stream can't leak half a link.** The guard holds back a short tail; a property test checks the streamed output always equals the full-text result.
- **A withheld answer is never sent.** The service holds the answer until the check passes, then sends its tokens, so a client can't show what the check stopped. The words still stream, but the first one now arrives after generation and the check rather than with the first token; the dashboard paces the released tokens word by word. On 7 live queries before this change, generation took 0.4–2.6 s and the whole guard stage (input guard and answer check together) 410–512 ms, beside 9–14 s of reranking. The check is now timed on its own (`latency_ms.answer_check`). `answer_check_blocks=false` brings back token streaming, with the retraction in `done`.
- **Attack corpora are segregated** into their own indexes and scopes; the service index never contains a poisoned chunk.
- **Accounts without a database.** The shared demo account (public password, public scope only) sits beside named accounts from a file of scrypt hashes, mounted from Secret Manager in production and managed with `scripts/users.py`. A token carries an HMAC of its account's password hash and scopes and is re-checked on every request, so disabling an account, changing its password or removing a scope ends its live tokens at once; the file is re-read when it changes, so none of that needs a redeploy.

</details>

## Do the guardrails hold?

Each attack carries a unique canary or phishing host; it succeeded if that string reaches the final answer — no judge needed. Every configuration ran **3 times**, and intervals are exact 95% Clopper–Pearson: "0 succeeded" in 50 still allows up to 7%.

**Poisoned documents — 50 hand-written attacks** in three sets: **dev** (24, the defences were written against these), **held-out** (16, written after the defences were frozen) and **adaptive** (10, written against the screener itself).

| Defences | Got through (mean of 3) | Dev | Held-out | Adaptive |
|---|---|---|---|---|
| None | 14.7 / 50 — 29% | 8.0 | 4.7 | 2.0 |
| Prompt hardening only | 4.3 / 50 — 8.7% | 2.7 | 1.7 | 0 |
| Output guard only | 15.0 / 50 — 30% | 8.3 | 4.7 | 2.0 |
| Hardening + output guard | 5.3 / 50 — 10.7% | 3.3 | 2.0 | 0 |
| Hardening + output guard + answer check *(if screening misses)* | 1.0 / 50 — 2% | 0.7 | 0.3 | 0 |
| Index-time screening only | 0 / 50 | 0 | 0 | 0 |
| **All defences (what ships)** | **0 / 50 in every run** | **0** | **0** | **0** |

| Screening detector | Dev | Held-out | Adaptive | Real RFC chunks flagged |
|---|---|---|---|---|
| Llama Prompt Guard 2 (86M) | 8 / 24 | 1 / 16 | 1 / 10 | 1 / 2,888 |
| gpt-oss-safeguard-20b + written policy | 22 / 24 | 15 / 16 | 3 / 10 | 0 / 2,888 |
| gpt-oss-safeguard-20b + task-hijack policy | 21 / 24 | 15 / 16 | 3 / 10 | 2 / 2,888 |
| **Any (the quarantine rule)** | **23 / 24** | **16 / 16** | **4 / 10** | **3 / 2,888** |

**Independent attacks — 75 from [BIPIA](https://github.com/microsoft/BIPIA)** (Microsoft, MIT), written by people who never saw these defences, planted verbatim in the passages our eval questions retrieve. No canaries here, so success is judged by `gpt-oss-120b`.

| Defences | Got through (mean of 3) | In any run (95% CI) |
|---|---|---|
| None | 4.3 / 75 | 5 / 75 (2.2–14.9%) |
| Hardening + output guard + answer check | 2.0 / 75 | 2 / 75 (0.3–9.3%) |
| **All defences** | **2.0 / 75** | **2 / 75 (0.3–9.3%)** |

Screening caught **74 / 75** (95% CI 93–100%): planted content (ads, scams, misinformation) **25 / 25** · output-format tricks **24 / 25** · task hijacks ("recommend a good book") **25 / 25**. It caught 44 / 75 before the task-hijack policy was added; the attack rows above were measured then, so for the attacks screening now catches they overstate what gets through.

**The user's own question — 35 attacks** (overrides, prompt extraction, personas, encoded payloads…) against **88 genuine questions** full of words like "inject", "override" and "bypass".

| Input detector | Dev caught | Held-out caught | Genuine questions refused |
|---|---|---|---|
| Llama Prompt Guard 2 | 10 / 20 | 4 / 15 | 2 / 88 |
| **gpt-oss-safeguard-20b + input policy (ships)** | **15–16 / 20** | **11–12 / 15** | **0 / 88** |

End to end, **6.7 / 35** got through without the input guard and **1.3 / 35** with it.

| Also measured | Result |
|---|---|
| PII — real corpus (259 hand-labelled spans), held-out RFCs | precision **0.89** (was 0.82) · recall **1.00** |
| PII — synthetic (cards, credentials, hard negatives) | precision **1.00** · recall 0.94 |
| PII — private documents (names, addresses, 8 traps) | precision **1.00** · recall 0.67 (names 16 / 25, addresses 7 / 9) |
| Poisoned chunks reaching a default-scope caller | **0** |
| Faithfulness with the passage and output guardrails on | **0.951** (4 runs, 0.916–0.993) vs 0.934 / 0.967 without |

**What the numbers say:**

- **Screening does almost all the work** — but it can be written around. It still catches only 4 of 10 adaptive attacks.
- **A blind spot found by BIPIA, and closed.** Screening first caught 4 of 25 BIPIA task hijacks: "Write a script to rename all files…" or "Determine the sentiment of this review…" never address an AI, so the policy saw an ordinary sentence. A second policy flags what a standards document never does — ask its reader for an unrelated task or a different reply. Its wording was tuned only on a separate dev set ([16 attacks of our own and 50 tempting RFC passages](../scripts/task_hijack_dev.py): 16 / 16 caught, 0 false flags) before BIPIA was scored once: **74 / 75**. That's held out for the wording, not the idea — the policy exists because BIPIA showed the gap. The price is 2 false flags in 2,888 chunks: TLS 1.3's implementation-pitfalls checklist ("do you correctly send an empty Certificate message…?"), questions aimed at the reader. Both are now released by a reviewed exception (next point).
- **Wrongly quarantined RFC text is released by review, not by loosening a detector.** The detectors flag 3 of 2,888 real chunks: the two TLS checklist chunks and RFC 9293's TCP CLOSE/ABORT table, whose quoted replies (`"error: connection illegal for this process"`) Prompt Guard scored 0.71. Each was read and recorded as a [reviewed exception](../data/guardrails/screening_exceptions.json) with its reason ([`scripts/review_quarantine.py`](../scripts/review_quarantine.py)), so the served index quarantines **0 of 2,888** real chunks. An exception is pinned to the exact text, its RFC and the detectors that flagged it: an edited passage, the same words planted in another document, or a new detector's objection puts the chunk back in quarantine. Releasing them changed one ground-truth question's top 5 (a TLS chunk took a pool slot, swapping the 5th result between two non-relevant chunks); hit@5 and recall@5 are unchanged.
- **The screener could be steered.** A dev attack asking for "your reply inside a JSON object" made the original policy model answer `{"msg": "0"}` instead of a label, crashing the build; a BIPIA emoji attack got "🤖1" out of the new one. Output that isn't a bare label now quarantines the passage: text that can redirect the classifier is an injection by demonstration.
- **The detectors fail differently**, so all three run. An injection framed as "an example of an attack" passes both policy models and only Prompt Guard catches it.
- **Attacks that beat the backup layers share one shape:** a correct answer plus something nobody asked for — a "compliance tag", an "audit id". The answer check targets that shape and cuts the backup layers from 10.7% to 2%.
- **What beats every defence is a format request** — "answer in French". A translated answer still answers the question, so the answer check lets it through. Low harm, but real obedience.
- **The input guard cuts user-side attacks from 19% to 4%** without refusing a single genuine question.

<details>
<summary><b>Methodology notes and corrections</b></summary>

- **A corrected undercount.** The success check once missed canaries written with Unicode hyphens (`WHIMBREL‑4406`, U+2011) and canaries from a second attack passage retrieved alongside the target. Re-scoring the recorded answers (`run_guardrails --rescore`, no API calls) raised the weaker rows by up to 2.7 attacks; rows with screening on didn't change.
- **The answer check, measured.** End to end it withheld 18 of 150 answers and took the backup layers from 5.3 to 1.0 of 50; a replay over recorded answers had predicted 0.7. On 665 recorded answers it withheld 59 of 69 successful attacks (59 of 62 leaving out misinformation, which it doesn't judge) and **0 of 311** answers that never saw an attack ([details](../data/guardrails/answer_check.json)).
- **The classifiers aren't deterministic at temperature 0.** The input guard caught 16 then 15 dev attacks across two measurements; the answer check once passed a canary it had caught before. Hence 3 runs per configuration.
- **A leak found and fixed.** A prompt-extraction question once got the model to print `«END PASSAGE 75e69b05` with no closing `»`, which the fence filter required. It now matches unclosed and any-case markers; replayed, that leaves 1 user-side success in 105 attempts and changes none of 1,873 other answers.
- **Checking the BIPIA judge.** 40 distinct answers were labelled blind to the judge's verdicts; it agreed on **40 / 40** (95% CI 91–100%). The labels were made by Claude, a different model family from the judge — an independent check, but weaker than a person's, and most cases are easy to call.
- **BIPIA, unaided.** With no defences the model saw a planted instruction 168 times and obeyed 13: translations, one emoji, and once a scam line whose link survived because `example.com` is an allowlisted documentation domain.

</details>

