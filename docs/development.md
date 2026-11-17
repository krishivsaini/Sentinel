# Development

[← README](../README.md) · [Results](results.md) · [Guardrails](guardrails.md) · [Design](design.md) · [Development](development.md)

Running Sentinel locally: building the indexes, serving, evaluating, the optional Postgres and Redis backends, configuration, and the repository layout. Deploying to Cloud Run is in [deploy/README.md](../deploy/README.md).


Python 3.11+ with [`uv`](https://docs.astral.sh/uv/) and a [Groq API key](https://console.groq.com/keys); Node 18+ for the dashboard.

```bash
git clone https://github.com/krishivsaini/Sentinel.git && cd Sentinel
uv sync                                  # exact versions pinned in uv.lock
cp .env.example .env                     # set GROQ_API_KEY and SENTINEL_JWT_SECRET
```

**Build the indexes** — local, no API key needed (screening verdicts are committed):

```bash
uv run python -m sentinel.ingest                 # the service index
uv run python -m sentinel.ingest --adversarial   # + the attack corpus, for the guardrail eval
```

**Keep the corpus current** — the RFC Editor's index says what changed; adding an RFC is a person's call:

```bash
python scripts/watch_rfcs.py                         # standing changes + candidate RFCs since the snapshot
python scripts/watch_rfcs.py add 9846 --cluster tls  # or: skip 9846 --reason "..."
python scripts/fetch_corpus.py && python scripts/fetch_rfc_status.py
uv run python -m sentinel.ingest --incremental       # embeds + screens only the new RFC
```

**Retrieve, serve, ask:**

```bash
uv run python -m sentinel.retrieve "What does the invalid_grant error mean?"   # top-5, all four scores
uv run python -m sentinel.serve                                                # http://localhost:8000
```

```bash
TOKEN=$(curl -s -X POST localhost:8000/token \
  -d 'username=demo&password=change-me' | jq -r .access_token)

curl -N -X POST localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"question": "What does the HTTP 429 status code indicate?"}'
```

Named accounts with their own scopes (the demo account gets `public` only):

```bash
uv run python scripts/users.py add alice --scopes public       # prompts for a password; stores a hash
SENTINEL_USERS_FILE=data/users.json uv run python -m sentinel.serve
curl -s -X POST localhost:8000/token -d 'username=alice&password=…&scope=public'   # scope is optional
```

**Dense search in Postgres (optional)** — local pgvector in Docker; FAISS stays the default:

```bash
uv sync --extra pgvector
docker compose -f deploy/pgvector/compose.yaml up -d      # Postgres 17 + pgvector, localhost:5433
uv run python -m sentinel.vectorstore sync                # copies indexes/ in; later syncs write only changes
SENTINEL_VECTOR_STORE=pgvector uv run python -m sentinel.serve
uv run python scripts/compare_vector_stores.py            # FAISS vs pgvector on every question, no API calls
```

**Semantic cache in Redis (optional)** — local Redis in Docker; the in-memory cache stays the default:

```bash
docker compose -f deploy/redis/compose.yaml up -d         # Redis 8, localhost:6390, no persistence
SENTINEL_CACHE_BACKEND=redis uv run python -m sentinel.serve
uv run pytest tests/test_cache.py                         # every cache test, in memory and in Redis
```

**Evaluate** — paid runs accept `--max-spend USD`:

```bash
uv run python -m sentinel.eval.run_eval                    # all 48 questions (resumable)
uv run python -m sentinel.eval.run_eval --ci               # the faithfulness gate
uv run python -m sentinel.eval.run_eval --write-baseline 4 # refresh main's baseline
uv run python -m sentinel.eval.run_guardrails              # every guardrail, 3 runs each
uv run python -m sentinel.eval.run_guardrails --ci         # the injection gate
uv run python -m sentinel.eval.run_bipia --max-spend 0.60  # the independent attack set
uv run python -m sentinel.eval.run_multistep --judge --max-spend 0.25  # multi-part questions, planned vs not
uv run pytest                                              # tests (live-LLM tests skipped by default)
```

**Dashboard** — ships with a committed data snapshot:

```bash
cd dashboard && npm install && npm run sync-data && npm run dev   # http://localhost:3000
```

## Free-tier Groq and configuration

A free Groq key runs everything, slowly — about two CI-sized evals a day. Set `SENTINEL_EVAL_ITEM_PAUSE_SECONDS=25`, `SENTINEL_CI_EVAL_SUBSET_SIZE=10` and `SENTINEL_CI_MIN_ITEMS=6`; [`.env.example`](../.env.example) explains each.

Everything tunable lives in [`config.py`](../sentinel/config.py) and can be overridden with `SENTINEL_*` variables.

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes | generation, the judge and the guardrail classifiers |
| `SENTINEL_JWT_SECRET` | yes | signs API tokens |
| `SENTINEL_DEMO_USER` / `SENTINEL_DEMO_PASSWORD` | no | the shared demo account (default scope only) |
| `SENTINEL_USERS_FILE` | no | named accounts: scrypt hashes and scopes, managed with `scripts/users.py` |
| `SENTINEL_SEMANTIC_CACHE` / `SENTINEL_CACHE_THRESHOLD` | no | `false` turns off answer reuse; the threshold defaults to the measured 0.98 |
| `SENTINEL_CACHE_BACKEND` / `SENTINEL_REDIS_URL` | no | `memory` (default) or `redis`, and the Redis to use (default: the local container; `rediss://` for TLS) |
| `SENTINEL_QUERY_PLANNING` | no | `false` turns off splitting multi-part questions into separate searches (on by default) |
| `SENTINEL_VECTOR_STORE` / `SENTINEL_DATABASE_URL` | no | `faiss` (default) or `pgvector`, and the Postgres to use (default: the local container) |
| `SENTINEL_SPEND_LIMIT_USD` | no | stop any run once its metered spend reaches this |
| `SENTINEL_TORCH_DEVICE` | no | `cpu` (default), `mps` or `cuda` for the local models |
| `SENTINEL_INPUT_GUARD` · `_INJECTION_SCREENING` · `_PROMPT_HARDENING` · `_REDACT_CONTEXT_PII` · `_OUTPUT_GUARD` · `_ANSWER_CHECK` | no | switch a guardrail off — the guardrail eval does exactly that |
| `OPENAI_API_KEY` / `DEEPSEEK_API_KEY` / `NVIDIA_API_KEY` / `GOOGLE_API_KEY` | no | alternate providers |


## Repository layout

```
Sentinel/
├── sentinel/
│   ├── config.py            # single source of truth: models, thresholds, paths
│   ├── schema.py            # Pydantic contracts shared by every stage
│   ├── ingest.py            # load → clean → chunk → embed → FAISS + BM25
│   ├── retrieve.py          # dense + sparse → RRF → cross-encoder rerank
│   ├── vectorstore.py       # dense search in Postgres + pgvector; incremental sync from an index
│   ├── plan.py              # query planner: splits a multi-part question into searches
│   ├── cache.py             # semantic cache: exact-term check + embedding similarity, scoped; in memory or Redis
│   ├── generate.py          # grounded, cited generation with abstention
│   ├── llm.py               # provider-agnostic chat-model factory
│   ├── serve.py             # FastAPI: JWT scopes, rate limiting, SSE, guardrails
│   ├── users.py             # accounts: the demo account + named ones (scrypt hashes, scopes)
│   ├── retry.py             # shared retry policy for every rate-limited API call
│   ├── spend.py             # meters every paid LLM call; caps a run in dollars
│   ├── logging_config.py    # structured logging + per-stage timers
│   ├── guardrails/
│   │   ├── input.py         # screens the user's question before retrieval
│   │   ├── injection.py     # index-time injection screening (Prompt Guard 2 + two policies)
│   │   ├── pii.py           # PII detection + redaction
│   │   ├── output.py        # streaming output guard: links, images, PII, citations
│   │   └── answer.py        # withholds answers that carried out a planted instruction
│   └── eval/
│       ├── run_eval.py      # Ragas pipeline, checkpointed + resumable, faithfulness gate
│       ├── run_guardrails.py# attack evals, detector + PII metrics, injection gate
│       ├── run_bipia.py     # the independent BIPIA attack set, judged, spend-capped
│       ├── run_multistep.py # multi-part questions: part coverage, over-splitting, judged answers
│       ├── stats.py         # exact confidence intervals for small samples
│       ├── attribution.py   # retrieval- vs generation-failure classifier
│       └── store.py         # SQLite persistence + JSON export
├── data/
│   ├── corpus/              # the 48 RFCs, verbatim
│   ├── corpus_manifest.json # pinned RFC list + clusters
│   ├── ground_truth.jsonl   # hand-built eval set
│   ├── multistep_questions.jsonl # 16 two-part questions, gold chunks + evidence per part
│   ├── multistep_eval.json  # single search vs planned, per question
│   ├── cache_pairs.jsonl    # paraphrases and near-misses of ground-truth questions
│   ├── cache_eval.json      # cache threshold sweep
│   ├── cache_live_check.json # the cache on the deployed service (in memory)
│   ├── cache_live_check_redis.json # ... with the cache in Redis, including after a cold start
│   ├── ground_truth_audit.md# second-pass review + judge calibration
│   ├── faithfulness_baseline.json # main's per-question scores, for the relative gate
│   ├── retrieval_ablation.json # dense / BM25 / fusion / rerank, per question
│   ├── vector_store_comparison.json # FAISS vs pgvector, per question
│   ├── latency_report.json  # per-stage p50/p95/p99
│   ├── adversarial/         # 50 labelled poisoned passages + questions (dev / held-out / adaptive)
│   ├── bipia/               # 75 BIPIA attacks planted in RFC passages (MIT, pinned)
│   └── guardrails/          # screening verdicts, PII labels, user-side prompts, the report
├── scripts/
│   ├── fetch_corpus.py              # download the corpus
│   ├── watch_rfcs.py                # what changed at the RFC Editor since the snapshot
│   ├── build_ground_truth.py        # surface candidate passages (--assist)
│   ├── assemble_ground_truth.py     # the hand-authored answers + gold passages
│   ├── judge_calibration.py         # correlate the judge with hand labels
│   ├── probe_judge.py               # does a candidate judge produce parseable output?
│   ├── compare_judge_hosts.py       # same judge, two providers, identical inputs
│   ├── retrieval_ablation.py        # does each retrieval stage earn its place?
│   ├── compare_vector_stores.py     # FAISS vs pgvector: same results under the same filters?
│   ├── cache_eval.py                # semantic cache threshold: paraphrases vs near-misses
│   ├── cache_live_check.py          # semantic cache on the deployed service: hit, reword, near-miss
│   ├── answer_check_eval.py         # the answer check on every recorded answer
│   ├── build_bipia.py               # fetch BIPIA (pinned) and plant it in RFC passages
│   ├── bench_latency.py             # per-stage p50/p95/p99
│   ├── users.py                     # add / disable / re-scope named accounts
│   └── export_dashboard_data.py     # cross-run history for the dashboard
├── dashboard/               # Next.js + Tailwind eval dashboard
├── tests/                   # retrieval, scope, guardrails, generation, service, eval, spend
├── ARCHITECTURE.md          # the system as built
├── docs/                    # requirements, product design, original architecture, plan
└── .github/workflows/eval-gate.yml
```


