# Sentinel

> **A production RAG service with an evaluation gate in CI.**
>
> Hybrid BM25 + dense retrieval with cross-encoder reranking, served over FastAPI with SSE
> streaming, JWT auth, and rate limiting. Every pull request runs a Ragas evaluation over a
> hand-built ground-truth set; merges are **blocked when faithfulness regresses below
> threshold**. The dashboard attributes every low-scoring answer to either a retrieval failure
> or a generation failure.

**🚧 Status: in development (Phase 0 — scaffold).** This README will lead with real, measured
numbers on completion. Per the project's honest-metrics discipline, no metrics are shown until
the eval actually produces them — see [`SENTINEL_BUILD_PLAN.md`](SENTINEL_BUILD_PLAN.md) §18.

## Planning docs

| Doc | Purpose |
|---|---|
| [docs/requirement.md](docs/requirement.md) | Testable functional + non-functional requirements, acceptance criteria |
| [docs/product_design.md](docs/product_design.md) | Product thesis, personas, surfaces, dashboard IA |
| [docs/architecture.md](docs/architecture.md) | System design, data flow, tech-stack rationale, deployment |
| [docs/implementation_plan.md](docs/implementation_plan.md) | Phased build plan, dependency graph, risk register |
| [SENTINEL_BUILD_PLAN.md](SENTINEL_BUILD_PLAN.md) | Source of truth (build brief) |

## Corpus

**IETF RFC web-protocol stack** (HTTP / URI / TLS / OAuth / JWT / cookies / TCP / DNS) — chosen
for its strong exact-match terms (RFC numbers, section refs, header/status-code tokens) that
justify hybrid retrieval over pure dense, and its clean license: the IETF Trust permits
whole-RFC reproduction; the pipeline quotes verbatim and creates no derivative works.

## Reproducing (target workflow — commands land as stages ship)

```bash
git clone <repo> && cd Sentinel
uv sync                          # pinned via uv.lock
cp .env.example .env             # set GOOGLE_API_KEY + SENTINEL_JWT_SECRET
python -m sentinel.ingest        # build FAISS + BM25 indexes  (Phase 1)
python -m sentinel.serve         # FastAPI service on :8000     (Phase 2)
python -m sentinel.eval.run_eval # full Ragas eval             (Phase 3)
```

## License

MIT — see [LICENSE](LICENSE).
