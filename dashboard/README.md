# Dashboard (placeholder)

Read-only Next.js eval dashboard (§14) — scaffolded in **Phase 5, Day 12**. Tailwind core
utilities only; no component libraries. Deploys to Vercel; consumes the JSON exports written
to `dashboard/data/` by `sentinel/eval/store.py` (no live DB connection).

Three views:
- **Trends** — faithfulness / relevance / recall over eval runs (by git SHA / time), with the
  gate-threshold line drawn.
- **Per-question drill-down** — each question's scores, retrieved chunks (all four scores),
  and the cited answer.
- **Failure attribution** — retrieval-failure vs generation-failure counts per run (the
  README screenshot).
