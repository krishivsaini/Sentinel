# Sentinel dashboard

Read-only Next.js + Tailwind dashboard for the eval pipeline (§14). It is a **pure JSON
consumer** (FR-D4): it reads static exports from `public/data/` and never opens a DB connection
or calls an LLM. Every number it shows is measured — see *Honesty* below.

```bash
npm install
npm run sync-data     # copy the latest eval exports from ../data into public/data
npm run dev           # http://localhost:3000
```

## Views

| View | What it shows | Source |
|---|---|---|
| **Overview** | metric cards, metric-trend chart vs the gate line, CI gate status, recent runs | `history.json` + `latest.json` |
| **Eval Runs** | every run keyed by git SHA, expandable for attribution + coverage | `history.json` |
| **Drill-down** | per-question scores; expand for the reranked chunks with **all four retrieval scores** (dense / sparse / RRF / rerank) and the grounded answer, with cited chunks highlighted | `latest.json` |
| **Failure attribution** | retrieval-failure vs generation-failure split and the offending questions — the README screenshot | `latest.json` |
| **Query playground** | live streaming answer from the running API (`/token` → JWT → `/query` SSE) | the API, optional |
| **Guardrails** | **roadmap only** — adversarial safety is not implemented, so this page deliberately shows no metrics | — |

Views are hash-routed (`#drill`, `#attr`, …), so any view is linkable.

## Data

`sentinel/eval/store.py` writes `eval_<run>.json` + `latest.json` per run;
`scripts/export_dashboard_data.py` writes `history.json` (all runs, for the trend).
`npm run sync-data` copies those plus `latency_report.json` into `public/data/`.

`dashboard/data/` is gitignored (regenerable), but **`dashboard/public/data/` is committed** so a
fresh clone or a Vercel build renders real numbers with no key and no local eval run.

## Query playground (optional)

Needs the API running:

```bash
uvicorn sentinel.serve:app --reload          # in the repo root
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Without it the playground shows a "no API configured" state; the four data views are unaffected.

## Honesty

- Numbers come from real eval runs — nothing is mocked or hand-tuned for the screenshot.
- A negative delta never renders green; scores are colour-coded < 0.60 (failure) / < 0.80 (below
  the gate line) so colour can't contradict the attribution verdict.
- The latency strip is the measured per-stage p95 from `latency_report.json` — per stage, never
  blended (NFR-3).
- Guardrails is a labelled roadmap, not a metrics page, because there is no implementation
  behind it yet.

Tailwind core utilities only; no component libraries. Colours are CSS variables
(`app/globals.css`) so light/dark themes swap the whole palette.
