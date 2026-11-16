"use client";

import type { OpsReport } from "@/lib/data";
import { Card, PageHead, SectionLabel, StatCard } from "./ui";

// Every number here comes from data/ops_report.json (scripts/ops_report.py), which summarises
// the live service's own structured logs. Nothing is computed in the UI beyond formatting.

const STAGE_LABEL: Record<string, string> = {
  retrieve: "Retrieve — dense + BM25 + RRF",
  rerank: "Rerank — cross-encoder, 50 candidates",
  generate: "Generate — gpt-oss-20b",
  guard: "Guards — input, redaction, output",
  answer_check: "Answer check",
  first_token: "First word reaches the reader",
  request: "Whole request",
};
const OUTCOME_LABEL: Record<string, string> = {
  answered: "Answered",
  abstained: "Abstained (not in the corpus)",
  withheld: "Withheld by the answer check",
  refused: "Refused by the input guard",
  budget_exhausted: "Turned away: daily budget used",
};

const ms = (v: number | undefined) =>
  v === undefined ? "—" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${Math.round(v)} ms`;
const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(0)}%`);
const day = (iso: string | null) => (iso ? iso.slice(0, 10) : "—");

export function Operations({ report }: { report: OpsReport | null }) {
  if (!report) {
    return (
      <div className="max-w-[1080px] px-8 pb-12 pt-7">
        <PageHead title="Operations" sub="The live service, from its own logs." />
        <div className="rounded-xl border border-dashed border-input-bd p-11 text-center text-[13px] text-text-4">
          No operations report yet. Run{" "}
          <span className="font-mono text-text-3">python scripts/ops_report.py --gcloud --project …</span>{" "}
          then <span className="font-mono text-text-3">npm run sync-data</span>.
        </div>
      </div>
    );
  }
  const stage = Object.fromEntries(report.latency.map((s) => [s.stage, s]));
  const answered = report.outcomes.answered;
  const e2e = stage.request ?? stage.rerank;
  const fb = report.feedback;
  const rated = fb.up + fb.down;

  return (
    <div className="max-w-[1080px] px-8 pb-12 pt-7">
      <PageHead
        title="Operations"
        sub={`The live Cloud Run service, summarised from its structured logs · ${report.requests} requests, ${day(report.window.from)} → ${day(report.window.to)}`}
      />

      <div className="mb-6 grid grid-cols-4 gap-3">
        <StatCard
          label="Requests"
          value={String(report.requests)}
          delta={answered ? `${pct(answered.rate)} answered` : undefined}
          deltaTone="muted"
          foot={report.source}
        />
        <StatCard
          label={stage.request ? "Whole request · p95" : "Rerank · p95"}
          value={ms(e2e?.p95_ms)}
          delta={e2e ? `p50 ${ms(e2e.p50_ms)}` : undefined}
          deltaTone="muted"
          foot={`${e2e?.n ?? 0} samples · inside the service`}
        />
        <StatCard
          label="Groq cost per request"
          value={report.cost_usd.per_request_mean == null ? "—" : `$${report.cost_usd.per_request_mean.toFixed(4)}`}
          delta={report.cost_usd.requests_with_cost ? `$${report.cost_usd.total.toFixed(3)} total` : undefined}
          deltaTone="muted"
          foot={`${report.cost_usd.requests_with_cost} requests metered`}
        />
        <StatCard
          label="Reader feedback"
          value={rated ? `${fb.up} 👍 · ${fb.down} 👎` : "—"}
          deltaTone="muted"
          foot={rated ? `${rated} answers rated` : "no ratings yet"}
        />
      </div>

      <div className="grid grid-cols-[1fr_320px] items-start gap-4">
        <Card className="overflow-hidden">
          <div className="border-b border-border-hair px-[18px] py-3 text-[12.5px] font-semibold">
            Latency per stage
          </div>
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="text-left text-[10.5px] uppercase tracking-[0.04em] text-text-4">
                <th className="px-[18px] py-2 font-semibold">Stage</th>
                <th className="px-2 py-2 text-right font-semibold">n</th>
                <th className="px-2 py-2 text-right font-semibold">p50</th>
                <th className="px-2 py-2 text-right font-semibold">p95</th>
                <th className="px-[18px] py-2 text-right font-semibold">p99</th>
              </tr>
            </thead>
            <tbody className="tabular-nums">
              {report.latency.map((s) => (
                <tr key={s.stage} className="border-t border-border-soft">
                  <td className="px-[18px] py-2 text-text-2">{STAGE_LABEL[s.stage] ?? s.stage}</td>
                  <td className="px-2 py-2 text-right font-mono text-text-4">{s.n}</td>
                  <td className="px-2 py-2 text-right font-mono">{ms(s.p50_ms)}</td>
                  <td className="px-2 py-2 text-right font-mono font-semibold">{ms(s.p95_ms)}</td>
                  <td className="px-[18px] py-2 text-right font-mono text-text-3">{ms(s.p99_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <div className="flex flex-col gap-4">
          <Card className="px-4 py-3.5">
            <SectionLabel>Outcomes</SectionLabel>
            {Object.entries(report.outcomes).map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between py-1 text-[12.5px]">
                <span className="text-text-2">{OUTCOME_LABEL[k] ?? k}</span>
                <span className="font-mono tabular-nums text-text-1">
                  {v.n} <span className="text-text-4">· {pct(v.rate)}</span>
                </span>
              </div>
            ))}
          </Card>
          {Object.keys(fb.by_outcome).length > 0 && (
            <Card className="px-4 py-3.5">
              <SectionLabel>Ratings by outcome</SectionLabel>
              {Object.entries(fb.by_outcome).map(([k, v]) => (
                <div key={k} className="flex items-baseline justify-between py-1 text-[12.5px]">
                  <span className="text-text-2">{OUTCOME_LABEL[k] ?? k}</span>
                  <span className="font-mono tabular-nums">
                    {v.up ?? 0} 👍 · {v.down ?? 0} 👎
                  </span>
                </div>
              ))}
            </Card>
          )}
        </div>
      </div>

      {report.daily.length > 1 && (
        <Card className="mt-4 overflow-hidden">
          <div className="border-b border-border-hair px-[18px] py-3 text-[12.5px] font-semibold">By day</div>
          <table className="w-full text-[12.5px] tabular-nums">
            <tbody>
              {report.daily.map((d) => (
                <tr key={d.date} className="border-t border-border-soft first:border-t-0">
                  <td className="px-[18px] py-2 font-mono text-text-2">{d.date}</td>
                  <td className="px-2 py-2 text-right font-mono">{d.requests} req</td>
                  <td className="px-2 py-2 text-right font-mono">${d.cost_usd.toFixed(4)}</td>
                  <td className="px-[18px] py-2 text-right font-mono text-text-3">
                    p95 {d.request_p95_ms == null ? "—" : ms(d.request_p95_ms)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      <p className="mt-4 text-[11.5px] leading-[1.6] text-text-4">
        Timings are measured inside the service, so they leave out network time to the reader. Cost is
        what this service's Groq calls were metered at for each request. Older log entries predate
        the end-to-end and cost fields, so those rows can have fewer samples.
      </p>
    </div>
  );
}
