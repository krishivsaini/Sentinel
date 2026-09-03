"use client";

import { stageP95, type DashboardData } from "@/lib/data";
import { f2, shortDate } from "@/lib/format";
import { CheckIcon } from "./icons";
import { TrendChart } from "./TrendChart";
import { Card, PageHead, StatCard } from "./ui";

function delta(runs: DashboardData["history"]["runs"], key: "faithfulness" | "answer_relevance" | "context_recall") {
  if (runs.length < 2) return undefined;
  const d = runs[runs.length - 1][key] - runs[runs.length - 2][key];
  return `${d >= 0 ? "+" : ""}${d.toFixed(2)}`;
}

export function Overview({ data }: { data: DashboardData }) {
  const { latest, history, latency } = data;
  const m = latest.means;
  const runs = history.runs;
  const gate = history.faithfulness_threshold;
  const passing = m.faithfulness >= gate;
  const totalP95 =
    (stageP95(latency, "retrieve") ?? 0) +
    (stageP95(latency, "rerank") ?? 0) +
    (stageP95(latency, "generate") ?? 0);
  const recent = runs.slice().reverse();
  // The latency report records which generation model it measured; if that differs from the model
  // the current runs use, the generate stage is stale and must not read as a current number.
  const latencyStale =
    !!latency?.generation_model &&
    !!history.generation_model &&
    latency.generation_model !== history.generation_model;

  return (
    <div className="max-w-[1180px] px-8 pb-12 pt-7">
      <PageHead
        title="Overview"
        sub="Ragas evaluation over a hand-built ground-truth set, wired into CI as a merge gate."
      />

      <div className="mb-[22px] grid grid-cols-4 gap-4">
        <StatCard
          label="Faithfulness"
          value={f2(m.faithfulness)}
          delta={delta(runs, "faithfulness")}
          deltaTone="auto"
          foot={`gate ≥ ${gate.toFixed(2)} · mean of ${latest.per_item.length}`}
        />
        <StatCard
          label="Answer relevance"
          value={f2(m.answer_relevance)}
          delta={delta(runs, "answer_relevance")}
          deltaTone="auto"
          foot={`Ragas judge · ${history.judge_model.split("/").pop()}`}
        />
        <StatCard
          label="Context recall"
          value={f2(m.context_recall)}
          delta={delta(runs, "context_recall")}
          deltaTone="auto"
          foot="retrieval-quality metric"
        />
        <StatCard
          label="p95 latency · total"
          value={totalP95 ? (totalP95 >= 1000 ? `${(totalP95 / 1000).toFixed(2)}s` : `${Math.round(totalP95)}ms`) : "—"}
          delta={latencyStale ? "stale" : "measured"}
          deltaTone="muted"
          // Never present a stale measurement as current: if the latency report was measured on a
          // different generation model than the one now configured, say so on the card itself.
          foot={
            latencyStale
              ? `generate measured on ${latency?.generation_model} · re-measure pending`
              : "per-stage · not blended"
          }
        />
      </div>

      <div className="mb-[22px] grid grid-cols-[1fr_320px] gap-4">
        <Card className="px-5 pb-3.5 pt-[18px]">
          <div className="mb-2.5 flex items-center justify-between">
            <div className="text-[13.5px] font-semibold">Metric trend · by run</div>
            <div className="flex gap-3.5 text-[11px] text-text-2">
              <Legend color="var(--line-faith)" label="faithfulness" />
              <Legend color="var(--line-rel)" label="relevance" />
              <Legend color="var(--line-rec)" label="recall" />
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 border-t-2 border-dashed" style={{ borderColor: "var(--dash)" }} />
                gate {gate.toFixed(2)}
              </span>
            </div>
          </div>
          <TrendChart runs={runs} threshold={gate} />
          {runs.length < 3 && (
            <div className="mt-1 font-mono text-[10px] text-text-4">
              trend fills out as more runs land · {runs.length} run{runs.length === 1 ? "" : "s"} so far
            </div>
          )}
        </Card>

        <Card className="flex flex-col px-5 py-[18px]">
          <div className="mb-3.5 text-[13.5px] font-semibold">CI faithfulness gate</div>
          <div
            className={`mb-3.5 flex items-center gap-2.5 rounded-[10px] border px-3.5 py-3 ${
              passing ? "border-ok-bd bg-ok-bg" : "border-bad-fg/30 bg-bad-bg"
            }`}
          >
            <span className={passing ? "text-ok-fg" : "text-bad-fg"}>
              <CheckIcon size={18} />
            </span>
            <div>
              <div className={`text-[12.5px] font-semibold ${passing ? "text-ok-fg" : "text-bad-fg"}`}>
                {passing ? "Passing" : "Blocked"}
              </div>
              <div className={`font-mono text-[10px] ${passing ? "text-ok-fg/80" : "text-bad-fg/80"}`}>
                {f2(m.faithfulness)} {passing ? "≥" : "<"} {gate.toFixed(2)} · {latest.per_item.length}-item subset
              </div>
            </div>
          </div>
          <div className="text-[12px] leading-[1.6] text-text-2">
            Every pull request runs a Ragas subset. Merges are{" "}
            <b className="text-text-1">blocked when mean faithfulness regresses below {gate.toFixed(2)}</b>. The full eval runs on demand.
          </div>
          <div className="mt-auto border-t border-border-hair pt-3.5 font-mono text-[10.5px] text-text-4">
            last run · {latest.git_sha} · {shortDate(latest.timestamp)}
          </div>
        </Card>
      </div>

      <Card className="overflow-hidden">
        <div className="border-b border-border-hair px-[18px] py-3.5 text-[13.5px] font-semibold">
          Recent eval runs
        </div>
        <div className="grid grid-cols-[130px_90px_1fr_1fr_1fr_120px] border-b border-border-soft px-[18px] py-[9px] text-[10.5px] font-semibold uppercase tracking-[0.05em] text-text-4">
          <span>commit</span><span>date</span><span>faithfulness</span><span>relevance</span><span>recall</span>
          <span className="text-right">gate</span>
        </div>
        {recent.map((r) => (
          <div key={r.run_id} className="grid grid-cols-[130px_90px_1fr_1fr_1fr_120px] items-center border-b border-border-soft px-[18px] py-[11px] text-[12.5px]">
            <span className="font-mono font-medium">{r.git_sha}</span>
            <span className="text-[12px] text-text-3">{shortDate(r.timestamp)}</span>
            <span className={`tabular-nums font-medium ${r.faithfulness < gate ? "text-bad-fg" : "text-text"}`}>{f2(r.faithfulness)}</span>
            <span className="tabular-nums">{f2(r.answer_relevance)}</span>
            <span className="tabular-nums">{f2(r.context_recall)}</span>
            <span className="text-right">
              <span className={`rounded-full px-2 py-0.5 text-[10.5px] font-semibold ${r.gate_pass ? "bg-ok-bg text-ok-fg" : "bg-bad-bg text-bad-fg"}`}>
                {r.gate_pass ? "passed" : "blocked"}
              </span>
            </span>
          </div>
        ))}
      </Card>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span className="h-[2.5px] w-[11px] rounded-sm" style={{ background: color }} />
      {label}
    </span>
  );
}
