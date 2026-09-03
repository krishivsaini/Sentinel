"use client";

import type { History } from "@/lib/data";
import { f2, shortDate } from "@/lib/format";
import { CaretIcon } from "./icons";
import { Card, PageHead } from "./ui";

export function EvalRuns({ history }: { history: History }) {
  const runs = history.runs.slice().reverse();
  const gate = history.faithfulness_threshold;
  return (
    <div className="max-w-[1080px] px-8 pb-12 pt-7">
      <PageHead
        title="Eval Runs"
        sub="Every run is tagged with its git SHA and stored in SQLite, then exported to JSON. Expand a row for failure attribution and coverage."
      />
      <Card className="overflow-hidden">
        <div className="grid grid-cols-[24px_150px_90px_1fr_1fr_1fr_110px] border-b border-border-hair px-[18px] py-[11px] text-[10.5px] font-semibold uppercase tracking-[0.05em] text-text-4">
          <span /><span>commit</span><span>date</span><span>faith</span><span>relevance</span><span>recall</span>
          <span className="text-right">gate</span>
        </div>
        {runs.map((r) => (
          <details key={r.run_id} className="border-b border-border-soft">
            <summary className="grid grid-cols-[24px_150px_90px_1fr_1fr_1fr_110px] items-center px-[18px] py-3 text-[12.5px]">
              <CaretIcon className="sc-caret text-text-5" />
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
            </summary>
            <div className="flex flex-wrap gap-[26px] bg-panel-2 px-[18px] py-[18px] pl-[42px]">
              <Detail label="Attribution">
                <span className="font-semibold text-ret-fg">{r.retrieval_fail}</span> retrieval ·{" "}
                <span className="font-semibold text-gen-fg">{r.generation_fail}</span> generation
              </Detail>
              <Detail label="Items scored">{r.items}</Detail>
              <Detail label="Run id"><span className="font-mono">{r.run_id}</span></Detail>
              <Detail label="Gate">{r.gate_pass ? "passed" : "blocked"} at ≥ {gate.toFixed(2)}</Detail>
            </div>
          </details>
        ))}
      </Card>
      <div className="mt-3.5 flex items-center gap-[7px] text-[11.5px] text-text-4">
        <span className="h-2 w-2 flex-none rounded-full bg-accent" />
        Runs are keyed by git SHA in SQLite (<span className="font-mono text-text-2">eval_runs.db</span>) and exported to JSON the dashboard reads.
      </div>
    </div>
  );
}

function Detail({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-[5px] text-[10px] font-semibold uppercase tracking-[0.05em] text-text-4">{label}</div>
      <div className="text-[12px] text-text-2 tabular-nums">{children}</div>
    </div>
  );
}
