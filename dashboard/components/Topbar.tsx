"use client";

import { stageP95, type LatencyReport, type RunResult } from "@/lib/data";
import { CheckIcon, ChevronDown } from "./icons";

function ms(v: number | null): string {
  if (v == null) return "—";
  return v >= 1000 ? `${(v / 1000).toFixed(2)}s` : `${Math.round(v)}ms`;
}

export function Topbar({
  latest,
  latency,
  threshold,
}: {
  latest: RunResult;
  latency: LatencyReport | null;
  threshold: number;
}) {
  const passing = latest.means.faithfulness >= threshold;
  return (
    <header className="flex h-[57px] flex-none items-center gap-3.5 border-b border-border bg-panel px-6">
      <div className="flex items-center gap-2 rounded-lg border border-border-2 bg-panel-3 px-2.5 py-[5px]">
        <span className="h-1.5 w-1.5 rounded-full bg-ok-fg" />
        <span className="text-[11px] text-text-3">run</span>
        <span className="font-mono text-[12px] font-medium">{latest.git_sha}</span>
        <ChevronDown className="text-text-4" />
      </div>

      <div
        className={`flex items-center gap-[7px] rounded-lg border px-[11px] py-[5px] ${
          passing ? "border-ok-bd bg-ok-bg" : "border-bad-fg/30 bg-bad-bg"
        }`}
      >
        <span className={passing ? "text-ok-fg" : "text-bad-fg"}>
          <CheckIcon />
        </span>
        <span
          className={`text-[11.5px] font-semibold ${passing ? "text-ok-fg" : "text-bad-fg"}`}
        >
          faithfulness gate · {passing ? "passing" : "blocked"}
        </span>
      </div>

      <div className="ml-auto flex items-center gap-4 font-mono text-[11px] text-text-3">
        <span>
          retrieve{" "}
          <b className="font-semibold text-text-2">{ms(stageP95(latency, "retrieve"))}</b>
        </span>
        <span className="text-border-2">·</span>
        <span>
          rerank <b className="font-semibold text-text-2">{ms(stageP95(latency, "rerank"))}</b>
        </span>
        <span className="text-border-2">·</span>
        <span>
          generate{" "}
          <b className="font-semibold text-text-2">{ms(stageP95(latency, "generate"))}</b>
        </span>
        <span className="text-border-2">·</span>
        <span>p95</span>
      </div>
    </header>
  );
}
