"use client";

import type { RunResult } from "@/lib/data";
import { f2 } from "@/lib/format";
import { Card, PageHead } from "./ui";

export function Attribution({ latest }: { latest: RunResult }) {
  const total = latest.per_item.length;
  const rFail = latest.per_item.filter((i) => i.attribution === "retrieval_fail");
  const gFail = latest.per_item.filter((i) => i.attribution === "generation_fail");
  const failing = rFail.length + gFail.length;
  const rPct = failing ? (rFail.length / failing) * 100 : 0;
  const gPct = failing ? (gFail.length / failing) * 100 : 0;

  return (
    <div className="max-w-[1080px] px-8 pb-12 pt-7">
      <PageHead
        title="Failure attribution"
        sub="Every low-scoring answer is classified: was the right context not retrieved (retrieval failure), or was it retrieved but the answer wrong or unfaithful (generation failure)?"
      />

      <div className="mb-5 grid grid-cols-2 gap-4">
        <Card className="px-5 py-[18px]">
          <div className="text-[11.5px] font-medium text-text-3">Retrieval failures</div>
          <div className="mt-2 flex items-baseline gap-2.5">
            <span className="text-[30px] font-semibold tabular-nums text-ret-fg">{rFail.length}</span>
            <span className="text-[12px] text-text-4">of {total} · context recall &lt; 0.50</span>
          </div>
        </Card>
        <Card className="px-5 py-[18px]">
          <div className="text-[11.5px] font-medium text-text-3">Generation failures</div>
          <div className="mt-2 flex items-baseline gap-2.5">
            <span className="text-[30px] font-semibold tabular-nums text-gen-fg">{gFail.length}</span>
            <span className="text-[12px] text-text-4">of {total} · recalled but unfaithful</span>
          </div>
        </Card>
      </div>

      <Card className="mb-5 px-5 pb-5 pt-[18px]">
        <div className="mb-3.5 text-[13.5px] font-semibold">Failure split · run {latest.git_sha}</div>
        {failing ? (
          <div className="mb-2.5 flex h-[34px] overflow-hidden rounded-lg">
            {rFail.length > 0 && (
              <div className="flex items-center pl-3 text-[12px] font-semibold text-white" style={{ width: `${rPct}%`, background: "var(--ret-solid)" }}>
                Retrieval · {rFail.length}
              </div>
            )}
            {gFail.length > 0 && (
              <div className="flex items-center pl-3 text-[12px] font-semibold text-white" style={{ width: `${gPct}%`, background: "var(--gen-solid)" }}>
                Generation · {gFail.length}
              </div>
            )}
          </div>
        ) : (
          <div className="mb-2.5 flex h-[34px] items-center justify-center rounded-lg bg-ok-bg text-[12px] font-semibold text-ok-fg">
            No failures this run
          </div>
        )}
        <div className="text-[12px] text-text-3">
          {failing} of {total} items scored below threshold.{" "}
          {gFail.length >= rFail.length
            ? "Generation failures dominate — the answer drifted from retrieved context on a small number of items."
            : "Retrieval failures dominate — the reranker recall is the lever with the most headroom."}
        </div>
      </Card>

      <div className="grid grid-cols-2 gap-4">
        <FailList title="Retrieval failures" dot="var(--ret-solid)" items={rFail} render={(i) => `recall ${f2(i.context_recall)} · relevant chunk not in top-50`} tone="text-ret-fg" />
        <FailList title="Generation failures" dot="var(--gen-solid)" items={gFail} render={(i) => `recall ${f2(i.context_recall)} · faith ${f2(i.faithfulness)} · answer drifted from context`} tone="text-gen-fg" />
      </div>
    </div>
  );
}

function FailList({
  title,
  dot,
  items,
  render,
  tone,
}: {
  title: string;
  dot: string;
  items: RunResult["per_item"];
  render: (i: RunResult["per_item"][number]) => string;
  tone: string;
}) {
  return (
    <Card className="overflow-hidden">
      <div className="flex items-center gap-2 border-b border-border-hair px-[18px] py-3">
        <span className="h-[9px] w-[9px] rounded-sm" style={{ background: dot }} />
        <span className="text-[13px] font-semibold">{title}</span>
      </div>
      {items.length === 0 ? (
        <div className="px-[18px] py-6 text-center text-[12px] text-text-4">None this run</div>
      ) : (
        items.map((i, idx) => (
          <div key={idx} className="border-b border-border-soft px-[18px] py-3">
            <div className="mb-[5px] text-[12.5px] leading-[1.45] text-text-1">{i.question}</div>
            <div className={`font-mono text-[10.5px] ${tone}`}>{render(i)}</div>
          </div>
        ))
      )}
    </Card>
  );
}
