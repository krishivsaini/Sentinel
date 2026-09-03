"use client";

import type { ItemResult, RunResult } from "@/lib/data";
import { citedChunkIds, docLabel, f2, kindMeta, kindOf, scoreClass } from "@/lib/format";
import { CaretIcon } from "./icons";
import { Card, PageHead } from "./ui";

const score = (v: number | null) => (v == null ? "—" : v.toFixed(v < 1 && v > -1 ? 2 : 1));

function Row({ item }: { item: ItemResult }) {
  const km = kindMeta[kindOf(item)];
  const cited = citedChunkIds(item);
  return (
    <details className="border-b border-border-soft">
      <summary className="grid grid-cols-[24px_1fr_78px_78px_78px_150px] items-center gap-1 px-[18px] py-[13px] text-[13px]">
        <CaretIcon className="sc-caret text-text-5" />
        <span className="pr-3.5 text-text-1">{item.question}</span>
        <span className={`text-right tabular-nums font-medium ${scoreClass(item.faithfulness)}`}>{f2(item.faithfulness)}</span>
        <span className="text-right tabular-nums text-text-2">{f2(item.answer_relevance)}</span>
        <span className={`text-right tabular-nums ${item.context_recall < 0.5 ? "text-ret-fg" : "text-text-2"}`}>{f2(item.context_recall)}</span>
        <span className="text-right">
          <span className={`rounded-full px-[9px] py-[3px] text-[10px] font-semibold ${km.fg} ${km.bg}`}>{km.label}</span>
        </span>
      </summary>
      <div className="bg-panel-2 px-5 pb-5 pl-[42px] pt-1.5">
        <div className="my-2 text-[10px] font-semibold uppercase tracking-[0.05em] text-text-4">
          Reranked context · top-{item.retrieved.length}
        </div>
        <div className="mb-4 flex flex-col gap-[7px]">
          {item.retrieved.map((c) => {
            const used = cited.has(c.chunk_id.toLowerCase());
            return (
              <div
                key={c.chunk_id}
                className={`rounded-[9px] border px-3 py-2.5 ${used ? "border-cu-bd bg-cu-bg" : "border-border-2 bg-panel"}`}
              >
                <div className="mb-1.5 flex items-center gap-2">
                  <span className="font-mono text-[11px] font-medium text-accent">{docLabel(c.chunk_id)}</span>
                  {used && (
                    <span className="rounded-[10px] bg-cu-chip-bg px-1.5 py-px text-[9px] font-semibold text-cu-chip-fg">cited</span>
                  )}
                  <span className="ml-auto flex gap-[9px] font-mono text-[10px] text-text-3">
                    <span title="dense">d {score(c.dense_score)}</span>
                    <span title="sparse · BM25">s {score(c.sparse_score)}</span>
                    <span title="RRF fused">f {score(c.fused_score)}</span>
                    <span className="font-semibold text-text-1" title="rerank">r {score(c.rerank_score)}</span>
                  </span>
                </div>
                <div className="text-[12px] leading-[1.5] text-text-2">
                  {c.text.replace(/\s+/g, " ").slice(0, 220)}…
                </div>
              </div>
            );
          })}
        </div>
        <div className="mb-[7px] text-[10px] font-semibold uppercase tracking-[0.05em] text-text-4">Grounded answer</div>
        <div className="border-l-2 border-border-2 pl-3.5 text-[13px] leading-[1.6] text-text-1">
          {item.generated_answer}
        </div>
      </div>
    </details>
  );
}

export function Drilldown({ latest }: { latest: RunResult }) {
  return (
    <div className="max-w-[1000px] px-8 pb-12 pt-7">
      <PageHead
        title="Per-question drill-down"
        sub="Each question with its Ragas scores. Expand to see the reranked chunks — carrying all four retrieval scores — and the grounded answer with citations."
        right={
          <div className="font-mono text-[12px] text-text-3">
            run {latest.git_sha} · {latest.per_item.length} items
          </div>
        }
      />
      <Card className="overflow-hidden">
        <div className="grid grid-cols-[24px_1fr_78px_78px_78px_150px] border-b border-border-hair px-[18px] py-[11px] text-[10.5px] font-semibold uppercase tracking-[0.05em] text-text-4">
          <span />
          <span>question</span>
          <span className="text-right">faith</span>
          <span className="text-right">rel</span>
          <span className="text-right">recall</span>
          <span className="text-right">verdict</span>
        </div>
        {latest.per_item.map((item, i) => (
          <Row key={i} item={item} />
        ))}
      </Card>
      <div className="mt-3.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-text-4">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-bad-fg" />
          &lt; 0.60 · counts as a failure
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-full bg-warn-fg" />
          &lt; 0.80 · below the CI gate line
        </span>
        <span>· four retrieval scores per chunk: d dense · s sparse (BM25) · f RRF · r rerank</span>
      </div>
    </div>
  );
}
