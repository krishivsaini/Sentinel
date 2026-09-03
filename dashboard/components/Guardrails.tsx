"use client";

import { ShieldIcon } from "./icons";
import { Card, PageHead } from "./ui";

const planned = [
  {
    title: "Adversarial-safety gate",
    body: "A second CI gate alongside faithfulness: measure prompt-injection success rate over a labeled adversarial corpus and block merges when it regresses.",
  },
  {
    title: "Injected-instruction defense",
    body: "Treat retrieved content strictly as data, never instructions — and prove it with cases where a chunk contains “ignore previous instructions”.",
  },
  {
    title: "PII detection on output",
    body: "Flag/redact emails, phone numbers, and secrets before returning, measured for recall/precision on a labeled subset.",
  },
  {
    title: "Ungrounded-citation rejection",
    body: "Reject an answer whose citations reference chunk IDs never retrieved (the generation layer already filters these — this surfaces the count).",
  },
];

export function Guardrails() {
  return (
    <div className="max-w-[1080px] px-8 pb-12 pt-7">
      <PageHead
        title="Guardrails"
        sub="Adversarial safety was out of scope for the core build. This view is a roadmap — it deliberately shows no metrics, because there is no implementation behind them yet."
        right={
          <span className="rounded-full bg-gen-bg px-2 py-0.5 text-[10.5px] font-semibold text-gen-fg">
            v2 · roadmap
          </span>
        }
      />

      <div className="mb-5 flex items-center gap-2.5 rounded-[11px] border border-border-2 bg-panel-2 px-4 py-3 card-shadow">
        <span className="text-text-3">
          <ShieldIcon size={17} />
        </span>
        <div className="text-[12.5px] text-text-2">
          <b className="text-text-1">Not implemented.</b> The rest of this dashboard shows only real,
          measured numbers — so this page stays a plan, not a set of invented metrics.
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {planned.map((p) => (
          <Card key={p.title} className="px-5 py-[18px]">
            <div className="mb-2 flex items-center gap-2">
              <span className="h-[7px] w-[7px] rounded-full bg-gen-solid" />
              <div className="text-[13px] font-semibold">{p.title}</div>
              <span className="ml-auto rounded-full bg-chip px-2 py-0.5 text-[10px] font-semibold text-text-4">
                planned
              </span>
            </div>
            <div className="text-[12.5px] leading-[1.55] text-text-2">{p.body}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}
