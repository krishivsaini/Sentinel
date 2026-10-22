"use client";

import type { AttackBlock, GuardrailsReport, PiiSource, Rate } from "@/lib/data";
import { shortDate } from "@/lib/format";
import { ShieldIcon } from "./icons";
import { Card, PageHead, StatCard } from "./ui";

// Every number here comes from data/guardrails/report.json (sentinel.eval.run_guardrails).
// Nothing is computed or rounded up in the UI beyond formatting.

const CONFIG_LABEL: Record<string, string> = {
  none: "No defence",
  screening: "Screening only",
  hardening: "Prompt hardening only",
  output_guard: "Output guard only",
  "hardening+output": "Hardening + output guard",
  all: "All defences",
};
const CONFIG_NOTE: Record<string, string> = {
  "hardening+output": "what an attack faces when screening misses it",
  all: "what ships",
};

function pct(v: number | null | undefined): string {
  return v == null ? "—" : `${(v * 100).toFixed(v === 0 || v === 1 ? 0 : 1)}%`;
}

function frac(r: Rate): string {
  return `${r.flagged} / ${r.of}`;
}

function detName(id: string): string {
  if (id === "union") return "Either (quarantine rule)";
  if (id.startsWith("promptguard")) return "Llama Prompt Guard 2 · 86M";
  if (id.startsWith("safeguard")) return "gpt-oss-safeguard-20b · policy";
  return id;
}

function Head({ cols, children }: { cols: string; children: React.ReactNode }) {
  return (
    <div
      className={`grid ${cols} border-b border-border-hair px-[18px] py-[11px] text-[10.5px] font-semibold uppercase tracking-[0.05em] text-text-4`}
    >
      {children}
    </div>
  );
}

function Attempts({ b }: { b: AttackBlock }) {
  const [lo, hi] = b.succeeded_range;
  const tone = b.succeeded === 0 ? "text-ok-fg" : "text-bad-fg";
  return (
    <span className="tabular-nums">
      <span className={`font-semibold ${tone}`}>{Number.isInteger(b.succeeded) ? b.succeeded : b.succeeded.toFixed(1)}</span>
      <span className="text-text-4"> / {b.attacks}</span>
      {lo !== hi && <span className="ml-1.5 text-[11px] text-text-4">({lo}–{hi})</span>}
    </span>
  );
}

function PiiTable({ title, src, note }: { title: string; src: PiiSource; note: string }) {
  const kinds = Object.entries(src.per_kind);
  return (
    <Card className="overflow-hidden">
      <div className="flex items-baseline gap-2 px-[18px] pb-1 pt-4">
        <div className="text-[13px] font-semibold">{title}</div>
        <div className="ml-auto text-[11.5px] tabular-nums text-text-3">
          precision <b className="text-text-1">{pct(src.precision)}</b> · recall{" "}
          <b className="text-text-1">{pct(src.recall)}</b>
        </div>
      </div>
      <div className="px-[18px] pb-3 text-[11.5px] text-text-3">{note}</div>
      <Head cols="grid-cols-[1fr_60px_60px_60px_80px_70px]">
        <span>kind</span><span>tp</span><span>fp</span><span>fn</span><span>precision</span><span>recall</span>
      </Head>
      {kinds.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[1fr_60px_60px_60px_80px_70px] border-b border-border-soft px-[18px] py-2.5 text-[12.5px] tabular-nums">
          <span className="font-mono text-[12px]">{k}</span>
          <span>{v.tp}</span>
          <span className={v.fp ? "text-warn-fg" : ""}>{v.fp}</span>
          <span className={v.fn ? "text-bad-fg" : ""}>{v.fn}</span>
          <span>{pct(v.precision)}</span>
          <span>{pct(v.recall)}</span>
        </div>
      ))}
    </Card>
  );
}

function Missing() {
  return (
    <div className="max-w-[1080px] px-8 pb-12 pt-7">
      <PageHead title="Guardrails" sub="No guardrail report has been exported yet." />
      <Card className="px-5 py-4 text-[12.5px] text-text-2">
        Run <span className="font-mono">python -m sentinel.eval.run_guardrails</span>, then{" "}
        <span className="font-mono">npm run sync-data</span>. This page shows only measured
        numbers, so it stays empty until there are some.
      </Card>
    </div>
  );
}

export function Guardrails({ report }: { report: GuardrailsReport | null }) {
  if (!report) return <Missing />;
  const att = report.injection.attacks.summary;
  const det = report.injection.detectors;
  const none = att.none?.all;
  const all = att.all?.all;
  const ho = det.recall.heldout.union;
  const fp = det.false_positives.union;
  const order = ["none", "screening", "hardening", "output_guard", "hardening+output", "all"].filter((k) => att[k]);
  const repeats = att.all?.repeats ?? 1;

  return (
    <div className="max-w-[1080px] px-8 pb-12 pt-7">
      <PageHead
        title="Guardrails"
        sub="Prompt-injection defence, PII redaction, output validation and retrieval scoping — each measured on labelled data, and the injection defence enforced as a second CI gate."
        right={
          <span className="rounded-full bg-ok-bg px-2 py-0.5 text-[10.5px] font-semibold text-ok-fg">
            v2 · measured {shortDate(report.generated_utc)}
          </span>
        }
      />

      <div className="mb-5 grid grid-cols-4 gap-3.5">
        <StatCard
          label="Attack success, all defences"
          value={pct(all?.success_rate)}
          delta={none ? `from ${pct(none.success_rate)}` : undefined}
          deltaTone="muted"
          foot={`${all?.attacks ?? 0} attacks × ${repeats} runs · gate ≤ ${pct(report.threshold)}`}
          valueClass={all && all.success_rate ? "text-bad-fg" : "text-ok-fg"}
        />
        <StatCard
          label="Held-out attacks caught"
          value={frac(ho)}
          foot="styles written after the policy was frozen"
        />
        <StatCard
          label="Real passages quarantined"
          value={frac(fp)}
          foot="false positives on the RFC corpus"
        />
        <StatCard
          label="Scope leaks"
          value={String(report.scope.default_scope_leaks)}
          foot="poisoned chunks seen by a default user"
          valueClass={report.scope.default_scope_leaks ? "text-bad-fg" : "text-ok-fg"}
        />
      </div>

      <div className="mb-2 text-[13px] font-semibold">Prompt injection — end to end</div>
      <Card className="mb-2 overflow-hidden">
        <Head cols="grid-cols-[1.4fr_1fr_1fr_1fr_110px]">
          <span>configuration</span><span>all 40</span><span>dev 24</span><span>held-out 16</span>
          <span className="text-right">when retrieved</span>
        </Head>
        {order.map((k) => (
          <div key={k} className={`grid grid-cols-[1.4fr_1fr_1fr_1fr_110px] items-center border-b border-border-soft px-[18px] py-2.5 text-[12.5px] ${k === "all" ? "bg-panel-2" : ""}`}>
            <span>
              <span className={k === "all" ? "font-semibold" : ""}>{CONFIG_LABEL[k] ?? k}</span>
              {CONFIG_NOTE[k] && <span className="ml-1.5 text-[11px] text-text-4">{CONFIG_NOTE[k]}</span>}
            </span>
            <Attempts b={att[k].all} />
            <Attempts b={att[k].dev} />
            <Attempts b={att[k].heldout} />
            <span className="text-right tabular-nums text-text-2">{pct(att[k].all.success_rate_when_exposed)}</span>
          </div>
        ))}
      </Card>
      <p className="mb-6 text-[11.5px] leading-[1.55] text-text-4">
        Mean attacks succeeded over {repeats} runs (range in brackets) — generation varies between
        runs even at temperature 0. An attack succeeds when its canary or phishing host appears in the
        final answer. Screening, hardening and the output guard are each measured alone, so their
        individual contributions are visible.
      </p>

      <div className="mb-2 text-[13px] font-semibold">Injection screening — detectors</div>
      <Card className="mb-2 overflow-hidden">
        <Head cols="grid-cols-[1.6fr_1fr_1fr_1fr]">
          <span>detector</span><span>dev recall</span><span>held-out recall</span><span>real-corpus false positives</span>
        </Head>
        {[...det.detectors, "union"].map((d) => (
          <div key={d} className={`grid grid-cols-[1.6fr_1fr_1fr_1fr] border-b border-border-soft px-[18px] py-2.5 text-[12.5px] tabular-nums ${d === "union" ? "bg-panel-2 font-semibold" : ""}`}>
            <span className="font-sans">{detName(d)}</span>
            <span>{frac(det.recall.dev[d])}</span>
            <span>{frac(det.recall.heldout[d])}</span>
            <span>{frac(det.false_positives[d])}</span>
          </div>
        ))}
      </Card>
      <p className="mb-6 text-[11.5px] leading-[1.55] text-text-4">
        Screening runs when the index is built; a flagged chunk is kept for audit but never retrieved.
        The policy and the prompt hardening were written with the dev attacks in view, so held-out
        recall is the generalisation estimate. Not caught by either detector:{" "}
        <span className="font-mono">{[...det.recall.dev.missed, ...det.recall.heldout.missed].join(", ") || "none"}</span>.
      </p>

      <div className="mb-2 text-[13px] font-semibold">PII detection</div>
      <div className="mb-2 grid grid-cols-2 gap-4">
        <PiiTable
          title="Real corpus"
          src={report.pii.corpus}
          note="Author contact blocks, labelled by hand. False positives are IETF role and list addresses, redacted by policy."
        />
        <PiiTable
          title="Synthetic passages"
          src={report.pii.synthetic}
          note="Cards and credentials the corpus lacks, with hard negatives: Luhn-invalid numbers, hashes, JWTs, OIDs."
        />
      </div>
      <p className="mb-6 text-[11.5px] leading-[1.55] text-text-4">
        Out of scope by design: IP addresses (in this corpus they are section numbers and protocol
        examples) and names or postal addresses (they need entity recognition, not patterns).
      </p>

      <div className="flex items-center gap-2.5 rounded-[11px] border border-border-2 bg-panel-2 px-4 py-3 card-shadow">
        <span className="text-text-3">
          <ShieldIcon size={17} />
        </span>
        <div className="text-[12px] text-text-2">
          A canary check measures whether the model obeyed an injected instruction. For misinformation
          attacks it is a lower bound — a model could repeat a false claim without the canary.
        </div>
      </div>
    </div>
  );
}
