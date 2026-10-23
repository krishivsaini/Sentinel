"use client";

import type { AttackBlock, GuardrailsReport, InputRule, PiiSource, Rate, Split } from "@/lib/data";
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
  "hardening+output+answer": "Hardening + output guard + answer check",
  all: "All defences",
};
const CONFIG_NOTE: Record<string, string> = {
  "hardening+output": "what an attack faced if screening missed it, before the answer check",
  "hardening+output+answer": "what an attack faces if screening misses it",
  all: "what ships",
};
const SPLIT_LABEL: Record<Split, string> = { dev: "dev", heldout: "held-out", adaptive: "adaptive" };

function pct(v: number | null | undefined): string {
  if (v == null) return "—";
  const p = v * 100;
  return `${p === 0 || p === 100 ? p.toFixed(0) : p < 1 ? p.toFixed(2) : p.toFixed(1)}%`;
}

function frac(r?: Rate): string {
  return r ? `${r.flagged} / ${r.of}` : "—";
}

function ci(r?: { ci95?: [number, number] }): string {
  return r?.ci95 ? `${pct(r.ci95[0])}–${pct(r.ci95[1])}` : "";
}

function detName(id: string): string {
  if (id === "union") return "Either (quarantine rule)";
  if (id === "configured") return "Shipped rule";
  if (id.startsWith("promptguard")) return "Llama Prompt Guard 2 · 86M";
  if (id.startsWith("input-safeguard")) return "gpt-oss-safeguard-20b · input policy";
  if (id.startsWith("safeguard")) return "gpt-oss-safeguard-20b · passage policy";
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

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <>
      <div className="mb-2 mt-1 text-[13px] font-semibold">{title}</div>
      {children}
    </>
  );
}

function Note({ children }: { children: React.ReactNode }) {
  return <p className="mb-6 mt-2 text-[11.5px] leading-[1.55] text-text-4">{children}</p>;
}

function Attempts({ b }: { b?: AttackBlock }) {
  if (!b) return <span className="text-text-5">—</span>;
  const [lo, hi] = b.succeeded_range;
  const tone = b.succeeded === 0 ? "text-ok-fg" : "text-bad-fg";
  return (
    <span className="tabular-nums">
      <span className={`font-semibold ${tone}`}>
        {Number.isInteger(b.succeeded) ? b.succeeded : b.succeeded.toFixed(1)}
      </span>
      <span className="text-text-4"> / {b.attacks}</span>
      {lo !== hi && <span className="ml-1.5 text-[11px] text-text-4">({lo}–{hi})</span>}
    </span>
  );
}

function PiiTable({ title, src, note }: { title: string; src: PiiSource; note: string }) {
  return (
    <Card className="overflow-hidden">
      <div className="flex items-baseline gap-2 px-[18px] pb-1 pt-4">
        <div className="text-[13px] font-semibold">{title}</div>
        <div className="ml-auto text-[11.5px] tabular-nums text-text-3">
          precision <b className="text-text-1">{pct(src.precision)}</b> · recall{" "}
          <b className="text-text-1">{pct(src.recall)}</b>
        </div>
      </div>
      <div className="px-[18px] pb-3 text-[11.5px] text-text-3">
        {note}
        {src.recall_ci95 && (
          <span className="text-text-4">
            {" "}95% CI: precision {pct(src.precision_ci95?.[0])}–{pct(src.precision_ci95?.[1])}, recall{" "}
            {pct(src.recall_ci95[0])}–{pct(src.recall_ci95[1])}.
          </span>
        )}
      </div>
      <Head cols="grid-cols-[1fr_60px_60px_60px_80px_70px]">
        <span>kind</span><span>tp</span><span>fp</span><span>fn</span><span>precision</span><span>recall</span>
      </Head>
      {Object.entries(src.per_kind).map(([k, v]) => (
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
  const splits = (["dev", "heldout", "adaptive"] as Split[]).filter((s) => det.recall[s]);
  const order = ["none", "screening", "hardening", "output_guard", "hardening+output", "hardening+output+answer", "all"].filter((k) => att[k]);
  const repeats = att.all?.repeats ?? 1;
  const allDoc = att.all?.all;
  const input = report.input;
  const usr = input?.end_to_end.with_input_guard.all;
  const inputRules = input
    ? Object.keys(input.detection).filter((k) => k !== "configured_detectors")
    : [];

  return (
    <div className="max-w-[1080px] px-8 pb-12 pt-7">
      <PageHead
        title="Guardrails"
        sub="Input screening, prompt-injection defence, PII redaction, output validation and retrieval scoping — each measured on labelled data, with the attack sets enforced as a second CI gate."
        right={
          <span className="rounded-full bg-ok-bg px-2 py-0.5 text-[10.5px] font-semibold text-ok-fg">
            v2 · measured {shortDate(report.generated_utc)}
          </span>
        }
      />

      <div className="mb-5 grid grid-cols-4 gap-3.5">
        <StatCard
          label="Poisoned-document attacks through"
          value={pct(allDoc?.success_rate)}
          delta={att.none ? `from ${pct(att.none.all.success_rate)}` : undefined}
          deltaTone="muted"
          foot={`${allDoc?.attacks ?? 0} attacks × ${repeats} runs · 95% CI ≤ ${pct(allDoc?.succeeded_in_any_run?.ci95[1])}`}
          valueClass={allDoc?.success_rate ? "text-bad-fg" : "text-ok-fg"}
        />
        <StatCard
          label="User-question attacks through"
          value={usr ? pct(usr.success_rate) : "—"}
          delta={input ? `from ${pct(input.end_to_end.without_input_guard.all.success_rate)}` : undefined}
          deltaTone="muted"
          foot={usr ? `${usr.attacks} attacks · 95% CI ≤ ${pct(usr.succeeded_in_any_run.ci95[1])}` : "no input-guard data"}
          valueClass={usr?.success_rate ? "text-bad-fg" : "text-ok-fg"}
        />
        <StatCard
          label="Screener: held-out · adaptive caught"
          value={`${frac(det.recall.heldout?.union)}`}
          delta={det.recall.adaptive ? `${frac(det.recall.adaptive.union)} adaptive` : undefined}
          deltaTone="muted"
          foot="adaptive = written to evade the screener"
        />
        <StatCard
          label="Real passages quarantined"
          value={frac(det.false_positives.union)}
          foot={`false positives · scope leaks: ${report.scope.default_scope_leaks}`}
        />
      </div>

      <Section title="Poisoned documents — end to end">
        <Card className="overflow-hidden">
          <Head cols="grid-cols-[1.5fr_1fr_1fr_1fr_1fr_90px]">
            <span>configuration</span><span>all</span>
            {splits.map((s) => <span key={s}>{SPLIT_LABEL[s]}</span>)}
            <span className="text-right">95% CI</span>
          </Head>
          {order.map((k) => (
            <div key={k} className={`grid grid-cols-[1.5fr_1fr_1fr_1fr_1fr_90px] items-center border-b border-border-soft px-[18px] py-2.5 text-[12.5px] ${k === "all" ? "bg-panel-2" : ""}`}>
              <span>
                <span className={k === "all" ? "font-semibold" : ""}>{CONFIG_LABEL[k] ?? k}</span>
                {CONFIG_NOTE[k] && <span className="ml-1.5 text-[11px] text-text-4">{CONFIG_NOTE[k]}</span>}
              </span>
              <Attempts b={att[k].all} />
              {splits.map((s) => <Attempts key={s} b={att[k][s]} />)}
              <span className="text-right text-[11px] tabular-nums text-text-3">{ci(att[k].all.succeeded_in_any_run)}</span>
            </div>
          ))}
        </Card>
        <Note>
          Mean attacks that got through over {repeats} runs (range in brackets) — generation varies
          even at temperature 0. An attack succeeds when its canary or phishing host appears in the
          final answer. The 95% interval is over attacks, counting one as successful if it got through
          in any run. <b className="text-text-3">Adaptive</b> attacks were written against the
          screening policy itself.
        </Note>
      </Section>

      <Section title="Injection screening — detectors">
        <Card className="overflow-hidden">
          <Head cols="grid-cols-[1.6fr_1fr_1fr_1fr_1fr]">
            <span>detector</span>
            {splits.map((s) => <span key={s}>{SPLIT_LABEL[s]} caught</span>)}
            <span>real-corpus false positives</span>
          </Head>
          {[...det.detectors, "union"].map((d) => (
            <div key={d} className={`grid grid-cols-[1.6fr_1fr_1fr_1fr_1fr] border-b border-border-soft px-[18px] py-2.5 text-[12.5px] tabular-nums ${d === "union" ? "bg-panel-2 font-semibold" : ""}`}>
              <span>{detName(d)}</span>
              {splits.map((s) => <span key={s}>{frac(det.recall[s]?.[d])}</span>)}
              <span>{frac(det.false_positives[d])}</span>
            </div>
          ))}
        </Card>
        <Note>
          Screening runs when the index is built; a flagged chunk is kept for audit but never
          retrieved. The policy and the prompt hardening were written with the dev attacks in view,
          so held-out recall is the generalisation estimate. Not caught by either detector:{" "}
          <span className="font-mono">
            {splits.flatMap((s) => det.recall[s]?.missed ?? []).join(", ") || "none"}
          </span>
          .
        </Note>
      </Section>

      {input && (
        <Section title="Input guard — the user's own question">
          <Card className="overflow-hidden">
            <Head cols="grid-cols-[1.6fr_1fr_1fr_1fr_1fr]">
              <span>detector</span><span>dev attacks caught</span><span>held-out caught</span>
              <span>genuine refused</span><span>ground truth refused</span>
            </Head>
            {inputRules.map((rule) => {
              const r = input.detection[rule] as InputRule;
              const refused = r.dev.benign_refused.flagged + r.heldout.benign_refused.flagged;
              const of = r.dev.benign_refused.of + r.heldout.benign_refused.of;
              return (
                <div key={rule} className={`grid grid-cols-[1.6fr_1fr_1fr_1fr_1fr] border-b border-border-soft px-[18px] py-2.5 text-[12.5px] tabular-nums ${rule === "configured" ? "bg-panel-2 font-semibold" : ""}`}>
                  <span>{detName(rule)}</span>
                  <span>{frac(r.dev.attacks_caught)}</span>
                  <span>{frac(r.heldout.attacks_caught)}</span>
                  <span className={refused ? "text-warn-fg" : ""}>{refused} / {of}</span>
                  <span>{frac(r.ground_truth_refused)}</span>
                </div>
              );
            })}
          </Card>
          <Note>
            Run end to end through the shipped pipeline, user-side attacks got through{" "}
            <b className="text-text-3">{input.end_to_end.without_input_guard.all.succeeded} / {input.end_to_end.without_input_guard.all.attacks}</b>{" "}
            without the input guard and{" "}
            <b className="text-text-3">{input.end_to_end.with_input_guard.all.succeeded} / {input.end_to_end.with_input_guard.all.attacks}</b>{" "}
            with it (mean of {input.end_to_end.repeats} runs). Genuine questions include all 48
            ground-truth questions and hard negatives full of words like “inject”, “override” and
            “ignore”.
          </Note>
        </Section>
      )}

      <Section title="PII detection">
        <div className="grid grid-cols-2 gap-4">
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
        <Note>
          Out of scope by design: IP addresses (in this corpus they are section numbers and protocol
          examples) and names or postal addresses (they need entity recognition, not patterns).
        </Note>
      </Section>

      <div className="flex items-center gap-2.5 rounded-[11px] border border-border-2 bg-panel-2 px-4 py-3 card-shadow">
        <span className="text-text-3">
          <ShieldIcon size={17} />
        </span>
        <div className="text-[12px] text-text-2">
          A canary check measures whether the model obeyed an injected instruction. For misinformation
          attacks it is a lower bound — a model could repeat a false claim without the canary. Every
          attack here was written by the same author as the defences; the held-out and adaptive splits
          limit, but don&apos;t remove, that bias.
        </div>
      </div>
    </div>
  );
}
