import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <div className={`rounded-xl border border-border-2 bg-panel card-shadow ${className}`}>
      {children}
    </div>
  );
}

export function PageHead({ title, sub, right }: { title: string; sub: string; right?: ReactNode }) {
  return (
    <>
      <div className="mb-1.5 flex items-baseline gap-3">
        <h1 className="text-[22px] font-semibold tracking-[-0.025em]">{title}</h1>
        {right}
      </div>
      <p className="mb-6 text-[13.5px] text-text-3">{sub}</p>
    </>
  );
}

export function StatCard({
  label,
  value,
  delta,
  deltaTone = "ok",
  foot,
  valueClass = "",
}: {
  label: string;
  value: string;
  delta?: string;
  // "auto" colors by sign (a regression must never read as green); "muted" is neutral.
  deltaTone?: "ok" | "muted" | "auto";
  foot: string;
  valueClass?: string;
}) {
  let toneClass = "bg-ok-bg text-ok-fg";
  if (deltaTone === "muted") toneClass = "bg-chip text-text-3";
  else if (deltaTone === "auto" && delta) {
    const n = parseFloat(delta);
    toneClass =
      n > 0 ? "bg-ok-bg text-ok-fg" : n < 0 ? "bg-bad-bg text-bad-fg" : "bg-chip text-text-3";
  }
  return (
    <Card className="px-[18px] py-4">
      <div className="text-[11.5px] font-medium text-text-3">{label}</div>
      <div className="mt-[9px] flex items-baseline gap-2">
        <span
          className={`text-[29px] font-semibold tracking-[-0.03em] tabular-nums ${valueClass}`}
        >
          {value}
        </span>
        {delta && (
          <span className={`rounded-[5px] px-1.5 py-0.5 text-[11px] font-semibold ${toneClass}`}>
            {delta}
          </span>
        )}
      </div>
      <div className="mt-2 font-mono text-[10px] text-text-4">{foot}</div>
    </Card>
  );
}

export function Pill({
  label,
  fg,
  bg,
}: {
  label: string;
  fg: string;
  bg: string;
}) {
  return (
    <span className={`rounded-full px-2 py-0.5 text-[10.5px] font-semibold ${fg} ${bg}`}>
      {label}
    </span>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.05em] text-text-4">
      {children}
    </div>
  );
}
