"use client";

import {
  AttrIcon,
  GridIcon,
  ListIcon,
  LogoMark,
  MoonIcon,
  QueryIcon,
  SearchIcon,
  ShieldIcon,
  SunIcon,
} from "./icons";
import type { View } from "./types";

const navBase =
  "flex w-full items-center gap-[9px] rounded-lg px-2.5 py-[7px] text-left text-[13px] font-medium tracking-[-0.01em] mb-px transition-colors";

function NavButton({
  id,
  view,
  setView,
  icon,
  children,
}: {
  id: View;
  view: View;
  setView: (v: View) => void;
  icon: React.ReactNode;
  children: React.ReactNode;
}) {
  const active = view === id;
  return (
    <button
      onClick={() => setView(id)}
      className={`${navBase} ${
        active
          ? "bg-accent-soft font-semibold text-accent-2"
          : "bg-transparent text-text-2 hover:bg-border-soft"
      }`}
    >
      {icon}
      {children}
    </button>
  );
}

function Group({ children }: { children: React.ReactNode }) {
  return (
    <div className="px-2.5 pb-1.5 pt-4 text-[10px] font-semibold uppercase tracking-[0.09em] text-text-5">
      {children}
    </div>
  );
}

export function Sidebar({
  view,
  setView,
  theme,
  setTheme,
}: {
  view: View;
  setView: (v: View) => void;
  theme: "light" | "dark";
  setTheme: (t: "light" | "dark") => void;
}) {
  const seg =
    "flex flex-1 items-center justify-center gap-1.5 rounded-[7px] px-1 py-1.5 text-[11.5px] font-semibold";
  return (
    <aside className="flex w-[236px] flex-none flex-col border-r border-border bg-panel px-3.5 py-[18px]">
      <div className="flex items-center gap-2.5 px-2 pb-[18px] pt-1">
        <div className="flex h-[26px] w-[26px] flex-none items-center justify-center rounded-[7px] bg-mark">
          <LogoMark />
        </div>
        <div className="leading-none">
          <div className="text-[14.5px] font-semibold tracking-[-0.02em]">Sentinel</div>
          <div className="mt-[3px] font-mono text-[9.5px] tracking-[0.02em] text-text-4">
            eval-gated RAG
          </div>
        </div>
      </div>

      <Group>Monitor</Group>
      <NavButton id="overview" view={view} setView={setView} icon={<GridIcon />}>
        Overview
      </NavButton>
      <NavButton id="runs" view={view} setView={setView} icon={<ListIcon />}>
        Eval Runs
      </NavButton>
      <NavButton id="drill" view={view} setView={setView} icon={<SearchIcon />}>
        Drill-down
      </NavButton>
      <NavButton id="attr" view={view} setView={setView} icon={<AttrIcon />}>
        Failure Attribution
      </NavButton>

      <Group>Service</Group>
      <NavButton id="query" view={view} setView={setView} icon={<QueryIcon />}>
        Query Playground
      </NavButton>

      <Group>Security · v2</Group>
      <NavButton id="guard" view={view} setView={setView} icon={<ShieldIcon />}>
        Guardrails
      </NavButton>

      <div className="mt-auto border-t border-border-hair px-1 pb-0.5 pt-3">
        <div className="mb-3.5 flex gap-[3px] rounded-[9px] bg-chip p-[3px]">
          <button
            onClick={() => setTheme("light")}
            className={`${seg} ${
              theme === "light"
                ? "bg-panel text-text shadow-[0_1px_2px_rgba(0,0,0,0.08)]"
                : "bg-transparent text-text-3"
            }`}
          >
            <SunIcon />
            Light
          </button>
          <button
            onClick={() => setTheme("dark")}
            className={`${seg} ${
              theme === "dark"
                ? "bg-panel text-text shadow-[0_1px_2px_rgba(0,0,0,0.08)]"
                : "bg-transparent text-text-3"
            }`}
          >
            <MoonIcon />
            Dark
          </button>
        </div>
        <div className="px-1.5 font-mono text-[10px] leading-[1.7] text-text-4">
          <div>corpus · IETF RFC web stack</div>
          <div>2,888 chunks · 48 RFCs</div>
        </div>
      </div>
    </aside>
  );
}
