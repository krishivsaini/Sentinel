"use client";

import { useEffect, useState } from "react";
import { Attribution } from "@/components/Attribution";
import { Drilldown } from "@/components/Drilldown";
import { EvalRuns } from "@/components/EvalRuns";
import { Guardrails } from "@/components/Guardrails";
import { Overview } from "@/components/Overview";
import { Playground } from "@/components/Playground";
import { Sidebar } from "@/components/Sidebar";
import { Topbar } from "@/components/Topbar";
import type { View } from "@/components/types";
import { loadDashboardData, type DashboardData } from "@/lib/data";

export default function Page() {
  const [view, setView] = useState<View>("overview");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [data, setData] = useState<DashboardData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    loadDashboardData().then(setData).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // Hash routing: each view gets a shareable URL (#drill, #attr, …) and the back button works.
  const VIEWS: View[] = ["overview", "runs", "drill", "attr", "query", "guard"];
  useEffect(() => {
    const sync = () => {
      const h = window.location.hash.replace("#", "") as View;
      if (VIEWS.includes(h)) setView(h);
    };
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const navigate = (v: View) => {
    setView(v);
    if (typeof window !== "undefined") window.location.hash = v;
  };

  return (
    <div className="flex h-screen w-full overflow-hidden bg-bg text-text">
      <Sidebar view={view} setView={navigate} theme={theme} setTheme={setTheme} />
      <main className="flex min-w-0 flex-1 flex-col">
        {data && (
          <Topbar
            latest={data.latest}
            latency={data.latency}
            threshold={data.history.faithfulness_threshold}
          />
        )}
        <div className="flex-1 overflow-y-auto overflow-x-hidden">
          {err && (
            <div className="p-8 text-[13px] text-bad-fg">
              Couldn&apos;t load eval data ({err}). Run{" "}
              <span className="font-mono">npm run sync-data</span> after an eval to refresh{" "}
              <span className="font-mono">public/data</span>.
            </div>
          )}
          {!data && !err && (
            <div className="p-8 text-[13px] text-text-4">Loading eval data…</div>
          )}
          {data && (
            <>
              {view === "overview" && <Overview data={data} />}
              {view === "runs" && <EvalRuns history={data.history} />}
              {view === "drill" && <Drilldown latest={data.latest} />}
              {view === "attr" && <Attribution latest={data.latest} />}
              {view === "query" && <Playground latest={data.latest} />}
              {view === "guard" && <Guardrails />}
            </>
          )}
        </div>
      </main>
    </div>
  );
}
