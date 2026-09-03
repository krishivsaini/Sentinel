"use client";

import { useState } from "react";
import type { RunResult } from "@/lib/data";
import { docLabel } from "@/lib/format";
import { SearchIcon } from "./icons";
import { Card, PageHead } from "./ui";

const API = process.env.NEXT_PUBLIC_API_URL ?? "";

type Phase = "idle" | "streaming" | "done" | "error";
type Chunk = { chunk_id: string; rerank_score?: number | null };

// Streams from the real Sentinel API: POST /token (demo creds) -> JWT, then POST /query and parse
// the SSE frames (`retrieved` -> chunks, `token` -> text, `done`). No API configured => idle state.
async function streamAnswer(
  question: string,
  onChunks: (c: Chunk[]) => void,
  onToken: (t: string) => void,
): Promise<void> {
  const tokenRes = await fetch(`${API}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: "demo", password: "change-me" }),
  });
  if (!tokenRes.ok) throw new Error(`auth ${tokenRes.status}`);
  const { access_token } = await tokenRes.json();

  const res = await fetch(`${API}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${access_token}` },
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) throw new Error(`query ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let event = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const frames = buf.split("\n\n");
    buf = frames.pop() ?? "";
    for (const frame of frames) {
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) {
          const data = line.slice(5).trim();
          if (event === "retrieved") {
            try {
              onChunks(JSON.parse(data));
            } catch {
              /* ignore malformed frame */
            }
          } else if (event === "token") {
            onToken(data);
          }
        }
      }
    }
  }
}

export function Playground({ latest }: { latest: RunResult }) {
  const samples = latest.per_item.slice(0, 3).map((i) => i.question);
  const [query, setQuery] = useState("");
  const [phase, setPhase] = useState<Phase>("idle");
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [error, setError] = useState("");

  async function ask(q: string) {
    if (!q.trim()) return;
    if (!API) {
      setPhase("error");
      setError("no-api");
      return;
    }
    setPhase("streaming");
    setAnswer("");
    setChunks([]);
    try {
      await streamAnswer(q, setChunks, (t) => setAnswer((a) => a + t));
      setPhase("done");
    } catch (e) {
      setError(String(e));
      setPhase("error");
    }
  }

  return (
    <div className="max-w-[1080px] px-8 pb-12 pt-7">
      <PageHead
        title="Query playground"
        sub="POST /query · JWT-authed · SSE token stream · hybrid retrieval → rerank → grounded generation."
      />

      <div className="mb-3 flex gap-2.5">
        <div className="flex flex-1 items-center gap-2.5 rounded-[11px] border border-input-bd bg-panel pl-4 pr-1 card-shadow">
          <SearchIcon size={16} className="text-text-5" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask(query)}
            placeholder="Ask a question about the RFC corpus…"
            className="flex-1 border-none bg-transparent py-3.5 text-[14px] text-text placeholder:text-text-4"
          />
          <button
            onClick={() => ask(query)}
            className="my-[5px] rounded-lg bg-btn-bg px-[18px] py-[9px] text-[13px] font-medium text-btn-fg"
          >
            Ask
          </button>
        </div>
      </div>
      <div className="mb-6 flex flex-wrap gap-2">
        {samples.map((s) => (
          <button
            key={s}
            onClick={() => {
              setQuery(s);
              ask(s);
            }}
            className="rounded-full border border-border-2 bg-panel px-3 py-1.5 text-[12px] text-text-2 hover:bg-border-soft"
          >
            {s.length > 52 ? s.slice(0, 52) + "…" : s}
          </button>
        ))}
      </div>

      {phase === "idle" && (
        <div className="rounded-xl border border-dashed border-input-bd p-11 text-center text-text-4">
          <div className="text-[13px]">
            Ask a question or pick a sample to stream the grounded answer, retrieved chunks, and
            per-stage latency.
          </div>
        </div>
      )}

      {phase === "error" && (
        <div className="rounded-xl border border-dashed border-input-bd p-11 text-center">
          <div className="mb-1.5 text-[13px] font-medium text-text-2">
            {error === "no-api" ? "No API endpoint configured" : "Couldn't reach the API"}
          </div>
          <div className="mx-auto max-w-[520px] text-[12px] leading-[1.6] text-text-4">
            The playground streams from a running Sentinel service. Start it with{" "}
            <span className="font-mono text-text-3">uvicorn sentinel.serve:app</span> and set{" "}
            <span className="font-mono text-text-3">NEXT_PUBLIC_API_URL</span> to its URL. The three
            data views above run entirely on exported JSON and need no backend.
          </div>
        </div>
      )}

      {(phase === "streaming" || phase === "done") && (
        <div className="grid grid-cols-[1fr_360px] items-start gap-4">
          <Card className="overflow-hidden">
            <div className="flex items-center gap-2 border-b border-border-hair px-[18px] py-3">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: phase === "streaming" ? "var(--warn-fg)" : "var(--ok-fg)" }} />
              <span className="text-[12.5px] font-semibold">{phase === "streaming" ? "streaming · SSE" : "answer"}</span>
              <span className="ml-auto font-mono text-[10.5px] text-text-4">grounded · cited</span>
            </div>
            <div className="min-h-[96px] p-[18px] text-[14px] leading-[1.7] text-text-1">
              {answer}
              {phase === "streaming" && <span className="text-accent">▍</span>}
            </div>
          </Card>
          <Card className="overflow-hidden">
            <div className="border-b border-border-hair px-4 py-3 text-[12.5px] font-semibold">
              Retrieved context · rerank top-5
            </div>
            {chunks.map((c) => (
              <div key={c.chunk_id} className="border-b border-border-soft px-4 py-2.5">
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-[10.5px] font-medium text-accent">{docLabel(c.chunk_id)}</span>
                  {c.rerank_score != null && (
                    <span className="ml-auto font-mono text-[10px] font-semibold text-text-1">
                      r {c.rerank_score.toFixed(2)}
                    </span>
                  )}
                </div>
              </div>
            ))}
            {chunks.length === 0 && <div className="px-4 py-4 text-[11.5px] text-text-4">waiting for retrieval…</div>}
          </Card>
        </div>
      )}
    </div>
  );
}
