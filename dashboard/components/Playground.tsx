"use client";

import { useEffect, useState } from "react";
import type { RunResult } from "@/lib/data";
import { docLabel } from "@/lib/format";
import { SearchIcon } from "./icons";
import { Card, PageHead } from "./ui";

const API = process.env.NEXT_PUBLIC_API_URL ?? "";

type Phase = "idle" | "streaming" | "checking" | "done" | "error";
type Chunk = { chunk_id: string; rerank_score?: number | null };
// The `done` event's Answer, the parts this view shows.
type Guardrails = {
  context_redactions: Record<string, number>;
  removed: Record<string, number>;
  unresolved_citations: string[];
  uncited: boolean;
  answer_withheld?: boolean;
};
// A citation carries its RFC's standing: later RFCs that replaced (obsoleted_by) or amended it.
type Citation = { chunk_id: string; doc_id: string; obsoleted_by?: string[]; updated_by?: string[] };
type Done = {
  text: string;
  citations?: Citation[];
  latency_ms: Record<string, number>;
  guardrails?: Guardrails | null;
  answer_id?: string | null;
  // The search queries a multi-part question was split into (query planning); empty otherwise.
  sub_questions?: string[];
};

const rfc = (docId: string) => docId.replace(/^rfc/i, "RFC ");

// Streams from the real Sentinel API: POST /token -> JWT, then POST /query and parse the SSE
// frames (`retrieved` -> chunks, `checking` -> the answer check is running, `token` -> text,
// `done` -> final text + latency + guardrail report). No API configured => idle state.
//
// Without a sign-in the playground uses the shared demo account (its password is public, so it
// only gets the public scope). A named account signs in below; its token is kept in memory for
// this tab only, and the password is never stored.
const DEMO = { username: "demo", password: "change-me" };
type Session = { token: string; username: string; scopes: string[]; shared: boolean; expires: number };

async function signIn(username: string, password: string): Promise<Session> {
  const res = await fetch(`${API}/token`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username, password }),
  });
  if (res.status === 401) throw new Error("Wrong username or password.");
  if (res.status === 429) throw new Error("Too many sign-in attempts. Wait a minute.");
  if (!res.ok) throw new Error(`auth ${res.status}`);
  const t = await res.json();
  const me = await fetch(`${API}/me`, { headers: { Authorization: `Bearer ${t.access_token}` } });
  // An API without /me (older deploys) still works: the demo account by name, public scope.
  const account = me.ok
    ? await me.json()
    : { username, scopes: t.scopes ?? ["public"], shared: username === DEMO.username };
  // Renewed a minute early so a question never starts on a token about to expire.
  const expires = Date.now() + ((t.expires_in || 3600) - 60) * 1000;
  return { token: t.access_token, ...account, expires };
}

async function streamAnswer(
  access_token: string,
  question: string,
  onChunks: (c: Chunk[]) => void,
  onToken: (t: string) => void,
  onDone: (d: Done) => void,
  onChecking: () => void,
): Promise<void> {
  const res = await fetch(`${API}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${access_token}` },
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) throw new Error(`query ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // SSE lines may end in CRLF, LF or CR — sse-starlette sends CRLF. Normalise to LF, holding
    // back a trailing CR whose LF may arrive in the next read.
    const held = buf.endsWith("\r") ? "\r" : "";
    buf = buf.slice(0, buf.length - held.length).replace(/\r\n?/g, "\n") + held;
    const frames = buf.split("\n\n");
    buf = frames.pop() ?? "";
    for (const frame of frames) {
      let event = "message";
      const data: string[] = [];
      for (const line of frame.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        // Per the SSE spec, strip ONE leading space after "data:". Trimming more would eat the
        // spaces between streamed tokens and glue words together.
        else if (line.startsWith("data:")) data.push(line.slice(line.startsWith("data: ") ? 6 : 5));
      }
      const payload = data.join("\n");
      if (event === "retrieved") {
        try {
          onChunks(JSON.parse(payload));
        } catch {
          /* ignore malformed frame */
        }
      } else if (event === "checking") {
        onChecking();
      } else if (event === "token") {
        onToken(payload);
      } else if (event === "done") {
        try {
          onDone(JSON.parse(payload));
        } catch {
          /* ignore malformed frame */
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
  // What's on screen: the answer revealed word by word. The API sends a checked answer's tokens
  // in a burst once the answer check passes, so pacing here keeps it reading like a stream.
  const [revealed, setRevealed] = useState("");
  const [chunks, setChunks] = useState<Chunk[]>([]);
  const [done, setDone] = useState<Done | null>(null);
  const [error, setError] = useState("");
  const [token, setToken] = useState("");
  const [session, setSession] = useState<Session | null>(null);

  useEffect(() => {
    if (!answer.startsWith(revealed)) {
      setRevealed(""); // the text was replaced (a withheld answer, a new question): start over
      return;
    }
    if (revealed.length === answer.length) return;
    const id = setTimeout(() => {
      const words = answer.slice(revealed.length).match(/\s*\S+/g) ?? [answer.slice(revealed.length)];
      // One word per tick, faster for a long backlog so no answer takes more than ~2 s to show.
      setRevealed(revealed + words.slice(0, Math.max(1, Math.ceil(words.length / 60))).join(""));
    }, 30);
    return () => clearTimeout(id);
  }, [answer, revealed]);
  const revealing = revealed.length < answer.length;

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
    setDone(null);
    try {
      let s = session;
      if (!s || Date.now() > s.expires) {
        if (s && !s.shared) {
          setSession(null);
          throw new Error("Your session expired. Sign in again.");
        }
        s = await signIn(DEMO.username, DEMO.password);
        setSession(s);
      }
      const t = s.token; // kept for this answer's feedback
      setToken(t);
      await streamAnswer(
        t,
        q,
        setChunks,
        (t) => {
          setPhase("streaming");
          setAnswer((a) => a + t);
        },
        (d) => {
          // The done event's text is final. The API holds the answer until the answer check
          // passes; if it's configured to stream instead, a withheld answer is replaced here.
          setAnswer(d.text);
          setDone(d);
        },
        () => setPhase("checking"),
      );
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
        sub="POST /query · JWT-authed · SSE token stream · hybrid retrieval → rerank → grounded generation → answer check."
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
      <AccountBar session={session} onSession={setSession} />
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
            {error === "no-api" ? "No API endpoint configured" : error.startsWith("Your session") ? error : "Couldn't reach the API"}
          </div>
          <div className="mx-auto max-w-[520px] text-[12px] leading-[1.6] text-text-4">
            The playground streams from a running Sentinel service. Start it with{" "}
            <span className="font-mono text-text-3">uvicorn sentinel.serve:app</span> and set{" "}
            <span className="font-mono text-text-3">NEXT_PUBLIC_API_URL</span> to its URL. The three
            data views above run entirely on exported JSON and need no backend.
          </div>
        </div>
      )}

      {(phase === "streaming" || phase === "checking" || phase === "done") && (
        <div className="grid grid-cols-[1fr_360px] items-start gap-4">
          <Card className="overflow-hidden">
            <div className="flex items-center gap-2 border-b border-border-hair px-[18px] py-3">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: phase === "done" ? "var(--ok-fg)" : "var(--warn-fg)" }} />
              <span className="text-[12.5px] font-semibold">
                {phase === "streaming" ? "streaming · SSE" : phase === "checking" ? "checking the answer…" : "answer"}
              </span>
              <span className="ml-auto font-mono text-[10.5px] text-text-4">grounded · cited</span>
            </div>
            <div className="min-h-[96px] p-[18px] text-[14px] leading-[1.7] text-text-1">
              {revealed}
              {phase === "checking" && !answer && (
                <span className="text-text-4">Generated. Holding it until the answer check passes.</span>
              )}
              {(phase === "streaming" || revealing) && <span className="text-accent">▍</span>}
            </div>
            {done && !revealing && <AnswerFooter done={done} />}
            {done && !revealing && done.answer_id && (
              <Feedback key={done.answer_id} answerId={done.answer_id} token={token} />
            )}
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

function AnswerFooter({ done }: { done: Done }) {
  const g = done.guardrails;
  const notes: string[] = [];
  if (g) {
    for (const [k, n] of Object.entries(g.context_redactions)) notes.push(`${n} ${k} redacted from context`);
    for (const [k, n] of Object.entries(g.removed))
      notes.push(k === "link" ? `${n} untrusted link removed` : k === "image" ? `${n} image removed` : `${n} ${k} redacted`);
    if (g.unresolved_citations.length) notes.push(`unresolved citation: ${g.unresolved_citations.join(", ")}`);
    if (g.uncited) notes.push("no valid citation");
    if (g.answer_withheld) notes.push("answer withheld by the answer check");
  }
  // One line per cited RFC, at most: replaced outright is a warning, amended is a note.
  const seen = new Set<string>();
  const sources = (done.citations ?? []).filter((c) => !seen.has(c.doc_id) && seen.add(c.doc_id));
  const outdated = sources.filter((c) => c.obsoleted_by?.length);
  const amended = sources.filter((c) => !c.obsoleted_by?.length && c.updated_by?.length);
  return (
    <div className="border-t border-border-hair px-[18px] py-3">
      {outdated.map((c) => (
        <div key={c.doc_id} className="mb-2 rounded-[6px] bg-warn-bg px-2.5 py-1.5 text-[11.5px] text-warn-fg">
          Cites {rfc(c.doc_id)}, which is obsoleted by {c.obsoleted_by!.map(rfc).join(", ")}; check the
          current specification.
        </div>
      ))}
      {amended.length > 0 && (
        <div className="mb-2 text-[11.5px] text-text-4">
          {amended
            .map((c) => `${rfc(c.doc_id)} is updated by ${c.updated_by!.length > 3
              ? `${c.updated_by!.length} later RFCs`
              : c.updated_by!.map(rfc).join(", ")}`)
            .join(" · ")}
        </div>
      )}
      {(done.sub_questions?.length ?? 0) > 1 && (
        <div className="mb-2 text-[11.5px] text-text-4">
          searched in {done.sub_questions!.length} parts: {done.sub_questions!.map((q) => `“${q}”`).join(" · ")}
        </div>
      )}
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(done.latency_ms)
          .filter(([stage, ms]) => stage !== "plan" || ms > 0)
          .map(([stage, ms]) => (
          <span key={stage} className="rounded-[5px] bg-chip px-1.5 py-0.5 font-mono text-[10.5px] text-text-3">
            {stage} {Math.round(ms)} ms
          </span>
        ))}
      </div>
      <div className={`mt-2 text-[11.5px] ${notes.length ? "text-warn-fg" : "text-text-4"}`}>
        guardrails: {notes.length ? notes.join(" · ") : "no changes"}
      </div>
    </div>
  );
}

// Thumbs up/down on one answer, sent to POST /feedback by its answer_id. A thumbs-down may carry a
// short comment; scripts/feedback_candidates.py turns them into eval candidates for review.
function Feedback({ answerId, token }: { answerId: string; token: string }) {
  const [state, setState] = useState<"ask" | "comment" | "sending" | "sent" | "failed">("ask");
  const [comment, setComment] = useState("");

  async function send(rating: "up" | "down") {
    setState("sending");
    try {
      const res = await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ answer_id: answerId, rating, comment: comment.trim() || null }),
      });
      setState(res.ok ? "sent" : "failed");
    } catch {
      setState("failed");
    }
  }

  const button =
    "rounded-[6px] border border-border-2 bg-panel px-2.5 py-1 text-[11.5px] text-text-2 hover:bg-border-soft";
  return (
    <div className="flex flex-wrap items-center gap-2 border-t border-border-hair px-[18px] py-2.5 text-[11.5px] text-text-4">
      {state === "ask" && (
        <>
          <span>Was this answer right?</span>
          <button className={button} onClick={() => send("up")} aria-label="Yes, helpful">
            👍 Yes
          </button>
          <button className={button} onClick={() => setState("comment")} aria-label="No, not helpful">
            👎 No
          </button>
        </>
      )}
      {state === "comment" && (
        <>
          <input
            value={comment}
            onChange={(e) => setComment(e.target.value.slice(0, 500))}
            onKeyDown={(e) => e.key === "Enter" && send("down")}
            placeholder="What was wrong? (optional)"
            className="min-w-[220px] flex-1 rounded-[6px] border border-input-bd bg-panel px-2.5 py-1 text-[12px] text-text"
            autoFocus
          />
          <button className={button} onClick={() => send("down")}>
            Send
          </button>
        </>
      )}
      {state === "sending" && <span>Sending…</span>}
      {state === "sent" && <span>Thanks, recorded.</span>}
      {state === "failed" && <span className="text-warn-fg">Couldn't send feedback.</span>}
    </div>
  );
}

// Who the playground asks as. Signing in swaps the token; nothing is written to storage.
function AccountBar({
  session,
  onSession,
}: {
  session: Session | null;
  onSession: (s: Session | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const named = session && !session.shared;

  async function submit() {
    if (!API || !username || !password) return;
    setBusy(true);
    setErr("");
    try {
      onSession(await signIn(username, password));
      setOpen(false);
      setPassword("");
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mb-3 text-[12px] text-text-3">
      <div className="flex flex-wrap items-center gap-2">
        <span>
          Asking as <span className="font-mono text-text-2">{named ? session.username : DEMO.username}</span>
          {" · scopes "}
          <span className="font-mono text-text-2">{(session?.scopes ?? ["public"]).join(", ")}</span>
          {!named && <span className="text-text-4"> (shared demo account)</span>}
        </span>
        {named ? (
          <button onClick={() => onSession(null)} className="text-accent hover:underline">
            Sign out
          </button>
        ) : (
          <button onClick={() => setOpen(!open)} className="text-accent hover:underline">
            {open ? "Cancel" : "Sign in with an account"}
          </button>
        )}
      </div>
      {open && !named && (
        <form
          className="mt-2 flex flex-wrap items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value.trim().toLowerCase())}
            placeholder="username"
            autoComplete="username"
            className="rounded-lg border border-input-bd bg-panel px-3 py-1.5 text-[12.5px] text-text"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="password"
            autoComplete="current-password"
            className="rounded-lg border border-input-bd bg-panel px-3 py-1.5 text-[12.5px] text-text"
          />
          <button
            type="submit"
            disabled={busy || !username || !password}
            className="rounded-lg bg-btn-bg px-3 py-1.5 text-[12.5px] font-medium text-btn-fg disabled:opacity-50"
          >
            {busy ? "Signing in…" : "Sign in"}
          </button>
          {err && <span className="text-[12px] text-bad-fg">{err}</span>}
          {!API && <span className="text-text-4">No API endpoint configured.</span>}
        </form>
      )}
    </div>
  );
}
