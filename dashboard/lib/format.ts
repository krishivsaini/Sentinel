import type { ItemResult } from "./data";

export const f2 = (n: number) => n.toFixed(2);
export const f3 = (n: number) => n.toFixed(3);

export function shortDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "2-digit" });
}

// A short document label from a chunk id like "rfc9110#0007" -> "RFC 9110 · #0007".
export function docLabel(chunkId: string): string {
  const m = chunkId.match(/^rfc(\d+)#(\d+)$/i);
  if (!m) return chunkId;
  return `RFC ${m[1]} · #${m[2]}`;
}

export function docName(chunkId: string): string {
  const m = chunkId.match(/^rfc(\d+)#/i);
  return m ? `RFC ${m[1]}` : chunkId;
}

// Which retrieved chunk ids the answer actually cited (inline [rfc..#..] or unicode-bracket forms).
export function citedChunkIds(item: ItemResult): Set<string> {
  const ids = new Set<string>();
  const re = /[[【]\s*(rfc\d+#\d+)\s*[\]】]/gi;
  let m: RegExpExecArray | null;
  while ((m = re.exec(item.generated_answer)) !== null) ids.add(m[1].toLowerCase());
  return ids;
}

// Three-level score encoding so colour never contradicts the attribution verdict:
//   < 0.60  red   — below the failure threshold (this is what makes an item a failure)
//   < 0.80  amber — below the CI gate line, but not classified a failure
//   else    normal
export function scoreClass(v: number): string {
  if (v < 0.6) return "text-bad-fg";
  if (v < 0.8) return "text-warn-fg";
  return "text-text";
}

export type Kind = "grounded" | "retrieval" | "generation";

export function kindOf(item: ItemResult): Kind {
  if (item.attribution === "retrieval_fail") return "retrieval";
  if (item.attribution === "generation_fail") return "generation";
  return "grounded";
}

export const kindMeta: Record<Kind, { label: string; fg: string; bg: string }> = {
  grounded: { label: "grounded", fg: "text-ok-fg", bg: "bg-ok-bg" },
  retrieval: { label: "retrieval", fg: "text-ret-fg", bg: "bg-ret-bg" },
  generation: { label: "generation", fg: "text-gen-fg", bg: "bg-gen-bg" },
};
