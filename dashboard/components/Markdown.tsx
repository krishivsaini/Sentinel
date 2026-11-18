import { Fragment, type ReactNode } from "react";

// The generator answers in light markdown: paragraphs, bullet and numbered lists, **bold**,
// *italic*, `code`, fenced code blocks and > quotes, with citations like [rfc2119#0000] (or the
// 【…】 form). This renders that subset as React elements — never as HTML, so nothing in an
// answer can inject markup. It copes with a half-streamed answer: an unclosed fence runs to the
// end, an unclosed ** stays literal until its partner arrives.

type Block =
  | { kind: "p"; text: string }
  | { kind: "h"; text: string }
  | { kind: "quote"; text: string }
  | { kind: "code"; text: string }
  | { kind: "ul" | "ol"; items: string[]; start: number };

const BULLET = /^\s*[*+-]\s+(.*)$/;
const NUMBERED = /^\s*(\d+)[.)]\s+(.*)$/;
const CITATION_ONLY = /^\s*(?:[[【]\s*[a-z][a-z0-9]*#\d+\s*[\]】]\s*[.,;]?\s*)+$/i;

function parseBlocks(src: string): Block[] {
  const lines = src.replace(/\r\n?/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  // Soft-wrapped lines join with a space; markdown's hard breaks are dropped, which also pulls a
  // citation the model put on its own line back next to the claim it supports.
  const join = (parts: string[]) => parts.map((s) => s.trim()).filter(Boolean).join(" ");
  const startsBlock = (l: string) =>
    /^\s*```/.test(l) || /^\s*#{1,6}\s/.test(l) || /^\s*>/.test(l) || BULLET.test(l) || NUMBERED.test(l);

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
    } else if (/^\s*```/.test(line)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) body.push(lines[i++]);
      i++; // the closing fence, if it has arrived
      blocks.push({ kind: "code", text: body.join("\n") });
    } else if (/^\s*#{1,6}\s/.test(line)) {
      blocks.push({ kind: "h", text: line.replace(/^\s*#{1,6}\s+/, "") });
      i++;
    } else if (/^\s*>/.test(line)) {
      const body: string[] = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) body.push(lines[i++].replace(/^\s*>\s?/, ""));
      blocks.push({ kind: "quote", text: join(body) });
    } else if (BULLET.test(line) || NUMBERED.test(line)) {
      const ordered = !BULLET.test(line);
      const marker = ordered ? NUMBERED : BULLET;
      const start = ordered ? Number(line.match(NUMBERED)![1]) : 1;
      const items: string[][] = [];
      // A list runs on across blank lines and indented continuation lines until something else
      // starts: loose lists ("* a\n\n* b") are one list, not one list per item.
      while (i < lines.length) {
        const l = lines[i];
        const m = l.match(marker);
        if (m) {
          items.push([ordered ? m[2] : m[1]]);
          i++;
        } else if (!l.trim()) {
          let j = i;
          while (j < lines.length && !lines[j].trim()) j++;
          if (j < lines.length && (marker.test(lines[j]) || /^\s{2,}\S/.test(lines[j]))) i = j;
          else break;
        } else if (/^\s{2,}\S/.test(l) || CITATION_ONLY.test(l) || !startsBlock(l)) {
          items[items.length - 1].push(l);
          i++;
        } else break;
      }
      blocks.push({ kind: ordered ? "ol" : "ul", items: items.map(join), start });
    } else {
      const body: string[] = [];
      while (i < lines.length && lines[i].trim() && !startsBlock(lines[i])) body.push(lines[i++]);
      blocks.push({ kind: "p", text: join(body) });
    }
  }
  return blocks;
}

// Inline: citations, `code`, **bold**, *italic*, and [text](https://…) links — the output guard
// has already removed links to hosts outside its allowlist before the text gets here.
const INLINE =
  /([[【]\s*([a-z][a-z0-9]*)#(\d+)\s*[\]】])|`([^`]+)`|\*\*([^*]+?)\*\*|__([^_]+?)__|\*([^*\s][^*]*?)\*|\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/gi;

function Citation({ doc, ordinal, forged }: { doc: string; ordinal: string; forged: boolean }) {
  const m = doc.match(/^rfc(\d+)$/i);
  const label = m ? `RFC ${m[1]}` : doc;
  const base =
    "mx-0.5 inline-flex items-center rounded-[5px] border px-1.5 py-px align-[1px] font-mono text-[10.5px] leading-[1.5] no-underline";
  // A citation to a passage that wasn't retrieved is flagged, never shown as a source.
  if (forged)
    return (
      <span title={`${label}, passage #${ordinal}: not one of the retrieved passages`} className={`${base} border-warn-fg bg-warn-bg text-warn-fg line-through`}>
        {label}
      </span>
    );
  const cls = `${base} border-border-2 bg-chip text-accent`;
  const title = `${label}, passage #${ordinal}`;
  return m ? (
    <a href={`https://www.rfc-editor.org/rfc/rfc${m[1]}`} target="_blank" rel="noreferrer" title={title} className={`${cls} hover:bg-border-soft`}>
      {label}
    </a>
  ) : (
    <span title={title} className={cls}>
      {label}
    </span>
  );
}

// `known` is the set of retrieved chunk ids; without it every citation is taken at face value.
function inline(text: string, keyPrefix: string, known?: Set<string>): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let k = 0;
  for (const m of text.matchAll(INLINE)) {
    const at = m.index ?? 0;
    if (at > last) out.push(text.slice(last, at));
    const key = `${keyPrefix}-${k++}`;
    if (m[1]) {
      const forged = !!known && !known.has(`${m[2]}#${m[3]}`.toLowerCase());
      out.push(<Citation key={key} doc={m[2]} ordinal={m[3]} forged={forged} />);
    }
    else if (m[4]) out.push(<code key={key} className="rounded bg-chip px-1 py-px font-mono text-[12.5px]">{m[4]}</code>);
    else if (m[5] || m[6]) out.push(<strong key={key} className="font-semibold text-text">{inline(m[5] ?? m[6], key, known)}</strong>);
    else if (m[7]) out.push(<em key={key}>{inline(m[7], key, known)}</em>);
    else if (m[8]) out.push(<a key={key} href={m[9]} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-2">{m[8]}</a>);
    last = at + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

// `tail` (the streaming cursor) is placed at the end of the last block rather than below it.
export function Markdown({ text, tail, known }: { text: string; tail?: ReactNode; known?: Set<string> }) {
  const blocks = parseBlocks(text);
  if (!blocks.length) return <>{tail}</>;
  return (
    <div className="space-y-3">
      {blocks.map((b, n) => {
        const end = n === blocks.length - 1 ? tail : null;
        const key = `b${n}`;
        switch (b.kind) {
          case "code":
            return (
              <Fragment key={key}>
                <pre className="overflow-x-auto rounded-lg bg-chip px-3.5 py-2.5 font-mono text-[12.5px] leading-[1.6] text-text-1">
                  {b.text}
                </pre>
                {end}
              </Fragment>
            );
          case "h":
            return <div key={key} className="font-semibold text-text">{inline(b.text, key, known)}{end}</div>;
          case "quote":
            return (
              <blockquote key={key} className="border-l-2 border-border-2 pl-3 text-text-2">
                {inline(b.text, key, known)}
                {end}
              </blockquote>
            );
          case "ul":
          case "ol": {
            const List = b.kind;
            return (
              <List
                key={key}
                start={b.kind === "ol" ? b.start : undefined}
                className={`space-y-1.5 pl-5 ${b.kind === "ol" ? "list-decimal" : "list-disc"} marker:text-text-4`}
              >
                {b.items.map((item, j) => (
                  <li key={j} className="pl-1">
                    {inline(item, `${key}-${j}`, known)}
                    {j === b.items.length - 1 ? end : null}
                  </li>
                ))}
              </List>
            );
          }
          default:
            return <p key={key}>{inline(b.text, key, known)}{end}</p>;
        }
      })}
    </div>
  );
}
