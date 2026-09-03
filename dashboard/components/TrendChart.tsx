import type { HistoryRun } from "@/lib/data";

// Metric-over-runs SVG trend, geometry adapted from the mockup. Domain is 0.70–1.00 so our real
// metrics (~0.90+) and the 0.80 gate line all fit with headroom.
const T = { top: 22, bot: 186, left: 46, right: 614 };
const LO = 0.7;
const HI = 1.0;

const yv = (v: number) => +(T.bot - ((v - LO) / (HI - LO)) * (T.bot - T.top)).toFixed(1);

export function TrendChart({ runs, threshold }: { runs: HistoryRun[]; threshold: number }) {
  const n = runs.length;
  const xi = (i: number) =>
    +(n <= 1 ? (T.left + T.right) / 2 : T.left + (i * (T.right - T.left)) / (n - 1)).toFixed(1);

  const pts = (key: "faithfulness" | "answer_relevance" | "context_recall") =>
    runs.map((r, i) => `${xi(i)},${yv(r[key])}`).join(" ");

  const faithPts = pts("faithfulness");
  const relPts = pts("answer_relevance");
  const recPts = pts("context_recall");
  const faithArea = `${T.left},${T.bot} ${faithPts} ${T.right},${T.bot}`;
  const gridVals = [0.7, 0.8, 0.9, 1.0];

  const dots: { x: number; y: number; r: number; color: string; tip: string }[] = [];
  runs.forEach((r, i) => {
    const last = i === n - 1;
    dots.push({ x: xi(i), y: yv(r.context_recall), r: last ? 3 : 2, color: "var(--line-rec)", tip: `recall ${r.context_recall.toFixed(2)} · ${r.git_sha}` });
    dots.push({ x: xi(i), y: yv(r.answer_relevance), r: last ? 3 : 2, color: "var(--line-rel)", tip: `relevance ${r.answer_relevance.toFixed(2)} · ${r.git_sha}` });
    dots.push({ x: xi(i), y: yv(r.faithfulness), r: last ? 3.6 : 2.4, color: "var(--line-faith)", tip: `faithfulness ${r.faithfulness.toFixed(2)} · ${r.git_sha}` });
  });

  return (
    <svg viewBox="0 0 640 214" className="block h-auto w-full">
      {gridVals.map((g) => (
        <g key={g}>
          <line x1="46" x2="614" y1={yv(g)} y2={yv(g)} stroke="var(--grid)" strokeWidth="1" />
          <text x="37" y={yv(g) + 3} textAnchor="end" fontSize="9" fill="var(--text-5)" fontFamily="var(--font-geist-mono), monospace">
            {g.toFixed(2)}
          </text>
        </g>
      ))}
      <polygon points={faithArea} fill="var(--accent-soft)" opacity="0.65" />
      <line x1="46" x2="614" y1={yv(threshold)} y2={yv(threshold)} stroke="var(--dash)" strokeWidth="1.2" strokeDasharray="4 4" />
      {n > 1 && (
        <>
          <polyline points={recPts} fill="none" stroke="var(--line-rec)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
          <polyline points={relPts} fill="none" stroke="var(--line-rel)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
          <polyline points={faithPts} fill="none" stroke="var(--line-faith)" strokeWidth="2" strokeLinejoin="round" strokeLinecap="round" />
        </>
      )}
      {dots.map((d, i) => (
        <circle key={i} cx={d.x} cy={d.y} r={d.r} fill={d.color} stroke="var(--panel)" strokeWidth="1.2">
          <title>{d.tip}</title>
        </circle>
      ))}
      {runs.map((r, i) => (
        <text key={i} x={xi(i)} y="205" textAnchor="middle" fontSize="8.5" fill="var(--text-5)" fontFamily="var(--font-geist-mono), monospace">
          {r.git_sha.slice(0, 6)}
        </text>
      ))}
    </svg>
  );
}
