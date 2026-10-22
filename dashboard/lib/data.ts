// Types + client fetch for the exported eval JSON. The dashboard is a pure JSON consumer
// (FR-D4): it reads static files from /public/data, never a live DB. `npm run sync-data` copies
// the latest eval exports into public/data before dev/build.

export type Retrieved = {
  chunk_id: string;
  text: string;
  dense_score: number | null;
  sparse_score: number | null;
  fused_score: number | null;
  rerank_score: number | null;
};

export type Attribution = "pass" | "retrieval_fail" | "generation_fail" | null;

export type ItemResult = {
  question: string;
  faithfulness: number;
  answer_relevance: number;
  context_recall: number;
  attribution: Attribution;
  generated_answer: string;
  retrieved: Retrieved[];
};

export type RunResult = {
  run_id: string;
  git_sha: string;
  timestamp: string;
  per_item: ItemResult[];
  means: { faithfulness: number; answer_relevance: number; context_recall: number };
  attribution_counts: { retrieval_fail: number; generation_fail: number };
};

export type HistoryRun = {
  run_id: string;
  git_sha: string;
  timestamp: string;
  faithfulness: number;
  answer_relevance: number;
  context_recall: number;
  retrieval_fail: number;
  generation_fail: number;
  items: number;
  gate_pass: boolean;
};

export type History = {
  faithfulness_threshold: number;
  judge_model: string;
  generation_model: string;
  runs: HistoryRun[];
};

export type LatencyStage = {
  stage: string;
  n: number;
  mean_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
};

export type LatencyReport = {
  measured_utc?: string;
  generation_model?: string;
  stages?: LatencyStage[];
};

export function stageP95(report: LatencyReport | null, stage: string): number | null {
  const s = report?.stages?.find((x) => x.stage === stage);
  return s ? s.p95_ms : null;
}


// sentinel/eval/run_guardrails.py -> data/guardrails/report.json (v2).
export type Rate = { flagged: number; of: number; rate: number | null };

export type AttackBlock = {
  attacks: number;
  exposed: number;
  succeeded: number; // mean over repeats
  succeeded_range: [number, number];
  success_rate: number | null;
  success_rate_when_exposed: number | null;
};

export type AttackConfigSummary = {
  repeats: number;
  all: AttackBlock;
  dev: AttackBlock;
  heldout: AttackBlock;
  by_attack: Record<string, { succeeded: number; of: number }>;
  succeeded_targets: Record<string, number>;
};

export type PiiKind = { tp: number; fp: number; fn: number; precision: number | null; recall: number | null };

export type PiiSource = {
  precision: number | null;
  recall: number | null;
  tp: number;
  fp: number;
  fn: number;
  per_kind: Record<string, PiiKind>;
  misses: { kind: string; text: string }[];
  false_alarms: { kind: string; text: string }[];
};

export type GuardrailsReport = {
  generated_utc: string;
  generation_model: string;
  threshold: number;
  injection: {
    detectors: {
      detectors: string[];
      recall: Record<"dev" | "heldout", Record<string, Rate> & { missed: string[] }>;
      false_positives: Record<string, Rate> & { flagged_chunks: string[]; unscored: number };
    };
    attacks: { summary: Record<string, AttackConfigSummary> };
  };
  pii: Record<"corpus" | "synthetic", PiiSource>;
  scope: { default_scope_leaks: number };
};

export type DashboardData = {
  latest: RunResult;
  history: History;
  latency: LatencyReport | null;
  guardrails: GuardrailsReport | null;
};

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

export async function loadDashboardData(): Promise<DashboardData> {
  const [latest, history] = await Promise.all([
    getJSON<RunResult>("/data/latest.json"),
    getJSON<History>("/data/history.json"),
  ]);
  let latency: LatencyReport | null = null;
  try {
    latency = await getJSON<LatencyReport>("/data/latency_report.json");
  } catch {
    latency = null;
  }
  // Optional: present once `python -m sentinel.eval.run_guardrails` has run and been synced.
  let guardrails: GuardrailsReport | null = null;
  try {
    guardrails = await getJSON<GuardrailsReport>("/data/guardrails.json");
  } catch {
    guardrails = null;
  }
  return { latest, history, latency, guardrails };
}
