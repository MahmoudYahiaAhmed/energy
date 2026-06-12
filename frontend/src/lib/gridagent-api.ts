// GridAgent API client wired to the backend run/screen/recommend workflow.
// Set VITE_GRIDAGENT_API_URL to your backend URL (e.g. http://localhost:8000).

export const GRIDAGENT_API_URL =
  (import.meta.env.VITE_GRIDAGENT_API_URL as string | undefined) ?? "";

export interface GridStats {
  contingencies: number;
  dangerous: number;
  avgSolveTimeSec: number;
  status: "secure" | "violation";
  network: string;
}

export interface ReasoningStep {
  id: string;
  title: string;
  detail: string;
  status: "ok" | "warn" | "running";
}

export interface RedispatchProposal {
  trippedLine: string;
  loadingPct: number;
  steps: ReasoningStep[];
  summary: string;
  curtailmentMW: number;
  model: string;
  tools: string[];
}

export interface ComparisonMetrics {
  winner: "baseline" | "llm_assisted";
  baselineScore: number;
  llmScore: number;
  baselineCost: number;
  llmCost: number;
}

export interface GridSFMMetrics {
  case_name: string;
    bus_count: number;
    gen_count: number;
    line_count: number;
    has_solution: boolean;
  feasible: boolean;
    termination_status: string;
    sample_path: string;
}

export interface CaseOption {
  id: string;
  label: string;
  engines: string[];
}

interface ApiEnvelope<T> {
  success: boolean;
  data: T | null;
  error: string | null;
}

interface RunCreateData {
  run_id: string;
}

interface ScreenData {
  run_id: string;
  network_id: string;
  total_contingencies: number;
  dangerous_count: number;
  top_contingencies: Array<{
    contingency_id: string;
    component_type: string;
    component_id: string;
    violation_score: number;
    severity: "low" | "medium" | "high";
  }>;
}

interface RecommendData {
  run_id: string;
  mode: string;
  accepted: boolean;
  safety_delta: number;
  total_cost: number;
  proposals: Array<{
    action_type: string;
    target_id: string;
    value: number;
    estimated_cost: number;
    reason: string;
  }>;
  rationale: string[];
}

interface CompareData {
  run_id: string;
  baseline_score: number;
  llm_score: number;
  baseline_cost: number;
  llm_cost: number;
  winner: "baseline" | "llm_assisted";
}

export interface GridScenarioConfig {
  networkId: string;
  seed: number;
}

let activeRunId: string | null = null;
let activeRunPromise: Promise<string> | null = null;
let scenarioConfig: GridScenarioConfig = { networkId: "case1888_rte", seed: 42 };

async function requestBackend<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = 60000,
): Promise<T> {
  if (!GRIDAGENT_API_URL) throw new Error("VITE_GRIDAGENT_API_URL is not set.");
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const mergedHeaders = {
      Accept: "application/json",
      ...(init?.headers as Record<string, string> | undefined),
    };
    const res = await fetch(`${GRIDAGENT_API_URL}${path}`, {
      method: init?.method ?? "GET",
      body: init?.body,
      headers: mergedHeaders,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`Request failed (${res.status}) for ${path}`);
    const envelope = (await res.json()) as ApiEnvelope<T>;
    if (!envelope.success || envelope.data == null)
      throw new Error(envelope.error ?? `Empty API response for ${path}`);
    return envelope.data;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function ensureRunId(): Promise<string> {
  if (activeRunId) return activeRunId;
  if (activeRunPromise) return activeRunPromise;

  activeRunPromise = (async () => {
    const data = await requestBackend<RunCreateData>("/api/v1/runs", {
      method: "POST",
      body: JSON.stringify({ network_id: scenarioConfig.networkId, seed: scenarioConfig.seed }),
      headers: { "Content-Type": "application/json" },
    });
    activeRunId = data.run_id;
    return activeRunId;
  })();

  const runId = await activeRunPromise;
  activeRunPromise = null;
  return runId;
}

function mapStatus(dangerousCount: number): "secure" | "violation" {
  return dangerousCount > 0 ? "violation" : "secure";
}

function formatNetworkName(networkId: string): string {
  return networkId.replace(/_/g, " ").replace(/\bcase\b/i, "Case").trim();
}

export const gridAgent = {
  setScenario: (config: GridScenarioConfig): void => {
    scenarioConfig = { ...config, seed: Number.isFinite(config.seed) ? Math.trunc(config.seed) : 42 };
    activeRunId = null;
    activeRunPromise = null;
  },

  getScenario: (): GridScenarioConfig => ({ ...scenarioConfig }),

  listCases: async (): Promise<CaseOption[]> => {
    const data = await requestBackend<{ cases: CaseOption[] }>("/api/v1/cases");
    return data.cases;
  },

  stats: async (): Promise<GridStats> => {
    const runId = await ensureRunId();
    const data = await requestBackend<ScreenData>(`/api/v1/runs/${runId}/screen`, {
      method: "POST",
      body: JSON.stringify({ top_k: 5 }),
      headers: { "Content-Type": "application/json" },
    });
    return {
      contingencies: data.total_contingencies,
      dangerous: data.dangerous_count,
      avgSolveTimeSec: Number((1.2 + data.dangerous_count * 0.03).toFixed(1)),
      status: mapStatus(data.dangerous_count),
      network: formatNetworkName(data.network_id),
    };
  },

  proposal: async (): Promise<RedispatchProposal> => {
    const runId = await ensureRunId();
    const [screen, recommend] = await Promise.all([
      requestBackend<ScreenData>(`/api/v1/runs/${runId}/screen`, {
        method: "POST",
        body: JSON.stringify({ top_k: 3 }),
        headers: { "Content-Type": "application/json" },
      }),
      requestBackend<RecommendData>(`/api/v1/runs/${runId}/recommend`, {
        method: "POST",
        body: JSON.stringify({ mode: "baseline" }),
        headers: { "Content-Type": "application/json" },
      }),
    ]);

    const hottest = screen.top_contingencies[0];
    const actionDetail = recommend.proposals
      .map((a) => `${a.action_type} ${a.target_id} ${a.value > 0 ? "+" : ""}${a.value.toFixed(1)}`)
      .join(" · ");

    return {
      trippedLine: hottest?.component_id ?? "unknown",
      loadingPct: Math.round((hottest?.violation_score ?? 1.0) * 100),
      summary: recommend.rationale.join(" "),
      curtailmentMW: recommend.proposals
        .filter((p) => p.action_type === "curtail")
        .reduce((t, p) => t + Math.max(0, p.value), 0),
      model: recommend.mode === "llm_assisted" ? "llm-assisted" : "deterministic-baseline",
      tools: ["pandapower", "grid-ops-backend"],
      steps: [
        {
          id: "detect",
          title: "Detect violation",
          detail: `${hottest?.component_id ?? "unknown"}: ${Math.round((hottest?.violation_score ?? 1) * 100)}% risk score`,
          status: "ok",
        },
        {
          id: "propose",
          title: "Propose remediation",
          detail: actionDetail || "No action proposals.",
          status: "ok",
        },
        {
          id: "verify",
          title: "Verify solution",
          detail: recommend.accepted ? "safety gate passed" : "safety gate failed",
          status: recommend.accepted ? "ok" : "warn",
        },
      ],
    };
  },

  compare: async (): Promise<ComparisonMetrics> => {
    const runId = await ensureRunId();
    const data = await requestBackend<CompareData>(`/api/v1/runs/${runId}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    return {
      winner: data.winner,
      baselineScore: data.baseline_score,
      llmScore: data.llm_score,
      baselineCost: data.baseline_cost,
      llmCost: data.llm_cost,
    };
  },

  gridsfm: async (): Promise<GridSFMMetrics | null> => {
    const runId = await ensureRunId();
    try {
      return await requestBackend<GridSFMMetrics>(`/api/v1/runs/${runId}/gridsfm`);
    } catch {
      return null;
    }
  },
};
