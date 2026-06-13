// GridAgent API client wired to the backend run/screen/recommend workflow.
// Set VITE_GRIDAGENT_API_URL to your backend URL (e.g. http://localhost:8000).

export const GRIDAGENT_API_URL =
  (import.meta.env.VITE_GRIDAGENT_API_URL as string | undefined) ?? "";

const BACKEND_BASE_URL = GRIDAGENT_API_URL.trim().replace(/\/$/, "");

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

export interface WorkflowLogEntry {
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

export interface AgentSummary {
  accepted: boolean;
  greedySteps: number;
  totalCost: number;
  safetyDelta: number;
  stopReason: string;
}

export interface DashboardWorkflowResult {
  runId: string;
  networkId: string;
  stats: GridStats;
  agentSummary: AgentSummary;
  proposal: RedispatchProposal;
  comparison: ComparisonMetrics;
  gridsfm: GridSFMMetrics | null;
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
  description?: string;
}

const STREAMLIT_NETWORK_OPTIONS: CaseOption[] = [
  {
    id: "ieee14",
    label: "IEEE 14-bus",
    description:
      "Small MATPOWER-derived system with generators, lines, loads, and transformers.",
    engines: ["pandapower"],
  },
  {
    id: "ieee9",
    label: "IEEE 9-bus",
    description: "Very small transmission test case for fast demonstrations.",
    engines: ["pandapower"],
  },
  {
    id: "ieee30",
    label: "IEEE 30-bus",
    description:
      "Medium-small test case with more branches and generators than IEEE 14-bus.",
    engines: ["pandapower"],
  },
  {
    id: "ieee57",
    label: "IEEE 57-bus",
    description: "Medium transmission test case with a richer N-1 contingency space.",
    engines: ["pandapower"],
  },
  {
    id: "ieee118",
    label: "IEEE 118-bus",
    description: "Larger IEEE test case suited for DC N-1 screening experiments.",
    engines: ["pandapower"],
  },
  {
    id: "case300",
    label: "IEEE 300-bus",
    description: "Large MATPOWER-derived case for more substantial DC screening runs.",
    engines: ["pandapower"],
  },
];

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

export type RecommendationMode = "baseline" | "llm_assisted";
export type ComputeProfile = "auto" | "balanced" | "fast" | "max_speed";

export interface WorkflowOptions {
  recommendationMode: RecommendationMode;
  computeProfile: ComputeProfile;
}

let activeRunId: string | null = null;
let activeRunPromise: Promise<string> | null = null;
let scenarioConfig: GridScenarioConfig = { networkId: "case1888_rte", seed: 42 };
let scenarioEpoch = 0;
let workflowOptions: WorkflowOptions = {
  recommendationMode: "baseline",
  computeProfile: "fast",
};

function profileToTopK(profile: ComputeProfile): number {
  if (profile === "max_speed") return 2;
  if (profile === "fast") return 5;
  if (profile === "balanced") return 8;
  return 5;
}

async function requestBackend<T>(
  path: string,
  init?: RequestInit,
  timeoutMs: number = 60000,
): Promise<T> {
  if (import.meta.env.PROD && !BACKEND_BASE_URL) {
    throw new Error(
      "VITE_GRIDAGENT_API_URL must be set for production builds (example: https://api.example.com).",
    );
  }
  if (import.meta.env.PROD && BACKEND_BASE_URL) {
    try {
      const parsed = new URL(BACKEND_BASE_URL);
      if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
        throw new Error("invalid protocol");
      }
    } catch {
      throw new Error("VITE_GRIDAGENT_API_URL must be a valid absolute http(s) URL.");
    }
  }
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const parseEnvelope = (envelope: ApiEnvelope<T>, requestUrl: string): T => {
    if (!envelope.success || envelope.data == null) {
      throw new Error(envelope.error ?? `Empty API response for ${requestUrl}`);
    }
    return envelope.data;
  };

  const fetchAndParse = async (requestUrl: string): Promise<T> => {
    const mergedHeaders = {
      Accept: "application/json",
      ...(init?.headers as Record<string, string> | undefined),
    };
    const res = await fetch(requestUrl, {
      method: init?.method ?? "GET",
      body: init?.body,
      headers: mergedHeaders,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(`Request failed (${res.status}) for ${requestUrl}`);
    const envelope = (await res.json()) as ApiEnvelope<T>;
    return parseEnvelope(envelope, requestUrl);
  };

  try {
    const requestUrl = `${BACKEND_BASE_URL}${path}`;
    try {
      return await fetchAndParse(requestUrl);
    } catch (primaryError) {
      if (!import.meta.env.DEV || !BACKEND_BASE_URL) {
        throw primaryError;
      }
      return await fetchAndParse(path);
    }
  } finally {
    clearTimeout(timeoutId);
  }
}

async function ensureRunId(): Promise<string> {
  if (activeRunId) return activeRunId;
  if (activeRunPromise) return activeRunPromise;

  const requestEpoch = scenarioEpoch;
  const requestScenario = { ...scenarioConfig };
  const runPromise = (async () => {
    const data = await requestBackend<RunCreateData>("/api/v1/runs", {
      method: "POST",
      body: JSON.stringify({ network_id: requestScenario.networkId, seed: requestScenario.seed }),
      headers: { "Content-Type": "application/json" },
    });

    // If scenario changed while creating a run, discard stale run ID and build a fresh run.
    if (requestEpoch !== scenarioEpoch) {
      return ensureRunId();
    }

    activeRunId = data.run_id;
    return activeRunId;
  })();
  activeRunPromise = runPromise;

  try {
    return await runPromise;
  } finally {
    if (activeRunPromise === runPromise) {
      activeRunPromise = null;
    }
  }
}

function mapStatus(dangerousCount: number): "secure" | "violation" {
  return dangerousCount > 0 ? "violation" : "secure";
}

function formatNetworkName(networkId: string): string {
  return networkId.replace(/_/g, " ").replace(/\bcase\b/i, "Case").trim();
}

function statsFromScreen(data: ScreenData): GridStats {
  return {
    contingencies: data.total_contingencies,
    dangerous: data.dangerous_count,
    avgSolveTimeSec: Number((1.2 + data.dangerous_count * 0.03).toFixed(1)),
    status: mapStatus(data.dangerous_count),
    network: formatNetworkName(data.network_id),
  };
}

function proposalFromResponses(screen: ScreenData, recommend: RecommendData): RedispatchProposal {
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
}

export const gridAgent = {
  setScenario: (config: GridScenarioConfig): void => {
    scenarioEpoch += 1;
    scenarioConfig = { ...config, seed: Number.isFinite(config.seed) ? Math.trunc(config.seed) : 42 };
    activeRunId = null;
    activeRunPromise = null;
  },

  getScenario: (): GridScenarioConfig => ({ ...scenarioConfig }),

  setWorkflowOptions: (options: Partial<WorkflowOptions>): void => {
    const nextProfile = options.computeProfile ?? workflowOptions.computeProfile;

    workflowOptions = {
      recommendationMode: options.recommendationMode ?? workflowOptions.recommendationMode,
      computeProfile: nextProfile,
    };
  },

  getWorkflowOptions: (): WorkflowOptions => ({ ...workflowOptions }),

  prepareRun: async (): Promise<string> => ensureRunId(),

  listCases: async (): Promise<CaseOption[]> => {
    const data = await requestBackend<{ cases: CaseOption[] }>("/api/v1/cases");
    const merged = new Map<string, CaseOption>();
    for (const item of STREAMLIT_NETWORK_OPTIONS) {
      merged.set(item.id, item);
    }
    for (const item of data.cases) {
      const existing = merged.get(item.id);
      merged.set(item.id, {
        ...existing,
        ...item,
        description: item.description ?? existing?.description,
      });
    }
    return Array.from(merged.values());
  },

  stats: async (): Promise<GridStats> => {
    const runId = await ensureRunId();
    const data = await requestBackend<ScreenData>(`/api/v1/runs/${runId}/screen`, {
      method: "POST",
      body: JSON.stringify({ top_k: profileToTopK(workflowOptions.computeProfile) }),
      headers: { "Content-Type": "application/json" },
    });
    return statsFromScreen(data);
  },

  proposal: async (): Promise<RedispatchProposal> => {
    const runId = await ensureRunId();
    const [screen, recommend] = await Promise.all([
      requestBackend<ScreenData>(`/api/v1/runs/${runId}/screen`, {
        method: "POST",
        body: JSON.stringify({ top_k: profileToTopK(workflowOptions.computeProfile) }),
        headers: { "Content-Type": "application/json" },
      }),
      requestBackend<RecommendData>(`/api/v1/runs/${runId}/recommend`, {
        method: "POST",
        body: JSON.stringify({ mode: workflowOptions.recommendationMode }),
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

  runWorkflow: async (
    onLog?: (entry: WorkflowLogEntry) => void,
  ): Promise<DashboardWorkflowResult> => {
    const emit = (entry: WorkflowLogEntry) => onLog?.(entry);

    emit({
      id: "create",
      title: "Create run",
      detail: `${formatNetworkName(scenarioConfig.networkId)} · seed ${scenarioConfig.seed}`,
      status: "running",
    });
    const runId = await ensureRunId();
    emit({ id: "create", title: "Create run", detail: runId, status: "ok" });

    emit({
      id: "screen",
      title: "Search contingency space",
      detail: "screening N-1 outages and ranking violations",
      status: "running",
    });
    const screen = await requestBackend<ScreenData>(`/api/v1/runs/${runId}/screen`, {
      method: "POST",
      body: JSON.stringify({ top_k: profileToTopK(workflowOptions.computeProfile) }),
      headers: { "Content-Type": "application/json" },
    });
    emit({
      id: "screen",
      title: "Search contingency space",
      detail: `${screen.total_contingencies} contingencies · ${screen.dangerous_count} dangerous`,
      status: screen.dangerous_count > 0 ? "warn" : "ok",
    });

    emit({
      id: "recommend",
      title: "Search corrective actions",
      detail: "evaluating capped redispatch, voltage, tap, load and switching candidates",
      status: "running",
    });
    const recommend = await requestBackend<RecommendData>(`/api/v1/runs/${runId}/recommend`, {
      method: "POST",
      body: JSON.stringify({ mode: workflowOptions.recommendationMode }),
      headers: { "Content-Type": "application/json" },
    });
    emit({
      id: "recommend",
      title: "Search corrective actions",
      detail: `${recommend.proposals.length} proposal(s) · cost ${recommend.total_cost.toFixed(2)}`,
      status: recommend.accepted ? "ok" : "warn",
    });

    emit({
      id: "compare",
      title: "Verify and compare",
      detail: "running baseline and assisted scoring using cached screening",
      status: "running",
    });
    const compare = await requestBackend<CompareData>(`/api/v1/runs/${runId}/compare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
    emit({
      id: "compare",
      title: "Verify and compare",
      detail: `${compare.winner} selected`,
      status: "ok",
    });

    emit({
      id: "gridsfm",
      title: "Load GridSFM metadata",
      detail: "checking sample availability",
      status: "running",
    });
    let gridsfm: GridSFMMetrics | null = null;
    try {
      gridsfm = await requestBackend<GridSFMMetrics>(`/api/v1/runs/${runId}/gridsfm`);
      emit({
        id: "gridsfm",
        title: "Load GridSFM metadata",
        detail: `${gridsfm.case_name} · ${gridsfm.bus_count} buses`,
        status: "ok",
      });
    } catch {
      emit({
        id: "gridsfm",
        title: "Load GridSFM metadata",
        detail: "not available for this run",
        status: "warn",
      });
    }

    return {
      runId,
      networkId: screen.network_id,
      stats: statsFromScreen(screen),
      agentSummary: {
        accepted: recommend.accepted,
        greedySteps: recommend.proposals.length,
        totalCost: recommend.total_cost,
        safetyDelta: recommend.safety_delta,
        stopReason: recommend.accepted ? "stable" : "no_improving_candidate",
      },
      proposal: proposalFromResponses(screen, recommend),
      comparison: {
        winner: compare.winner,
        baselineScore: compare.baseline_score,
        llmScore: compare.llm_score,
        baselineCost: compare.baseline_cost,
        llmCost: compare.llm_cost,
      },
      gridsfm,
    };
  },
};
