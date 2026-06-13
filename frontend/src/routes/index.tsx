import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  CheckCircle2,
  Database,
  Github,
  LineChart,
  Loader2,
  Play,
  Sparkles,
  Wrench,
  Bell,
  Brain,
  Scale,
  Zap,
} from "lucide-react";
import {
  gridAgent,
  type AgentSummary,
  type CaseOption,
  type ComparisonMetrics,
  type ComputeProfile,
  type GridSFMMetrics,
  type GridStats,
  type RecommendationMode,
  type RedispatchProposal,
  type WorkflowLogEntry,
} from "@/lib/gridagent-api";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "GridAgent — AI that keeps a stressed power grid secure" },
      {
        name: "description",
        content:
          "GridAgent screens N-1 contingencies, identifies dangerous failures and proposes redispatch actions — narrating its reasoning at every step.",
      },
      { property: "og:title", content: "GridAgent — N-1 Security AI" },
      {
        property: "og:description",
        content:
          "An AI agent for power grid operators. Detects violations, proposes redispatch, verifies with pandapower.",
      },
    ],
  }),
  component: Index,
});

function Index() {
  const [stats, setStats] = useState<GridStats | null>(null);
  const [agentSummary, setAgentSummary] = useState<AgentSummary | null>(null);
  const [proposal, setProposal] = useState<RedispatchProposal | null>(null);
  const [comparison, setComparison] = useState<ComparisonMetrics | null>(null);
  const [gridsfm, setGridsfm] = useState<GridSFMMetrics | null>(null);
  const [cases, setCases] = useState<CaseOption[]>([]);
  const currentScenario = gridAgent.getScenario();
  const currentOptions = gridAgent.getWorkflowOptions();
  const [selectedCase, setSelectedCase] = useState(currentScenario.networkId);
  const [seed, setSeed] = useState(currentScenario.seed);
  const [computeProfile, setComputeProfile] = useState<ComputeProfile>(currentOptions.computeProfile);
  const [recommendationMode, setRecommendationMode] =
    useState<RecommendationMode>(currentOptions.recommendationMode);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [workflowLogs, setWorkflowLogs] = useState<WorkflowLogEntry[]>([]);
  const [lastRunId, setLastRunId] = useState<string | null>(null);
  const [networkId, setNetworkId] = useState<string>(currentScenario.networkId);
  const [isRunning, setIsRunning] = useState(false);
  const loadedRef = useRef(false);
  const selectedCaseLabel =
    cases.find((item) => item.id === selectedCase)?.label ?? selectedCase;

  async function loadDashboard(): Promise<void> {
    if (isRunning) return;
    try {
      setIsRunning(true);
      setErrorMessage(null);
      setStats(null);
      setAgentSummary(null);
      setProposal(null);
      setComparison(null);
      setGridsfm(null);
      setWorkflowLogs([]);

      const result = await gridAgent.runWorkflow((entry) => {
        setWorkflowLogs((current) => {
          const next = current.filter((item) => item.id !== entry.id);
          return [...next, entry];
        });
      });
      setLastRunId(result.runId);
      setNetworkId(result.networkId);
      setStats(result.stats);
      setAgentSummary(result.agentSummary);
      setProposal(result.proposal);
      setComparison(result.comparison);
      setGridsfm(result.gridsfm);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown backend error.";
      setErrorMessage(message);
      setStats(null);
      setAgentSummary(null);
      setProposal(null);
      setComparison(null);
      setGridsfm(null);
      setLastRunId(null);
      setWorkflowLogs((current) => [
        ...current.filter((item) => item.id !== "failed"),
        { id: "failed", title: "Workflow stopped", detail: message, status: "warn" },
      ]);
    } finally {
      setIsRunning(false);
    }
  }

  function applyScenario(): void {
    gridAgent.setScenario({ networkId: selectedCase, seed });
    gridAgent.setWorkflowOptions({
      computeProfile,
      recommendationMode,
    });
    void loadDashboard();
  }

  function runWithNewSeed(): void {
    const nextSeed = seed + 1;
    setSeed(nextSeed);
    gridAgent.setScenario({ networkId: selectedCase, seed: nextSeed });
    gridAgent.setWorkflowOptions({
      computeProfile,
      recommendationMode,
    });
    void loadDashboard();
  }

  useEffect(() => {
    if (loadedRef.current) return;
    loadedRef.current = true;
    gridAgent.listCases().then(setCases).catch(() => {});
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-40" />
      <div className="pointer-events-none absolute inset-0 bg-hero-glow" />

      <div className="relative mx-auto max-w-6xl px-5 py-6 sm:px-8">
        <Nav />
        <Hero />
        {errorMessage && (
          <div className="mt-6 rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive">
            Backend error: {errorMessage}
          </div>
        )}
        <CaseSelector
          cases={cases}
          selected={selectedCase}
          seed={seed}
          computeProfile={computeProfile}
          recommendationMode={recommendationMode}
          lastRunId={lastRunId}
          onSelect={setSelectedCase}
          onSeedChange={setSeed}
          onComputeProfileChange={setComputeProfile}
          onRecommendationModeChange={setRecommendationMode}
          onRun={applyScenario}
          onRunNewSeed={runWithNewSeed}
          isRunning={isRunning}
        />
        <WorkflowLog logs={workflowLogs} isRunning={isRunning} />
        <Stats stats={stats} agentSummary={agentSummary} />
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <GridStatePanel
            proposal={proposal}
            networkLabel={stats?.network ?? selectedCaseLabel}
            networkId={networkId}
            hasStats={Boolean(stats)}
            runId={lastRunId}
            seed={seed}
          />
          <ReasoningPanel proposal={proposal} agentSummary={agentSummary} />
        </div>
        <ComparisonPanel comparison={comparison} />
        <GridSFMPanel gridsfm={gridsfm} />
        <Architecture />
        <ApiIntegration />
        <Footer />
      </div>
    </div>
  );
}

function CaseSelector({
  cases,
  selected,
  seed,
  computeProfile,
  recommendationMode,
  lastRunId,
  onSelect,
  onSeedChange,
  onComputeProfileChange,
  onRecommendationModeChange,
  onRun,
  onRunNewSeed,
  isRunning,
}: {
  cases: CaseOption[];
  selected: string;
  seed: number;
  computeProfile: ComputeProfile;
  recommendationMode: RecommendationMode;
  lastRunId: string | null;
  onSelect: (id: string) => void;
  onSeedChange: (value: number) => void;
  onComputeProfileChange: (value: ComputeProfile) => void;
  onRecommendationModeChange: (value: RecommendationMode) => void;
  onRun: () => void;
  onRunNewSeed: () => void;
  isRunning: boolean;
}) {
  const grouped = cases.reduce<Record<string, CaseOption[]>>((acc, c) => {
    const group =
      c.id.startsWith("msr_") ? "MSR US States (GridSFM)"
      : c.engines.includes("pandapower") && c.engines.includes("gridsfm") ? "PEGASE / Polish (Pandapower + GridSFM)"
      : c.engines.includes("gridsfm") ? "GridSFM Only"
      : "IEEE Standard (Pandapower)";
    acc[group] = acc[group] ?? [];
    acc[group].push(c);
    return acc;
  }, {});

  const selected_case = cases.find((c) => c.id === selected);

  return (
    <section className="mt-6 rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur">
      <div className="flex flex-wrap items-end gap-3">
        <div className="flex-1 min-w-64">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Select Case / Network
          </label>
          <select
            value={selected}
            onChange={(e) => onSelect(e.target.value)}
            className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm outline-none ring-primary/30 focus:ring"
          >
            {Object.entries(grouped).map(([group, items]) => (
              <optgroup key={group} label={group}>
                {items.map((c) => (
                  <option key={c.id} value={c.id}>{c.label}</option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <div className="w-32">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Seed
          </label>
          <input
            type="number"
            value={seed}
            min={0}
            max={2147483647}
            step={1}
            onChange={(e) => {
              const next = Number.parseInt(e.target.value || "42", 10);
              const clamped = Number.isFinite(next)
                ? Math.min(2147483647, Math.max(0, next))
                : 42;
              onSeedChange(clamped);
            }}
            className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm outline-none ring-primary/30 focus:ring"
          />
        </div>
        <div className="w-40">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Compute profile
          </label>
          <select
            value={computeProfile}
            onChange={(e) => onComputeProfileChange(e.target.value as ComputeProfile)}
            className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm outline-none ring-primary/30 focus:ring"
          >
            <option value="auto">Auto</option>
            <option value="balanced">Balanced</option>
            <option value="fast">Fast</option>
            <option value="max_speed">Max speed</option>
          </select>
        </div>
        <div className="w-44">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Recommendation mode
          </label>
          <select
            value={recommendationMode}
            onChange={(e) => onRecommendationModeChange(e.target.value as RecommendationMode)}
            className="mt-1 w-full rounded-xl border border-border bg-background px-3 py-2 text-sm outline-none ring-primary/30 focus:ring"
          >
            <option value="baseline">Baseline</option>
            <option value="llm_assisted">LLM assisted</option>
          </select>
        </div>
        <button
          onClick={onRun}
          disabled={isRunning}
          className="inline-flex min-w-36 items-center justify-center gap-2 rounded-xl border border-primary/50 bg-primary/10 px-5 py-2 text-sm font-semibold text-primary transition hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
          {isRunning ? "Searching" : "Start Search"}
        </button>
        <button
          onClick={onRunNewSeed}
          disabled={isRunning}
          className="inline-flex min-w-36 items-center justify-center gap-2 rounded-xl border border-border px-5 py-2 text-sm font-semibold transition hover:border-primary/50 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {isRunning ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
          {isRunning ? "Searching" : "New Seed Run"}
        </button>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        Deterministic mode: same network + same seed returns the same result. Use <strong>New Seed Run</strong> to generate a new scenario quickly.
      </p>
      {selected_case && (
        <div className="mt-3 flex flex-wrap gap-2">
          {selected_case.engines.map((eng) => (
            <span
              key={eng}
              className={`rounded-full px-2.5 py-1 text-[11px] font-medium border ${
                eng === "gridsfm"
                  ? "border-accent/40 bg-accent/10 text-accent"
                  : "border-primary/40 bg-primary/10 text-primary"
              }`}
            >
              {eng}
            </span>
          ))}
          <span className="text-xs text-muted-foreground self-center">
            streamlit-style controls applied at frontend workflow level
          </span>
          {selected_case.description && (
            <span className="w-full text-xs text-muted-foreground">{selected_case.description}</span>
          )}
          {lastRunId && (
            <span className="w-full text-xs text-muted-foreground">Last run: {lastRunId}</span>
          )}
        </div>
      )}
    </section>
  );
}

function WorkflowLog({ logs, isRunning }: { logs: WorkflowLogEntry[]; isRunning: boolean }) {
  const visibleLogs =
    logs.length > 0
      ? logs
      : [
          {
            id: "idle",
            title: "Ready",
            detail: "choose a case and start the search",
            status: "ok" as const,
          },
        ];

  return (
    <section className="mt-4 rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Search log
          </div>
          <h3 className="mt-1 text-base font-semibold">
            {isRunning ? "Exploring candidate space" : "Workflow state"}
          </h3>
        </div>
        <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${
          isRunning
            ? "border-accent/40 bg-accent/10 text-accent"
            : "border-primary/40 bg-primary/10 text-primary"
        }`}>
          {isRunning && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
          {isRunning ? "loading" : "idle"}
        </span>
      </div>
      <ol className="mt-4 grid gap-2">
        {visibleLogs.map((entry) => (
          <li
            key={entry.id}
            className="flex items-start gap-3 rounded-xl border border-border/50 bg-background/50 p-3"
          >
            <div className="mt-0.5">
              {entry.status === "running" ? (
                <Loader2 className="h-4 w-4 animate-spin text-accent" />
              ) : entry.status === "warn" ? (
                <Activity className="h-4 w-4 text-[color:var(--warning)]" />
              ) : (
                <CheckCircle2 className="h-4 w-4 text-primary" />
              )}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold">{entry.title}</div>
              <div className="mt-1 break-words font-mono text-xs text-muted-foreground">
                {entry.detail}
                {entry.status === "running" && <span className="animate-blink">▍</span>}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function GridSFMPanel({ gridsfm }: { gridsfm: GridSFMMetrics | null }) {
  const sampleName = gridsfm?.sample_path.split(/[\\/]/).pop() ?? null;

  return (
    <section className="mt-6 rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur">
      <div className="flex items-center gap-2">
        <Zap className="h-4 w-4 text-accent" />
        <h3 className="text-lg font-semibold">GridSFM — Foundation Model Prediction</h3>
        <span className="rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-[10px] font-medium text-accent uppercase tracking-wider">
          GNN · PINN
        </span>
      </div>
      {!gridsfm ? (
        <p className="mt-3 text-sm text-muted-foreground">
          Not available for this case — GridSFM requires{" "}
          <code className="text-xs">GRIDSFM_MODEL_DIR</code> set and the case to be in its sample
          set.
        </p>
      ) : (
        <>
          <p className="mt-1 text-xs text-muted-foreground">
            Loaded real sample metadata for <strong className="text-foreground">{gridsfm.case_name}</strong> from
            the configured GridSFM samples directory.
          </p>
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {[
              { label: "Case", value: gridsfm.case_name, unit: "" },
              { label: "Buses", value: String(gridsfm.bus_count), unit: "count" },
              { label: "Generators", value: String(gridsfm.gen_count), unit: "count" },
              { label: "Lines", value: String(gridsfm.line_count), unit: "count" },
              { label: "Solution", value: gridsfm.has_solution ? "present" : "missing", unit: "" },
              {
                label: "Feasibility",
                value: gridsfm.feasible ? "feasible" : "infeasible",
                unit: gridsfm.feasible ? "✓" : "✗",
              },
            ].map((m) => (
              <div key={m.label} className="rounded-xl border border-border/60 bg-background/60 p-3">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {m.label}
                </div>
                <div className="mt-1 text-base font-semibold">{m.value}</div>
                {m.unit && <div className="text-[10px] text-muted-foreground">{m.unit}</div>}
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center gap-3 rounded-xl border border-border/60 bg-background/40 p-3 text-xs text-muted-foreground">
            <Zap className="h-3.5 w-3.5 text-accent shrink-0" />
            <span>
              Termination status: <strong className="text-foreground">{gridsfm.termination_status}</strong>
              {sampleName ? (
                <>
                  {" "}from sample file <strong className="text-foreground">{sampleName}</strong>.
                </>
              ) : (
                "."
              )}
            </span>
          </div>
        </>
      )}
    </section>
  );
}

function Nav() {
  return (
    <nav className="flex items-center justify-between rounded-2xl border border-border/60 bg-card/40 px-4 py-3 backdrop-blur">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full border border-primary/40 bg-primary/10 text-primary">
          <Activity className="h-4 w-4" />
        </div>
        <span className="font-semibold tracking-tight">GridAgent</span>
        <span className="ml-1 rounded-full border border-accent/40 bg-accent/10 px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-accent">
          beta
        </span>
      </div>
      <div className="hidden items-center gap-7 text-sm text-muted-foreground sm:flex">
        <a href="#docs" className="hover:text-foreground transition">Docs</a>
        <a href="#architecture" className="hover:text-foreground transition">Architecture</a>
        <a href="#demo" className="hover:text-foreground transition">Demo</a>
        <a
          href="#demo"
          className="text-foreground font-medium inline-flex items-center gap-1 hover:text-primary transition"
        >
          Launch <ArrowUpRight className="h-3.5 w-3.5" />
        </a>
      </div>
    </nav>
  );
}

function Hero() {
  return (
    <header className="mt-6 rounded-3xl border border-border/60 bg-card/50 p-8 backdrop-blur sm:p-12">
      <div className="flex flex-wrap gap-2">
        <Tag color="accent">Munich Hackathon 2026</Tag>
        <Tag color="warning">N-1 Security</Tag>
      </div>
      <h1 className="mt-6 max-w-3xl text-balance text-4xl font-bold leading-[1.05] tracking-tight sm:text-5xl md:text-6xl">
        An AI agent that keeps a <span className="text-primary">stressed power grid</span> secure
      </h1>
      <p className="mt-5 max-w-2xl text-base text-muted-foreground sm:text-lg">
        When a line trips, the grid has minutes before overloads cascade. GridAgent screens N-1
        contingencies, identifies dangerous failures, and proposes redispatch actions — narrating
        its reasoning at every step.
      </p>
      <div className="mt-7 flex flex-wrap gap-3">
        <a
          href="#demo"
          className="inline-flex items-center gap-2 rounded-xl border border-border bg-secondary px-4 py-2.5 text-sm font-medium text-secondary-foreground transition hover:border-primary/60 hover:text-primary"
        >
          Run live demo <ArrowUpRight className="h-4 w-4" />
        </a>
        <a
          href="https://github.com"
          className="inline-flex items-center gap-2 rounded-xl border border-border bg-transparent px-4 py-2.5 text-sm font-medium transition hover:border-primary/60"
        >
          <Github className="h-4 w-4" /> View on GitHub
        </a>
      </div>
    </header>
  );
}

function Tag({ children, color }: { children: React.ReactNode; color: "accent" | "warning" }) {
  const cls =
    color === "accent"
      ? "border-accent/40 bg-accent/10 text-accent"
      : "border-[color:var(--warning)]/40 bg-[color:var(--warning)]/10 text-[color:var(--warning)]";
  return (
    <span className={`rounded-full border px-3 py-1 text-xs font-medium ${cls}`}>{children}</span>
  );
}

function toneClass(tone: string | undefined, type: "text" | "sub"): string {
  if (type === "text") {
    if (tone === "danger") return "text-destructive";
    if (tone === "ok") return "text-primary";
    return "";
  }
  return tone === "danger" ? "text-destructive/80" : "text-muted-foreground";
}

function Stats({ stats, agentSummary }: { stats: GridStats | null; agentSummary: AgentSummary | null }) {
  const curtailment = agentSummary == null ? null : agentSummary.accepted ? "not Required" : "Required";
  const curtailmentTone: "ok" | "danger" | undefined = agentSummary == null
    ? undefined
    : agentSummary.accepted ? "ok" : "danger";
  type StatsItem = { label: string; value: string | number; sub: string; tone?: "ok" | "warn" | "danger" };
  const items: StatsItem[] = [
    {
      label: "Post-contingency violations",
      value: stats?.dangerous ?? "-",
      sub: stats?.network ?? "-",
      tone: stats && stats.dangerous > 0 ? "danger" : "ok",
    },
    {
      label: "Greedy steps",
      value: agentSummary?.greedySteps ?? "-",
      sub: "accepted corrective actions",
    },
    {
      label: "Curtailment",
      value: curtailment ?? "-",
      sub: agentSummary ? "cost " + agentSummary.totalCost.toFixed(1) : "waiting for run",
      tone: curtailmentTone,
    },
    {
      label: "Safety delta",
      value: agentSummary != null ? agentSummary.safetyDelta.toFixed(2) : "-",
      sub: agentSummary != null && agentSummary.safetyDelta > 0 ? "improvement" : "no improvement",
      tone: agentSummary != null && agentSummary.safetyDelta > 0 ? "ok" : undefined,
    },
  ];
  return (
    <section id="demo" className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map((it) => (
        <div key={it.label} className="rounded-2xl border border-border/60 bg-card/50 p-4 backdrop-blur">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            {it.label}
          </div>
          <div className={"mt-2 text-3xl font-bold tracking-tight " + toneClass(it.tone, "text")}>
            {it.value}
          </div>
          <div className={"mt-1 text-xs " + toneClass(it.tone, "sub")}>
            {it.sub}
          </div>
        </div>
      ))}
    </section>
  );
}

function GridStatePanel({
  proposal,
  networkLabel,
  networkId,
  hasStats,
  runId,
  seed,
}: {
  proposal: RedispatchProposal | null;
  networkLabel: string;
  networkId: string;
  hasStats: boolean;
  runId: string | null;
  seed: number;
}) {
  return (
    <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Grid state
          </div>
          <h3 className="mt-1 text-lg font-semibold">
            {networkLabel} — after contingency {proposal?.trippedLine ?? "—"}
          </h3>
        </div>
        {hasStats && (
          <span className="rounded-full border border-[color:var(--warning)]/40 bg-[color:var(--warning)]/10 px-3 py-1 text-xs font-medium text-[color:var(--warning)]">
            Overload detected
          </span>
        )}
      </div>
      <div className="mt-2 text-xs text-muted-foreground">
        Run {runId ?? "-"} • Seed {seed}
      </div>
      <NetworkDiagram networkId={networkId} trippedLine={proposal?.trippedLine ?? null} loadingPct={proposal?.loadingPct ?? 0} />
      <div className="mt-4 flex flex-wrap gap-4 text-xs text-muted-foreground">
        <Legend swatch={<span className="inline-block h-[2px] w-6 border-t-2 border-dashed border-destructive" />}>
          tripped line
        </Legend>
        <Legend swatch={<span className="inline-block h-[3px] w-6 bg-[color:var(--warning)]" />}>
          overloaded
        </Legend>
        <Legend swatch={<span className="inline-block h-3 w-3 rounded-full border border-accent bg-accent/20" />}>
          generator
        </Legend>
      </div>
    </div>
  );
}

function Legend({ swatch, children }: { swatch: React.ReactNode; children: React.ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2">
      {swatch}
      {children}
    </span>
  );
}


// ─────────────────────────────────────────────────────────────────────────────
// Per-network topology layouts
// ─────────────────────────────────────────────────────────────────────────────
interface BusDef { id: number; x: number; y: number; gen: boolean }
interface EdgeDef { id: number; from: number; to: number; trafo?: boolean }
interface NetLayout { buses: BusDef[]; edges: EdgeDef[] }

// IEEE 9-bus (WSCC), pandapower 0-indexed buses + lines
const LAYOUT_IEEE9: NetLayout = {
  buses: [
    { id: 0, x: 250, y: 22,  gen: true  },
    { id: 1, x:  38, y: 212, gen: true  },
    { id: 2, x: 462, y: 212, gen: true  },
    { id: 3, x: 250, y: 80,  gen: false },
    { id: 4, x: 375, y: 148, gen: false },
    { id: 5, x: 375, y: 210, gen: false },
    { id: 6, x: 250, y: 252, gen: false },
    { id: 7, x: 125, y: 210, gen: false },
    { id: 8, x: 125, y: 148, gen: false },
  ],
  edges: [
    { id: 0, from: 0, to: 3, trafo: true  }, // 1-4
    { id: 1, from: 3, to: 4               }, // 4-5
    { id: 2, from: 4, to: 5               }, // 5-6
    { id: 3, from: 2, to: 5, trafo: true  }, // 3-6
    { id: 4, from: 5, to: 6               }, // 6-7
    { id: 5, from: 6, to: 7               }, // 7-8
    { id: 6, from: 7, to: 1, trafo: true  }, // 8-2
    { id: 7, from: 7, to: 8               }, // 8-9
    { id: 8, from: 8, to: 3               }, // 9-4
  ],
};

// IEEE 14-bus (MATPOWER), 5 generators, 0-indexed
const LAYOUT_IEEE14: NetLayout = {
  buses: [
    { id:  0, x:  65, y:  50, gen: true  },
    { id:  1, x: 190, y:  30, gen: true  },
    { id:  2, x: 305, y:  62, gen: true  },
    { id:  3, x: 235, y: 112, gen: false },
    { id:  4, x: 128, y: 132, gen: false },
    { id:  5, x: 420, y:  88, gen: true  },
    { id:  6, x: 378, y: 148, gen: false },
    { id:  7, x: 460, y: 132, gen: true  },
    { id:  8, x: 368, y: 192, gen: false },
    { id:  9, x: 292, y: 208, gen: false },
    { id: 10, x: 378, y: 232, gen: false },
    { id: 11, x: 332, y: 255, gen: false },
    { id: 12, x: 238, y: 252, gen: false },
    { id: 13, x: 168, y: 212, gen: false },
  ],
  edges: [
    { id:  0, from:  0, to:  1               },
    { id:  1, from:  0, to:  4               },
    { id:  2, from:  1, to:  2               },
    { id:  3, from:  1, to:  3               },
    { id:  4, from:  1, to:  4               },
    { id:  5, from:  2, to:  3               },
    { id:  6, from:  3, to:  4               },
    { id:  7, from:  3, to:  6, trafo: true  },
    { id:  8, from:  3, to:  8, trafo: true  },
    { id:  9, from:  4, to:  5, trafo: true  },
    { id: 10, from:  5, to: 10               },
    { id: 11, from:  5, to: 11               },
    { id: 12, from:  5, to: 12               },
    { id: 13, from:  6, to:  7, trafo: true  },
    { id: 14, from:  6, to:  8               },
    { id: 15, from:  8, to:  9               },
    { id: 16, from:  8, to: 13               },
    { id: 17, from:  9, to: 10               },
    { id: 18, from: 11, to: 12               },
    { id: 19, from: 12, to: 13               },
  ],
};

function buildRingLayout(display: number): NetLayout {
  const buses: BusDef[] = [];
  const edges: EdgeDef[] = [];
  const cx = 250, cy = 148;
  const hasInner = display > 12;
  const outer = hasInner ? Math.ceil(display * 0.65) : display;
  const inner = display - outer;
  const outerR = hasInner ? 118 : 112;
  const innerR = 58;
  for (let i = 0; i < outer; i++) {
    const a = (2 * Math.PI * i / outer) - Math.PI / 2;
    buses.push({ id: i, x: Math.round(cx + outerR * Math.cos(a)), y: Math.round(cy + outerR * Math.sin(a)), gen: i % Math.max(2, Math.floor(outer / 4)) === 0 });
    edges.push({ id: i, from: i, to: (i + 1) % outer });
  }
  for (let i = 0; i < inner; i++) {
    const a = (2 * Math.PI * i / inner) - Math.PI / 2;
    buses.push({ id: outer + i, x: Math.round(cx + innerR * Math.cos(a)), y: Math.round(cy + innerR * Math.sin(a)), gen: false });
    edges.push({ id: outer + i, from: outer + i, to: outer + (i + 1) % inner });
    edges.push({ id: outer + inner + i, from: i % outer, to: outer + i });
  }
  return { buses, edges };
}

const BUS_COUNTS: Record<string, number> = {
  ieee9: 9, case9: 9,
  ieee14: 14, case14: 14,
  ieee30: 30, case30: 30,
  ieee39: 39, case39: 39,
  ieee57: 57, case57: 57,
  ieee118: 118, case118: 118,
  case300: 300,
};
const DISPLAY_COUNTS: Record<string, number> = {
  ieee30: 30, case30: 30,
  ieee39: 20, case39: 20,
  ieee57: 20, case57: 20,
  ieee118: 18, case118: 18,
  case300: 18,
};

function resolveLayout(networkId: string | null | undefined): { layout: NetLayout; busCount: number } {
  if (!networkId) return { layout: buildRingLayout(12), busCount: 0 };
  const key = networkId.toLowerCase().replace(/[^a-z0-9]/g, '');
  if (key === 'ieee9' || key === 'case9') return { layout: LAYOUT_IEEE9, busCount: 9 };
  if (key === 'ieee14' || key === 'case14') return { layout: LAYOUT_IEEE14, busCount: 14 };
  const busCount = BUS_COUNTS[key] ?? 0;
  const display = DISPLAY_COUNTS[key] ?? 12;
  return { layout: buildRingLayout(display), busCount };
}

function parseComponent(id: string): { type: string; idx: number } | null {
  const m = id.match(/^(line|gen(?:erator)?)_(\d+)$/);
  if (!m) return null;
  return { type: m[1].startsWith('gen') ? 'generator' : 'line', idx: parseInt(m[2], 10) };
}

function NetworkDiagram({
  networkId,
  trippedLine,
  loadingPct,
}: {
  networkId: string | null | undefined;
  trippedLine: string | null;
  loadingPct: number;
}) {
  const { layout, busCount } = resolveLayout(networkId);
  const { buses, edges } = layout;

  let trippedEdgeId: number | null = null;
  let trippedBusId: number | null = null;
  if (trippedLine) {
    const parsed = parseComponent(trippedLine);
    if (parsed) {
      if (parsed.type === 'line') {
        const edge = edges.find((e) => e.id === parsed.idx) ?? edges[parsed.idx % Math.max(1, edges.length)];
        if (edge) trippedEdgeId = edge.id;
      } else {
        const genBuses = buses.filter((b) => b.gen);
        const bus = buses.find((b) => b.id === parsed.idx && b.gen) ?? genBuses[parsed.idx % Math.max(1, genBuses.length)];
        if (bus) trippedBusId = bus.id;
      }
    }
  }

  const busMap = new Map(buses.map((b) => [b.id, b]));

  const subtitle = busCount > buses.length
    ? networkId + ' — ' + buses.length + ' of ' + busCount + ' buses shown'
    : networkId + ' — ' + busCount + (busCount === 1 ? ' bus' : ' buses');

  return (
    <div className="mt-4 rounded-xl border border-border/60 bg-background/60 p-3">
      <p className="mb-2 text-[11px] text-muted-foreground">{subtitle}</p>
      <svg viewBox="0 0 500 280" className="h-64 w-full">
        {edges.map((edge) => {
          const a = busMap.get(edge.from);
          const b = busMap.get(edge.to);
          if (!a || !b) return null;
          const isTripped = edge.id === trippedEdgeId;
          return (
            <path
              key={'e' + edge.id}
              d={'M' + a.x + ' ' + a.y + ' L' + b.x + ' ' + b.y}
              stroke={isTripped ? 'var(--destructive)' : edge.trafo ? 'var(--color-primary)' : 'var(--color-muted-foreground)'}
              strokeOpacity={isTripped ? 1 : edge.trafo ? 0.7 : 0.35}
              strokeWidth={isTripped ? 2.5 : edge.trafo ? 2 : 1.5}
              strokeDasharray={isTripped ? '6 5' : 'none'}
              fill="none"
            />
          );
        })}
        {trippedEdgeId !== null && (() => {
          const edge = edges.find((e) => e.id === trippedEdgeId);
          if (!edge) return null;
          const a = busMap.get(edge.from);
          const b = busMap.get(edge.to);
          if (!a || !b) return null;
          const mx = (a.x + b.x) / 2;
          const my = (a.y + b.y) / 2;
          return (
            <g key="trip-label">
              <text x={mx} y={my - 14} textAnchor="middle" fontSize="10" fill="var(--destructive)" fontStyle="italic">tripped</text>
              <text x={mx} y={my + 2}  textAnchor="middle" fontSize="11" fill="var(--destructive)" fontWeight="600">{loadingPct}% loading</text>
            </g>
          );
        })()}
        {trippedBusId !== null && (() => {
          const b = busMap.get(trippedBusId);
          if (!b) return null;
          return (
            <g key="trip-bus">
              <circle cx={b.x} cy={b.y} r={14} fill="var(--destructive)" fillOpacity={0.15} stroke="var(--destructive)" strokeWidth={1.5} />
              <text x={b.x} y={b.y - 20} textAnchor="middle" fontSize="10" fill="var(--destructive)" fontStyle="italic">tripped</text>
              <text x={b.x} y={b.y + 26} textAnchor="middle" fontSize="11" fill="var(--destructive)" fontWeight="600">{loadingPct}% loading</text>
            </g>
          );
        })()}
        {buses.map((bus) => {
          const isTripped = bus.id === trippedBusId;
          return (
            <g key={'b' + bus.id}>
              {isTripped && <circle cx={bus.x} cy={bus.y} r={bus.gen ? 12 : 9} fill="var(--destructive)" fillOpacity={0.2} />}
              <circle
                cx={bus.x}
                cy={bus.y}
                r={bus.gen ? 8 : 5}
                fill={isTripped ? 'var(--destructive)' : bus.gen ? 'transparent' : 'var(--color-muted-foreground)'}
                stroke={isTripped ? 'var(--destructive)' : bus.gen ? 'var(--accent)' : 'var(--color-muted-foreground)'}
                strokeWidth={bus.gen ? 2 : 1}
                fillOpacity={isTripped ? 1 : 0.7}
              />
              {bus.gen && !isTripped && (
                <text x={bus.x} y={bus.y + 3} textAnchor="middle" fontSize="8" fill="var(--accent)" fontWeight="700">G</text>
              )}
              <text
                x={bus.x}
                y={bus.y + (bus.y > 148 ? 20 : -13)}
                textAnchor="middle"
                fontSize="9"
                fill="var(--color-muted-foreground)"
              >
                {bus.id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}


function ReasoningPanel({
  proposal,
  agentSummary,
}: {
  proposal: RedispatchProposal | null;
  agentSummary: AgentSummary | null;
}) {
  const STOP_REASON_LABELS: Record<string, string> = {
    already_stable: "post-contingency case was already stable",
    stable: "reached a stable grid state",
    no_candidates: "no candidate actions were generated",
    no_converged_candidate: "no candidate action produced a converged power flow",
    no_improving_candidate: "no candidate improved the score",
    max_steps_reached: "maximum greedy step count reached",
  };
  const stopLabel = agentSummary
    ? (STOP_REASON_LABELS[agentSummary.stopReason] ?? agentSummary.stopReason)
    : null;

  return (
    <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        Agent reasoning
      </div>
      <h3 className="mt-1 text-lg font-semibold">Live redispatch proposal</h3>

      {agentSummary && (
        <div className="mt-3 rounded-xl border border-border/60 bg-background/50 p-3 text-xs space-y-1">
          <div className="flex gap-2">
            <span className="text-muted-foreground w-28 shrink-0">Stop reason</span>
            <span className="font-mono">{stopLabel}</span>
          </div>
          <div className="flex gap-2">
            <span className="text-muted-foreground w-28 shrink-0">Greedy steps</span>
            <span className="font-mono">{agentSummary.greedySteps}</span>
          </div>
          <div className="flex gap-2">
            <span className="text-muted-foreground w-28 shrink-0">Curtailment</span>
            <span className={`font-mono font-semibold ${agentSummary.accepted ? "text-primary" : "text-destructive"}`}>
              {agentSummary.accepted ? "not Required" : "Required"}
            </span>
          </div>
        </div>
      )}

      <ol className="mt-4 space-y-4">
        {(proposal?.steps ?? []).map((s) => (
          <li key={s.id} className="flex gap-3">
            <div className="mt-0.5">
              {s.status === "running" ? (
                <Loader2 className="h-5 w-5 animate-spin text-accent" />
              ) : (
                <CheckCircle2 className="h-5 w-5 text-primary" />
              )}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-semibold">{s.title}</div>
              <div className="mt-1 font-mono text-xs text-muted-foreground">
                {s.detail}
                {s.status === "running" && <span className="animate-blink">▍</span>}
              </div>
            </div>
          </li>
        ))}
      </ol>

      {proposal && (
        <>
          <p className="mt-5 border-l-2 border-primary/50 pl-3 text-sm italic text-muted-foreground">
            “{proposal.summary}”
          </p>
          <div className="mt-4 flex flex-wrap gap-2">
            {proposal.tools.map((t) => (
              <span
                key={t}
                className="inline-flex items-center gap-1 rounded-full border border-border bg-secondary/60 px-2.5 py-1 text-[11px] text-secondary-foreground"
              >
                <Sparkles className="h-3 w-3 text-primary" /> {t}
              </span>
            ))}
            <span className="inline-flex items-center rounded-full border border-accent/40 bg-accent/10 px-2.5 py-1 text-[11px] text-accent">
              {proposal.model}
            </span>
          </div>
        </>
      )}
    </div>
  );
}

function ComparisonPanel({ comparison }: { comparison: ComparisonMetrics | null }) {
  return (
    <section className="mt-6 rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur">
      <div className="flex items-center gap-2">
        <Scale className="h-4 w-4 text-primary" />
        <h3 className="text-lg font-semibold">Baseline vs LLM-Assisted Comparison</h3>
      </div>
      {!comparison ? (
        <p className="mt-3 text-sm text-muted-foreground">Comparison data not available yet.</p>
      ) : (
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <MetricCard label="Winner" value={comparison.winner} />
          <MetricCard label="Baseline score" value={comparison.baselineScore.toFixed(2)} />
          <MetricCard label="LLM-assisted score" value={comparison.llmScore.toFixed(2)} />
          <MetricCard label="Baseline cost" value={comparison.baselineCost.toFixed(2)} />
          <MetricCard label="LLM-assisted cost" value={comparison.llmCost.toFixed(2)} />
        </div>
      )}
    </section>
  );
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-background/60 p-3">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold">{value}</div>
    </div>
  );
}

function Architecture() {
  const steps = [
    { icon: Database, title: "Grid data", sub: "pandapower cases" },
    { icon: Wrench, title: "MCP tools", sub: "run_powerflow · screen_n1" },
    { icon: Brain, title: "LLM agent", sub: "reason · propose · explain", active: true },
    { icon: LineChart, title: "Result analysis", sub: "performance summary · archive findings" },
    { icon: Bell, title: "Alert system", sub: "report critical issues · operator interface" },
  ];
  return (
    <section
      id="architecture"
      className="mt-6 rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur"
    >
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        System architecture
      </div>
      <div className="mt-4 flex flex-wrap items-stretch gap-3 sm:flex-nowrap">
        {steps.map((s, i) => (
          <div key={s.title} className="flex flex-1 items-center gap-2 min-w-[140px]">
            <div
              className={`flex w-full flex-col items-center gap-2 rounded-xl border p-3 text-center ${
                s.active
                  ? "border-accent/60 bg-accent/10"
                  : "border-border bg-background/40"
              }`}
            >
              <s.icon
                className={`h-5 w-5 ${s.active ? "text-accent" : "text-muted-foreground"}`}
              />
              <div className="text-xs font-semibold">{s.title}</div>
              <div className="text-[10px] text-muted-foreground">{s.sub}</div>
            </div>
            {i < steps.length - 1 && (
              <ArrowRight className="hidden h-4 w-4 shrink-0 text-muted-foreground sm:block" />
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function ApiIntegration() {
  return (
    <section id="docs" className="mt-6 rounded-2xl border border-border/60 bg-card/50 p-6 backdrop-blur">
      <div className="flex items-center gap-2">
        <Sparkles className="h-4 w-4 text-primary" />
        <h3 className="text-lg font-semibold">Ready to integrate with your Python backend</h3>
      </div>
      <p className="mt-2 text-sm text-muted-foreground">
        The dev server now proxies backend routes automatically to <span className="font-mono">localhost:8000</span>.
        For production builds, set <span className="font-mono">VITE_GRIDAGENT_API_URL</span> to your backend URL.
      </p>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <CodeBlock
          title="Optional .env override"
          code={`VITE_GRIDAGENT_API_URL=http://localhost:8000`}
        />
        <CodeBlock
          title="Backend API flow"
          code={`POST /api/v1/runs
POST /api/v1/runs/{run_id}/screen
POST /api/v1/runs/{run_id}/recommend
POST /api/v1/runs/{run_id}/compare
GET  /api/v1/runs/{run_id}/events (SSE)

All responses use:
{ "success": true, "data": { ... }, "error": null }`}
        />
      </div>
    </section>
  );
}

function CodeBlock({ title, code }: { title: string; code: string }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border bg-background/70">
      <div className="border-b border-border px-3 py-2 text-[11px] font-mono text-muted-foreground">
        {title}
      </div>
      <pre className="overflow-x-auto p-3 text-xs leading-relaxed text-foreground/90">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function Footer() {
  return (
    <footer className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-border/60 pt-5 text-xs text-muted-foreground">
      <span>© {new Date().getFullYear()} GridAgent · Built for grid operators</span>
      <span className="inline-flex items-center gap-1">
        <span className="h-2 w-2 rounded-full bg-primary animate-pulse" /> agent online
      </span>
    </footer>
  );
}
