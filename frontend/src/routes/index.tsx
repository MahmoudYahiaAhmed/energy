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
  Sparkles,
  Wrench,
  Bell,
  Brain,
  Scale,
  Zap,
} from "lucide-react";
import {
  gridAgent,
  type CaseOption,
  type ComparisonMetrics,
  type GridSFMMetrics,
  type GridStats,
  type RedispatchProposal,
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
  const [proposal, setProposal] = useState<RedispatchProposal | null>(null);
  const [comparison, setComparison] = useState<ComparisonMetrics | null>(null);
  const [gridsfm, setGridsfm] = useState<GridSFMMetrics | null>(null);
  const [cases, setCases] = useState<CaseOption[]>([]);
  const [selectedCase, setSelectedCase] = useState(gridAgent.getScenario().networkId);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const loadedRef = useRef(false);

  async function loadDashboard(): Promise<void> {
    try {
      setErrorMessage(null);
      const [nextStats, nextProposal, nextComparison, nextGridsfm] = await Promise.all([
        gridAgent.stats(),
        gridAgent.proposal(),
        gridAgent.compare(),
        gridAgent.gridsfm(),
      ]);
      setStats(nextStats);
      setProposal(nextProposal);
      setComparison(nextComparison);
      setGridsfm(nextGridsfm);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown backend error.";
      setErrorMessage(message);
      setStats(null);
      setProposal(null);
      setComparison(null);
      setGridsfm(null);
    }
  }

  function applyScenario(): void {
    gridAgent.setScenario({ networkId: selectedCase, seed: 42 });
    void loadDashboard();
  }

  useEffect(() => {
    if (loadedRef.current) return;
    loadedRef.current = true;
    gridAgent.listCases().then(setCases).catch(() => {});
    void loadDashboard();
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
          onSelect={setSelectedCase}
          onRun={applyScenario}
        />
        <Stats stats={stats} />
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <GridStatePanel proposal={proposal} />
          <ReasoningPanel proposal={proposal} />
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
  onSelect,
  onRun,
}: {
  cases: CaseOption[];
  selected: string;
  onSelect: (id: string) => void;
  onRun: () => void;
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
        <button
          onClick={onRun}
          className="rounded-xl border border-primary/50 bg-primary/10 px-5 py-2 text-sm font-semibold text-primary transition hover:bg-primary/20"
        >
          Run Scenario
        </button>
      </div>
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
          <span className="text-xs text-muted-foreground self-center">available for this case</span>
        </div>
      )}
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
        <Tag color="accent">Munich Hackathon 2025</Tag>
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

function Stats({ stats }: { stats: GridStats | null }) {
  const items = [
    { label: "Contingencies", value: stats?.contingencies ?? "—", sub: stats?.network ?? "IEEE 118-bus" },
    {
      label: "Dangerous",
      value: stats?.dangerous ?? "—",
      sub: "require redispatch",
      tone: "danger" as const,
    },
    { label: "Avg. solve time", value: stats ? `${stats.avgSolveTimeSec}s` : "—", sub: "per contingency" },
    {
      label: "Status",
      value: stats?.status === "secure" ? "N-1 ✓" : "N-1 ⚠",
      sub: stats?.status === "secure" ? "all lines secure" : "violation detected",
      tone: stats?.status === "secure" ? ("ok" as const) : ("warn" as const),
    },
  ];
  return (
    <section id="demo" className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {items.map((it) => (
          <div key={it.label} className="rounded-2xl border border-border/60 bg-card/50 p-4 backdrop-blur">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              {it.label}
            </div>
            <div className={`mt-2 text-3xl font-bold tracking-tight ${it.tone === "danger" ? "text-destructive" : it.tone === "ok" ? "text-primary" : it.tone === "warn" ? "text-[color:var(--warning)]" : ""}`}>
              {it.value}
            </div>
            <div className={`mt-1 text-xs ${it.tone === "danger" ? "text-destructive/80" : "text-muted-foreground"}`}>
              {it.sub}
            </div>
          </div>
        ))}
      </section>
  );
}

function GridStatePanel({ proposal }: { proposal: RedispatchProposal | null }) {
  return (
    <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Grid state
          </div>
          <h3 className="mt-1 text-lg font-semibold">
            IEEE 118-bus — after line {proposal?.trippedLine ?? "—"} trip
          </h3>
        </div>
        <span className="rounded-full border border-[color:var(--warning)]/40 bg-[color:var(--warning)]/10 px-3 py-1 text-xs font-medium text-[color:var(--warning)]">
          Overload detected
        </span>
      </div>
      <GridDiagram loadingPct={proposal?.loadingPct ?? 109} />
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

function GridDiagram({ loadingPct }: { loadingPct: number }) {
  // Stylized 6-bus subnetwork inspired by the screenshot
  const buses = [
    { id: "15", x: 70, y: 70, gen: true },
    { id: "17", x: 200, y: 55, hot: true },
    { id: "13", x: 430, y: 95, gen: true },
    { id: "16", x: 80, y: 200 },
    { id: "19", x: 250, y: 220 },
    { id: "20", x: 370, y: 215 },
  ];
  return (
    <div className="mt-4 rounded-xl border border-border/60 bg-background/60 p-3">
      <svg viewBox="0 0 500 280" className="h-64 w-full">
        {/* base lines */}
        <g stroke="var(--color-muted-foreground)" strokeOpacity="0.4" strokeWidth="1.5" fill="none">
          <path d="M70 70 L80 200" />
          <path d="M80 200 L250 220" />
          <path d="M250 220 L370 215" />
          <path d="M370 215 L430 95" />
          <path d="M200 55 L430 95" />
        </g>
        {/* overloaded lines */}
        <g
          stroke="var(--warning)"
          strokeWidth="3"
          strokeLinecap="round"
          fill="none"
          className="pulse-line"
        >
          <path d="M70 70 L200 55" />
          <path d="M200 55 L250 220" />
          <path d="M250 220 L430 95" />
        </g>
        {/* tripped line */}
        <path
          d="M200 55 L430 95"
          stroke="var(--destructive)"
          strokeWidth="2"
          strokeDasharray="6 6"
          className="animate-dash"
          fill="none"
        />
        <text x="285" y="78" fontSize="10" fill="var(--destructive)" fontStyle="italic">
          tripped
        </text>
        <text x="285" y="40" fontSize="11" fill="var(--warning)" fontWeight="600">
          {loadingPct}% loading
        </text>
        {/* buses */}
        {buses.map((b) => (
          <g key={b.id}>
            <circle
              cx={b.x}
              cy={b.y}
              r={b.hot ? 9 : 6}
              fill={b.hot ? "var(--warning)" : b.gen ? "transparent" : "var(--color-muted-foreground)"}
              stroke={b.gen ? "var(--accent)" : b.hot ? "var(--warning)" : "var(--color-muted-foreground)"}
              strokeWidth={b.gen ? 2 : 1}
            />
            {b.gen && (
              <text x={b.x} y={b.y + 3} textAnchor="middle" fontSize="9" fill="var(--accent)" fontWeight="700">
                G
              </text>
            )}
            <text
              x={b.x}
              y={b.y + (b.y > 150 ? 22 : -14)}
              textAnchor="middle"
              fontSize="10"
              fill="var(--color-muted-foreground)"
            >
              bus {b.id}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function ReasoningPanel({ proposal }: { proposal: RedispatchProposal | null }) {
  return (
    <div className="rounded-2xl border border-border/60 bg-card/50 p-5 backdrop-blur">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
        Agent reasoning
      </div>
      <h3 className="mt-1 text-lg font-semibold">Live redispatch proposal</h3>

      <ol className="mt-5 space-y-4">
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
    { icon: Database, title: "Grid data", sub: "pandapower / SMARD" },
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
        Point this UI at the backend by setting one env var. The UI calls a run-based API flow:
        create run, screen contingencies, then request a remediation recommendation.
      </p>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <CodeBlock
          title=".env"
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
