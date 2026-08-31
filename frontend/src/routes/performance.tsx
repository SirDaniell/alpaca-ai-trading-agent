import { useQuery } from "@tanstack/react-query";
import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/AppShell";
import { Metric, Panel } from "@/components/Panel";
import { api, fmtMoney, fmtPct, fmtSigned } from "@/lib/api";
import { fmtDateTime } from "@/lib/api";

export const Route = createFileRoute("/performance")({
  head: () => ({
    meta: [
      { title: "AXE Genesis — Performance Analytics" },
      {
        name: "description",
        content:
          "Multi-period PnL performance for the AXE Genesis options agent: session, day, week and month PnL, win rate, equity, cash and buying power.",
      },
      { property: "og:title", content: "AXE Genesis — Performance Analytics" },
      {
        property: "og:description",
        content: "Session, day, week and month PnL plus key trading KPIs for the AXE Genesis agent.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: PerformancePage,
});

function PerformancePage() {
  const { data: summary } = useQuery({
    queryKey: ["summary"],
    queryFn: api.summary,
    refetchInterval: 10_000,
  });
  const { data: status } = useQuery({
    queryKey: ["status"],
    queryFn: api.status,
    refetchInterval: 10_000,
  });

  const periods = summary
    ? [
        { label: "Session PnL", pnl: summary.session_pnl, pct: summary.session_pnl_pct, accent: "blue" as const },
        { label: "Day PnL", pnl: summary.day_pnl, pct: summary.day_pnl_pct, accent: "gold" as const },
        { label: "Week PnL", pnl: summary.week_pnl, pct: summary.week_pnl_pct, accent: "silver" as const },
        { label: "Month PnL", pnl: summary.month_pnl, pct: summary.month_pnl_pct },
      ]
    : [];

  return (
    <AppShell>
      <div className="space-y-4 sm:space-y-5">
        <Panel
          title="Performance"
          subtitle={
            summary ? `Session started ${fmtDateTime(summary.session_start_time)}` : "Loading…"
          }
        >
          <div className="grid grid-cols-2 gap-2.5 sm:gap-3 lg:grid-cols-4">
            {periods.map((p) => (
              <Metric
                key={p.label}
                label={p.label}
                value={fmtSigned(p.pnl)}
                sub={fmtPct(p.pct)}
                tone={p.pnl >= 0 ? "up" : "down"}
                accent={p.accent}
              />
            ))}
          </div>
        </Panel>

        <Panel title="Key indicators" subtitle="Trading account KPIs">
          <div className="grid grid-cols-2 gap-2.5 sm:gap-3 lg:grid-cols-3">
            <Metric label="Portfolio equity" value={fmtMoney(summary?.current_equity ?? status?.portfolio_value ?? 0)} accent="blue" />
            <Metric label="Cash balance" value={fmtMoney(status?.cash ?? 0)} accent="silver" />
            <Metric label="Buying power" value={fmtMoney(status?.buying_power ?? 0)} accent="gold" />
            <Metric label="Total trades" value={String(summary?.total_trades ?? 0)} />
            <Metric
              label="Win rate"
              value={`${(summary?.win_rate_pct ?? 0).toFixed(1)}%`}
              tone={(summary?.win_rate_pct ?? 0) >= 50 ? "up" : "down"}
            />
            <Metric
              label="Wins / losses"
              value={`${summary?.wins ?? 0} / ${summary?.losses ?? 0}`}
              sub={`${summary?.wins ?? 0} wins · ${summary?.losses ?? 0} losses`}
            />
          </div>
        </Panel>
      </div>
    </AppShell>
  );
}
