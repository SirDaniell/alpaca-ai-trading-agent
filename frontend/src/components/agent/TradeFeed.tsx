import { useQuery } from "@tanstack/react-query";
import { Panel } from "@/components/Panel";
import { api, fmtDateTime, fmtDuration, fmtMoney, fmtPct } from "@/lib/api";
import { cn } from "@/lib/utils";

const reasonClass: Record<string, string> = {
  TAKE_PROFIT: "bg-bull/15 text-bull",
  STOP_LOSS: "bg-destructive/15 text-bear",
  EXPIRY_REACHED: "bg-gold/15 text-gold",
};

export function TradeFeed({ limit = 25 }: { limit?: number }) {
  const { data } = useQuery({
    queryKey: ["trades", limit],
    queryFn: () => api.trades(limit),
    refetchInterval: 15_000,
  });
  const trades = data?.trades ?? [];

  return (
    <Panel title="Execution feed" subtitle={`${trades.length} completed trades`} bodyClassName="p-3 sm:p-4">
      <div className="space-y-2.5">
        {trades.map((t) => {
          const up = t.realized_pnl >= 0;
          return (
            <article key={t.trade_id} className="glass rounded-2xl px-3.5 py-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <p className="num truncate text-[13px] font-semibold">{t.symbol}</p>
                  <p className="mt-0.5 text-[11px] text-muted-foreground">
                    {fmtDateTime(t.start_time)} → {fmtDateTime(t.end_time)} · held{" "}
                    {fmtDuration(t.hold_duration_sec)}
                  </p>
                </div>
                <div className="text-right">
                  <p className={cn("num text-sm font-semibold", up ? "text-bull" : "text-bear")}>
                    {fmtMoney(t.realized_pnl)}
                  </p>
                  <p className={cn("num text-[11px]", up ? "text-bull/80" : "text-bear/80")}>
                    {fmtPct(t.realized_pnl_pct)}
                  </p>
                </div>
              </div>
              <div className="mt-2.5 flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                <span
                  className={cn(
                    "rounded-full px-2 py-0.5 font-semibold tracking-widest uppercase",
                    reasonClass[t.exit_reason] ?? "bg-secondary text-silver",
                  )}
                >
                  {t.exit_reason.replace(/_/g, " ")}
                </span>
                <span className="num">qty {t.qty}</span>
                <span className="num">
                  entry {t.entry_price.toFixed(2)} · exit {t.exit_price.toFixed(2)}
                </span>
              </div>
            </article>
          );
        })}
        {trades.length === 0 && (
          <p className="py-6 text-center text-sm text-muted-foreground">No trades recorded yet.</p>
        )}
      </div>
    </Panel>
  );
}
