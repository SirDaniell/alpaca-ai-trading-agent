import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Panel } from "@/components/Panel";
import { api, fmtMoney, fmtPct } from "@/lib/api";
import type { Position } from "@/lib/api-types";
import { cn } from "@/lib/utils";

function Countdown({ expiresAt }: { expiresAt?: string | undefined }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  if (!expiresAt) return <span className="text-muted-foreground">—</span>;
  const left = Math.max(0, Math.floor((new Date(expiresAt).getTime() - now) / 1000));
  const m = Math.floor(left / 60);
  const s = left % 60;
  return (
    <span className={cn("num", left < 300 ? "text-bear" : "text-gold")}>
      {m}:{String(s).padStart(2, "0")}
    </span>
  );
}

function Row({ p }: { p: Position }) {
  const up = p.unrealized_pl >= 0;
  return (
    <div className="glass rounded-2xl p-3.5 lg:hidden">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold">{p.underlying}</p>
          <p className="num text-[11px] text-muted-foreground">{p.symbol}</p>
        </div>
        <span
          className={cn(
            "rounded-full px-2.5 py-0.5 text-[10px] font-semibold tracking-widest uppercase",
            p.contract_type === "CALL" ? "bg-primary/20 text-primary" : "bg-destructive/20 text-bear",
          )}
        >
          {p.contract_type}
        </span>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 text-[11px] text-muted-foreground">
        <span>
          Qty <span className="num block text-foreground">{p.qty}</span>
        </span>
        <span>
          Fill <span className="num block text-foreground">{p.avg_fill_price.toFixed(2)}</span>
        </span>
        <span>
          Last <span className="num block text-foreground">{p.current_price.toFixed(2)}</span>
        </span>
        <span>
          Unrealized
          <span className={cn("num block", up ? "text-bull" : "text-bear")}>
            {fmtMoney(p.unrealized_pl)}
          </span>
        </span>
        <span>
          Change
          <span className={cn("num block", up ? "text-bull" : "text-bear")}>
            {fmtPct(p.unrealized_plpc)}
          </span>
        </span>
        <span>
          Expiry
          <span className="block">
            <Countdown expiresAt={p.expires_at} />
          </span>
        </span>
      </div>
    </div>
  );
}

export function PositionsPanel() {
  const { data } = useQuery({ queryKey: ["positions"], queryFn: api.positions, refetchInterval: 8000 });
  const positions = data?.positions ?? [];

  return (
    <Panel
      title="Open positions"
      subtitle={`${positions.length} contract${positions.length === 1 ? "" : "s"} live`}
      bodyClassName="p-3 sm:p-4"
    >
      {positions.length === 0 ? (
        <p className="px-1 py-6 text-center text-sm text-muted-foreground">No open positions.</p>
      ) : (
        <>
          <div className="space-y-2.5 lg:hidden">
            {positions.map((p) => (
              <Row key={p.symbol} p={p} />
            ))}
          </div>
          <div className="hidden lg:block">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[11px] tracking-[0.12em] text-muted-foreground uppercase">
                  <th className="px-3 py-2 text-left font-medium">Underlying</th>
                  <th className="px-3 py-2 text-left font-medium">Contract</th>
                  <th className="px-3 py-2 text-right font-medium">Qty</th>
                  <th className="px-3 py-2 text-right font-medium">Avg fill</th>
                  <th className="px-3 py-2 text-right font-medium">Last</th>
                  <th className="px-3 py-2 text-right font-medium">Unrealized</th>
                  <th className="px-3 py-2 text-right font-medium">%</th>
                  <th className="px-3 py-2 text-right font-medium">Expiry in</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((p) => {
                  const up = p.unrealized_pl >= 0;
                  return (
                    <tr key={p.symbol} className="border-t border-border/60">
                      <td className="px-3 py-3 font-semibold">{p.underlying}</td>
                      <td className="num px-3 py-3 text-xs text-muted-foreground">
                        {p.symbol}
                        <span
                          className={cn(
                            "ml-2 rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-widest uppercase",
                            p.contract_type === "CALL"
                              ? "bg-primary/20 text-primary"
                              : "bg-destructive/20 text-bear",
                          )}
                        >
                          {p.contract_type}
                        </span>
                      </td>
                      <td className="num px-3 py-3 text-right">{p.qty}</td>
                      <td className="num px-3 py-3 text-right">{p.avg_fill_price.toFixed(2)}</td>
                      <td className="num px-3 py-3 text-right">{p.current_price.toFixed(2)}</td>
                      <td className={cn("num px-3 py-3 text-right", up ? "text-bull" : "text-bear")}>
                        {fmtMoney(p.unrealized_pl)}
                      </td>
                      <td className={cn("num px-3 py-3 text-right", up ? "text-bull" : "text-bear")}>
                        {fmtPct(p.unrealized_plpc)}
                      </td>
                      <td className="px-3 py-3 text-right">
                        <Countdown expiresAt={p.expires_at} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Panel>
  );
}
