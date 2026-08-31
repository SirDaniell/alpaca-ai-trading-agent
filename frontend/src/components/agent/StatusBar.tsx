import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CircleDot, Pause, Play, Zap } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { api, fmtMoney, getLastSource } from "@/lib/api";
import { cn } from "@/lib/utils";

function Dot({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
      <span className={cn("h-1.5 w-1.5 rounded-full", ok ? "bg-bull" : "bg-bear")} />
      {label}
    </span>
  );
}

export function StatusBar({ symbol }: { symbol: string }) {
  const qc = useQueryClient();
  const { data: status } = useQuery({
    queryKey: ["status"],
    queryFn: api.status,
    refetchInterval: 10_000,
  });
  const { data: health } = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 30_000 });

  const invalidate = () => {
    qc.invalidateQueries({ queryKey: ["status"] });
    qc.invalidateQueries({ queryKey: ["positions"] });
    qc.invalidateQueries({ queryKey: ["trades"] });
  };

  const start = useMutation({
    mutationFn: () => api.start(300),
    onSuccess: (r) => {
      toast.success(`Autopilot ${r.loop_state}`, { description: `Interval ${r.interval_seconds}s` });
      invalidate();
    },
  });
  const stop = useMutation({
    mutationFn: () => api.stop(),
    onSuccess: () => {
      toast.success("Autopilot paused");
      invalidate();
    },
  });
  const cycle = useMutation({
    mutationFn: () => api.runCycle(symbol),
    onSuccess: (r) => {
      toast.success(`Cycle ${r.cycle_id}`, {
        description: `${r.executed_action} · ${r.symbol} · ${r.expiry_selected} · conviction ${r.meta_conviction}`,
      });
      invalidate();
    },
  });

  const state = status?.loop_state ?? "stopped";
  const running = state === "running";

  return (
    <section className="glass rounded-3xl p-4 sm:p-5">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-center gap-2.5">
            <span
              className={cn(
                "flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold tracking-[0.14em] uppercase",
                running ? "bg-bull/15 text-bull" : "bg-gold/15 text-gold",
              )}
            >
              <CircleDot className="h-3 w-3" />
              {state}
            </span>
            <span className="rounded-full bg-secondary px-3 py-1 text-[11px] font-semibold tracking-[0.14em] text-silver uppercase">
              Health {health?.status ?? "—"}
            </span>
            <span className="rounded-full border border-border px-3 py-1 text-[11px] tracking-[0.14em] text-muted-foreground uppercase">
              {getLastSource() === "live" ? "Live backend" : "Synthetic data"}
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
            <Dot ok={!!status?.alpaca_connected} label="Alpaca" />
            <Dot ok={!!status?.market_open} label={status?.market_open ? "NYSE open" : "NYSE closed"} />
            <Dot ok={!!health?.signal_inference} label="Signal inference" />
            <span className="num text-xs text-muted-foreground">
              Equity {fmtMoney(status?.portfolio_value ?? 0)}
            </span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:w-auto">
          <Button
            onClick={() => start.mutate()}
            disabled={start.isPending || running}
            className="rounded-full"
          >
            <Play className="h-4 w-4" /> Start
          </Button>
          <Button
            onClick={() => stop.mutate()}
            disabled={stop.isPending || !running}
            variant="secondary"
            className="rounded-full bg-destructive/15 text-destructive hover:bg-destructive/25"
          >
            <Pause className="h-4 w-4" /> Pause
          </Button>
          <Button
            onClick={() => cycle.mutate()}
            disabled={cycle.isPending}
            variant="secondary"
            className="col-span-2 rounded-full bg-gold text-gold-foreground hover:bg-gold/90 sm:col-span-1"
          >
            <Zap className="h-4 w-4" /> {cycle.isPending ? "Running…" : "Cycle now"}
          </Button>
        </div>
      </div>
    </section>
  );
}
