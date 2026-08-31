import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Trash2, Zap } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import { Panel } from "@/components/Panel";
import { PretrainDialog } from "@/components/models/PretrainDialog";
import { Button } from "@/components/ui/button";
import { api, fmtDateTime } from "@/lib/api";
import type { ModelCheckpoint } from "@/lib/api-types";
import { cn } from "@/lib/utils";

const symbols = ["ALL", "AAPL", "MSFT", "SPY", "NVDA"];

function Metrics({ m }: { m: ModelCheckpoint }) {
  return (
    <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4">
      {[
        ["Final loss", m.final_loss.toExponential(2)],
        ["Steps", String(m.train_steps)],
        ["Win rate", m.metrics.win_rate_pct ? `${m.metrics.win_rate_pct}%` : "—"],
        ["Samples", String(m.metrics.scaler_n_samples)],
      ].map(([k, v]) => (
        <div key={k} className="rounded-xl border border-border/70 px-2.5 py-1.5">
          <p className="text-muted-foreground">{k}</p>
          <p className="num mt-0.5 text-xs text-foreground">{v}</p>
        </div>
      ))}
    </div>
  );
}

export function ModelsPanel() {
  const qc = useQueryClient();
  const [symbol, setSymbol] = useState("ALL");
  const [expanded, setExpanded] = useState<string | null>(null);

  const { data } = useQuery({
    queryKey: ["models", symbol],
    queryFn: () => api.models(symbol === "ALL" ? undefined : symbol),
  });

  const activate = useMutation({
    mutationFn: (id: string) => api.activate(id),
    onSuccess: (r) => {
      toast.success(`Hot-swapped to ${r.checkpoint_id}`);
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });
  const remove = useMutation({
    mutationFn: (id: string) => api.remove(id),
    onSuccess: (r) => {
      toast.success(`Deleted ${r.checkpoint_id}`);
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });

  const models = (data?.models ?? []).filter((m) => symbol === "ALL" || m.symbol === symbol);

  return (
    <Panel
      title="Model registry"
      subtitle={`${models.length} checkpoints`}
      actions={<PretrainDialog />}
      bodyClassName="p-3 sm:p-4"
    >
      <div className="mb-3 flex gap-1.5 overflow-x-auto pb-1">
        {symbols.map((s) => (
          <button
            key={s}
            onClick={() => setSymbol(s)}
            className={cn(
              "num rounded-full px-3.5 py-1.5 text-xs font-medium whitespace-nowrap transition-colors",
              symbol === s
                ? "bg-primary text-primary-foreground"
                : "border border-border text-muted-foreground hover:bg-accent",
            )}
          >
            {s}
          </button>
        ))}
      </div>

      <div className="space-y-2.5">
        {models.map((m) => (
          <article key={m.checkpoint_id} className="glass rounded-2xl px-3.5 py-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-semibold">{m.symbol}</p>
                  {m.active && (
                    <span className="flex items-center gap-1 rounded-full bg-bull/15 px-2 py-0.5 text-[10px] font-semibold tracking-widest text-bull uppercase">
                      <CheckCircle2 className="h-3 w-3" /> Active
                    </span>
                  )}
                  <span
                    className={cn(
                      "rounded-full px-2 py-0.5 text-[10px] font-semibold tracking-widest uppercase",
                      m.metrics.scaler_fitted ? "bg-secondary text-silver" : "bg-gold/15 text-gold",
                    )}
                  >
                    {m.metrics.scaler_fitted ? "Scaler fitted" : "No scaler"}
                  </span>
                </div>
                <p className="num mt-1 text-[11px] text-muted-foreground">
                  {m.checkpoint_id} · {m.scope} · {fmtDateTime(m.created_at)}
                </p>
              </div>

              <div className="flex flex-wrap gap-1.5">
                <Button
                  size="sm"
                  variant="secondary"
                  className="rounded-full"
                  onClick={() => setExpanded(expanded === m.checkpoint_id ? null : m.checkpoint_id)}
                >
                  Evaluation
                </Button>
                <Button
                  size="sm"
                  className="rounded-full bg-gold text-gold-foreground hover:bg-gold/90"
                  disabled={m.active || activate.isPending}
                  onClick={() => activate.mutate(m.checkpoint_id)}
                >
                  <Zap className="h-3.5 w-3.5" /> Activate
                </Button>
                <Button
                  size="sm"
                  variant="secondary"
                  className="rounded-full bg-destructive/15 text-destructive hover:bg-destructive/25"
                  disabled={remove.isPending}
                  onClick={() => remove.mutate(m.checkpoint_id)}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </Button>
              </div>
            </div>

            <Metrics m={m} />

            {expanded === m.checkpoint_id && (
              <div className="mt-3 border-t border-border/60 pt-3">
                <p className="text-[11px] tracking-[0.12em] text-muted-foreground uppercase">
                  Horizon alignment
                </p>
                <div className="mt-2 grid grid-cols-4 gap-2">
                  {Object.entries(m.metrics.horizon_alignment ?? {}).map(([h, v]) => (
                    <div key={h} className="rounded-xl border border-border/70 px-2 py-2 text-center">
                      <p className="text-[10px] text-muted-foreground">{h}</p>
                      <p className="num text-sm">{v.toFixed(2)}</p>
                    </div>
                  ))}
                </div>
                <p className="num mt-3 text-xs text-muted-foreground">
                  Sharpe {m.metrics.sharpe ?? "—"} · val loss{" "}
                  {m.metrics.val_loss?.toExponential(2) ?? "—"}
                </p>
              </div>
            )}
          </article>
        ))}
        {models.length === 0 && (
          <p className="py-6 text-center text-sm text-muted-foreground">No checkpoints for {symbol}.</p>
        )}
      </div>
    </Panel>
  );
}
