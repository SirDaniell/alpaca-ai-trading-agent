import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api } from "@/lib/api";
import type { PretrainRequest, PretrainResponse } from "@/lib/api-types";

const defaults: PretrainRequest = {
  symbol: "MSFT",
  scope: "dev-scope-v1",
  num_candles: 1500,
  warmup_bars: 80,
  train_steps: 20,
  batch_size: 32,
  seed: 42,
  persist: true,
};

const numFields = [
  ["num_candles", "Candles"],
  ["warmup_bars", "Warmup bars"],
  ["train_steps", "Train steps"],
  ["batch_size", "Batch size"],
  ["seed", "Seed"],
] as const;

export function PretrainDialog() {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const [form, setForm] = useState<PretrainRequest>(defaults);
  const [result, setResult] = useState<PretrainResponse | null>(null);

  const run = useMutation({
    mutationFn: () => api.pretrain(form),
    onSuccess: (r) => {
      setResult(r);
      toast.success(`Checkpoint ${r.checkpoint_id} trained`, {
        description: `final loss ${r.final_loss} · ${r.experiences_recorded} experiences`,
      });
      qc.invalidateQueries({ queryKey: ["models"] });
    },
  });

  const curve = (result?.metrics.loss_curve ?? []).map((loss, step) => ({ step: step + 1, loss }));

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="rounded-full">Pretrain model</Button>
      </DialogTrigger>
      <DialogContent className="glass max-h-[90vh] overflow-y-auto rounded-3xl sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Off-line pretraining run</DialogTitle>
          <DialogDescription>
            Trains a checkpoint on synthetic or real candles, then reports metrics.
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-2 gap-3">
          <div className="space-y-1.5">
            <Label htmlFor="symbol">Symbol</Label>
            <Input
              id="symbol"
              value={form.symbol}
              onChange={(e) => setForm({ ...form, symbol: e.target.value.toUpperCase() })}
              className="num rounded-xl"
            />
          </div>
          <div className="space-y-1.5">
            <Label htmlFor="scope">Scope</Label>
            <Input
              id="scope"
              value={form.scope}
              onChange={(e) => setForm({ ...form, scope: e.target.value })}
              className="num rounded-xl"
            />
          </div>
          {numFields.map(([key, label]) => (
            <div key={key} className="space-y-1.5">
              <Label htmlFor={key}>{label}</Label>
              <Input
                id={key}
                type="number"
                value={form[key]}
                onChange={(e) => setForm({ ...form, [key]: Number(e.target.value) })}
                className="num rounded-xl"
              />
            </div>
          ))}
        </div>

        <Button
          onClick={() => run.mutate()}
          disabled={run.isPending}
          className="mt-1 w-full rounded-full"
        >
          {run.isPending ? "Training…" : "Run pretraining"}
        </Button>

        {result && (
          <div className="mt-2 space-y-3">
            <div className="h-36 rounded-2xl border border-border/70 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curve}>
                  <YAxis hide />
                  <Tooltip
                    contentStyle={{
                      background: "var(--popover)",
                      border: "1px solid var(--border)",
                      borderRadius: 12,
                      fontSize: 12,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="loss"
                    stroke="var(--color-gold)"
                    strokeWidth={1.8}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <dl className="grid grid-cols-2 gap-2 text-xs">
              {[
                ["Final loss", result.final_loss.toExponential(2)],
                ["Val loss", result.metrics.val_loss?.toExponential(2) ?? "—"],
                ["Win rate", `${result.metrics.win_rate_pct ?? "—"}%`],
                ["Sharpe", String(result.metrics.sharpe ?? "—")],
                ["Scaler samples", String(result.metrics.scaler_n_samples)],
                ["Experiences", String(result.experiences_recorded)],
              ].map(([k, v]) => (
                <div key={k} className="rounded-xl border border-border/70 px-3 py-2">
                  <dt className="text-muted-foreground">{k}</dt>
                  <dd className="num mt-0.5 text-sm">{v}</dd>
                </div>
              ))}
            </dl>
            <div className="grid grid-cols-4 gap-2">
              {Object.entries(result.metrics.horizon_alignment ?? {}).map(([h, v]) => (
                <div key={h} className="rounded-xl border border-border/70 px-2 py-2 text-center">
                  <p className="text-[10px] text-muted-foreground">{h}</p>
                  <p className="num text-sm">{v.toFixed(2)}</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
