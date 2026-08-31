import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { AppShell } from "@/components/AppShell";
import { PositionsPanel } from "@/components/agent/PositionsPanel";
import { SignalCard } from "@/components/agent/SignalCard";
import { StatusBar } from "@/components/agent/StatusBar";
import { TradeFeed } from "@/components/agent/TradeFeed";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "AXE Genesis — Autopilot Control & Execution Feed" },
      {
        name: "description",
        content:
          "Monitor the AXE Genesis autonomous options trading agent: autopilot controls, live positions, execution feed, and multi-horizon signal intelligence.",
      },
      { property: "og:title", content: "AXE Genesis — Autopilot Control" },
      {
        property: "og:description",
        content:
          "Real-time monitoring and control center for the AXE Genesis autonomous options trading agent.",
      },
      { property: "og:type", content: "website" },
      { name: "twitter:card", content: "summary_large_image" },
    ],
  }),
  component: AutopilotPage,
});

const symbols = ["AAPL", "MSFT", "SPY", "NVDA", "TSLA"];

function AutopilotPage() {
  const [symbol, setSymbol] = useState("AAPL");

  return (
    <AppShell>
      <div className="space-y-4 sm:space-y-5">
        <StatusBar symbol={symbol} />

        <div className="flex gap-1.5 overflow-x-auto pb-1">
          {symbols.map((s) => (
            <button
              key={s}
              onClick={() => setSymbol(s)}
              className={cn(
                "num rounded-full px-4 py-1.5 text-xs font-medium whitespace-nowrap transition-colors",
                symbol === s
                  ? "bg-primary text-primary-foreground"
                  : "glass text-muted-foreground hover:text-foreground",
              )}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="grid gap-4 sm:gap-5 lg:grid-cols-3">
          <div className="lg:col-span-1">
            <SignalCard symbol={symbol} />
          </div>
          <div className="lg:col-span-2">
            <PositionsPanel />
          </div>
        </div>

        <TradeFeed limit={25} />
      </div>
    </AppShell>
  );
}
