import type {
  AgentStatus,
  CycleResponse,
  Health,
  ModelsResponse,
  PerformanceSummary,
  PositionsResponse,
  PretrainRequest,
  PretrainResponse,
  SignalBundle,
  TradesResponse,
} from "./api-types";
import {
  mockHealth,
  mockModels,
  mockPositions,
  mockSignal,
  mockStatus,
  mockSummary,
  mockTrades,
} from "./mock-data";

export const API_BASE =
  (import.meta.env["VITE_API_BASE_URL"] as string | undefined)?.replace(/\/$/, "") ??
  "http://localhost:8000";

export type Source = "live" | "mock";

let lastSource: Source = "live";
export const getLastSource = () => lastSource;

async function request<T>(path: string, init?: RequestInit, timeoutMs = 8000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      ...init,
      signal: controller.signal,
      headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    lastSource = "live";
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  health: () => request<Health>("/health"),
  status: () => request<AgentStatus>("/status"),
  summary: () => request<PerformanceSummary>("/performance/summary"),
  positions: () => request<PositionsResponse>("/positions"),
  trades: (limit = 50) => request<TradesResponse>(`/performance/trades?limit=${limit}`),
  models: (symbol?: string) => request<ModelsResponse>(symbol ? `/models?symbol=${symbol}` : "/models"),
  signal: (symbol = "AAPL") => request<SignalBundle>(`/signal/latest?symbol=${symbol}`),

  start: (interval_seconds = 300) =>
    request<{ status: string; loop_state: string; interval_seconds: number }>(
      "/agent/start",
      { method: "POST", body: JSON.stringify({ interval_seconds }) },
    ),
  stop: () =>
    request<{ status: string; loop_state: string }>(
      "/agent/stop",
      { method: "POST" },
    ),
  runCycle: (symbol: string, dry_run = false) =>
    request<CycleResponse>(
      "/agent/run-cycle",
      { method: "POST", body: JSON.stringify({ symbol, dry_run }) },
    ),
  pretrain: (body: PretrainRequest) =>
    request<PretrainResponse>(
      "/models/pretrain",
      { method: "POST", body: JSON.stringify(body) },
    ),
  activate: (checkpoint_id: string) =>
    request<{ status: string; checkpoint_id: string }>(
      `/models/${checkpoint_id}/activate`,
      { method: "POST" },
    ),
  remove: (checkpoint_id: string) =>
    request<{ status: string; checkpoint_id: string }>(
      `/models/${checkpoint_id}`,
      { method: "DELETE" },
    ),
};

export const fmtMoney = (n: number | string | null | undefined, digits = 2) => {
  const num = typeof n === "string" ? parseFloat(n) : typeof n === "number" ? n : 0;
  const val = isNaN(num) ? 0 : num;
  return `${val < 0 ? "-" : ""}$${Math.abs(val).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}`;
};

export const fmtPct = (n: number | string | null | undefined) => {
  const num = typeof n === "string" ? parseFloat(n) : typeof n === "number" ? n : 0;
  const val = isNaN(num) ? 0 : num;
  return `${val > 0 ? "+" : ""}${val.toFixed(2)}%`;
};

export const fmtSigned = (n: number | string | null | undefined) => {
  const num = typeof n === "string" ? parseFloat(n) : typeof n === "number" ? n : 0;
  const val = isNaN(num) ? 0 : num;
  return `${val > 0 ? "+" : ""}${fmtMoney(val)}`;
};

export const fmtDuration = (sec: number) => {
  if (sec < 60) return `${sec}s`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m}m`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
};

export const fmtTime = (t: string) =>
  new Date(t).toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });

export const fmtDateTime = (t: string) =>
  new Date(t).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
