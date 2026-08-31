import type {
  AgentStatus,
  Health,
  ModelsResponse,
  PerformanceSummary,
  PositionsResponse,
  SignalBundle,
  TradesResponse,
} from "./api-types";

const iso = (minsAgo: number) => new Date(Date.now() - minsAgo * 60_000).toISOString();

export const mockHealth: Health = {
  status: "ok",
  mode: "backend-only",
  signal_inference: true,
  meta_learner_synthetic_train: true,
  timestamp: iso(0),
};

export const mockStatus: AgentStatus = {
  agent: "active",
  loop_state: "running",
  frontend_display_only: true,
  backend_inference: ["mtf_rsi", "divergence_scale", "signal_bundle", "meta_learner"],
  alpaca_connected: true,
  market_open: true,
  portfolio_value: 102450.75,
  cash: 95200.5,
  buying_power: 380802.0,
};

export const mockSummary: PerformanceSummary = {
  session_start_time: iso(210),
  session_start_equity: 100000,
  current_equity: 102450.75,
  session_pnl: 2450.75,
  session_pnl_pct: 2.45,
  day_pnl: 1820.5,
  day_pnl_pct: 1.81,
  week_pnl: 5340.2,
  week_pnl_pct: 5.49,
  month_pnl: 12450,
  month_pnl_pct: 13.83,
  total_trades: 24,
  wins: 18,
  losses: 6,
  win_rate_pct: 75,
};

export const mockPositions: PositionsResponse = {
  portfolio_value: 102450.75,
  cash: 95200.5,
  buying_power: 380802.0,
  positions: [
    {
      symbol: "AAPL260904C00235000",
      underlying: "AAPL",
      contract_type: "CALL",
      qty: 2,
      avg_fill_price: 3.45,
      current_price: 4.1,
      unrealized_pl: 130.0,
      unrealized_plpc: 18.84,
      expires_at: new Date(Date.now() + 26 * 60_000).toISOString(),
    },
    {
      symbol: "SPY260904P00548000",
      underlying: "SPY",
      contract_type: "PUT",
      qty: 4,
      avg_fill_price: 1.92,
      current_price: 1.71,
      unrealized_pl: -84.0,
      unrealized_plpc: -10.94,
      expires_at: new Date(Date.now() + 8 * 60_000).toISOString(),
    },
    {
      symbol: "MSFT260904C00432500",
      underlying: "MSFT",
      contract_type: "CALL",
      qty: 1,
      avg_fill_price: 5.6,
      current_price: 6.35,
      unrealized_pl: 75.0,
      unrealized_plpc: 13.39,
      expires_at: new Date(Date.now() + 52 * 60_000).toISOString(),
    },
  ],
};

const reasons = ["EXPIRY_REACHED", "TAKE_PROFIT", "STOP_LOSS"] as const;
const syms = ["AAPL", "MSFT", "SPY", "NVDA", "TSLA"];

export const mockTrades: TradesResponse = {
  count: 12,
  trades: Array.from({ length: 12 }, (_, i) => {
    const win = i % 3 !== 2;
    const entry = 2 + ((i * 7) % 9) / 3;
    const exit = win ? entry * 1.18 : entry * 0.86;
    const sym = syms[i % syms.length];
    const durations = [60, 900, 1800, 3600] as const;
    const dur = durations[i % durations.length] ?? 1800;
    return {
      trade_id: `trd-${String(100 + i)}`,
      symbol: `${sym}260904${i % 2 ? "P" : "C"}00${230 + i}000`,
      side: "buy",
      qty: 1 + (i % 3),
      asset_type: "option",
      start_time: iso(30 + i * 18),
      end_time: iso(30 + i * 18 - dur / 60),
      entry_price: Number(entry.toFixed(2)),
      exit_price: Number(exit.toFixed(2)),
      realized_pnl: Number(((exit - entry) * 100 * (1 + (i % 3))).toFixed(2)),
      realized_pnl_pct: Number((((exit - entry) / entry) * 100).toFixed(2)),
      hold_duration_sec: dur,
      exit_reason: win ? (reasons[i % 2] ?? "TAKE_PROFIT") : "STOP_LOSS",
    };
  }),
};

const curve = (steps: number, seed = 1) =>
  Array.from({ length: steps }, (_, i) => Number((0.42 / (1 + i * 0.55 * seed) + 0.0012).toFixed(5)));

export const mockModels: ModelsResponse = {
  count: 4,
  models: [
    {
      checkpoint_id: "chk-aapl-v1",
      symbol: "AAPL",
      scope: "prod-v1",
      active: true,
      train_steps: 50,
      final_loss: 1.24e-3,
      metrics: {
        scaler_fitted: true,
        scaler_n_samples: 420,
        val_loss: 1.9e-3,
        win_rate_pct: 71.2,
        sharpe: 1.84,
        horizon_alignment: { "5m": 0.62, "15m": 0.74, "30m": 0.69, "1h": 0.58 },
        loss_curve: curve(20),
      },
      created_at: iso(320),
    },
    {
      checkpoint_id: "chk-msft-v2",
      symbol: "MSFT",
      scope: "dev-scope-v1",
      active: false,
      train_steps: 20,
      final_loss: 2.14e-4,
      metrics: {
        scaler_fitted: true,
        scaler_n_samples: 476,
        val_loss: 3.12e-4,
        win_rate_pct: 68.5,
        sharpe: 1.42,
        horizon_alignment: { "5m": 0.55, "15m": 0.66, "30m": 0.72, "1h": 0.61 },
        loss_curve: curve(20, 1.6),
      },
      created_at: iso(180),
    },
    {
      checkpoint_id: "chk-spy-v3",
      symbol: "SPY",
      scope: "prod-v1",
      active: true,
      train_steps: 80,
      final_loss: 8.4e-4,
      metrics: {
        scaler_fitted: true,
        scaler_n_samples: 910,
        val_loss: 9.9e-4,
        win_rate_pct: 74.8,
        sharpe: 2.06,
        horizon_alignment: { "5m": 0.7, "15m": 0.78, "30m": 0.66, "1h": 0.6 },
        loss_curve: curve(24, 0.8),
      },
      created_art: undefined,
      created_at: iso(90),
    } as ModelsResponse["models"][number],
    {
      checkpoint_id: "chk-nvda-v1",
      symbol: "NVDA",
      scope: "sandbox",
      active: false,
      train_steps: 12,
      final_loss: 4.7e-3,
      metrics: {
        scaler_fitted: false,
        scaler_n_samples: 0,
        win_rate_pct: 51.4,
        sharpe: 0.44,
        horizon_alignment: { "5m": 0.41, "15m": 0.48, "30m": 0.5, "1h": 0.39 },
        loss_curve: curve(12, 2.4),
      },
      created_at: iso(35),
    },
  ],
};

export const mockSignal: SignalBundle = {
  symbol: "AAPL",
  meta_conviction: 0.84,
  bias: "BULLISH",
  recommended_expiry: "15m",
  reversal_risk_pct: 18.5,
  expected_mfe_pips: 42,
  expected_mae_pips: 14,
  dxy_divergence: "BEARISH",
  horizons: [
    { horizon: "5m", score: 0.71 },
    { horizon: "15m", score: 0.86 },
    { horizon: "30m", score: 0.79 },
    { horizon: "1h", score: 0.63 },
  ],
  candles: Array.from({ length: 40 }, (_, i) => ({
    timestamp: iso(200 - i * 5),
    close: Number((150 + Math.sin(i / 3) * 1.6 + i * 0.055).toFixed(2)),
  })),
};
