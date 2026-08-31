export type LoopState = "running" | "paused" | "stopped";

export interface Health {
  status: string;
  mode: string;
  signal_inference: boolean;
  meta_learner_synthetic_train: boolean;
  timestamp: string;
}

export interface AgentStatus {
  agent: string;
  loop_state: LoopState;
  frontend_display_only: boolean;
  backend_inference: string[];
  alpaca_connected: boolean;
  market_open: boolean;
  portfolio_value: number;
  cash: number;
  buying_power: number;
}

export interface PerformanceSummary {
  session_start_time: string;
  session_start_equity: number;
  current_equity: number;
  session_pnl: number;
  session_pnl_pct: number;
  day_pnl: number;
  day_pnl_pct: number;
  week_pnl: number;
  week_pnl_pct: number;
  month_pnl: number;
  month_pnl_pct: number;
  total_trades: number;
  wins: number;
  losses: number;
  win_rate_pct: number;
}

export interface Position {
  symbol: string;
  underlying: string;
  contract_type: "CALL" | "PUT";
  qty: number;
  avg_fill_price: number;
  current_price: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  expires_at?: string;
}

export interface PositionsResponse {
  portfolio_value: number;
  cash: number;
  buying_power: number;
  positions: Position[];
}

export interface Trade {
  trade_id: string;
  symbol: string;
  side: string;
  qty: number;
  asset_type: string;
  start_time: string;
  end_time: string;
  entry_price: number;
  exit_price: number;
  realized_pnl: number;
  realized_pnl_pct: number;
  hold_duration_sec: number;
  exit_reason: "EXPIRY_REACHED" | "TAKE_PROFIT" | "STOP_LOSS" | string;
}

export interface TradesResponse {
  count: number;
  trades: Trade[];
}

export interface ModelCheckpoint {
  checkpoint_id: string;
  symbol: string;
  scope: string;
  active: boolean;
  train_steps: number;
  final_loss: number;
  metrics: {
    scaler_fitted: boolean;
    scaler_n_samples: number;
    val_loss?: number;
    win_rate_pct?: number;
    sharpe?: number;
    horizon_alignment?: Record<string, number>;
    loss_curve?: number[];
  };
  created_at: string;
}

export interface ModelsResponse {
  count: number;
  models: ModelCheckpoint[];
}

export interface PretrainRequest {
  symbol: string;
  scope: string;
  num_candles: number;
  warmup_bars: number;
  train_steps: number;
  batch_size: number;
  seed: number;
  persist: boolean;
}

export interface PretrainResponse {
  checkpoint_id: string;
  symbol: string;
  scope: string;
  experiences_recorded: number;
  train_steps: number;
  final_loss: number;
  metrics: ModelCheckpoint["metrics"];
}

export interface CycleResponse {
  status: string;
  cycle_id: string;
  executed_action: string;
  symbol: string;
  expiry_selected: string;
  meta_conviction: number;
  timestamp: string;
}

export interface SignalBundle {
  symbol: string;
  meta_conviction: number;
  bias: "BULLISH" | "BEARISH" | "NEUTRAL";
  recommended_expiry: "5m" | "15m" | "30m" | "1h";
  reversal_risk_pct: number;
  expected_mfe_pips: number;
  expected_mae_pips: number;
  dxy_divergence: "BULLISH" | "BEARISH" | "NEUTRAL";
  horizons: { horizon: string; score: number }[];
  candles: { timestamp: string; close: number }[];
}
