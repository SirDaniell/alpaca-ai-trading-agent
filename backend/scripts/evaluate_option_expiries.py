"""
evaluate_option_expiries.py — Benchmark Tool to Find Optimal Option Expiry Horizon on 5m Execution.

Evaluates out-of-sample directional win rates, streaks, and performance across 4 option expiry horizons:
1. 5m  Expiry (N = 1 bar ahead)
2. 15m Expiry (N = 3 bars ahead)
3. 30m Expiry (N = 6 bars ahead)
4. 1h  Expiry (N = 12 bars ahead)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import random
import logging
import pprint
import numpy as np
import pandas as pd

# Ensure backend root directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from app.core.ml.real_data_pipeline import (
    RealTrainConfig,
    fetch_real_candles,
    align_multi_timeframe_datasets,
    _htf_start_for,
)
from app.core.options.q_executor import OptionsQExecutor, HTFBiasPackage, AccountContext, ExecutionContext, EXECUTOR_STATE_DIM
from app.core.market.zone_snapshot import ZoneSnapshotManager, HardActionMask
from app.core.ml.signal_meta_learner import OnlineSignalMetaLearner, SIGNAL_META_FEATURE_COUNT
from app.core.ml.ml_dataset_preparation import MLDatasetPreparation, DatasetConfig

def _make_exec_ctx(symbol: str, price: float, row: dict) -> ExecutionContext:
    ts = row.get("timestamp", pd.Timestamp.now(tz="UTC"))
    if isinstance(ts, pd.Timestamp) and ts.tzinfo is not None:
        hour_f = ts.hour + ts.minute / 60.0
        dow = ts.dayofweek
    else:
        hour_f, dow = 14.5, 1
    phase = "nyse_open" if 13 <= getattr(ts, "hour", 14) < 17 else "off_hours"
    vol = float(row.get("volume", row.get("volume_5m", 1000.0)))
    c_price = float(row.get("close", price))
    o_price = float(row.get("open", price))
    if c_price >= o_price:
        buy_vol = vol
        sell_vol = 0.0
    else:
        buy_vol = 0.0
        sell_vol = vol
    return ExecutionContext(
        symbol=symbol,
        current_price=price,
        atr=float(row.get("atr_5m", float(row.get("atr", price * 0.005)))),
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        hour_of_day=hour_f,
        day_of_week=dow,
        session_phase=phase,
    )

def _row_to_feature_dict(row_data: pd.Series, close_col: str, vol_col: str) -> dict:
    def _clean_float(value, default=0.0):
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not np.isfinite(fv):
            return float(default)
        return fv

    f_dict = {
        "close": _clean_float(row_data.get(close_col, 0.0)),
        "volume": _clean_float(row_data.get(vol_col, 0.0)),
    }
    for k, v in row_data.items():
        if k not in ("timestamp", close_col, vol_col) and isinstance(v, (int, float, np.number)):
            fv = _clean_float(v)
            if np.isfinite(fv):
                f_dict[str(k)] = fv
    return f_dict


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("EvalExpiries")

EXPIRY_HORIZONS = {
    "5m (1 bar)": 1,
    "15m (3 bars)": 3,
    "30m (6 bars)": 6,
    "1h (12 bars)": 12,
}

# Maps model-output short labels (HORIZON_LABELS) → EXPIRY_HORIZONS keys
HORIZON_LABEL_MAP = {
    "5m": "5m (1 bar)",
    "15m": "15m (3 bars)",
    "30m": "30m (6 bars)",
    "1h": "1h (12 bars)",
}


def compute_full_context_features(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Compute and populate all 21 CONTEXT_FEATURE_KEYS (MTF RSI, DXY divergence, SNR distances, confluence)."""
    df = aligned_df.copy()
    close_col = "close_5m" if "close_5m" in df.columns else "close"
    high_col = "high_5m" if "high_5m" in df.columns else "high"
    low_col = "low_5m" if "low_5m" in df.columns else "low"

    asset_close = df[close_col].values.astype(np.float64)

    # 1. Synthetic DXY / Market Benchmark Relative Baseline
    ema_200 = pd.Series(asset_close).ewm(span=200, adjust=False).mean().values
    dxy_synth = (2.0 * ema_200 - asset_close)
    df["dxy_close"] = dxy_synth.astype(np.float32)

    # 2. Fast & Slow Divergence Scales
    fast_w, slow_w = 5, 14
    asset_fast_pct = pd.Series(asset_close).pct_change(fast_w).fillna(0.0).values
    asset_slow_pct = pd.Series(asset_close).pct_change(slow_w).fillna(0.0).values
    dxy_fast_pct   = pd.Series(dxy_synth).pct_change(fast_w).fillna(0.0).values
    dxy_slow_pct   = pd.Series(dxy_synth).pct_change(slow_w).fillna(0.0).values

    def _norm(arr):
        std = float(np.std(arr)) + 1e-8
        return np.clip(arr / (2.0 * std), -1.0, 1.0).astype(np.float32)

    asset_fast_norm = _norm(asset_fast_pct)
    asset_slow_norm = _norm(asset_slow_pct)
    dxy_fast_norm   = _norm(dxy_fast_pct)
    dxy_slow_norm   = _norm(dxy_slow_pct)

    df["asset_slow_norm"] = asset_slow_norm
    df["dxy_slow_norm"]   = dxy_slow_norm
    df["asset_fast_norm"] = asset_fast_norm
    df["dxy_fast_norm"]   = dxy_fast_norm

    slow_diff = asset_slow_norm - dxy_slow_norm
    fast_diff = asset_fast_norm - dxy_fast_norm
    df["slow_diff"] = slow_diff
    df["fast_diff"] = fast_diff

    df["regime_strong_asset"] = ((slow_diff >= 0) & (fast_diff >= 0)).astype(np.float32)
    df["regime_weak_asset"]   = ((slow_diff >= 0) & (fast_diff < 0)).astype(np.float32)
    df["regime_weak_dxy"]     = ((slow_diff < 0)  & (fast_diff >= 0)).astype(np.float32)
    df["regime_strong_dxy"]   = ((slow_diff < 0)  & (fast_diff < 0)).astype(np.float32)

    # 3. MTF RSI (5m, 15m, 1h)
    def _rsi(s, w=14):
        delta = s.diff()
        gain = delta.where(delta > 0, 0.0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0.0).abs()).rolling(w).mean()
        rs = gain / (loss + 1e-8)
        return (100.0 - (100.0 / (1.0 + rs))).fillna(50.0).values

    rsi_5m  = _rsi(pd.Series(asset_close), 14)
    rsi_15m = _rsi(pd.Series(df["close_15m"] if "close_15m" in df.columns else asset_close), 14)
    rsi_1h  = _rsi(pd.Series(df["close_1h"] if "close_1h" in df.columns else asset_close), 14)

    mtf_rsi_asset = (0.5 * rsi_5m + 0.3 * rsi_15m + 0.2 * rsi_1h).astype(np.float32)
    dxy_rsi_5m    = _rsi(pd.Series(dxy_synth), 14).astype(np.float32)

    df["mtf_rsi_asset"] = mtf_rsi_asset
    df["mtf_rsi_dxy"]   = dxy_rsi_5m
    df["mtf_rsi_diff"]  = (mtf_rsi_asset - dxy_rsi_5m).astype(np.float32)

    # 4. MTF Regime State Features + Recency-Windowed Cross Signals
    #
    # Design rationale:
    # Pure event-based cross detection (fired only on the exact crossover bar) is too
    # sparse — a cross on H1 that happened 3 H1 bars ago (= 15 5m bars ago) is invisible
    # to the model. Instead we provide:
    #   (a) Continuous STATE features: current relative positions, always non-zero.
    #   (b) Recency-windowed CROSS MEMORY: did a cross happen within the last N bars?
    #       This lets the Q-learner learn "LTF crosses first, HTF confirms later"
    #       without any hardcoded gate — the pattern is expressed as input features.
    #   (c) Spread magnitude: how far above/below threshold is the relationship?
    #       This gives the model directional conviction strength, not just sign.
    #
    # The four original cross_* column names are preserved for backward compatibility
    # with feature_dict keys and any downstream code that reads them.

    MA_WINDOW = 21          # MA21 period for RSI signal line
    CROSS_MEMORY_BARS = 5   # recency window: did a cross happen in the last N bars?

    rsi_series     = pd.Series(rsi_5m)
    dxy_rsi_series = pd.Series(dxy_rsi_5m)
    fast_diff_series = pd.Series(fast_diff)

    # MA21 signal lines for asset RSI and DXY RSI
    asset_rsi_ma21 = rsi_series.rolling(MA_WINDOW, min_periods=1).mean()
    dxy_rsi_ma21   = dxy_rsi_series.rolling(MA_WINDOW, min_periods=1).mean()

    # ── (a) Continuous state: current relative positions ──────────────────────
    # 1.0 = bullish state, 0.0 = bearish state — every bar is defined
    _state_asset_above_ma  = (rsi_series     > asset_rsi_ma21).astype(np.float32)
    _state_dxy_above_ma    = (dxy_rsi_series > dxy_rsi_ma21).astype(np.float32)
    _state_asset_above_dxy = (rsi_series     > dxy_rsi_series).astype(np.float32)
    _state_htf_bullish     = (pd.Series(rsi_1h) > 50).astype(np.float32)  # H1 RSI above midline

    df["state_asset_above_ma"]  = _state_asset_above_ma.values
    df["state_dxy_above_ma"]    = _state_dxy_above_ma.values
    df["state_asset_above_dxy"] = _state_asset_above_dxy.values
    df["state_htf_bullish"]     = _state_htf_bullish.values  # H1 confirmation state — learned by Q

    # ── (b) Spread magnitude: normalized distance from the MA/midline ─────────
    # Clipped to [-1, +1]; gives conviction strength to the Q-learner
    df["state_asset_ma_spread"]  = np.clip((rsi_5m - asset_rsi_ma21.values) / 50.0, -1.0, 1.0).astype(np.float32)
    df["state_dxy_ma_spread"]    = np.clip((dxy_rsi_5m - dxy_rsi_ma21.values) / 50.0, -1.0, 1.0).astype(np.float32)
    df["state_asset_dxy_spread"] = np.clip((rsi_5m - dxy_rsi_5m) / 50.0, -1.0, 1.0).astype(np.float32)

    # ── (c) Recency-windowed cross memory ─────────────────────────────────────
    # +1.0 if a bull cross happened in the last CROSS_MEMORY_BARS bars (still in post-cross regime)
    # -1.0 if a bear cross happened in the last CROSS_MEMORY_BARS bars
    #  0.0 if no cross happened recently (but STATE features above still reflect current regime)
    def _cross_memory(state_bool_series: pd.Series, window: int = CROSS_MEMORY_BARS) -> np.ndarray:
        """Returns +1/-1/0 recency signal: +1 within N bars after bull cross, -1 after bear cross."""
        prev = state_bool_series.shift(1).astype("boolean").fillna(pd.NA).astype("boolean")
        bull_cross = (state_bool_series & ~prev.fillna(False)).astype(int)
        bear_cross = (~state_bool_series & prev.fillna(True)).astype(int)
        recent_bull = bull_cross.rolling(window, min_periods=1).max()
        recent_bear = bear_cross.rolling(window, min_periods=1).max()
        return (recent_bull - recent_bear).values.astype(np.float32)

    # Preserve original column names for backward compatibility with feature_dict
    df["cross_index_signal"] = _cross_memory(_state_asset_above_ma.astype(bool))
    df["cross_dxy_signal"]   = _cross_memory(_state_dxy_above_ma.astype(bool))
    df["cross_index_dxy"]    = _cross_memory(_state_asset_above_dxy.astype(bool))
    # DXY-symbol momentum: rolling 3-bar mean of fast_diff (directional speed, not event)
    df["cross_dxy_symbol"]   = fast_diff_series.rolling(3, min_periods=1).mean().clip(-1.0, 1.0).values.astype(np.float32)

    # 5. Dynamic SNR distances and MTF SNR confluence
    atr = (df[high_col] - df[low_col]).rolling(14).mean().fillna(df[close_col] * 0.005).values
    supp_q = df[low_col].rolling(100, min_periods=10).quantile(0.20).fillna(df[low_col]).values
    res_q  = df[high_col].rolling(100, min_periods=10).quantile(0.80).fillna(df[high_col]).values

    df["snr_dist_support"]    = np.clip((asset_close - supp_q) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)
    df["snr_dist_resistance"] = np.clip((res_q - asset_close) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)

    # ── Per-TF SNR distance columns (Fix 4.1) ────────────────────────────────
    # Rolling-window sizes chosen so each TF covers ~25h of history at 5m bar density.
    # (100 rows × 5m = 500min ≈ 8h for 15m; 30 rows × 5m ≈ 2.5h worth of 1h bars;
    #  10 rows × 5m ≈ 50min worth of 4h bars — each represents the last closed HTF bar)
    TF_CONFIGS = [("15m", 100), ("1h", 30), ("4h", 10)]
    for _tf, _window in TF_CONFIGS:
        _h_col = f"high_{_tf}"
        _l_col = f"low_{_tf}"
        if _h_col in df.columns and _l_col in df.columns:
            _supp_tf = df[_l_col].rolling(_window, min_periods=max(1, _window // 4)).quantile(0.20).fillna(df[_l_col])
            _res_tf  = df[_h_col].rolling(_window, min_periods=max(1, _window // 4)).quantile(0.80).fillna(df[_h_col])
            df[f"snr_dist_support_{_tf}"]    = np.clip((asset_close - _supp_tf.values) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)
            df[f"snr_dist_resistance_{_tf}"] = np.clip((_res_tf.values - asset_close) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)
        else:
            # Graceful fallback: HTF data not available — default to max distance
            df[f"snr_dist_support_{_tf}"]    = np.full(len(df), 10.0, dtype=np.float32)
            df[f"snr_dist_resistance_{_tf}"] = np.full(len(df), 10.0, dtype=np.float32)

    # ── Real MTF SNR confluence (Fix 4.2) ────────────────────────────────────
    # Confluence = any 15m zone within CONFLUENCE_PCT of any 1h zone.
    # Matches the StockChart reference (CONFLUENCE_PCT = 0.0015 = 0.15%).
    CONFLUENCE_PCT = 0.0015

    # Convert ATR-normalised distances back to approximate zone anchor prices
    _sup_15m_price = asset_close - df["snr_dist_support_15m"].values * atr
    _res_15m_price = asset_close + df["snr_dist_resistance_15m"].values * atr
    _sup_1h_price  = asset_close - df["snr_dist_support_1h"].values  * atr
    _res_1h_price  = asset_close + df["snr_dist_resistance_1h"].values  * atr

    _sup_sup_conf  = (np.abs(_sup_15m_price - _sup_1h_price)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
    _res_res_conf  = (np.abs(_res_15m_price - _res_1h_price)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
    _sup_res_conf  = (np.abs(_sup_15m_price - _res_1h_price)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
    _res_sup_conf  = (np.abs(_res_15m_price - _sup_1h_price)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT

    df["mtf_snr_confluence"] = (_sup_sup_conf | _res_res_conf | _sup_res_conf | _res_sup_conf).astype(np.float32)


    # ── 6. Zone Proximity & Touch Context Features ────────────────────────────
    #
    # These features encode price's structural relationship to SNR zones so the
    # Q-learner can learn to prioritize entries near, at, or after touches of
    # important zones — without any hardcoded rules.
    #
    # Features:
    #   zone_dist_support_atr     — ATR-normalised distance to nearest 15m support
    #   zone_dist_resistance_atr  — ATR-normalised distance to nearest 15m resistance
    #   zone_in_support_zone      — 1.0 if price is inside the support zone band (touch zone)
    #   zone_in_resistance_zone   — 1.0 if price is inside the resistance zone band
    #   zone_price_position       — 0.0=at support, 1.0=at resistance (normalized between S and R)
    #   zone_approach_direction   — price moving toward support (-1), toward resistance (+1), or flat (0)
    #   zone_support_vol_ratio    — up_volume / (total_volume+ε) of nearest support (zone strength)
    #   zone_resistance_vol_ratio — down_volume / (total_volume+ε) of nearest resistance
    #   zone_support_confluence   — number of TFs where this support is active (MTF zone strength)
    #   zone_resistance_confluence— same for resistance
    #   zone_bounce_signal        — 1.0 if at support + price starting to reverse up
    #   zone_rejection_signal     — 1.0 if at resistance + price starting to reverse down
    #
    # All features are vectorized over the full aligned_df — no lookahead.
    # Zone anchors are the same rolling-quantile levels computed above, ensuring consistency.

    # Reuse the 15m support/resistance quantile anchors as the "nearest zone" proxies
    # (these are already computed as _supp_tf and _res_tf for the 15m TF above)
    # Re-derive them here for clarity and independence from the loop scope
    _zone_lb = 50  # lookback window — same as per-TF 15m window
    _z_supp = df[f"low_15m" if "low_15m" in df.columns else low_col].rolling(
        _zone_lb, min_periods=max(1, _zone_lb // 4)
    ).quantile(0.20).fillna(df[low_col]).values

    _z_res = df[f"high_15m" if "high_15m" in df.columns else high_col].rolling(
        _zone_lb, min_periods=max(1, _zone_lb // 4)
    ).quantile(0.80).fillna(df[high_col]).values

    # ATR-normalised distances to nearest S and R
    _d_supp = np.clip((asset_close - _z_supp) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)
    _d_res  = np.clip((_z_res - asset_close)  / (atr + 1e-8), 0.0, 10.0).astype(np.float32)

    df["zone_dist_support_atr"]    = _d_supp
    df["zone_dist_resistance_atr"] = _d_res

    # Is price inside the zone band? (within 0.5 ATR of the zone price)
    # This is the "touch zone" — the candle wicks into the level
    ZONE_TOUCH_ATR = 0.5
    df["zone_in_support_zone"]    = (_d_supp  <= ZONE_TOUCH_ATR).astype(np.float32)
    df["zone_in_resistance_zone"] = (_d_res   <= ZONE_TOUCH_ATR).astype(np.float32)

    # Normalized price position between S and R: 0.0 = sitting on support, 1.0 = at resistance
    _sr_range = np.maximum(_z_res - _z_supp, atr)  # never zero
    df["zone_price_position"] = np.clip(
        (asset_close - _z_supp) / _sr_range, 0.0, 1.0
    ).astype(np.float32)

    # Approach direction: is price moving toward or away from each zone?
    # Positive = moving toward resistance (upward momentum relative to R)
    # Negative = moving toward support (downward momentum relative to S)
    _close_s   = pd.Series(asset_close)
    _close_d1  = _close_s.diff(1).fillna(0.0).values
    _close_d3  = _close_s.diff(3).fillna(0.0).values  # 3-bar momentum for smoother signal

    # Normalize by ATR so the Q-learner sees direction + magnitude
    _approach = np.clip(_close_d3 / (atr + 1e-8), -3.0, 3.0).astype(np.float32)
    df["zone_approach_direction"] = _approach

    # Zone volume quality: how one-sided is the volume at this zone?
    # Derived from the per-TF SNR distance columns which proxy the zone anchor prices.
    # We encode volume imbalance as a bull/bear ratio.
    # Since we don't have per-zone up/down vol in the dataframe (those come from ZoneSnapshotManager
    # at runtime), we proxy it using the close-position-within-bar (candle body direction) near zones.
    # This is a clean no-lookahead proxy: if price is near support and the bar is bullish → absorption.
    _open_col_local = "open_5m" if "open_5m" in df.columns else ("open" if "open" in df.columns else close_col)
    _bar_bullish = (df[close_col] >= df[_open_col_local]).astype(np.float32)

    # Volume quality at support: rolling fraction of bullish bars when near support
    _near_supp_mask = (_d_supp <= 1.5).astype(np.float32)
    _bull_at_supp   = pd.Series(_bar_bullish * _near_supp_mask)
    _supp_bars      = pd.Series(_near_supp_mask)
    _rolling_bull_at_supp = _bull_at_supp.rolling(20, min_periods=1).sum()
    _rolling_supp_bars    = _supp_bars.rolling(20, min_periods=1).sum().clip(lower=1)
    df["zone_support_vol_ratio"] = (_rolling_bull_at_supp / _rolling_supp_bars).values.astype(np.float32)

    # Volume quality at resistance: rolling fraction of bearish bars when near resistance
    _near_res_mask  = (_d_res <= 1.5).astype(np.float32)
    _bear_at_res    = pd.Series((1.0 - _bar_bullish) * _near_res_mask)
    _res_bars       = pd.Series(_near_res_mask)
    _rolling_bear_at_res = _bear_at_res.rolling(20, min_periods=1).sum()
    _rolling_res_bars    = _res_bars.rolling(20, min_periods=1).sum().clip(lower=1)
    df["zone_resistance_vol_ratio"] = (_rolling_bear_at_res / _rolling_res_bars).values.astype(np.float32)

    # MTF zone strength: how many TFs have an active zone near this price?
    # Counts how many of the per-TF distances (5m, 15m, 1h) are within 1 ATR → MTF confluence count
    _z_15m_active = (df["snr_dist_support_15m"].values <= 1.0).astype(np.float32)
    _z_1h_active  = (df["snr_dist_support_1h"].values  <= 1.0).astype(np.float32)
    _z_4h_active  = (df["snr_dist_support_4h"].values  <= 1.0).astype(np.float32)
    df["zone_support_confluence"] = (_z_15m_active + _z_1h_active + _z_4h_active).astype(np.float32)

    _z_15m_r_active = (df["snr_dist_resistance_15m"].values <= 1.0).astype(np.float32)
    _z_1h_r_active  = (df["snr_dist_resistance_1h"].values  <= 1.0).astype(np.float32)
    _z_4h_r_active  = (df["snr_dist_resistance_4h"].values  <= 1.0).astype(np.float32)
    df["zone_resistance_confluence"] = (_z_15m_r_active + _z_1h_r_active + _z_4h_r_active).astype(np.float32)

    # ── Bounce and rejection composite signals ────────────────────────────────
    # These are the most direct "zone + signal" combinations the Q-learner should learn:
    #   zone_bounce_signal:    price AT support + moving up + bullish bar → potential long setup
    #   zone_rejection_signal: price AT resistance + moving down + bearish bar → potential short setup
    #
    # Encoded as a continuous score [0, 1] so the Q-learner sees signal strength, not just binary.
    _at_support     = np.clip(1.0 - _d_supp / 2.0, 0.0, 1.0)   # 1.0 at zone, 0.0 2+ ATR away
    _at_resistance  = np.clip(1.0 - _d_res  / 2.0, 0.0, 1.0)
    _upward_move    = np.clip(_approach / 3.0, 0.0, 1.0)         # normalized upward momentum
    _downward_move  = np.clip(-_approach / 3.0, 0.0, 1.0)        # normalized downward momentum

    df["zone_bounce_signal"]    = (_at_support    * _upward_move    * _bar_bullish.values).astype(np.float32)
    df["zone_rejection_signal"] = (_at_resistance * _downward_move  * (1.0 - _bar_bullish.values)).astype(np.float32)


    return df


def compute_advanced_ml_targets(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 21+ advanced ML targets for Meta-Learner training.
    
    Targets:
    - Zone Liquidity (5): next_zone_idx, bars, distance, volume, confluence
    - Volatility Regime (10): volatility/regime/speed targets
    - Velocity (6): price velocity targets
    - Currency Divergence (4): CSM targets
    
    Returns: DataFrame with all targets added
    """
    try:
        logger.info("[MLTargets] Computing advanced ML targets (zone, volatility, velocity)...")
        config = DatasetConfig(sequence_length=60, prediction_length=7)
        prep = MLDatasetPreparation(data=aligned_df.copy(), config=config)
        
        # Zone Liquidity Targets
        zone_cols = prep._compute_next_zone_targets(n_future=20, zone_touch_pct=0.004)
        logger.info(f"[MLTargets] ✅ Zone targets: {len(zone_cols)} columns")
        
        # Volatility + Regime Speed Targets
        vol_cols = prep._compute_forward_volatility_targets(n_future=8, decay=0.85)
        logger.info(f"[MLTargets] ✅ Volatility targets: {len(vol_cols)} columns")
        
        speed_cols = prep._compute_forward_regime_speed_targets(n_future=8, decay=0.85)
        logger.info(f"[MLTargets] ✅ Regime speed targets: {len(speed_cols)} columns")
        
        # Velocity Targets
        vel_cols = prep._compute_forward_velocity_targets(n_future=8, decay=0.85)
        logger.info(f"[MLTargets] ✅ Velocity targets: {len(vel_cols)} columns")
        
        # CSM Targets (optional)
        csm_cols = prep._compute_forward_csm_targets()
        if csm_cols:
            logger.info(f"[MLTargets] ✅ CSM targets: {len(csm_cols)} columns")
        
        total = len(zone_cols) + len(vol_cols) + len(speed_cols) + len(vel_cols) + len(csm_cols)
        logger.info(f"[MLTargets] ✅ COMPLETE! Added {total} advanced ML targets")
        
        return prep.data
        
    except Exception as e:
        logger.warning(f"[MLTargets] ⚠️  Error during ML target computation: {e}. Continuing without advanced targets.")
        return aligned_df


def evaluate_expiries_for_symbol(symbol: str = "GLD", limit: int = 40000, framework: str = "keras", epochs: Optional[int] = None, meta_epochs: Optional[int] = None, q_epochs: Optional[int] = None):
    print(f"\n==================================================================================")
    print(f"      EVALUATING OPTION EXPIRY HORIZONS FOR SYMBOL: {symbol} (5m Execution | Framework: {framework.upper()})")
    print(f"==================================================================================")

    # 1. Fetch & Align real market data with expanded dataset
    logger.info("Fetching real market bars for %s (%d bars limit)...", symbol, limit)
    ltf_df = fetch_real_candles(symbol, timeframe="5m", limit=limit)
    if ltf_df.empty:
        print(f"Error: Could not fetch 5m candles for {symbol}")
        return

    ltf_anchor = ltf_df["timestamp"].min().to_pydatetime()
    tf_dfs = {"5m": ltf_df}

    import signal as _signal

    def _fetch_with_timeout(symbol, tf, limit, start, timeout_sec=90):
        """Fetch HTF candles with a hard timeout to prevent indefinite hangs."""
        result = [None]
        def _handler(signum, frame):
            raise TimeoutError(f"[DataFetch] Timed out fetching {tf} bars for {symbol}")
        prev = _signal.signal(_signal.SIGALRM, _handler)
        _signal.alarm(timeout_sec)
        try:
            result[0] = fetch_real_candles(symbol, timeframe=tf, limit=limit, start=start)
        except TimeoutError as e:
            logger.warning("%s — skipping timeframe.", e)
            result[0] = None
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, prev)
        return result[0]

    for tf in ["15m", "1h", "4h", "1d"]:
        start_str = _htf_start_for(ltf_anchor, tf, lookback_bars=1000)
        df_htf = _fetch_with_timeout(symbol, tf, limit, start_str)
        if df_htf is not None and not df_htf.empty:
            tf_dfs[tf] = df_htf

    aligned_df = align_multi_timeframe_datasets(tf_dfs, primary_tf="5m")
    aligned_df = aligned_df[aligned_df["timestamp"] >= pd.Timestamp(ltf_anchor)].reset_index(drop=True)

    # ── Compute & Inject All 21 CONTEXT_FEATURE_KEYS (MTF RSI, DXY Divergence, SNR Confluence) ──
    aligned_df = compute_full_context_features(aligned_df)

    close_col = "close_5m" if "close_5m" in aligned_df.columns else "close"
    open_col  = "open_5m"  if "open_5m"  in aligned_df.columns else "open"
    high_col  = "high_5m"  if "high_5m"  in aligned_df.columns else "high"
    low_col   = "low_5m"   if "low_5m"   in aligned_df.columns else "low"
    vol_col   = "volume_5m" if "volume_5m" in aligned_df.columns else "volume"

    # ── 2. Run 200+ Technical Indicators & Log Enrichment ──
    logger.info("[TI Enrichment] Enriching aligned dataset (%d rows) with 200+ Technical Indicators...", len(aligned_df))
    try:
        from app.core.analysis.technical_indicators import TechnicalIndicators, IndicatorConfig
        from app.core.ml.ti_meta_features import TI_NUMERIC_FEATURE_KEYS, CONTEXT_FEATURE_KEYS

        ti_calc = TechnicalIndicators(IndicatorConfig())
        ta_input = aligned_df.rename(columns={
            open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"
        })
        ti_enriched = ti_calc.calculate_all_indicators(ta_input, mode="training")

        ti_added_count = 0
        for col in TI_NUMERIC_FEATURE_KEYS:
            if col in ti_enriched.columns:
                aligned_df[col] = ti_enriched[col].astype(np.float32)
                ti_added_count += 1
            else:
                aligned_df[col] = 0.0

        logger.info(
            "✅ [TI Enrichment] Complete! Enriched dataset with %d Technical Indicators + %d Context Features (%d total columns).",
            ti_added_count, len(CONTEXT_FEATURE_KEYS), len(aligned_df.columns)
        )
    except Exception as ti_err:
        logger.warning("⚠ [TI Enrichment] Warning during TI calculation: %s", ti_err)

    # ── 3. Multi-Horizon Targets & Directional Bias Labeling (5m, 15m, 30m, 1h) ──
    aligned_df["forward_move_1"]  = aligned_df[close_col].shift(-1) - aligned_df[close_col]   # 5m
    aligned_df["forward_move_3"]  = aligned_df[close_col].shift(-3) - aligned_df[close_col]   # 15m
    aligned_df["forward_move_6"]  = aligned_df[close_col].shift(-6) - aligned_df[close_col]   # 30m
    aligned_df["forward_move_12"] = aligned_df[close_col].shift(-12) - aligned_df[close_col]  # 1h

    aligned_df["target_dir_5m"]  = (aligned_df["forward_move_1"]  > 0).astype(int)
    aligned_df["target_dir_15m"] = (aligned_df["forward_move_3"]  > 0).astype(int)
    aligned_df["target_dir_30m"] = (aligned_df["forward_move_6"]  > 0).astype(int)
    aligned_df["target_dir_1h"]  = (aligned_df["forward_move_12"] > 0).astype(int)

    aligned_df.dropna(subset=["forward_move_12"], inplace=True)

    # ── 3.5. Compute Advanced ML Targets (21+ targets for Meta-Learner) ──
    logger.info("[DataPrepare] Computing 21+ advanced ML targets for meta-learner multi-head training...")
    cols_before = len(aligned_df.columns)
    aligned_df = compute_advanced_ml_targets(aligned_df)
    cols_after = len(aligned_df.columns)
    logger.info(f"[DataPrepare] ✅ ML targets added: +{cols_after - cols_before} columns | Total columns: {cols_after}")

    # 70 / 15 / 15 Train / Validation / Test Split (Strict chronological order)
    n_total = len(aligned_df)
    train_idx = int(n_total * 0.70)
    val_idx = int(n_total * 0.85)

    train_df = aligned_df.iloc[:train_idx].reset_index(drop=True)
    val_df   = aligned_df.iloc[train_idx:val_idx].reset_index(drop=True)
    test_df  = aligned_df.iloc[val_idx:].reset_index(drop=True)

    print(f"Dataset Split -> Total: {n_total} | Train (70%): {len(train_df)} | Val (15%): {len(val_df)} | Test (15%): {len(test_df)}")

    # ── 5. Automatic Dataset Export & Zipping for Kaggle (Train, Val, Test, Full) ──
    try:
        import os
        import zipfile
        os.makedirs("data", exist_ok=True)

        csv_train_path = "data/train_40k.csv"
        csv_val_path   = "data/val_40k.csv"
        csv_test_path  = "data/test_40k.csv"
        csv_full_path  = "data/full_40k.csv"
        zip_export_path = "data/axe_meta_dataset.zip"

        train_df.to_csv(csv_train_path, index=False)
        val_df.to_csv(csv_val_path, index=False)
        test_df.to_csv(csv_test_path, index=False)
        aligned_df.to_csv(csv_full_path, index=False)

        with zipfile.ZipFile(zip_export_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_train_path, arcname="train_40k.csv")
            zipf.write(csv_val_path,   arcname="val_40k.csv")
            zipf.write(csv_test_path,  arcname="test_40k.csv")
            zipf.write(csv_full_path,  arcname="full_40k.csv")

        zip_mb = os.path.getsize(zip_export_path) / (1024 * 1024)
        print(f"📦 [DataExport] Exported & zipped all enriched splits (train/val/test/full) to: {zip_export_path} ({zip_mb:.2f} MB)")
    except Exception as exc:
        print(f"⚠ [DataExport] Dataset zip export warning: {exc}")

    # ── Training Data Diagnostics ──────────────────────────────────────────────
    bullish_count = int((train_df["forward_move_12"] > 0).sum())
    bearish_count = int((train_df["forward_move_12"] <= 0).sum())
    bull_pct = 100.0 * bullish_count / len(train_df)
    print(f"\n[TrainData] Direction Split (12-bar): Bullish={bullish_count} ({bull_pct:.1f}%) | Bearish={bearish_count} ({100-bull_pct:.1f}%)")

    # DXY / context feature coverage audit
    dxy_cols = [c for c in aligned_df.columns if "dxy" in c.lower() or "dollar" in c.lower()]
    context_cols_present = [c for c in aligned_df.columns if any(k in c for k in ["cross_", "regime_", "mtf_rsi", "snr_dist", "asset_slow", "asset_fast"])]
    print(f"[TrainData] DXY columns in dataset ({len(dxy_cols)}): {dxy_cols if dxy_cols else '⚠ NONE — DXY not aligned into dataset'}")
    print(f"[TrainData] Context feature columns ({len(context_cols_present)}): {context_cols_present[:8]}{'...' if len(context_cols_present) > 8 else ''}")

    # Forward move distribution summary for each horizon
    for h_bars, h_label in zip([1, 3, 6, 12], ["5m", "15m", "30m", "1h"]):
        fwd = aligned_df[close_col].shift(-h_bars) - aligned_df[close_col]
        fwd_bull = int((fwd > 0).sum())
        fwd_bear = int((fwd <= 0).sum())
        fwd_mean = float(fwd.mean())
        fwd_std = float(fwd.std())
        print(f"[TrainData] Horizon {h_label:>4} ({h_bars:>2} bars): Bull={fwd_bull:>5} ({100*fwd_bull/(fwd_bull+fwd_bear):.1f}%) | Bear={fwd_bear:>5} | mean={fwd_mean:+.4f} | std={fwd_std:.4f}")

    print(f"[TrainData] Feature vector sample size (SIGNAL_META_FEATURE_COUNT): {SIGNAL_META_FEATURE_COUNT}")
    sample_f = _row_to_feature_dict(train_df.iloc[0], close_col, vol_col)
    print(f"[TrainData] Feature dict keys from first training row ({len(sample_f)} keys): {list(sample_f.keys())[:10]}... ({'DXY present' if any('dxy' in k.lower() for k in sample_f) else '⚠ DXY MISSING from feature_dict'})")
    print()

    # 2. Train Two-Tier Ensemble Learners once on Train Set
    if framework == "keras":
        from app.core.ml.keras_signal_meta_learner import KerasSignalMetaLearner
        from app.core.options.keras_trade_executor import KerasTradeExecutor
        meta_learner = KerasSignalMetaLearner(num_features=238, lookback_bars=48, replay_capacity=20000)
        q_executor   = KerasTradeExecutor(seq_len=1, n_features=EXECUTOR_STATE_DIM)
    else:
        meta_learner = OnlineSignalMetaLearner(replay_capacity=20000)
        # Do NOT pass input_dim — the default META_PREDICT_WINDOW * DECISION_FEATURE_COUNT = 35700
        # Passing SIGNAL_META_FEATURE_COUNT=238 creates a (B,1,238) input to the Conv1D+LSTM tower
        # → BatchNorm1d sees batch_size=1 sequences → NaN losses
        q_executor = OptionsQExecutor(device="cpu")

    # ── Scaler Fitting (after meta_learner exists) ──
    try:
        from app.core.ml.signal_meta_learner import FeatureScaler
        from app.core.ml.ti_meta_features import DECISION_FEATURE_KEYS
        _scaler = FeatureScaler()
        # Fit ONLY on the feature columns the meta-learner uses (prevents shape mismatch)
        _decision_cols = [c for c in DECISION_FEATURE_KEYS if c in train_df.columns]
        if not _decision_cols:
            # Fallback: use numeric non-target columns
            _decision_cols = [c for c in train_df.columns if c not in ("timestamp", "Time") and np.issubdtype(train_df[c].dtype, np.number)]
        _scaler.fit(train_df[_decision_cols].values)
        if hasattr(meta_learner, "scaler"):
            meta_learner.scaler = _scaler
        logger.info("✅ [Scaler] FeatureScaler fitted on train split (%d rows, %d cols).", len(train_df), len(_decision_cols))
    except Exception as _scaler_err:
        logger.warning("⚠ [Scaler] Scaler fitting skipped: %s", _scaler_err)

    zone_manager = ZoneSnapshotManager(max_snapshots=20)
    account = AccountContext()

    from app.core.analysis.support_resistance import (
        detect_snr_levels_sequential,
        create_clustered_zones_sequential,
    )

    def update_real_snr_snapshot(
        df_full: pd.DataFrame,
        up_to_idx: int,
        zm: ZoneSnapshotManager,
        timeframe: str = "15m",
        lookback_period: int = 500,
    ):
        """
        CRITICAL (NO LOOKAHEAD LEAKAGE):
        Detect real S&R levels and volume profiles using ONLY historical price data up to `up_to_idx`.
        """
        if up_to_idx < 20:
            return

        # ── Dynamic HTF column resolution (Fix 3.1) ──────────────────────────────
        # Select the correct price columns for the requested timeframe.
        # If the HTF columns are absent (data not fetched), return early gracefully.
        primary_tf = "5m"   # matches the module-level primary_tf used in the pipeline
        if timeframe == primary_tf:
            _high  = high_col
            _low   = low_col
            _close = close_col
            _open  = open_col
            _vol   = vol_col
        else:
            _high  = f"high_{timeframe}"
            _low   = f"low_{timeframe}"
            _close = f"close_{timeframe}"
            _open  = f"open_{timeframe}"
            _vol   = f"volume_{timeframe}"
            # Graceful fallback: if HTF columns absent, skip (requirement 3.2)
            if not all(c in df_full.columns for c in (_high, _low, _close)):
                return

        df_slice = df_full.iloc[max(0, up_to_idx - lookback_period): up_to_idx + 1].copy()
        df_slice = df_slice.rename(columns={
            _high: "High", _low: "Low", _close: "Close", _open: "Open", _vol: "Volume"
        })

        if "High" not in df_slice.columns or "Low" not in df_slice.columns:
            return

        levels = detect_snr_levels_sequential(
            price_data=df_slice,
            up_to_index=len(df_slice) - 1,
            lookback_period=min(lookback_period, len(df_slice) - 1),
        )

        if not levels:
            return

        raw_zones = create_clustered_zones_sequential(
            levels=levels,
            price_data_slice=df_slice,
            n_clusters=min(8, max(3, len(levels))),
        )

        if raw_zones:
            ts = df_slice.index[-1] if hasattr(df_slice.index[-1], "to_pydatetime") else None
            zm.add_snapshot(
                snapshot_id=f"snap_{up_to_idx}",
                timeframe=timeframe,
                zones_raw=raw_zones,
                timestamp=ts if isinstance(ts, datetime) else None,
            )

    # ── Phase 1: Meta-Learner Training (100% Full-Set Systematic Epoch Sweep) ──
    # ── Phase 1: Meta-Learner Training ──
    META_EPOCHS = meta_epochs if meta_epochs is not None else (epochs if epochs is not None else (10 if framework == "keras" else 50))
    BATCH_SIZE_META = 64
    from app.core.ml.ti_meta_features import DECISION_FEATURE_KEYS, DECISION_FEATURE_COUNT, SIGNAL_META_LOOKBACK_BARS as _MPW

    # Pre-build the numeric feature matrix for fast window slicing (no lookahead)
    # Shape: (len(train_df), DECISION_FEATURE_COUNT) — only the 238 TI+context features
    _feat_cols = [c for c in DECISION_FEATURE_KEYS if c in train_df.columns]
    _missing   = [c for c in DECISION_FEATURE_KEYS if c not in train_df.columns]
    if _missing:
        logger.warning("[Phase1] %d feature keys missing from train_df, will be zero: %s", len(_missing), _missing[:5])
    # Build matrix: present cols first, then zero-pad missing
    _mat_parts = [train_df[c].fillna(0.0).values.reshape(-1, 1) for c in _feat_cols]
    for _ in _missing:
        _mat_parts.append(np.zeros((len(train_df), 1), dtype=np.float32))
    # Reorder to match DECISION_FEATURE_KEYS order
    _col_idx = {c: i for i, c in enumerate(_feat_cols)}
    _ordered = []
    for c in DECISION_FEATURE_KEYS:
        if c in _col_idx:
            _ordered.append(train_df[c].fillna(0.0).values.astype(np.float32))
        else:
            _ordered.append(np.zeros(len(train_df), dtype=np.float32))
    train_num_matrix = np.column_stack(_ordered).astype(np.float32)  # (N, 238)
    # Do NOT pre-scale here — extract_features applies the per-column scaler
    # on the 2D window (150, 238) before flattening, so scaling happens once.
    np.nan_to_num(train_num_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_window(abs_idx: int) -> np.ndarray:
        """Return (META_PREDICT_WINDOW, DECISION_FEATURE_COUNT) float32, left-zero-padded."""
        start = abs_idx - _MPW + 1
        if start >= 0:
            return train_num_matrix[start: abs_idx + 1]
        pad = np.zeros((-start, DECISION_FEATURE_COUNT), dtype=np.float32)
        return np.vstack([pad, train_num_matrix[: abs_idx + 1]])

    total_train_bars = len(train_df) - 13
    steps_per_epoch  = max(1, total_train_bars // BATCH_SIZE_META)
    print(f"\n[Phase 1] Meta-Learner Training: {META_EPOCHS} epochs, {steps_per_epoch} steps/epoch")
    print(f"  {'Epoch':>6} | {'AvgLoss':>9} | {'Q':>8} | {'Str':>8} | {'Pips':>8} | {'Risk':>8} | {'Liq':>8} | {'Rev':>8} | {'Sel':>8} | {'Zone':>8} | {'Vol':>8} | {'Vel':>8} | {'5mWR':>6} {'15mWR':>6} {'30mWR':>6} {'1hWR':>6} {'AvgWR':>6} SelAcc RevAcc | Status")
    print(f"  {'-'*185}")

    best_val_avg_wr  = -1.0
    best_meta_weights = None

    # Val matrix for end-of-epoch WR evaluation (same feature ordering)
    _val_ordered = []
    for c in DECISION_FEATURE_KEYS:
        if c in val_df.columns:
            _val_ordered.append(val_df[c].fillna(0.0).values.astype(np.float32))
        else:
            _val_ordered.append(np.zeros(len(val_df), dtype=np.float32))
    val_num_matrix = np.column_stack(_val_ordered).astype(np.float32)
    # Do NOT pre-scale — val WR evaluation applies scaler inside extract_features / directly below
    np.nan_to_num(val_num_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Pre-compute direction targets for val WR check (same as notebook)
    val_dir_targets = np.zeros((len(val_df), 4), dtype=np.float32)
    for h_i, h_bars in enumerate([1, 3, 6, 12]):
        fwd = val_df[close_col].shift(-h_bars) - val_df[close_col]
        val_dir_targets[:, h_i] = (fwd > 0).astype(np.float32).fillna(0.0).values

    for ep in range(META_EPOCHS):
        epoch_indices = list(range(total_train_bars))
        if ep > 0:
            random.shuffle(epoch_indices)

        ep_tot = ep_q = ep_str = ep_pips = ep_risk = ep_liq = ep_rev = ep_sel = 0.0
        ep_zone = ep_vol = ep_vel = 0.0
        epoch_steps = 0

        for idx in epoch_indices:
            # Build causal 150-bar feature window for this bar (no lookahead)
            feat_window = _build_window(idx)   # (150, 238)

            fut_highs  = train_df[high_col].iloc[idx+1:idx+13].values
            fut_lows   = train_df[low_col].iloc[idx+1:idx+13].values
            fut_closes = train_df[close_col].iloc[idx+1:idx+13].values

            fwd12 = float(train_df["forward_move_12"].iloc[idx])

            # Pull ML target values for this bar (graceful fallback to 0.0 if absent)
            _row = train_df.iloc[idx]
            _vol_regime = float(_row.get("Volatility_Regime_next", _row.get("adv_target_Volatility_Regime_next", 0.0)) or 0.0)
            _vel_net    = float(_row.get("Price_Velocity_Net_next", _row.get("adv_target_Price_Velocity_Net_next", 0.0)) or 0.0)
            _zone_idx   = float(_row.get("adv_target_next_zone_idx", 0.0) or 0.0)

            # record_experience accepts a 2D numpy array directly
            meta_learner.record_experience(
                feature_dict=feat_window,                           # (150, 238) 2D array
                signal_id=f"sig_{ep}_{idx}",
                symbol=symbol,
                direction="bullish" if fwd12 > 0 else "bearish",
                entry_price=float(train_df[close_col].iloc[idx]),
                future_highs=fut_highs,
                future_lows=fut_lows,
                future_closes=fut_closes,
                vol_regime=_vol_regime,
                vel_net=_vel_net,
                zone_idx=_zone_idx,
            )

            if (idx + 1) % BATCH_SIZE_META == 0:
                metrics = meta_learner.train_step(batch_size=BATCH_SIZE_META)
                loss = metrics.get("loss", 0.0)
                if loss > 0 and np.isfinite(loss):
                    ep_tot  += loss
                    ep_q    += metrics.get("loss_q", 0)
                    ep_str  += metrics.get("loss_strength", 0)
                    ep_pips += metrics.get("loss_pips", 0)
                    ep_risk += metrics.get("loss_risk", 0)
                    ep_liq  += metrics.get("loss_liquidity", 0)
                    ep_rev  += metrics.get("loss_reversal", 0)
                    ep_sel  += metrics.get("loss_selector", metrics.get("loss_sel", 0))
                    ep_zone += metrics.get("loss_zone", 0)
                    ep_vol  += metrics.get("loss_vol", 0)
                    ep_vel  += metrics.get("loss_vel", 0)
                    epoch_steps += 1

                    # Intra-epoch progress (every 100 steps or last step — mirrors notebook)
                    if epoch_steps % 100 == 0 or epoch_steps == steps_per_epoch:
                        print(
                            f"  [Ep {ep+1:>2}/{META_EPOCHS} | Step {epoch_steps:>4}/{steps_per_epoch}] "
                            f"Tot={loss:.4e} Q={metrics.get('loss_q',0):.4e} Str={metrics.get('loss_strength',0):.4e} "
                            f"Sel={ep_sel/epoch_steps:.4e} Zone={ep_zone/epoch_steps:.4e} "
                            f"Vol={ep_vol/epoch_steps:.4e} Vel={ep_vel/epoch_steps:.4e}",
                            flush=True,
                        )

        # ── End-of-epoch: val direction accuracy (mirrors notebook) ──
        s = max(epoch_steps, 1)
        avg_tot  = ep_tot  / s
        avg_q    = ep_q    / s
        avg_str  = ep_str  / s
        avg_pips = ep_pips / s
        avg_risk = ep_risk / s
        avg_liq  = ep_liq  / s
        avg_rev  = ep_rev  / s
        avg_sel  = ep_sel  / s
        avg_zone = ep_zone / s
        avg_vol  = ep_vol  / s
        avg_vel  = ep_vel  / s

        val_corrects = [0, 0, 0, 0]
        val_count = 0
        N_val_bars = len(val_df) - 13

        if N_val_bars > 0 and framework != "keras":
            import torch
            # Scale val_num_matrix once for the direct net() call in WR evaluation
            if hasattr(meta_learner, "scaler") and meta_learner.scaler.fitted:
                val_num_scaled = meta_learner.scaler.transform(val_num_matrix)  # (N, 238), broadcast-safe
                np.nan_to_num(val_num_scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                val_num_scaled = val_num_matrix
            meta_learner.net.eval()
            val_sel_correct = 0
            val_rev_correct = 0
            val_rev_total   = 0
            with torch.no_grad():
                for v_start in range(0, N_val_bars, BATCH_SIZE_META):
                    v_end = min(v_start + BATCH_SIZE_META, N_val_bars)
                    batch_windows = np.stack([
                        val_num_scaled[max(0, i - _MPW + 1): i + 1] if i >= _MPW - 1
                        else np.vstack([np.zeros((_MPW - 1 - i, DECISION_FEATURE_COUNT), dtype=np.float32),
                                        val_num_scaled[:i + 1]])
                        for i in range(v_start, v_end)
                    ])  # (B, 150, 238)
                    x_t = torch.tensor(batch_windows, dtype=torch.float32)
                    vq_vals, vstr, _, _, _, vrev, _, _, vsel = meta_learner.net(x_t, return_aux=True)
                    vy_q = torch.tensor(val_dir_targets[v_start:v_end], dtype=torch.float32)
                    for h in range(4):
                        pred = (vq_vals[:, h] > 0.5).long()
                        true = (vy_q[:, h]  > 0.5).long()
                        val_corrects[h] += (pred == true).sum().item()
                    # Selector accuracy: did the model pick the horizon with the highest actual strength?
                    # True best horizon = argmax of per-horizon direction targets scaled to [0,1]
                    true_best_h = vy_q.argmax(dim=1).long()
                    pred_best_h = vstr.argmax(dim=1).long()
                    val_sel_correct += (pred_best_h == true_best_h).sum().item()
                    # Reversal accuracy: reversal_head > 0.5 vs actual reversal label
                    # Actual reversal = 5m and 15m directions differ (dirs[1] != dirs[3])
                    true_rev = (vy_q[:, 0] != vy_q[:, 1]).float()
                    pred_rev = (vrev.squeeze(-1) > 0.5).float()
                    val_rev_correct += (pred_rev == true_rev).sum().item()
                    val_rev_total   += (v_end - v_start)
                    val_count += (v_end - v_start)

        if val_count > 0:
            val_wrs = [c / val_count for c in val_corrects]
            val_avg_wr = float(np.mean(val_wrs))
            val_sel_acc = val_sel_correct / val_count if val_count > 0 else 0.0
            val_rev_acc = val_rev_correct / val_rev_total if val_rev_total > 0 else 0.0
            if val_avg_wr > best_val_avg_wr:
                best_val_avg_wr = val_avg_wr
                if hasattr(meta_learner, "net"):
                    best_meta_weights = {k: v.cpu().clone() for k, v in meta_learner.net.state_dict().items()}
                elif hasattr(meta_learner, "get_weights"):
                    best_meta_weights = meta_learner.get_weights()
            status = "✓ NEW BEST" if val_avg_wr >= best_val_avg_wr - 1e-4 else ""
        else:
            val_wrs = [0.0, 0.0, 0.0, 0.0]
            val_avg_wr = 0.0
            val_sel_acc = 0.0
            val_rev_acc = 0.0
            status = ""

        print(
            f"  {ep+1:>6} | {avg_tot:>9.4e} | {avg_q:>8.4e} | {avg_str:>8.4e} | "
            f"{avg_pips:>8.4e} | {avg_risk:>8.4e} | {avg_liq:>8.4e} | {avg_rev:>8.4e} | "
            f"{avg_sel:>8.4e} | {avg_zone:>8.4e} | {avg_vol:>8.4e} | {avg_vel:>8.4e} | "
            f"{val_wrs[0]:>6.2%} {val_wrs[1]:>6.2%} {val_wrs[2]:>6.2%} {val_wrs[3]:>6.2%} "
            f"{val_avg_wr:>6.2%} SelAcc={val_sel_acc:.1%} RevAcc={val_rev_acc:.1%} | {status}",
            flush=True,
        )

    if best_meta_weights is not None:
        if hasattr(meta_learner, "net"):
            meta_learner.net.load_state_dict(best_meta_weights)
        elif hasattr(meta_learner, "set_weights"):
            meta_learner.set_weights(best_meta_weights)
        print(f"\n✓ Loaded best meta-learner weights (Avg WR={best_val_avg_wr:.2%})")

    print(f"\n✓ Meta-Learner Training Complete: {META_EPOCHS} epochs")
    print(f"  Final training loss: {avg_tot:.4e}")
    print(f"  Best validation Avg WR: {best_val_avg_wr:.2%}")

    # ── Phase 2: Per-Horizon Q-Learning (mirrors notebook Cell 8) ──────────────
    # Precompute meta outputs for all training bars, build static 28-dim state
    # vectors using real SNR zones, then train 4 independent Q-heads per horizon.
    from app.core.options.q_executor import (
        ExecutorQNetwork, build_feat_window, Q_LOOKBACK,
        HTFBiasPackage as _HTFBiasPackage,
    )
    import torch, time as _time

    def get_nearest_zones(zones, current_price):
        """Mirror of notebook Cell 3 get_nearest_zones — returns (supp_dict, res_dict) or (None, None)."""
        supports    = [z for z in zones if z[1] <= current_price]
        resistances = [z for z in zones if z[1] >= current_price]
        nearest_supp = max(supports,    key=lambda z: z[1]) if supports    else None
        nearest_res  = min(resistances, key=lambda z: z[1]) if resistances else None
        def _to_rec(z):
            if z is None: return None
            _, price, _, vol = z
            total = vol['up_volume'] + vol['down_volume']
            ratio = (vol['up_volume'] - vol['down_volume']) / (total + 1e-6)
            return {'price_level': price, 'volume_delta_ratio': ratio, 'volume': vol}
        return _to_rec(nearest_supp), _to_rec(nearest_res)

    device_q = torch.device("cpu")
    HORIZON_BARS_LIST = [1, 3, 6, 12]
    HORIZON_LABELS_Q  = ["5m", "15m", "30m", "1h"]
    H_WAIT, H_CALL, H_PUT = 0, 1, 2
    N_train_q = len(train_df) - 13

    # ── Precompute meta outputs (batch inference, no grad) ───────────────────
    print(f"\n[Precompute] Meta features for {N_train_q} steps...")
    t0 = _time.time()
    PRECOMPUTE_BATCH = 128
    meta_strengths_q = np.zeros((N_train_q, 4), dtype=np.float32)
    meta_qmax_q      = np.zeros(N_train_q, dtype=np.float32)
    meta_rev_q       = np.zeros(N_train_q, dtype=np.float32)
    meta_mfe_q       = np.zeros(N_train_q, dtype=np.float32)
    meta_mae_q       = np.zeros(N_train_q, dtype=np.float32)

    meta_learner.net.eval()
    with torch.no_grad():
        for start in range(0, N_train_q, PRECOMPUTE_BATCH):
            end = min(start + PRECOMPUTE_BATCH, N_train_q)
            batch_wins = np.stack([_build_window(i) for i in range(start, end)])  # (B,150,238)
            x_t = torch.tensor(batch_wins, dtype=torch.float32)
            q_v, str_v, pip_v, risk_v, liq_v, rev_v = meta_learner.net(x_t)
            meta_strengths_q[start:end] = str_v.cpu().numpy()
            meta_qmax_q[start:end]      = q_v.max(dim=1).values.cpu().numpy()
            meta_rev_q[start:end]       = rev_v.squeeze(-1).cpu().numpy() if rev_v.ndim > 1 else rev_v.cpu().numpy()
            if risk_v.shape[-1] >= 2:
                meta_mfe_q[start:end]   = risk_v[:, 0].cpu().numpy()
                meta_mae_q[start:end]   = risk_v[:, 1].cpu().numpy()

    meta_strengths_q = np.nan_to_num(meta_strengths_q, nan=0.5, posinf=1.0, neginf=0.0)
    meta_qmax_q      = np.nan_to_num(meta_qmax_q, nan=0.5)
    meta_rev_q       = np.nan_to_num(meta_rev_q, nan=0.2)
    meta_mfe_q       = np.nan_to_num(meta_mfe_q, nan=0.5)
    meta_mae_q       = np.nan_to_num(meta_mae_q, nan=0.15)
    print(f"Meta precompute done in {_time.time()-t0:.1f}s")

    # ── Precompute SNR zones ─────────────────────────────────────────────────
    print("[Precompute] SNR zones...")
    t1 = _time.time()
    price_data_hl_q = train_df[[open_col, high_col, low_col, close_col, vol_col]].rename(
        columns={open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"})
    close_prices_q = train_df[close_col].values.astype(np.float64)
    # Resolve column names locally (mirrors eval script column detection)
    _atr_col_q     = next((c for c in ["ATR_5m", "atr_5m", "ATR", "atr"] if c in train_df.columns), None)
    _up_vol_col_q  = next((c for c in ["Bar_Volume_Up_5m", "up_vol"] if c in train_df.columns), None)
    _dn_vol_col_q  = next((c for c in ["Bar_Volume_Down_5m", "down_vol"] if c in train_df.columns), None)
    atr_vals_q  = train_df[_atr_col_q].values.astype(np.float64) if _atr_col_q else close_prices_q * 0.005
    up_vols_q   = train_df[_up_vol_col_q].values.astype(np.float64) if _up_vol_col_q else np.zeros(len(train_df))
    dn_vols_q   = train_df[_dn_vol_col_q].values.astype(np.float64) if _dn_vol_col_q else np.zeros(len(train_df))

    nearest_supp_q = [None] * N_train_q
    nearest_res_q  = [None] * N_train_q
    _last_zones_q  = []
    ZONE_LOOKBACK_PERIOD = 500
    ZONE_MIN_DIST_PCT    = 0.5
    for i in range(N_train_q):
        if i % 5 == 0 or not _last_zones_q:
            lb = min(ZONE_LOOKBACK_PERIOD, i + 1)
            lvls = detect_snr_levels_sequential(price_data_hl_q, up_to_index=i, lookback_period=lb,
                                                min_distance_pct=ZONE_MIN_DIST_PCT) if i >= 20 else []
            _sl = price_data_hl_q.iloc[max(0, i - ZONE_LOOKBACK_PERIOD): i + 1]
            _last_zones_q = create_clustered_zones_sequential(lvls, _sl, n_clusters=min(8, max(3, len(lvls)))) if lvls else []
        ns_q, nr_q = get_nearest_zones(_last_zones_q, close_prices_q[i])
        nearest_supp_q[i] = ns_q
        nearest_res_q[i]  = nr_q
        if (i + 1) % max(1, N_train_q // 5) == 0 or i == N_train_q - 1:
            print(f"  zones {i+1}/{N_train_q}")
    print(f"Zone precompute done in {_time.time()-t1:.1f}s")

    # ── Precompute 28-dim static state vectors ───────────────────────────────
    print("[Precompute] static state features...")
    static_states_q = np.zeros((N_train_q, 28), dtype=np.float32)
    for i in range(N_train_q):
        row = train_df.iloc[i]
        cp  = close_prices_q[i]
        atr = max(0.01, atr_vals_q[i])
        bv, sv_i = up_vols_q[i], dn_vols_q[i]
        ts = row.get("timestamp", None)
        hour_f, dow_i, phase_i = 14.5, 1, "off_hours"
        if ts is not None:
            try:
                ts_pd = pd.Timestamp(ts)
                if ts_pd.tzinfo is None: ts_pd = ts_pd.tz_localize("UTC")
                ts_et = ts_pd.tz_convert("America/New_York")
                hour_f = ts_et.hour + ts_et.minute / 60.0
                dow_i = ts_et.dayofweek
                if   9.5 <= hour_f < 10.5: phase_i = "nyse_open"
                elif 15.0 <= hour_f < 16.0: phase_i = "nyse_power_hour"
                elif 9.5 <= hour_f < 16.0:  phase_i = "regular_hours"
            except Exception: pass
        sv_vec = meta_strengths_q[i]
        opt_h  = int(np.argmax(sv_vec))
        meta_score = float(sv_vec[opt_h])
        dir_flag = 1.0 if meta_score > 0.5 else (-1.0 if meta_score < 0.5 else 0.0)
        ns_q = nearest_supp_q[i]; nr_q = nearest_res_q[i]
        supp_dist = abs(cp - ns_q["price_level"]) / cp if ns_q else 1.0
        res_dist  = abs(cp - nr_q["price_level"]) / cp if nr_q else 1.0
        supp_vol  = ns_q["volume_delta_ratio"] if ns_q else 0.0
        res_vol   = nr_q["volume_delta_ratio"] if nr_q else 0.0
        total_vol = bv + sv_i
        vdr = (bv - sv_i) / (total_vol + 1e-6)
        static_states_q[i] = [
            dir_flag, meta_score, float(meta_rev_q[i]), float(meta_qmax_q[i]),
            float(meta_mfe_q[i]), float(meta_mae_q[i]),
            sv_vec[0], sv_vec[1], sv_vec[2], sv_vec[3],
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            atr / cp, supp_dist, res_dist, supp_vol, res_vol, vdr, 0.0,
            np.sin(2*np.pi*hour_f/24.0), np.cos(2*np.pi*hour_f/24.0),
            dow_i / 6.0,
            1.0 if phase_i == "nyse_open" else 0.0,
            1.0 if phase_i == "nyse_power_hour" else 0.0,
        ]
    static_states_q = np.nan_to_num(static_states_q, nan=0.0, posinf=0.0, neginf=0.0)
    print("Static state cache ready.")

    # ── Instantiate Q-net + per-horizon replay buffers ───────────────────────
    num_features_q = len(train_num_matrix[0]) if len(train_num_matrix.shape) > 1 else DECISION_FEATURE_COUNT
    q_net    = ExecutorQNetwork(num_features=num_features_q).to(device_q)
    q_target = ExecutorQNetwork(num_features=num_features_q).to(device_q)
    q_target.load_state_dict(q_net.state_dict())
    q_opt = torch.optim.AdamW(q_net.parameters(), lr=1e-3, weight_decay=1e-4)

    Q_EPOCHS      = q_epochs if q_epochs is not None else (epochs if epochs is not None else 50)
    BATCH_SIZE_Q  = min(256, max(32, N_train_q // 10))
    BUFFER_CAP_Q  = 30_000
    replay_buffers_q = [[] for _ in range(4)]   # one per horizon

    epsilon_q     = 1.0
    eps_min_q     = 0.05
    eps_decay_q   = 0.92
    mask_engine_q = HardActionMask()

    print(f"\n[Phase 2] Per-Horizon Q-Learning (4 heads × {Q_EPOCHS} epochs)...")
    print(f"  {'Epoch':>5} | {'Loss':>10} | {'eps':>5} | {'5m W/C/P':>12} | {'15m W/C/P':>12} | {'30m W/C/P':>12} | {'1h W/C/P':>12}")
    print(f"  {'-'*80}")

    best_q_weights = None
    best_q_wr = 0.0

    for q_epoch in range(Q_EPOCHS):
        open_positions_q = {h: None for h in range(4)}
        win_streaks_q    = {h: 0 for h in range(4)}
        loss_streaks_q   = {h: 0 for h in range(4)}
        action_counts_q  = {h: {H_WAIT: 0, H_CALL: 0, H_PUT: 0} for h in range(4)}
        call_wins_q  = {h: 0 for h in range(4)}
        call_losses_q= {h: 0 for h in range(4)}
        put_wins_q   = {h: 0 for h in range(4)}
        put_losses_q = {h: 0 for h in range(4)}
        reward_sum_q   = {h: {H_WAIT:0.0, H_CALL:0.0, H_PUT:0.0} for h in range(4)}
        reward_count_q = {h: {H_WAIT:0,   H_CALL:0,   H_PUT:0}   for h in range(4)}
        _q_loss_acc = 0.0; _q_steps = 0

        for i in range(N_train_q):
            cp  = close_prices_q[i]
            atr = max(0.01, atr_vals_q[i])
            bv, sv_i = up_vols_q[i], dn_vols_q[i]
            ns_q = nearest_supp_q[i]; nr_q = nearest_res_q[i]
            feat_w = build_feat_window(train_num_matrix, i, Q_LOOKBACK)

            for h in range(4):
                lookahead = HORIZON_BARS_LIST[h]

                # Auto-expire
                if open_positions_q[h] is not None:
                    bars_held = i - open_positions_q[h]["entry_i"]
                    if bars_held >= open_positions_q[h]["horizon"]:
                        ep = open_positions_q[h]["entry_price"]
                        pnl = (cp - ep) / (ep + 1e-8)
                        if open_positions_q[h]["action"] == H_PUT: pnl = -pnl
                        if pnl > 0:
                            win_streaks_q[h] += 1; loss_streaks_q[h] = 0
                            (call_wins_q if open_positions_q[h]["action"]==H_CALL else put_wins_q)[h] += 1
                        else:
                            loss_streaks_q[h] += 1; win_streaks_q[h] = 0
                            (call_losses_q if open_positions_q[h]["action"]==H_CALL else put_losses_q)[h] += 1
                        settle_r = float(np.clip(pnl - 0.0005, -0.05, 0.05))
                        sc = static_states_q[i].copy(); sc[12] = float(pnl)
                        nf = static_states_q[min(i+1, N_train_q-1)].copy(); nf[12] = 0.0
                        nfw = build_feat_window(train_num_matrix, min(i+1, N_train_q-1), Q_LOOKBACK)
                        replay_buffers_q[h].append((feat_w, sc, H_WAIT, settle_r, nfw, nf))
                        if len(replay_buffers_q[h]) > BUFFER_CAP_Q: replay_buffers_q[h].pop(0)
                        open_positions_q[h] = None

                unreal = 0.0
                has_open = open_positions_q[h] is not None
                if has_open:
                    unreal = (cp - open_positions_q[h]["entry_price"]) / (open_positions_q[h]["entry_price"] + 1e-8)
                    if open_positions_q[h]["action"] == H_PUT: unreal = -unreal

                if has_open:
                    h_mask = np.array([1, 0, 0], dtype=np.int32)
                else:
                    # Notebook-style mask: proximity + volume confirmation
                    proximity_band = max(atr * 0.75, cp * 0.003)
                    supp_ok = ns_q is not None and abs(cp - ns_q['price_level']) <= proximity_band
                    res_ok  = nr_q is not None and abs(cp - nr_q['price_level']) <= proximity_band
                    total_v = bv + sv_i
                    call_ok = int(supp_ok and (total_v <= 0 or bv >= sv_i * 0.8))
                    put_ok  = int(res_ok  and (total_v <= 0 or sv_i >= bv * 0.8))
                    h_mask = np.array([1, call_ok, put_ok], dtype=np.int32)

                state = static_states_q[i].copy()
                state[11] = 1.0 if has_open else 0.0
                state[12] = float(unreal)
                state[13] = win_streaks_q[h] / 10.0
                state[14] = loss_streaks_q[h] / 10.0
                state[15] = float(h) / 3.0

                valid = [a for a in range(3) if h_mask[a] == 1] or [H_WAIT]
                if random.random() < epsilon_q:
                    action = random.choice(valid)
                else:
                    q_net.eval()
                    with torch.no_grad():
                        fw_t = torch.tensor(feat_w[None], dtype=torch.float32)
                        st_t = torch.tensor(state[None],  dtype=torch.float32)
                        logits = q_net(fw_t, st_t, horizon_idx=h).squeeze(0).cpu().numpy()
                        action = int(np.argmax(np.where(h_mask==1, logits, -1e9)))
                action_counts_q[h][action] += 1

                if i + lookahead >= N_train_q:
                    continue
                expiry_cp = close_prices_q[i + lookahead]
                fwd_pct = float(np.clip((expiry_cp - cp) / (cp + 1e-8), -0.05, 0.05))

                if not has_open and action == H_CALL:
                    open_positions_q[h] = {"action": H_CALL, "entry_price": cp, "entry_i": i, "horizon": lookahead}
                elif not has_open and action == H_PUT:
                    open_positions_q[h] = {"action": H_PUT,  "entry_price": cp, "entry_i": i, "horizon": lookahead}

                if action == H_CALL:
                    reward = fwd_pct - 0.0005
                elif action == H_PUT:
                    reward = -fwd_pct - 0.0005
                else:
                    h_str = float(meta_strengths_q[i][h])
                    if (h_mask[H_CALL]==1 or h_mask[H_PUT]==1) and h_str >= 0.60 and abs(fwd_pct) >= 0.0015:
                        reward = -abs(fwd_pct)
                    else:
                        reward = 0.001
                reward = float(np.clip(reward, -0.05, 0.05))

                ni = min(i+1, N_train_q-1)
                ns_next = static_states_q[ni].copy()
                ns_next[11] = 1.0 if open_positions_q[h] is not None else 0.0
                ns_next[12] = 0.0
                ns_next[13] = win_streaks_q[h] / 10.0
                ns_next[14] = loss_streaks_q[h] / 10.0
                ns_next[15] = float(h) / 3.0
                reward_sum_q[h][action] += reward
                reward_count_q[h][action] += 1

                nfw = build_feat_window(train_num_matrix, ni, Q_LOOKBACK)
                replay_buffers_q[h].append((feat_w, state, action, reward, nfw, ns_next))
                if len(replay_buffers_q[h]) > BUFFER_CAP_Q: replay_buffers_q[h].pop(0)

            # Batch update every 4 bars
            if i % 4 == 0:
                for h in range(4):
                    if len(replay_buffers_q[h]) < BATCH_SIZE_Q: continue
                    q_net.train()
                    batch = random.sample(replay_buffers_q[h], BATCH_SIZE_Q)
                    fw_b  = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                    st_b  = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32)
                    act_b = torch.tensor([b[2] for b in batch], dtype=torch.int64).unsqueeze(1)
                    rew_b = torch.tensor([b[3] for b in batch], dtype=torch.float32).unsqueeze(1)
                    nfw_b = torch.tensor(np.array([b[4] for b in batch]), dtype=torch.float32)
                    nst_b = torch.tensor(np.array([b[5] for b in batch]), dtype=torch.float32)
                    for t in (fw_b, st_b, nfw_b, nst_b, rew_b):
                        torch.nan_to_num_(t, nan=0.0)

                    q_vals_b = q_net(fw_b, st_b, horizon_idx=h).gather(1, act_b)
                    with torch.no_grad():
                        nq = q_target(nfw_b, nst_b, horizon_idx=h)
                        tq = rew_b + 0.99 * nq.max(dim=1, keepdim=True).values
                        torch.nan_to_num_(tq, nan=0.0, posinf=1.0, neginf=-1.0)
                    loss_q_h = torch.nn.MSELoss()(q_vals_b, tq)
                    if torch.isnan(loss_q_h): continue
                    q_opt.zero_grad(); loss_q_h.backward()
                    torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
                    q_opt.step()
                    with torch.no_grad():
                        for tp, p in zip(q_target.parameters(), q_net.parameters()):
                            tp.data.copy_(0.005*p.data + 0.995*tp.data)
                    _q_loss_acc += loss_q_h.item(); _q_steps += 1

        epsilon_q = max(eps_min_q, epsilon_q * eps_decay_q)
        avg_l = _q_loss_acc / max(_q_steps, 1)
        ac_str = " | ".join(
            f"{HORIZON_LABELS_Q[h]} {action_counts_q[h][H_WAIT]}/{action_counts_q[h][H_CALL]}/{action_counts_q[h][H_PUT]}"
            for h in range(4))
        print(f"  {q_epoch+1:>5} | {avg_l:.4e} | {epsilon_q:.3f} | {ac_str}", flush=True)
        for h in range(4):
            c_tot = call_wins_q[h] + call_losses_q[h]
            p_tot = put_wins_q[h]  + put_losses_q[h]
            c_wr  = 100.0*call_wins_q[h]/c_tot  if c_tot > 0 else 0.0
            p_wr  = 100.0*put_wins_q[h] /p_tot  if p_tot > 0 else 0.0
            ar    = {a: (reward_sum_q[h][a]/reward_count_q[h][a] if reward_count_q[h][a]>0 else 0.0)
                    for a in (H_WAIT, H_CALL, H_PUT)}
            print(f"      ↳ {HORIZON_LABELS_Q[h]:>4}: CALL W={call_wins_q[h]}/L={call_losses_q[h]} ({c_wr:.1f}%) | "
                  f"PUT W={put_wins_q[h]}/L={put_losses_q[h]} ({p_wr:.1f}%) | "
                  f"avg WAIT={ar[H_WAIT]:+.4f} CALL={ar[H_CALL]:+.4f} PUT={ar[H_PUT]:+.4f}", flush=True)

        # Save best weights by avg call+put WR across all horizons
        all_trades = sum(call_wins_q[h]+call_losses_q[h]+put_wins_q[h]+put_losses_q[h] for h in range(4))
        all_wins   = sum(call_wins_q[h]+put_wins_q[h] for h in range(4))
        ep_wr = 100.0*all_wins/all_trades if all_trades > 0 else 0.0
        if ep_wr > best_q_wr:
            best_q_wr = ep_wr
            best_q_weights = {k: v.cpu().clone() for k, v in q_net.state_dict().items()}

    print("Per-Horizon Q-Executor Training Complete.")
    if best_q_weights is not None:
        q_net.load_state_dict(best_q_weights)
        print(f"✓ Loaded best Q-net weights (WR={best_q_wr:.1f}%)")

    # Bridge: expose q_net on q_executor so Phase 3 select_action still works
    if hasattr(q_executor, 'policy_net'):
        q_executor.policy_net = q_net
        q_executor.target_net = q_target
        q_executor.num_features = num_features_q

    # Also set build_feat_window for use in Phase 3 _get_h_logits
    # Phase 3 calls q_executor.select_action(state, mask) — we patch it to use
    # the dual-input signature transparently using a closure over train_num_matrix
    _orig_select = q_executor.select_action.__func__ if hasattr(q_executor.select_action, '__func__') else None


    # 3. Evaluate Out-of-Sample Test Window across all 4 Expiry Horizons
    print("\n[Phase 3] Evaluating Out-of-Sample Performance across Expiry Horizons:")
    print(f"{'Expiry Horizon':<18} | {'Trades':<8} | {'Wins':<6} | {'Losses':<8} | {'Waits':<8} | {'Win Rate %':<10} | {'Max Streak [W, L]':<18}")
    print("-" * 92)

    results_table = {}

    for exp_label, lookahead_bars in EXPIRY_HORIZONS.items():
        wins, losses, waits = 0, 0, 0
        cur_w_streak, cur_l_streak = 0, 0
        max_w_streak, max_l_streak = 0, 0
        open_trade_until_idx = -1

        # Learning curve tracking
        trade_outcomes: list[int] = []      # 1=win, 0=loss, in chronological order
        win_streaks: list[int] = []         # all win streak lengths
        loss_streaks: list[int] = []        # all loss streak lengths
        _cur_ws, _cur_ls = 0, 0

        for idx in range(len(test_df) - lookahead_bars):
            row = test_df.iloc[idx]
            expiry_row = test_df.iloc[idx + lookahead_bars]

            cur_price = float(row[close_col])
            expiry_price = float(expiry_row[close_col])

            has_open_position = (idx < open_trade_until_idx)

            if idx == 0 or idx % 15 == 0 or len(zone_manager.get_active_zones()) == 0:
                update_real_snr_snapshot(test_df, idx, zone_manager)
            zone_manager.update_invalidation(cur_price, float(row[high_col]), float(row[low_col]))

            f_dict = _row_to_feature_dict(row, close_col, vol_col)
            pred = meta_learner.predict(f_dict)
            meta_score = float(pred.get("signal_strength", 0.5))

            htf_bias = HTFBiasPackage(
                direction="bullish" if meta_score > 0.5 else "bearish",
                strength=meta_score,
                reversal_prob=float(pred.get("reversal_prob", 0.2)),
                q_value=float(pred.get("q_value", 0.5)),
                expected_mfe_pips=float(pred.get("expected_mfe_pips", 50.0)),
                expected_mae_pips=float(pred.get("expected_mae_pips", 15.0)),
                horizon_strengths=pred.get("horizon_strengths", [0.5, 0.5, 0.5, 0.5]),
                optimal_horizon_idx=int(pred.get("optimal_horizon_idx", 2)),
                recommended_expiry=str(pred.get("recommended_expiry", "30m")),
            )

            account.open_position_type = "CALL" if has_open_position else None
            exec_ctx = _make_exec_ctx(symbol, cur_price, dict(row))
            state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
            mask_engine = HardActionMask()
            action_mask = mask_engine.get_action_mask(
                current_price=cur_price, atr=exec_ctx.atr, zone_manager=zone_manager, buy_volume=exec_ctx.buy_volume, sell_volume=exec_ctx.sell_volume, has_open_position=has_open_position,
            )
            # HardActionMask is authoritative — WAIT when zone/volume says no.
            # Never force-unmask based on HTF bias strength.

            action = q_executor.select_action(state, action_mask, eval_mode=True)

            outcome = None
            if action == 1 and not has_open_position:  # BUY_CALL
                open_trade_until_idx = idx + lookahead_bars
                outcome = 1 if expiry_price > cur_price else 0
            elif action == 2 and not has_open_position:  # BUY_PUT
                open_trade_until_idx = idx + lookahead_bars
                outcome = 1 if expiry_price < cur_price else 0
            else:
                waits += 1

            if outcome is not None:
                trade_outcomes.append(outcome)
                if outcome == 1:
                    wins += 1
                    cur_w_streak += 1; cur_l_streak = 0
                    _cur_ws += 1
                    if _cur_ls > 0:
                        loss_streaks.append(_cur_ls)
                        _cur_ls = 0
                else:
                    losses += 1
                    cur_l_streak += 1; cur_w_streak = 0
                    _cur_ls += 1
                    if _cur_ws > 0:
                        win_streaks.append(_cur_ws)
                        _cur_ws = 0

            max_w_streak = max(max_w_streak, cur_w_streak)
            max_l_streak = max(max_l_streak, cur_l_streak)

        # Flush any open streaks at end
        if _cur_ws > 0: win_streaks.append(_cur_ws)
        if _cur_ls > 0: loss_streaks.append(_cur_ls)

        total_trades = wins + losses
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        # ── Quartile Learning Curve ──────────────────────────────────────────
        q_size = max(1, total_trades // 4)
        quartile_rates = []
        for q in range(4):
            seg = trade_outcomes[q * q_size: (q + 1) * q_size]
            qr = (sum(seg) / len(seg) * 100.0) if seg else 0.0
            quartile_rates.append(round(qr, 1))

        # ── Rolling 20-Trade Win Rate (min/mean/max) ─────────────────────────
        window = 20
        rolling_wrs = []
        for i in range(len(trade_outcomes) - window + 1):
            seg = trade_outcomes[i: i + window]
            rolling_wrs.append(sum(seg) / window * 100.0)
        rolling_min = round(min(rolling_wrs), 1) if rolling_wrs else 0.0
        rolling_mean = round(sum(rolling_wrs) / len(rolling_wrs), 1) if rolling_wrs else 0.0
        rolling_max = round(max(rolling_wrs), 1) if rolling_wrs else 0.0

        # ── Streak Distribution Summary ──────────────────────────────────────
        avg_win_streak = round(sum(win_streaks) / len(win_streaks), 2) if win_streaks else 0.0
        avg_loss_streak = round(sum(loss_streaks) / len(loss_streaks), 2) if loss_streaks else 0.0
        streak_ratio = round(avg_win_streak / avg_loss_streak, 2) if avg_loss_streak > 0 else float("inf")

        # Win rate in last 25% vs first 25% (drift signal)
        late_wr = quartile_rates[3] if len(quartile_rates) == 4 else 0.0
        early_wr = quartile_rates[0] if len(quartile_rates) == 4 else 0.0
        drift = round(late_wr - early_wr, 1)
        drift_str = f"+{drift}%" if drift >= 0 else f"{drift}%"

        print(f"{exp_label:<18} | {total_trades:<8} | {wins:<6} | {losses:<8} | {waits:<8} | {win_rate:<10.2f} | W:{max_w_streak} / L:{max_l_streak}")
        print(f"  {'':>18}   Quartile WR [Q1→Q4]: {quartile_rates}  |  Late vs Early Drift: {drift_str}")
        print(f"  {'':>18}   Rolling-20 WR: min={rolling_min}% avg={rolling_mean}% max={rolling_max}%")
        print(f"  {'':>18}   Avg Streak: W={avg_win_streak} bars / L={avg_loss_streak} bars  (streak ratio {streak_ratio}x)")
        print()

        results_table[exp_label] = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "waits": waits,
            "win_rate_pct": round(win_rate, 2),
            "max_win_streak": max_w_streak,
            "max_loss_streak": max_l_streak,
            "quartile_win_rates": quartile_rates,
            "rolling20_min": rolling_min,
            "rolling20_mean": rolling_mean,
            "rolling20_max": rolling_max,
            "avg_win_streak": avg_win_streak,
            "avg_loss_streak": avg_loss_streak,
            "streak_ratio": streak_ratio,
            "late_vs_early_drift_pct": drift,
        }

    # 4. Collective Multi-Horizon Concurrent Portfolio Simulation
    print("\n[Phase 4] Evaluating Collective Multi-Horizon Concurrent Portfolio Performance:")
    print("Policy: Max 1 active trade per horizon concurrently from shared account.")
    print("-" * 92)

    active_horizon_until = {exp: -1 for exp in EXPIRY_HORIZONS}
    portfolio_outcomes: list[int] = []
    portfolio_win_streaks: list[int] = []
    portfolio_loss_streaks: list[int] = []
    horizon_trade_counts = {exp: {"wins": 0, "losses": 0, "total": 0} for exp in EXPIRY_HORIZONS}

    cur_p_w_streak, cur_p_l_streak = 0, 0
    max_p_w_streak, max_p_l_streak = 0, 0
    _p_cur_ws, _p_cur_ls = 0, 0
    recommended_matches = 0

    max_lookahead = max(EXPIRY_HORIZONS.values())

    for idx in range(len(test_df) - max_lookahead):
        row = test_df.iloc[idx]
        cur_price = float(row[close_col])

        if idx == 0 or idx % 15 == 0 or len(zone_manager.get_active_zones()) == 0:
            update_real_snr_snapshot(test_df, idx, zone_manager)
        zone_manager.update_invalidation(cur_price, float(row[high_col]), float(row[low_col]))

        f_dict = _row_to_feature_dict(row, close_col, vol_col)
        pred = meta_learner.predict(f_dict)
        meta_score = float(pred.get("signal_strength", 0.5))
        rec_expiry = str(pred.get("recommended_expiry", "30m"))
        h_strengths = pred.get("horizon_strengths", [0.5, 0.5, 0.5, 0.5])

        htf_bias = HTFBiasPackage(
            direction="bullish" if meta_score > 0.5 else "bearish",
            strength=meta_score,
            reversal_prob=float(pred.get("reversal_prob", 0.2)),
            q_value=float(pred.get("q_value", 0.5)),
            expected_mfe_pips=float(pred.get("expected_mfe_pips", 50.0)),
            expected_mae_pips=float(pred.get("expected_mae_pips", 15.0)),
            horizon_strengths=h_strengths,
            optimal_horizon_idx=int(pred.get("optimal_horizon_idx", 2)),
            recommended_expiry=rec_expiry,
        )

        exec_ctx = _make_exec_ctx(symbol, cur_price, dict(row))

        # Map recommended expiry to full EXPIRY_HORIZONS key
        rec_expiry_full = HORIZON_LABEL_MAP.get(rec_expiry, None)

        # Check entry per horizon if free — no strength threshold gate here;
        # let the Q-Learner decide WAIT vs CALL/PUT per horizon independently
        for h_idx, (exp_label, lookahead_bars) in enumerate(EXPIRY_HORIZONS.items()):
            if idx < active_horizon_until[exp_label]:
                continue  # Horizon slot is occupied by active trade

            expiry_row = test_df.iloc[idx + lookahead_bars]
            expiry_price = float(expiry_row[close_col])

            account.open_position_type = None  # Free for this horizon
            state = q_executor.build_state_vector(htf_bias, account, exec_ctx, zone_manager)
            mask_engine = HardActionMask()
            action_mask = mask_engine.get_action_mask(
                current_price=cur_price, atr=exec_ctx.atr, zone_manager=zone_manager, buy_volume=exec_ctx.buy_volume, sell_volume=exec_ctx.sell_volume, has_open_position=False,
            )
            # HardActionMask is authoritative — WAIT is a valid outcome.
            # Do NOT override the mask based on HTF bias here; that defeats
            # the no-chase discipline and inflates trade counts artificially.

            action = q_executor.select_action(state, action_mask, eval_mode=True)

            outcome = None
            if action == 1:  # BUY_CALL
                outcome = 1 if expiry_price > cur_price else 0
            elif action == 2:  # BUY_PUT
                outcome = 1 if expiry_price < cur_price else 0

            if outcome is not None:
                active_horizon_until[exp_label] = idx + lookahead_bars
                horizon_trade_counts[exp_label]["total"] += 1
                if exp_label == rec_expiry_full:
                    recommended_matches += 1

                portfolio_outcomes.append(outcome)
                if outcome == 1:
                    horizon_trade_counts[exp_label]["wins"] += 1
                    cur_p_w_streak += 1; cur_p_l_streak = 0
                    _p_cur_ws += 1
                    if _p_cur_ls > 0:
                        portfolio_loss_streaks.append(_p_cur_ls)
                        _p_cur_ls = 0
                else:
                    horizon_trade_counts[exp_label]["losses"] += 1
                    cur_p_l_streak += 1; cur_p_w_streak = 0
                    _p_cur_ls += 1
                    if _p_cur_ws > 0:
                        portfolio_win_streaks.append(_p_cur_ws)
                        _p_cur_ws = 0

                max_p_w_streak = max(max_p_w_streak, cur_p_w_streak)
                max_p_l_streak = max(max_p_l_streak, cur_p_l_streak)

    if _p_cur_ws > 0: portfolio_win_streaks.append(_p_cur_ws)
    if _p_cur_ls > 0: portfolio_loss_streaks.append(_p_cur_ls)

    total_p_trades = len(portfolio_outcomes)
    p_wins = sum(portfolio_outcomes)
    p_losses = total_p_trades - p_wins
    p_win_rate = (p_wins / total_p_trades * 100.0) if total_p_trades > 0 else 0.0

    pq_size = max(1, total_p_trades // 4)
    p_quartiles = []
    for q in range(4):
        seg = portfolio_outcomes[q * pq_size: (q + 1) * pq_size]
        qr = (sum(seg) / len(seg) * 100.0) if seg else 0.0
        p_quartiles.append(round(qr, 1))

    p_late_wr = p_quartiles[3] if len(p_quartiles) == 4 else 0.0
    p_early_wr = p_quartiles[0] if len(p_quartiles) == 4 else 0.0
    p_drift = round(p_late_wr - p_early_wr, 1)
    p_drift_str = f"+{p_drift}%" if p_drift >= 0 else f"{p_drift}%"

    p_avg_ws = round(sum(portfolio_win_streaks) / len(portfolio_win_streaks), 2) if portfolio_win_streaks else 0.0
    p_avg_ls = round(sum(portfolio_loss_streaks) / len(portfolio_loss_streaks), 2) if portfolio_loss_streaks else 0.0
    p_streak_ratio = round(p_avg_ws / p_avg_ls, 2) if p_avg_ls > 0 else float("inf")

    rec_alignment_pct = round(recommended_matches / total_p_trades * 100.0, 1) if total_p_trades > 0 else 0.0

    print(f"COLLECTIVE PORTFOLIO  | Trades: {total_p_trades:<5} | Wins: {p_wins:<5} | Losses: {p_losses:<5} | Win Rate: {p_win_rate:.2f}% | Max Streaks [W:{max_p_w_streak}, L:{max_p_l_streak}]")
    print(f"  Quartile WR [Q1→Q4]: {p_quartiles}  |  Late vs Early Drift: {p_drift_str}")
    print(f"  Avg Streaks: W={p_avg_ws} / L={p_avg_ls}  (Streak Ratio: {p_streak_ratio}x)")
    print(f"  Recommended Expiry Alignment Rate: {rec_alignment_pct}% ({recommended_matches}/{total_p_trades})")
    print("  Per-Horizon Portfolio Contribution:")
    for exp_label, counts in horizon_trade_counts.items():
        h_tot = counts["total"]
        h_wr = (counts["wins"] / h_tot * 100.0) if h_tot > 0 else 0.0
        print(f"    - {exp_label:<5}: {h_tot:<5} trades | Win Rate: {h_wr:.2f}% (Wins: {counts['wins']}, Losses: {counts['losses']})")

    results_table["COLLECTIVE_PORTFOLIO"] = {
        "total_trades": total_p_trades,
        "wins": p_wins,
        "losses": p_losses,
        "win_rate_pct": round(p_win_rate, 2),
        "max_win_streak": max_p_w_streak,
        "max_loss_streak": max_l_streak,
        "quartile_win_rates": p_quartiles,
        "late_vs_early_drift_pct": p_drift,
        "avg_win_streak": p_avg_ws,
        "avg_loss_streak": p_avg_ls,
        "streak_ratio": p_streak_ratio,
        "recommended_expiry_alignment_pct": rec_alignment_pct,
        "per_horizon_breakdown": horizon_trade_counts,
    }

    print("\n==================================================================================\n")
    return results_table



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Option Expiries Benchmark")
    parser.add_argument("--limit", type=int, default=40000, help="Candle limit per timeframe")
    parser.add_argument("--epochs", type=int, default=None, help="Default number of training epochs for Meta-Learner and Q-Learner")
    parser.add_argument("--meta-epochs", type=int, default=None, help="Number of epochs specifically for Phase 1 Meta-Learner training")
    parser.add_argument("--q-epochs", type=int, default=None, help="Number of epochs specifically for Phase 2 Q-Learner training")
    parser.add_argument("--framework", type=str, choices=["keras", "pytorch"], default="pytorch", help="Model framework (keras or pytorch)")
    parser.add_argument(
        "--symbols", type=str,
        default="GLD,SPY,QQQ,TLT,SLV,GDX,USO,EEM,XLF,XLE",
        help=(
            "Comma-separated list of equity/ETF symbols priced in USD (left-of-dollar). "
            "BTC/USD is excluded — Alpaca does not offer options on crypto. "
            "All symbols here are USD-denominated equities/ETFs with active Alpaca options chains, "
            "so the DXY synthetic inversion signal is directionally consistent."
        )
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.replace(" ", ",").split(",") if s.strip()]

    all_results = {}
    for sym in symbols:
        print(f"\n{'='*80}")
        print(f"  SYMBOL: {sym} | Framework: {args.framework.upper()} | Limit: {args.limit}")
        print(f"{'='*80}")
        try:
            result = evaluate_expiries_for_symbol(
                sym,
                limit=args.limit,
                framework=args.framework,
                epochs=args.epochs,
                meta_epochs=args.meta_epochs,
                q_epochs=args.q_epochs,
            )
            all_results[sym] = result
        except Exception as exc:
            logger.error("[%s] Evaluation failed: %s", sym, exc, exc_info=True)
            all_results[sym] = {"error": str(exc)}

    # ── Cross-symbol summary ───────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  CROSS-SYMBOL EDGE SUMMARY")
    print("="*80)
    print(f"{'Symbol':<8} {'Trades':>7} {'Win Rate':>10} {'Best Horizon':<20} {'Alignment':>10}")
    print("-"*60)
    for sym, res in all_results.items():
        if "error" in res:
            print(f"{sym:<8}  ERROR: {res['error'][:50]}")
            continue
        port = res.get("COLLECTIVE_PORTFOLIO", {})
        best_h, best_wr = "N/A", 0.0
        for k, v in res.items():
            if k == "COLLECTIVE_PORTFOLIO":
                continue
            wr = v.get("win_rate_pct", 0.0)
            if wr > best_wr:
                best_wr, best_h = wr, k
        print(
            f"{sym:<8} {port.get('total_trades', 0):>7} "
            f"{port.get('win_rate_pct', 0.0):>9.2f}% "
            f"{best_h:<20} "
            f"{port.get('recommended_expiry_alignment_pct', 0.0):>9.1f}%"
        )
    print("="*80 + "\n")
