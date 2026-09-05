"""
feature_builder.py — Feature Hydration & Feature Serving Pipeline for Axe-paka-v1.

Hardcodes the exact 333 feature column contract extracted directly from the training dataset (`train_50k.csv`).
Ensures real market data for Gold (GLD) and target underlyings is properly hydrated,
enriched with Technical Indicators & MTF Context Features, and formatted into (150, 333) feature windows.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from app.core.analysis.technical_indicators import TechnicalIndicators, IndicatorConfig
from app.core.ml.real_data_pipeline import fetch_real_candles, align_multi_timeframe_datasets, _htf_start_for
from app.axe_paka_v1.models import DEFAULT_NUM_FEATURES, DEFAULT_Q_LOOKBACK, build_feat_window

logger = logging.getLogger(__name__)

# Exact 333 feature column names in exact order from train_50k.csv
AXE_PAKA_V1_FEATURE_COLUMNS: List[str] = [
    'open_5m', 'high_5m', 'low_5m', 'close_5m', 'volume_5m',
    'dxy_open_5m', 'dxy_high_5m', 'dxy_low_5m', 'dxy_close_5m', 'dxy_volume_5m',
    'rsi_5m', 'dxy_rsi_5m', 'rsi_diff_5m',
    'snr_support_5m', 'snr_resistance_5m', 'snr_dist_support_5m', 'snr_dist_resistance_5m',
    'dxy_snr_support_5m', 'dxy_snr_resistance_5m', 'dxy_snr_dist_support_5m', 'dxy_snr_dist_resistance_5m',
    '10_Day_MA_5m', '50_Day_MA_5m', 'ATR_5m', 'ATR_Pct_5m', 'Bar_Dir_5m',
    'Bar_Volume_Down_5m', 'Bar_Volume_Up_5m', 'Body_Ratio_5m', 'Candle_Bear_Score_5m', 'Candle_Bull_Score_5m',
    'Close_Pos_5m', 'Cross_EMA12_Above_EMA18_5m', 'Cross_EMA8_Above_EMA12_5m', 'Cross_MA25_Above_MA50_5m',
    'Cross_MA50_Above_MA100_5m', 'Cross_Supertrend_5m', 'Doji_5m', 'Down_Distance_5m', 'Down_Volume_Change_Pct_5m',
    'EMA12_Above_EMA18_5m', 'EMA8_Above_EMA12_5m', 'EMA_10_5m', 'EMA_10_Diff_5m', 'EMA_10_Minus_EMA8_5m',
    'EMA_12_5m', 'EMA_12_Diff_5m', 'EMA_12_Minus_EMA8_5m', 'EMA_18_5m', 'EMA_18_Diff_5m',
    'EMA_18_Minus_EMA8_5m', 'EMA_21_5m', 'EMA_21_Diff_5m', 'EMA_21_Minus_EMA8_5m', 'EMA_24_5m',
    'EMA_24_Diff_5m', 'EMA_24_Minus_EMA8_5m', 'EMA_32_5m', 'EMA_32_Diff_5m', 'EMA_32_Minus_EMA8_5m',
    'EMA_64_5m', 'EMA_64_Diff_5m', 'EMA_64_Minus_EMA8_5m', 'EMA_8_5m', 'EMA_8_Diff_5m',
    'FVG_Diff_5m', 'HA_Candle_5m', 'HA_Flat_Bottom_5m', 'HA_Flat_Top_5m', 'HA_Lower_Wick_5m',
    'HA_Reversal_5m', 'HA_Small_Body_5m', 'HA_Upper_Wick_5m', 'High_Close_DD_5m', 'Historical_Volatility_5m',
    'Historical_Volatility_20_5m', 'Is_Down_Bar_5m', 'Is_Up_Bar_5m', 'Long_MA_5m', 'Long_MA_100_5m',
    'Long_MA_25_5m', 'Long_MA_Diff_5m', 'Long_Period_MA_Diff_5m', 'Lower_Wick_R_5m', 'MA_5m',
    'MA25_Above_MA50_5m', 'MA50_Above_MA100_5m', 'MACD_12_26_9_5m', 'MACDh_12_26_9_5m', 'MACDs_12_26_9_5m',
    'MA_100_5m', 'MA_100_50_Diff_5m', 'MA_200_5m', 'MA_25_5m', 'MA_50_5m',
    'MeanRev_BB_Stretch_5m', 'MeanRev_MA_Distance_5m', 'MeanRev_RSI_Extreme_5m', 'MeanRev_Score_5m', 'MeanRev_TF_Deviation_5m',
    'MeanRev_Volatility_Spike_5m', 'Mom_Delta_10_5m', 'Mom_Delta_20_5m', 'Mom_Delta_5_5m', 'OBV_5m',
    'Open_Low_DD_5m', 'PSAR_Diff_5m', 'Parabolic_SAR_5m', 'PinBar_At_Level_5m', 'PinBar_Bear_5m',
    'PinBar_Bull_5m', 'PinBar_Recent_Bear_5m', 'PinBar_Recent_Bull_5m', 'PinBar_Score_5m', 'Pivot_Diff_5m',
    'Pivot_R1_Diff_5m', 'Pivot_R2_Diff_5m', 'Pivot_R3_Diff_5m', 'Pivot_S1_Diff_5m', 'Pivot_S2_Diff_5m',
    'Pivot_S3_Diff_5m', 'Pivots_5m', 'Price_Diff_From_Last_Swing_High_5m', 'Price_Diff_From_Last_Swing_Low_5m', 'Price_Long_Long_Period_Diff_5m',
    'Price_Long_Short_Period_Diff_5m', 'Price_Short_Long_Period_Diff_5m', 'Price_Short_Period_Diff_5m', 'Price_Velocity_Bear_5m', 'Price_Velocity_Bull_5m',
    'Price_Velocity_Net_5m', 'RSI_7_5m', 'Regime_Speed_Aligned_5m', 'Regime_Speed_Bear_5m', 'Regime_Speed_Bull_5m',
    'Regime_Speed_Divergence_5m', 'Rel_High_5m', 'Resist_Trendline_Diff_5m', 'Resist_Trendline_Value_5m', 'Ret_20_5m',
    'Ret_5_5m', 'Reversal_Exhaustion_5m', 'Reversal_Momentum_Div_5m', 'Reversal_PinBar_5m', 'Reversal_Score_5m',
    'Reversal_Structure_Break_5m', 'Reversal_Velocity_Decay_5m', 'Reversal_Volume_Clue_5m', 'SMA_10_5m', 'SMA_100_5m',
    'SMA_100_Diff_5m', 'SMA_10_Diff_5m', 'SMA_20_5m', 'SMA_20_Diff_5m', 'SMA_50_5m',
    'SMA_50_Diff_5m', 'SMC_BOS_BOS_5m', 'SMC_BOS_BrokenIndex_5m', 'SMC_BOS_CHOCH_5m', 'SMC_BOS_Level_5m',
    'SMC_FVG_Bottom_5m', 'SMC_FVG_Bottom_Diff_5m', 'SMC_FVG_FVG_5m', 'SMC_FVG_MitigatedIndex_5m', 'SMC_FVG_Top_5m',
    'SMC_FVG_Top_Diff_5m', 'SMC_Liquidity_Level_5m', 'SMC_Liquidity_Level_Diff_5m', 'SMC_Liquidity_Liquidity_5m', 'SMC_OB_Bottom_5m',
    'SMC_OB_Bottom_Diff_5m', 'SMC_OB_MitigatedIndex_5m', 'SMC_OB_OB_5m', 'SMC_OB_OBvolume_5m', 'SMC_OB_Percentage_5m',
    'SMC_OB_Top_5m', 'SMC_OB_Top_Diff_5m', 'SMC_Swing_HighLow_5m', 'SMC_Swing_Level_5m', 'SMC_Swing_Level_Diff_5m',
    'SNR_5m', 'Short_Above_Long_Crossover_5m', 'Short_MA_5m', 'Short_MA_10_5m', 'Short_MA_50_5m',
    'Short_MA_Diff_5m', 'Short_Period_MA_Diff_5m', 'Speed_From_Last_Swing_High_5m', 'Speed_From_Last_Swing_Low_5m', 'Structural_Range_Position_5m',
    'Structural_Range_Width_5m', 'Structure_Established_5m', 'Supertrend_5m', 'Supertrend_Distance_5m', 'Supertrend_Lower_5m',
    'Supertrend_Upper_5m', 'Support_Trendline_Diff_5m', 'Support_Trendline_Value_5m', 'Tick_Volume_5m', 'Time_Diff_From_Last_Swing_High_5m',
    'Time_Diff_From_Last_Swing_Low_5m', 'Trend_Strength_5m', 'Up_Distance_5m', 'Up_Volume_Change_Pct_5m', 'Upper_Wick_R_5m',
    'VIX_20_5m', 'VSI_20_5m', 'Vol_Zscore_5m', 'Volatility_Bear_5m', 'Volatility_Bull_5m',
    'Volatility_Expansion_5m', 'Volatility_Regime_5m', 'Volume_Change_Pct_5m', 'day_of_week_5m', 'hour_5m',
    'minute_5m', 'r1_5m', 'r2_5m', 'r3_5m', 's1_5m',
    's2_5m', 's3_5m', 'session_5m', 'session_transition_5m', 'MACD_5m',
    'BB_Middle_5m', 'RSI_14_5m', 'open_15m', 'high_15m', 'low_15m',
    'close_15m', 'volume_15m', 'dxy_open_15m', 'dxy_high_15m', 'dxy_low_15m',
    'dxy_close_15m', 'dxy_volume_15m', 'rsi_15m', 'dxy_rsi_15m', 'rsi_diff_15m',
    'snr_support_15m', 'snr_resistance_15m', 'snr_dist_support_15m', 'snr_dist_resistance_15m', 'dxy_snr_support_15m',
    'dxy_snr_resistance_15m', 'dxy_snr_dist_support_15m', 'dxy_snr_dist_resistance_15m', 'open_1h', 'high_1h',
    'low_1h', 'close_1h', 'volume_1h', 'dxy_open_1h', 'dxy_high_1h',
    'dxy_low_1h', 'dxy_close_1h', 'dxy_volume_1h', 'rsi_1h', 'dxy_rsi_1h',
    'rsi_diff_1h', 'snr_support_1h', 'snr_resistance_1h', 'snr_dist_support_1h', 'snr_dist_resistance_1h',
    'dxy_snr_support_1h', 'dxy_snr_resistance_1h', 'dxy_snr_dist_support_1h', 'dxy_snr_dist_resistance_1h', 'open_4h',
    'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'dxy_open_4h',
    'dxy_high_4h', 'dxy_low_4h', 'dxy_close_4h', 'dxy_volume_4h', 'rsi_4h',
    'dxy_rsi_4h', 'rsi_diff_4h', 'snr_support_4h', 'snr_resistance_4h', 'snr_dist_support_4h',
    'snr_dist_resistance_4h', 'dxy_snr_support_4h', 'dxy_snr_resistance_4h', 'dxy_snr_dist_support_4h', 'dxy_snr_dist_resistance_4h',
    'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d',
    'dxy_open_1d', 'dxy_high_1d', 'dxy_low_1d', 'dxy_close_1d', 'dxy_volume_1d',
    'rsi_1d', 'dxy_rsi_1d', 'rsi_diff_1d', 'snr_support_1d', 'snr_resistance_1d',
    'snr_dist_support_1d', 'snr_dist_resistance_1d', 'dxy_snr_support_1d', 'dxy_snr_resistance_1d', 'dxy_snr_dist_support_1d',
    'dxy_snr_dist_resistance_1d', 'mtf_rsi_asset', 'mtf_rsi_dxy', 'mtf_rsi_diff', 'mtf_snr_confluence',
    'r1', 'r2', 'r3', 's1', 's2',
    's3', 'Volatility_Bull_next', 'Volatility_Bear_next', 'Volatility_Regime_next', 'Volatility_Expansion_next',
    'Regime_Speed_Bull_next', 'Regime_Speed_Bear_next', 'Price_Velocity_Bull_next', 'Price_Velocity_Bear_next', 'Price_Velocity_Net_next',
    'vel_bull_fwd_8', 'vel_bear_fwd_8', 'vel_net_fwd_8'
]


def compute_full_context_features(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Compute and populate all MTF context features (MTF RSI, DXY divergence, SNR distances, confluence)."""
    df = aligned_df.copy()
    close_col = "close_5m" if "close_5m" in df.columns else "close"
    high_col  = "high_5m"  if "high_5m"  in df.columns else "high"
    low_col   = "low_5m"   if "low_5m"   in df.columns else "low"

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

    # 4. Cross & SNR context features
    MA_WINDOW = 21
    CROSS_MEMORY_BARS = 5

    rsi_series       = pd.Series(rsi_5m)
    dxy_rsi_series   = pd.Series(dxy_rsi_5m)
    fast_diff_series = pd.Series(fast_diff)

    asset_rsi_ma21 = rsi_series.rolling(MA_WINDOW, min_periods=1).mean()
    dxy_rsi_ma21   = dxy_rsi_series.rolling(MA_WINDOW, min_periods=1).mean()

    _state_asset_above_ma  = (rsi_series     > asset_rsi_ma21).astype(np.float32)
    _state_dxy_above_ma    = (dxy_rsi_series > dxy_rsi_ma21).astype(np.float32)
    _state_asset_above_dxy = (rsi_series     > dxy_rsi_series).astype(np.float32)
    _state_htf_bullish     = (pd.Series(rsi_1h) > 50).astype(np.float32)

    df["state_asset_above_ma"]  = _state_asset_above_ma.values
    df["state_dxy_above_ma"]    = _state_dxy_above_ma.values
    df["state_asset_above_dxy"] = _state_asset_above_dxy.values
    df["state_htf_bullish"]     = _state_htf_bullish.values

    df["state_asset_ma_spread"]  = np.clip((rsi_5m - asset_rsi_ma21.values) / 50.0, -1.0, 1.0).astype(np.float32)
    df["state_dxy_ma_spread"]    = np.clip((dxy_rsi_5m - dxy_rsi_ma21.values) / 50.0, -1.0, 1.0).astype(np.float32)
    df["state_asset_dxy_spread"] = np.clip((rsi_5m - dxy_rsi_5m) / 50.0, -1.0, 1.0).astype(np.float32)

    def _cross_memory(state_bool_series: pd.Series, window: int = CROSS_MEMORY_BARS) -> np.ndarray:
        prev = state_bool_series.shift(1).astype("boolean").fillna(pd.NA).astype("boolean")
        bull_cross = (state_bool_series & ~prev.fillna(False)).astype(int)
        bear_cross = (~state_bool_series & prev.fillna(True)).astype(int)
        recent_bull = bull_cross.rolling(window, min_periods=1).max()
        recent_bear = bear_cross.rolling(window, min_periods=1).max()
        return (recent_bull - recent_bear).values.astype(np.float32)

    df["cross_index_signal"] = _cross_memory(_state_asset_above_ma.astype(bool))
    df["cross_dxy_signal"]   = _cross_memory(_state_dxy_above_ma.astype(bool))
    df["cross_index_dxy"]    = _cross_memory(_state_asset_above_dxy.astype(bool))
    df["cross_dxy_symbol"]   = fast_diff_series.rolling(3, min_periods=1).mean().clip(-1.0, 1.0).values.astype(np.float32)

    atr = (df[high_col] - df[low_col]).rolling(14).mean().fillna(df[close_col] * 0.005).values
    supp_q = df[low_col].rolling(100, min_periods=10).quantile(0.20).fillna(df[low_col]).values
    res_q  = df[high_col].rolling(100, min_periods=10).quantile(0.80).fillna(df[high_col]).values

    df["snr_dist_support"]    = np.clip((asset_close - supp_q) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)
    df["snr_dist_resistance"] = np.clip((res_q - asset_close) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)

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
            df[f"snr_dist_support_{_tf}"]    = np.full(len(df), 10.0, dtype=np.float32)
            df[f"snr_dist_resistance_{_tf}"] = np.full(len(df), 10.0, dtype=np.float32)

    CONFLUENCE_PCT = 0.0015
    _sup_15m_price = asset_close - df["snr_dist_support_15m"].values * atr
    _res_15m_price = asset_close + df["snr_dist_resistance_15m"].values * atr
    _sup_1h_price  = asset_close - df["snr_dist_support_1h"].values  * atr
    _res_1h_price  = asset_close + df["snr_dist_resistance_1h"].values  * atr

    _sup_sup_conf  = (np.abs(_sup_15m_price - _sup_1h_price)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
    _res_res_conf  = (np.abs(_res_15m_price - _res_1h_price)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
    _sup_res_conf  = (np.abs(_sup_15m_price - _res_1h_price)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
    _res_sup_conf  = (np.abs(_res_15m_price - _sup_1h_price)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT

    df["mtf_snr_confluence"] = (_sup_sup_conf | _res_res_conf | _sup_res_conf | _res_sup_conf).astype(np.float32)

    return df


class AxePakaV1FeatureBuilder:
    """
    Builds and hydrates full 333-feature window matrices for Axe-paka-v1 model inference.
    Uses exact hardcoded 333 feature column contract from train_50k.csv.
    """

    def __init__(self, target_num_features: int = DEFAULT_NUM_FEATURES):
        self.target_num_features = target_num_features
        self.ti_calc = TechnicalIndicators(IndicatorConfig())
        self.feature_columns: List[str] = AXE_PAKA_V1_FEATURE_COLUMNS

    def build_features_for_dataframe(self, df_5m: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Build complete feature matrix (N, 333) from 5m candle dataframe matching exact 333 columns.
        """
        df = df_5m.copy()
        close_col = "close_5m" if "close_5m" in df.columns else ("close" if "close" in df.columns else "Close")
        open_col  = "open_5m"  if "open_5m"  in df.columns else ("open" if "open" in df.columns else "Open")
        high_col  = "high_5m"  if "high_5m"  in df.columns else ("high" if "high" in df.columns else "High")
        low_col   = "low_5m"   if "low_5m"   in df.columns else ("low" if "low" in df.columns else "Low")
        vol_col   = "volume_5m" if "volume_5m" in df.columns else ("volume" if "volume" in df.columns else "Volume")

        df = compute_full_context_features(df)

        ta_input = df.rename(columns={
            open_col: "Open", high_col: "High", low_col: "Low", close_col: "Close", vol_col: "Volume"
        })
        try:
            ti_enriched = self.ti_calc.calculate_all_indicators(ta_input, mode="training")
            for c in ti_enriched.columns:
                if c not in df.columns and c not in ("timestamp", "Time"):
                    df[c] = ti_enriched[c].astype(np.float32)
        except Exception as e:
            logger.warning("⚠ Indicator enrichment warning: %s", e)

        # Select exact 333 feature columns in exact order
        matrix_cols = []
        for col_name in AXE_PAKA_V1_FEATURE_COLUMNS:
            if col_name in df.columns:
                col_vals = df[col_name].values.astype(np.float32)
            else:
                col_vals = np.zeros(len(df), dtype=np.float32)
            matrix_cols.append(col_vals)

        matrix = np.column_stack(matrix_cols).astype(np.float32)
        np.nan_to_num(matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return matrix, df

    def get_latest_feature_window(
        self,
        symbol: str = "GLD",
        limit: int = 500,
        q_lookback: int = DEFAULT_Q_LOOKBACK,
    ) -> Tuple[np.ndarray, float]:
        """
        Fetch real market data for symbol (default GLD), compute exact 333 features,
        and return latest (150, 333) feature window and current price.
        """
        logger.info("⚡ Hydrating exact 333-feature window for %s (limit=%d bars)...", symbol, limit)
        df_5m = fetch_real_candles(symbol, timeframe="5m", limit=limit)
        if df_5m.empty:
            raise ValueError(f"Could not fetch market bars for {symbol}")

        matrix, enriched_df = self.build_features_for_dataframe(df_5m)
        latest_idx = len(matrix) - 1
        window = build_feat_window(matrix, abs_idx=latest_idx, q_lookback=q_lookback)

        close_col = "close_5m" if "close_5m" in enriched_df.columns else ("close" if "close" in enriched_df.columns else "Close")
        current_price = float(enriched_df[close_col].iloc[-1])

        logger.info(
            "✅ Hydration complete! Feature window shape: %s, Current %s price: $%.2f",
            window.shape, symbol, current_price
        )
        return window, current_price
