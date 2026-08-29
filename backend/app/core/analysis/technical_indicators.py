"""
TechnicalIndicators module for calculating a comprehensive set of technical indicators on trading data.

"""

# ============================================================================
# ⚠️  DO NOT MODIFY PANDAS API CALLS WITHOUT READING THIS FIRST  ⚠️
# ============================================================================
#
# Several bugs were introduced by AI agents using outdated pandas knowledge.
# The fixes below are INTENTIONAL and CORRECT for pandas 2.x+.
# If you are an AI agent editing this file, preserve all of the following:
#
# 1. DEPRECATED fillna(method=...) — DO NOT USE
#    ❌ WRONG (raises FutureWarning / will raise in future):
#         series.fillna(method="ffill")
#         series.fillna(method="bfill")
#    ✅ CORRECT:
#         series.ffill()
#         series.bfill()
#
# 2. DEPRECATED replace() downcasting — DO NOT USE
#    ❌ WRONG (triggers FutureWarning, silently changes dtype):
#         series.replace([np.inf, -np.inf], 0).fillna(0)
#    ✅ CORRECT (replace inf with NaN first, lock dtype, then fill):
#         series.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).fillna(0)
#
# 3. DEPRECATED frame.insert in loops — DO NOT USE
#    ❌ WRONG (causes PerformanceWarning: DataFrame highly fragmented):
#         for col in col_list:
#             df[col] = np.nan          # calls frame.insert N times
#    ✅ CORRECT (single allocation):
#         missing = [c for c in col_list if c not in df.columns]
#         if missing:
#             df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing)], axis=1)
#
# 4. AMBIGUOUS bool(DataFrame) — DO NOT USE
#    ❌ WRONG (raises ValueError: ambiguous truth value):
#         if self.config:               # self.config may be a DataFrame
#         if not some_df_variable:
#    ✅ CORRECT:
#         if isinstance(self.config, dict) and self.config:
#         if some_df_variable is not None and not some_df_variable.empty:
#
# These warnings were introduced by AI agents with outdated pandas 1.x knowledge.
# This comment block exists to prevent regressions. Last fixed: 2026-08-09.
# ============================================================================

import warnings
import pandas as pd
import numpy as np
from app.core.analysis.trendline_automation import fit_trendlines_high_low
import pandas_ta as ta
from tqdm import tqdm
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Any, Callable, Set

from app.core.analysis.candles import heikinashi, doji
from app.core.analysis.smc import smc
import logging
import gc
from app.core.processing.tasks import TaskStore, TaskCancelledException
from app.core.processing.progress_reporter import ProgressEvent, ProgressReporter, ThrottlingStrategy
from app.core.analysis.pivots_points import pivots_points
logger = logging.getLogger(__name__)

# Sentinel value for diff columns when indicator is unavailable (warm-up period, missing data).
# Using a very large value so the model understands: "price has NO known relationship to this level".
# DO NOT use 0 — that would mean price == indicator (perfect closeness), which is semantically wrong.

LARGE_DIFF_SENTINEL_DEFAULT = 999999.0


# Legacy progress helpers removed in favor of unified ProgressReporter infrastructure.


class IndicatorConfig(BaseModel):
    """Configuration class for technical indicators"""

    # Basic parameters
    bars: int = 50                  # Generous bar count — captures retracements + primary trend
    lookback_window: int = 50       # Extended lookback to see retracements and primary trend context
    pivot_levels: int = 3
    diff_column: str = (
        "Close"  # Column to calculate differences against (e.g., 'Close', 'High', 'Low')
    )

    # Moving averages
    sma_periods: List[int] = [10, 20, 50, 100]
    sma_range_periods: Tuple[int, int] = (
        2,
        14,
    )  # Range for SMA differences (e.g., SMA_diff_2 to SMA_diff_13)
    ema_periods: List[int] = [8, 10, 12, 18, 21, 24, 32, 64]

    # Standard SMA configurations
    sma_short_ma_10: int = 10
    sma_long_ma_25: int = 25
    sma_short_ma_50: int = 50
    sma_long_ma_100: int = 100
    sma_10_day_ma: int = 10
    sma_50_day_ma: int = 25  # Note: This was 25 in original
    sma_20: int = 20

    # Additional SMA mappings
    sma_ma_25: int = 8
    sma_ma_50: int = 50
    sma_ma_100: int = 100
    sma_ma_200: int = 250

    # Short/Long term MAs
    short_ma: int = 8
    long_ma: int = 25
    ma: int = 8

    # RSI parameters
    rsi_periods: List[int] = [7, 14]
    rsi_change_periods: int = 5

    # Bollinger Bands
    bb_length: int = 20
    bb_std: float = 2.0

    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Other indicators
    historical_volatility_length: int = 20
    atr_length: int = 14  # Default ATR period
    psar_af: float = 0.02  # Parabolic SAR acceleration factor
    psar_max_af: float = 0.2  # Parabolic SAR max acceleration

    # Historical price levels
    historical_days: int = (
        3  # Number of historical days to track (Low_Day_X, High_Day_X)
    )
    prev_close_periods: int = 5  # Number of previous close differences to calculate

    # Volume parameters
    enable_volume_analysis: bool = True

    # Pattern analysis
    enable_heikinashi: bool = True
    enable_doji: bool = True
    enable_smc: bool = True
    enable_supertrend: bool = True
    enable_pivots: bool = True

    # Supertrend parameters
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0

    # Peak/Valley Pivot parameters
    pivot_up_thresh: float = 0.01
    pivot_down_thresh: float = -0.01

    # Additional features
    enable_additional_features: bool = True
    momentum_window: int = 30
    
    # Momentum & Trend Features
    enable_momentum_features: bool = True  # MOM_t, MR_t, TF_t
    trend_factor_window: int = 30  # Window for TF_t calculation
    
    # Legacy feature names (compact representation)
    enable_lettered_variables: bool = True  # a, b, c, d, e, f, g, h, i
    enable_ema_change_series: bool = True  # MA_Change0-4 (EMA shifts)
    enable_sma_range_diffs: bool = True  # SMA0-SMA11 (SMA differences for range periods)

    # ── Advanced Market Microstructure Indicators ─────────────────────────────
    # These use only historical lookback data (no future rows).
    snr_window: int = 20                    # Rolling window for SNR & VIX_20 calculation
    vsi_window: int = 20                    # Rolling window for Volume Sentiment Index
    enable_snr_vix: bool = True             # Toggle SNR (dB) + VIX_20 (%) features
    enable_candle_structure: bool = True    # Toggle per-bar candle structure features
    enable_candle_bull_score: bool = True   # Toggle composite candle bull score
    
    # ── Pin Bar / Hammer / Shooting-Star Detection ───────────────────────────
    # A pin bar is valid when: long_wick / candle_range >= pin_wick_ratio_min
    #                    AND:  body / candle_range      <= pin_body_ratio_max
    pin_wick_ratio_min: float = 0.60        # Long wick must be ≥ 60% of total range
    pin_body_ratio_max: float = 0.30        # Body must be ≤ 30% of total range
    pin_nose_ratio_max: float = 0.10        # Opposite (short) wick ≤ 10% of range (classic pin)
    # Recent-bar scoring weights: bar[-1] carries the most signal weight
    pin_weight_bar_minus1: float = 1.00     # Current/last completed bar (heaviest)
    pin_weight_bar_minus2: float = 0.55     # Previous bar
    pin_weight_bar_minus3: float = 0.25     # Two bars ago (lightest)
    # Key-level proximity amplifier: if price is within N×ATR of a key level,
    # the pin bar signal is amplified by this factor (capped so score stays ≤ 1)
    pin_level_proximity_atr: float = 1.5    # ATR multiples = "near a key level"
    pin_level_amplifier: float = 1.40       # Multiply pin score when near a key level

    # ── Regime Transition Scoring Features ────────────────────────────────────
    # Reversal Score parameters
    reversal_lookback: int = 5              # Bars for divergence detection
    reversal_structure_atr_threshold: float = 0.5  # Structure break sensitivity (× ATR)
    
    # Config for Dual-Head label blending (mirrors writing/create_dual_labels)
    dual_head_lookahead: int = 1            # Bars ahead for base return signal
    dual_head_n_future: int = 8             # Regime window length
    dual_head_w_base: float = 0.35         # Weight on binary base return
    dual_head_w_regime: float = 0.65       # Weight on exponentially decayed regime

    # ── Footprint / Volume Profile (tick-derived, optional) ──────────────────
    enable_footprint: bool = False          # Off by default — requires pre-built fp_* columns
    footprint_cum_delta_window: int = 20    # Rolling window for FP_Cum_Delta
    footprint_rejection_lookback: int = 1   # Bars to check for delta divergence

    def max_lookback_period(self) -> int:
        """
        Calculate the maximum lookback period required across all indicators.
        This ensures ProcessingManager overlap is sufficient for all indicator calculations.
        
        Rationale: When slicing data, we need overlap periods large enough so that:
        1. The 250-period MA_200 (sma_ma_200) has full history
        2. All other technical indicators (RSI, MACD, Bollinger Bands, etc.) are valid
        3. SNR signal detection can look back far enough for pattern recognition
        
        Returns:
            int: Maximum lookback period in bars needed for valid calculations
        """
        periods = [
            # Moving averages - need full history for each period
            max(self.sma_periods) if self.sma_periods else 100,
            max(self.ema_periods) if self.ema_periods else 64,
            self.sma_ma_200,  # 250-period SMA (critical limit)
            
            # Oscillators
            max(self.rsi_periods) if self.rsi_periods else 14,
            self.bb_length,
            
            # MACD needs both slow EMA + signal line
            self.macd_slow + self.macd_signal,  # 26 + 9 = 35
            
            # Other indicators
            self.historical_volatility_length,
            self.atr_length,
            self.supertrend_period,
            self.lookback_window,
            self.prev_close_periods,
            self.momentum_window,

            # Advanced microstructure indicators
            self.snr_window,   # SNR / VIX rolling window
            self.vsi_window,   # Volume Sentiment Index window
            20,                # Candle structure uses 20-bar rolling (Rel_High, Vol_Zscore)
            20,                # Ret_20 lookback
            self.footprint_cum_delta_window,   # FP_Cum_Delta rolling window
        ]
        return max(periods)
    
    def calculate_required_overlap(self, snr_lookback_period: int = 200, safety_offset: int = 50) -> int:
        """
        Calculate the minimum overlap required between slices to ensure valid calculations.
        
        Args:
            snr_lookback_period: Lookback period for SNR signal detection
            safety_offset: Additional buffer for edge cases (default 50 bars)
        
        Returns:
            int: Required overlap in bars
            
        Example:
            >>> config = IndicatorConfig()
            >>> required = config.calculate_required_overlap(snr_lookback_period=200, safety_offset=50)
            >>> # Returns: max(200, max_indicator_lookback) + 50
        """
        max_indicator_lookback = self.max_lookback_period()
        return max(snr_lookback_period, max_indicator_lookback) + safety_offset

    def get_output_columns(self) -> Set[str]:
        """
        Return all possible output columns that TechnicalIndicators can generate.
        This is the source of truth for what technical analysis produces.
        
        Used by ProcessingManager's AnalysisColumnContract to validate outputs.
        
        Returns:
            Set of column names that technical analysis will add to the DataFrame
        """
        columns = set()
        
        # Moving Averages
        for period in self.sma_periods:
            columns.add(f"SMA_{period}")
            columns.add(f"SMA_{period}_Diff")
        
        columns.update([
            "Short_MA_10", 
            "Long_MA_25", 
            "Short_MA_50", 
            "Long_MA_100",
            "10_Day_MA", 
            "50_Day_MA", 
            "SMA_20",
            "MA_25", "MA_50", "MA_100", "MA_200",
            "Short_MA", "Long_MA", "MA",
            "Short_Period_MA_Diff", "Long_Period_MA_Diff",
            "Price_Short_Period_Diff", "Price_Short_Long_Period_Diff",
            "Price_Long_Short_Period_Diff", "Price_Long_Long_Period_Diff",
        ])
        
        # EMA — driven by config.ema_periods, same as the actual calculation
        base_ema = min(self.ema_periods) if self.ema_periods else 8
        for period in self.ema_periods:
            columns.add(f"EMA_{period}")
            columns.add(f"EMA_{period}_Diff")
            if period != base_ema:
                columns.add(f"EMA_{period}_Minus_EMA{base_ema}")
        columns.add("MA_100_50_Diff")
        columns.update(["Short_MA", "Long_MA", "MA", "Short_MA_Diff", "Long_MA_Diff"])

        # MA relative-position flags and Cross_ event columns
        # Cross_* = change from prev bar (purely lookback — NOT future-confirmed).
        # Named Cross_ to distinguish from Signal_bounce/breakout_* (signal_generator,
        # forward-confirmed targets) which must be excluded from input features.
        columns.update([
            "Short_Above_Long_Crossover",
            "EMA8_Above_EMA12", "EMA12_Above_EMA18",
            "MA25_Above_MA50", "MA50_Above_MA100",
            "Cross_EMA8_Above_EMA12",   # +1 when EMA8 just crossed above EMA12, -1 below, 0 flat
            "Cross_EMA12_Above_EMA18",
            "Cross_MA25_Above_MA50",
            "Cross_MA50_Above_MA100",
        ])
        
        # RSI
        for period in self.rsi_periods:
            columns.add(f"RSI_{period}")
        
        # Bollinger Bands
        columns.update([
            "BB_Upper", "BB_Middle", "BB_Lower", "BB_UpperBand", "BB_MiddleBand", "BB_LowerBand",
            "BB_Upper_Diff", "BB_Lower_Diff", "BB_Mid_Diff", "BB_Squeeze",
        ])

        # MACD
        columns.update(["MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9", "MACD", "MACD_Signal", "MACD_Histogram"])

        # Supertrend
        if self.enable_supertrend:
            columns.update([
                "Supertrend_Upper", "Supertrend_Lower", "Supertrend",
                "Supertrend_Distance",  # Consolidated distance to active band
                "Cross_Supertrend",
            ])

        # ATR & Volatility
        hv_col = f"Historical_Volatility_{self.historical_volatility_length}"
        columns.update([
            "ATR", "ATR_Pct",
            "Historical_Volatility", hv_col,
            "Parabolic_SAR", "PSAR_Diff",
        ])
        
        # Volume
        if self.enable_volume_analysis:
            columns.update([
                "Tick_Volume", "OBV", "Is_Up_Bar", "Is_Down_Bar",
                "Bar_Volume_Up", "Bar_Volume_Down", "Up_Distance", "Down_Distance",
                "Volume_Change_Pct", "Up_Volume_Change_Pct", "Down_Volume_Change_Pct",
                "Price_Diff_From_Last_Swing_Low", "Price_Diff_From_Last_Swing_High",
                "Time_Diff_From_Last_Swing_Low", "Time_Diff_From_Last_Swing_High",
                "Speed_From_Last_Swing_Low", "Speed_From_Last_Swing_High",
            ])
        
        # Heikin-Ashi
        if self.enable_heikinashi:
            columns.update([
                "HA_Flat_Bottom", "HA_Flat_Top", "HA_Small_Body",
                "HA_Candle", "HA_Reversal", "HA_Lower_Wick", "HA_Upper_Wick"
            ])
        
        # Doji
        if self.enable_doji:
            columns.update(["Doji", "Doji_Type"])
        
        # SMC
        if self.enable_smc:
            columns.update([
                "SMC_Swing_HighLow", "SMC_Swing_Level", "SMC_Swing_Level_Diff",
                "SMC_FVG_FVG", "SMC_FVG_Top", "SMC_FVG_Bottom", "SMC_FVG_MitigatedIndex",
                "SMC_FVG_Top_Diff", "SMC_FVG_Bottom_Diff",
                "SMC_OB_OB", "SMC_OB_Top", "SMC_OB_Bottom",
                "SMC_OB_OBvolume", "SMC_OB_Percentage", "SMC_OB_MitigatedIndex",
                "SMC_OB_Top_Diff", "SMC_OB_Bottom_Diff",
                "SMC_BOS_BOS", "SMC_BOS_CHOCH", "SMC_BOS_Level", "SMC_BOS_BrokenIndex",
                "SMC_Liquidity_Liquidity", "SMC_Liquidity_Level", "SMC_Liquidity_Level_Diff",
                "FVG_Diff",
            ])
        
        # Pivots
        if self.enable_pivots:
            columns.add("Pivots")
            for i in range(1, self.pivot_levels + 1):
                columns.add(f"r{i}")
                columns.add(f"s{i}")
            columns.add("Pivot_Diff")
            for level in range(1, self.pivot_levels + 1):
                columns.add(f"Pivot_R{level}_Diff")
                columns.add(f"Pivot_S{level}_Diff")
        
        # Trendlines
        columns.update([
            "Support_Trendline_Value", "Resist_Trendline_Value",
            "Support_Trendline_Diff", "Resist_Trendline_Diff"
        ])
        
        # Structural Range
        columns.update([
            "Structural_Range_Position", "Structural_Range_Width",
            "Structure_Established", "Peak_Freshness", "Valley_Freshness"
        ])
        
        # Session & Time Features
        columns.update([
            "session", "session_transition",
            "day_of_week", "hour", "minute"
        ])
        
        # Utility column added by technical analysis
        columns.add("time_index")

        # ── Advanced Microstructure Indicators ───────────────────────────────
        if self.enable_snr_vix:
            columns.update(["SNR", "VIX_20"])

        # Trend Strength (always computed — uses trend_factor_window)
        columns.add("Trend_Strength")

        # Volume Sentiment Index
        if self.enable_volume_analysis:
            columns.add("VSI_20")

        # Per-bar candle structure features (port of candle_features() from Dual-Head v6)
        if self.enable_candle_structure:
            columns.update([
                "Bar_Dir", "Body_Ratio", "Open_Low_DD", "High_Close_DD",
                "Close_Pos", "Upper_Wick_R", "Lower_Wick_R",
                "Ret_5", "Ret_20", "Vol_Zscore", "Rel_High",
            ])

        # Composite candle bull conviction score
        if self.enable_candle_bull_score:
            columns.add("Candle_Bull_Score")
            # Independent bear score (NOT the complement of bull score)
            columns.add("Candle_Bear_Score")

        # ── Footprint / Volume Profile (tick-derived) ────────────────────────
        if self.enable_footprint:
            columns.update([
                "FP_POC_Diff", "FP_VAH_Diff", "FP_VAL_Diff",
                "FP_Delta", "FP_Cum_Delta", "FP_Delta_Divergence",
                "FP_Imbalance_Max", "FP_High_Vol_Rejection", "FP_Data_Available",
            ])

        # ── Price Velocity (directional pips-per-second, bull & bear independent) ──
        columns.update([
            "Price_Velocity_Bull",   # upward pip rate, ATR-normalised [0, 3]
            "Price_Velocity_Bear",   # downward pip rate, ATR-normalised [0, 3]
            "Price_Velocity_Net",    # signed balance [-1, 1]
        ])

        # ── Volatility Regime ────────────────────────────────────────────────
        columns.update([
            "Volatility_Regime",     # current vol vs 95th-pct history [0, 1]
            "Volatility_Expansion",  # short-vol / long-vol (expanding=1, contracting=0) [0, 1]
            "Volatility_Bull",       # fraction of vol that was upward [0, 1]
            "Volatility_Bear",       # fraction of vol that was downward [0, 1]
        ])

        # ── Regime Speed (trend-gated directional advance rate) ─────────────
        columns.update([
            "Regime_Speed_Bull",        # bull advance speed × trend quality [0, 1]
            "Regime_Speed_Bear",        # bear advance speed × trend quality [0, 1]
            "Regime_Speed_Aligned",     # on-trend direction speed [0, 1]
            "Regime_Speed_Divergence",  # bull−bear speed balance [-1, 1]
        ])

        # ── Reversal Score (trend death probability) ─────────────────────────
        columns.update([
            "Reversal_Score",              # composite reversal probability [0, 1]
            "Reversal_Momentum_Div",       # momentum divergence component [0, 1]
            "Reversal_Volume_Clue",        # volume anomaly component [0, 1]
            "Reversal_Structure_Break",    # structural break component [0, 1]
            "Reversal_Exhaustion",         # RSI exhaustion component [0, 1]
            "Reversal_Velocity_Decay",     # velocity decay component [0, 1]
            "Reversal_PinBar",             # pin bar / hammer reversal component [0, 1]
        ])

        # ── Mean Reversion Score (snap-back probability) ─────────────────────
        columns.update([
            "MeanRev_Score",               # composite mean reversion probability [0, 1]
            "MeanRev_BB_Stretch",          # Bollinger Band stretch component [0, 1]
            "MeanRev_RSI_Extreme",         # RSI extreme component [0, 1]
            "MeanRev_Volatility_Spike",    # volatility spike component [0, 1]
            "MeanRev_MA_Distance",         # MA distance component [0, 1]
            "MeanRev_TF_Deviation",        # trend factor deviation component [0, 1]
        ])

        # ── Momentum Delta (strength comparison) ─────────────────────────────
        columns.update([
            "Mom_Delta_5",                 # momentum change vs 5 bars ago [-2, 2]
            "Mom_Delta_10",                # momentum change vs 10 bars ago [-2, 2]
            "Mom_Delta_20",                # momentum change vs 20 bars ago [-2, 2]
        ])

        # ── Pin Bar / Hammer / Shooting-Star Detection ───────────────────────
        columns.update([
            "PinBar_Bull",          # Bullish pin bar (hammer): lower wick dominates [0, 1]
            "PinBar_Bear",          # Bearish pin bar (shooting star): upper wick dominates [0, 1]
            "PinBar_Score",         # Signed net pin score: +1 = strong bull pin, -1 = strong bear pin
            "PinBar_Recent_Bull",   # Weighted 3-bar lookback bull pin influence [0, 1]
            "PinBar_Recent_Bear",   # Weighted 3-bar lookback bear pin influence [0, 1]
            "PinBar_At_Level",      # 1 if pin bar coincides with a key structural level
        ])

        return columns


class TechnicalIndicators:
    """Modular techAPInical indicators calculator"""

    def __init__(self, config: IndicatorConfig = None):
        # 🛡️ DEFENSIVE: Ensure config is a valid object with required fields
        if config is None:
            self.config = IndicatorConfig()
        else:
            self.config = config
            # Ensure critical list fields are initialized (prevents 'NoneType' has no len() crashes)
            for attr in ['sma_periods', 'ema_periods', 'rsi_periods']:
                val = getattr(self.config, attr) if hasattr(self.config, attr) else None
                if val is None:
                    default_val = getattr(IndicatorConfig(), attr)
                    try:
                        setattr(self.config, attr, default_val)
                        logger.warning(f"🛡️ [TI] Fixed null '{attr}' in config by restoring defaults")
                    except Exception as e:
                        logger.error(f"❌ [TI] Failed to restore default for '{attr}': {e}")
        self._calculation_mode = "training"  # Default mode
        self.dynamic_diff_sentinel = LARGE_DIFF_SENTINEL_DEFAULT
        self._validate_config()
        self.logger = logger

    def _validate_config(self):
        """Validate configuration parameters"""
        valid_diff_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "open",
            "high",
            "low",
            "close",
        ]
        if self.config.diff_column not in valid_diff_columns:
            raise ValueError(
                f"diff_column must be one of {valid_diff_columns}, got {self.config.diff_column}"
            )

    def _clean_inf(self, s: Any, fill_value: Optional[float] = 0.0) -> Any:
        """Safely clean inf/-inf values without triggering pandas 2.x downcasting warnings."""
        if s is None:
            return s
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s = pd.to_numeric(s, errors="coerce")
        cleaned = s.mask(np.isinf(s), np.nan)
        if fill_value is not None:
            return cleaned.fillna(fill_value)
        return cleaned

    def calculate_all_indicators(
        self, 
        data: pd.DataFrame, 
        helper_class=None, 
        task_id=None, 
        progress_store=None, 
        calculate_last_n: Optional[int] = None, 
        mode: str = "training", 
        slice_context: Optional[Dict] = None,
        reporter: Optional[ProgressReporter] = None
    ) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators for trading data, with slice-aware progress.
        """
        self._calculation_mode = mode
        
        # Initialize or use the provided unified ProgressReporter
        if not reporter:
            reporter = ProgressReporter(task_id, progress_store, slice_context)
            # Ensure task_id is set for report() calls
            if task_id: reporter.task_id = task_id
        
        # Store total rows for progress reporting
        # 🛡️ DEFENSIVE: Guard against None data
        if data is None:
            logger.error("❌ [TI] calculate_all_indicators received None data")
            raise ValueError("Input data cannot be None")

        n_rows = len(data)
        if n_rows == 0:
            logger.warning("⚠️ [TI] calculate_all_indicators received empty DataFrame")
            return data.copy()
        
        # 0. Duplicate-column guard — must run BEFORE _prepare_data so that any
        #    downstream arithmetic (Series - Series, pd.DataFrame constructor, etc.)
        #    never encounters ambiguous duplicate labels.  This can happen when the
        #    input comes from the currency_indices step which may carry prefixed TI
        #    columns added on top of the base TA output; if that merge introduced any
        #    duplicate names they must be collapsed here.
        if data.columns.duplicated().any():
            dup_names = data.columns[data.columns.duplicated(keep=False)].unique().tolist()
            logger.warning(
                "⚠️ [TI] Input DataFrame has %d duplicate column name(s) — "
                "deduplicating before calculation (keep first): %s",
                len(dup_names), dup_names[:20],
            )
            data = data.loc[:, ~data.columns.duplicated(keep="first")].copy()
            n_rows = len(data)  # re-read in case shape changed (shouldn't, but be safe)

        # 1. Preparation & Cleanup
        reporter.report(
            progress=10,
            message="Data Preparation",
            message2=f"Standardizing {n_rows:,} bars and validating OHLCV columns",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Data_Prep"
        )
        try:
            df = self._prepare_data(data.copy())
        except Exception as e:
            logger.error(f"❌ Critical error in _prepare_data: {e}", exc_info=True)
            return data.copy() # Return copy of original if preparation fails
        
        # Check cancellation after major data copy
        reporter.check_cancellation()
        
        # 2. Basic Features
        reporter.report(
            progress=15,
            message="Basic Features",
            message2=f"Calculating price changes, direction, and historical levels for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Price_Features"
        )
        try:
            df = self._calculate_basic_price_features(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_basic_price_features: {e}", exc_info=True)
        
        # 2b. Momentum & Trend Features (MOM_t, MR_t, TF_t)
        if self.config.enable_momentum_features:
            reporter.check_cancellation()
            try:
                df = self._calculate_momentum_features(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_momentum_features: {e}", exc_info=True)

        # 2c. Advanced Microstructure Indicators (all purely historical)
        # NOTE: Candle structure must run BEFORE candle_bull_score (score depends on structure cols)
        #       SNR/VIX and Trend Strength are independent, can run in any order.
        if self.config.enable_candle_structure:
            reporter.check_cancellation()
            reporter.report(
                progress=19,
                message="Candle Structure",
                message2=f"Computing per-bar candle microstructure features for {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="Candle_Structure"
            )
            try:
                df = self._calculate_candle_structure(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_candle_structure: {e}", exc_info=True)

        if self.config.enable_snr_vix:
            reporter.check_cancellation()
            reporter.report(
                progress=19,
                message="SNR & VIX",
                message2=f"Computing Signal-to-Noise Ratio and VIX-like volatility for {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="SNR_VIX"
            )
            try:
                df = self._calculate_snr_vix_features(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_snr_vix_features: {e}", exc_info=True)

        reporter.check_cancellation()
        try:
            df = self._calculate_trend_strength(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_trend_strength: {e}", exc_info=True)

        if self.config.enable_volume_analysis and "Volume" in df.columns:
            reporter.check_cancellation()
            try:
                df = self._calculate_vsi(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_vsi: {e}", exc_info=True)

        # Candle Bull Score depends on candle structure AND RSI — run after both are ready
        # RSI is computed in step 4 below, so candle_bull_score is deferred until after RSI.
        
        # 2c. Session & Time Features (trading session, day, hour, minute)
        reporter.check_cancellation()
        reporter.report(
            progress=18,
            message="Time Features",
            message2=f"Calculating session classification and time-of-day features for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Session_Time"
        )
        try:
            df = self._calculate_session_time_features(df)
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating session/time features: {e}")
        
        if self.config.enable_pivots:
            reporter.check_cancellation()
            reporter.report(
                progress=18,
                message="Pivot Series",
                message2=f"Computing series pivots across {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="Pivot_Series"
            )
            try:
                df = self._calculate_pivots_series(df)
                df = self._calculate_structural_range_features(df)
            except Exception as e:
                logger.error(f"❌ Error in pivot/structural range calculations: {e}", exc_info=True)

        # 3. Core Indicators (Moving Averages) (20-30%)
        reporter.check_cancellation()
        try:
            df = self._calculate_moving_averages(df, reporter=reporter)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_moving_averages: {e}", exc_info=True)
        
        # 3b. EMA Change Series & Lettered Variables
        if self.config.enable_ema_change_series:
            reporter.check_cancellation()
            try:
                df = self._calculate_ema_change_series(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_ema_change_series: {e}", exc_info=True)
        
        if self.config.enable_lettered_variables:
            reporter.check_cancellation()
            try:
                df = self._calculate_lettered_variables(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_lettered_variables: {e}", exc_info=True)
        
        # 4. Core Indicators (RSI) (30-35%)
        reporter.check_cancellation()
        try:
            df = self._calculate_rsi_indicators(df, reporter=reporter)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_rsi_indicators: {e}", exc_info=True)

        # 4b. Candle Bull Score — deferred until after RSI (score uses RSI_14 / RSI_7)
        # Pin Bar features run first so candle_bull_score can use pin suppression.
        if self.config.enable_candle_structure:
            reporter.check_cancellation()
            reporter.report(
                progress=47,
                message="Pin Bar Detection",
                message2=f"Scoring pin bars, hammers, and shooting stars across {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="PinBar"
            )
            try:
                df = self._calculate_pin_bar_features(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_pin_bar_features: {e}", exc_info=True)

        if self.config.enable_candle_bull_score:
            reporter.check_cancellation()
            try:
                df = self._calculate_candle_bull_score(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_candle_bull_score: {e}", exc_info=True)

        # 4c. Price Velocity — directional pips-per-second (bull & bear independent)
        #     Depends on: ATR (from _calculate_other_indicators, step 6), so deferred to after step 6.
        #     Wired here as a post-step-6 call via flag; actual call is after _calculate_other_indicators.
        #     (See step 6b below.)
        
        # 5. Core Indicators (Signals & Bands) (35-50%)
        reporter.check_cancellation()
        reporter.report(
            progress=35,
            message="Signal Detection",
            message2=f"Detecting MA and RSI crossover signals across {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Crossovers"
        )
        try:
            df = self._calculate_crossovers(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_crossovers: {e}", exc_info=True)
        
        reporter.check_cancellation()
        reporter.report(
            progress=40,
            message="Bollinger Bands",
            message2=f"Computing bands (length={self.config.bb_length}, std={self.config.bb_std}) and squeeze signals",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="BB"
        )
        try:
            df = self._calculate_bollinger_bands(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_bollinger_bands: {e}", exc_info=True)
        
        reporter.check_cancellation()
        reporter.report(
            progress=45,
            message="MACD",
            message2=f"Computing MACD ({self.config.macd_fast}/{self.config.macd_slow}/{self.config.macd_signal}) and histogram",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="MACD"
        )
        try:
            df = self._calculate_macd(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_macd: {e}", exc_info=True)

        # 6. Advanced Indicators (50-62%)
        if self.config.enable_supertrend:
            reporter.check_cancellation()
            reporter.report(
                progress=50,
                message="Supertrend",
                message2=f"Computing Supertrend (period={self.config.supertrend_period}, multiplier={self.config.supertrend_multiplier})",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="Supertrend"
            )
            try:
                df = self._calculate_supertrend(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_supertrend: {e}", exc_info=True)

        reporter.check_cancellation()
        reporter.report(
            progress=52,
            message="Volatility",
            message2=f"Computing ATR({self.config.atr_length}) and historical volatility across {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="ATR"
        )
        try:
            df = self._calculate_other_indicators(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_other_indicators: {e}", exc_info=True)

        # 6b. Price Velocity — directional speed (pips/s, bull & bear independent)
        #     Requires ATR (computed in step 6 above). Must come AFTER _calculate_other_indicators.
        reporter.check_cancellation()
        reporter.report(
            progress=54,
            message="Price Velocity",
            message2=f"Computing directional bull/bear pips-per-second for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Price_Velocity"
        )
        try:
            df = self._calculate_price_velocity(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_price_velocity: {e}", exc_info=True)

        # 6c. Volatility Regime — normalised volatility vs history, directional split
        #     Requires VIX_20/ATR_Pct (available after steps 6 and 2c).
        reporter.check_cancellation()
        reporter.report(
            progress=55,
            message="Volatility Regime",
            message2=f"Computing volatility regime (expansion/contraction, bull/bear split) for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Volatility_Regime"
        )
        try:
            df = self._calculate_volatility_regime(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_volatility_regime: {e}", exc_info=True)

        # 6d. Regime Speed — trend-gated directional pace
        #     Requires Price_Velocity_Bull/Bear (step 6b) + Trend_Strength (step 2c).
        #     Supertrend (step 6a) used for direction gating if available.
        reporter.check_cancellation()
        reporter.report(
            progress=56,
            message="Regime Speed",
            message2=f"Computing trend-aligned advance speed for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Regime_Speed"
        )
        try:
            df = self._calculate_regime_speed(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_regime_speed: {e}", exc_info=True)

        # 6e. Reversal Score — trend death probability (58%)
        reporter.check_cancellation()
        reporter.report(
            progress=58,
            message="Reversal Score",
            message2=f"Computing trend reversal probability for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Reversal_Score"
        )
        try:
            df = self._calculate_reversal_score(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_reversal_score: {e}", exc_info=True)

        # 6f. Mean Reversion Score — snap-back probability (59%)
        reporter.check_cancellation()
        reporter.report(
            progress=59,
            message="Mean Reversion Score",
            message2=f"Computing mean reversion probability for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="MeanRev_Score"
        )
        try:
            df = self._calculate_mean_reversion_score(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_mean_reversion_score: {e}", exc_info=True)

        # 6g. Momentum Delta — strength comparison (60%)
        reporter.check_cancellation()
        reporter.report(
            progress=60,
            message="Momentum Delta",
            message2=f"Computing momentum strength comparison for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Mom_Delta"
        )
        try:
            df = self._calculate_momentum_delta(df)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_momentum_delta: {e}", exc_info=True)

        # 6h. Footprint / Volume Profile (tick-derived, optional)
        if self.config.enable_footprint:
            reporter.check_cancellation()
            reporter.report(
                progress=61,
                message="Footprint Features",
                message2=f"Computing volume-at-price footprint features for {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="Footprint"
            )
            try:
                df = self._calculate_footprint_features(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_footprint_features: {e}", exc_info=True)

        # 7. Volume Analysis
        if self.config.enable_volume_analysis and "Volume" in df.columns:
            reporter.check_cancellation()
            try:
                df = self._calculate_volume_indicators(df, calculate_last_n=calculate_last_n, reporter=reporter)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_volume_indicators: {e}", exc_info=True)

        # 8. Pattern Features
        if self.config.enable_heikinashi:
            reporter.check_cancellation()
            reporter.report(
                progress=62,
                message="Pattern Features",
                message2=f"Computing Heikin-Ashi Candles for {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="Heikin_Ashi"
            )
            try:
                df = self._calculate_heikinashi(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_heikinashi: {e}", exc_info=True)

        if self.config.enable_doji:
            reporter.check_cancellation()
            reporter.report(
                progress=65,
                message="Pattern Detection",
                message2=f"Detecting Doji patterns across {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="Doji"
            )
            try:
                df = self._calculate_doji(df)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_doji: {e}", exc_info=True)
            
        if self.config.enable_smc:
            reporter.check_cancellation()
            try:
                df = self._calculate_smc_indicators(df, reporter=reporter)
            except Exception as e:
                logger.error(f"❌ Error in _calculate_smc_indicators: {e}", exc_info=True)

        # 9. Support/Resistance & Trendlines
        reporter.check_cancellation()
        try:
            df = self._calculate_pivot_points(df, reporter=reporter)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_pivot_points: {e}", exc_info=True)

        reporter.check_cancellation()
        try:
            df = self._calculate_trendlines(df, calculate_last_n=calculate_last_n, reporter=reporter)
        except Exception as e:
            logger.error(f"❌ Error in _calculate_trendlines: {e}", exc_info=True)

        # 10. Finalization
        if self.config.enable_additional_features:
            reporter.check_cancellation()
            reporter.report(
                progress=90,
                message="Additional Features",
                message2=f"Computing additional metrics for {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="Additional"
            )
            # Logic handled by _calculate_structural_range_features after pivots
            pass

        reporter.check_cancellation()
        reporter.report(
            progress=95,
            message="Finalization",
            message2=f"Sorting and cleaning {len(df.columns)} columns for {n_rows:,} bars",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Cleanup"
        )
        
        try:
            result = self._finalize_dataframe(df)
        except Exception as e:
            logger.error(f"❌ Error in _finalize_dataframe: {e}", exc_info=True)
            result = df # Return current state if finalization fails
        
   
        reporter.report(
            progress=98,
            message="Processing Complete",
            message2=f"Calculated {len(result.columns)} indicators - awaiting storage",
            processed_bars=n_rows,
            total_bars=n_rows,
            current_indicator="Ready"
        )
        
        # Explicit memory cleanup
        del df
        gc.collect()
        
        return result

    def calculate_prebuilt_indicators(self, data: pd.DataFrame, indicators: List[Dict], task_id=None, progress_store=None, mode: str = "training", reporter: Optional[ProgressReporter] = None) -> pd.DataFrame:
        if not reporter:
            reporter = ProgressReporter(task_id, progress_store)
            if task_id: reporter.task_id = task_id
        
        if task_id and (progress_store or reporter):
            reporter.report(10, "Preparing data for prebuilt indicators...")
        
        self._calculation_mode = mode  # Store mode for use in sub-methods
        
        df = self._prepare_data(data.copy())
        
        total_indicators = len(indicators)
        for i, indicator_config in enumerate(indicators):
            try:
                indicator_id = indicator_config.get('id', '').lower()
                params = indicator_config.get('parameters', {})
                
                if task_id and (progress_store or reporter):
                    progress = 15 + int((i / total_indicators) * 70)
                    indicator_name = indicator_config.get('name', indicator_id)
                    reporter.report(progress, message=f"Calculating Indicators...", message2=f"Current: {indicator_name}")

                if 'sma' in indicator_id:
                    df[f"SMA_{params.get('length', 10)}"] = ta.sma(df[self.config.diff_column], length=params.get('length', 10))
                elif 'ema' in indicator_id:
                    df[f"EMA_{params.get('length', 10)}"] = ta.ema(df[self.config.diff_column], length=params.get('length', 10))
                elif 'rsi' in indicator_id:
                    df[f"RSI_{params.get('length', 14)}"] = ta.rsi(df[self.config.diff_column], length=params.get('length', 14))
                elif 'macd' in indicator_id:
                    df.ta.macd(
                        close=self.config.diff_column,
                        fast=params.get('fast', 12),
                        slow=params.get('slow', 26),
                        signal=params.get('signal', 9),
                        append=True,
                    )
                elif 'bbands' in indicator_id:
                    df.ta.bbands(
                        close=self.config.diff_column,
                        length=params.get('length', 20),
                        std=params.get('std', 2.0),
                        append=True,
                    )
                # Add other prebuilt indicators here based on their ID
            except Exception as e:
                logger.error(f"❌ Error calculating prebuilt indicator {indicator_config.get('id')}: {e}", exc_info=True)
                continue
            
        if task_id and (progress_store or reporter):
            reporter.report(90, message="Finalizing data...")

        result = self._finalize_dataframe(df)
        
        # Explicit memory cleanup
        del df
        gc.collect()
        
        return result

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Consolidated data preparation for all technical indicators.
        Merges standardization logic from analysis_route_helpers into core module.
        """
        # 1. Standardize column names (mapping variations from different providers)
        STANDARD_MAPPING = {
            'o': 'Open', 'open': 'Open', 'OPEN': 'Open',
            'h': 'High', 'high': 'High', 'HIGH': 'High',
            'l': 'Low', 'low': 'Low', 'LOW': 'Low',
            'c': 'Close', 'close': 'Close', 'CLOSE': 'Close',
            'v': 'Volume', 'vol': 'Volume', 'volume': 'Volume', 'VOLUME': 'Volume',
            'tick_volume': 'TickVolume', 'tick_vol': 'TickVolume', 'tickvolume': 'TickVolume',
            't': 'Time', 'time': 'Time', 'timestamp': 'Time', 'date': 'Time',
        }
        
        # Use lowercase mapping to resolve variations
        lowercase_actual_cols = {c.lower(): c for c in df.columns}
        rename_map = {}
        for norm_key, standard_name in STANDARD_MAPPING.items():
            if norm_key in lowercase_actual_cols:
                actual_col = lowercase_actual_cols[norm_key]
                rename_map[actual_col] = standard_name
        # 0. Deduplicate columns immediately (Case-Insensitive) to prevent "same-caps" issues
        df = df.loc[:, ~df.columns.str.lower().duplicated(keep='first')].copy()
        
        df.rename(columns=rename_map, inplace=True)

        # 2. Validate required OHLC columns
        required_ohlc = ["Open", "High", "Low", "Close"]
        missing_cols = [col for col in required_ohlc if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required OHLC columns: {missing_cols}")

        # 3. Ensure Volume column (synthetic if missing/empty)
        if 'Volume' not in df.columns or not (df['Volume'] > 0).any():
            if 'TickVolume' in df.columns:
                df['Volume'] = df['TickVolume']
            else:
                df['Volume'] = 1.0

        # 4. Standardize Time column and set Index
        if "Time" in df.columns:
            # Only convert if not already datetime
            if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
                if pd.api.types.is_numeric_dtype(df["Time"]):
                    df["Time"] = pd.to_datetime(df["Time"], unit='s', errors='coerce')
                else:
                    df["Time"] = pd.to_datetime(df["Time"], errors='coerce')
            
            # use drop=True to avoid duplicate columns. 
            # We will restore the "Time" column in _finalize_dataframe.
            df.set_index("Time", inplace=True, drop=True)
            df.index.name = "time_index"
        elif not isinstance(df.index, pd.DatetimeIndex):
            try:
                # Only use unit='s' if numeric
                if pd.api.types.is_numeric_dtype(df.index):
                    df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                else:
                    df.index = pd.to_datetime(df.index, errors='coerce')
            except Exception as e:
                logger.warning(f"Could not convert index to DatetimeIndex: {e}")

        # 5. Remove duplicates and reset index to integer for processing
        df = df.drop_duplicates()
        
        # 🛡️ DEFENSIVE: Strip duplicate index labels to prevent "cannot reindex on an axis with duplicate labels"
        if df.index.has_duplicates:
            df = df.loc[~df.index.duplicated(keep='last')].copy()
        
        # 6. Standardize diff_column to capitalized version if it's one of OHLC
        target_diff = self.config.diff_column.lower()
        if target_diff in ["open", "high", "low", "close"]:
            self.config.diff_column = target_diff.capitalize()
        
        if self.config.diff_column not in df.columns:
            # Try to find it case-insensitively
            found = False
            for col in df.columns:
                if col.lower() == target_diff:
                    self.config.diff_column = col
                    found = True
                    break
            if not found:
                raise ValueError(f"diff_column '{self.config.diff_column}' not found after standardization")

        # Calculate dynamic diff sentinel based on max OHLC
        try:
            max_val = df[["Open", "High", "Low", "Close"]].max().max()
            if pd.notna(max_val) and max_val > 0:
                self.dynamic_diff_sentinel = float(max_val)
            else:
                self.dynamic_diff_sentinel = LARGE_DIFF_SENTINEL_DEFAULT
                logger.warning("⚠️ OHLC values are invalid, using default LARGE_DIFF_SENTINEL.")
        except Exception as e:
            self.dynamic_diff_sentinel = LARGE_DIFF_SENTINEL_DEFAULT
            logger.warning(f"⚠️ Could not calculate dynamic sentinel from OHLC: {e}")

        return df

    def _calculate_basic_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate basic price-based features"""
        diff_col = self.config.diff_column
        new_cols = {}

        # Price changes and direction
        # ⚠️ Price_Change must NOT use future data (shift(-1)) - it's a feature, not a target
        # Use previous price momentum: current - previous
        new_cols["Price_Change"] = df[diff_col].diff().fillna(0)  # current - previous bar
        new_cols["Direction"] = (new_cols["Price_Change"] > 0).astype(int)
        
        # ⚠️ WARNING: Target_Close uses FUTURE DATA (shift(-1)) - only for training mode
        mode = getattr(self, '_calculation_mode', 'training')
        if mode == 'training':
            # Only in training: target is next bar's close (for supervised learning)
            new_cols["Target_Close"] = df[diff_col].shift(-1).fillna(df[diff_col])
        else:
            # In inference: target is current close (no future data)
            new_cols["Target_Close"] = df[diff_col]
        
        new_cols["Previous_Close"] = df[diff_col].shift(1)
        new_cols[f"{diff_col}PCT"] = df[diff_col].pct_change()

        # Historical price levels (always use actual OHLC)
        for i in range(1, 4):
            new_cols[f"Low_Day_{i}"] = df["Low"].shift(i)
            new_cols[f"High_Day_{i}"] = df["High"].shift(i)

        # Previous differences (use diff_col)
        for i in range(1, 6):
            new_cols[f"Prev_{i}_{diff_col}"] = df[diff_col].shift(i) - df[diff_col]

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum and trend features from andria_indicators.py:
        - MOM_t: Momentum (Close difference)
        - MR_t: Momentum Rate (percentage change)
        - TF_t: Trend Factor (detrended price)
        """
        diff_col = self.config.diff_column
        new_cols = {}
        
        # MOM_t: Raw momentum (difference from previous close)
        new_cols["MOM_t"] = df[diff_col].diff()
        
        # MR_t: Momentum Rate (percentage change)
        new_cols["MR_t"] = df[diff_col].pct_change()
        
        # TF_t: Trend Factor (detrended: deviation from rolling mean normalized by std)
        window = self.config.trend_factor_window
        rolling_mean = df[diff_col].rolling(window=window).mean()
        rolling_std = df[diff_col].rolling(window=window).std()
        new_cols["TF_t"] = (df[diff_col] - rolling_mean) / rolling_std.replace(0, np.nan)
        
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        return df

    def _calculate_session_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading session classification and time-of-day features.
        
        Enables ML models to learn time-specific patterns:
        - Asia trading hours: 0 (00:00 - 08:00 UTC)
        - US trading hours: 1 (08:00 - 16:00 UTC)
        - Europe/overlap: 2 (16:00 - 24:00 UTC)
        
        Returns:
            DataFrame with added columns:
            - session: 0, 1, or 2 (trading session category)
            - session_transition: 1 if session changed from previous bar, 0 otherwise
            - day_of_week: 0-6 (Monday=0, Sunday=6)
            - hour: 0-23 (hour of day)
            - minute: 0-59 (minute of hour)
        """
        new_cols = {}
        
        # ──────────────────────────────────────────────────────────────────────────
        # Extract time features from index (either DatetimeIndex or Time column)
        # ──────────────────────────────────────────────────────────────────────────
        if isinstance(df.index, pd.DatetimeIndex):
            time_index = df.index
        elif "Time" in df.columns:
            try:
                time_index = pd.to_datetime(df["Time"])
            except Exception as e:
                self.logger.warning(f"⚠️ Could not parse Time column: {e}. Using defaults.")
                new_cols["session"] = 1  # Default to US hours
                new_cols["session_transition"] = 0
                new_cols["day_of_week"] = 0
                new_cols["hour"] = 12
                new_cols["minute"] = 0
                for col_name, col_data in new_cols.items():
                    df[col_name] = col_data if isinstance(col_data, pd.Series) else col_data
                return df
        else:
            self.logger.warning("⚠️ No Time column or DatetimeIndex found. Using default values.")
            new_cols["session"] = 1
            new_cols["session_transition"] = 0
            new_cols["day_of_week"] = 0
            new_cols["hour"] = 12
            new_cols["minute"] = 0
            for col_name, col_data in new_cols.items():
                df[col_name] = col_data if isinstance(col_data, pd.Series) else col_data
            return df
        
        # ──────────────────────────────────────────────────────────────────────────
        # Session Classification (UTC-based trading hours)
        # ──────────────────────────────────────────────────────────────────────────
        # Session 0: 00:00 - 08:00 (Asia trading)
        # Session 1: 08:00 - 16:00 (US trading)
        # Session 2: 16:00 - 24:00 (Europe/evening)
        hours = time_index.hour
        new_cols["session"] = np.where(
            (hours >= 0) & (hours < 8), 0,
            np.where((hours >= 8) & (hours < 16), 1, 2)
        )
        
        # ──────────────────────────────────────────────────────────────────────────
        # Session Transition Detection
        # ──────────────────────────────────────────────────────────────────────────
        # Flags when session changes from previous bar (useful for structure breaks)
        session_series = pd.Series(new_cols["session"], index=df.index)
        new_cols["session_transition"] = (
            (session_series.shift(1) != session_series) & (session_series.shift(1).notna())
        ).astype(int).values
        
        # ──────────────────────────────────────────────────────────────────────────
        # Time-of-Day Features
        # ──────────────────────────────────────────────────────────────────────────
        # day_of_week: 0=Monday, 1=Tuesday, ..., 6=Sunday
        new_cols["day_of_week"] = time_index.dayofweek.values
        
        # hour: 0-23 (hour of day)
        new_cols["hour"] = hours.values
        
        # minute: 0-59 (minute of hour)
        new_cols["minute"] = time_index.minute.values
        
        # ──────────────────────────────────────────────────────────────────────────
        # Add all time features to DataFrame
        # ──────────────────────────────────────────────────────────────────────────
        for col_name, col_data in new_cols.items():
            df[col_name] = col_data
        
        self.logger.debug(
            f"✅ Session/time features calculated: "
            f"sessions={df['session'].nunique()}, "
            f"transitions={df['session_transition'].sum()}, "
            f"days={df['day_of_week'].nunique()}, "
            f"hours={df['hour'].nunique()}, "
            f"minutes={df['minute'].nunique()}"
        )
        
        return df

    def _calculate_moving_averages(self, df: pd.DataFrame, reporter: Optional[ProgressReporter] = None) -> pd.DataFrame:
        """Calculate moving averages and their differences"""
        diff_col = self.config.diff_column
        n_rows = len(df)
        new_cols = {}
        
        # Calculate total operations for progress tracking
        total_sma_range = self.config.sma_range_periods[1] - self.config.sma_range_periods[0]
        total_sma_configs = 7  # Standard SMA configs
        total_sma_mappings = 4  # Additional SMA mappings
        total_sma_dynamic = len(self.config.sma_periods)
        total_ema = len(self.config.ema_periods)
        total_ops = total_sma_range + total_sma_configs + total_sma_mappings + 3 + total_sma_dynamic + total_ema
        completed_ops = 0
        
        if reporter:
            reporter.report(
                progress=20,
                message="Moving Averages (0%)",
                message2=f"Initializing SMA/EMA calculations for {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="MA_Init"
            )

        # SMA calculations for range periods (difference from diff_col)
        start_period, end_period = self.config.sma_range_periods
        for period in range(start_period, end_period):
            try:
                sma_val = ta.sma(df[diff_col], length=period)
                if sma_val is None:
                    logger.warning(f"⚠️ ta.sma returned None for SMA_Diff_{period} (series too short or column missing)")
                    continue
                new_cols[f"SMA_Diff_{period}"] = sma_val - df[diff_col]
                # LEGACY NAMING: SMA0-SMA11 (for andria_indicators compatibility)
                if self.config.enable_sma_range_diffs and start_period == 2 and end_period == 14:
                    new_cols[f"SMA{period-2}"] = sma_val - df[diff_col]
            except Exception as e:
                logger.error(f"❌ Error calculating SMA_Diff_{period}: {e}")
            completed_ops += 1
            
            # Report every 5 periods or at key milestones
            if completed_ops % 5 == 0 or period == end_period - 1:
                progress_pct = (completed_ops / total_ops) * 100
                if reporter:
                    reporter.report(
                        progress=20 + int(progress_pct * 0.05),  # 20-25% range
                        message=f"Moving Averages ({int(progress_pct)}%)",
                        message2=f"Computing SMA_Diff({period}) - range {start_period} to {end_period}",
                        processed_bars=n_rows,
                        total_bars=n_rows,
                        current_indicator=f"SMA_Diff_{period}"
                    )

        # Standard SMA periods (actual values, not differences)
        sma_configs = [
            (10, "Short_MA_10"),
            (25, "Long_MA_25"),
            (50, "Short_MA_50"),
            (100, "Long_MA_100"),
            (10, "10_Day_MA"),
            (25, "50_Day_MA"),  # Note: This was 25 in original
            (20, "SMA_20"),
        ]

        for period, name in sma_configs:
            new_cols[name] = df[diff_col].rolling(window=period).mean()
            completed_ops += 1
            
            progress_pct = (completed_ops / total_ops) * 100
            if reporter:
                reporter.report(
                    progress=20 + int(progress_pct * 0.05),
                    message=f"Moving Averages ({int(progress_pct)}%)",
                    message2=f"Computing {name} (period={period})",
                    processed_bars=n_rows,
                    total_bars=n_rows,
                    current_indicator=name
                )

        # Additional SMA calculations with specific naming — driven by config
        sma_mappings = [
            (self.config.sma_ma_25,  "MA_25"),
            (self.config.sma_ma_50,  "MA_50"),
            (self.config.sma_ma_100, "MA_100"),
            (self.config.sma_ma_200, "MA_200"),
        ]
        for period, name in sma_mappings:
            new_cols[name] = ta.sma(df[diff_col], length=period)
            completed_ops += 1

        # Short and long-term moving averages — use config periods
        new_cols["Short_MA"] = ta.sma(df[diff_col], length=self.config.short_ma)
        new_cols["Long_MA"] = ta.sma(df[diff_col], length=self.config.long_ma)
        new_cols["MA"] = ta.sma(df[diff_col], length=self.config.ma)
        completed_ops += 3

        # Dynamic MA lengths
        for i, length in enumerate(self.config.sma_periods):
            new_cols[f"SMA_{length}"] = ta.sma(df[diff_col], length=length)
            completed_ops += 1
            
            progress_pct = (completed_ops / total_ops) * 100
            if reporter:
                reporter.report(
                    progress=20 + int(progress_pct * 0.05),
                    message=f"Moving Averages ({int(progress_pct)}%)",
                    message2=f"Computing SMA({length}) - {i+1}/{len(self.config.sma_periods)} configured periods",
                    processed_bars=n_rows,
                    total_bars=n_rows,
                    current_indicator=f"SMA_{length}"
                )

        # EMA calculations
        for i, period in enumerate(self.config.ema_periods):
            new_cols[f"EMA_{period}"] = ta.ema(df[diff_col], length=period)
            completed_ops += 1
            
            progress_pct = (completed_ops / total_ops) * 100
            if reporter:
                reporter.report(
                    progress=20 + int(progress_pct * 0.05),
                    message=f"Moving Averages ({int(progress_pct)}%)",
                    message2=f"Computing EMA({period}) with exponential weighting - {i+1}/{len(self.config.ema_periods)}",
                    processed_bars=n_rows,
                    total_bars=n_rows,
                    current_indicator=f"EMA_{period}"
                )

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        # Calculate MA differences
        self._calculate_ma_differences(df, diff_col, reporter=reporter)

        return df

    def _calculate_ma_differences(self, df: pd.DataFrame, diff_col: str, reporter: Optional[ProgressReporter] = None):
        """Calculate moving average differences and relationships (driven by config, not hardcoded)."""
        n_rows = len(df)
        new_cols = {}

        if reporter:
            reporter.report(
                progress=25,
                message="MA Relationships",
                message2=f"Analyzing MA divergence, crossovers, and price relationships across {n_rows:,} bars",
                processed_bars=n_rows,
                total_bars=n_rows,
                current_indicator="MA_Divergence"
            )

        # ── Price vs standard named MAs ──────────────────────────────────────
        new_cols["Short_Period_MA_Diff"] = df["Long_MA_25"] - df["Short_MA_10"]
        new_cols["Long_Period_MA_Diff"] = df["Long_MA_100"] - df["Short_MA_50"]
        new_cols["Price_Short_Period_Diff"] = df[diff_col] - df["Short_MA_10"]
        new_cols["Price_Short_Long_Period_Diff"] = df[diff_col] - df["Long_MA_25"]
        new_cols["Price_Long_Short_Period_Diff"] = df[diff_col] - df["Short_MA_50"]
        new_cols["Price_Long_Long_Period_Diff"] = df[diff_col] - df["Long_MA_100"]

        new_cols["Volume_Change_Pct"] = self._clean_inf(df["Volume"].pct_change(), 0.0)
        
        if "Bar_Volume_Up" in df.columns:
            new_cols["Up_Volume_Change_Pct"] = self._clean_inf(df["Bar_Volume_Up"].pct_change(), 0.0)
        if "Bar_Volume_Down" in df.columns:
            new_cols["Down_Volume_Change_Pct"] = self._clean_inf(df["Bar_Volume_Down"].pct_change(), 0.0)
        # Price vs config-driven Short_MA / Long_MA
        if "Short_MA" in df.columns:
            new_cols["Short_MA_Diff"] = df[diff_col] - df["Short_MA"]
        if "Long_MA" in df.columns:
            new_cols["Long_MA_Diff"] = df[diff_col] - df["Long_MA"]

        # ── Named MA diffs (MA_25/50/100/200) ───────────────────────────────
        named_ma_diffs = [
            ("MA_25",  "MA_25_Diff"),
            ("MA_50",  "MA_50_Diff"),
            ("MA_100", "MA_100_Diff"),
            ("MA_200", "MA_200_Diff"),
        ]
        for ma_col, diff_name in named_ma_diffs:
            if ma_col in df.columns:
                new_cols[diff_name] = df[ma_col] - df[diff_col]

        # ── EMA diffs — driven by config.ema_periods (no hardcoded list) ────
        for period in self.config.ema_periods:
            ema_col = f"EMA_{period}"
            if ema_col in df.columns:
                new_cols[f"EMA_{period}_Diff"] = df[ema_col] - df[diff_col]

        # EMA cross-period diffs (each period vs shortest configured EMA)
        base_ema_period = min(self.config.ema_periods) if self.config.ema_periods else 8
        base_ema_col = f"EMA_{base_ema_period}"
        for period in self.config.ema_periods:
            if period != base_ema_period:
                ema_col = f"EMA_{period}"
                if ema_col in df.columns and base_ema_col in df.columns:
                    new_cols[f"EMA_{period}_Minus_EMA{base_ema_period}"] = df[ema_col] - df[base_ema_col]

        # ── MA-200 vs lagged price (trend context) ───────────────────────────
        if "MA_200" in df.columns:
            for i in range(5):
                new_cols[f"MA_200_Change_{i}"] = df["MA_200"] - df[diff_col].shift(i)

        # ── Dynamic SMA diffs (from config.sma_periods) ─────────────────────
        for length in self.config.sma_periods:
            sma_col = f"SMA_{length}"
            if sma_col in df.columns:
                new_cols[f"SMA_{length}_Diff"] = df[sma_col] - df[diff_col]

        # ── SMA_100 vs SMA_50 spread ─────────────────────────────────────────
        if "SMA_100" in df.columns and "SMA_50" in df.columns:
            new_cols["MA_100_50_Diff"] = df["SMA_100"] - df["SMA_50"]
        else:
            # Not 0 — that would mean equal. Use sentinel: spread is unknowable.
            new_cols["MA_100_50_Diff"] = self.dynamic_diff_sentinel

        if new_cols:
            combined = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
            # Update the original dataframe content in place
            for col in new_cols:
                df[col] = combined[col]
            del combined
            gc.collect()

    def _calculate_lettered_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate compact lettered variable representation (a-i) from andria_indicators.py:
        - a = MA_25 - Close
        - b = MA_50 - Close
        - c = MA_100 - Close
        - d = MA_200 - Close
        - e = EMA_8 - Close
        - f = EMA_10 - Close
        - g = EMA_12 - Close
        - h = EMA_24 - Close
        - i = EMA_32 - Close
        """
        diff_col = self.config.diff_column
        new_cols = {}
        
        # MA-based letters
        ma_mappings = [
            ("MA_25", "a"),
            ("MA_50", "b"),
            ("MA_100", "c"),
            ("MA_200", "d"),
        ]
        for ma_col, letter in ma_mappings:
            if ma_col in df.columns:
                new_cols[letter] = df[ma_col] - df[diff_col]
        
        # EMA-based letters
        ema_mappings = [
            ("EMA_8", "e"),
            ("EMA_10", "f"),
            ("EMA_12", "g"),
            ("EMA_24", "h"),
            ("EMA_32", "i"),
        ]
        for ema_col, letter in ema_mappings:
            if ema_col in df.columns:
                new_cols[letter] = df[ema_col] - df[diff_col]
        
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        return df

    def _calculate_ema_change_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate EMA change series (MA_Change0-4) from andria_indicators.py:
        These represent differences between longer EMAs and EMA-8 baseline.
        - MA_Change0 = EMA_64 - EMA_8
        - MA_Change1 = EMA_32 - EMA_8
        - MA_Change2 = EMA_24 - EMA_8
        - MA_Change3 = EMA_21 - EMA_8
        - MA_Change4 = EMA_18 - EMA_8
        """
        new_cols = {}
        
        base_ema = "EMA_8"
        change_periods = [64, 32, 24, 21, 18]
        
        if base_ema in df.columns:
            for idx, period in enumerate(change_periods):
                ema_col = f"EMA_{period}"
                if ema_col in df.columns:
                    new_cols[f"MA_Change{idx}"] = df[ema_col] - df[base_ema]
        
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        return df

    def _calculate_rsi_indicators(self, df: pd.DataFrame, reporter: Optional[ProgressReporter] = None) -> pd.DataFrame:
        """Calculate RSI indicators"""
        diff_col = self.config.diff_column
        new_cols = {}
        
        if reporter:
            reporter.report(30, "Calculating core indicators...", "Computing Relative Strength Index (RSI)")

        # Calculate RSI for all configured periods
        for period in self.config.rsi_periods:
            try:
                rsi_series = ta.rsi(df[diff_col], length=period)
                if rsi_series is None:
                    logger.warning(f"⚠️ ta.rsi returned None for period={period} (series too short or column missing)")
                    continue
                col_name = f"RSI_{period}"
                new_cols[col_name] = rsi_series
                
                # Setup RSI change calculations if it's the first configured period (usually 7 or 14)
                if period == self.config.rsi_periods[0]:
                    new_cols["RSI"] = rsi_series  # Main RSI column for consistency
                    # INF-SAFE: Handle RSI=0 division-by-zero
                    pct_change = rsi_series.pct_change()
                    new_cols[f"RSI_{period}_Pct_Change"] = self._clean_inf(pct_change, 0.0)
                    
                    # RSI change lags
                    for i in range(1, self.config.rsi_change_periods + 1):
                        new_cols[f"RSI_{period}_Change_Lag_{i}"] = rsi_series - rsi_series.shift(i)
            except Exception as e:
                logger.error(f"❌ Error calculating RSI_{period}: {e}")

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_crossovers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate moving average crossovers"""
        new_cols = {}

        crossovers = [
            # (fast_ma, slow_ma, above_name, diff_name)
            ("Short_MA", "Long_MA", "Short_Above_Long_Crossover", None),
            ("EMA_8", "EMA_12", "EMA8_Above_EMA12", "Cross_EMA8_Above_EMA12"),
            ("EMA_12", "EMA_18", "EMA12_Above_EMA18", "Cross_EMA12_Above_EMA18"),
            ("MA_25", "MA_50", "MA25_Above_MA50", "Cross_MA25_Above_MA50"),
            ("MA_50", "MA_100", "MA50_Above_MA100", "Cross_MA50_Above_MA100"),
        ]

        for fast_ma, slow_ma, above_name, diff_name in crossovers:
            if fast_ma in df.columns and slow_ma in df.columns:
                # Ensure columns are numeric and handle NaN/None
                s_fast = pd.to_numeric(df[fast_ma], errors='coerce').fillna(0)
                s_slow = pd.to_numeric(df[slow_ma], errors='coerce').fillna(0)
                
                if above_name == "Short_Above_Long_Crossover":
                    new_cols[above_name] = np.where(s_fast > s_slow, 1, 0)
                else:
                    new_cols[above_name] = (s_fast >= s_slow).astype(float)
                    if diff_name:
                        new_cols[diff_name] = pd.Series(new_cols[above_name], index=df.index).diff().fillna(0).astype(float)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        diff_col = self.config.diff_column

        # Calculate Bollinger Bands using diff_col
        df.ta.bbands(
            close=diff_col,
            length=self.config.bb_length,
            std=self.config.bb_std,
            append=True,
        )

        # Calculate BB differences
        bb_upper_col = f"BBU_{self.config.bb_length}_{self.config.bb_std}"
        bb_lower_col = f"BBL_{self.config.bb_length}_{self.config.bb_std}"
        bb_mid_col = f"BBM_{self.config.bb_length}_{self.config.bb_std}"

        if all(col in df.columns for col in [bb_upper_col, bb_lower_col, bb_mid_col]):
            df["BB_Upper_Diff"] = df[bb_upper_col] - df[diff_col]
            df["BB_Lower_Diff"] = df[bb_lower_col] - df[diff_col]
            # Mid-band diff: tells the model whether price is above/below the BB centre
            df["BB_Mid_Diff"] = df[bb_mid_col] - df[diff_col]
            df["BB_Squeeze"] = (df[bb_upper_col] - df[bb_lower_col]) / df[
                bb_mid_col
            ].replace(0, np.nan)

        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD"""
        df.ta.macd(
            close=self.config.diff_column,
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal,
            append=True,
        )
        return df

    def _calculate_other_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate other technical indicators"""
        diff_col = self.config.diff_column

        # ── Historical Volatility — column name uses config period ───────────────────
        hv_col = f"Historical_Volatility_{self.config.historical_volatility_length}"
        df[hv_col] = ta.stdev(df[diff_col], length=self.config.historical_volatility_length)
        # Alias without period suffix for backward compatibility
        df["Historical_Volatility"] = df[hv_col]
        df[hv_col] = df[hv_col].ffill()
        df["Historical_Volatility"] = df["Historical_Volatility"].ffill()

        # ── Parabolic SAR — column name derived from config (no hardcoded string) ──
        psar_af = self.config.psar_af
        psar_max_af = self.config.psar_max_af
        # pandas_ta names the column: PSARl_{af}_{max_af}  (long) or PSARs (short)
        psar_col_long = f"PSARl_{psar_af}_{psar_max_af}"
        psar_result = ta.psar(df["High"], df["Low"], df[diff_col],
                              af=psar_af, max_af=psar_max_af)
        if psar_result is not None and psar_col_long in psar_result.columns:
            df["Parabolic_SAR"] = psar_result[psar_col_long]
        else:
            # Fall back to first available PSAR column
            psar_cols = [c for c in (psar_result.columns if psar_result is not None else []) if c.startswith("PSAR")]
            df["Parabolic_SAR"] = psar_result[psar_cols[0]] if psar_cols else np.nan
        df["Parabolic_SAR"] = df["Parabolic_SAR"].ffill()

        # PSAR diff: distance between PSAR level and current price
        df["PSAR_Diff"] = df["Parabolic_SAR"] - df[diff_col]

        # ── ATR ────────────────────────────────────────────────────────────────────────
        df["ATR"] = ta.atr(df["High"], df["Low"], df[diff_col], length=self.config.atr_length)
        df["ATR"] = df["ATR"].ffill()
        # ATR as % of price: normalised volatility context for the model
        df["ATR_Pct"] = (df["ATR"] / df[diff_col].replace(0, np.nan)) * 100

        # ── Candle calculations ──────────────────────────────────────────────────
        # Candle_Size: range within current candle (High - Low)
        df["Candle_Size"] = df["High"] - df["Low"]
        # ⚠️ FIXED: Candle_Range must NOT use future data (shift(-1))
        # Changed from: Close - Close.shift(-1) [uses next close - LEAKAGE]
        # To: Close - Open [range within same candle - NO LEAKAGE]
        df["Candle_Range"] = df[diff_col] - df["Open"]
        df["Candle_Range"] = df["Candle_Range"].fillna(0)

        return df

    def _calculate_volume_indicators(self, df: pd.DataFrame, calculate_last_n: Optional[int] = None, reporter: Optional[ProgressReporter] = None) -> pd.DataFrame:
        """Calculate volume-based indicators"""
        diff_col = self.config.diff_column

        if reporter:
            reporter.report(60, "Performing Volume Analysis...", "Computing OBV & Bar Direction", current_indicator="Volume_OBV")

        df["Tick_Volume"] = df["Volume"]
        df["OBV"] = ta.obv(df[diff_col], df["Tick_Volume"])

        # Bar direction based on diff_column
        df["Is_Up_Bar"] = df[diff_col].diff() > 0
        df["Is_Down_Bar"] = ~df["Is_Up_Bar"]

        # Initialize volume columns
        volume_cols = [
            "Bar_Volume_Up",
            "Bar_Volume_Down",
            "Up_Distance",
            "Down_Distance",
        ]
        for col in volume_cols:
            df[col] = 0.0

        # Initialize swing tracking
        swing_cols = [
            "Price_Diff_From_Last_Swing_Low",
            "Price_Diff_From_Last_Swing_High",
            "Time_Diff_From_Last_Swing_Low",
            "Time_Diff_From_Last_Swing_High",
            "Speed_From_Last_Swing_Low",
            "Speed_From_Last_Swing_High",
        ]
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=[c for c in swing_cols if c not in df.columns])], axis=1)

        # Volume and swing calculations
        self._calculate_volume_metrics(df, diff_col, calculate_last_n=calculate_last_n, reporter=reporter)

        # Volume change percentages
        if reporter:
            reporter.report(61.8, "Performing Volume Analysis...", "Computing volume change percentages", current_indicator="Volume_Pct_Change")

        volume_change_cols = [
            ("Tick_Volume", "Volume_Change_Pct"),
            ("Bar_Volume_Up", "Up_Volume_Change_Pct"),
            ("Bar_Volume_Down", "Down_Volume_Change_Pct"),
        ]

        volume_change_df = {}
        for source_col, target_col in volume_change_cols:
            pct_change = df[source_col].pct_change(fill_method=None) * 100
            volume_change_df[target_col] = self._clean_inf(pct_change, 0.0)

        if volume_change_df:
            df = pd.concat([df, pd.DataFrame(volume_change_df, index=df.index)], axis=1)

        if reporter:
            reporter.report(61.9, "Performing Volume Analysis...", "Finalizing volume indicators", current_indicator="Volume_Finalize")
        
        gc.collect()

        return df

    def _calculate_volume_metrics(self, df: pd.DataFrame, diff_col: str, calculate_last_n: Optional[int] = None, reporter: Optional[ProgressReporter] = None):
        """Calculate rolling volume and distance metrics"""

        if reporter:
            reporter.report(60.1, "Performing Volume Analysis...", "Initializing swing and pivot tracking", current_indicator="Volume_Metrics_Init")

        last_swing_low_index = None
        last_swing_high_index = None
        low_pivot_label = df["Pivots"].min() if "Pivots" in df.columns else np.nan
        high_pivot_label = df["Pivots"].max() if "Pivots" in df.columns else np.nan
        use_pivots = (
            "Pivots" in df.columns
            and not pd.isna(low_pivot_label)
            and not pd.isna(high_pivot_label)
        )

        # Create a copy of time data to avoid mutating the original Time column
        # This prevents issues with time difference calculations
        time_data = None
        if "Time" in df.columns:
            time_data = df["Time"].copy()
        elif isinstance(df.index, pd.DatetimeIndex):
            time_data = pd.Series(df.index, index=df.index)

        start_i = 0
        if calculate_last_n is not None:
            start_i = max(0, len(df) - calculate_last_n)

        total_iterations = len(df) - start_i
        
        # Determine reporting step based on total iterations to avoid flooding the task queue
        reporting_step = max(1, total_iterations // 200)

        for i in range(start_i, len(df)):
            timestamp = df.index[i]

            if not calculate_last_n or calculate_last_n > 50:
                if i % 100 == 0:
                    logger.debug(f"Volume metrics progress: {i}/{len(df)}")
           
            if reporter and total_iterations > 0:
                if i % reporting_step == 0 or i == start_i or i == len(df) - 1:
                    reporter.check_cancellation()
                    reporter.report_loop(
                        i - start_i, 
                        total_iterations, 
                        message="Performing Volume Analysis...", 
                        message2="Analyzing Candle {current}/{total} (Vol/Price relationship)",
                        base_progress=60.1,
                        progress_range=1.6,
                        processed_bars=i,
                        current_indicator="Volume_Metrics"
                    )
            
            window_start = max(i - self.config.bars + 1, 0)
            window_data = df.iloc[window_start : i + 1]

            # Volume calculations
            df.at[timestamp, "Bar_Volume_Up"] = window_data[
                window_data["Is_Up_Bar"] == True
            ]["Tick_Volume"].sum()
            df.at[timestamp, "Bar_Volume_Down"] = window_data[
                window_data["Is_Down_Bar"] == True
            ]["Tick_Volume"].sum()

            # Distance calculations
            df.at[timestamp, "Up_Distance"] = window_data[
                window_data["Is_Up_Bar"] == True
            ]["Candle_Range"].sum()
            df.at[timestamp, "Down_Distance"] = window_data[
                window_data["Is_Down_Bar"] == True
            ]["Candle_Range"].sum()

            # Update swing indices based on diff_col
            if use_pivots:
                if not pd.isna(df.at[timestamp, "Pivots"]):
                    if df.at[timestamp, "Pivots"] == low_pivot_label:
                        last_swing_low_index = i
                    elif df.at[timestamp, "Pivots"] == high_pivot_label:
                        last_swing_high_index = i

                last_swing_low_close = (
                    df.iloc[last_swing_low_index]["Pivot_Price"]
                    if last_swing_low_index is not None and not pd.isna(df.iloc[last_swing_low_index].get("Pivot_Price", np.nan))
                    else df.iloc[last_swing_low_index][diff_col] if last_swing_low_index is not None else np.nan
                )
                last_swing_high_close = (
                    df.iloc[last_swing_high_index]["Pivot_Price"]
                    if last_swing_high_index is not None and not pd.isna(df.iloc[last_swing_high_index].get("Pivot_Price", np.nan))
                    else df.iloc[last_swing_high_index][diff_col] if last_swing_high_index is not None else np.nan
                )

                if time_data is not None:
                    last_swing_low_time = (
                        time_data.iloc[last_swing_low_index]
                        if last_swing_low_index is not None
                        else None
                    )
                    last_swing_high_time = (
                        time_data.iloc[last_swing_high_index]
                        if last_swing_high_index is not None
                        else None
                    )
                else:
                    last_swing_low_time = None
                    last_swing_high_time = None
            else:
                close_so_far = df.iloc[: i + 1][diff_col]
                last_swing_low_index = close_so_far.argmin()
                last_swing_high_index = close_so_far.argmax()
                last_swing_low_close = close_so_far.iloc[last_swing_low_index]
                last_swing_high_close = close_so_far.iloc[last_swing_high_index]

                if time_data is not None:
                    last_swing_low_time = time_data.iloc[last_swing_low_index]
                    last_swing_high_time = time_data.iloc[last_swing_high_index]
                else:
                    last_swing_low_time = None
                    last_swing_high_time = None

            # Price differences from swings
            df.at[timestamp, "Price_Diff_From_Last_Swing_Low"] = (
                df.at[timestamp, diff_col] - last_swing_low_close
                if not pd.isna(last_swing_low_close)
                else np.nan
            )
            df.at[timestamp, "Price_Diff_From_Last_Swing_High"] = (
                df.at[timestamp, diff_col] - last_swing_high_close
                if not pd.isna(last_swing_high_close)
                else np.nan
            )

            # Time differences and speed calculations
            # 🛡️ Safety: Only convert if not already a Timestamp to avoid 1970 bug
            if isinstance(timestamp, pd.Timestamp):
                current_time = timestamp
            else:
                current_time = pd.to_datetime(timestamp, unit='s', errors='coerce')

            # Low swing calculations
            if last_swing_low_time is not None:
                time_diff_low = (current_time - last_swing_low_time).total_seconds()
                df.at[timestamp, "Time_Diff_From_Last_Swing_Low"] = time_diff_low
                df.at[timestamp, "Speed_From_Last_Swing_Low"] = (
                    df.at[timestamp, "Price_Diff_From_Last_Swing_Low"] / time_diff_low
                    if time_diff_low != 0
                    else 0
                )
            else:
                df.at[timestamp, "Time_Diff_From_Last_Swing_Low"] = 0
                df.at[timestamp, "Speed_From_Last_Swing_Low"] = 0

            # High swing calculations
            if last_swing_high_time is not None:
                time_diff_high = (current_time - last_swing_high_time).total_seconds()
                df.at[timestamp, "Time_Diff_From_Last_Swing_High"] = time_diff_high
                df.at[timestamp, "Speed_From_Last_Swing_High"] = (
                    df.at[timestamp, "Price_Diff_From_Last_Swing_High"] / time_diff_high
                    if time_diff_high != 0
                    else 0
                )
            else:
                df.at[timestamp, "Time_Diff_From_Last_Swing_High"] = 0
                df.at[timestamp, "Speed_From_Last_Swing_High"] = 0
            
            # 🧹 CLEANUP: Current loop data
            del window_data
        
        # 🧹 FINAL CLEANUP: Single collection after full loop
        gc.collect()

    def _calculate_pivot_points(self, df: pd.DataFrame, reporter: Optional[ProgressReporter] = None) -> pd.DataFrame:
        """Calculate pivot points and differences"""
        diff_col = self.config.diff_column
        new_cols = {}

        # Calculate pivot points using external module if not already present
        if "pivot" not in df.columns:
            try:
                pivot_df = pivots_points(df, timeperiod=self.config.lookback_window, levels=self.config.pivot_levels)
                for col in pivot_df.columns:
                    if col not in df.columns:
                        new_cols[col] = pivot_df[col]
            except Exception as e:
                self.logger.warning(f"Failed to compute pivot points: {e}")

        # Initialize pivot columns to NaN if computation failed or returned missing columns
        pivot_cols = (
            ["pivot"]
            + [f"r{i}" for i in range(1, self.config.pivot_levels + 1)]
            + [f"s{i}" for i in range(1, self.config.pivot_levels + 1)]
        )
        for col in pivot_cols:
            if col not in df.columns and col not in new_cols:
                new_cols[col] = np.nan

        # Calculate pivot differences using diff_col
        # NOTE: We use Pivot_Price (last swing high/low) for the main Pivot_Diff
        pivot_price_series = df["Pivot_Price"] if "Pivot_Price" in df.columns else (new_cols["Pivot_Price"] if "Pivot_Price" in new_cols else None)
        if pivot_price_series is not None:
            new_cols["Pivot_Diff"] = pivot_price_series.ffill() - df[diff_col]

        # Support and resistance level differences
        for level in range(1, self.config.pivot_levels + 1):
            if reporter is not None:
                reporter.report(75, "Calculating Pivot Points...", f"Processing Support/Resistance Level {level}")
            r_col = f"r{level}"
            s_col = f"s{level}"
            r_diff_col = f"Pivot_R{level}_Diff"
            s_diff_col = f"Pivot_S{level}_Diff"

            # Get level values from df or new_cols
            r_val = df[r_col] if r_col in df.columns else new_cols.get(r_col)
            s_val = df[s_col] if s_col in df.columns else new_cols.get(s_col)
            
            if r_val is not None:
                new_cols[r_diff_col] = r_val.ffill() - df[diff_col]
            if s_val is not None:
                new_cols[s_diff_col] = s_val.ffill() - df[diff_col]

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_trendlines(self, df: pd.DataFrame, calculate_last_n: Optional[int] = None, reporter: Optional[ProgressReporter] = None) -> pd.DataFrame:
        """Calculate trendline analysis"""
        diff_col = self.config.diff_column

        # Initialize trendline columns
        trendline_cols = [
            "Support_Trendline_Value",
            "Resist_Trendline_Value",
            "Support_Trendline_Diff",
            "Resist_Trendline_Diff",
        ]
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=[c for c in trendline_cols if c not in df.columns])], axis=1)

        start_i = self.config.lookback_window - 1
        if calculate_last_n is not None:
            start_i = max(start_i, len(df) - calculate_last_n)

        total_iterations = len(df) - start_i
        for i in range(start_i, len(df)):
            timestamp = df.index[i]
            if reporter and i % 25 == 0:
                # Trendlines progress spans from 80% to 95%
                loop_progress = 80 + int(((i - start_i) / total_iterations) * 15)
                reporter.report(
                    loop_progress, 
                    "Drawing Trendlines...", 
                    f"Refining High-Low Regression Segment {i}/{len(df)}"
                )
            # ...
            timestamp = df.index[i]
            trendline_slice = df.iloc[
                i - self.config.lookback_window + 1 : i + 1
            ].copy()
            high_vals = (
                trendline_slice["High"].astype(float).ffill().bfill().fillna(0).values
            )
            low_vals = (
                trendline_slice["Low"].astype(float).ffill().bfill().fillna(0).values
            )
            close_vals = (
                trendline_slice[diff_col].astype(float).ffill().bfill().fillna(0).values
            )

            try:
                # Use original fit_trendlines_high_low function
                support_coefs, resist_coefs = fit_trendlines_high_low(
                    high_vals, low_vals, close_vals
                )

                # Calculate trendline values at the end of the slice
                support_slope, support_intercept = support_coefs
                resist_slope, resist_intercept = resist_coefs
                support_val_at_end = (
                    support_slope * (len(trendline_slice) - 1) + support_intercept
                )
                resist_val_at_end = (
                    resist_slope * (len(trendline_slice) - 1) + resist_intercept
                )

                df.at[timestamp, "Support_Trendline_Value"] = support_val_at_end
                df.at[timestamp, "Resist_Trendline_Value"] = resist_val_at_end

            except Exception as e:
                print(f"Warning: Trendline calculation failed for row {i}: {e}")
                continue
            finally:
                # 🧹 CLEANUP: Current loop data
                if 'trendline_slice' in locals(): del trendline_slice
                if 'high_vals' in locals(): del high_vals
                if 'low_vals' in locals(): del low_vals
                if 'close_vals' in locals(): del close_vals
        
        gc.collect()

        # Calculate trendline differences using diff_col
        df["Support_Trendline_Diff"] = df["Support_Trendline_Value"] - df[diff_col]
        df["Resist_Trendline_Diff"] = df["Resist_Trendline_Value"] - df[diff_col]

        return df

    def _calculate_heikinashi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Heikin-Ashi patterns"""
        # Initialize Heikin-Ashi columns
        ha_cols = [
            "HA_Flat_Bottom",
            "HA_Flat_Top",
            "HA_Small_Body",
            "HA_Candle",
            "HA_Reversal",
            "HA_Lower_Wick",
            "HA_Upper_Wick",
        ]
        df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=[c for c in ha_cols if c not in df.columns])], axis=1)

        try:
            heikinashi_data = heikinashi(df.copy())
            if heikinashi_data is not None and not heikinashi_data.empty:
                column_mapping = {
                    'flat_bottom': 'HA_Flat_Bottom',
                    'flat_top': 'HA_Flat_Top',
                    'small_body': 'HA_Small_Body',
                    'candle': 'HA_Candle',
                    'reversal': 'HA_Reversal',
                    'lower_wick': 'HA_Lower_Wick',
                    'upper_wick': 'HA_Upper_Wick',
                }
                
                for source_col, target_col in column_mapping.items():
                    if source_col in heikinashi_data.columns:
                        df[target_col] = heikinashi_data[source_col]
        except Exception as e:
            print(f"Warning: Heikin-Ashi calculation failed: {e}")
        finally:
            # 🧹 CLEANUP: Delete copy
            if 'heikinashi_data' in locals(): del heikinashi_data
            gc.collect()

        return df

    def _calculate_doji(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Doji patterns"""
        if "Doji" not in df.columns:
            df = pd.concat([df, pd.DataFrame({"Doji": np.nan}, index=df.index)], axis=1)
        else:
            df["Doji"] = np.nan

        try:
            doji_result = doji(df.copy())
            if isinstance(doji_result, pd.Series) and not doji_result.empty:
                df["Doji"] = doji_result
        except Exception as e:
            print(f"Warning: Doji calculation failed: {e}")
        finally:
            if 'doji_result' in locals(): del doji_result
            gc.collect()

        return df

    def _calculate_smc_indicators(self, df: pd.DataFrame, reporter: Optional[ProgressReporter] = None) -> pd.DataFrame:
        """Calculate Smart Money Concepts indicators (FVG, Order Blocks, Liquidity)"""
        if reporter:
            reporter.report(68, "Adding SMC Features...", "Detecting Fair Value Gaps (FVG)")
        diff_col = self.config.diff_column

        # Initialize SMC columns
        smc_cols = [
            "SMC_Swing_HighLow",
            "SMC_Swing_Level",
            "SMC_FVG_FVG",
            "SMC_FVG_Top",
            "SMC_FVG_Bottom",
            "SMC_FVG_MitigatedIndex",
            "SMC_OB_OB",
            "SMC_OB_Top",
            "SMC_OB_Bottom",
            "SMC_OB_OBvolume",
            "SMC_OB_Percentage",
            "SMC_OB_MitigatedIndex",
            "SMC_BOS_BOS",
            "SMC_BOS_CHOCH",
            "SMC_BOS_Level",
            "SMC_BOS_BrokenIndex",
            "SMC_Liquidity_Liquidity",
            "SMC_Liquidity_Level",
            "FVG_Diff",
        ]
        missing_smc = [c for c in smc_cols if c not in df.columns]
        if missing_smc:
            df = pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=missing_smc)], axis=1)

        try:
            ohlc_for_smc = df.copy()  # SMC uses full OHLC

            # Smaller datasets need smaller swing windows to detect enough swings
            dataset_size = len(ohlc_for_smc)
            if dataset_size < 200:
                swing_length = 10
            elif dataset_size < 500:
                swing_length = 20
            elif dataset_size < 1000:
                swing_length = 30
            else:
                swing_length = 50  # Default for large datasets
            
            # Calculate Swing Highs/Lows first, as others depend on it
            swing_highs_lows_data = smc.swing_highs_lows(ohlc_for_smc, swing_length=swing_length)

            if swing_highs_lows_data is not None and not swing_highs_lows_data.empty:
                df["SMC_Swing_HighLow"] = swing_highs_lows_data.get("highlow", swing_highs_lows_data.get("HighLow", np.nan))
                df["SMC_Swing_Level"] = swing_highs_lows_data.get("Level", np.nan)

            # Calculate FVG
            if reporter:
                reporter.report(62, "Adding SMC Features...", "Identifying Fair Value Gaps (FVG)")
            fvg_data = smc.fvg(ohlc_for_smc)
            if fvg_data is not None and not fvg_data.empty:
                df["SMC_FVG_FVG"] = fvg_data.get("FVG", np.nan)
                df["SMC_FVG_Top"] = fvg_data.get("Top", np.nan)
                df["SMC_FVG_Bottom"] = fvg_data.get("Bottom", np.nan)
                # MitigatedIndex=0 means "not yet mitigated" — replace sentinel with NaN
                mit_fvg = fvg_data.get("MitigatedIndex", pd.Series(dtype=float))
                df["SMC_FVG_MitigatedIndex"] = mit_fvg.where(mit_fvg > 0, np.nan)

            # Calculate Order Blocks (requires swing_highs_lows_data)
            if swing_highs_lows_data is not None and not swing_highs_lows_data.empty:
                if reporter:
                    reporter.report(65, "Adding SMC Features...", "Detecting Order Blocks (OB)")
                ob_data = smc.ob(ohlc_for_smc, swing_highs_lows_data)
                if ob_data is not None and not ob_data.empty:
                    df["SMC_OB_OB"] = ob_data.get("OB", np.nan)
                    df["SMC_OB_Top"] = ob_data.get("Top", np.nan)
                    df["SMC_OB_Bottom"] = ob_data.get("Bottom", np.nan)
                    df["SMC_OB_OBvolume"] = ob_data.get("OBvolume", np.nan)
                    df["SMC_OB_Percentage"] = ob_data.get("Percentage", np.nan)
                    # MitigatedIndex=0 is the sentinel for "not yet mitigated" (numpy
                    # zero-init). Replace 0 with NaN so downstream code can use notna().
                    mit_ob = ob_data.get("MitigatedIndex", pd.Series(dtype=float))
                    df["SMC_OB_MitigatedIndex"] = mit_ob.where(mit_ob > 0, np.nan)

            # Calculate BOS / CHoCH (requires swing_highs_lows_data)
            if swing_highs_lows_data is not None and not swing_highs_lows_data.empty:
                if reporter:
                    reporter.report(67, "Adding SMC Features...", "Detecting BOS/CHoCH")
                try:
                    bos_data = smc.bos_choch(ohlc_for_smc, swing_highs_lows_data, close_break=True)
                    if bos_data is not None and not bos_data.empty:
                        df["SMC_BOS_BOS"] = bos_data.get("BOS", np.nan)
                        df["SMC_BOS_CHOCH"] = bos_data.get("CHOCH", np.nan)
                        df["SMC_BOS_Level"] = bos_data.get("Level", np.nan)
                        bi = bos_data.get("BrokenIndex", pd.Series(dtype=float))
                        df["SMC_BOS_BrokenIndex"] = bi.where(bi > 0, np.nan)
                except Exception as _bos_e:
                    self.logger.debug(f"BOS/CHoCH detection failed: {_bos_e}")

            # Calculate Liquidity (requires swing_highs_lows_data)
            if swing_highs_lows_data is not None and not swing_highs_lows_data.empty:
                if reporter:
                    reporter.report(68, "Adding SMC Features...", "Mapping Market Liquidity Zones")
                
                # Try multiple range_percent values to find liquidity zones
                # Liquidity detection requires clustered swing points, which may not exist
                # in all datasets. We try progressively larger ranges to find zones.
                liquidity_data = None
                range_percents = [0.01, 0.02, 0.05, 0.10]  # Try increasing ranges only if the default 1% doesn't yield results
                
                for range_pct in range_percents:
                    try:
                        liquidity_data = smc.liquidity(ohlc_for_smc, swing_highs_lows_data, range_percent=range_pct)
                        if liquidity_data is not None and not liquidity_data.empty:
                            # Check if any liquidity zones were actually detected
                            if liquidity_data['Liquidity'].notna().sum() > 0:
                                break  # Found liquidity zones, use this result
                    except Exception as e:
                        self.logger.debug(f"Liquidity detection failed with range_percent={range_pct}: {e}")
                        continue
                
                if liquidity_data is not None and not liquidity_data.empty:
                    df["SMC_Liquidity_Liquidity"] = liquidity_data.get(
                        "Liquidity", np.nan
                    )
                    df["SMC_Liquidity_Level"] = liquidity_data.get("Level", np.nan)
                else:
                    # No liquidity zones detected - this is normal for datasets without
                    # clustered swing points. Leave as NaN (already initialized above).
                    self.logger.debug("No SMC liquidity zones detected - insufficient clustered swing points")

        except Exception as e:
            print(f"Warning: SMC calculation failed: {e}")
        finally:
            if 'ohlc_for_smc' in locals(): del ohlc_for_smc
            if 'swing_highs_lows_data' in locals(): del swing_highs_lows_data
            if 'fvg_data' in locals(): del fvg_data
            if 'ob_data' in locals(): del ob_data
            if 'bos_data' in locals(): del bos_data
            if 'liquidity_data' in locals(): del liquidity_data
            gc.collect()

        # ── Diff calculations: distance from price to each SMC level ─────────────────
        diff_df = {
            "FVG_Diff": df["SMC_FVG_FVG"].ffill() - df[diff_col],
            "SMC_FVG_Top_Diff": df["SMC_FVG_Top"].ffill() - df[diff_col],
            "SMC_FVG_Bottom_Diff": df["SMC_FVG_Bottom"].ffill() - df[diff_col],
            "SMC_OB_Top_Diff": df["SMC_OB_Top"].ffill() - df[diff_col],
            "SMC_OB_Bottom_Diff": df["SMC_OB_Bottom"].ffill() - df[diff_col],
            "SMC_Swing_Level_Diff": df["SMC_Swing_Level"].ffill() - df[diff_col],
            "SMC_Liquidity_Level_Diff": df["SMC_Liquidity_Level"].ffill() - df[diff_col],
        }

        if diff_df:
            for col in list(diff_df):
                if col in df.columns:
                    df = df.drop(columns=[col])
            df = pd.concat([df, pd.DataFrame(diff_df, index=df.index)], axis=1)

        return df

    def _calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend indicator with consolidated single-column output.
        
        Supertrend is a trend-following indicator with two bands (upper/lower).
        Only ONE band is active at a time based on trend direction.
        
        Returns:
            - Supertrend: Boolean (True=bullish, False=bearish)
            - Supertrend_Distance: Signed distance to active band (consolidated feature)
              * Positive = price is "safe" relative to trend
              * Negative = price breached band (potential reversal)
            - Cross_Supertrend: Crossover signals (1=buy, -1=sell, 0=hold)
        """
        high = df["High"]
        low = df["Low"]
        diff_col = self.config.diff_column
        price = df[diff_col]  # Use configured diff column (usually Close)
        
        atr_period = self.config.supertrend_period
        multiplier = self.config.supertrend_multiplier

        # calculate ATR
        price_diffs = [
            high - low,
            high - price.shift(),
            price.shift() - low,
        ]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        # default ATR calculation in supertrend indicator
        atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()

        # HL2 is simply the average of high and low prices
        hl2 = (high + low) / 2
        # upperband and lowerband calculation
        final_upperband = hl2 + (multiplier * atr)
        final_lowerband = hl2 - (multiplier * atr)

        # initialize Supertrend column to True
        supertrend = [True] * len(df)

        for i in range(1, len(df.index)):
            curr, prev = i, i - 1

            # if current price crosses above upperband
            if price.iloc[curr] > final_upperband.iloc[prev]:
                supertrend[curr] = True
            # if current price crosses below lowerband
            elif price.iloc[curr] < final_lowerband.iloc[prev]:
                supertrend[curr] = False
            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]

                # adjustment to the final bands
                if (
                    supertrend[curr] == True
                    and final_lowerband.iloc[curr] < final_lowerband.iloc[prev]
                ):
                    final_lowerband.iloc[curr] = final_lowerband.iloc[prev]
                if (
                    supertrend[curr] == False
                    and final_upperband.iloc[curr] > final_upperband.iloc[prev]
                ):
                    final_upperband.iloc[curr] = final_upperband.iloc[prev]

            # to remove bands according to the trend direction
            if supertrend[curr] == True:
                final_upperband.iloc[curr] = np.nan
            else:
                final_lowerband.iloc[curr] = np.nan

        df["Supertrend"] = supertrend
        df["Supertrend_Lower"] = final_lowerband
        df["Supertrend_Upper"] = final_upperband

        # Single distance to active Supertrend band
        # This replaces the two separate diff columns (Upper_Diff, Lower_Diff)
        # which had sentinel values and were redundant.
        #
        # Interpretation:
        #   Positive = price is "safe" (above support in uptrend, below resistance in downtrend)
        #   Negative = price breached the band (potential reversal signal)
        #   Magnitude = strength of trend / distance from reversal
        df["Supertrend_Distance"] = np.where(
            df["Supertrend"] == True,  # Bullish trend
            df[diff_col] - df["Supertrend_Lower"],   # Distance above support
            df["Supertrend_Upper"] - df[diff_col]    # Distance below resistance
        )

        # Standardized signal column for automatic detection
        # True = Bullish, False = Bearish
        supertrend_signals = np.zeros(len(df))
        for i in range(1, len(df)):
            if supertrend[i] and not supertrend[i - 1]:
                supertrend_signals[i] = 1.0  # Buy
            elif not supertrend[i] and supertrend[i - 1]:
                supertrend_signals[i] = -1.0  # Sell
        
        df["Cross_Supertrend"] = supertrend_signals

        del price_diffs
        del true_range
        del atr
        del hl2
        del final_upperband
        del final_lowerband
        del supertrend
        del supertrend_signals
        gc.collect()

        return df

    def _calculate_pivots_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate peak/valley pivots"""

        def _identify_initial_pivot(X, up_thresh, down_thresh):
            x_0 = X[0]
            max_x, max_t = x_0, 0
            min_x, min_t = x_0, 0
            up_thresh += 1
            down_thresh += 1
            for t in range(1, len(X)):
                x_t = X[t]
                if x_t / min_x >= up_thresh:
                    return -1 if min_t == 0 else 1
                if x_t / max_x <= down_thresh:
                    return 1 if max_t == 0 else -1
                if x_t > max_x:
                    max_x, max_t = x_t, t
                if x_t < min_x:
                    min_x, min_t = x_t, t
            return -1 if x_0 < X[len(X) - 1] else 1

        close = df["Close"].values
        high = df["High"].values
        low = df["Low"].values
        up_thresh = self.config.pivot_up_thresh
        down_thresh = self.config.pivot_down_thresh

        if down_thresh > 0:
            down_thresh = -down_thresh

        initial_pivot = _identify_initial_pivot(close, up_thresh, down_thresh)
        t_n = len(close)
        pivots = np.zeros(t_n, dtype="i1")
        pivots[0] = initial_pivot

        up_thresh += 1
        down_thresh += 1
        trend = -initial_pivot
        last_pivot_t = 0
        last_pivot_x = close[0]

        for t in range(1, t_n):
            if trend == -1:
                x = low[t]
                r = x / last_pivot_x
                if r >= up_thresh:
                    pivots[last_pivot_t] = trend
                    trend = 1
                    last_pivot_x = high[t]
                    last_pivot_t = t
                elif x < last_pivot_x:
                    last_pivot_x = x
                    last_pivot_t = t
            else:
                x = high[t]
                r = x / last_pivot_x
                if r <= down_thresh:
                    pivots[last_pivot_t] = trend
                    trend = -1
                    last_pivot_x = low[t]
                    last_pivot_t = t
                elif x > last_pivot_x:
                    last_pivot_x = x
                    last_pivot_t = t

        if last_pivot_t == t_n - 1:
            pivots[last_pivot_t] = trend
        elif pivots[t_n - 1] == 0:
            pivots[t_n - 1] = trend

        df["Pivots"] = pivots
        df["Pivot_Price"] = np.nan
        df.loc[df["Pivots"] == 1, "Pivot_Price"] = df["High"]
        df.loc[df["Pivots"] == -1, "Pivot_Price"] = df["Low"]
        return df

    def _calculate_structural_range_features(self, df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        """
        Calculate relative price position based on rolling structural range.
        Ensures absolute prices are converted into stationary [0, 1] positions.
        """
        if 'Pivots' not in df.columns: return df
        peaks = df[df['Pivots'] == 1]['High']
        valleys = df[df['Pivots'] == -1]['Low']
        last_peak = peaks.reindex(df.index).ffill()
        last_valley = valleys.reindex(df.index).ffill()
        df['Rolling_Range_High'] = last_peak.rolling(window=window, min_periods=20).max()
        df['Rolling_Range_Low'] = last_valley.rolling(window=window, min_periods=20).min()
        rolling_width = df['Rolling_Range_High'] - df['Rolling_Range_Low']
        denom = rolling_width.replace(0, np.nan)
        for col in ['Open', 'High', 'Low', 'Close']:
            df[f'{col}_Norm'] = (df[col] - df['Rolling_Range_Low']) / denom
        df['Structural_Range_Position'] = df['Close_Norm']
        df['Structural_Range_Width'] = (denom / df['Close']).clip(upper=0.5)
        df['Structure_Established'] = (~df['Rolling_Range_High'].isna()).astype(float)
        return df

    def _add_legacy_aliases(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add andria_indicators.py-compatible column aliases for backward compatibility.
        
        This method handles 57 missing columns from the old file by:
        1. Creating aliases for renamed columns (dash → underscore, case changes)
        2. Creating crossover signal aliases
        3. Creating RSI change lag aliases
        4. Creating aliases for preserved but renamed columns
        
        Returns:
            DataFrame with all legacy aliases added
        """
        # Guard: deduplicate by exact name before touching any column.
        # If duplicates entered (e.g. from a cached currency-indices run that
        # stored un-prefixed TI columns alongside the base-pair TI columns),
        # df[source_col] would return a DataFrame instead of a Series and the
        # assignment df[legacy_name] = df[source_col] would raise ValueError.
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

        aliases = {}
        
        # ────────────────────────────────────────────────────────────────────
        # 1. EMA NOTATION: Dash → Underscore (EMA-8 → EMA_8)
        # ────────────────────────────────────────────────────────────────────
        for period in [8, 10, 12, 18, 21, 24, 32, 64]:
            dash_name = f'EMA-{period}'
            underscore_name = f'EMA_{period}'
            if underscore_name in df.columns and dash_name not in df.columns:
                aliases[dash_name] = underscore_name
        
        # ────────────────────────────────────────────────────────────────────
        # 2. MA NOTATION: Dash → Underscore (MA-25 → MA_25)
        # ────────────────────────────────────────────────────────────────────
        for period in [25, 50, 100, 200]:
            dash_name = f'MA-{period}'
            underscore_name = f'MA_{period}'
            if underscore_name in df.columns and dash_name not in df.columns:
                aliases[dash_name] = underscore_name
        
        # ────────────────────────────────────────────────────────────────────
        # 3. RSI NOTATION: Dash → Underscore (RSI-7 → RSI_7)
        # ────────────────────────────────────────────────────────────────────
        for period in [7, 14]:
            dash_name = f'RSI-{period}'
            underscore_name = f'RSI_{period}'
            if underscore_name in df.columns and dash_name not in df.columns:
                aliases[dash_name] = underscore_name
        
        # ────────────────────────────────────────────────────────────────────
        # 4. BOLLINGER BANDS: Suffix variants (BBL_20_2.0_2.0)
        # ────────────────────────────────────────────────────────────────────
        bb_mappings = [
            ('BBL_20_2.0_2.0', 'BBL_20_2.0'),
            ('BBM_20_2.0_2.0', 'BBM_20_2.0'),
            ('BBU_20_2.0_2.0', 'BBU_20_2.0'),
            ('BBB_20_2.0_2.0', 'BBB_20_2.0'),
            ('BBP_20_2.0_2.0', 'BBP_20_2.0'),
        ]
        for old_name, new_name in bb_mappings:
            if new_name in df.columns and old_name not in df.columns:
                aliases[old_name] = new_name
        
        # ────────────────────────────────────────────────────────────────────
        # 5. MA DIFFERENCES: Case-sensitive and underscore variants
        # ────────────────────────────────────────────────────────────────────
        ma_diff_mappings = [
            ('ma10_diff', 'SMA_10_Diff'),
            ('ma20_diff', 'SMA_20_Diff'),
            ('ma50_diff', 'MA_50_Diff'),
            ('ma100_diff', 'MA_100_Diff'),
        ]
        for old_name, new_name in ma_diff_mappings:
            if new_name in df.columns and old_name not in df.columns:
                aliases[old_name] = new_name
        
        # ────────────────────────────────────────────────────────────────────
        # 6. PRICE PERIOD DIFFERENCES: Capitalization variants
        # ────────────────────────────────────────────────────────────────────
        period_diff_mappings = [
            ('price_longlong_period_diff', 'Price_Long_Long_Period_Diff'),
            ('price_longshort_period_diff', 'Price_Long_Short_Period_Diff'),
            ('price_shortlong_period_diff', 'Price_Short_Long_Period_Diff'),
        ]
        for old_name, new_name in period_diff_mappings:
            if new_name in df.columns and old_name not in df.columns:
                aliases[old_name] = new_name
        
        # ────────────────────────────────────────────────────────────────────
        # 7. MA_200 CHANGES: Underscore before digit (MA_200_Change_0)
        # ────────────────────────────────────────────────────────────────────
        for i in range(5):
            old_name = f'MA_200_Change{i}'
            new_name = f'MA_200_Change_{i}'
            if new_name in df.columns and old_name not in df.columns:
                aliases[old_name] = new_name
        
        # ────────────────────────────────────────────────────────────────────
        # 8. SUPERTREND BANDS: Name variants
        # ────────────────────────────────────────────────────────────────────
        if 'Supertrend_Lower' in df.columns and 'Final Lowerband' not in df.columns:
            aliases['Final Lowerband'] = 'Supertrend_Lower'
        if 'Supertrend_Upper' in df.columns and 'Final Upperband' not in df.columns:
            aliases['Final Upperband'] = 'Supertrend_Upper'
        
        # ────────────────────────────────────────────────────────────────────
        # 9. TRENDLINE VALUES: Naming variants
        # ────────────────────────────────────────────────────────────────────
        trendline_mappings = [
            ('support_trendline_val', 'Support_Trendline_Value'),
            ('resist_trendline_val', 'Resist_Trendline_Value'),
        ]
        for old_name, new_name in trendline_mappings:
            if new_name in df.columns and old_name not in df.columns:
                aliases[old_name] = new_name
        
        # ────────────────────────────────────────────────────────────────────
        # 10. VOLUME CHANGE: Casing variant
        # ────────────────────────────────────────────────────────────────────
        if 'Down_Volume_Change_Pct' in df.columns and 'Dn_volume_change_pct' not in df.columns:
            aliases['Dn_volume_change_pct'] = 'Down_Volume_Change_Pct'
        
        # ────────────────────────────────────────────────────────────────────
        # 11. PIVOT NAMES: Casing variant
        # ────────────────────────────────────────────────────────────────────
        if 'Pivot_Price' in df.columns and 'Pivot Price' not in df.columns:
            aliases['Pivot Price'] = 'Pivot_Price'
        
        # ────────────────────────────────────────────────────────────────────
        # 12. CROSSOVER SIGNALS: Friendly name variants
        # ────────────────────────────────────────────────────────────────────
        crossover_mappings = [
            ('8_above_12', 'EMA8_Above_EMA12'),
            ('12_above_8', 'Cross_EMA8_Above_EMA12'),
            ('12_above_18', 'EMA12_Above_EMA18'),
            ('18_above_12', 'Cross_EMA12_Above_EMA18'),
            ('25_above_50', 'MA25_Above_MA50'),
            ('50_above_25', 'Cross_MA25_Above_MA50'),
            ('50_above_100', 'MA50_Above_MA100'),
            ('100_above_50', 'Cross_MA50_Above_MA100'),
            ('Crossover', 'Short_Above_Long_Crossover'),
        ]
        for old_name, new_name in crossover_mappings:
            if new_name in df.columns and old_name not in df.columns:
                aliases[old_name] = new_name
        
        # ────────────────────────────────────────────────────────────────────
        # 13. RSI CHANGE LAGS: Lag numbering variants (RSI-Change0 → RSI_7_Change_Lag_1)
        # ────────────────────────────────────────────────────────────────────
        rsi_change_mappings = [
            ('RSI-Change0', 'RSI_7_Change_Lag_1'),
            ('RSI-Change1', 'RSI_7_Change_Lag_2'),
            ('RSI-Change2', 'RSI_7_Change_Lag_3'),
            ('RSI-Change3', 'RSI_7_Change_Lag_4'),
        ]
        for old_name, new_name in rsi_change_mappings:
            if new_name in df.columns and old_name not in df.columns:
                aliases[old_name] = new_name
        
        # ────────────────────────────────────────────────────────────────────
        # 14. UP BAR / DOWN BAR 
        # ────────────────────────────────────────────────────────────────────
        # is_up_bar: Lowercase alias of Is_Up_Bar
        if 'Is_Up_Bar' in df.columns and 'is_up_bar' not in df.columns:
            aliases['is_up_bar'] = 'Is_Up_Bar'
        
        # is_down_bar: Lowercase alias of Is_Down_Bar
        if 'Is_Down_Bar' in df.columns and 'is_down_bar' not in df.columns:
            aliases['is_down_bar'] = 'Is_Down_Bar'
        
        # open: Lowercase alias of Open
        if 'Open' in df.columns and 'open' not in df.columns:
            aliases['open'] = 'Open'

        # ────────────────────────────────────────────────────────────────────
        # 15. PRICE VELOCITY
        # ────────────────────────────────────────────────────────────────────
        # price_velocity_bull: Lowercase alias of Price_Velocity_Bull
        if 'Price_Velocity_Bull' in df.columns and 'price_velocity_bull' not in df.columns:
            aliases['price_velocity_bull'] = 'Price_Velocity_Bull'
        
        # price_velocity_bear: Lowercase alias of Price_Velocity_Bear
        if 'Price_Velocity_Bear' in df.columns and 'price_velocity_bear' not in df.columns:
            aliases['price_velocity_bear'] = 'Price_Velocity_Bear'
        
        # price_velocity_net: Lowercase alias of Price_Velocity_Net
        if 'Price_Velocity_Net' in df.columns and 'price_velocity_net' not in df.columns:
            aliases['price_velocity_net'] = 'Price_Velocity_Net'
        # ────────────────────────────────────────────────────────────────────
        # Apply all aliases to DataFrame — single pd.concat instead of N inserts
        # ⚠️ DO NOT revert to loop: df[col] = df[src] inside a loop calls
        # frame.insert N times on an already-fragmented 600+ col DataFrame,
        # causing PerformanceWarning and measurable slowdown. See warning block
        # at top of file.
        # ────────────────────────────────────────────────────────────────────
        new_alias_cols = {
            legacy_name: df[source_col]
            for legacy_name, source_col in aliases.items()
            if source_col in df.columns and legacy_name not in df.columns
        }
        if new_alias_cols:
            df = pd.concat(
                [df, pd.DataFrame(new_alias_cols, index=df.index)],
                axis=1
            )
        
        return df


    # =========================================================================
    # ADVANCED MICROSTRUCTURE INDICATORS
    # These methods compute purely historical features (lookback only).
    # All safe as input features — no future data.
    # =========================================================================

    def _calculate_snr_vix_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Signal-to-Noise Ratio (SNR) and VIX-like volatility index.

        SNR = 20 * log10(Signal_RMS / Noise_RMS)   [dB]
          Signal_RMS = sqrt(mean(ΔClose²))           over snr_window bars
          Noise_RMS  = sqrt(mean(((H−L)/C)²))        over snr_window bars

        VIX_20 = sqrt(mean((H/C−1)² + (C/L−1)²)) × 100   [% annualised proxy]

        Both are purely historical (no lookahead).
        Reference: UNIFIED_PREDICTIVE_BULL_BEAR_STRENGTH_MODEL.md §1.1, §1.2
        """
        window = self.config.snr_window
        new_cols = {}

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]

        # Signal RMS — root-mean-square of close-to-close returns
        delta_close = close.diff()
        signal_rms  = (delta_close ** 2).rolling(window=window).mean() ** 0.5

        # Noise RMS — intrabar range relative to close price
        noise_term = ((high - low) / close.replace(0, np.nan)) ** 2
        noise_rms  = noise_term.rolling(window=window).mean() ** 0.5

        # SNR in dB — guard against zero denominators
        safe_signal = signal_rms.replace(0, np.nan)
        safe_noise  = noise_rms.replace(0, np.nan)
        snr = 20.0 * np.log10((safe_signal / safe_noise).replace(0, np.nan))
        new_cols["SNR"] = self._clean_inf(snr, 0.0)

        # VIX_20 — normalized intrabar volatility index (percentage)
        vix_term = (
            (high  / close.replace(0, np.nan) - 1.0) ** 2 +
            (close / low.replace(0, np.nan)   - 1.0) ** 2
        )
        new_cols["VIX_20"] = self._clean_inf(
            (vix_term.rolling(window=window).mean() ** 0.5 * 100.0),
            0.0,
        )

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Trend Strength via the KAMA Efficiency Ratio.

        EfficiencyRatio = |Close_n - Close_0| / Σ|Close_i - Close_{i-1}|
        TrendStrength   = EfficiencyRatio  clipped to [0, 1]

        Interpretation:
          0.0 = pure noise / no trend
          1.0 = perfectly directional trend (no zigzag)

        Reference: UNIFIED_PREDICTIVE_BULL_BEAR_STRENGTH_MODEL.md §1.3
        """
        window   = self.config.trend_factor_window
        diff_col = self.config.diff_column
        new_cols = {}

        price = df[diff_col]

        # Net directional move over the window
        net_change = price.diff(window).abs()

        # Total bar-by-bar movement (sum of absolute 1-bar changes)
        total_movement = price.diff().abs().rolling(window=window).sum()

        new_cols["Trend_Strength"] = (
            (net_change / total_movement.replace(0, np.nan))
            .clip(0.0, 1.0)
            .fillna(0.0)
        )

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_vsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Sentiment Index (VSI).

        VSI = (BullishVolume − BearishVolume) / (BullishVolume + BearishVolume) × 100

        Bullish candles: Close > Open
        Bearish candles: Close < Open
        Window: config.vsi_window bars (default 20)

        Range: −100 (all bearish vol) to +100 (all bullish vol).
        Reference: UNIFIED_PREDICTIVE_BULL_BEAR_STRENGTH_MODEL.md §1.5
        """
        if "Volume" not in df.columns:
            logger.debug("[TI] _calculate_vsi: no Volume column, skipping")
            return df

        window   = self.config.vsi_window
        new_cols = {}

        is_bull  = (df["Close"] > df["Open"]).astype(float)
        is_bear  = (df["Close"] < df["Open"]).astype(float)
        vol      = df["Volume"].fillna(0.0)

        bull_vol = (vol * is_bull).rolling(window=window).sum()
        bear_vol = (vol * is_bear).rolling(window=window).sum()
        total    = (bull_vol + bear_vol).replace(0, np.nan)

        new_cols["VSI_20"] = self._clean_inf(((bull_vol - bear_vol) / total * 100.0), 0.0)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_candle_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate per-bar candle structure features.

        Exact port of candle_features() from the Dual-Head v6/v7 writing file.
        These feed the CNN branch of the dual-head model as discriminators
        for BEAR/WAIT/BULL classification.

        Output columns (all purely historical, no lookahead):
          Bar_Dir       {−1, 0, +1}   candle direction
          Body_Ratio    [0, 1]        |Close−Open| / range
          Open_Low_DD   [0, 1]        (Open−Low) / range   (lower shadow)
          High_Close_DD [0, 1]        (High−Close) / range (upper shadow)
          Close_Pos     [0, 1]        (Close−Low) / range  (close position)
          Upper_Wick_R  [0, 1]        upper wick / range
          Lower_Wick_R  [0, 1]        lower wick / range
          Ret_5         float         log(Close / Close.shift(5))
          Ret_20        float         log(Close / Close.shift(20))
          Vol_Zscore    float         volume z-score (20-bar)
          Rel_High      [0, 1]        High position in 20-bar rolling range
        """
        o = df["Open"]
        h = df["High"]
        l = df["Low"]
        c = df["Close"]

        rng = (h - l).replace(0, np.nan)  # candle range; NaN for doji-range bars

        new_cols = {}

        # Bar direction: +1 bull, −1 bear, 0 doji
        new_cols["Bar_Dir"] = np.sign(c - o).fillna(0.0)

        # Body size relative to range
        new_cols["Body_Ratio"] = ((c - o).abs() / rng).clip(0.0, 1.0).fillna(0.5)

        # Lower shadow (open-low for bull, analogous for bear)
        new_cols["Open_Low_DD"] = ((o - l) / rng).clip(0.0, 1.0).fillna(0.5)

        # Upper shadow: High minus close (bearish pressure above close)
        new_cols["High_Close_DD"] = ((h - c) / rng).clip(0.0, 1.0).fillna(0.5)

        # Close position within bar range — key directional signal
        new_cols["Close_Pos"] = ((c - l) / rng).clip(0.0, 1.0).fillna(0.5)

        # Upper and lower wicks
        body_top = pd.concat([o, c], axis=1).max(axis=1)
        body_bot = pd.concat([o, c], axis=1).min(axis=1)
        new_cols["Upper_Wick_R"] = ((h - body_top) / rng).clip(0.0, 1.0).fillna(0.0)
        new_cols["Lower_Wick_R"] = ((body_bot - l)  / rng).clip(0.0, 1.0).fillna(0.0)

        # Multi-bar log returns (Ret_5, Ret_20) — diff-class features
        safe_c = c.replace(0, np.nan)
        log_c  = np.log(safe_c)
        ret_5  = self._clean_inf(log_c - log_c.shift(5), 0.0)
        ret_20 = self._clean_inf(log_c - log_c.shift(20), 0.0)
        new_cols["Ret_5"]  = ret_5
        new_cols["Ret_20"] = ret_20

        # Volume z-score (20-bar rolling)
        vol = (
            df["Volume"].fillna(0.0).astype(float)
            if "Volume" in df.columns
            else pd.Series(1.0, index=df.index)
        )
        vol_mean = vol.rolling(20).mean()
        vol_std  = vol.rolling(20).std().replace(0, np.nan)
        new_cols["Vol_Zscore"] = self._clean_inf((vol - vol_mean) / vol_std, 0.0)

        # Relative position of High within its 20-bar rolling range
        h_max = h.rolling(20).max()
        h_min = l.rolling(20).min()
        new_cols["Rel_High"] = (
            ((h - h_min) / (h_max - h_min).replace(0, np.nan))
            .clip(0.0, 1.0)
            .fillna(0.5)
        )

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_candle_bull_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the composite Candle Bull Score.

        Exact port of candle_bull_score() from the Dual-Head v6/v7 writing file.

        score = (
            Body_Ratio          × 0.25 +
            Close_Pos           × 0.20 +
            Lower_Wick_R        × 0.15 +
            (1 − High_Close_DD) × 0.15 +
            Rel_High            × 0.10 +
            RSI_norm            × 0.10 +   (RSI_14 / 100, fallback RSI_7)
            clip(Ret_1/0.05,−1,1)×0.5+0.5 × 0.05
        )
        For bearish candles (Bar_Dir < 0): score = 1 − score
        Final: clip to [0, 1]

        This score serves TWO roles:
          1. An input feature capturing per-bar bull conviction.
          2. The per-bar regime strength signal used by _enrich_with_targets()
             to generate BEAR/WAIT/BULL labels via the dual-head label scheme.

        IMPORTANT: Requires candle structure columns from _calculate_candle_structure()
        AND RSI from _calculate_rsi_indicators(). Both must run first.
        """
        required = [
            "Body_Ratio", "Close_Pos", "Lower_Wick_R",
            "High_Close_DD", "Rel_High", "Bar_Dir",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(
                f"⚠️ [TI] Skipping Candle_Bull_Score — missing structure cols: {missing}. "
                f"Ensure enable_candle_structure=True runs before this step."
            )
            return df

        # RSI normalised to [0, 1]: prefer 14-period, fall back to 7, then 0.5
        if "RSI_14" in df.columns:
            rsi_norm = (df["RSI_14"] / 100.0).clip(0.0, 1.0)
        elif "RSI_7" in df.columns:
            rsi_norm = (df["RSI_7"] / 100.0).clip(0.0, 1.0)
        else:
            rsi_norm = pd.Series(0.5, index=df.index)

        # Single-bar log return, normalised to [−1, 1] window of ±5%
        diff_col = self.config.diff_column
        close    = df[diff_col].replace(0, np.nan)
        ret_1    = np.log(close / close.shift(1).replace(0, np.nan))
        ret_1    = self._clean_inf(ret_1, 0.0)
        ret_norm = (ret_1 / 0.05).clip(-1.0, 1.0)  # normalised to [-1, 1]

        # Weighted combination (exact weights from writing candle_bull_score)
        score = (
            df["Body_Ratio"]            * 0.25 +
            df["Close_Pos"]             * 0.20 +
            df["Lower_Wick_R"]          * 0.15 +
            (1.0 - df["High_Close_DD"]) * 0.15 +
            df["Rel_High"]              * 0.10 +
            rsi_norm                    * 0.10 +
            (ret_norm * 0.5 + 0.5)      * 0.05
        )

        # Invert score for bearish candles
        is_bear = df["Bar_Dir"] < 0
        score   = score.where(~is_bear, 1.0 - score)

        df["Candle_Bull_Score"] = score.clip(0.0, 1.0).fillna(0.5)

        # ── Independent Bear Score ────────────────────────────────────────────────        #
        # Design principles (v2 — corrected):
        #
        #   Body size (Body_Ratio) is DIRECTION-NEUTRAL — a big body means conviction
        #   regardless of direction. We use it the same way in both scores.
        #
        #   The DIRECTIONAL discriminators are:
        #     Bull: Close_Pos HIGH (close near range top), Lower_Wick_R HIGH (buyers absorbed)
        #     Bear: Close_Pos LOW  (close near range bottom), High_Close_DD HIGH (sellers rejected)
        #
        #   NO inversion logic — the score naturally separates:
        #     Strong BULL bar:  Close_Pos≈1, High_Close_DD≈0 → bear_score LOW
        #     Strong BEAR bar:  Close_Pos≈0, High_Close_DD≈1 → bear_score HIGH
        #     DOJI/neutral bar: Close_Pos≈0.5, body small    → both scores ≈ 0.3-0.5
        #
        #   Both scores can be low simultaneously (e.g. tight range inside bar with no
        #   directional follow-through) — this is meaningful: "market is undecided".

        # RSI inverted: low RSI = bear momentum context
        rsi_bear_norm = (1.0 - rsi_norm)

        # Log return inverted: negative return = bear confirmation
        ret_bear_norm = (-ret_norm * 0.5 + 0.5)   # [-1,1] → [1,0] (negative ret → 1.0)

        bear_score = (
            df["Body_Ratio"]            * 0.25 +   # Same as bull: large body = conviction
            (1.0 - df["Close_Pos"])     * 0.20 +   # Close near LOW  = bear pressure (key discriminator)
            df["High_Close_DD"]         * 0.15 +   # Large upper wick = rejection from high (bear signature)
            (1.0 - df["Lower_Wick_R"])  * 0.15 +   # No lower wick = no buyer absorption below
            (1.0 - df["Rel_High"])      * 0.10 +   # High near 20-bar low = bear regime context
            rsi_bear_norm               * 0.10 +   # Low RSI = bear momentum
            ret_bear_norm               * 0.05     # Falling single-bar return = bear confirmation
        )
        # No inversion: score is naturally discriminative
        # A bull bar has Close_Pos≈1 → (1-Close_Pos)≈0 and High_Close_DD≈0, so bear_score is low.
        # A bear bar has Close_Pos≈0 → (1-Close_Pos)≈1 and High_Close_DD≈1, so bear_score is high.

        df["Candle_Bear_Score"] = bear_score.clip(0.0, 1.0).fillna(0.5)

        # ── Pin Bar Dampening of Bull/Bear Conviction ─────────────────────────────
        #
        # A recent pin bar is the market's "warning shot" — price tried to extend but
        # was aggressively rejected. The system should reduce its enthusiasm for the
        # current trend direction when pin bars appear in the last 3 bars.
        #
        # Logic:
        #   • A recent BEAR pin (shooting star) suppresses Candle_Bull_Score:
        #       bull_score *= (1 − suppression_weight × PinBar_Recent_Bear)
        #   • A recent BULL pin (hammer) suppresses Candle_Bear_Score:
        #       bear_score *= (1 − suppression_weight × PinBar_Recent_Bull)
        #
        # suppression_weight = 0.50 → a perfect recent pin reduces conviction by up to 50%
        # The 3-bar recency weighting already handles the decay (bar[-1] = 1.00,
        # bar[-2] = 0.55, bar[-3] = 0.25), so we just multiply in.
        if "PinBar_Recent_Bear" in df.columns and "PinBar_Recent_Bull" in df.columns:
            suppression_weight = 0.50
            pb_bear = df["PinBar_Recent_Bear"].clip(0.0, 1.0).fillna(0.0)
            pb_bull = df["PinBar_Recent_Bull"].clip(0.0, 1.0).fillna(0.0)

            # Bear pin suppresses bull enthusiasm
            df["Candle_Bull_Score"] = (
                df["Candle_Bull_Score"] * (1.0 - suppression_weight * pb_bear)
            ).clip(0.0, 1.0)

            # Bull pin suppresses bear enthusiasm
            df["Candle_Bear_Score"] = (
                df["Candle_Bear_Score"] * (1.0 - suppression_weight * pb_bull)
            ).clip(0.0, 1.0)

        return df

    def _calculate_price_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate directional price velocity (pips per second) for bull and bear regimes.

        DESIGN:
          Velocity answers: "How fast is price moving in the direction of the current regime?"
          It is NOT symmetric — bull velocity measures upward pip progress per unit time,
          bear velocity measures downward pip progress per unit time.

        Three complementary columns:

          Price_Velocity_Bull  [pips/s, ≥ 0]
            Rate of upward price movement over rolling window.
            = sum of positive bar-to-bar price changes / time span of window.
            Normalised by ATR_Pct so the value is regime-relative (not absolute price).

          Price_Velocity_Bear  [pips/s, ≥ 0]
            Rate of downward price movement over rolling window.
            = sum of |negative bar-to-bar price changes| / time span of window.
            Normalised by ATR_Pct.

          Price_Velocity_Net   [−1, 1]
            Signed composite: (Bull_raw − Bear_raw) / (Bull_raw + Bear_raw)
            = direction-weighted velocity balance.
            +1 = all upward movement; −1 = all downward; 0 = balanced chop.

        Inputs used from existing columns:
          Speed_From_Last_Swing_Low   (pip/s from last pivot low  — existing)
          Speed_From_Last_Swing_High  (pip/s from last pivot high — existing)
          ATR                         (normalisation baseline)
          Time_Diff_From_Last_Swing_* (seconds elapsed — for reference)

        Window: config.snr_window (default 20 bars) — same as SNR/VIX for consistency.
        """
        window   = self.config.snr_window
        diff_col = self.config.diff_column
        new_cols = {}

        close    = df[diff_col]
        bar_diff = close.diff()  # bar-to-bar price change (positive=up, negative=down)

        # ── Per-bar directional moves ─────────────────────────────────────────
        bull_moves = bar_diff.clip(lower=0)       # positive changes only
        bear_moves = (-bar_diff).clip(lower=0)    # absolute negative changes only

        # Rolling sum of directional movement over window bars
        bull_sum = bull_moves.rolling(window=window).sum()
        bear_sum = bear_moves.rolling(window=window).sum()

        # ── Time span of the rolling window (seconds) ────────────────────────
        # Use the actual timestamp difference between bar[t] and bar[t-window+1]
        # Falls back to bar_seconds estimate if index is not DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            # Shift index by window bars, compute elapsed seconds
            shifted_index = df.index.to_series().shift(window)
            elapsed_s = (df.index.to_series() - shifted_index).dt.total_seconds()
            elapsed_s = elapsed_s.replace(0, np.nan)
        else:
            # Estimate: assume uniform bar duration from mean gap in Time_Diff columns
            if "Time_Diff_From_Last_Swing_Low" in df.columns:
                mean_bar_s = df["Time_Diff_From_Last_Swing_Low"].replace(0, np.nan).median()
                mean_bar_s = float(mean_bar_s) if pd.notna(mean_bar_s) else 3600.0
            else:
                mean_bar_s = 3600.0  # default: 1-hour bars
            elapsed_s = pd.Series(mean_bar_s * window, index=df.index)

        # ── Raw velocity (pips per second) ───────────────────────────────────
        bull_vel_raw = bull_sum / elapsed_s   # positive: how fast price moves up
        bear_vel_raw = bear_sum / elapsed_s   # positive: how fast price moves down

        # ── Normalise by ATR to get regime-relative velocity ─────────────────
        # ATR represents "typical pip movement per bar", so dividing by ATR gives
        # a dimensionless ratio: "how many ATR units per second of directional movement"
        # We further scale by elapsed_s/window to convert ATR-per-bar → ATR-per-second
        if "ATR" in df.columns:
            atr_per_s = (df["ATR"] / elapsed_s * window).replace(0, np.nan)
            bull_vel_norm = self._clean_inf(bull_vel_raw / atr_per_s, 0.0)
            bear_vel_norm = self._clean_inf(bear_vel_raw / atr_per_s, 0.0)
        else:
            # No ATR — use min-max normalisation across the rolling window
            bull_max = bull_vel_raw.rolling(window * 5).max().replace(0, np.nan)
            bear_max = bear_vel_raw.rolling(window * 5).max().replace(0, np.nan)
            bull_vel_norm = self._clean_inf(bull_vel_raw / bull_max, 0.0)
            bear_vel_norm = self._clean_inf(bear_vel_raw / bear_max, 0.0)

        # Clip normalised velocities to [0, 3] — values above 3 indicate extreme moves
        new_cols["Price_Velocity_Bull"] = bull_vel_norm.clip(0.0, 3.0).fillna(0.0)
        new_cols["Price_Velocity_Bear"] = bear_vel_norm.clip(0.0, 3.0).fillna(0.0)

        # ── Net velocity: signed direction balance ────────────────────────────
        total_vel = (bull_vel_raw + bear_vel_raw).replace(0, np.nan)
        net_vel   = (bull_vel_raw - bear_vel_raw) / total_vel
        new_cols["Price_Velocity_Net"] = self._clean_inf(net_vel, 0.0).clip(-1.0, 1.0)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate normalised volatility regime indicators.

        DESIGN:
          Volatility regime answers: "How much is price moving relative to what's normal?"
          It separates MAGNITUDE (how big) from DIRECTION, using time as the denominator
          so the model understands pace — not just size.

        Four columns:

          Volatility_Regime   [0, 1]
            Combined intrabar + interbar volatility, normalised by rolling percentile.
            0 = dead market (below historical median).
            1 = extreme volatility (at or above 95th percentile of last 252 bars).
            Formula: clip(VIX_20 / VIX_95pct, 0, 1)
            Falls back to ATR_Pct normalised if VIX_20 unavailable.

          Volatility_Expansion [0, 1]
            Detects volatility EXPANSION vs contraction.
            = short_vol / long_vol, where:
              short_vol = rolling std of returns, window=5 bars
              long_vol  = rolling std of returns, window=20 bars
            >1 = expanding (clipped to 1); <1 = contracting.
            Normalised: clip((short/long), 0, 2) / 2

          Volatility_Bull  [0, 1]
            Fraction of total volatility that was UPWARD movement.
            = rolling std of bull_moves / total_std
            High when rallies are volatile (thrust moves); low during quiet uptrends.

          Volatility_Bear  [0, 1]
            Fraction of total volatility that was DOWNWARD movement.
            = rolling std of bear_moves / total_std
            Independent of Volatility_Bull — both can be high simultaneously (chop).

        Inputs from existing columns:
          VIX_20, ATR, ATR_Pct, Historical_Volatility, Candle_Size
        """
        window   = self.config.snr_window   # 20
        diff_col = self.config.diff_column
        new_cols = {}

        close     = df[diff_col]
        bar_diff  = close.diff()
        bull_moves = bar_diff.clip(lower=0)
        bear_moves = (-bar_diff).clip(lower=0)

        # ── Volatility_Regime: how extreme is current vol vs history ──────────
        if "VIX_20" in df.columns:
            vix = df["VIX_20"].replace(0, np.nan)
            # 95th percentile of VIX over a 252-bar lookback (approx 1 trading year)
            vix_95 = vix.rolling(252, min_periods=window).quantile(0.95).replace(0, np.nan)
            vol_regime = (vix / vix_95).clip(0.0, 1.0).fillna(0.5)
        elif "ATR_Pct" in df.columns:
            atr_pct = df["ATR_Pct"].replace(0, np.nan)
            atr_95  = atr_pct.rolling(252, min_periods=window).quantile(0.95).replace(0, np.nan)
            vol_regime = (atr_pct / atr_95).clip(0.0, 1.0).fillna(0.5)
        else:
            std_now = bar_diff.rolling(window).std().replace(0, np.nan)
            std_95  = std_now.rolling(252, min_periods=window).quantile(0.95).replace(0, np.nan)
            vol_regime = (std_now / std_95).clip(0.0, 1.0).fillna(0.5)

        new_cols["Volatility_Regime"] = vol_regime

        # ── Volatility_Expansion: is vol expanding or contracting? ───────────
        short_std = bar_diff.rolling(5).std().replace(0, np.nan)
        long_std  = bar_diff.rolling(window).std().replace(0, np.nan)
        expansion = self._clean_inf(short_std / long_std, np.nan)
        # Map [0, 2] → [0, 1]: 0.5 means short_std == long_std (stable)
        new_cols["Volatility_Expansion"] = (expansion / 2.0).clip(0.0, 1.0).fillna(0.5)

        # ── Directional volatility split ──────────────────────────────────────
        # Std of bull and bear moves independently, then normalise by total std
        bull_std = bull_moves.rolling(window).std().replace(0, np.nan)
        bear_std = bear_moves.rolling(window).std().replace(0, np.nan)
        total_std = bar_diff.abs().rolling(window).std().replace(0, np.nan)

        new_cols["Volatility_Bull"] = self._clean_inf(bull_std / total_std, np.nan).clip(0.0, 1.0).fillna(0.5)
        new_cols["Volatility_Bear"] = self._clean_inf(bear_std / total_std, np.nan).clip(0.0, 1.0).fillna(0.5)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_regime_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-aligned regime speed — pip progress per unit time
        in the DIRECTION of the current trend.

        DESIGN:
          Regime Speed answers: "Given the current trend, how fast is price
          advancing toward its objective?"

          It uses Trend_Strength to gate the signal:
            - In a strong trend (Trend_Strength → 1): speed measures clean directional progress.
            - In chop (Trend_Strength → 0): speed is near zero regardless of pip count,
              because distance / elapsed_time cancels out random walks.

        Four columns:

          Regime_Speed_Bull  [0, 1]
            Speed of upward price advance, gated by trend quality.
            = Price_Velocity_Bull × Trend_Strength
            Normalised to [0, 1] by 95th percentile.

          Regime_Speed_Bear  [0, 1]
            Speed of downward price advance, gated by trend quality.
            = Price_Velocity_Bear × Trend_Strength
            Normalised to [0, 1] by 95th percentile.

          Regime_Speed_Aligned  [0, 1]
            Speed aligned with the Supertrend direction (the "on-regime" speed).
            Uses Supertrend to determine direction:
              Bull regime → uses Regime_Speed_Bull
              Bear regime → uses Regime_Speed_Bear
            This is the primary feature: "how fast is the trend advancing right now?"

          Regime_Speed_Divergence  [−1, 1]
            Signed difference: Regime_Speed_Bull − Regime_Speed_Bear.
            +1 = bull speed dominates; −1 = bear speed dominates; 0 = balance.
            Independent of Volatility_Net: this is purely about trend-aligned pace.

        Requires: Trend_Strength, Price_Velocity_Bull/Bear (from _calculate_price_velocity).
        Optional: Supertrend (for Regime_Speed_Aligned direction gating).
        """
        new_cols = {}

        # ── Guard: require velocity columns ──────────────────────────────────
        if "Price_Velocity_Bull" not in df.columns or "Price_Velocity_Bear" not in df.columns:
            logger.warning(
                "⚠️ [TI] _calculate_regime_speed: Price_Velocity_Bull/Bear missing. "
                "Run _calculate_price_velocity first."
            )
            return df

        vel_bull = df["Price_Velocity_Bull"]
        vel_bear = df["Price_Velocity_Bear"]

        # ── Trend quality gate ────────────────────────────────────────────────
        if "Trend_Strength" in df.columns:
            trend_gate = df["Trend_Strength"].clip(0.0, 1.0)
        else:
            # Fallback: use Candle_Bull_Score as a directional proxy
            trend_gate = df["Candle_Bull_Score"].clip(0.0, 1.0) if "Candle_Bull_Score" in df.columns else pd.Series(0.5, index=df.index)

        # Raw regime speeds (velocity × trend quality)
        raw_bull = vel_bull * trend_gate
        raw_bear = vel_bear * trend_gate

        # ── Normalise by rolling 95th percentile ─────────────────────────────
        window = self.config.snr_window
        hist_window = min(252, len(df))
        # min_periods must not exceed the rolling window size — clamp defensively
        effective_min_periods = min(window, hist_window)
        bull_95 = raw_bull.rolling(hist_window, min_periods=effective_min_periods).quantile(0.95).replace(0, np.nan)
        bear_95 = raw_bear.rolling(hist_window, min_periods=effective_min_periods).quantile(0.95).replace(0, np.nan)

        new_cols["Regime_Speed_Bull"] = (raw_bull / bull_95).clip(0.0, 1.0).fillna(0.0)
        new_cols["Regime_Speed_Bear"] = (raw_bear / bear_95).clip(0.0, 1.0).fillna(0.0)

        # ── Regime-aligned speed (on-trend direction only) ────────────────────
        # Use Supertrend to determine which direction is "on-regime"
        if "Supertrend" in df.columns:
            # Supertrend == True → bull regime → use bull speed
            # Supertrend == False → bear regime → use bear speed
            is_bull_regime = df["Supertrend"].astype(bool)
            new_cols["Regime_Speed_Aligned"] = pd.Series(
                np.where(is_bull_regime,
                         new_cols["Regime_Speed_Bull"],
                         new_cols["Regime_Speed_Bear"]),
                index=df.index
            )
        else:
            # Fallback: use net velocity direction
            net_vel = df.get("Price_Velocity_Net", pd.Series(0.0, index=df.index))
            new_cols["Regime_Speed_Aligned"] = pd.Series(
                np.where(net_vel >= 0,
                         new_cols["Regime_Speed_Bull"],
                         new_cols["Regime_Speed_Bear"]),
                index=df.index
            )

        # ── Divergence: which direction is faster? ────────────────────────────
        denom = (new_cols["Regime_Speed_Bull"] + new_cols["Regime_Speed_Bear"])
        denom = pd.Series(denom, index=df.index).replace(0, np.nan)
        divergence = (
            (pd.Series(new_cols["Regime_Speed_Bull"], index=df.index) -
             pd.Series(new_cols["Regime_Speed_Bear"], index=df.index)) / denom
        )
        new_cols["Regime_Speed_Divergence"] = self._clean_inf(divergence, 0.0).clip(-1.0, 1.0)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_pin_bar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and score Pin Bars (Hammers & Shooting Stars) with ratio-based grading.

        A Pin Bar is a reversal candle with:
          • A long wick (≥ pin_wick_ratio_min of total range) on one side
          • A small body (≤ pin_body_ratio_max of total range)
          • A tiny opposite wick (≤ pin_nose_ratio_max) — classic 'nose' of the pin

        BULL PIN (Hammer): long LOWER wick → buyers violently rejected price downward
            → suppresses trend-following BEAR enthusiasm
        BEAR PIN (Shooting Star): long UPPER wick → sellers violently rejected rally
            → suppresses trend-following BULL enthusiasm

        ── Ratio-based scoring (continuous, not binary) ────────────────────
        Rather than a hard flag, we compute a strength score [0, 1] based on
        how well each candle satisfies the three ratio criteria:
            wick_score  = clip((wick / range − threshold) / (1 − threshold), 0, 1)
            body_score  = clip(1 − body / range / body_max, 0, 1)
            nose_score  = clip(1 − nose / range / nose_max, 0, 1)
            pin_raw     = wick_score × body_score × nose_score          [0, 1]

        ── 3-bar recency window with heavy weighting on bar[-1] ─────────────
        The system MUST see a recent pin and reduce trend enthusiasm accordingly.
        Weighted combination of the last 3 bars (indices -1, -2, -3):
            PinBar_Recent_Bull = w1×pin_bull[t] + w2×pin_bull[t-1] + w3×pin_bull[t-2]
        where w1 > w2 > w3  (config: pin_weight_bar_minus1/2/3)

        ── Key-level proximity amplifier ────────────────────────────────────
        A pin bar at a key structural level (pivot, swing high/low, BB band,
        SMC OB/FVG) is far more significant. We amplify by pin_level_amplifier
        when price is within pin_level_proximity_atr × ATR of such a level.

        Output columns:
          PinBar_Bull        [0, 1]  Raw bull pin score for THIS bar
          PinBar_Bear        [0, 1]  Raw bear pin score for THIS bar
          PinBar_Score       [-1,1]  Signed: PinBar_Bull − PinBar_Bear
          PinBar_Recent_Bull [0, 1]  3-bar recency-weighted bull pin influence
          PinBar_Recent_Bear [0, 1]  3-bar recency-weighted bear pin influence
          PinBar_At_Level    {0, 1}  1 when pin bar coincides with a key level
        """
        cfg = self.config
        new_cols = {}

        o = df["Open"]
        h = df["High"]
        l = df["Low"]
        c = df["Close"]

        rng = (h - l).replace(0, np.nan)  # total candle range

        body_top = pd.concat([o, c], axis=1).max(axis=1)
        body_bot = pd.concat([o, c], axis=1).min(axis=1)

        body      = (body_top - body_bot)          # absolute body size
        lower_wick = body_bot - l                  # lower shadow length
        upper_wick = h - body_top                  # upper shadow length

        # Ratios relative to total range
        body_ratio  = (body       / rng).clip(0.0, 1.0).fillna(0.5)
        lower_ratio = (lower_wick / rng).clip(0.0, 1.0).fillna(0.0)
        upper_ratio = (upper_wick / rng).clip(0.0, 1.0).fillna(0.0)

        wr_min  = cfg.pin_wick_ratio_min    # e.g. 0.60
        br_max  = cfg.pin_body_ratio_max    # e.g. 0.30
        nr_max  = cfg.pin_nose_ratio_max    # e.g. 0.10

        # ── Continuous strength scores ────────────────────────────────────────
        # Body quality: penalise fat bodies — the smaller the body the better
        body_quality = (1.0 - body_ratio / br_max).clip(0.0, 1.0)

        # BULL PIN: long lower wick, short upper wick (nose)
        bull_wick_score = ((lower_ratio - wr_min) / (1.0 - wr_min)).clip(0.0, 1.0)
        bull_nose_score = (1.0 - upper_ratio / nr_max).clip(0.0, 1.0)
        pin_bull_raw = (bull_wick_score * body_quality * bull_nose_score).fillna(0.0)

        # BEAR PIN: long upper wick, short lower wick (nose)
        bear_wick_score = ((upper_ratio - wr_min) / (1.0 - wr_min)).clip(0.0, 1.0)
        bear_nose_score = (1.0 - lower_ratio / nr_max).clip(0.0, 1.0)
        pin_bear_raw = (bear_wick_score * body_quality * bear_nose_score).fillna(0.0)

        # ── Key-level proximity amplifier ─────────────────────────────────────
        atr = df.get("ATR", pd.Series(np.nan, index=df.index)).ffill().fillna(0.0)
        prox_atr = cfg.pin_level_proximity_atr   # e.g. 1.5× ATR = "near a level"
        amp      = cfg.pin_level_amplifier        # e.g. 1.40

        # Collect key levels into a single best-distance series
        level_sources = []
        for col in ["r1", "s1", "r2", "s2", "r3", "s3",
                    "BB_Upper", "BB_Lower",
                    "SMC_OB_Top", "SMC_OB_Bottom",
                    "SMC_FVG_Top", "SMC_FVG_Bottom",
                    "Support_Trendline_Value", "Resist_Trendline_Value",
                    "Supertrend_Lower", "Supertrend_Upper"]:
            if col in df.columns:
                level_sources.append((df[col] - c).abs())

        if level_sources:
            min_dist_to_level = pd.concat(level_sources, axis=1).min(axis=1)
            near_level = (min_dist_to_level <= atr * prox_atr).astype(float)
        else:
            near_level = pd.Series(0.0, index=df.index)

        new_cols["PinBar_At_Level"] = near_level

        # Apply amplifier — but cap so scores stay ≤ 1
        level_factor = 1.0 + near_level * (amp - 1.0)
        pin_bull = (pin_bull_raw * level_factor).clip(0.0, 1.0)
        pin_bear = (pin_bear_raw * level_factor).clip(0.0, 1.0)

        new_cols["PinBar_Bull"] = pin_bull.fillna(0.0)
        new_cols["PinBar_Bear"] = pin_bear.fillna(0.0)
        new_cols["PinBar_Score"] = (pin_bull - pin_bear).clip(-1.0, 1.0).fillna(0.0)

        # ── 3-bar recency window (bar[-1] is heaviest) ────────────────────────
        # bar[-1] = current bar (t), bar[-2] = t-1, bar[-3] = t-2
        w1 = cfg.pin_weight_bar_minus1   # 1.00  current bar
        w2 = cfg.pin_weight_bar_minus2   # 0.55  one bar ago
        w3 = cfg.pin_weight_bar_minus3   # 0.25  two bars ago
        w_total = w1 + w2 + w3

        new_cols["PinBar_Recent_Bull"] = (
            (pin_bull * w1 + pin_bull.shift(1).fillna(0.0) * w2 + pin_bull.shift(2).fillna(0.0) * w3)
            / w_total
        ).clip(0.0, 1.0)

        new_cols["PinBar_Recent_Bear"] = (
            (pin_bear * w1 + pin_bear.shift(1).fillna(0.0) * w2 + pin_bear.shift(2).fillna(0.0) * w3)
            / w_total
        ).clip(0.0, 1.0)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _calculate_reversal_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Reversal Score — probability that the current trend is dying/reversing.
        
        Reversal Score is a 5-component composite that detects exhaustion signals:
        
        Components (with weights):
          1. Momentum Divergence (30%): Price vs momentum direction mismatch
          2. Structure Break (25%): Price breaking structural support/resistance
          3. Volume Clue (20%): Volume pattern anomalies (climax or drying up)
          4. Exhaustion (15%): Overbought/oversold extremes
          5. Velocity Decay (10%): Price momentum slowing down
        
        Output: Reversal_Score [0,1] + 5 sub-component columns
        Higher score = higher probability of trend reversal
        
        Requires: RSI_14, MACD, ATR, Volume, Supertrend (optional)
        """
        new_cols = {}
        
        # Initialize sub-components
        momentum_div = pd.Series(0.0, index=df.index)
        volume_clue = pd.Series(0.0, index=df.index)
        structure_break = pd.Series(0.0, index=df.index)
        exhaustion = pd.Series(0.0, index=df.index)
        velocity_decay = pd.Series(0.0, index=df.index)
        
        lookback = getattr(self.config, 'reversal_lookback', 5)
        
        # ── 1. Momentum Divergence (30%): Price direction != Momentum direction ──
        if 'MACD' in df.columns and 'Close' in df.columns:
            # Price direction (5-bar change)
            price_change = df['Close'].diff(lookback)
            macd_change = df['MACD'].diff(lookback)
            
            # Divergence when signs differ
            divergence = ((price_change > 0) & (macd_change < 0)) | ((price_change < 0) & (macd_change > 0))
            momentum_div = divergence.astype(float)
        
        # ── 2. Structure Break (25%): Price breaking structural levels ──
        if 'ATR' in df.columns and 'Close' in df.columns:
            atr_threshold = getattr(self.config, 'reversal_structure_atr_threshold', 0.5)
            
            # Use Supertrend or pivot levels if available
            if 'Supertrend_Upper' in df.columns and 'Supertrend_Lower' in df.columns:
                upper = df['Supertrend_Upper']
                lower = df['Supertrend_Lower']
                
                # Break when price crosses bands by more than threshold × ATR
                upbreak = (df['Close'] - upper) > (df['ATR'] * atr_threshold)
                downbreak = (lower - df['Close']) > (df['ATR'] * atr_threshold)
                structure_break = (upbreak | downbreak).astype(float)
            elif 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
                # Fallback to Bollinger Bands
                upbreak = df['Close'] > df['BB_Upper']
                downbreak = df['Close'] < df['BB_Lower']
                structure_break = (upbreak | downbreak).astype(float)
        
        # ── 3. Volume Clue (20%): Volume pattern anomalies ──
        if 'Volume' in df.columns:
            vol_ma = df['Volume'].rolling(20, min_periods=1).mean()
            vol_std = df['Volume'].rolling(20, min_periods=1).std()
            
            # Volume spike (climax) or volume drying up
            spike = df['Volume'] > (vol_ma + 2 * vol_std)
            dryup = df['Volume'] < (vol_ma - 0.5 * vol_std)
            volume_clue = (spike | dryup).astype(float)
        
        # ── 4. Exhaustion (15%): RSI extremes ──
        if 'RSI_14' in df.columns:
            overbought = df['RSI_14'] > 70
            oversold = df['RSI_14'] < 30
            exhaustion = (overbought | oversold).astype(float)
        
        # ── 5. Velocity Decay (10%): Price momentum slowing ──
        if 'Close' in df.columns:
            # Calculate velocity (rate of change) and its change
            velocity = df['Close'].diff(lookback)
            velocity_change = velocity.diff(lookback)
            
            # Decay when velocity is decreasing (approaching zero)
            decay = (velocity_change < 0) & (velocity.abs() > 0)
            velocity_decay = decay.astype(float)
        
        # ── 6. Pin Bar Reversal Signal (bonus overlay) ──────────────────────────
        # Pin bars on bar[-1] are the single strongest single-candle reversal cue.
        # We fold them into the reversal score using the already-weighted 3-bar
        # recency series (PinBar_Recent_Bull/Bear). The net pin signal acts as an
        # additive boost that can push the reversal score toward 1.0.
        # Weight: 0.20 — meaningful but subordinate to structural/momentum signals.
        pin_reversal = pd.Series(0.0, index=df.index)
        if 'PinBar_Recent_Bull' in df.columns and 'PinBar_Recent_Bear' in df.columns:
            # Max of bull/bear pin activity: any strong recent pin elevates reversal prob
            pin_reversal = pd.concat([
                df['PinBar_Recent_Bull'],
                df['PinBar_Recent_Bear']
            ], axis=1).max(axis=1).fillna(0.0)
        
        # ── Composite Reversal Score (weighted sum) ──
        # Weights re-normalised to sum to 1.0 after adding pin bar component.
        weights = {
            'momentum_div':   0.25,   # reduced from 0.30 to make room for pin bar
            'structure_break': 0.20,  # reduced from 0.25
            'volume_clue':    0.18,   # reduced from 0.20
            'exhaustion':     0.12,   # reduced from 0.15
            'velocity_decay': 0.05,   # reduced from 0.10
            'pin_reversal':   0.20,   # NEW — pin bar 3-bar weighted signal
        }
        
        reversal_score = (
            momentum_div    * weights['momentum_div'] +
            structure_break * weights['structure_break'] +
            volume_clue     * weights['volume_clue'] +
            exhaustion      * weights['exhaustion'] +
            velocity_decay  * weights['velocity_decay'] +
            pin_reversal    * weights['pin_reversal']
        )
        
        new_cols['Reversal_Score'] = reversal_score.clip(0.0, 1.0).fillna(0.0)
        new_cols['Reversal_Momentum_Div'] = momentum_div.fillna(0.0)
        new_cols['Reversal_Structure_Break'] = structure_break.fillna(0.0)
        new_cols['Reversal_Volume_Clue'] = volume_clue.fillna(0.0)
        new_cols['Reversal_Exhaustion'] = exhaustion.fillna(0.0)
        new_cols['Reversal_Velocity_Decay'] = velocity_decay.fillna(0.0)
        new_cols['Reversal_PinBar'] = pin_reversal.fillna(0.0)   # pin bar sub-component
        
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        return df

    def _calculate_mean_reversion_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Mean Reversion Score — probability that price will snap back to mean.
        
        Mean Reversion Score is a 5-component composite that detects rubber-band stretch:
        
        Components (with weights):
          1. BB Stretch (25%): Distance from Bollinger Band mean
          2. RSI Extreme (25%): Overbought/oversold levels
          3. Volatility Spike (20%): Abnormal volatility expansion
          4. MA Distance (20%): Distance from moving average
          5. TF Deviation (10%): Distance from trend factor baseline
        
        Output: MeanRev_Score [0,1] + 5 sub-component columns
        Higher score = higher probability of mean reversion
        
        Requires: BB_Middle, RSI_14, ATR, EMA_21, Trend_Strength
        """
        new_cols = {}
        
        # Initialize sub-components
        bb_stretch = pd.Series(0.0, index=df.index)
        rsi_extreme = pd.Series(0.0, index=df.index)
        vol_spike = pd.Series(0.0, index=df.index)
        ma_distance = pd.Series(0.0, index=df.index)
        tf_deviation = pd.Series(0.0, index=df.index)
        
        # ── 1. BB Stretch (25%): Normalized distance from BB mean ──
        if all(col in df.columns for col in ['Close', 'BB_Middle', 'BB_Upper', 'BB_Lower']):
            bb_width = df['BB_Upper'] - df['BB_Lower']
            bb_width = bb_width.replace(0, np.nan)
            
            # Normalized distance from middle band
            bb_stretch = ((df['Close'] - df['BB_Middle']).abs() / bb_width).clip(0.0, 1.0).fillna(0.0)
        
        # ── 2. RSI Extreme (25%): How far from 50 (neutral) ──
        if 'RSI_14' in df.columns:
            # Normalized distance from 50 (neutral RSI)
            rsi_extreme = ((df['RSI_14'] - 50).abs() / 50).clip(0.0, 1.0).fillna(0.0)
        
        # ── 3. Volatility Spike (20%): Current vol vs historical ──
        if 'ATR' in df.columns:
            atr_ma = df['ATR'].rolling(20, min_periods=1).mean()
            atr_ma = atr_ma.replace(0, np.nan)
            
            # Current ATR vs average ATR
            vol_spike = (df['ATR'] / atr_ma - 1.0).clip(0.0, 1.0).infer_objects(copy=False).fillna(0.0)
        
        # ── 4. MA Distance (20%): Distance from EMA ──
        # Try EMA_21, fall back to EMA_20, then SMA_20
        ma_col = None
        for col in ['EMA_21', 'EMA_20', 'SMA_20', 'MA']:
            if col in df.columns:
                ma_col = col
                break
        
        if ma_col and 'Close' in df.columns and 'ATR' in df.columns:
            atr_safe = df['ATR'].replace(0, np.nan)
            # ATR-normalized distance from MA
            ma_distance = ((df['Close'] - df[ma_col]).abs() / atr_safe).clip(0.0, 1.0).infer_objects(copy=False).fillna(0.0)
        
        # ── 5. TF Deviation (10%): Distance from trend baseline ──
        if 'Trend_Strength' in df.columns:
            # Trend_Strength near 0 = chop = high reversion probability
            # Trend_Strength near 1 = strong trend = low reversion probability
            tf_deviation = (1.0 - df['Trend_Strength']).clip(0.0, 1.0).fillna(0.0)
        
        # ── Composite Mean Reversion Score (weighted sum) ──
        weights = {
            'bb_stretch': 0.25,
            'rsi_extreme': 0.25,
            'vol_spike': 0.20,
            'ma_distance': 0.20,
            'tf_deviation': 0.10
        }
        
        mean_rev_score = (
            bb_stretch * weights['bb_stretch'] +
            rsi_extreme * weights['rsi_extreme'] +
            vol_spike * weights['vol_spike'] +
            ma_distance * weights['ma_distance'] +
            tf_deviation * weights['tf_deviation']
        )
        
        new_cols['MeanRev_Score'] = mean_rev_score.clip(0.0, 1.0).fillna(0.0)
        new_cols['MeanRev_BB_Stretch'] = bb_stretch
        new_cols['MeanRev_RSI_Extreme'] = rsi_extreme
        new_cols['MeanRev_Volatility_Spike'] = vol_spike
        new_cols['MeanRev_MA_Distance'] = ma_distance
        new_cols['MeanRev_TF_Deviation'] = tf_deviation
        
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        return df

    def _calculate_momentum_delta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Momentum Delta — comparison of current momentum vs N bars ago.
        
        Momentum Delta measures whether momentum is strengthening or weakening:
        
        Outputs:
          Mom_Delta_5:  Current momentum - momentum 5 bars ago
          Mom_Delta_10: Current momentum - momentum 10 bars ago
          Mom_Delta_20: Current momentum - momentum 20 bars ago
        
        Positive delta = momentum strengthening
        Negative delta = momentum weakening
        
        Uses RSI_14 as the momentum proxy (normalized to [0,1], then centered around 0.5)
        
        Requires: RSI_14
        """
        new_cols = {}
        
        if 'RSI_14' not in df.columns:
            logger.warning("⚠️ [TI] _calculate_momentum_delta: RSI_14 missing, cannot calculate momentum deltas")
            # Return zero columns as fallback
            new_cols['Mom_Delta_5'] = pd.Series(0.0, index=df.index)
            new_cols['Mom_Delta_10'] = pd.Series(0.0, index=df.index)
            new_cols['Mom_Delta_20'] = pd.Series(0.0, index=df.index)
        else:
            # Normalize RSI to [0,1] and center around 0 (so 50 RSI = 0 momentum)
            momentum = (df['RSI_14'] - 50) / 50
            
            # Calculate deltas for different lookback periods
            new_cols['Mom_Delta_5'] = (momentum - momentum.shift(5)).fillna(0.0)
            new_cols['Mom_Delta_10'] = (momentum - momentum.shift(10)).fillna(0.0)
            new_cols['Mom_Delta_20'] = (momentum - momentum.shift(20)).fillna(0.0)
        
        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        return df

    def _calculate_footprint_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive scalar footprint features from pre-merged raw tick-derived columns
        (fp_poc, fp_vah, fp_val, fp_delta, fp_imbalance_max, fp_high_vol_rejection,
        fp_data_available). These raw columns must already exist on `df` — they are
        produced by app.core.analysis.footprint.build_footprint_table() and merged
        onto the OHLCV DataFrame BEFORE it reaches calculate_all_indicators().

        If raw fp_* columns are absent (tick data not backfilled for this
        symbol/timeframe), this method logs a warning and returns df unchanged —
        FP_* columns will simply not appear in get_output_columns() output for
        this run (enable_footprint should be False in that case).
        """
        required_raw = ["fp_poc", "fp_vah", "fp_val", "fp_delta",
                         "fp_imbalance_max", "fp_high_vol_rejection", "fp_data_available"]
        missing = [c for c in required_raw if c not in df.columns]
        if missing:
            logger.warning(
                f"⚠️ [TI] Skipping footprint features — missing raw columns: {missing}. "
                f"Merge footprint.build_footprint_table() output before calling calculate_all_indicators(), "
                f"or set enable_footprint=False."
            )
            return df

        diff_col = self.config.diff_column
        window = self.config.footprint_cum_delta_window
        new_cols = {}

        # ── Price-unit diffs (auto-routed to diff bucket by name via "_Diff" suffix) ──
        new_cols["FP_POC_Diff"] = df["fp_poc"] - df[diff_col]
        new_cols["FP_VAH_Diff"] = df["fp_vah"] - df[diff_col]
        new_cols["FP_VAL_Diff"] = df["fp_val"] - df[diff_col]

        # ── Delta / cumulative delta (volume-scale, signed — NOT a price diff) ──
        new_cols["FP_Delta"] = df["fp_delta"].fillna(0.0)
        new_cols["FP_Cum_Delta"] = new_cols["FP_Delta"].rolling(window=window, min_periods=1).sum()

        # ── Imbalance (already [0,1] ratio) ──
        new_cols["FP_Imbalance_Max"] = df["fp_imbalance_max"].clip(0.0, 1.0).fillna(0.0)

        # ── Delta divergence: price extends but delta doesn't confirm (absorption) ──
        lb = self.config.footprint_rejection_lookback
        price_new_high = df[diff_col] > df[diff_col].shift(lb)
        price_new_low  = df[diff_col] < df[diff_col].shift(lb)
        delta_series = pd.Series(new_cols["FP_Delta"], index=df.index)
        weak_up_delta   = delta_series <= delta_series.shift(lb)
        weak_down_delta = delta_series >= delta_series.shift(lb)
        divergence = (price_new_high & weak_up_delta) | (price_new_low & weak_down_delta)
        new_cols["FP_Delta_Divergence"] = divergence.astype(float).fillna(0.0)

        # ── Rejection flag (already binary from ingestion) ──
        new_cols["FP_High_Vol_Rejection"] = df["fp_high_vol_rejection"].fillna(0.0)

        # ── Availability flag — lets the model learn to discount FP_* on bars
        #     where tick data wasn't available (avoids sentinel confusion) ──
        new_cols["FP_Data_Available"] = df["fp_data_available"].fillna(0.0)

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _finalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize the DataFrame: clean up, fill NaNs, and sort columns alphabetically for readability"""

        # Apply legacy aliases for backward compatibility with andria_indicators.py
        # This adds 57 missing columns via aliases (renames, capitalization variants, etc.)
        df = self._add_legacy_aliases(df)

        # Drop duplicate columns first to avoid Series TypeError during coercion
        df = df.loc[:, ~df.columns.duplicated(keep='first')].copy()

        # Clean up time column if temporary
        # 1. Clean up potential duplicate time/date columns
        if "Time" in df.columns:
            for extra in ["time", "Date", "date", "timestamp"]:
                if extra in df.columns:
                    df = df.drop(extra, axis=1)

        # Forward fill specific columns
        numeric_cols = [
            "Price_Diff_From_Last_Swing_Low",
            "Time_Diff_From_Last_Swing_Low",
            "Speed_From_Last_Swing_Low",
            "Price_Diff_From_Last_Swing_High",
            "Time_Diff_From_Last_Swing_High",
            "Speed_From_Last_Swing_High",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        # Coerce all columns to numeric, making non-numeric NaN, then fill
        for col in df.columns:
            if col not in ["Time", "time", "Date", "date", "timestamp", "Timestamp"]:  # Don't coerce time columns
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # ── Smart NaN fill: diff columns use LARGE_DIFF_SENTINEL, others use 0 ──────
        # fillna(0) on a diff column is semantically WRONG: 0 means "price == indicator"
        # (perfect closeness). During warm-up / when an indicator is absent, the correct
        # signal is LARGE_DIFF_SENTINEL — "price has no known relationship to this level".
        #
        # A column is treated as a diff column if:
        #   (a) its name ends with '_Diff', OR
        #   (b) it is in the explicit set of known distance columns below.
        explicit_diff_cols = {
            "Pivot_Diff", "FVG_Diff",
            "MA_100_50_Diff",
            "Short_Period_MA_Diff", "Long_Period_MA_Diff",
            "Price_Short_Period_Diff", "Price_Short_Long_Period_Diff",
            "Price_Long_Short_Period_Diff", "Price_Long_Long_Period_Diff",
            "Short_MA_Diff", "Long_MA_Diff",
            "Price_Diff_From_Last_Swing_Low", "Price_Diff_From_Last_Swing_High",
        }

        def _is_diff_col(col_name: str) -> bool:
            return col_name.endswith("_Diff") or col_name in explicit_diff_cols or col_name.startswith("Diff_")

        diff_cols_list = [c for c in df.columns if _is_diff_col(c)]
        non_diff_num_cols = [
            c for c in df.columns
            if not _is_diff_col(c) and c not in ("Time", "time")
        ]

        if diff_cols_list:
            df[diff_cols_list] = df[diff_cols_list].fillna(self.dynamic_diff_sentinel)
            
        if non_diff_num_cols:
            # ffill level/price columns first, then zero-fill remaining NaNs
            level_keywords = ["Level", "Price", "Top", "Bottom", "Trendline_Value", "pivot", "_SAR", "SMA_", "EMA_", "MA_"]
            level_cols = [c for c in non_diff_num_cols if any(k in c for k in level_keywords) or (len(c) <= 2 and c[0] in ['r', 's'] and c[1:].isdigit())]
            df[level_cols] = df[level_cols].ffill()
            
            # SMC Liquidity columns need special handling for ML readiness
            # When no liquidity zones exist, we use LARGE_DIFF_SENTINEL to indicate
            # "no liquidity zone detected" (not 0, which would mean "price is at liquidity level")
            smc_liquidity_cols = ["SMC_Liquidity_Level", "SMC_Liquidity_Liquidity"]
            for col in smc_liquidity_cols:
                if col in df.columns:
                    # If the entire column is NaN (no liquidity zones detected anywhere),
                    # fill with sentinel value to indicate "feature not applicable"
                    if df[col].isna().all():
                        df[col] = self.dynamic_diff_sentinel
                    else:
                        # If some liquidity zones exist, forward-fill then use sentinel for remaining NaNs
                        df[col] = df[col].ffill().fillna(self.dynamic_diff_sentinel)
            
            # Recalculate SMC_Liquidity_Level_Diff after filling with sentinel
            # This ensures the diff column reflects the sentinel value properly
            if "SMC_Liquidity_Level" in df.columns and "SMC_Liquidity_Level_Diff" in df.columns:
                diff_col = self.config.diff_column
                # When SMC_Liquidity_Level is sentinel, the diff should also be sentinel
                # (indicating "no liquidity zone to compare against")
                df["SMC_Liquidity_Level_Diff"] = np.where(
                    df["SMC_Liquidity_Level"] == self.dynamic_diff_sentinel,
                    self.dynamic_diff_sentinel,
                    df["SMC_Liquidity_Level"] - df[diff_col]
                )
            
            # Zero-fill all other non-diff numeric columns
            cols_to_zero_fill = [c for c in non_diff_num_cols if c not in smc_liquidity_cols]
            if cols_to_zero_fill:
                df[cols_to_zero_fill] = df[cols_to_zero_fill].fillna(0)

        # Sort indicator columns alphabetically (preserve original OHLCV/Time first)
        # We standardize on Title Case: 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'
        original_cols = ["Time", "Open", "High", "Low", "Close", "Volume"]
        
        # Restore Time from index if missing (because of drop=True in _prepare_data)
        if "Time" not in df.columns and isinstance(df.index, pd.DatetimeIndex):
            time_df = pd.DataFrame({"Time": df.index}, index=df.index)
            df = pd.concat([df, time_df], axis=1)
        
        # Ensure Time column is always datetime type if present
        if "Time" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["Time"]):
                try:
                    df["Time"] = pd.to_datetime(df["Time"])
                except Exception:
                    # If conversion fails, keep as-is but log
                    logger.warning(f"Could not convert Time column to datetime: {df['Time'].dtype}")

        original_cols_present = [col for col in original_cols if col in df.columns]
        
        # Deduplicate columns (Case-Insensitive) to prevent "same-caps" issues
        df = df.loc[:, ~df.columns.str.lower().duplicated(keep='first')].copy()
        
        indicator_cols = sorted(
            [col for col in df.columns if col not in original_cols_present]
        )

        # Reorder: original + sorted indicators
        df = df[original_cols_present + indicator_cols]

        return df