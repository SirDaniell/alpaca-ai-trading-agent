# Feature Registration Inventory
## Complete Catalog of Existing Feature Calculation Functions

**Version:** 1.0  
**Status:** Inventory Phase  
**Purpose:** Catalog all existing mutation functions for Feature Provenance System registration  

---

## Overview

This document inventories **ALL** feature calculation functions currently in the pipeline across three main analysis modules:

1. **Technical Indicators** (`technical_indicators.py`)
2. **Astronomical Features** (`astronomical.py`)
3. **SNR Signal Generation** (`signal_generator.py`)

Each function will need to be registered in the Feature Registry with full provenance tracking.

---

## 1. Technical Indicators Module

**Source File:** `Backend/app/core/analysis/technical_indicators.py`  
**Class:** `TechnicalIndicators`  
**Total Methods:** 20+ feature calculation methods

### 1.1 Data Preparation

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_prepare_data()` | Standardizes OHLCV columns | 0 | ✅ Safe | P0 |

### 1.2 Basic Price Features

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_basic_price_features()` | `Price_Change`, `Price_Change_Pct`, `Direction`, `High_Low_Range`, `High_Low_Range_Pct`, `Low_Day_1` through `Low_Day_N`, `High_Day_1` through `High_Day_N`, `Prev_Close_Diff_1` through `Prev_Close_Diff_N` | 1-N days | ✅ Safe (lookback only) | P0 |

**Registration Template:**
```python
registry.register_feature(
    feature_name="Price_Change",
    calculation_function=TechnicalIndicators._calculate_basic_price_features,
    category="technical",
    lookback_period=1,
    required_columns=["Close"],
    description="Bar-to-bar price change (Close[t] - Close[t-1])"
)
```

### 1.3 Moving Averages

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_moving_averages()` | `SMA_10`, `SMA_20`, `SMA_50`, `SMA_100`, `SMA_200`, `SMA_{period}_Diff`, `EMA_8`, `EMA_10`, `EMA_12`, `EMA_18`, `EMA_21`, `EMA_24`, `EMA_32`, `EMA_64`, `EMA_{period}_Diff`, `Short_MA`, `Long_MA`, `MA`, `Short_MA_10`, `Long_MA_25`, etc. | 2-250 bars | ✅ Safe (rolling windows) | P0 |
| `_calculate_ma_differences()` | `Short_Period_MA_Diff`, `Long_Period_MA_Diff`, `Price_Short_Period_Diff`, `Price_Long_Period_Diff`, `MA_100_50_Diff`, `EMA_{period}_Minus_EMA{base}` | Varies | ✅ Safe (derived from MAs) | P1 |

**Key Features:**
- SMA periods: 10, 20, 50, 100, 200, 250
- EMA periods: 8, 10, 12, 18, 21, 24, 32, 64
- All use `.rolling()` with lookback only

**Registration Template:**
```python
for period in [10, 20, 50, 100, 200, 250]:
    registry.register_feature(
        feature_name=f"SMA_{period}",
        calculation_function=TechnicalIndicators._calculate_moving_averages,
        category="technical",
        lookback_period=period,
        required_columns=["Close"],
        description=f"{period}-period Simple Moving Average"
    )
```

### 1.4 RSI Indicators

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_rsi_indicators()` | `RSI_7`, `RSI_14`, `RSI_Change_{period}` | 7-14 bars | ✅ Safe (pandas_ta.rsi) | P0 |

**Registration Template:**
```python
for period in [7, 14]:
    registry.register_feature(
        feature_name=f"RSI_{period}",
        calculation_function=TechnicalIndicators._calculate_rsi_indicators,
        category="technical",
        lookback_period=period,
        required_columns=["Close"],
        description=f"{period}-period Relative Strength Index"
    )
```

### 1.5 Crossover Signals

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_crossovers()` | `Signal_MA_Cross`, `Signal_RSI_Oversold`, `Signal_RSI_Overbought` | 2 bars | ✅ Safe (compares current vs previous) | P1 |

### 1.6 Bollinger Bands

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_bollinger_bands()` | `BB_Upper`, `BB_Middle`, `BB_Lower`, `BB_UpperBand`, `BB_MiddleBand`, `BB_LowerBand`, `BB_Upper_Diff`, `BB_Lower_Diff`, `BB_Mid_Diff`, `BB_Squeeze` | 20 bars (configurable) | ✅ Safe (pandas_ta.bbands) | P0 |

**Registration Template:**
```python
registry.register_feature(
    feature_name="BB_Upper",
    calculation_function=TechnicalIndicators._calculate_bollinger_bands,
    category="technical",
    lookback_period=20,  # config.bb_length
    required_columns=["Close"],
    description="Bollinger Band Upper (SMA + 2*std)"
)
```

### 1.7 MACD

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_macd()` | `MACD_12_26_9`, `MACDh_12_26_9`, `MACDs_12_26_9`, `MACD`, `MACD_Signal`, `MACD_Histogram` | 26 bars (slow EMA) | ✅ Safe (pandas_ta.macd) | P0 |

### 1.8 Supertrend

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_supertrend()` | `Supertrend_Upper`, `Supertrend_Lower`, `Supertrend`, `Supertrend_Distance`, `Signal_Supertrend` | 10 bars (configurable) | ✅ Safe (ATR-based) | P1 |

### 1.9 Other Indicators

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_other_indicators()` | `ATR`, `ATR_Pct`, `Historical_Volatility`, `Historical_Volatility_{length}`, `Parabolic_SAR`, `PSAR_Diff` | 14-20 bars | ✅ Safe (pandas_ta) | P0 |

### 1.10 Volume Indicators

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_volume_indicators()` | `Tick_Volume`, `OBV`, `Is_Up_Bar`, `Is_Down_Bar`, `Bar_Volume_Up`, `Bar_Volume_Down`, `Up_Distance`, `Down_Distance`, `Volume_Change_Pct`, `Up_Volume_Change_Pct`, `Down_Volume_Change_Pct` | 1 bar | ✅ Safe | P1 |
| `_calculate_volume_metrics()` | `Price_Diff_From_Last_Swing_Low`, `Price_Diff_From_Last_Swing_High`, `Time_Diff_From_Last_Swing_Low`, `Time_Diff_From_Last_Swing_High`, `Speed_From_Last_Swing_Low`, `Speed_From_Last_Swing_High` | Varies | ✅ Safe (lookback to last swing) | P2 |

### 1.11 Pivot Points

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_pivot_points()` | `Pivots`, `r1`, `r2`, `r3`, `s1`, `s2`, `s3`, `Pivot_Diff`, `Pivot_R1_Diff`, `Pivot_R2_Diff`, `Pivot_R3_Diff`, `Pivot_S1_Diff`, `Pivot_S2_Diff`, `Pivot_S3_Diff` | 1 bar (previous day) | ✅ Safe (uses previous OHLC) | P1 |

### 1.12 Trendlines

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_trendlines()` | `Support_Trendline_Value`, `Resist_Trendline_Value`, `Support_Trendline_Diff`, `Resist_Trendline_Diff` | 20 bars (configurable) | ✅ Safe (fit_trendlines_high_low) | P2 |

### 1.13 Heikin-Ashi

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_heikinashi()` | `HA_Flat_Bottom`, `HA_Flat_Top`, `HA_Small_Body`, `HA_Candle`, `HA_Reversal`, `HA_Lower_Wick`, `HA_Upper_Wick` | 1 bar | ✅ Safe (heikinashi function) | P2 |

### 1.14 Doji Patterns

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_doji()` | `Doji`, `Doji_Type` | 1 bar | ✅ Safe (doji function) | P2 |

### 1.15 Smart Money Concepts (SMC)

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_smc_indicators()` | `SMC_Swing_HighLow`, `SMC_Swing_Level`, `SMC_Swing_Level_Diff`, `SMC_FVG_FVG`, `SMC_FVG_Top`, `SMC_FVG_Bottom`, `SMC_FVG_Top_Diff`, `SMC_FVG_Bottom_Diff`, `SMC_OB_OB`, `SMC_OB_Top`, `SMC_OB_Bottom`, `SMC_OB_Top_Diff`, `SMC_OB_Bottom_Diff`, `SMC_Liquidity_Liquidity`, `SMC_Liquidity_Level`, `SMC_Liquidity_Level_Diff`, `FVG_Diff` | 3-5 bars | ✅ Safe (smc function) | P2 |

### 1.16 Pivot Series (Peak/Valley Detection)

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_pivots_series()` | `Pivots` (peak/valley markers) | Varies | ✅ Safe (threshold-based detection) | P2 |

### 1.17 Structural Range Features

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `_calculate_structural_range_features()` | `Structural_Range_Position`, `Structural_Range_Width`, `Structure_Established`, `Peak_Freshness`, `Valley_Freshness` | 252 bars (1 year) | ✅ Safe (rolling window) | P2 |

---

## 2. Astronomical Features Module

**Source File:** `Backend/app/core/analysis/astronomy/astronomical.py`  
**Class:** `AstronomicalFeatureGenerator`  
**Total Methods:** 15+ feature calculation methods

### 2.1 Core Calculation Methods

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `calculate_moon_details()` | `Moon_Phase`, `Moon_Type`, `Moon_Illumination`, `Moon_Age_Days`, `Moon_Distance_KM`, `Moon_Angular_Diameter` | 0 (ephemeris calculation) | ✅ Safe (astronomical calculation) | P0 |
| `calculate_zodiac_house()` | `{Body}_House` for all bodies | 0 | ✅ Safe (longitude / 30) | P0 |
| `calculate_aspect()` | `{Body1}_{Body2}_{aspect}`, `{Body1}_{Body2}_{aspect}_orb` | 0 | ✅ Safe (angular separation) | P1 |
| `calculate_synodic_phase()` | `{Body1}_{Body2}_Synodic` | 0 | ✅ Safe (phase calculation) | P1 |
| `calculate_seasonal_proximity()` | `Spring_Equinox_Proximity`, `Summer_Solstice_Proximity`, `Autumn_Equinox_Proximity`, `Winter_Solstice_Proximity` | 0 | ✅ Safe (angular distance) | P1 |

### 2.2 Planetary Positions

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `EphemCalculator.get_position()` | `{Planet}_Position_Deg` for Sun, Moon, Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune, Pluto | 0 | ✅ Safe (PyEphem) | P0 |
| `EphemCalculator.get_speed()` | `{Planet}_Speed` | 1 day | ✅ Safe (position difference) | P1 |
| `EphemCalculator.get_lunar_nodes()` | `North_Node_Position_Deg`, `South_Node_Position_Deg` | 0 | ✅ Safe (IAU formula) | P1 |
| `EphemCalculator.get_angles()` | `MC`, `IC`, `Ascendant`, `Descendant` | 0 | ✅ Safe (sidereal time calculation) | P1 |
| `EphemCalculator.get_lilith()` | `Lilith_Position_Deg` | 0 | ✅ Safe (mean apogee formula) | P2 |
| `EphemCalculator.get_earth_heliocentric()` | `Earth_Heliocentric_Position_Deg`, `Earth_Distance_AU`, `Earth_Declination` | 0 | ✅ Safe (Sun position + 180°) | P2 |

### 2.3 Asteroid Positions

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `EphemCalculator._add_asteroids()` | `Ceres_Position_Deg`, `Pallas_Position_Deg`, `Juno_Position_Deg`, `Vesta_Position_Deg`, `Chiron_Position_Deg` | 0 | ✅ Safe (orbital elements) | P2 |

### 2.4 Aspect Detection

**Generated Features (all planet pairs × all aspects):**
- Major Aspects: `conjunction`, `sextile`, `square`, `trine`, `opposition`
- Minor Aspects (if enabled): `semisextile`, `semisquare`, `quintile`, `sesquisquare`, `biquintile`, `quincunx`

**Example Features:**
- `Sun_Moon_conjunction`, `Sun_Moon_trine`, `Venus_Mars_square`, etc.
- Total: ~450 aspect features (10 planets × 9 other planets × 5 aspects)

### 2.5 Eclipse Detection

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `detect_stellium()` | `Solar_Eclipse_Potential`, `Lunar_Eclipse_Potential`, `Eclipse_Season` | 0 | ✅ Safe (node proximity) | P2 |

### 2.6 Temporal Features

| Method | Features Generated | Lookback | Temporal Safety | Priority |
|--------|-------------------|----------|-----------------|----------|
| `generate_features_for_date()` | `Day_of_Week`, `Day_of_Week_Num`, `Planetary_Day`, `Dominant_Element` | 0 | ✅ Safe (date extraction) | P1 |

**Registration Template:**
```python
registry.register_feature(
    feature_name="Moon_Phase",
    calculation_function=AstronomicalFeatureGenerator.calculate_moon_details,
    category="astronomical",
    lookback_period=0,
    required_columns=[],  # Uses date only
    description="Synodic moon phase (0-1 cycle, 0=New, 0.5=Full)"
)

# Planetary positions
for planet in ["Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]:
    registry.register_feature(
        feature_name=f"{planet}_Position_Deg",
        calculation_function=EphemCalculator.get_position,
        category="astronomical",
        lookback_period=0,
        required_columns=[],
        description=f"Ecliptic longitude of {planet} in degrees (0-360)"
    )
```

---

## 3. SNR Signal Generation Module

**Source File:** `Backend/app/core/analysis/trading/signal_generator.py`  
**Functions:** Multiple analysis functions for signal generation  
**Total Functions:** 15+ feature calculation functions

### 3.1 Movement Analysis Functions

| Function | Features Generated | Lookback | Temporal Safety | Priority |
|----------|-------------------|----------|-----------------|----------|
| `_calculate_basic_movement_metrics()` | `max_favorable_move`, `max_adverse_move`, `final_move`, `max_favorable_pct`, `max_adverse_pct`, `final_move_pct`, `signal_strength` | N/A | ⚠️ **USES FUTURE DATA** | P0 |
| `_find_optimal_entry_exit_points()` | `best_entry_price`, `best_entry_improvement`, `best_entry_candle`, `optimal_exit_price`, `optimal_exit_return`, `optimal_exit_candle` | N/A | ⚠️ **USES FUTURE DATA** | P0 |
| `_calculate_risk_reward_metrics()` | `max_return_potential`, `max_risk_exposure`, `risk_reward_ratio` | N/A | ⚠️ **USES FUTURE DATA** | P0 |

**⚠️ CRITICAL NOTE:** These functions analyze **FUTURE** price movement after a signal. They are used for:
1. **Training label generation** (supervised learning targets)
2. **Backtesting evaluation** (historical performance analysis)
3. **Signal quality scoring** (post-hoc validation)

**They MUST NOT be used as features for prediction** - they represent the outcome we're trying to predict.

### 3.2 Level Interaction Analysis

| Function | Features Generated | Lookback | Temporal Safety | Priority |
|----------|-------------------|----------|-----------------|----------|
| `_analyze_level_interaction_patterns()` | `last_touch_price`, `last_touch_candle`, `touches_after_signal`, `level_respect_score`, `level_break_strength`, `retest_occurred` | N/A | ⚠️ **USES FUTURE DATA** | P0 |

### 3.3 Volume Pattern Analysis

| Function | Features Generated | Lookback | Temporal Safety | Priority |
|----------|-------------------|----------|-----------------|----------|
| `_analyze_movement_volume_patterns()` | `avg_volume_during_move`, `volume_surge_factor`, `volume_consistency` | N/A | ⚠️ **USES FUTURE DATA** | P1 |
| `_analyze_historical_volume_patterns()` | `historical_avg_volume`, `volume_surge_vs_historical`, `up_volume_dominance`, `down_volume_dominance`, `volume_distance_ratio`, `level_touch_volume_avg`, `level_touch_volume_surge`, `pre_signal_volume_trend` | 200 bars | ✅ Safe (lookback only) | P0 |
| `_analyze_level_touch_volume()` | `level_touch_volume_avg`, `level_touch_volume_surge` | Varies | ✅ Safe (historical touches) | P1 |

### 3.4 Candlestick Pattern Analysis

| Function | Features Generated | Lookback | Temporal Safety | Priority |
|----------|-------------------|----------|-----------------|----------|
| `_analyze_candlestick_patterns()` | `doji_count`, `hammer_count`, `shooting_star_count`, `engulfing_bullish_count`, `engulfing_bearish_count`, `spinning_top_count`, `marubozu_count`, `pattern_strength_score`, `reversal_pattern_strength`, `continuation_pattern_strength` | 10 bars | ✅ Safe (lookback window) | P1 |
| `_detect_candlestick_pattern()` | Individual pattern detection | 1 bar | ✅ Safe (current candle only) | P1 |
| `_detect_engulfing_patterns()` | Engulfing pattern detection | 2 bars | ✅ Safe (current + previous) | P1 |

### 3.5 Timing Analysis

| Function | Features Generated | Lookback | Temporal Safety | Priority |
|----------|-------------------|----------|-----------------|----------|
| `_analyze_movement_timing()` | `time_to_max_favorable`, `time_to_max_adverse`, `time_to_target_1pct` | N/A | ⚠️ **USES FUTURE DATA** | P0 |

### 3.6 Volatility Analysis

| Function | Features Generated | Lookback | Temporal Safety | Priority |
|----------|-------------------|----------|-----------------|----------|
| `_analyze_movement_volatility()` | `avg_volatility`, `volatility_surge`, `volatility_consistency` | N/A | ⚠️ **USES FUTURE DATA** | P1 |

### 3.7 Pullback Analysis

| Function | Features Generated | Lookback | Temporal Safety | Priority |
|----------|-------------------|----------|-----------------|----------|
| `_analyze_pullback_patterns()` | `pullback_occurred`, `max_pullback_pct`, `pullback_recovery_candles` | N/A | ⚠️ **USES FUTURE DATA** | P1 |

### 3.8 SNR Level Detection (SAFE Features)

| Function | Features Generated | Lookback | Temporal Safety | Priority |
|----------|-------------------|----------|-----------------|----------|
| `detect_snr_levels_sequential()` | Support/Resistance levels | 50-200 bars | ✅ Safe (lookback only) | P0 |
| `create_clustered_zones_sequential()` | Clustered SNR zones | Varies | ✅ Safe (derived from levels) | P0 |
| `extract_snr_features()` | `snr_level_price`, `snr_level_strength`, `snr_level_touches`, `snr_distance_to_level`, etc. | Varies | ✅ Safe (historical analysis) | P0 |

**Registration Template for SAFE SNR Features:**
```python
registry.register_feature(
    feature_name="snr_distance_to_nearest_support",
    calculation_function=extract_snr_features,
    category="snr",
    lookback_period=200,
    required_columns=["High", "Low", "Close"],
    description="Distance from current price to nearest support level (lookback only)"
)
```

---

## 4. Feature Categories Summary

### 4.1 By Temporal Safety

| Category | Count | Status | Notes |
|----------|-------|--------|-------|
| ✅ **Safe Features** | ~200+ | Ready for registration | Pure lookback, no future bias |
| ⚠️ **Future-Looking** | ~30 | **LABELS ONLY** | Used for training targets, NOT features |
| 🔍 **Needs Review** | ~10 | Manual inspection needed | Complex calculations requiring validation |

### 4.2 By Priority

| Priority | Description | Count | Timeline |
|----------|-------------|-------|----------|
| **P0** | Critical features (OHLCV, basic indicators) | ~80 | Phase 1 (Weeks 1-2) |
| **P1** | Important features (volume, aspects, crossovers) | ~70 | Phase 2 (Weeks 3-4) |
| **P2** | Advanced features (SMC, patterns, asteroids) | ~50 | Phase 3 (Weeks 5-6) |

### 4.3 By Module

| Module | Safe Features | Future-Looking | Total |
|--------|---------------|----------------|-------|
| Technical Indicators | ~120 | 0 | ~120 |
| Astronomical | ~80 | 0 | ~80 |
| SNR Signals | ~20 | ~30 | ~50 |
| **Total** | **~220** | **~30** | **~250** |

---

## 5. Registration Priority Queue

### Phase 1: Foundation (P0 Features)

**Week 1-2: Core Technical Indicators**
1. Basic price features (OHLCV, changes, ranges)
2. Moving averages (SMA, EMA)
3. RSI indicators
4. Bollinger Bands
5. MACD
6. ATR and volatility

**Week 1-2: Core Astronomical Features**
1. Moon phase and details
2. Planetary positions (Sun, Moon, inner planets)
3. Zodiac houses
4. Basic aspects (conjunction, opposition, square, trine)

### Phase 2: Extended Features (P1)

**Week 3-4: Volume and Signals**
1. Volume indicators (OBV, volume changes)
2. Crossover signals
3. Supertrend
4. Pivot points
5. Synodic cycles
6. Seasonal proximity

### Phase 3: Advanced Features (P2)

**Week 5-6: Patterns and Advanced**
1. SMC indicators (FVG, Order Blocks)
2. Candlestick patterns
3. Heikin-Ashi
4. Trendlines
5. Structural range
6. Asteroids and minor aspects

---

## 6. Special Cases: Future-Looking Features

### 6.1 Training Labels (NOT Features)

These functions generate **training labels** for supervised learning. They analyze future price movement to determine if a signal was successful. They MUST be excluded from feature sets:

```python
# ❌ NEVER use as features
FUTURE_LOOKING_LABELS = [
    "max_favorable_move",
    "max_adverse_move",
    "final_move",
    "optimal_exit_price",
    "time_to_max_favorable",
    "touches_after_signal",
    "level_respect_score",
    # ... all movement analysis outputs
]

# ✅ Use ONLY as training targets
y_train = df["max_favorable_pct"]  # Predict this
X_train = df[SAFE_FEATURES]  # Using only these
```

### 6.2 Registration for Labels

Labels should still be registered for audit purposes, but marked as future-looking:

```python
registry.register_feature(
    feature_name="max_favorable_pct",
    calculation_function=_calculate_basic_movement_metrics,
    category="snr_label",
    lookback_period=0,
    required_columns=["High", "Low", "Close"],
    description="Maximum favorable price movement after signal (FUTURE DATA - LABEL ONLY)",
    uses_future_data=True,  # ⚠️ CRITICAL FLAG
    allowed_usage="training_label_only"
)
```

---

## 7. Next Steps

### Immediate Actions

1. **Review this inventory** with the team
2. **Validate temporal safety** for "Needs Review" features
3. **Create registration scripts** for automated bulk registration
4. **Begin Phase 1 registration** (P0 features)

### Registration Script Template

```python
# scripts/register_technical_indicators.py

from app.core.processing.feature_registry import FeatureRegistry
from app.core.analysis.technical_indicators import TechnicalIndicators

def register_all_technical_indicators():
    """Bulk register all technical indicator features."""
    
    registry = FeatureRegistry(db_session)
    
    # Basic price features
    registry.register_feature(
        feature_name="Price_Change",
        calculation_function=TechnicalIndicators._calculate_basic_price_features,
        category="technical",
        lookback_period=1,
        required_columns=["Close"],
        description="Bar-to-bar price change"
    )
    
    # Moving averages
    for period in [10, 20, 50, 100, 200, 250]:
        registry.register_feature(
            feature_name=f"SMA_{period}",
            calculation_function=TechnicalIndicators._calculate_moving_averages,
            category="technical",
            lookback_period=period,
            required_columns=["Close"],
            description=f"{period}-period Simple Moving Average"
        )
    
    # ... continue for all features
    
    print(f"✅ Registered {len(registry.registered_features)} technical indicators")

if __name__ == "__main__":
    register_all_technical_indicators()
```

---

## 8. Validation Checklist

For each feature being registered, verify:

- [ ] Function source file and line number identified
- [ ] Lookback period calculated correctly
- [ ] Required columns documented
- [ ] Temporal safety validated (AST + runtime test)
- [ ] Dependencies identified
- [ ] Description written
- [ ] Test case created
- [ ] Registration script updated

---

## References

- **Technical Indicators:** `Backend/app/core/analysis/technical_indicators.py`
- **Astronomical Features:** `Backend/app/core/analysis/astronomy/astronomical.py`
- **SNR Signals:** `Backend/app/core/analysis/trading/signal_generator.py`
- **Feature Provenance System:** `Backend/app/core/processing/FEATURE_PROVENANCE_SYSTEM.md`

---

**Document Status:** ✅ Ready for Team Review  
**Total Features Inventoried:** ~250  
**Safe Features:** ~220  
**Future-Looking Labels:** ~30  
**Last Updated:** 2026-05-19
