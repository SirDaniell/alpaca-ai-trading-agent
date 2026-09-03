# Signal Generation & Confluence Analysis
## Research Findings from Codebase

---

## 1. Signal Classes (Core Types)

The system generates **4 primary signal types** based on Support/Resistance zone interactions:

### A. **Bounce Signals** (Price reverses at level)
- `Signal_bounce_support`: Price bounces UP from support level
  - **Condition**: Support zone, price bounces (low stays ≥ zone * 0.995), closes ABOVE entry
  - **Interpretation**: Bullish reversal from lower level
  
- `Signal_bounce_resistance`: Price bounces DOWN from resistance level
  - **Condition**: Resistance zone, price bounces (high stays ≤ zone * 1.005), closes BELOW entry
  - **Interpretation**: Bearish reversal from upper level

### B. **Breakout Signals** (Price breaks through level)
- `Signal_breakout_support`: Price breaks DOWN through support
  - **Condition**: Support zone, price breaks (close stays < zone * 1.005 for confirmation period)
  - **Interpretation**: Bearish breakdown below lower level
  
- `Signal_breakout_resistance`: Price breaks UP through resistance
  - **Condition**: Resistance zone, price breaks (close stays > zone * 0.995 for confirmation period)
  - **Interpretation**: Bullish breakout above upper level

---

## 2. Confluence Framework

### A. **Multi-Timeframe (MTF) SNR Confluence**
**Location**: `evaluate_option_expiries.py` lines 270-286

**Definition**: Confluence occurs when support/resistance levels from different timeframes align within a tight price band.

```
CONFLUENCE_PCT = 0.0015  (0.15% or 15 basis points)
```

**Confluence Types** (All checked):
1. **Support-to-Support Confluence**: 15m support ≈ 1h support (within 0.15%)
2. **Resistance-to-Resistance Confluence**: 15m resistance ≈ 1h resistance (within 0.15%)
3. **Support-to-Resistance Confluence**: 15m support ≈ 1h resistance (within 0.15%)
4. **Resistance-to-Support Confluence**: 15m resistance ≈ 1h support (within 0.15%)

**Output**: 
- `mtf_snr_confluence`: Boolean (1.0/0.0) — binary flag indicating ANY confluence exists

### B. **Zone Support Confluence** 
**Location**: `evaluate_option_expiries.py` lines 379-389

**Definition**: Counts how many timeframes (5m, 15m, 1h, 4h) have an active zone near current price.

```
Zone is "active" if distance ≤ 1 ATR
```

**Confluence Counting**:
```python
df["zone_support_confluence"]    = count(5m_active, 15m_active, 1h_active, 4h_active)  [0-4]
df["zone_resistance_confluence"] = count(5m_active, 15m_active, 1h_active, 4h_active)  [0-4]
```

**Interpretation**:
- `0`: No confluence (isolated zone)
- `1`: One timeframe (weak)
- `2`: Two timeframes (moderate)
- `3`: Three timeframes (strong)
- `4`: All four timeframes (very strong confluence)

### C. **Signal Strength & Directional Conviction**
**Location**: `evaluate_option_expiries.py` lines 179-244

**Composite Score** combines:
1. **Directional MA Cross Signal** (asset vs DXY benchmark)
   - Cross signal recency: +1 within N bars after bullish cross, -1 after bearish cross
   - Clipped to [-1, +1] to encode conviction strength
   - Stored in: `cross_index_signal`, `cross_dxy_signal`

2. **Regime State** (4 binary flags, mutually exclusive)
   ```
   regime_strong_asset   = (slow_diff ≥ 0) AND (fast_diff ≥ 0)  → Bullish
   regime_weak_asset     = (slow_diff ≥ 0) AND (fast_diff < 0)  → Weakening bullish
   regime_weak_dxy       = (slow_diff < 0)  AND (fast_diff ≥ 0)  → Weak bearish
   regime_strong_dxy     = (slow_diff < 0)  AND (fast_diff < 0)  → Strong bearish
   ```

3. **MTF RSI Agreement** 
   - 5m RSI, 15m RSI, 1h RSI all point same direction = higher conviction
   - RSI signals stored: `mtf_rsi_asset`, `mtf_rsi_dxy`

### D. **Zone-Specific Signal Strength**
**Location**: `evaluate_option_expiries.py` lines 391-403

**Bounce/Rejection Composite Signals**:
```python
zone_bounce_signal = 
    (_at_support) * (_upward_move) * (_bar_bullish)
    # Price AT support + moving up + bullish bar structure

zone_rejection_signal = 
    (_at_resistance) * (_downward_move) * (1.0 - _bar_bullish)
    # Price AT resistance + moving down + bearish bar structure
```

**Scoring**: Continuous [0, 1] representing signal strength:
- `0.0`: No confluence at zone
- `0.5`: Partial confluence (1-2 factors present)
- `1.0`: Full confluence (all factors present)

---

## 3. Confirmation Logic

### A. **Confirmation Period**
- Default: 5 future bars
- **Critical**: Not ALL bars must satisfy conditions (was too noisy)
- **Tolerance**: `CONFIRMATION_TOLERANCE = 0.80` (80% of bars)

### B. **Bounce Confirmation** 
For Support:
```python
bounce_fraction = (future_candles[low].values >= zone_price * 0.995).mean()
bounced = bounce_fraction >= 0.80  # At least 80% of future bars hold above 99.5% of zone
AND close[-1] > entry_price        # Final close is higher than entry
```

For Resistance:
```python
bounce_fraction = (future_candles[high].values <= zone_price * 1.005).mean()
bounced = bounce_fraction >= 0.80  # At least 80% of future bars hold below 100.5% of zone
AND close[-1] < entry_price        # Final close is lower than entry
```

### C. **Breakout Confirmation**
For Support Breakdown:
```python
breakout = (future_candles[close].values < zone_price * 1.005).mean() >= 0.80
# At least 80% of future bars stay below 100.5% of zone (already broken through)
```

For Resistance Breakout:
```python
breakout = (future_candles[close].values > zone_price * 0.995).mean() >= 0.80
# At least 80% of future bars stay above 99.5% of zone (already broken through)
```

---

## 4. Required Feature Vectors for Strong Naive Signal

A **truly strong signal** should have:

### Minimum Requirements (ALL must be true):
1. ✅ **Zone Confluence ≥ 2** (support or resistance level active on 2+ timeframes)
2. ✅ **MTF SNR Confluence = 1.0** (at least one multi-timeframe level alignment within 0.15%)
3. ✅ **Directional Conviction ≥ 0.60** (signal_strength, from meta-learner)
4. ✅ **Confirmation Window ≥ 80%** (80% of 5 future bars sustain the move)

### Strong Confluence Boost (2+ of these):
- `cross_index_signal` in [-1, -0.5] or [+0.5, +1.0]  (strong recency cross signal)
- `mtf_rsi_asset` AND `mtf_rsi_dxy` both align directionally
- `zone_bounce_signal` OR `zone_rejection_signal` ≥ 0.7  (high zone strength)
- `regime_strong_asset` OR `regime_strong_dxy` = 1.0  (strong regime alignment)

### Very Strong Signals (Rare):
- Zone Confluence = 4 (all timeframes)
- MTF SNR Confluence = 1.0
- Directional Conviction ≥ 0.75
- Confirmation = 100%
- ALL strong boost factors present

---

## 5. Signal Count & Generation Stats

From evaluate_option_expiries.py, tracking per 1000 bars:

```
Signal Counts Track:
  - bounce_support
  - bounce_resistance
  - breakout_support
  - breakout_resistance

Typical Ratio (GLD, 1K bars):
  - Total bounces: ~10-20% of bars
  - Total breakouts: ~5-10% of bars
  - High-confluence signals: ~1-3% of bars (truly strong)
  - Very high confluence: <0.5% of bars (extremely rare)
```

**Key Insight**: Most generated signals are noise-tolerant. Strong signals require 2+ confluence factors.

---

## 6. ML Target Integration

The 4 signal types feed into **25+ ML targets** for training:

### Primary ML Targets (from 21+ context features):
- **Zone Targets (5)**: next_zone_idx, bars_to_zone, distance_to_zone, volume_at_zone, confluence_count
- **Volatility Targets (10)**: regime state, speed, directional persistence, volatility clusters
- **Velocity Targets (6)**: forward momentum, acceleration, deceleration signals
- **CSM Targets (4)**: composite strength metrics per horizon

**Training Flow**:
1. Signal detected (one of 4 types)
2. Movement analyzed for 20-bar lookahead (no future leakage)
3. ML targets computed from movement outcomes
4. Meta-learner trained to predict signal type + 21+ targets jointly
5. Q-learner trained on per-horizon rewards based on predicted targets

---

## 7. Code Locations Summary

| Component | File | Lines |
|-----------|------|-------|
| Signal Types (bounce/breakout) | signal_generator.py | 851-1200 |
| MTF Confluence | evaluate_option_expiries.py | 270-286 |
| Zone Confluence | evaluate_option_expiries.py | 379-389 |
| Zone Bounce/Rejection | evaluate_option_expiries.py | 391-403 |
| Confirmation Logic | signal_generator.py | 1145-1185 |
| Feature Dict Construction | evaluate_option_expiries.py | 63-82 |
| Context Features (21) | evaluate_option_expiries.py | 116-345 |
| ML Target Integration | ml_dataset_preparation.py | 400-600 |

---

## 8. Key Takeaways

1. **Confluence is NOT Binary**: It's a multi-level system (0-4 timeframes, 0.15% MTF alignment)
2. **Strong Signals Are Rare**: Only ~1-3% of generated signals meet high confluence criteria
3. **80% Rule**: Not all confirmation bars need to satisfy conditions; 80% tolerance filters noise
4. **No Single Metric**: Signal strength combines zone, regime, cross, RSI, and momentum alignment
5. **For Kaggle**: Focus on **Zone Confluence ≥ 2** + **Directional Conviction ≥ 0.65** as minimum filter

---

**Research Completed**: 2026-09-02
**Status**: Ready for evaluation and Kaggle submission
