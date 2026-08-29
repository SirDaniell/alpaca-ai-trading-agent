# Competition Signal Architecture Handoff

## Purpose

This document captures the reusable trading intelligence from the current project so a new competition project can borrow the signal logic without copying the dashboard UI or app shell.

The goal is to preserve the real edge: the combination of

- multi-timeframe RSI context,
- SNR structural zoning,
- directional regime features,
- labeled event generation,
- forward-looking target enrichment,
- and a multi-target ML training setup.

The emphasis is on logic extraction, not UI porting.

---

## 1. High-level summary

This codebase is not a simple “indicator app.” It is a signal-generation + learning pipeline.

At a high level, the flow is:

1. Load OHLCV market data.
2. Compute historical technical features.
3. Detect structural support/resistance zones.
4. Detect MTF RSI / divergence states.
5. Label validated market events (`bounce`, `breakout`, `reversal`).
6. Enrich data with forward-looking targets for ML.
7. Train on the enriched dataset and use it for decisioning.

The app already encodes a strong pattern:

> price structure + regime context + liquidity zones + forward targets

This is exactly the stack we want for the Alpaca options competition.

---

## 2. The core reusable assets

### 2.1 Keep for logic reuse

These files are the reusable core and should be ported to a leaner competition repo.

- [Backend/app/core/analysis/technical_indicators.py](../Backend/app/core/analysis/technical_indicators.py#L2873-L3976)
- [Backend/app/core/analysis/trading/signal_generator.py](../Backend/app/core/analysis/trading/signal_generator.py#L867-L1525)
- [Backend/app/core/ml/ml_dataset_preparation.py](../Backend/app/core/ml/ml_dataset_preparation.py#L1041-L1450)
- [Backend/app/core/analysis/support_resistance.py](../Backend/app/core/analysis/support_resistance.py)
- [Backend/app/core/analysis/trendline_automation.py](../Backend/app/core/analysis/trendline_automation.py)
- [Backend/app/core/analysis/candles.py](../Backend/app/core/analysis/candles.py)
- [Backend/app/core/analysis/smc.py](../Backend/app/core/analysis/smc.py)
- [Backend/app/core/analysis/pivots_points.py](../Backend/app/core/analysis/pivots_points.py)
- [Frontend/src/lib/technical/mtf-rsi.ts](../Frontend/src/lib/technical/mtf-rsi.ts#L1-L260)

### 2.2 Do not copy directly

These are app-specific and product UI layers, not reusable strategy intelligence:

- [Frontend/src/components/StockChart.tsx](../Frontend/src/components/StockChart.tsx)
- the dashboard shell and chart orchestration logic
- the live websocket/UI state management around the model
- the application-level session persistence layer

Rule: copy the signal logic, not the product shell.

---

## 3. Architecture: what the app is doing conceptually

The app combines three layers:

### 3.1 Feature generation layer

This is where technical indicators and microstructure features are created.

The main file is:

- [Backend/app/core/analysis/technical_indicators.py](../Backend/app/core/analysis/technical_indicators.py#L2873-L3976)

This file creates features such as:

- SNR / noise ratio features
- volatility regime features
- trend strength
- candle structure and conviction scores
- velocity and regime speed
- pin-bar and reversal scores
- mean reversion scores
- price-volume and footprint context

The file includes explicit comments stating that these features are historical and safe for input features. This is the key design principle.

Example from the indicator file:

```python
# These methods compute purely historical features (lookback only).
# All safe as input features — no future data.
```

This is a strong signal that the feature pipeline is intentionally designed to remain a “clean model input” layer rather than a leakage-prone target pipeline.

### 3.2 Structural event layer

The event layer converts price structure into discrete labels.

Main file:

- [Backend/app/core/analysis/trading/signal_generator.py](../Backend/app/core/analysis/trading/signal_generator.py#L867-L1525)

This file does the following:

1. detects SNR levels and zones,
2. looks for the current candle near a structural level,
3. checks whether the next confirmation window bounces off or breaks the level,
4. records label metadata and ML-ready sequence metadata.

This is the actual “tradeable signal” engine.

### 3.3 Target enrichment layer

The training label generation is centralized in:

- [Backend/app/core/ml/ml_dataset_preparation.py](../Backend/app/core/ml/ml_dataset_preparation.py#L1041-L1450)

This file creates forward-looking labels that are meant to be used as prediction targets, not as inputs.

Example:

```python
# Map target names to their look-ahead periods
# We handle the default targets: Next_Day_Return, Next_3_Day_Return, Next_5_Day_Return
```

and:

```python
self.data[target_col] = (
    self.data[close_col].shift(-period) - self.data[close_col]
) / self.data[close_col]
```

This is a clear, clean logic pattern for returning future relative price moves.

---

## 4. MTF RSI + divergence engine

The chart-side logic is in:

- [Frontend/src/lib/technical/mtf-rsi.ts](../Frontend/src/lib/technical/mtf-rsi.ts#L1-L260)

This file is crucial for understanding the model’s market regime logic.

### 4.1 Core functions

The main reusable logic is:

- `calculateMovingAverage()` — smoothing of RSI values
- `detectPairCrosses()` — bounce detection between signal lines
- `detectMtfRsiCrossSignals()` — multi-timeframe crossover detection
- `calculateWilderRsiDetailed()` — stable RSI calculation using Wilder smoothing

These functions are the core of the multi-timeframe momentum stack.

Example:

```ts
export function calculateWilderRsiDetailed(
    data: CandlestickData[],
    period = MTF_RSI_PERIOD,
): DetailedWilderRsiResult {
```

This is a detailed RSI state machine that tracks internal average gain/loss values and produces a continuous stateful signal.

### 4.2 Why this matters

The app is not using a single flat RSI. It is using a hierarchy of timeframes and signal comparison logic.

This is important for a competition bot because a single-period RSI often fails during regime change, but a multi-timeframe structure provides better context.

### 4.3 Chart integration pattern

The live chart integrates these signals directly in:

- [Frontend/src/components/StockChart.tsx](../Frontend/src/components/StockChart.tsx#L578-L860)

This is the visual context layer that binds together:

- OHLCV data,
- MTF RSI,
- SNR zones,
- divergence visualizers,
- and forecast overlays.

This is conceptually useful but should be treated as a UI wrapper, not a reusable model component.

---

## 5. SNR zone logic and signal classification

The actual structural event labeling is in:

- [Backend/app/core/analysis/trading/signal_generator.py](../Backend/app/core/analysis/trading/signal_generator.py#L867-L1525)

### 5.1 Flow

The signal generator uses the sequence below:

1. detect current SNR levels,
2. create clustered SNR zones,
3. evaluate current price versus zone center,
4. inspect future candles for confirmation,
5. classify the event type,
6. accumulate ML metadata.

The critical part is the event decision logic around the zone and confirmation window.

Example:

```python
if z_type == "support":
    bounce_fraction = (future_candles[low_col].values >= zone_price * 0.995).mean()
    bounced = bounce_fraction >= CONFIRMATION_TOLERANCE
    if bounced and future_candles[close_col].iloc[-1] > current_price:
        signal_data = { "type": "bounce_support", "price": zone_price }
```

and:

```python
elif z_type == "resistance":
    bounce_fraction = (future_candles[high_col].values <= zone_price * 1.005).mean()
    bounced = bounce_fraction >= CONFIRMATION_TOLERANCE
    if bounced and future_candles[close_col].iloc[-1] < current_price:
        signal_data = { "type": "bounce_resistance", "price": zone_price }
```

This is a very useful pattern for a competition project: do not classify on a single candle; confirm on a fraction of the future validation window.

### 5.2 Label interpretation

This makes the labels meaningful:

- `bounce_support`: support holds and price rallies
- `bounce_resistance`: resistance holds and price falls
- `breakout_support`: support breaks and price continues lower
- `breakout_resistance`: resistance breaks and price continues higher

This is an action-oriented market label set, not just a statistical target.

---

## 6. Technical indicators worth borrowing into the new project

The strongest reusable indicator groups from [Backend/app/core/analysis/technical_indicators.py](../Backend/app/core/analysis/technical_indicators.py#L2873-L3976) are the following.

### 6.1 Regime and structure features

- `SNR`
- `VIX_20`
- `Trend_Strength`
- `Volatility_Regime`
- `Regime_Speed_Aligned`
- `Regime_Speed_Divergence`

These capture whether the market is currently in a clean trend, noisy regime, or high-volatility expansion phase.

### 6.2 Microstructure candle features

- `Bar_Dir`
- `Body_Ratio`
- `Open_Low_DD`
- `High_Close_DD`
- `Close_Pos`
- `Candle_Bull_Score`
- `Candle_Bear_Score`

These are excellent for learning directional conviction at the candle level.

Example from the code:

```python
# Exact port of candle_bull_score() from the Dual-Head v6/v7 writing file.
score = (
    df["Body_Ratio"]            * 0.25 +
    df["Close_Pos"]             * 0.20 +
    df["Lower_Wick_R"]          * 0.15 +
    (1.0 - df["High_Close_DD"]) * 0.15 +
    df["Rel_High"]              * 0.10 +
    rsi_norm                    * 0.10 +
    (ret_norm * 0.5 + 0.5)      * 0.05
)
```

This is a strongly interpretable signal.

### 6.3 Directional and exhaustion features

- `Price_Velocity_Bull`
- `Price_Velocity_Bear`
- `Price_Velocity_Net`
- `PinBar_Recent_Bull`
- `PinBar_Recent_Bear`
- `Reversal_Score`
- `MeanRev_Score`

These are exactly the kinds of features that matter in a live options strategy: directional speed, reversal risk, and exhaustion pressure.

---

## 7. Enrich targets: the real training labels

The target logic is in:

- [Backend/app/core/ml/ml_dataset_preparation.py](../Backend/app/core/ml/ml_dataset_preparation.py#L1041-L1450)

This is the part we must not skip when designing the new project.

### 7.1 Basic return targets

The code creates the following default forward return targets:

- `Next_Day_Return`
- `Next_3_Day_Return`
- `Next_5_Day_Return`
- `Next_Day_Direction`

Code pattern:

```python
target_mapping = {
    "Next_Day_Return": 1,
    "Next_3_Day_Return": 3,
    "Next_5_Day_Return": 5
}
```

and:

```python
self.data[target_col] = (
    self.data[close_col].shift(-period) - self.data[close_col]
) / self.data[close_col]
```

This is a clean and useful label design for modeling directional moves.

### 7.2 Advanced targets

When `prepare_advanced_ml_targets` is enabled, the dataset adds many additional targets, including:

- next momentum values: `adv_target_MOM_t_next`, `adv_target_RSI_14_next`, etc.
- OHLCV scalar targets: `adv_target_Open_t1`, `adv_target_High_t2`, ...
- dual-head classification labels
- MFE/MAE targets
- forward log-return targets
- bull/bear strength targets
- velocity and volatility regime targets
- regime speed targets
- structural targets for trendlines / SNR geometry
- next-zone liquidity targets
- SNR sequence targets
- reversal labels

Key example from the code:

```python
# ── Next-Zone Liquidity targets ────────────────────────────────────
# "Price moves from liquidity to liquidity."
next_zone_added = self._compute_next_zone_targets(
    n_future=getattr(self.config, 'next_zone_n_future', 20),
    zone_touch_pct=getattr(self.config, 'next_zone_touch_pct', 0.004),
)
```

and:

```python
# ── SNR Zone Sequence targets ──────────────────────────────────────
# Ordered two-touch prediction: which SNR zone does price reach first
```

This is the richest part of the pipeline and the strongest conceptual fit for options learning.

### 7.3 Why this target stack matters

The app is not trying to predict a single scalar price. It is trying to learn a market regime from a target bundle that includes:

- expected forward return,
- direction and speed,
- local reversal probability,
- which zone is likely to be reached,
- and which regime the price likely continues in.

That is exactly the kind of target stack a competition bot can use to make options decisions.

---

## 8. File-by-file map for the competition project

### Copy / port these as logic modules

1. [Backend/app/core/analysis/technical_indicators.py](../Backend/app/core/analysis/technical_indicators.py)
   - Main feature factory.
   - Keep the historical and regime features.

2. [Backend/app/core/analysis/trading/signal_generator.py](../Backend/app/core/analysis/trading/signal_generator.py)
   - Main event-generation engine.
   - Converts structure + confirmation into labels.

3. [Backend/app/core/ml/ml_dataset_preparation.py](../Backend/app/core/ml/ml_dataset_preparation.py)
   - Main target-generation logic.
   - Use as the label-generation base.

4. [Backend/app/core/analysis/support_resistance.py](../Backend/app/core/analysis/support_resistance.py)
   - SNR level extraction and zone clustering context.

5. [Backend/app/core/analysis/trendline_automation.py](../Backend/app/core/analysis/trendline_automation.py)
   - Structural trends and trendline context.

6. [Backend/app/core/analysis/candles.py](../Backend/app/core/analysis/candles.py)
   - Candle logic and pattern-specific features.

7. [Backend/app/core/analysis/smc.py](../Backend/app/core/analysis/smc.py)
   - Market structure concepts like order blocks / fair value gaps if needed.

8. [Backend/app/core/analysis/pivots_points.py](../Backend/app/core/analysis/pivots_points.py)
   - Pivot-based support/resistance features.

9. [Frontend/src/lib/technical/mtf-rsi.ts](../Frontend/src/lib/technical/mtf-rsi.ts)
   - Borrow the logic for multi-timeframe momentum and cross detection.

### Reuse only conceptually, not as direct code

- [Frontend/src/components/StockChart.tsx](../Frontend/src/components/StockChart.tsx)
- all UI/chart orchestration logic
- app shell and product composition logic

---

## 9. Inferred competition strategy from the code

This codebase strongly suggests the best competition approach is not a simple “buy/sell indicator” bot. The better design is:

1. generate structural and regime features,
2. detect structurally meaningful SNR events,
3. label those events with forward confirmation,
4. train on multi-target future market behavior,
5. convert predictions into a strategy layer that respects options risk and capital constraints.

That is the correct abstraction boundary for the hackathon.

---

## 10. Recommended construction plan for a new competition repo

The new repo should follow this order:

### Step 1 — Data layer

- OHLCV ingestion
- time alignment across multiple timeframes
- clean column normalization

### Step 2 — Feature layer

- run the historical technical indicator pipeline
- compute SNR, RSI, trend, volatility, and candle structure features

### Step 3 — Structural layer

- detect support/resistance and clustered zones
- compute zone proximity and liquidity context

### Step 4 — Event layer

- generate MTF RSI cross and divergence states
- classify bounce vs breakout events using confirmation windows

### Step 5 — Target layer

- compute forward return targets
- compute multi-target structural labels and next-zone targets

### Step 6 — Training layer

- split by time, not random rows
- train on feature matrix and target bundle
- validate on nearby holdout windows

### Step 7 — Strategy layer

- turn model outputs into an options-trading policy
- enforce risk limits, max exposure, and position sizing
- integrate with Alpaca paper trading (not in UI code here)

---

## 11. Bug watchlist and cleanup notes

This codebase contains a number of useful warning signs that should be tracked for later cleanup.

### 11.1 Pin-bar bug candidate

In [Backend/app/core/analysis/technical_indicators.py](../Backend/app/core/analysis/technical_indicators.py#L3494-L3527), there is a suspicious block involving proximity to key levels. The code creates `level_sources` and then references a variable `c` that is not defined in that scope.

This is a real bug candidate and should be reviewed in a later cleanup pass.

### 11.2 Historical pandas warning remediation

The file includes an explicit note about pandas 2.x deprecations and says not to use old patterns such as:

- `fillna(method="ffill")`
- `fillna(method="bfill")`
- deprecated `replace` downcasting patterns

This indicates there were multiple regressions caused by outdated pandas knowledge.

### 11.3 Scope mismatch

The project is engineered as a larger app, not as a minimal competition pipeline. We should keep the signal stack but trim the operational and UI complexity for the new competition build.

---

## 12. Final recommendation

The correct competition move is not to copy the dashboard. It is to extract the intelligence layer and rebuild the project around it.

The winning reusable backbone is:

- indicators
- SNR zones
- MTF RSI/divergence
- event labels
- enriched forward targets
- multi-task modeling

That is the architecture already embedded in this codebase, and it is the correct starting point for a competition-grade options agent.

---

## 13. Agent instruction summary

The next agent should:

1. create a new competition repo skeleton,
2. port the signal feature and target logic, not the chart UI,
3. document the exact flow from OHLCV to model labels,
4. preserve the structure and target design from the current system,
5. reduce complexity only where necessary for the hackathon,
6. keep a bug watchlist for later cleanup.

This document is the intended source of truth for that migration.
