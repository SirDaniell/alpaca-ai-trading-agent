# MTF SNR Signal Alignment Bugfix Design

## Overview

The ML training pipeline in `evaluate_option_expiries.py` contains four related defects that
cause the meta-learner to train on structurally misaligned, mislabelled data. The
`update_real_snr_snapshot()` function ignores its `timeframe` parameter and always uses 5m
columns for S/R detection. `compute_full_context_features()` replaces real per-TF SNR distances
with rolling-quantile approximations and computes MTF confluence against those same proxies.
The Phase 1 training loop records every bar unconditionally, including bars where the HTF RSI
direction directly contradicts the 5m forward-move label.

The fix is minimal and surgical: three targeted changes inside `evaluate_option_expiries.py`
only. No new dependencies, no architectural rework, no changes to `support_resistance.py` or
`real_data_pipeline.py`.

---

## Glossary

- **Bug_Condition (C)**: Any of four conditions that cause structural misalignment between
  the SNR detection, feature computation, or training-sample labelling logic and the HTF
  price data already present in the aligned dataframe.
- **Property (P)**: The desired behavior after the fix — HTF price columns drive HTF SNR
  detection; per-TF distance features appear in the feature vector; confluence uses real zone
  proximity; training samples require HTF bias confirmation.
- **Preservation**: All existing 5m SNR detection, no-lookahead guarantees, graceful missing-
  column fallbacks, feature dict construction, and chronological data-split behaviour must
  remain unchanged.
- **`update_real_snr_snapshot()`**: Function in `evaluate_option_expiries.py` that detects S/R
  levels up to a given bar and pushes a snapshot to `ZoneSnapshotManager`. Currently always
  reads the 5m price columns regardless of the `timeframe` argument.
- **`compute_full_context_features()`**: Function in `evaluate_option_expiries.py` that enriches
  the aligned dataframe with derived features (MTF RSI, DXY divergence, SNR distances,
  confluence). Currently uses rolling-quantile proxies for SNR proximity features.
- **`aligned_df`**: The multi-timeframe dataframe produced by `align_multi_timeframe_datasets()`,
  with columns named `{field}_{tf}` (e.g. `close_15m`, `high_1h`, `low_4h`). Anti-lookahead
  shift is already applied upstream.
- **HTF bias gate**: A pre-recording filter that checks whether the H1/M15 RSI-derived
  directional signal at a bar aligns with the 5m forward-move label before the bar is
  submitted to `meta_learner.record_experience()`.
- **CONFLUENCE_PCT**: `0.0015` (0.15%) — the maximum relative price distance between two zones
  from different timeframes for them to be considered confluent, matching the reference
  `StockChart.tsx` implementation.

---

## Bug Details

### Bug Condition

The bug manifests in four distinct but related code paths, all triggered during the normal
training and feature-computation flow on any bar where HTF price data is available.

**Formal Specification:**

```
FUNCTION isBugCondition(bar, df_aligned, context)
  INPUT: bar index, aligned dataframe, context string identifying which defect path
  OUTPUT: boolean

  IF context == "SNR_TF_IGNORED" THEN
    // update_real_snr_snapshot ignores the timeframe param and always uses 5m columns
    RETURN (df_aligned contains column "high_15m" OR "high_1h" OR "high_4h")
       AND (SNR detection uses "high_5m" column regardless of requested timeframe)

  ELSE IF context == "SNR_PROXIMITY_APPROX" THEN
    // Per-TF SNR distance features are absent; rolling quantile proxies used instead
    RETURN ("snr_dist_support_15m" NOT IN df_aligned.columns)
       AND ("snr_dist_support_1h"  NOT IN df_aligned.columns)

  ELSE IF context == "CONFLUENCE_APPROX" THEN
    // Confluence computed from rolling quantile counts, not real zone proximity
    RETURN (mtf_snr_confluence derived from rolling_quantile_count)

  ELSE IF context == "HTF_GATE_MISSING" THEN
    // Bars with contradicting HTF bias are recorded as directional training samples
    RETURN (fwd_move_12_sign == +1 AND htf_rsi_direction == "bearish")
        OR (fwd_move_12_sign == -1 AND htf_rsi_direction == "bullish")

  END IF
END FUNCTION
```

### Examples

- **SNR TF Ignored**: `update_real_snr_snapshot(df, idx=500, zm, timeframe="1h")` — the
  function renames `high_5m`/`low_5m` to `High`/`Low` and detects S/R pivots from 5m
  candlesticks. The resulting zones represent 5m micro-structure, not the 1h structural
  pivots that the snapshot was supposed to capture.

- **SNR Proximity Approx**: After `compute_full_context_features()` runs, the dataframe
  contains `snr_dist_support` and `snr_dist_resistance` (single rolling-quantile distances)
  but no `snr_dist_support_15m`, `snr_dist_resistance_1h`, etc. The meta-learner feature
  vector therefore never receives per-TF structural proximity signals.

- **Confluence Approx**: `mtf_snr_confluence` for a bar counts how many of four
  rolling-quantile distances are within 1 ATR. A bar near the 20th-percentile low of a
  100-bar window gets `confluence = 1` even when no real 15m or 1h pivot is nearby.

- **HTF Gate Missing**: A bar where `forward_move_12 > 0` (bullish 5m outcome) but
  `rsi_1h < 45` (bearish H1 RSI) is recorded as `direction="bullish"`. The meta-learner
  trains on a label that contradicts the higher-timeframe market structure.

---

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- When a bar index is below the minimum lookback threshold (< 20), `update_real_snr_snapshot()`
  returns immediately without error, exactly as today.
- When an HTF column is absent from the aligned dataframe (e.g. `high_4h` not present because
  4h data was not fetched), the system gracefully skips that timeframe's SNR computation
  without raising an exception.
- `detect_snr_levels_sequential` and `create_clustered_zones_sequential` are called with only
  historical data up to the current bar — the no-lookahead guarantee is fully preserved.
- `_row_to_feature_dict` continues to include all numeric columns already in the row,
  including aligned HTF OHLCV columns; new per-TF SNR distance columns are appended the
  same way.
- The 70/15/15 chronological train/val/test split is unchanged.
- Bars that pass the HTF confirmation gate continue to use `forward_move_12` as the
  multi-horizon outcome label.
- The existing cross-signal features (`cross_index_signal`, `cross_dxy_signal`,
  `cross_index_dxy`, `cross_dxy_symbol`) are derived from the same RSI and divergence logic
  with no changes.

**Scope:**
All logic paths that do NOT involve HTF SNR detection, per-TF distance feature computation,
real confluence detection, or the HTF confirmation gate are completely unaffected by this fix.
This includes:
- The 5m SNR detection path inside `update_real_snr_snapshot()` when called with
  `timeframe="5m"`.
- The DXY synthetic divergence computation in `compute_full_context_features()`.
- The MTF RSI blend computation (`rsi_5m`, `rsi_15m`, `rsi_1h`).
- Phase 1b, Phase 2, Phase 3, and Phase 4 evaluation loops (no changes to those sections).

---

## Hypothesized Root Cause

Based on analysis of the code, the four defects share a common ancestry: the original
implementation was written to bootstrap quickly using only 5m price data, with HTF columns
added later via `align_multi_timeframe_datasets()`. The downstream code was never updated to
consume those HTF columns for structural analysis.

1. **`update_real_snr_snapshot()` column mapping is hardcoded to `high_col`/`low_col`**:
   The function receives `timeframe` as a parameter but uses module-level `high_col`/`low_col`
   variables that are set to `"high_5m"`/`"low_5m"` at the top of
   `evaluate_expiries_for_symbol()`. No branch exists to select the corresponding HTF columns
   when a different timeframe is requested. The `timeframe` argument only controls the snapshot
   label string, not the price columns used.

2. **`compute_full_context_features()` never calls the SNR pipeline**:
   The function computes all features from the aligned dataframe using vectorized operations.
   It calculates `snr_dist_support` and `snr_dist_resistance` using `rolling(100).quantile()`
   — a cheap approximation. There is no call to `detect_snr_levels_sequential` or
   `create_clustered_zones_sequential` inside this function, so no pivot-based distances exist.

3. **`mtf_snr_confluence` counts quantile proximity, not zone proximity**:
   The confluence calculation sums four boolean expressions (`snr_dist_support <= 1.0`, etc.)
   that all reference the same rolling-quantile approximations. There is no cross-TF zone
   comparison and no 0.15% distance threshold logic.

4. **Phase 1 training loop records every bar unconditionally**:
   The recording call `meta_learner.record_experience(... direction="bullish" if row["forward_move_12"] > 0 else "bearish" ...)` 
   has no preceding filter that checks whether `rsi_1h` or `rsi_15m` supports that
   direction. The gate was either never added or was removed during an earlier refactor.

---

## Correctness Properties

Property 1: Bug Condition — HTF SNR Detection Uses Correct Price Columns

_For any_ bar index where `update_real_snr_snapshot()` is called with `timeframe` in
`["15m", "1h", "4h"]` and the corresponding HTF price columns exist in the aligned dataframe,
the fixed function SHALL build the price slice from `high_{tf}`, `low_{tf}`, `close_{tf}`,
`open_{tf}`, and `volume_{tf}` columns rather than the 5m equivalents, so that all detected
S/R levels reflect the structural pivots of the requested timeframe.

**Validates: Requirements 2.1, 2.2**

Property 2: Bug Condition — Per-TF SNR Distance Features Present in Feature Vector

_For any_ aligned dataframe row produced after the fix, the fixed
`compute_full_context_features()` SHALL include columns `snr_dist_support_15m`,
`snr_dist_resistance_15m`, `snr_dist_support_1h`, `snr_dist_resistance_1h`,
`snr_dist_support_4h`, and `snr_dist_resistance_4h`, each containing the ATR-normalised
distance from the current 5m close to the nearest detected pivot level in that timeframe —
replacing the former rolling-quantile single-distance columns for inter-TF analysis.

**Validates: Requirements 2.3**

Property 3: Bug Condition — MTF Confluence Reflects Real Zone Proximity

_For any_ bar in the aligned dataframe, the fixed `compute_full_context_features()` SHALL set
`mtf_snr_confluence` to a value greater than zero if and only if at least one 15m detected
zone and one 1h detected zone have prices within `CONFLUENCE_PCT = 0.0015` (0.15%) of each
other at that bar — the rolling-quantile count approach SHALL be replaced entirely for this
feature.

**Validates: Requirements 2.4**

Property 4: Bug Condition — HTF Confirmation Gate Filters Training Samples

_For any_ bar in the Phase 1 training loop where the `rsi_1h`-derived directional bias is
available and directly contradicts the sign of `forward_move_12`, the fixed training loop SHALL
skip that bar (or record it as `"hold"`) rather than recording it as a `"bullish"` or
`"bearish"` directional sample, so the meta-learner replay buffer contains only HTF-confirmed
directional experiences.

**Validates: Requirements 2.5**

Property 5: Preservation — Non-Buggy Inputs Unaffected

_For any_ input where the bug condition does NOT hold — specifically, 5m SNR detection calls,
bars where HTF RSI aligns with the 5m forward move, graceful-fallback code paths when HTF
columns are absent, and all code paths outside the three targeted change sites — the fixed
code SHALL produce the same result as the original code, preserving all existing behaviour.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6**

---

## Fix Implementation

### Changes Required

Assuming the root cause analysis above is correct:

**File**: `backend/scripts/evaluate_option_expiries.py`

---

**Change 1 — `update_real_snr_snapshot()`: Select HTF columns by timeframe**

**Specific Changes**:
- After the guard `if up_to_idx < 20: return`, resolve the four column names
  (`high_col_tf`, `low_col_tf`, `close_col_tf`, `open_col_tf`, `vol_col_tf`) dynamically:
  - If `timeframe == "5m"` (or the primary TF), use the existing module-level
    `high_col`/`low_col`/etc. variables.
  - Otherwise, construct column names as `f"high_{timeframe}"`, `f"low_{timeframe}"`, etc.
    and check their presence in `df_full.columns`. If absent, return early (preserving
    requirement 3.2).
- Replace the rename mapping `{high_col: "High", low_col: "Low", ...}` with the dynamically
  resolved `{high_col_tf: "High", low_col_tf: "Low", ...}`.
- No other changes to this function — detection call, clustering call, and
  `ZoneSnapshotManager.add_snapshot()` logic remain identical.

---

**Change 2 — `compute_full_context_features()`: Compute per-TF SNR distances and real confluence**

**Specific Changes**:
1. **Per-TF SNR distance columns**: After the existing ATR computation, add a loop over
   `["15m", "1h", "4h"]`. For each TF where `high_{tf}` and `low_{tf}` exist:
   - Extract the HTF high/low arrays from the dataframe.
   - Compute the rolling 20th-percentile low as the proxy support and 80th-percentile high
     as the proxy resistance, using a lookback appropriate to the TF's bar density in the
     5m-aligned frame (e.g. 100 rows for 15m, 30 rows for 1h — each representing roughly
     25h of history).
   - Store ATR-normalised distances as `snr_dist_support_{tf}` and
     `snr_dist_resistance_{tf}`.
   - Default to `10.0` when the column is absent (preserving requirement 3.2).
   
   > **Note**: Full `detect_snr_levels_sequential` over the entire dataframe inside
   > `compute_full_context_features()` would be prohibitively slow (it runs once per bar
   > when called from the training loop). The rolling-quantile approach is retained for
   > the per-row vectorised pass, but is applied per-TF rather than only on 5m data.
   > `detect_snr_levels_sequential` is already called correctly per-bar via
   > `update_real_snr_snapshot()` — Change 1 fixes that path. Change 2 fixes the static
   > feature columns baked into the dataframe.

2. **MTF confluence**: Replace the four-condition `confluence` integer sum with:
   - Extract `snr_dist_support_15m`, `snr_dist_resistance_15m`, `snr_dist_support_1h`,
     `snr_dist_resistance_1h` vectors (computed in step 1).
   - Convert to absolute price distances using ATR: `dist_price = dist_atr * atr`.
   - Define `zone_15m_support_price` and `zone_15m_resistance_price` as
     `close - dist_price_sup_15m` and `close + dist_price_res_15m` (approximate zone
     anchor prices).
   - Similarly for 1h zones.
   - `confluence = (|zone_15m_support - zone_1h_support| / close <= 0.0015)
                 | (|zone_15m_resistance - zone_1h_resistance| / close <= 0.0015)`
     cast to `int` (0 or 1).
   - Store as `mtf_snr_confluence`.

---

**Change 3 — Phase 1 training loop: HTF confirmation gate**

**Specific Changes**:
- Before the `meta_learner.record_experience(...)` call, read `rsi_1h` from the current row
  (it is already present in `aligned_df` via `compute_full_context_features()`).
- Compute `htf_bias`: `"bullish"` if `rsi_1h > 52`, `"bearish"` if `rsi_1h < 48`,
  `"neutral"` otherwise. The 52/48 band avoids hair-trigger flips around the 50 midline.
- Apply gate:
  - If `forward_move_12 > 0` and `htf_bias == "bearish"`: skip (continue).
  - If `forward_move_12 <= 0` and `htf_bias == "bullish"`: skip (continue).
  - All other combinations (aligned, or neutral HTF bias): record as before.
- The `direction` argument to `record_experience()` is unchanged for bars that pass the gate.

---

## Testing Strategy

### Validation Approach

Testing follows a two-phase approach: first surface counterexamples that demonstrate each
defect on the unfixed code, then verify the fix works correctly and that no existing
behaviour regresses.

---

### Exploratory Bug Condition Checking

**Goal**: Confirm each of the four bug conditions is demonstrable on the current (unfixed) code
before writing the fix. If any exploration test passes unexpectedly, re-examine the hypothesis.

**Test Plan**: Write isolated unit tests that call the relevant functions with controlled
inputs and assert the failing behaviour.

**Test Cases**:

1. **SNR TF Ignored — Exploration**:
   Build a minimal aligned dataframe with distinct 5m and 15m price series (e.g. 5m highs
   clustered around 100, 15m highs around 200). Call `update_real_snr_snapshot()` with
   `timeframe="15m"` and `up_to_idx=50`. Assert that the zone prices returned are near 200,
   not near 100. On unfixed code: zones will be near 100 — confirming the bug.

2. **SNR Proximity Approx — Exploration**:
   Call `compute_full_context_features()` on a small aligned dataframe that includes
   `high_15m` and `high_1h` columns. After the call, assert that
   `"snr_dist_support_15m" in df.columns`. On unfixed code: the column will be absent —
   confirming the bug.

3. **Confluence Approx — Exploration**:
   Construct a dataframe where 15m and 1h support zones are more than 1.5% apart. Assert
   that `mtf_snr_confluence` is 0. On unfixed code: the rolling-quantile counter may still
   produce a non-zero value — confirming the bug.

4. **HTF Gate Missing — Exploration**:
   Mock `meta_learner.record_experience()` to capture calls. Run a short slice of the Phase 1
   loop over rows where `forward_move_12 > 0` but `rsi_1h < 48`. Assert that none of those
   rows appear in the recorded experience buffer. On unfixed code: all rows will be recorded —
   confirming the bug.

**Expected Counterexamples**:
- SNR zone prices reflect 5m pivots, not HTF pivots.
- Per-TF SNR distance columns are absent from the feature vector.
- `mtf_snr_confluence` is non-zero even when no real HTF zone overlap exists.
- HTF-contradicting bars appear in the meta-learner replay buffer as directional samples.

---

### Fix Checking

**Goal**: Verify that for all inputs where each bug condition holds, the fixed functions
produce the expected correct behavior.

**Pseudocode:**
```
FOR ALL (bar, df, tf) WHERE isBugCondition_SNR_TF_Ignored(bar, df, tf) DO
  result := update_real_snr_snapshot_fixed(df, bar, zm, tf)
  ASSERT ALL zone IN zm.snapshots: zone.price NEAR df[f"high_{tf}"] pivots
END FOR

FOR ALL df_aligned WHERE isBugCondition_SNR_Proximity_Approx(df_aligned) DO
  df' := compute_full_context_features_fixed(df_aligned)
  ASSERT "snr_dist_support_15m" IN df'.columns
  ASSERT "snr_dist_resistance_1h" IN df'.columns
  ASSERT "mtf_snr_confluence" in df'.columns
  ASSERT df'["mtf_snr_confluence"].dtype == float32
END FOR

FOR ALL bar WHERE isBugCondition_HTFGateMissing(bar, htf_dir, fwd_sign) DO
  record_calls := capture(meta_learner.record_experience)
  run_phase1_loop_fixed_on(bar)
  ASSERT bar NOT IN record_calls
END FOR
```

---

### Preservation Checking

**Goal**: Verify that for all inputs where the bug conditions do NOT hold, the fixed
functions produce the same result as the original functions.

**Pseudocode:**
```
FOR ALL (bar, df) WHERE timeframe == "5m" DO
  // 5m SNR path is unchanged
  ASSERT update_real_snr_snapshot_original(df, bar, zm, "5m")
      == update_real_snr_snapshot_fixed(df, bar, zm, "5m")
END FOR

FOR ALL df_aligned WHERE "high_4h" NOT IN df_aligned.columns DO
  // Graceful fallback: no exception, 4h columns default to 10.0
  df' := compute_full_context_features_fixed(df_aligned)
  ASSERT "snr_dist_support_4h" IN df'.columns
  ASSERT df'["snr_dist_support_4h"].eq(10.0).all()
END FOR

FOR ALL bar WHERE htf_rsi_direction ALIGNS WITH fwd_move_12_sign DO
  // Bar recorded with correct direction, same as before
  ASSERT record_experience_called_with(direction == expected_direction)
END FOR
```

**Testing Approach**: Property-based testing is recommended for the preservation checks
because:
- It generates many combinations of aligned-dataframe shapes and column presence automatically.
- It catches edge cases where a column is present but has NaN values at the boundary.
- It provides strong guarantees that the fallback/no-op paths are truly unchanged.

---

### Unit Tests

- Test `update_real_snr_snapshot()` with each of `"5m"`, `"15m"`, `"1h"`, `"4h"` to verify
  zone prices reflect the correct TF price series.
- Test `update_real_snr_snapshot()` with a missing HTF column — verify no exception, no
  snapshot added.
- Test `compute_full_context_features()` — verify all six per-TF SNR distance columns exist.
- Test `compute_full_context_features()` on a dataframe missing `high_4h` — verify default
  `10.0` fill for 4h columns.
- Test `mtf_snr_confluence`: construct two zone configs — one where 15m and 1h zones are
  within 0.15%, one where they are 0.5% apart — verify `confluence == 1` and `confluence == 0`
  respectively.
- Test Phase 1 HTF gate: verify HTF-contradicting bars are skipped, aligned bars are recorded.

### Property-Based Tests

- Generate random aligned dataframes with varying HTF price levels; verify that per-TF SNR
  distance columns are always present and non-negative.
- Generate random combinations of 15m/1h zone prices; verify that `mtf_snr_confluence == 1`
  iff `|z15 - z1h| / price <= 0.0015`.
- Generate random (rsi_1h, forward_move_12) pairs; verify that the gate filters exactly the
  contradicting combinations and passes all others.
- Generate aligned dataframes with and without `high_4h`; verify no exception in either case.

### Integration Tests

- Run `evaluate_expiries_for_symbol()` on a small synthetic dataset (200 bars); verify
  `aligned_df` columns include `snr_dist_support_15m` and `mtf_snr_confluence` after the
  enrichment phase.
- Verify that the Phase 1 training loop completes without exception and that the meta-learner
  replay buffer contains only HTF-confirmed samples when synthetic HTF RSI is fully controlled.
- Verify that out-of-sample evaluation (Phase 3) and portfolio simulation (Phase 4) run
  correctly on the enriched feature set — no KeyError on new column names, no dtype errors.
