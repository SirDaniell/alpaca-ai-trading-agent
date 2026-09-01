# Bugfix Requirements Document

## Introduction

The ML training pipeline in `evaluate_option_expiries.py` collects signal samples and computes
feature context using only the 5m execution timeframe's price series. It never applies the
`detect_snr_levels_sequential` + `create_clustered_zones_sequential` pipeline to higher timeframe
(HTF) price columns (`close_15m`, `high_1h`, `low_4h`, etc.) that are already present in the
aligned dataset. Consequently, (1) SNR zones are structurally wrong because they ignore HTF
pivots, (2) the feature vector carries no per-HTF SNR distance features, (3) MTF confluence is
approximated by rolling quantiles rather than real zone overlap, and (4) training samples are
recorded unconditionally — no HTF directional bias gate is applied before a bar is labelled
as a CALL or PUT candidate. This set of five related defects causes the meta-learner to train
on mislabelled, structurally misaligned data, producing a model that cannot correctly exploit
HTF confirmation bias at execution time.

---

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `update_real_snr_snapshot()` is called for any 5m bar index, THEN the system renames
    the aligned dataframe's `high_5m`/`low_5m`/`close_5m` columns to `High`/`Low`/`Close` and
    passes that slice to `detect_snr_levels_sequential`, so all detected S/R levels are derived
    exclusively from the 5m price series regardless of the `timeframe` parameter value.

1.2 WHEN `update_real_snr_snapshot()` receives a `timeframe` argument of `"15m"`, `"1h"`, or
    `"4h"`, THEN the system does not select the corresponding HTF price columns (`high_15m`,
    `low_1h`, etc.) from the aligned dataframe — the `timeframe` parameter is ignored.

1.3 WHEN `compute_full_context_features()` calculates `snr_dist_support` and
    `snr_dist_resistance`, THEN the system uses `df[low_col].rolling(100).quantile(0.20)` and
    `df[high_col].rolling(100).quantile(0.80)` as proxies, meaning no per-timeframe SNR distance
    columns (`snr_dist_support_15m`, `snr_dist_resistance_15m`, `snr_dist_support_1h`, etc.) are
    ever computed or added to the feature vector.

1.4 WHEN `compute_full_context_features()` calculates `mtf_snr_confluence`, THEN the system
    counts how many of the four rolling-quantile approximations are within 1 ATR of the current
    price — it never detects actual pivot-based zone overlap across timeframes, so confluence
    signals are structurally unsound.

1.5 WHEN `meta_learner.record_experience()` is called during the Phase 1 training sweep, THEN
    the system records every bar as a directional sample based solely on `forward_move_12` sign,
    with no check that the H1 or M30 RSI cross or directional bias aligns with the 5m signal
    direction before the sample is labelled `"bullish"` or `"bearish"`.

---

### Expected Behavior (Correct)

2.1 WHEN `update_real_snr_snapshot()` is called with `timeframe="15m"` (or `"1h"`, `"4h"`),
    THEN the system SHALL select the HTF price columns for that timeframe (`high_15m`/`low_15m`
    etc.) from the aligned dataframe, rename them to `High`/`Low`/`Close`/`Open`/`Volume`, and
    pass that per-TF slice to `detect_snr_levels_sequential`, so detected zones reflect the
    structural pivots of that specific timeframe.

2.2 WHEN SNR zone detection is triggered for a given 5m bar index, THEN the system SHALL invoke
    `update_real_snr_snapshot()` separately for each available HTF (`15m`, `1h`, `4h`) using
    only the HTF rows whose timestamps fall at or before the current 5m bar (no-lookahead), and
    SHALL tag each resulting snapshot with the originating timeframe.

2.3 WHEN `compute_full_context_features()` computes SNR proximity features, THEN the system
    SHALL produce per-TF distance columns — `snr_dist_support_15m`, `snr_dist_resistance_15m`,
    `snr_dist_support_1h`, `snr_dist_resistance_1h`, `snr_dist_support_4h`,
    `snr_dist_resistance_4h` — by running `detect_snr_levels_sequential` on the aligned HTF
    price columns for each timeframe and measuring the distance from the current 5m close to
    the nearest detected zone in each TF.

2.4 WHEN `compute_full_context_features()` computes `mtf_snr_confluence`, THEN the system SHALL
    set `mtf_snr_confluence = True` for a bar if and only if at least one 15m zone and one 1h
    zone have prices within 0.15% of each other (matching the CONFLUENCE_PCT threshold used in
    the reference StockChart), and SHALL replace the rolling-quantile approximation entirely.

2.5 WHEN `meta_learner.record_experience()` is called during the Phase 1 training sweep, THEN
    the system SHALL only record a bar as a `"bullish"` training sample if the H1 or M30
    RSI-derived directional bias (derived from `rsi_1h` / `rsi_15m` columns already present in
    the aligned dataframe) confirms an upward bias at that bar; and SHALL only record it as
    `"bearish"` if that same HTF bias confirms a downward bias — bars where the HTF bias is
    neutral or contradicts the 5m forward move SHALL be skipped or recorded as `"hold"`.

---

### Unchanged Behavior (Regression Prevention)

3.1 WHEN a 5m bar index is below the minimum lookback threshold (< 20 bars), THEN the system
    SHALL CONTINUE TO skip SNR zone computation without error, as it does today.

3.2 WHEN the aligned dataframe does not contain a column for a given HTF (e.g., `"high_4h"` is
    absent because 4h data was not fetched), THEN the system SHALL CONTINUE TO gracefully fall
    back to the next available HTF or skip that timeframe's SNR computation without raising an
    exception.

3.3 WHEN `detect_snr_levels_sequential` and `create_clustered_zones_sequential` are called,
    THEN the system SHALL CONTINUE TO use only data up to the current bar index with no
    lookahead, preserving the existing no-leakage guarantee.

3.4 WHEN `_row_to_feature_dict` builds a feature dictionary from a row, THEN the system SHALL
    CONTINUE TO include all numeric columns already present in the row (including aligned HTF
    OHLCV columns such as `close_15m`, `close_1h`), and the new per-TF SNR distance columns
    SHALL be appended in the same way.

3.5 WHEN a bar passes the HTF confirmation gate and is recorded as a training sample, THEN the
    system SHALL CONTINUE TO use `forward_move_12` (12 × 5m = 1h) as the outcome label for
    multi-horizon target computation, and SHALL CONTINUE TO respect the 70/15/15
    train/val/test chronological split without modification.

3.6 WHEN the 5m cross-signal features (`cross_index_signal`, `cross_dxy_signal`,
    `cross_index_dxy`, `cross_dxy_symbol`) are computed in `compute_full_context_features`,
    THEN the system SHALL CONTINUE TO derive them from the same RSI and divergence logic
    currently implemented, with no change to those calculations.

---

## Bug Condition Pseudocode

### Bug Condition Functions

```pascal
FUNCTION isBugCondition_SNR_TF_Ignored(bar, df_aligned, timeframe)
  INPUT: bar index, aligned dataframe, requested timeframe string
  OUTPUT: boolean

  // True when HTF SNR detection is requested but 5m columns are used instead
  RETURN (timeframe IN ["15m", "1h", "4h"])
     AND (df_aligned contains column "high_" + timeframe)
     AND (update_real_snr_snapshot uses "high_5m" or "high" for detection)
END FUNCTION

FUNCTION isBugCondition_SNR_Proximity_Approx(df_aligned)
  INPUT: aligned dataframe after compute_full_context_features
  OUTPUT: boolean

  // True when per-TF SNR distance features are absent from the feature vector
  RETURN ("snr_dist_support_15m" NOT IN df_aligned.columns)
     AND ("snr_dist_support_1h"  NOT IN df_aligned.columns)
END FUNCTION

FUNCTION isBugCondition_Confluence_Approx(df_aligned)
  INPUT: aligned dataframe
  OUTPUT: boolean

  // True when confluence uses rolling quantile rather than real zone proximity
  RETURN mtf_snr_confluence_computed_via == "rolling_quantile"
END FUNCTION

FUNCTION isBugCondition_HTFGateMissing(bar, htf_rsi_direction, fwd_move_12_sign)
  INPUT: bar features, HTF RSI direction, 5m forward move sign
  OUTPUT: boolean

  // True when a contradicting-HTF bar is recorded as a directional training sample
  RETURN (fwd_move_12_sign == +1 AND htf_rsi_direction == "bearish")
      OR (fwd_move_12_sign == -1 AND htf_rsi_direction == "bullish")
END FUNCTION
```

### Fix-Checking Properties

```pascal
// Property: SNR zones use per-TF price columns
FOR ALL bar WHERE isBugCondition_SNR_TF_Ignored(bar, df, tf) DO
  slice ← build_htf_slice(df, bar, tf)   // selects high_{tf}, low_{tf}, etc.
  levels ← detect_snr_levels_sequential(slice, ...)
  ASSERT ALL level IN levels: level.source_column == "high_" + tf OR "low_" + tf
END FOR

// Property: Per-TF SNR distance columns exist and are non-trivial
FOR ALL df_aligned' WHERE isBugCondition_SNR_Proximity_Approx(df_aligned') DO
  ASSERT "snr_dist_support_15m" IN df_aligned'.columns
  ASSERT "snr_dist_resistance_15m" IN df_aligned'.columns
  ASSERT "snr_dist_support_1h" IN df_aligned'.columns
  ASSERT "snr_dist_resistance_1h" IN df_aligned'.columns
END FOR

// Property: Confluence uses real zone proximity (0.15% threshold)
FOR ALL bar WHERE isBugCondition_Confluence_Approx(df_aligned) DO
  zones_15m ← detected_zones(df_aligned, bar, "15m")
  zones_1h  ← detected_zones(df_aligned, bar, "1h")
  confluence ← EXISTS z15 IN zones_15m, z1h IN zones_1h:
               ABS(z15.price - z1h.price) / z1h.price <= 0.0015
  ASSERT df_aligned'[bar]["mtf_snr_confluence"] == confluence
END FOR

// Property: HTF-contradicting bars are not recorded as directional samples
FOR ALL bar WHERE isBugCondition_HTFGateMissing(bar, htf_dir, fwd_sign) DO
  ASSERT bar NOT IN meta_learner.replay_buffer
      OR meta_learner.replay_buffer[bar].direction == "hold"
END FOR
```

### Preservation Property

```pascal
// Property: Preservation Checking — non-buggy inputs are unaffected
FOR ALL bar WHERE NOT isBugCondition_SNR_TF_Ignored(bar, df, "5m") DO
  // 5m SNR detection path is unchanged
  ASSERT F(bar) == F'(bar)
END FOR

FOR ALL bar WHERE htf_rsi_direction ALIGNS WITH fwd_move_12_sign DO
  // Bar still recorded as before; label unchanged
  ASSERT meta_learner.replay_buffer[bar].direction ==
         ("bullish" IF fwd_move_12_sign > 0 ELSE "bearish")
END FOR
```
