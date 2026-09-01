# Implementation Plan

- [x] 1. Write bug condition exploration tests (BEFORE implementing the fix)
  - **Property 1: Bug Condition** - Four-Defect MTF SNR Misalignment
  - **CRITICAL**: These tests MUST FAIL on unfixed code — failure confirms each bug exists
  - **DO NOT attempt to fix the test or the code when they fail**
  - **NOTE**: These tests encode the expected behavior — they will validate the fix when they pass after implementation
  - **GOAL**: Surface counterexamples that demonstrate all four bug conditions
  - **Scoped PBT Approach**: For each deterministic defect, scope to the concrete failing case to ensure reproducibility

  Create `backend/tests/test_mtf_snr_bugfix_exploration.py` with the following four scoped tests:

  **Test 1 — SNR TF Ignored**:
  - Build a minimal aligned dataframe with distinct 5m highs clustered around 100 and 15m highs clustered around 200
  - Call `update_real_snr_snapshot(df, up_to_idx=50, zm, timeframe="15m")` on the unfixed code
  - Assert that ALL zone prices in `zm` are near 200, not near 100
  - **EXPECTED OUTCOME on unfixed code**: zones will be near 100 → FAIL ✓ (confirms Bug Condition 1.1 / 1.2)
  - Document counterexample: `update_real_snr_snapshot(df, 50, zm, "15m")` returns 5m-derived zones (≈100) instead of 15m-derived zones (≈200)
  - Mark task complete when test is written, run, and failure is documented

  **Test 2 — SNR Proximity Approx**:
  - Build a small aligned dataframe that includes `high_15m`, `low_15m`, `high_1h`, `low_1h` columns
  - Call `compute_full_context_features(df)` on the unfixed code
  - Assert that `"snr_dist_support_15m"` is in `df.columns`
  - Assert that `"snr_dist_resistance_1h"` is in `df.columns`
  - **EXPECTED OUTCOME on unfixed code**: columns absent → FAIL ✓ (confirms Bug Condition 1.3)
  - Document counterexample: after `compute_full_context_features()`, per-TF distance columns (`snr_dist_support_15m`, etc.) are absent from the feature vector

  **Test 3 — Confluence Approx**:
  - Build a dataframe where the rolling-quantile-derived 15m proxy support and 1h proxy resistance are more than 1.5% apart from any real pivot overlap
  - Call `compute_full_context_features(df)` on the unfixed code
  - Assert that `mtf_snr_confluence` is 0 for those rows
  - **EXPECTED OUTCOME on unfixed code**: rolling-quantile counter produces non-zero values → FAIL ✓ (confirms Bug Condition 1.4)
  - Document counterexample: `mtf_snr_confluence = 2` even when no 15m/1h pivot prices overlap within 0.15%

  **Test 4 — HTF Gate Missing**:
  - Mock `meta_learner.record_experience()` to capture all calls
  - Construct a train_df slice of 10 bars where all rows have `forward_move_12 > 0` (bullish 5m label) but `rsi_1h < 48` (bearish HTF RSI)
  - Run the Phase 1 training loop over those rows on unfixed code
  - Assert that NO row from that slice appears in the captured `record_experience` calls as `direction="bullish"`
  - **EXPECTED OUTCOME on unfixed code**: all 10 rows are recorded as `"bullish"` → FAIL ✓ (confirms Bug Condition 1.5)
  - Document counterexample: HTF-contradicting bars (fwd_move_12>0, rsi_1h<48) recorded as `direction="bullish"` in replay buffer

  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Write preservation property tests (BEFORE implementing the fix)
  - **Property 2: Preservation** - Non-Buggy Input Paths Are Unaffected
  - **IMPORTANT**: Follow observation-first methodology — run unfixed code on non-buggy inputs and record actual outputs
  - Create `backend/tests/test_mtf_snr_preservation.py` covering:

  **Preservation Test A — 5m SNR path unchanged**:
  - Call `update_real_snr_snapshot(df, up_to_idx=50, zm, timeframe="5m")` on unfixed code and record zone prices
  - Write property-based test: for any `up_to_idx >= 20` and any aligned dataframe, the 5m SNR path returns the same zones before and after the fix
  - Verify test PASSES on unfixed code (confirming baseline)

  **Preservation Test B — Early-return guard (bar < 20)**:
  - Observe: `update_real_snr_snapshot(df, up_to_idx=5, zm, timeframe="15m")` returns immediately, adds nothing to `zm`
  - Write property-based test: for all `up_to_idx < 20`, the function exits without adding any snapshot to the ZoneSnapshotManager
  - Verify test PASSES on unfixed code

  **Preservation Test C — Missing HTF column graceful fallback**:
  - Observe: calling `compute_full_context_features(df)` on a dataframe WITHOUT `high_4h`/`low_4h` raises no exception
  - Write property-based test: for any aligned dataframe missing `high_4h`, `compute_full_context_features` completes without exception and produces valid output for all present TFs
  - Verify test PASSES on unfixed code

  **Preservation Test D — HTF-aligned bars still recorded**:
  - Observe: bars where `forward_move_12 > 0` AND `rsi_1h > 52` are recorded as `direction="bullish"` by the unfixed Phase 1 loop
  - Write property-based test: for all (rsi_1h, fwd_move_12) pairs where HTF aligns with the 5m label, `record_experience` is called with the correct direction
  - Verify test PASSES on unfixed code (these bars are already correct behavior)

  **Preservation Test E — Chronological train/val/test split unchanged**:
  - Observe: the 70/15/15 split index positions for a given dataframe length
  - Write test asserting split boundaries are identical before and after the fix
  - Verify test PASSES on unfixed code

  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [x] 3. Fix `update_real_snr_snapshot()` — dynamic HTF column selection

  - [x] 3.1 Resolve HTF column names dynamically from the `timeframe` parameter
    - After the existing `if up_to_idx < 20: return` guard, add logic to determine the correct price columns:
      - If `timeframe == "5m"` (or the module-level primary TF), use existing `high_col`, `low_col`, `close_col`, `open_col`, `vol_col`
      - Otherwise, construct `high_col_tf = f"high_{timeframe}"`, `low_col_tf = f"low_{timeframe}"`, `close_col_tf = f"close_{timeframe}"`, `open_col_tf = f"open_{timeframe}"`, `vol_col_tf = f"volume_{timeframe}"`
      - If any of the four price columns are absent from `df_full.columns`, return early without raising an exception (preserves requirement 3.2)
    - Replace the hardcoded rename mapping `{high_col: "High", low_col: "Low", close_col: "Close", open_col: "Open", vol_col: "Volume"}` with the dynamically resolved column map
    - No other changes: `detect_snr_levels_sequential`, `create_clustered_zones_sequential`, and `ZoneSnapshotManager.add_snapshot()` calls remain identical
    - _Bug_Condition: isBugCondition_SNR_TF_Ignored(bar, df, tf) where tf IN ["15m", "1h", "4h"] and df contains high_{tf}_
    - _Expected_Behavior: zones in zm reflect pivots from high_{tf}/low_{tf}, not high_5m/low_5m_
    - _Preservation: 5m SNR path (tf=="5m") is structurally identical to before; early-return guard for up_to_idx < 20 unchanged; missing-column early-return added for 3.2_
    - _Requirements: 2.1, 2.2, 3.1, 3.2, 3.3_

  - [x] 3.2 Call `update_real_snr_snapshot()` once per HTF at the same cadence as today
    - In the evaluation loops (Phase 1 training, Phase 1b, Phase 3, Phase 4) where `update_real_snr_snapshot` is already called with `timeframe="15m"`, ensure the call correctly receives the aligned dataframe as `df_full` (no other structural changes to the call sites are needed — Change 1 fixes the column selection inside the function)
    - Verify no duplicate snapshot accumulation by confirming the existing `idx % 15 == 0` cadence gate is unchanged
    - _Requirements: 2.2, 3.3_

- [x] 4. Fix `compute_full_context_features()` — per-TF SNR distance columns + real confluence

  - [x] 4.1 Add per-TF SNR distance columns for `15m`, `1h`, `4h`
    - After the existing ATR computation (line computing `atr`), add a loop over `TF_CONFIGS = [("15m", 100), ("1h", 30), ("4h", 10)]` where the second element is the rolling window size (each roughly represents ~25h of history at the 5m bar density)
    - For each `(tf, window)`:
      - Check if `f"high_{tf}"` and `f"low_{tf}"` exist in `df.columns`
      - If present: compute `supp_tf = df[f"low_{tf}"].rolling(window, min_periods=max(1, window//4)).quantile(0.20).fillna(df[f"low_{tf}"])` and `res_tf = df[f"high_{tf}"].rolling(window, min_periods=max(1, window//4)).quantile(0.80).fillna(df[f"high_{tf}"])`; then set `df[f"snr_dist_support_{tf}"] = np.clip((asset_close - supp_tf.values) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)` and `df[f"snr_dist_resistance_{tf}"] = np.clip((res_tf.values - asset_close) / (atr + 1e-8), 0.0, 10.0).astype(np.float32)`
      - If absent: set both columns to `np.full(len(df), 10.0, dtype=np.float32)` (graceful fallback per requirement 3.2)
    - Remove or retain the original `snr_dist_support` / `snr_dist_resistance` single-TF columns for backward compatibility with any existing code that reads them — do not delete them
    - _Bug_Condition: isBugCondition_SNR_Proximity_Approx(df) — per-TF columns absent after compute_full_context_features_
    - _Expected_Behavior: snr_dist_support_15m, snr_dist_resistance_15m, snr_dist_support_1h, snr_dist_resistance_1h, snr_dist_support_4h, snr_dist_resistance_4h all present and non-negative in df_
    - _Preservation: existing snr_dist_support and snr_dist_resistance columns retained; missing-column path defaults to 10.0 without exception_
    - _Requirements: 2.3, 3.2, 3.4_

  - [x] 4.2 Replace rolling-quantile confluence with real zone-proximity confluence
    - After task 4.1's per-TF distance columns are computed, replace the four-condition integer sum that produces `mtf_snr_confluence` with:
      ```python
      # Convert ATR-normalised distances back to approximate price distances
      dist_sup_15m_price = df["snr_dist_support_15m"].values * atr
      dist_res_15m_price = df["snr_dist_resistance_15m"].values * atr
      dist_sup_1h_price  = df["snr_dist_support_1h"].values  * atr
      dist_res_1h_price  = df["snr_dist_resistance_1h"].values  * atr

      # Approximate zone anchor prices
      zone_15m_sup = asset_close - dist_sup_15m_price
      zone_15m_res = asset_close + dist_res_15m_price
      zone_1h_sup  = asset_close - dist_sup_1h_price
      zone_1h_res  = asset_close + dist_res_1h_price

      # Confluence: any 15m zone within CONFLUENCE_PCT=0.0015 of any 1h zone
      CONFLUENCE_PCT = 0.0015
      sup_confluence  = (np.abs(zone_15m_sup - zone_1h_sup)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
      res_confluence  = (np.abs(zone_15m_res - zone_1h_res)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
      cross_sup_res   = (np.abs(zone_15m_sup - zone_1h_res)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT
      cross_res_sup   = (np.abs(zone_15m_res - zone_1h_sup)  / (asset_close + 1e-8)) <= CONFLUENCE_PCT

      df["mtf_snr_confluence"] = (sup_confluence | res_confluence | cross_sup_res | cross_res_sup).astype(np.float32)
      ```
    - Delete the old four-condition sum block (the `confluence = ((df["snr_dist_support"].values <= 1.0).astype(int) + ...)` lines)
    - _Bug_Condition: isBugCondition_Confluence_Approx — mtf_snr_confluence uses rolling_quantile count_
    - _Expected_Behavior: mtf_snr_confluence == 1 iff at least one 15m zone and one 1h zone are within CONFLUENCE_PCT=0.0015 of each other_
    - _Preservation: column name "mtf_snr_confluence" unchanged; dtype float32 unchanged_
    - _Requirements: 2.4, 3.6_

- [x] 5. Fix Phase 1 training loop — add HTF confirmation gate before `record_experience()`

  - [x] 5.1 Read `rsi_1h` from the current row and derive `htf_bias`
    - Inside the Phase 1 epoch loop, immediately before the `meta_learner.record_experience(...)` call, add:
      ```python
      rsi_1h_val = float(row.get("rsi_1h", 50.0))
      if rsi_1h_val > 52:
          htf_bias = "bullish"
      elif rsi_1h_val < 48:
          htf_bias = "bearish"
      else:
          htf_bias = "neutral"
      ```
    - The `rsi_1h` column is populated by `compute_full_context_features()` via the MTF RSI blend (already present in `aligned_df`)
    - _Requirements: 2.5_

  - [x] 5.2 Apply the HTF gate to filter contradicting training samples
    - Immediately after computing `htf_bias`, add the filter:
      ```python
      fwd_sign = row["forward_move_12"]
      if fwd_sign > 0 and htf_bias == "bearish":
          continue   # HTF contradiction — skip bullish label
      if fwd_sign <= 0 and htf_bias == "bullish":
          continue   # HTF contradiction — skip bearish label
      # Neutral or aligned: fall through to record_experience as before
      ```
    - The `direction` argument passed to `record_experience()` is unchanged for all bars that pass the gate
    - _Bug_Condition: isBugCondition_HTFGateMissing — bars where fwd_move_12>0 and htf_bias=="bearish" (or vice versa) are recorded as directional samples_
    - _Expected_Behavior: those bars are skipped (continue) and never reach record_experience_
    - _Preservation: bars where htf_bias is neutral or aligns with fwd_sign are recorded with the same direction label as before_
    - _Requirements: 2.5, 3.5_

- [x] 6. Verify exploration test (Property 1) now passes after the fix
  - **Property 1: Expected Behavior** - Four-Defect MTF SNR Misalignment
  - **IMPORTANT**: Re-run the SAME four tests from task 1 — do NOT write new tests
  - Run `backend/tests/test_mtf_snr_bugfix_exploration.py`
  - **EXPECTED OUTCOME**: All four tests PASS (confirms all four bug conditions are resolved)
    - Test 1: zones from `update_real_snr_snapshot(df, 50, zm, "15m")` are near 200 (HTF pivots), not 100
    - Test 2: `snr_dist_support_15m` and `snr_dist_resistance_1h` present in `df.columns`
    - Test 3: `mtf_snr_confluence == 0` when 15m and 1h zones are >1.5% apart
    - Test 4: HTF-contradicting bars absent from `record_experience` replay buffer
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 7. Verify preservation tests (Property 2) still pass after the fix
  - **Property 2: Preservation** - Non-Buggy Input Paths Are Unaffected
  - **IMPORTANT**: Re-run the SAME tests from task 2 — do NOT write new tests
  - Run `backend/tests/test_mtf_snr_preservation.py`
  - **EXPECTED OUTCOME**: All preservation tests PASS (confirms no regressions)
    - 5m SNR path produces identical zone sets
    - Early-return guard for `up_to_idx < 20` still works
    - Missing `high_4h` column triggers no exception; defaults to 10.0
    - HTF-aligned bars still recorded with the correct direction label
    - 70/15/15 chronological split boundaries unchanged
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

- [ ] 8. Write unit tests for all targeted fix surfaces
  - Create `backend/tests/test_mtf_snr_unit.py` covering:
  - `update_real_snr_snapshot()` with `timeframe="5m"` — zones reflect 5m pivots
  - `update_real_snr_snapshot()` with `timeframe="15m"` — zones reflect 15m pivots
  - `update_real_snr_snapshot()` with `timeframe="1h"` — zones reflect 1h pivots
  - `update_real_snr_snapshot()` with `timeframe="4h"` — zones reflect 4h pivots
  - `update_real_snr_snapshot()` with missing `high_15m` column — no exception, nothing added to ZoneSnapshotManager
  - `compute_full_context_features()` — all six per-TF distance columns present: `snr_dist_support_15m`, `snr_dist_resistance_15m`, `snr_dist_support_1h`, `snr_dist_resistance_1h`, `snr_dist_support_4h`, `snr_dist_resistance_4h`
  - `compute_full_context_features()` on dataframe missing `high_4h` — 4h columns default to `10.0`, no exception
  - `mtf_snr_confluence`: construct a case where 15m and 1h zones are within 0.15% — assert `confluence == 1.0`
  - `mtf_snr_confluence`: construct a case where 15m and 1h zones are 0.5% apart — assert `confluence == 0.0`
  - HTF gate: `fwd_move_12 > 0` + `rsi_1h = 45` (bearish) → `record_experience` NOT called
  - HTF gate: `fwd_move_12 > 0` + `rsi_1h = 55` (bullish) → `record_experience` called as `direction="bullish"`
  - HTF gate: `fwd_move_12 > 0` + `rsi_1h = 50` (neutral) → `record_experience` called as `direction="bullish"`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2_

- [ ] 9. Write property-based tests for stronger universal guarantees
  - Create `backend/tests/test_mtf_snr_pbt.py` using `hypothesis` (install if not present: `pip install hypothesis`)
  - **Property A**: For any aligned dataframe with randomized HTF price levels, all six per-TF SNR distance columns are always present and contain only non-negative, finite values
  - **Property B**: For any (zone_15m_price, zone_1h_price, current_price) triple, `mtf_snr_confluence == 1` iff `abs(zone_15m_price - zone_1h_price) / current_price <= 0.0015`
  - **Property C**: For any (rsi_1h, forward_move_12) pair, the gate filters exactly the contradicting cases (`rsi_1h < 48` + `fwd > 0`, or `rsi_1h > 52` + `fwd <= 0`) and passes all others (neutral band 48–52 or aligned direction)
  - **Property D**: For any aligned dataframe with or without `high_4h`/`low_4h`, `compute_full_context_features()` raises no exception; when 4h columns are absent, `snr_dist_support_4h` equals `10.0` everywhere
  - _Requirements: 2.3, 2.4, 2.5, 3.2_

- [ ] 10. Checkpoint — Ensure all tests pass
  - Run full test suite: `cd backend && python -m pytest tests/test_mtf_snr_bugfix_exploration.py tests/test_mtf_snr_preservation.py tests/test_mtf_snr_unit.py tests/test_mtf_snr_pbt.py -v`
  - All exploration tests (task 1 / task 6) must PASS after the fix
  - All preservation tests (task 2 / task 7) must PASS
  - All unit tests (task 8) must PASS
  - All property-based tests (task 9) must PASS
  - Ensure no `KeyError` on new column names and no dtype errors across Phase 1, Phase 1b, Phase 3, Phase 4 loops
  - Ask the user if any questions arise
