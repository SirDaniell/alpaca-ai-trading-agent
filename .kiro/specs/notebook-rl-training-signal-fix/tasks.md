# Implementation Plan

- [x] 1. Write bug condition exploration tests (BEFORE implementing any fixes)
  - **Property 1: Bug Condition** - Seven Confirmed RL Training Bugs
  - **CRITICAL**: Write these property-based tests BEFORE implementing any fixes
  - **GOAL**: Surface counterexamples that demonstrate each bug exists on unfixed code
  - **DO NOT attempt to fix the test or the code when it fails**
  - **NOTE**: These tests encode the expected behavior — they will validate the fix when they pass after implementation
  - **Scoped PBT Approach**: For deterministic bugs, scope each property to the concrete failing case(s) to ensure reproducibility
  - **Sub-tests to write (all in a single test cell or test file `test_bug_conditions.py`):**

  - **1a — Bug 6 (Backbone Gradient Probe)**: Register a backward hook on `SignalMetaNetwork.b1_conv1.weight`. Run one training step on a minimal 8-row batch. Assert that `grad_norm_from_l_pips_l_risk` is zero (`.detach()` severs the path). This is the counterexample proving Bug 6 exists.
    - Condition from design: `branch_cat = ...torch.cat([b1_out, b2_out, b3_out], dim=-1).detach()` → `b1_conv1.weight.grad` reflects only `l_q` + `l_str`, not pips/risk/liq/rev
    - Expected outcome on unfixed code: backbone gradient norm driven by ≤ 2 heads; pips/risk/liq/rev contribute 0

  - **1b — Bug 3 (Coin-Flip Label Check)**: Compute `train_df["target_dir_5m"].mean()` on the training CSV. Assert that `abs(mean - 0.5) < 0.02` — confirming the coin-flip condition. Then run 10 training steps and assert `l_q > 0.24` (at or above the MSE floor for 50/50 iid binary labels).
    - Condition from design: `isBugCondition_Bug3(direction_labels)` where `abs(mean(labels) - 0.5) < 0.02`
    - Expected outcome on unfixed code: `target_dir_5m.mean() ≈ 0.503`; MSE plateau at ~0.25

  - **1c — Bug 5 (Q_LOOKBACK Execution Order Test)**: Execute cells in the order Cell 2 → Cell 7 → Cell 10. After all three cells run, check `build_feat_window.__defaults__` (the value bound at function-definition time in Cell 7) vs the `Q_LOOKBACK` used to instantiate `ExecutorQNetwork` in Cell 10. Assert that they differ — confirming the consistency bug.
    - Condition from design: `NOT (q_lookback_network == q_lookback_window_fn == q_lookback_train)`
    - Expected outcome on unfixed code: `build_feat_window` default = 300 (from Cell 7), `ExecutorQNetwork.q_lookback` = 64 (from Cell 10's last assignment)

  - **1d — Bug 1 (Synthesized ML Target Coverage)**: Call `_fill_ml_targets(train_df)` without prior Cell 8 synthesis. Count keys where `abs(v).mean() <= 1e-7`. Assert that count > 0 (some targets are zero). Specifically verify that `vel_bull_fwd_8`, `Volatility_Regime_next`, and zone columns are zero if CSV lacks them.
    - Condition from design: `COUNT(k IN ml_targets WHERE abs(ml_targets[k]).mean() <= 1e-7) > 0`
    - Expected outcome on unfixed code: "ML targets filled | non-zero keys: 0 / 22" (confirmed in notebook output)

  - **1e — Bug 7 (Feature Scale Check)**: Compute `train_num_matrix[:, _PRICE_FEAT_INDICES].mean()` and `.std()` on the raw (unfixed) feature matrix. Assert that mean > 50.0 (absolute GLD price level) — confirming non-normalized absolute values.
    - Condition from design: `isBugCondition_Bug7(window)` where `price_std > 1.0`
    - Expected outcome on unfixed code: close_5m mean ≈ 185–210 (GLD absolute price level), std ≈ 10–20

  - **1f — Bug 2 (WAIT Reward Dominance)**: Run 3 Q-training epochs on the unfixed reward function (WAIT = +0.001 unconditionally). After epoch 3, compute `action_counts[0][H_WAIT] / (action_counts[0][H_WAIT] + action_counts[0][H_CALL] + action_counts[0][H_PUT])`. Assert that WAIT fraction > 0.85 — confirming policy collapse.
    - Condition from design: `reward_WAIT > MAX(reward_CALL_avg, reward_PUT_avg) + epsilon`
    - Expected outcome on unfixed code: ~27699 WAIT vs ~3803 CALL vs ~3327 PUT (epoch 20 from notebook output)

  - **1g — Bug 4 (Buffer Wipe Check)**: Place `replay_buffers = [[] for _ in range(NUM_HORIZONS)]` inside the loop for 2 iterations. After iteration 1, add 100 items; after iteration 2, assert `len(replay_buffers[0]) == 0` — confirming the wipe.
    - Condition from design: `buffer_init_location == "inside_epoch_loop"`
    - Expected outcome on unfixed code: buffer resets to empty every epoch; <3% of experience retained within an epoch

  - Run all sub-tests on UNFIXED code
  - **EXPECTED OUTCOME**: All 7 sub-tests FAIL (i.e., each assertion detecting buggy behavior passes, confirming all bugs exist)
  - Document counterexamples found for each bug (copy from notebook cell outputs in the bug report)
  - Mark task complete when all 7 sub-tests are written, run, and counterexamples documented
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 6.1, 6.2, 7.1, 7.2_

- [x] 2. Write preservation property tests (BEFORE implementing any fixes)
  - **Property 2: Preservation** - Existing Correct Behaviors in SignalMetaNetwork, Q-Executor, and Reward Pipeline
  - **IMPORTANT**: Follow observation-first methodology — observe behavior on UNFIXED code for non-buggy inputs
  - **Observe on unfixed code:**
    - `SignalMetaNetwork` forward produces shapes: `q_vals (B,4)`, `strength (B,4)`, `pips (B,4)`, `risk (B,8)`, `liq (B,2)`, `rev (B,1)`
    - Phase 1 weight-freeze works: `net.b1_conv1.weight.requires_grad` is True before Phase 2, toggled to False after
    - CALL reward = `fwd_pct - 0.0005`, PUT reward = `-fwd_pct - 0.0005` — PnL-based, unchanged by WAIT fix
    - `_batch_from_index_replay` produces 6-element tuples `(fw, ctx, action, reward, nfw, nctx)` compatible with Q-update
    - `feature_cols` has 326 columns; normalization is additive post-processing, not a column removal
    - `build_feat_window` returns shape `(Q_LOOKBACK, num_features)`; shape must be preserved after normalization
    - End-to-end cell execution completes without manual intervention between Phase 1 and Phase 2
  - **Write property-based tests capturing observed behavior patterns (from Preservation Requirements in design section 3.1–3.10):**
    - For all valid batch sizes B ∈ {1, 16, 64}: `SignalMetaNetwork` output shapes are constant
    - For all CALL/PUT transitions in replay: `reward == fwd_pct ± 0.0005` (WAIT reward changes must not affect this)
    - For all `build_feat_window` calls after normalization: shape `(Q_LOOKBACK, num_features)` is preserved
    - For any number of cell re-runs: `feature_cols` length remains 326
    - For any Q-training step: `_batch_from_index_replay` tuple length remains 6
  - Verify tests PASS on UNFIXED code (baseline behavior confirmed)
  - **EXPECTED OUTCOME**: All preservation tests PASS (confirms baseline behaviors to preserve)
  - Mark task complete when tests are written, run, and passing on unfixed code
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10_

- [x] 3. Fix Bug 6 — Remove `.detach()` from Auxiliary Head Inputs (Cell 6)

  - [x] 3.1 Remove `.detach()` from `branch_cat` and `aux1`/`aux2` in `SignalMetaNetwork.forward()`
    - In Cell 6, `SignalMetaNetwork.forward()`, lines 126–128: change `aux1 = self.aux1_head(b1_out.detach())` → `aux1 = self.aux1_head(b1_out)` and `aux2 = self.aux2_head(b2_out.detach())` → `aux2 = self.aux2_head(b2_out)`
    - In Cell 6, line 141: change `branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1).detach())` → `branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1))`
    - Keep `aux1_sg = aux1.detach()` and `aux2_sg = aux2.detach()` unchanged — these prevent double-counting through the fusion concat
    - Update the comment from "Zero Gradient Interference via feat.detach()" to "Aux heads connected to backbone — gradients flow through b1_out, b2_out, b3_out and branch_cat to Conv1D/LSTM towers"
    - _Bug_Condition: `isBugCondition_Bug6` where `branch_cat` is detached, severing pips/risk/liq/rev gradient paths to Conv1D/LSTM_
    - _Expected_Behavior: After fix, `b1_conv1.weight.grad.abs().mean()` reflects contributions from all 6 heads, not just l_q and l_str_
    - _Preservation: SignalMetaNetwork output shapes unchanged; aux1_sg and aux2_sg remain detached before fusion concat_
    - _Requirements: 2.15, 2.16, 3.5_

  - [x] 3.2 Verify Bug 6 exploration test now passes (Property 1: Expected Behavior — Backbone Gradient)
    - **Property 1: Expected Behavior** - Backbone Gradient Reaches All 6 Heads
    - **IMPORTANT**: Re-run the SAME sub-test 1a from task 1 — do NOT write a new test
    - Run the backward hook test: after one training step, assert backbone gradient norm increases beyond the value measured on unfixed code
    - Assert `b1_conv1.weight.grad is not None` and grad norm > the pre-fix baseline
    - **EXPECTED OUTCOME**: Test PASSES (confirms Bug 6 is fixed)
    - _Requirements: 2.15, 2.16_

  - [x] 3.3 Verify preservation tests still pass
    - **Property 2: Preservation** - SignalMetaNetwork Output Shapes Unchanged
    - **IMPORTANT**: Re-run the SAME preservation tests from task 2 — do NOT write new tests
    - Assert `q_vals.shape == (B, 4)`, `strength.shape == (B, 4)`, `pips.shape == (B, 4)`, `risk.shape == (B, 8)`, `liq.shape == (B, 2)`, `rev.shape == (B, 1)` for batch sizes 1, 16, 64
    - **EXPECTED OUTCOME**: Tests PASS (confirms no regressions)

- [x] 4. Fix Bug 3 — Zone-Conditional Direction Labels with ATR-Threshold Weighting (Cell 9)

  - [x] 4.1 Add ATR-threshold direction weights to Cell 9 target preparation block
    - After `_rebuild_targets` is called, compute `atr_vals` from `atr_col` or high-low rolling mean
    - Compute `direction_weight = (np.abs(fwd_move_1) > 0.30 * atr_vals).astype(np.float32)` — excludes micro-move bars from the direction loss
    - Add validation assert: `effective_pos_rate = train_df.loc[direction_weight.astype(bool), "target_dir_5m"].mean()` and assert `abs(effective_pos_rate - 0.5) > 0.02`
    - Print: `[Bug3-fix] Effective direction label positive rate: {effective_pos_rate:.3f} on {direction_weight.sum():.0f} / {len(direction_weight)} bars`
    - _Bug_Condition: `isBugCondition_Bug3` where `abs(mean(target_dir_5m) - 0.5) < 0.02` (coin-flip labels)_
    - _Expected_Behavior: Effective positive rate after ATR-threshold filtering differs from 0.50 by > 0.02; direction head MSE decreases below 0.24 within 30 epochs_
    - _Preservation: `_rebuild_targets` output unchanged; `feature_cols` unchanged; existing pips/risk/liq/rev targets unchanged_
    - _Requirements: 2.7, 2.8, 3.1, 3.5_

  - [x] 4.2 Apply `direction_weight` to `l_q` in the Phase 1 training loop (Cell 9)
    - In the training loop's loss computation block, replace `l_q = nn.MSELoss()(q_vals, y_q_t)` with weighted MSE: `w_t = torch.tensor(direction_weight[ti], ...).unsqueeze(1).expand_as(q_vals)` and `l_q = (w_t * (q_vals - y_q_t) ** 2).sum() / (w_t.sum() + 1e-6)`
    - Ensure `direction_weight` array is computed once before the epoch loop and indexed via `ti` batch indices
    - _Requirements: 2.7, 2.8_

  - [x] 4.3 Verify Bug 3 exploration test now passes (Property 1: Expected Behavior — Direction Labels)
    - **Property 1: Expected Behavior** - Direction Labels Are Discriminative
    - **IMPORTANT**: Re-run the SAME sub-test 1b from task 1 — do NOT write a new test
    - Assert `abs(effective_pos_rate - 0.5) > 0.02` after filtering
    - After 10+ training steps: assert `l_q < 0.24` (below the 50/50 MSE floor)
    - **EXPECTED OUTCOME**: Test PASSES (confirms Bug 3 is fixed)
    - _Requirements: 2.7, 2.8_

  - [x] 4.4 Verify preservation tests still pass
    - **Property 2: Preservation** - Target Arrays and Feature Columns Unchanged
    - **IMPORTANT**: Re-run the SAME preservation tests from task 2
    - Assert `len(feature_cols) == 326`; assert pips/risk/liq/rev target shapes unchanged
    - **EXPECTED OUTCOME**: Tests PASS

- [x] 5. Fix Bug 5 — Consolidate Q_LOOKBACK to Single Definition (Cells 2, 7, 10)

  - [x] 5.1 Remove Q_LOOKBACK re-definitions from Cells 7 and 10; add execution-order assertion
    - In Cell 7: remove the line `Q_LOOKBACK = 300` and update the comment to "uses Q_LOOKBACK from Cell 2 config"
    - In Cell 10: remove both `Q_LOOKBACK = 150` (line 0) and `Q_LOOKBACK = 64` (line 11); replace with `assert 'Q_LOOKBACK' in dir(), "Run Cell 2 (config) before Cell 10"` and `print(f"[Config check] Q_LOOKBACK = {Q_LOOKBACK}")`
    - In Cell 2, ensure `Q_LOOKBACK = 150` is the sole authoritative definition with a clear comment
    - _Bug_Condition: `isBugCondition_Bug5` where `q_lookback_network != q_lookback_window_fn != q_lookback_train`_
    - _Expected_Behavior: All three Q_LOOKBACK usage sites (network instantiation, build_feat_window default, training window construction) read the same value regardless of cell execution order_
    - _Preservation: ExecutorQNetwork continues using AdaptiveAvgPool1d; no change to network architecture or output shapes_
    - _Requirements: 2.13, 2.14, 3.10_

  - [x] 5.2 Fix `build_feat_window` default argument binding to use call-time Q_LOOKBACK
    - Change `def build_feat_window(num_matrix, abs_idx, q_lookback=Q_LOOKBACK, ...)` to `def build_feat_window(num_matrix, abs_idx, q_lookback=None, ...)` with `if q_lookback is None: q_lookback = Q_LOOKBACK` as the first line of the function body
    - This ensures the default evaluates at call time (when Q_LOOKBACK is already set by Cell 2), not at function-definition time (when execution order is unpredictable)
    - _Requirements: 2.13, 2.14_

  - [x] 5.3 Verify Bug 5 exploration test now passes (Property 1: Expected Behavior — Q_LOOKBACK Consistency)
    - **Property 1: Expected Behavior** - Q_LOOKBACK Consistent Across All Usage Sites
    - **IMPORTANT**: Re-run the SAME sub-test 1c from task 1 — do NOT write a new test
    - Run cells in order 2 → 7 → 10; assert `build_feat_window` uses the same value as `ExecutorQNetwork.q_lookback`
    - Assert `q_net.q_lookback == Q_LOOKBACK == 150`
    - **EXPECTED OUTCOME**: Test PASSES (confirms Bug 5 is fixed)
    - _Requirements: 2.13, 2.14_

  - [x] 5.4 Verify preservation tests still pass
    - **Property 2: Preservation** - ExecutorQNetwork Architecture Unchanged
    - **IMPORTANT**: Re-run the SAME preservation tests from task 2
    - Assert `build_feat_window` output shape `(Q_LOOKBACK, num_features)` unchanged; assert `AdaptiveAvgPool1d` still present in architecture
    - **EXPECTED OUTCOME**: Tests PASS

- [x] 6. Fix Bug 1 — Harden ML Target Synthesis and Loss Gating (Cells 8 and 9)

  - [x] 6.1 Define explicit permanently-zero key set and add post-synthesis validation in Cell 8
    - Add `_PERMANENTLY_ZERO_KEYS` set: zone index/bars/distance/type and 4 CSM columns (8 keys total that cannot be synthesized from OHLCV data)
    - Add `_REQUIRED_NONZERO_KEYS = set(_ML_KEYS) - _PERMANENTLY_ZERO_KEYS` (14 synthesizable keys)
    - After `train_ml_targets = _fill_ml_targets(train_df)`, add assert loop: `for k in _REQUIRED_NONZERO_KEYS: assert abs(train_ml_targets[k]).mean() > 1e-6, f"Synthesized target '{k}' is all-zero"`
    - Print: `ML targets validated: {len(_REQUIRED_NONZERO_KEYS)} synthesized keys non-zero, {len(_PERMANENTLY_ZERO_KEYS)} permanently excluded from loss`
    - _Bug_Condition: `isBugCondition_Bug1` where any required target array has `abs(v).mean() <= 1e-7` or contains NaN_
    - _Expected_Behavior: All 14 synthesizable keys have `abs(v).mean() > 1e-6`; 8 permanently-missing keys are excluded from loss via `_PERMANENTLY_ZERO_KEYS` guard, not per-step runtime mean check_
    - _Preservation: Loss weights for l_zone, l_vol, l_vel unchanged (0.20, 0.15, 0.15); existing graceful fallback for CSV-absent columns preserved_
    - _Requirements: 2.1, 2.2, 2.3, 3.1_

  - [x] 6.2 Replace per-step runtime mean checks in Cell 9 loss computation with pre-computed exclusion set
    - Replace the `if _y_zone_idx_mean > 1e-7:` / `if _y_zone_bars_mean > 1e-7:` pattern with `if "adv_target_next_zone_bars" not in _PERMANENTLY_ZERO_KEYS and ...` guards
    - This eliminates the per-step CPU-side mean computation overhead while making the exclusion logic explicit
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 6.3 Verify Bug 1 exploration test now passes (Property 1: Expected Behavior — ML Targets Non-Zero)
    - **Property 1: Expected Behavior** - 14 Synthesizable ML Targets Are Non-Zero
    - **IMPORTANT**: Re-run the SAME sub-test 1d from task 1 — do NOT write a new test
    - Assert `sum(1 for k,v in train_ml_targets.items() if abs(v).mean() <= 1e-7) == 8` (exactly the 8 permanently-missing keys)
    - Assert `sum(1 for k in _REQUIRED_NONZERO_KEYS if abs(train_ml_targets[k]).mean() > 1e-6) == 14`
    - **EXPECTED OUTCOME**: Test PASSES (confirms Bug 1 is fixed)
    - _Requirements: 2.1, 2.2, 2.3_

  - [x] 6.4 Verify preservation tests still pass
    - **Property 2: Preservation** - Auxiliary Loss Weights and CSV-Present Column Fallback Unchanged
    - **IMPORTANT**: Re-run the SAME preservation tests from task 2
    - Assert that when CSV columns ARE present, `_fill_ml_targets` uses them directly (not synthesis)
    - **EXPECTED OUTCOME**: Tests PASS

- [ ] 7. Fix Bug 2 — Decouple WAIT Reward from Meta_Strength and Strengthen Robustness (Cell 10)

  - [-] 7.1 Replace `meta_strength`-dependent WAIT penalty with ATR-normalized move size proxy
    - In Cell 10, WAIT reward branch: replace the `h_strength >= 0.58` condition with `atr_norm = abs(fwd_pct) / max(abs(atr_vals[abs_idx]) / cp, 1e-6)` and threshold `atr_norm >= 1.5` for the penalty
    - This decouples the WAIT penalty from Phase 1 meta_strength (which may be degenerate from broken backbone) and makes it robust to Bug 6 being fixed concurrently
    - Keep the other branches: `reward = 0.002` for flat bars (`abs(fwd_pct) < 0.0003`), `reward = 0.0` for ambiguous bars, `reward = 0.0` when no valid entry exists
    - _Bug_Condition: `isBugCondition_Bug2` where `reward_WAIT > MAX(reward_CALL_avg, reward_PUT_avg) + epsilon`_
    - _Expected_Behavior: After fix, WAIT reward does not unconditionally dominate; epoch WAIT fraction < 85% once training stabilizes_
    - _Preservation: CALL reward = `fwd_pct - 0.0005` and PUT reward = `-fwd_pct - 0.0005` unchanged; reward clipping to `[-0.05, 0.05]` unchanged_
    - _Requirements: 2.4, 2.5, 2.6, 3.4_

  - [~] 7.2 Add per-epoch reward distribution logging and dominance assertion
    - After each Q epoch's print statement, compute `avg_r = {a: reward_sum[h][a] / max(reward_count[h][a], 1) for a in (H_WAIT, H_CALL, H_PUT)}`
    - Add assertion: `assert avg_r[H_WAIT] <= max(avg_r[H_CALL], avg_r[H_PUT]) + 0.002, f"WAIT reward dominance at horizon {h}"`
    - This catches future reward degeneration early without requiring manual inspection
    - _Requirements: 2.4, 2.5, 2.6_

  - [~] 7.3 Verify Bug 2 exploration test now passes (Property 1: Expected Behavior — WAIT Fraction)
    - **Property 1: Expected Behavior** - WAIT Policy Fraction Below 85%
    - **IMPORTANT**: Re-run the SAME sub-test 1f from task 1 — do NOT write a new test
    - After 3 Q-training epochs with fixed reward, assert `action_counts[0][H_WAIT] / total_actions < 0.85` for at least one horizon
    - **EXPECTED OUTCOME**: Test PASSES (confirms Bug 2 is fixed)
    - _Requirements: 2.4, 2.5, 2.6_

  - [~] 7.4 Verify preservation tests still pass
    - **Property 2: Preservation** - CALL/PUT Reward Computation Unchanged
    - **IMPORTANT**: Re-run the SAME preservation tests from task 2
    - Assert CALL reward = `fwd_pct - 0.0005` for a concrete sample transition; assert PUT reward = `-fwd_pct - 0.0005`
    - **EXPECTED OUTCOME**: Tests PASS

- [ ] 8. Fix Bug 4 — Harden Replay Buffer: Explicit Import, Capacity Alignment, Length Guard (Cell 10)

  - [~] 8.1 Add explicit `import collections` and align `BUFFER_CAPACITY` with deque maxlen
    - Add `import collections` at the top of Cell 10 (do not rely on import from another cell)
    - Set `BUFFER_CAPACITY = 50000` (>= N_steps_per_epoch of 34,829) and use it as `maxlen` in deque init
    - Update the deque initialization guard to also check `len(replay_buffers) == NUM_HORIZONS` and `replay_buffers[0].maxlen == BUFFER_CAPACITY` to handle `NUM_HORIZONS` changes between cell runs
    - Add per-epoch buffer size logging: `print(f"[Epoch {q_epoch+1}] Buffer sizes: {[len(b) for b in replay_buffers]}")`
    - _Bug_Condition: `isBugCondition_Bug4` where buffer is list / inside epoch loop / capacity < N_steps_per_epoch / O(N) eviction_
    - _Expected_Behavior: After one full epoch, `len(replay_buffers[0]) > 30000`; O(1) deque eviction; buffer persists across epoch boundaries_
    - _Preservation: Tuple format `(abs_idx, state, action, reward, next_abs_idx, next_state)` unchanged; CALL/PUT oversampling 3:1 preserved; `_batch_from_index_replay` interface unchanged_
    - _Requirements: 2.9, 2.10, 2.11, 2.12, 3.6_

  - [~] 8.2 Verify Bug 4 exploration test now passes (Property 1: Expected Behavior — Buffer Persists)
    - **Property 1: Expected Behavior** - Replay Buffer Persists Across Epochs with Sufficient Capacity
    - **IMPORTANT**: Re-run the SAME sub-test 1g from task 1 — do NOT write a new test
    - After one full Q-training epoch: assert `len(replay_buffers[0]) > 30000`
    - After two epochs: assert buffer sizes only grow up to `maxlen` (not reset to 0)
    - **EXPECTED OUTCOME**: Test PASSES (confirms Bug 4 is fixed)
    - _Requirements: 2.9, 2.10, 2.11, 2.12_

  - [~] 8.3 Verify preservation tests still pass
    - **Property 2: Preservation** - Replay Tuple Format and Oversampling Unchanged
    - **IMPORTANT**: Re-run the SAME preservation tests from task 2
    - Assert `_batch_from_index_replay` still returns 6-element tuples; assert CALL/PUT oversampling branch still present
    - **EXPECTED OUTCOME**: Tests PASS

- [ ] 9. Fix Bug 7 — Per-Window Feature Normalization in `build_feat_window` (Cells 6 and 7)

  - [~] 9.1 Identify price-level feature indices after `feature_cols` is defined (Cell 6)
    - After `feature_cols = [...]` is computed in Cell 6, add:
      ```python
      _PRICE_FEATURE_NAMES = {"close_5m", "open_5m", "high_5m", "low_5m", "close", "open", "high", "low"}
      _PRICE_FEAT_INDICES = np.array([i for i, col in enumerate(feature_cols) if col in _PRICE_FEATURE_NAMES], dtype=np.int64)
      ```
    - Print the identified indices for verification: `print(f"Price feature indices: {_PRICE_FEAT_INDICES} ({[feature_cols[i] for i in _PRICE_FEAT_INDICES]})")`
    - _Bug_Condition: `isBugCondition_Bug7` where price feature std > 1.0 (absolute scale, not normalized)_
    - _Expected_Behavior: After normalization, price features have mean ≈ 0 and std ≈ 1 within each window_

  - [~] 9.2 Compute training-set normalization statistics (Cell 9, before training loop)
    - After `train_num_matrix` is defined, compute `_feat_mean = np.nanmean(train_num_matrix, axis=0, keepdims=True).astype(np.float32)` and `_feat_std = np.maximum(np.nanstd(train_num_matrix, axis=0, keepdims=True).astype(np.float32), 1e-6)`
    - These training-set statistics are used for non-price features only (stable, no lookahead)
    - _Requirements: 2.17, 2.18_

  - [~] 9.3 Update `build_feat_window` to apply per-window price normalization and training-set normalization for other features (Cell 7)
    - Change signature to `def build_feat_window(num_matrix, abs_idx, q_lookback=None, price_feat_idx=None, feat_mean=None, feat_std=None):`
    - Add `if q_lookback is None: q_lookback = Q_LOOKBACK` as first line (also fixes Bug 5 default binding)
    - After constructing the raw window, add: per-window z-score for price indices using `window[:, price_feat_idx]` own mean/std; training-set z-score for non-price features using `feat_mean`/`feat_std`
    - Add final `return np.clip(window, -10.0, 10.0)`
    - Add a convenience wrapper `_fw(num_matrix, abs_idx)` that binds `_PRICE_FEAT_INDICES`, `_feat_mean`, `_feat_std` from the global scope
    - _Bug_Condition: Raw price levels with GLD mean ≈ 190, std ≈ 10–20_
    - _Expected_Behavior: Normalized price features mean ≈ 0, std ≈ 1 per window; non-price features z-scored to training-set distribution_
    - _Preservation: Output shape `(Q_LOOKBACK, num_features)` unchanged; `feature_cols` length unchanged; existing call sites can use `_fw` wrapper with no signature change_
    - _Requirements: 2.17, 2.18, 3.2, 3.8_

  - [~] 9.4 Update all `build_feat_window` call sites in Cells 10, 11, and `_batch_from_index_replay` to use `_fw` wrapper
    - Replace all `build_feat_window(train_num_matrix, abs_idx, Q_LOOKBACK)` calls in Cell 10 with `_fw(train_num_matrix, abs_idx)`
    - Replace all `build_feat_window(test_matrix, abs_idx, Q_LOOKBACK)` calls in Cell 11 with a test-set-aware wrapper `_fw_test(test_matrix, abs_idx)` that uses the same `_feat_mean`/`_feat_std` from training (no data leakage)
    - Update `_batch_from_index_replay` to pass through normalization params or use wrapper
    - _Requirements: 2.17, 2.18, 3.2_

  - [~] 9.5 Verify Bug 7 exploration test now passes (Property 1: Expected Behavior — Normalized Feature Windows)
    - **Property 1: Expected Behavior** - Price Features Normalized Per Window
    - **IMPORTANT**: Re-run the SAME sub-test 1e from task 1 — do NOT write a new test
    - Call `_fw(train_num_matrix, 500)` (a non-padded window); assert `price_features.mean() ∈ [-0.1, 0.1]` and `price_features.std() ∈ [0.9, 1.1]`
    - Repeat for 5 random absolute indices across different market regimes
    - **EXPECTED OUTCOME**: Test PASSES (confirms Bug 7 is fixed)
    - _Requirements: 2.17, 2.18_

  - [~] 9.6 Verify preservation tests still pass
    - **Property 2: Preservation** - Feature Window Shape and Column Count Unchanged
    - **IMPORTANT**: Re-run the SAME preservation tests from task 2
    - Assert output shape `(Q_LOOKBACK, num_features)` unchanged; assert `len(feature_cols) == 326`
    - **EXPECTED OUTCOME**: Tests PASS

- [~] 10. Checkpoint — Ensure all bug condition tests pass and all preservation tests still pass

  - Run all 7 bug condition sub-tests from task 1 on the fully fixed code
  - Assert each sub-test now PASSES (bug fixed) rather than detecting the bug
  - Run the complete preservation test suite from task 2 on the fully fixed code
  - Assert all preservation tests still PASS (no regressions)
  - Run integration validation:
    - Phase 1: 3 epochs on a small synthetic dataset; assert `strength.std() > 0.05`, `l_q < 0.249`, backbone gradient norm non-zero for all parameters
    - Phase 2: 2 epochs; assert `len(replay_buffers[0]) > 30000`, WAIT fraction < 85%, `Q_LOOKBACK` consistent
    - Phase 3: assert strict gate fires on > 10 test trades (vs 4 before fix)
    - End-to-end: all cells execute in order without manual intervention; Cell 12 checkpoint export completes
  - Ensure all tests pass; ask the user if questions arise.
