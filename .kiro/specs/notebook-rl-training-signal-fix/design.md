# RL Training Notebook Bugfix Design

## Overview

`notebooks/training/axe_signal_shaped_rl_training.ipynb` trains a two-phase model for options trading: Phase 1 is a `SignalMetaNetwork` (multi-task supervised learning for direction, strength, pips, risk, and liquidity) and Phase 2 is a DQN `ExecutorQNetwork` that uses frozen Phase 1 outputs as part of its state. Seven confirmed bugs collectively prevent the pipeline from producing a model better than random chance. Observed symptoms include: Phase 1 validation win rate peaking at ~52.89% (barely above 50% random baseline), strength outputs stuck near 0.528 across all epochs, Q-loss collapsing to 1e-5 (degenerate WAIT policy), and Phase 3 out-of-sample win rate of 49.87% (indistinguishable from coin-flip always-CALL at 49.91% or always-PUT at 50.61%).

This document describes the bug conditions for all seven bugs, the expected correct behaviors, hypothesized root causes based on code inspection, and the implementation and testing strategy. The notebook contains partial fixes already applied (Bugs 1, 2, and 4 have attempts in the current code); the remaining work addresses Bugs 3, 5, 6, and 7 fully, and hardens the partial fixes for Bugs 1, 2, and 4.

---

## Glossary

- **Bug_Condition (C)**: The input condition that triggers one of the seven defective behaviors
- **Property (P)**: The desired correct behavior when the bug condition holds
- **Preservation**: Behaviors that must remain unchanged after the fix — five output heads with correct shapes, frozen Phase 1 weights before Phase 2, PnL-based CALL/PUT rewards, and end-to-end notebook execution
- **`SignalMetaNetwork`**: The Phase 1 model in Cell 6 (`notebooks/training/axe_signal_shaped_rl_training.ipynb`), a multi-branch Conv1D+LSTM+Fusion network with six output heads: direction (`q_head`), strength, pips, risk, liquidity, reversal
- **`ExecutorQNetwork`**: The Phase 2 DQN in Cell 7, dual-input (feature window + 28-dim context), with four independent horizon heads each producing 3 Q-values (WAIT/CALL/PUT)
- **`build_feat_window`**: The helper in Cell 7 that slices `(q_lookback, num_features)` raw feature windows for the Q-network
- **`_rebuild_targets`**: The target-construction function in Cell 9 that computes ATR-normalized soft strength and pips targets from close prices
- **`_fill_ml_targets`**: The function in Cells 8/9 that populates the 22 auxiliary ML target arrays, falling back to zeros if columns are absent from the CSV
- **`Q_LOOKBACK`**: The lookback length (in bars) for `ExecutorQNetwork` feature windows — defined in Cells 2, 7, and 10
- **`backbone_input`**: The concatenated branch output `[b1_out, b2_out, b3_out]` in `SignalMetaNetwork.forward()` that feeds both the fusion stack and the auxiliary heads
- **`target_dir_5m`**: The binary direction label for the 5-minute horizon, computed as `(forward_move_1 > 0)`

---

## Bug Details

### Bug 1 — Missing Auxiliary ML Targets (35% Loss Budget Corrupted)

The 22 ML auxiliary target arrays (`adv_target_*`, `Volatility_Regime_next`, `vel_bull_fwd_8`, etc.) default to `np.zeros` when the corresponding columns are absent from the CSV. The auxiliary losses `l_zone`, `l_vol`, and `l_vel` then train the strength, pips, and liquidity heads toward zero, directly opposing any real signal.

**Current State**: Cell 8 synthesizes 14 of 22 targets from available CSV columns (ATR, forward returns, volume delta). Cell 9 gates the auxiliary losses — `l_zone/l_vol/l_vel` are only applied when `abs(target).mean() > 1e-7`. The remaining 8 targets (zone index, zone bars, zone distance, zone type, and 4 CSM keys) stay zero because they require external data not in the CSV.

**Remaining Issue**: The zero-gating is correct for the permanently-missing columns, but the synthesis logic in Cell 8 can produce arrays that are non-zero in aggregate yet near-zero for certain rows (e.g., `vol_regime` starts as rolling NaN for the first 5 bars). The assert on `non_zero_keys` in Cell 9 passes, but training rows near index boundaries may still receive near-zero targets for those features.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug1(df, ml_targets)
  INPUT: df (loaded CSV DataFrame), ml_targets (dict of 22 target arrays)
  OUTPUT: boolean

  RETURN (
    COUNT(k IN ml_targets WHERE abs(ml_targets[k]).mean() <= 1e-7) > 0
    OR COUNT(k IN ml_targets WHERE ISNAN(ml_targets[k]).any()) > 0
  )
  -- Bug fires when any target array is all-zero or contains NaN
END FUNCTION
```

**Examples:**
- CSV missing `vel_bull_fwd_8` → array filled with zeros → `l_vel` pulls pips head toward 0 for entire training run
- CSV missing all 4 `adv_target_*` zone columns → zone index/bars/distance/type all zero → `l_zone` contributes no useful gradient (currently gated out, so benign when gating works)
- CSV present but `Volatility_Regime_next` has NaN rows → `nan_to_num` converts to 0 → those training rows push strength head toward zero

### Bug 2 — Degenerate WAIT Reward (Q-Policy Collapses to WAIT-Always)

The original code returns `reward = +0.001` unconditionally for WAIT. Since CALL/PUT actions average −0.0005 on a coin-flip market, the Q-network correctly learns WAIT-always as the mathematically optimal strategy.

**Current State**: Cell 10 has replaced the fixed `+0.001` with context-sensitive WAIT rewards:
- `0.0` when no valid entry is available (hard mask blocks CALL and PUT)
- `-0.5 * abs(fwd_pct)` when `meta_strength >= 0.58` and `abs(fwd_pct) >= 0.001` (penalizes missing a clear move)
- `+0.0002` when `abs(fwd_pct) < 0.0005` (rewards correctly avoiding chop)
- `0.0` otherwise (ambiguous bar)

**Remaining Issue**: The penalty branch `(-0.5 * abs(fwd_pct))` uses `meta_strength` from the Phase 1 model precomputed before Phase 2 training begins. If Phase 1 produced degenerate strength outputs (stuck near 0.528), the `>= 0.58` threshold will rarely fire, and the penalty branch becomes a dead path — WAIT reward defaults to `0.0`, which still dominates a coin-flip CALL/PUT average near zero. The fix must be robust to a Phase 1 model that hasn't yet learned meaningful strength.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug2(reward_WAIT, reward_CALL_avg, reward_PUT_avg)
  INPUT: rewards for each action type
  OUTPUT: boolean

  RETURN reward_WAIT > MAX(reward_CALL_avg, reward_PUT_avg) + epsilon
  -- Bug fires when WAIT dominates unconditionally, collapsing policy
END FUNCTION
```

**Examples:**
- WAIT = +0.001, CALL_avg = −0.0005, PUT_avg = −0.0005 → Q converges to WAIT-always (original bug, confirmed)
- WAIT = 0.0, CALL/PUT = 0.0 average → Q has no gradient signal to differentiate actions → slow degenerate convergence to WAIT (residual risk with current fix)
- WAIT = −0.001 when mask allows trades, CALL/PUT = +fwd_pct − 0.0005 → Q correctly learns to take profitable entries

### Bug 3 — Coin-Flip Direction Labels (~50.3% Positive Rate)

`target_dir_5m = (forward_move_1 > 0)` is the unconditional next-bar direction for GLD. This is approximately 50/50 by market microstructure. The q_head trained on these labels with MSE loss plateaus at MSE ≈ 0.25 (the theoretical floor for a binary {0, 1} target under MSE when labels are 50/50 iid). No amount of training epochs can improve this because the labels contain no discriminative information.

**Current State**: Cell 9 still uses `(df["forward_move_1"] > 0).astype(np.float32)` for `target_dir_5m`. No conditional filtering or zone-anchored weighting is applied. This remains unaddressed in the current notebook.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug3(direction_labels)
  INPUT: direction_labels array of {0, 1} floats
  OUTPUT: boolean

  positive_rate = mean(direction_labels)
  RETURN abs(positive_rate - 0.5) < 0.02
  -- Bug fires when label positive rate is within 2% of 50% (coin flip)
END FUNCTION
```

**Examples:**
- GLD 5m, 50,000 rows: `target_dir_5m.mean() ≈ 0.503` → MSE floor ≈ 0.25, confirmed in notebook output
- Using zone-conditional filter: only label bars where price is within ATR×0.75 of a support/resistance zone → expected positive rate shifts toward 55%+ if zone analysis has any predictive power
- Using ATR-scaled forward return threshold: `target_dir_5m = (forward_move_1 > 0.3 * ATR)` → labels exclude micro-moves, positive rate drops toward ~35%, but the surviving labels are more discriminative

### Bug 4 — Replay Buffer Destroyed Every Epoch / O(N) Eviction / Under-Capacity

The original code placed `replay_buffers = [[] for _ in range(NUM_HORIZONS)]` inside the epoch loop, wiping all accumulated experience at epoch start. Buffer capacity of 4,000 across 34,829 steps/epoch means <3% retention. `pop(0)` is O(N).

**Current State**: Cell 10 now uses:
```python
if 'replay_buffers' not in dir() or not isinstance(replay_buffers[0], collections.deque):
    replay_buffers = [collections.deque(maxlen=50000) for _ in range(NUM_HORIZONS)]
```
This is outside the epoch loop, persists across cell re-runs, and uses `deque(maxlen=50000)` for O(1) eviction. CALL/PUT oversampling at 3:1 is also implemented. However, Cell 10 line 1 re-defines `BUFFER_CAPACITY = 3000` (unused for the deque init, which hardcodes 50000), and the `collections` import is not explicitly shown in Cell 10 — it may rely on an import in another cell.

**Remaining Issue**: The `collections` module must be explicitly imported. The hardcoded `maxlen=50000` is inconsistent with the `BUFFER_CAPACITY` variable. The deque guard should also handle the case where `replay_buffers` exists but has wrong `NUM_HORIZONS` length (e.g., when `NUM_HORIZONS` changes between cell runs).

**Formal Specification:**
```
FUNCTION isBugCondition_Bug4(buffer_init_location, buffer_type, buffer_capacity, eviction_complexity)
  INPUT: buffer configuration
  OUTPUT: boolean

  RETURN (
    buffer_init_location == "inside_epoch_loop"
    OR buffer_type == "list"
    OR buffer_capacity < N_steps_per_epoch
    OR eviction_complexity == "O(N)"
  )
END FUNCTION
```

**Examples:**
- `replay_buffers = [[] for _ ...]` inside epoch loop → 0 experience persists across epochs (original bug)
- `deque(maxlen=4000)` outside loop → O(1) eviction, but capacity 4000 < 34829 steps → <3% retained per epoch (partially fixed, capacity still too small)
- `deque(maxlen=50000)` outside loop → full epoch retained, O(1) eviction (current state — correct capacity)

### Bug 5 — Q_LOOKBACK Defined Inconsistently Across Cells (Three Conflicting Values)

`Q_LOOKBACK` is defined three times with three different values:
- Cell 2: `Q_LOOKBACK = 150`
- Cell 7: `Q_LOOKBACK = 300` (overrides Cell 2 when Cell 7 executes)
- Cell 10: `Q_LOOKBACK = 150` then `Q_LOOKBACK = 64` (two assignments in the same cell, line 0 and line 11)

`ExecutorQNetwork` is instantiated with `q_lookback=Q_LOOKBACK` at Cell 10 line 157, which reads the value of `64` (last assignment in Cell 10). However, `build_feat_window` uses `q_lookback: int = Q_LOOKBACK` as a default argument — this default is bound at function-definition time (Cell 7 execution), which captures `300`. If Cell 7 runs after Cell 10, it will capture 64; if before, it captures whatever value existed when Cell 7 was executed.

The `ExecutorQNetwork`'s Conv1D/LSTM architecture uses `AdaptiveAvgPool1d` which silently accepts any input length — no crash, no warning.

**Current State**: No consolidation has been applied. All three conflicting assignments remain.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug5(q_lookback_network, q_lookback_window_fn, q_lookback_train)
  INPUT: the Q_LOOKBACK value used at network instantiation, at window construction, and in training
  OUTPUT: boolean

  RETURN NOT (q_lookback_network == q_lookback_window_fn == q_lookback_train)
  -- Bug fires when any of the three Q_LOOKBACK uses disagree
END FUNCTION
```

**Examples:**
- Cell 7 runs → `Q_LOOKBACK = 300` → `build_feat_window` default bound to 300 → Cell 10 runs → `Q_LOOKBACK = 64` → network instantiated with 64, but build_feat_window still uses 300 as default → shape mismatch
- Cell 10 runs first → `Q_LOOKBACK = 64` → Cell 7 runs → redefines to 300, rebinds default → all subsequent calls use 300 but network expects 64
- Single definition in Cell 2 used everywhere → consistent (desired state)

### Bug 6 — Auxiliary Heads Fully Detached, Blocking Backbone Regularization

In `SignalMetaNetwork.forward()` (Cell 6, lines 126–146), the backbone branches are detached before feeding auxiliary heads:

```python
# Line 127-128: aux1/aux2 detached
aux1 = self.aux1_head(b1_out.detach())
aux2 = self.aux2_head(b2_out.detach())

# Line 142: pips/risk/liq/rev detached
branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1).detach())
pips      = self.pips_head(self.pips_ln(self.pips_proj(branch_cat)))
risk      = self.risk_head(self.risk_ln(self.risk_proj(branch_cat)))
liquidity = self.liquidity_head(self.liq_ln(self.liq_proj(branch_cat)))
reversal  = self.reversal_head(self.rev_ln(self.rev_proj(branch_cat)))
```

The gradients from `l_pips`, `l_risk`, `l_liq`, and `l_rev` never reach the Conv1D+LSTM towers. Only `l_q` (on 50/50 coin-flip direction labels) and `l_str` (partially corrupted when `l_vol`/`l_vel` are zero) propagate through `feat → fusion → backbone`. The backbone effectively learns only from the coin-flip signal, which explains the strength output stuck near 0.528 (Sigmoid(≈0)).

**Current State**: The detach calls are still present and the comment explicitly says "Zero Gradient Interference via feat.detach()". This is the design intent, but it is architecturally incorrect — it means 4 of the 6 multi-task losses contribute zero backbone regularization.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug6(gradient_path_pips, gradient_path_risk, gradient_path_liq, gradient_path_rev)
  INPUT: boolean flags indicating whether each auxiliary head's gradients reach the backbone
  OUTPUT: boolean

  RETURN NOT (gradient_path_pips AND gradient_path_risk AND gradient_path_liq AND gradient_path_rev)
  -- Bug fires when any auxiliary head is disconnected from backbone training
END FUNCTION
```

**Examples:**
- `branch_cat = ...(detach())` → pips/risk/liq/rev gradients stop at `branch_cat`, never enter Conv1D/LSTM → backbone trains only on coin-flip `l_q` + partially-useful `l_str`
- `branch_cat = ...(no detach)` → all 6 heads contribute backbone gradients → richer temporal representations, diversity of training signal
- Partial detach (only reversal detached) → 5 of 6 heads train backbone → acceptable compromise if reversal target quality is suspect

### Bug 7 — Feature Windows Not Normalized Per-Window

`build_feat_window` (Cell 7, lines 121–134) returns raw feature values without normalization:

```python
def build_feat_window(num_matrix, abs_idx, q_lookback=Q_LOOKBACK):
    start = abs_idx - q_lookback + 1
    if start >= 0:
        return num_matrix[start: abs_idx + 1].astype(np.float32)
    # left pad with zeros
    window = np.zeros((q_lookback, num_matrix.shape[1]), dtype=np.float32)
    window[-len(available):] = available
    return window
```

The 326 feature columns include absolute price levels (`close_5m`, `open_5m`, `high_5m`, `low_5m`) that vary with market level. GLD trades at ~$175–$240 over the dataset range. A Conv1D filter that sees raw close prices learns absolute level information rather than structural patterns (momentum, breakouts, range compression). This prevents generalization across different time periods and market regimes.

**Current State**: No per-window normalization has been applied. `train_num_matrix` is filled with `nan_to_num` but otherwise raw.

**Formal Specification:**
```
FUNCTION isBugCondition_Bug7(feature_window)
  INPUT: (q_lookback, num_features) feature array
  OUTPUT: boolean

  price_features = feature_window[:, PRICE_FEATURE_INDICES]
  price_std = std(price_features)
  RETURN price_std > 1.0  -- absolute scale, not relative
  -- Bug fires when price features carry absolute level information
END FUNCTION
```

**Examples:**
- GLD window Jan 2022: close values ~175–177 → Conv1D encodes "GLD is at 176" not "close is 0.3 ATR above open"
- GLD window Jan 2024: close values ~196–198 → same structural pattern encoded as different value → filter generalizes poorly
- After z-score normalization per window → close values ~[−1.5, 1.5] relative to window mean → same structural pattern produces same Conv1D activation regardless of market level

---

## Expected Behavior

### Preservation Requirements

**Unchanged Behaviors:**
- `SignalMetaNetwork` forward pass produces outputs for all five heads (direction, strength, pips, risk, liquidity) with the same tensor shapes and value ranges as before the fix
- Phase 1 weights are frozen before Phase 2 training begins (the two-phase separation is preserved)
- CALL/PUT action rewards continue to be computed from realized PnL (`fwd_pct ± 0.0005` spread), not from a synthetic signal
- Replay sampling continues to produce `(state, action, reward, next_state, done)` tuples compatible with the existing Q-learning update step
- The strict gate in Phase 3 continues to apply the same confidence threshold logic
- Feature engineering continues to produce the same 326 columns in the same order; normalization is a post-processing step applied in `build_feat_window`, not a change to feature selection
- Notebook cells continue to execute end-to-end without manual intervention between phases
- `AdaptiveAvgPool1d` continues to be used in the Q-network architecture

**Note:** The actual expected correct behaviors per bug are defined in the Correctness Properties section below (Properties 1–7). This section focuses on what must NOT change.

---

## Hypothesized Root Cause

### Bug 1 — Zero ML Targets
The CSV exported from the data pipeline (`train_50k.csv`) was generated without the auxiliary target computation step. The auxiliary columns were never written to the file. `_fill_ml_targets` correctly falls back to zeros, but zeros are treated as valid training signal by the loss function (no gating existed originally). The Cell 8 synthesis and Cell 9 gating represent a correct partial fix; the remaining gap is that 8 of 22 targets (zone index/bars/distance/type and 4 CSM columns) cannot be synthesized from OHLCV data alone and must either be generated offline or permanently excluded from the loss.

### Bug 2 — Degenerate WAIT Reward
The original `+0.001` WAIT reward was chosen as a "do-nothing is slightly positive" incentive without accounting for the fact that CALL/PUT returns on a coin-flip market are near-zero negative in expectation. The fix in Cell 10 is structurally correct but depends on `meta_strengths` from Phase 1. If Phase 1 is itself broken (Bug 6 means backbone trains only on coin-flip signal), the strength scores will be near 0.528 and the penalty branch for high-confidence skipped moves rarely fires. The bugs are coupled: Bug 6 undermines Bug 2's fix.

### Bug 3 — Coin-Flip Labels
The label construction `(forward_move_1 > 0)` is the standard approach for a directional model on any asset, but it is only predictive if the underlying market has some autocorrelation or if the model can condition on features that predict direction. GLD's 5-minute bars are close to iid under unconditional labeling. The fix requires either (a) conditioning labels on zone proximity so only "near a S/R zone" bars are labeled, or (b) using an ATR threshold to exclude micro-moves, or (c) switching from MSE to `BCEWithLogitsLoss` with sample weights that up-weight zone-anchored bars.

### Bug 4 — Replay Buffer
The original placement inside the epoch loop was almost certainly a copy-paste error from pseudocode or an initialization pattern used outside a loop. The O(N) `list.pop(0)` is the standard Python list approach that works in small experiments but degrades at scale. Both are classic beginner mistakes in DQN implementations. The current fix (deque outside epoch loop) is correct.

### Bug 5 — Q_LOOKBACK Multi-Definition
The constant was tuned over multiple editing sessions: initially 150, then increased to 300 to capture a full zone lifecycle, then forced back to 64 to avoid OOM on Kaggle GPUs. Each session added a new definition without removing the previous one. The Cell 7 default argument binding makes this particularly dangerous — Python captures the default value at function-definition time, so which value ends up bound depends on cell execution order.

### Bug 6 — Detached Auxiliary Heads
The comment "Zero Gradient Interference via feat.detach()" suggests this was an intentional design decision to prevent auxiliary task gradients from "interfering" with the primary direction head. This is a common misapplication of the gradient-stopping pattern: `detach()` is appropriate when you want a head to train without affecting an upstream encoder that is already well-trained (e.g., a frozen backbone). In Phase 1, the backbone is not pre-trained — it is jointly trained with all heads. Detaching auxiliary heads removes the most informative training signal (pips, risk, liquidity are more discriminative than coin-flip direction) and leaves the backbone to train only on noise.

### Bug 7 — No Per-Window Normalization
The pipeline normalizes using `nan_to_num` (NaN replacement) but never applies z-score or min-max normalization to the feature matrix. Price-level features were likely included because "more information is better," but absolute values create a distribution shift between training windows and make the Conv1D filters encode level rather than structure. Rolling standardization (using only past data) is the standard fix and is numerically safe since it only requires the window's own statistics.

---

## Correctness Properties

Property 1: Bug Condition — ML Targets Are Non-Zero and Meaningful

_For any_ training step where `_fill_ml_targets` is called, the synthesized target arrays SHALL have `abs(v).mean() > 1e-6` for all 14 synthesizable keys, and the remaining 8 permanently-missing keys SHALL be excluded from the loss computation via explicit gating. No auxiliary loss SHALL train any head toward zero unless the true target value is genuinely zero for that sample.

**Validates: Requirements 2.1, 2.2, 2.3**

Property 2: Bug Condition — WAIT Reward Does Not Unconditionally Dominate

_For any_ state where the hard action mask permits CALL or PUT, the WAIT reward SHALL be less than or equal to the maximum expected CALL/PUT reward at that state. The Q-network SHALL NOT converge to a WAIT-only policy when profitable trades exist. Epoch-level WAIT action fraction SHALL be below 85% once training stabilizes.

**Validates: Requirements 2.4, 2.5, 2.6**

Property 3: Bug Condition — Direction Labels Are Discriminative

_For any_ direction label array constructed from the training CSV, the validation MSE on the direction head SHALL decrease below 0.24 within 30 epochs (compared to the 0.25 floor for 50/50 iid labels). The positive label rate SHALL be validated to differ meaningfully from 0.50 ± 0.02, or a zone-conditional weighting scheme SHALL be applied that makes the effective label distribution learnable.

**Validates: Requirements 2.7, 2.8**

Property 4: Bug Condition — Replay Buffer Persists and Has Sufficient Capacity

_For any_ Q-training run, the replay buffer SHALL be initialized exactly once before the epoch loop, SHALL use a `collections.deque` with O(1) eviction, SHALL have `maxlen ≥ N_steps_per_epoch` (≥ 34,829), and SHALL not be wiped between epochs. CALL/PUT transitions SHALL be oversampled at ≥ 3:1 relative to WAIT transitions during batch construction.

**Validates: Requirements 2.9, 2.10, 2.11, 2.12**

Property 5: Bug Condition — Q_LOOKBACK Is Consistent

_For any_ execution of notebook cells in any order, the value of `Q_LOOKBACK` used at `ExecutorQNetwork` instantiation, at `build_feat_window` default binding, and at training window construction SHALL be identical. A single canonical definition SHALL exist and SHALL be the only source of truth.

**Validates: Requirements 2.13, 2.14**

Property 6: Bug Condition — Auxiliary Head Gradients Reach the Backbone

_For any_ backward pass through `SignalMetaNetwork`, the gradients from `l_pips`, `l_risk`, `l_liq`, and `l_rev` SHALL flow through to the Conv1D and LSTM parameters. The `branch_cat` tensor fed to auxiliary heads SHALL NOT be detached from the computational graph. Backbone parameter gradient norms after a training step SHALL be non-zero and SHALL reflect contributions from more than just `l_q` and `l_str`.

**Validates: Requirements 2.15, 2.16**

Property 7: Bug Condition — Feature Windows Are Per-Window Normalized

_For any_ feature window returned by `build_feat_window`, price-level features (close, open, high, low) SHALL be z-score normalized using the window's own mean and standard deviation, computed from only the bars within that window (no lookahead). The normalized values SHALL have mean ≈ 0 and std ≈ 1 across the window. All other features (indicators, volume ratios, derived features) SHALL be normalized on a per-feature basis using statistics from the training set computed before training begins, to avoid distribution shift between training and inference.

**Validates: Requirements 2.17, 2.18**

---

## Fix Implementation

### Assumptions

All five of the remaining/incomplete fixes assume the root cause analyses above are correct. The implementations below are ordered by coupling: Bug 6 (detach removal) should be done first because it directly affects Bug 2's fix viability (strength outputs need to be meaningful for the WAIT penalty branch to fire). Bug 3 (direction labels) should be done second. Bugs 5, 7, and the hardening of Bugs 1 and 4 can be done in any order thereafter.

---

### Fix 6 — Remove `.detach()` from Auxiliary Head Inputs (Cell 6)

**File**: `notebooks/training/axe_signal_shaped_rl_training.ipynb`, Cell 6 (`SignalMetaNetwork.forward`)

**Lines to change**: 127–128, 141–142

**Before:**
```python
# Lines 126-130
# Aux heads (detached)
aux1 = self.aux1_head(b1_out.detach())
aux2 = self.aux2_head(b2_out.detach())
aux1_sg = aux1.detach()
aux2_sg = aux2.detach()

# Lines 141-146
# Private Aux Heads — detached so their gradients cannot contaminate branch towers
branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1).detach())
pips      = self.pips_head(self.pips_ln(self.pips_proj(branch_cat)))
risk      = self.risk_head(self.risk_ln(self.risk_proj(branch_cat)))
liquidity = self.liquidity_head(self.liq_ln(self.liq_proj(branch_cat)))
reversal  = self.reversal_head(self.rev_ln(self.rev_proj(branch_cat)))
```

**After:**
```python
# Aux heads (gradient flows to branches)
aux1 = self.aux1_head(b1_out)
aux2 = self.aux2_head(b2_out)
aux1_sg = aux1.detach()   # still detach before fusion concat to avoid double-counting
aux2_sg = aux2.detach()

# Auxiliary Heads — connected to backbone for regularization
branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1))
pips      = self.pips_head(self.pips_ln(self.pips_proj(branch_cat)))
risk      = self.risk_head(self.risk_ln(self.risk_proj(branch_cat)))
liquidity = self.liquidity_head(self.liq_ln(self.liq_proj(branch_cat)))
reversal  = self.reversal_head(self.rev_ln(self.rev_proj(branch_cat)))
```

**Rationale**: `aux1_sg` and `aux2_sg` remain detached before being concatenated into `fusion_in` at line 133, so the fusion stack doesn't receive gradient through those auxiliary paths (avoiding double-counting). The backbone Conv1D+LSTM branches receive gradient from all six heads through `b1_out`, `b2_out`, `b3_out`, and `branch_cat`.

---

### Fix 3 — Zone-Conditional Direction Labels (Cell 9)

**File**: `notebooks/training/axe_signal_shaped_rl_training.ipynb`, Cell 9

**Function**: Target construction block, after `_rebuild_targets` is called

**Specific Changes**:

1. **Replace unconditional binary label with ATR-threshold label**: Exclude micro-moves by requiring `|forward_move_1| > threshold * ATR` before assigning a label. Rows below threshold are masked out of the direction loss (not assigned label 0 — they are simply excluded from `l_q` via a sample weight of 0).

```python
# Compute zone-proximity and ATR-threshold weights for direction head
atr_vals = train_df[atr_col].values if atr_col else (
    pd.Series(train_df[high_col].values - train_df[low_col].values).rolling(14, min_periods=1).mean().values
)
atr_vals = np.maximum(np.nan_to_num(atr_vals, nan=0.001), 1e-4).astype(np.float32)

# Label is valid only if the move is > 0.3 ATR (exclude microstructure noise)
DIRECTION_THRESHOLD = 0.30  # tunable; 0.3 ATR excludes ~40-50% of smallest moves
fwd_move_1 = train_df["forward_move_1"].values.astype(np.float32)
direction_weight = (np.abs(fwd_move_1) > DIRECTION_THRESHOLD * atr_vals).astype(np.float32)
# direction_weight = 0 for micro-move bars → their l_q contribution = 0
```

2. **Apply weights to `l_q` in the training loop**:

```python
# In the training loop, replace:
l_q = nn.MSELoss()(q_vals, y_q_t)
# With weighted MSE:
w_t = torch.tensor(direction_weight[ti], dtype=torch.float32, device=device).unsqueeze(1).expand_as(q_vals)
l_q = (w_t * (q_vals - y_q_t) ** 2).sum() / (w_t.sum() + 1e-6)
```

3. **Validate label discriminability** before training:

```python
effective_pos_rate = train_df.loc[direction_weight.astype(bool), "target_dir_5m"].mean()
assert abs(effective_pos_rate - 0.5) > 0.02, \
    f"Filtered direction labels still ~50/50 ({effective_pos_rate:.3f}). Increase DIRECTION_THRESHOLD."
print(f"[Bug3-fix] Effective direction label positive rate: {effective_pos_rate:.3f} "
      f"(on {direction_weight.sum():.0f} / {len(direction_weight)} bars)")
```

---

### Fix 5 — Consolidate Q_LOOKBACK to Single Definition (Cells 2, 7, 10)

**File**: `notebooks/training/axe_signal_shaped_rl_training.ipynb`, Cells 2, 7, and 10

**Specific Changes**:

1. **Keep the single authoritative definition in Cell 2** (the configuration cell):
   ```python
   Q_LOOKBACK = 150  # single definition — change only here
   ```

2. **Remove the re-definition from Cell 7** (lines 3 and 15):
   - Delete the line `Q_LOOKBACK = 300`
   - Update the comment to reference Cell 2

3. **Remove both re-definitions from Cell 10** (lines 0 and 11):
   - Delete `Q_LOOKBACK = 150` and `Q_LOOKBACK = 64`
   - Add an assertion to catch stale execution order:
   ```python
   assert 'Q_LOOKBACK' in dir(), "Run Cell 2 (config) before Cell 10"
   print(f"[Config check] Q_LOOKBACK = {Q_LOOKBACK}")
   ```

4. **Re-bind `build_feat_window` default** in Cell 7: Since Python captures default argument values at function-definition time, move the `build_feat_window` definition to Cell 10 (after the Cell 2 `Q_LOOKBACK` is confirmed), or use `q_lookback=None` with a `if q_lookback is None: q_lookback = Q_LOOKBACK` inside the function body (which evaluates at call time).

---

### Fix 1 — Harden ML Target Synthesis and Gating (Cells 8 and 9)

**File**: `notebooks/training/axe_signal_shaped_rl_training.ipynb`, Cell 8

**Revised approach**: Zone targets (`adv_target_next_zone_*`) are synthesized from `snr_support_5m`, `snr_resistance_5m`, `snr_dist_support_5m`, `snr_dist_resistance_5m` columns already in the CSV. A forward-looking loop over the next 12 bars computes minimum ATR-normalized distance to zones, bars-to-proximity, and zone type. A new `_fill_ml_targets_phase1()` function replaces the bare `_fill_ml_targets()` redefinition in Phase 1 to prevent zero-overwrite: it checks if the synthesizer is available and falls back to it when CSV columns are all-zero. This synthesizes all 18 non-CSM targets (previously only 14) and makes all zone targets active in the loss.

**Specific Changes**:

1. **Add zone synthesis from SNR CSV columns in Cell 8** (`_synthesize_ml_targets`): Use `snr_support_5m`, `snr_resistance_5m`, `snr_dist_support_5m`, `snr_dist_resistance_5m` from the CSV. A forward-looking loop over the next 12 bars computes `min_dist_fwd` (minimum ATR-normalized distance to nearest zone), `bars_to_zone` (bars until zone proximity threshold of 1.5 ATR), `zone_type` (0=support, 1=resistance), and `zone_idx` (percentile bucket 0–15 of the midpoint level). This populates all 4 zone targets, previously left as zero.

2. **Replace `_fill_ml_targets()` re-definition in Phase 1 (Cell 9) with `_fill_ml_targets_phase1()`**: Prefers CSV columns; falls back to the Cell-8 synthesizer when CSV columns are entirely zero; never leaves synthesizable keys zeroed. Guards with `assert _nz >= 6` to catch silent failures at training start.

---

### Fix 2 — Strengthen WAIT Reward Robustness (Cell 10)

**File**: `notebooks/training/axe_signal_shaped_rl_training.ipynb`, Cell 10

**Alternative fix (preferred)**: **Signal-quality shaping approach**: Instead of only changing the WAIT penalty, introduce `compute_signal_quality_score(action, meta_strength_h, dir_flag, supp_dist, res_dist, vol_delta_ratio, row)` which computes a quality score [0,1] from 5 components: meta conviction (25%), directional alignment with RSI-diff if available (25%), zone proximity using CSV `snr_dist_*` columns (20%), volume confirmation (10%), MTF SNR confluence and DXY alignment (20%). Then `compute_shaped_reward(base_reward, signal_quality)` adds `ALIGNMENT_WEIGHT * SHAPING_SCALE * quality_centered` to any base reward. This rewards high-quality CALL/PUT trades more than low-quality ones, raising the ceiling rather than only lowering the WAIT floor.

**Specific Changes**:

1. **Decouple WAIT penalty from meta_strength** (which may be degenerate from a broken Phase 1): Replace the `meta_strength >= 0.58` threshold with a direct signal-quality proxy based on `abs(fwd_pct)` and ATR, not meta_strength:

   ```python
   else:  # WAIT
       if h_mask[H_CALL] == 0 and h_mask[H_PUT] == 0:
           reward = 0.0
       else:
           abs_move = abs(fwd_pct)
           atr_norm = abs_move / max(abs(atr_vals[i]) / exec_ctxs[i].current_price, 1e-6)
           if atr_norm >= 1.5:
               # Large directional move was available and skipped — penalize
               reward = -0.5 * abs_move
           elif abs_move < 0.0003:
               # Flat bar — WAIT was correct
               reward = 0.0002
           else:
               reward = 0.0
   ```

2. **Add reward distribution logging per epoch** to catch future degeneration early:
   ```python
   # After each epoch print:
   for h in range(NUM_HORIZONS):
       avg_r = {a: (reward_sum[h][a] / reward_count[h][a]) if reward_count[h][a] > 0 else 0.0
                for a in (H_WAIT, H_CALL, H_PUT)}
       assert avg_r[H_WAIT] <= avg_r[H_CALL] + 0.002 or avg_r[H_WAIT] <= avg_r[H_PUT] + 0.002, \
           f"WAIT reward dominance detected at horizon {h} — check reward function"
   ```

---

### Fix 4 — Harden Replay Buffer (Cell 10)

**File**: `notebooks/training/axe_signal_shaped_rl_training.ipynb`, Cell 10

**Specific Changes**:

1. **Add explicit `import collections`** at top of Cell 10
2. **Align `BUFFER_CAPACITY` variable with the deque maxlen**:
   ```python
   BUFFER_CAPACITY = 50000  # >= N_steps_per_epoch (34829)
   replay_buffers = (
       replay_buffers
       if 'replay_buffers' in dir()
           and isinstance(replay_buffers, list)
           and len(replay_buffers) == NUM_HORIZONS
           and isinstance(replay_buffers[0], collections.deque)
           and replay_buffers[0].maxlen == BUFFER_CAPACITY
       else [collections.deque(maxlen=BUFFER_CAPACITY) for _ in range(NUM_HORIZONS)]
   )
   ```
3. **Verify buffer state** at start of each epoch:
   ```python
   print(f"[Epoch {q_epoch+1}] Buffer sizes: {[len(b) for b in replay_buffers]}")
   ```

---

### Fix 7 — Per-Window Feature Normalization (Cell 7)

**File**: `notebooks/training/axe_signal_shaped_rl_training.ipynb`, Cell 7

**Function**: `build_feat_window`

**Specific Changes**:

1. **Identify price-level feature indices** once during Cell 6 setup (after `feature_cols` is defined):
   ```python
   _PRICE_FEATURE_NAMES = {"close_5m", "open_5m", "high_5m", "low_5m", "close", "open", "high", "low"}
   _PRICE_FEAT_INDICES = np.array([
       i for i, col in enumerate(feature_cols) if col in _PRICE_FEATURE_NAMES
   ], dtype=np.int64)
   _OTHER_FEAT_INDICES = np.array([
       i for i in range(len(feature_cols)) if i not in set(_PRICE_FEAT_INDICES)
   ], dtype=np.int64)
   ```

2. **Compute training-set feature statistics** (Cell 9, before training loop):
   ```python
   _feat_mean = np.nanmean(train_num_matrix, axis=0, keepdims=True).astype(np.float32)
   _feat_std  = np.nanstd(train_num_matrix, axis=0, keepdims=True).astype(np.float32)
   _feat_std  = np.maximum(_feat_std, 1e-6)
   ```

3. **Update `build_feat_window` to normalize**:
   ```python
   def build_feat_window(num_matrix, abs_idx, q_lookback=Q_LOOKBACK,
                         price_feat_idx=None, feat_mean=None, feat_std=None):
       start = abs_idx - q_lookback + 1
       if start >= 0:
           window = num_matrix[start: abs_idx + 1].astype(np.float32)
       else:
           window = np.zeros((q_lookback, num_matrix.shape[1]), dtype=np.float32)
           available = num_matrix[: abs_idx + 1]
           window[-len(available):] = available.astype(np.float32)
   
       # Per-window z-score for price-level features (no lookahead: only uses window's own stats)
       if price_feat_idx is not None and len(price_feat_idx) > 0:
           price_slice = window[:, price_feat_idx]
           w_mean = price_slice.mean(axis=0, keepdims=True)
           w_std  = np.maximum(price_slice.std(axis=0, keepdims=True), 1e-6)
           window[:, price_feat_idx] = (price_slice - w_mean) / w_std
   
       # Training-set z-score for non-price features (stable statistics)
       if feat_mean is not None and feat_std is not None:
           if len(price_feat_idx) < window.shape[1]:  # there are non-price features
               other_idx = [i for i in range(window.shape[1]) if i not in set(price_feat_idx)]
               window[:, other_idx] = (window[:, other_idx] - feat_mean[:, other_idx]) / feat_std[:, other_idx]
   
       return np.clip(window, -10.0, 10.0)
   ```

4. **Thread the normalization parameters through all call sites** in Cells 10, 11, and `_batch_from_index_replay`:
   ```python
   # Convenience wrapper using global normalization params
   def _fw(num_matrix, abs_idx, q_lookback=Q_LOOKBACK):
       return build_feat_window(num_matrix, abs_idx, q_lookback,
                                price_feat_idx=_PRICE_FEAT_INDICES,
                                feat_mean=_feat_mean, feat_std=_feat_std)
   ```

---

## Testing Strategy

### Validation Approach

The testing strategy follows a two-phase approach: first, surface counterexamples that demonstrate each bug on unfixed code; then verify each fix works correctly and preserves existing behavior. Because the bugs are coupled (Bug 6 undermines Bug 2's fix; Bug 1 poisons Bug 6's gradient even after detach is removed), tests are ordered to verify the root bugs before the dependent bugs.

### Exploratory Bug Condition Checking

**Goal**: Confirm or refute each root cause hypothesis before applying fixes.

**Test Cases:**

1. **Bug 6 — Backbone Gradient Probe** (run on unfixed code): Add a backward hook to `b1_conv1.weight` and log its gradient norm after one training step. Expected: gradient norm is non-zero but small, driven only by `l_q` and `l_str`. After fix: gradient norm increases substantially as all six head losses contribute.

2. **Bug 3 — Label Positive Rate Check**: Print `train_df["target_dir_5m"].mean()` and `train_df["target_dir_5m"].value_counts()`. Expected: ~50.3% positive rate confirming coin-flip hypothesis. Run MSE on random 50/50 predictions: expected ≈ 0.25, matching the observed plateau.

3. **Bug 5 — Q_LOOKBACK Execution Order Test**: Run cells in order 2 → 7 → 10, then check `build_feat_window.__defaults__[1]` vs `Q_LOOKBACK`. Expected: mismatch (300 vs 64). After fix: match (150 vs 150).

4. **Bug 1 — Synthesized Target Coverage**: Run Cell 8 synthesis and print `{k: v.mean() for k,v in train_ml_targets.items()}`. Expected: 8 permanently-zero keys and 14 non-zero keys. Verify the 8 permanently-zero keys are properly excluded from loss in Cell 9.

5. **Bug 7 — Feature Scale Check**: Print `train_num_matrix[:, price_feat_idx].mean()` and `.std()`. Expected: mean ≈ current GLD price (~190), std ≈ 10–20 (absolute scale). After fix: mean ≈ 0, std ≈ 1 per window.

**Expected Counterexamples:**
- Backbone gradient norm < 0.01 when 4 of 6 heads are detached (Bug 6 confirmed)
- MSE plateau at 0.25 exactly after 10 epochs (Bug 3 confirmed)
- `build_feat_window.__defaults__` captures wrong Q_LOOKBACK value (Bug 5 confirmed)

### Fix Checking

**Goal**: Verify that for all inputs where each bug condition holds, the fixed function produces the expected correct behavior.

**Pseudocode:**
```
FOR ALL input WHERE isBugCondition_BugN(input) DO
  result := fixedFunction(input)
  ASSERT propertyN(result)
END FOR
```

**Concrete Fix Checks:**

1. **Bug 6 fix**: After removing detach, run one backward pass and assert `b1_conv1.weight.grad.abs().mean() > threshold_before_fix`
2. **Bug 3 fix**: Assert `effective_pos_rate` after ATR-threshold filtering differs from 0.50 by > 0.02
3. **Bug 5 fix**: Assert all three Q_LOOKBACK usage sites read the same value
4. **Bug 1 fix**: Assert `len(_REQUIRED_NONZERO_KEYS - {k for k,v in train_ml_targets.items() if v.mean() > 1e-6}) == 0`
5. **Bug 7 fix**: Assert per-window normalized price features have mean ≈ 0 and std ≈ 1
6. **Bug 2 fix**: After 5 Q-training epochs, assert WAIT fraction < 85% for at least one horizon
7. **Bug 4 fix**: Assert `len(replay_buffers[0]) > 30000` after one full epoch

### Preservation Checking

**Goal**: Verify that non-buggy behaviors are unchanged.

**Pseudocode:**
```
FOR ALL input WHERE NOT isBugCondition(input) DO
  ASSERT originalFunction(input) = fixedFunction(input)
END FOR
```

**Concrete Preservation Checks:**

1. Assert `SignalMetaNetwork` output shapes: `q_vals.shape == (B, 4)`, `strength.shape == (B, 4)`, `pips.shape == (B, 4)`, `risk.shape == (B, 8)`, `liq.shape == (B, 2)`, `rev.shape == (B, 1)` — unchanged after detach removal
2. Assert Phase 1 weights are frozen in Phase 2: `net.b1_conv1.weight.requires_grad == True` before freezing, `== False` after
3. Assert CALL/PUT rewards still use `fwd_pct ± 0.0005`: reward for CALL = `fwd_pct - 0.0005`; only WAIT reward changes
4. Assert `_batch_from_index_replay` produces tuples with 6 elements `(fw, ctx, action, reward, nfw, nctx)` — unchanged
5. Assert feature column count unchanged: `len(feature_cols) == 326` after adding normalization
6. Assert `build_feat_window` output shape: `(Q_LOOKBACK, num_features)` — unchanged

### Unit Tests

- Test `_synthesize_ml_targets` on a minimal 100-row synthetic DataFrame: verify 14 non-zero keys, 8 zero keys, no NaN
- Test `build_feat_window` with normalization: input raw price window at market level ~190, output normalized mean ≈ 0, std ≈ 1
- Test WAIT reward function with three branches: (a) hard mask blocks all entries → reward = 0.0, (b) large move missed → reward < 0, (c) flat bar → reward = 0.0002
- Test `replay_buffers` persistence: run epoch loop 3 times, assert buffer sizes only increase up to `maxlen`
- Test `SignalMetaNetwork` backward with detach removed: assert `b1_conv1.weight.grad is not None` and `norm > 0`
- Test Q_LOOKBACK consistency: assert `build_feat_window` default matches `Q_LOOKBACK` variable after Cell 7 executes

### Property-Based Tests

- **Property 1 (ML Targets)**: For any random subset of CSV rows, `_fill_ml_targets` should return arrays with `abs(v).mean() > 1e-6` for all 14 synthesizable keys and exactly 0.0 for all 8 permanently-missing keys
- **Property 4 (Replay Buffer)**: For any sequence of N insertions into `replay_buffers`, after N insertions where N > maxlen, the buffer length equals maxlen (not grows unbounded), and no insert takes O(N) time
- **Property 5 (Q_LOOKBACK)**: Regardless of cell execution order, `ExecutorQNetwork.q_lookback` and `build_feat_window` default SHALL agree — generate random execution orderings and verify
- **Property 7 (Normalization)**: For any feature window generated by `build_feat_window` with normalization enabled, `price_features.mean()` is in `[-0.1, 0.1]` and `price_features.std()` is in `[0.9, 1.1]` — hold across windows from different market regimes (bull, bear, high-volatility)

### Integration Tests

- Run Phase 1 for 3 epochs on a small synthetic dataset and assert: strength `std > 0.05`, direction head MSE drops below 0.249 (below the 50/50 floor), backbone gradient norm is non-zero for all parameters
- Run Phase 2 for 2 epochs and assert: `replay_buffers` sizes exceed 30,000, WAIT fraction < 85%, `Q_LOOKBACK` used in `build_feat_window` matches network `q_lookback`
- Run Phase 3 on test set and assert: strict gate fires on more than 10 trades (vs. 4 before fix), OOS win rate is not statistically worse than 50% at p=0.05
- Run full notebook end-to-end (all cells in order) without manual intervention and assert: Cell 12 checkpoint export completes without error
