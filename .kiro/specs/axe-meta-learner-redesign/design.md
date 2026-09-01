# Design: AXE Meta-Learner Redesign & Notebook-App Parity

## Overview

This document defines the complete architecture, data flow, and parity contract between the Kaggle training notebook (`notebook42ef966279(6).ipynb`) and the in-house backend (`app/core/ml/signal_meta_learner.py`, `app/core/options/q_executor.py`).

Three confirmed bugs block training progress. Additionally, the Q-executor needs a dual-input architecture upgrade to match the notebook, and both systems need a regret buffer for missed-opportunity learning.

---

## Architecture

### Dual-Window SignalMetaNetwork

The meta-learner uses two distinct context windows serving different roles:

```
Input: (B, 150, F)  — 150-bar "prompt" window (sliced from 1000-bar TI output)

        ┌──────────────────────────────────────────────────┐
        │            Branch 1 — Full Sequence (100%)        │
        │   Conv1D(F→64, k=3) → BN → SiLU                 │
        │   Conv1D(64→32, k=3) → BN → SiLU                │
        │   LSTM(32→32) → [b1_last ‖ b1_gap] = b1_out 64  │
        └──────────────────────────────────────────────────┘
        
        ┌─────────────────────┐  ┌─────────────────────┐
        │ Branch 2 (50% mid)  │  │ Branch 3 (30% rec.) │
        │ Conv1D(F→32)→GAP    │  │ Conv1D(F→32)→GAP    │
        │ FC(32→32)           │  │ FC(32→32)           │
        │ b2_out (32-dim)     │  │ b3_out (32-dim)     │
        └─────────────────────┘  └─────────────────────┘
        
        Aux heads (detached inputs — no gradient to branches):
          aux1 = aux1_head(b1_out.detach())  → (5,)
          aux2 = aux2_head(b2_out.detach())  → (5,)
        
        ┌────────────────────────────────────────────────────┐
        │  Gated Fusion: cat([b1,b2,b3,aux1_sg,aux2_sg])    │
        │  138→128→128 (two FC layers with LN + SiLU each)  │
        │  feat (128-dim)                                     │
        └────────────────────────────────────────────────────┘
        
        PRIMARY HEADS (gradient through fusion → branches):
          q_head        → (B, 4)  direction per horizon
          strength_head → (B, 4)  horizon edge [0,1]
          selector_head → (B, 4)  optimal horizon logits (CE loss)
        
        PRIVATE HEADS (branch_cat.detach() — NO gradient to branches):
          branch_cat = LN(cat([b1,b2,b3]).detach())  ← critical fix
          pips_head      → (B, 4)  forward pips per horizon
          risk_head      → (B, 8)  [MFE, MAE] × 4 horizons
          liquidity_head → (B, 2)  zone dist + vol
          reversal_head  → (B, 1)  reversal probability
```

**Why `.detach()` on `branch_cat` matters**: The reversal loss and pips loss must NOT flow into the branch towers. Without `.detach()`, gradients from all private heads mix into the shared Conv1D encoders, degrading the fusion-path representations that drive direction and horizon selection.

### Dual-Input ExecutorQNetwork

The Q-executor receives both a full-indicator feature window and a compact context vector:

```
feat_window: (B, 300, F)   — 300-bar zone history
ctx:         (B, 28)        — meta outputs + account + session

  Branch A (feature window):
    Conv1D(F→64) → BN → SiLU → Conv1D(64→64) → BN → SiLU
    AdaptiveAvgPool1d(1) → FC(64→64) → SiLU  =  feat_h (64)

  Branch B (context):
    Dense path:  b1_fc1(28→128)→LN→SiLU → b1_fc2(128→64)→LN  = b1_out (64)
    Group path:  meta[0:10]→16, risk[10:15]→16,
                 zone[15:23]→16, time[23:28]→16
                 cat(64)→FC(64→64)→LN→SiLU  = b2_out (64)

  Fusion: cat([feat_h, b1_out, b2_out]) = 192 → FC(192→128) → LN → SiLU

  4 Independent Horizon Heads:
    Each: FC(128→64) → LN → SiLU → FC(64→3)
    Actions per head: WAIT(0) / CALL(1) / PUT(2)
    Output: (B, 4, 3) or (B, 3) when horizon_idx specified
```

### Regret Buffer (Track B)

```
self.regret_buffer = collections.deque(maxlen=10_000)

RegretTransition = namedtuple('RegretTransition', [
    'feat_window',          # (300, F)
    'ctx_state',            # (28,)
    'action',               # always H_WAIT=0
    'regret_reward',        # -abs(fwd_pct) * 5.0, capped -3.0
    'next_feat_window',     # (300, F)
    'next_ctx_state',       # (28,)
    'next_action_mask',     # (3,)
    'horizon_idx',          # which horizon was profitable
    'counterfactual_pct',   # what % move was missed
])

Trigger: action=WAIT AND |fwd_pct_h| >= 0.0015 AND direction matches bias
Sample rate: 20% of training batches draw from regret buffer
```

---

## Components and Interfaces

### SignalMetaNetwork (PyTorch)

**Location**: `app/core/ml/signal_meta_learner.py` (in-house), Cell 5 (notebook)  
**Interface**:
```python
forward(x: Tensor[B, T, F], return_aux=False)
  → (q_vals[B,4], strength[B,4], pips[B,4], risk[B,8],
     liquidity[B,2], reversal[B,1])
  # with return_aux=True also returns (aux1[B,5], aux2[B,5], selector[B,4])
```

**Critical invariant**: `branch_cat = branch_ln(cat([b1,b2,b3]).detach())` — the `.detach()` call is non-negotiable.

### ExecutorQNetwork (PyTorch)

**Location**: `app/core/options/q_executor.py` (in-house), Cell 6 (notebook)  
**Interface** (upgraded):
```python
forward(feat_window: Tensor[B,T,F], ctx: Tensor[B,28], horizon_idx=None)
  → Tensor[B, 4, 3]  # all horizons
  # or Tensor[B, 3]   # single horizon when horizon_idx provided
```

### HTFBiasPackage

**Location**: `app/core/options/q_executor.py`  
**Fields**:
```python
direction: str              # "bullish" | "bearish" | "neutral"
strength: float             # [0,1]
reversal_prob: float        # [0,1]
q_value: float              # max Q-value
expected_mfe_pips: float
expected_mae_pips: float
horizon_strengths: List[float]   # [5m, 15m, 30m, 1h]
optimal_horizon_idx: int    # 0-3
recommended_expiry: str     # "5m" | "15m" | "30m" | "1h"
```

### State Vector (28-dim)

```
Slots 0-9:   Meta + Multi-Horizon (dir, strength, rev_prob, q_val, mfe, mae, h_s[4])
Slots 10-14: Account context (drawdown, pos_type, pnl, win_streak, loss_streak)
Slots 15-22: Execution + Zone (tf_flag, atr/price, supp_dist, res_dist,
                               supp_vol, res_vol, vol_delta, reentries)
Slots 23-27: Time/Session (sin_hour, cos_hour, dow/6, is_nyse, is_power)
```

This layout is identical between Notebook Cell 4 and `OptionsQExecutor.build_state_vector()`.

---

## Data Models

### Target Schema (training CSV)

Basic targets (always recomputed from close prices):
```
forward_move_1   = close.shift(-1)  - close    # 5m move in price units
forward_move_3   = close.shift(-3)  - close    # 15m move
forward_move_6   = close.shift(-6)  - close    # 30m move
forward_move_12  = close.shift(-12) - close    # 1h move
target_dir_5m    = int(forward_move_1  > 0)    # binary direction
target_dir_15m   = int(forward_move_3  > 0)
target_dir_30m   = int(forward_move_6  > 0)
target_dir_1h    = int(forward_move_12 > 0)
```

Derived strength targets (computed in `_extract_targets()` in Cell 7):
```
strength_target[h] = clip(forward_move[h] / (ATR * sqrt(h_bars)) * 0.5 + 0.5, 0.05, 0.95)
```

Rich targets (from `ml_dataset_preparation.py`, already in CSV when `prepare_advanced_ml_targets=True`):
```
adv_target_MFE                      float  max upside excursion
adv_target_MAE                      float  max downside excursion (negative)
adv_target_bull_conf                float  {0,1}
adv_target_bear_conf                float  {0,1}
adv_target_bull_prob                float  [0,1]
adv_target_reversal_prob            float  [0,1]
adv_target_trend_continuation_prob  float  [0,1]
adv_target_next_zone_idx            float  {0..6}  which S/R zone reached next
adv_target_next_zone_bars           float  bars to reach zone
adv_target_next_zone_distance       float  ATR-normalised distance
```

### Checkpoint Format

```python
{
    'format_version': 2,
    'input_dim': META_PREDICT_WINDOW * DECISION_FEATURE_COUNT,
    'meta_predict_window': 150,        # new field
    'num_actions': 4,
    'feature_contract_version': str,
    'feature_keys': list[str],
    'network': state_dict,
    'target_network': state_dict,
    'optimizer': state_dict,
    'replay': dict,
    'scaler': dict,
}
```

---

## Confirmed Bugs

### Bug 1 — Zero Forward-Move Targets (Cell 7)

**Root cause**: `if "forward_move_1" not in df.columns:` guard prevents recomputation. CSV columns exist but are zero. All strength/pips/risk heads collapse to constant output.

**Fix**: Remove the conditional — always recompute from actual close prices in the loaded dataframes.

**Verification**: After fix, `_extract_targets()` must return `strength_target` with mean ≠ 0.5 and std > 0.1.

### Bug 2 — Q_LOOKBACK NameError (Cell 8)

**Root cause**: `Q_LOOKBACK` defined in Cell 6, referenced in Cell 8. Session restart + out-of-order execution → NameError.

**Fix**: Move `Q_LOOKBACK = 300` to Cell 1. Add guard in Cell 8.

### Bug 3 — Missing branch_cat.detach() (Cell 5)

**Root cause**: Notebook Cell 5 concatenates `branch_cat` without `.detach()`. Reversal/pips/risk gradients flow into branch towers, contaminating the fusion path.

**Fix**: Match in-house code: `branch_cat = branch_ln(cat([b1,b2,b3]).detach())`.

---

## Correctness Properties

Property 1: `branch_cat.requires_grad == False` after `.detach()` — verified by unit test
Property 2: Strength targets have std > 0 after Bug 1 fix — verified by assert in training cell
Property 3: `Q_LOOKBACK` in `dir()` at Cell 8 execution time — verified by guard
Property 4: State vector slots 0-27 identical between notebook and in-house — verified by parity script
Property 5: `ExecutorQNetwork` output shape `(B, 4, 3)` for `horizon_idx=None` inputs
Property 6: `RegretTransition.regret_reward` always negative (missed opportunity)

---

## Error Handling

- **Zero ATR**: `_extract_targets()` guard: `if atr < 1e-6: atr = 1.0` — prevents division by zero in strength normalization
- **Missing columns**: Graceful fallback in `_extract_targets()` when `adv_target_*` columns absent — zeros as defaults, log warning
- **Empty regret buffer**: `record_regret_transition()` is a no-op if buffer full; `train_step()` skips regret sampling if `len(regret_buffer) < batch_size // 5`
- **Checkpoint mismatch**: `import_checkpoint()` raises `ValueError` if `meta_predict_window` doesn't match current learner config

---

## Testing Strategy

1. **Unit test — Bug 1 regression**: Load CSV with zero forward_move columns → assert targets have std > 0 after fix
2. **Unit test — Bug 3**: Run `forward()` → assert `branch_cat.requires_grad == False`
3. **Integration test — parity script** (`scripts/verify_meta_q_parity.py`): Same weights, same input → all outputs match within `atol=1e-5`
4. **Shape test**: `ExecutorQNetwork(num_features=50).forward(rand(1,300,50), rand(1,28))` → shape `(1, 4, 3)`
5. **Training smoke test**: Run Phase 1 for 1 epoch → all head losses non-zero, strength std > 0
