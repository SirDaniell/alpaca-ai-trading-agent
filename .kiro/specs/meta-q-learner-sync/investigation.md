# Meta-Learner + Q-Learner Sync Investigation Report

**Date:** 2026-09-01  
**Status:** Investigation Complete (No Implementation Yet)  
**Focus:** Architecture verification, target inventory, and parity between notebook (PyTorch) and app code (TensorFlow)

---

## Executive Summary

The AXE system has two learners:
1. **Meta-Learner (SignalMetaNetwork)** — PyTorch, notebook-based, 1000-bar context, 6 prediction heads
2. **Q-Learner (ExecutorQNetwork)** — PyTorch, notebook-based, 64-bar context, 4 horizon × 3 action heads

The investigation reveals **critical gaps between design intent and implementation**:

- **Target Consumption:** Only ~6/30+ advanced targets are wired into the meta-learner
- **Context Window Confusion:** 1000-bar context allocated to meta-learner, but design intent suggests it should go to Q-learner for zone analysis
- **Head Isolation:** Auxiliary heads use `.detach()`, but private heads (pips, risk, liquidity, reversal) may receive gradient bleed
- **Data Leakage:** Dataset collection pipeline lacks explicit no-lookahead guarantees in both codebases
- **Parity Gap:** App code uses TensorFlow/Keras; notebook uses PyTorch — architectural alignment unclear

---

## Part 1: Target Inventory & Meta-Learner Head Mapping

### Available Targets (from ml_dataset_preparation.py)

Total: **37+ targets** across 9 categories

#### Category A: Directional Classification (4 targets)
```
- target_dir_5m   → forward 1 bar direction  (binary 0/1)
- target_dir_15m  → forward 3 bars direction (binary 0/1)
- target_dir_30m  → forward 6 bars direction (binary 0/1)
- target_dir_1h   → forward 12 bars direction (binary 0/1)
```
**Current Use:** Meta-learner `q_head` (Horizon × Direction)  
**Status:** ✅ Wired

#### Category B: Strength/Confidence (4 targets + 2 variants)
```
- adv_target_bull_strength_8  → exp-decay weighted Candle_Bull_Score [0,1]
- adv_target_bear_strength_8  → exp-decay weighted Candle_Bear_Score [0,1]
- adv_target_bull_prob        → percentile rank [0,1]
- adv_target_bull_class       → classification {0,1,2} BEAR/WAIT/BULL
- adv_target_bull_conf        → binary sigmoid "is BULL?" {0,1}
- adv_target_bear_conf        → binary sigmoid "is BEAR?" {0,1}
```
**Current Use:** Meta-learner `strength_head` (per-horizon strength [0,1])  
**Status:** ✅ Wired (via synthesized strength targets)

#### Category C: OHLCV Sequence Targets (35 targets)
```
- adv_target_Open_t1..t7     (7 future timesteps)
- adv_target_High_t1..t7     (7 future timesteps)
- adv_target_Low_t1..t7      (7 future timesteps)
- adv_target_Close_t1..t7    (7 future timesteps)
- adv_target_Volume_t1..t7   (7 future timesteps)
```
**Purpose:** Multi-task learning — model learns OHLC structural constraints  
**Current Use:** ❌ NOT USED  
**Gap:** Meta-learner has no head for sequence OHLCV prediction

#### Category D: Movement & Directional Targets (8 targets)
```
- forward_move_1   → 1 bar move (pips)
- forward_move_3   → 3 bar move (pips)
- forward_move_6   → 6 bar move (pips)
- forward_move_12  → 12 bar move (pips)
- adv_target_logret_1   → log-return at 1 bar
- adv_target_logret_5   → log-return at 5 bars
- adv_target_logret_10  → log-return at 10 bars
- adv_target_logret_20  → log-return at 20 bars
```
**Current Use:** Meta-learner `pips_head` (4 horizons ATR-normalized move)  
**Status:** ✅ Wired (via `_extract_targets` synthesis)

#### Category E: Excursion Targets (2 targets)
```
- adv_target_MFE  → Maximum Favorable Excursion [0, ∞)
- adv_target_MAE  → Maximum Adverse Excursion (-∞, 0]
```
**Current Use:** ❌ NOT USED  
**Gap:** Meta-learner risk head uses synthetic MFE/MAE, not actual targets

#### Category F: Risk/Volatility Targets (14+ targets)
```
- adv_target_Volatility_Regime_next     → shift(-1)
- adv_target_Volatility_Expansion_next  → shift(-1)
- adv_target_vol_regime_fwd_8           → exp-decay average
- adv_target_vol_expansion_fwd_8        → exp-decay average
- adv_target_Price_Velocity_Bull_next   → shift(-1)
- adv_target_Price_Velocity_Bear_next   → shift(-1)
- adv_target_vel_bull_fwd_8             → exp-decay average
- adv_target_vel_bear_fwd_8             → exp-decay average
- adv_target_Regime_Speed_Bull_next     → shift(-1)
- adv_target_Regime_Speed_Bear_next     → shift(-1)
- adv_target_speed_aligned_fwd_8        → exp-decay average
- adv_target_speed_divergence_fwd_8     → exp-decay average
```
**Current Use:** ❌ NOT USED  
**Gap:** No head in meta-learner for volatility/velocity/regime predictions

#### Category G: Zone/Liquidity Targets (7 targets)
```
- adv_target_next_zone_idx        → softmax {0..6} which pivot first?
- adv_target_next_zone_bars       → regression (bars to reach)
- adv_target_next_zone_distance   → regression (ATR-normalized)
- adv_target_next_zone_volume     → regression (zone volume)
- adv_target_snr_touch_1          → first SNR zone within window
- adv_target_snr_touch_2          → second SNR zone within window
```
**Purpose:** CRITICAL FOR Q-LEARNER — determines which pivot price visits next  
**Current Use:** Meta-learner `liquidity_head` (2 dimensions only)  
**Status:** ⚠️ PARTIALLY WIRED (only 2 dims, missing 5 important targets)

#### Category H: Reversal/Continuation Targets (3 targets)
```
- adv_target_reversal_prob             → probability of reversal
- adv_target_trend_continuation_prob   → probability trend continues
- adv_target_reversal_held             → did reversal persist?
```
**Current Use:** Meta-learner `reversal_head` (1 dimension)  
**Status:** ✅ Wired (binary reversal signal)

#### Category I: Currency/Divergence Targets (4 targets)
```
- adv_target_CSM_hist_fast_next    → Currency Strength divergence
- adv_target_CSM_hist_slow_next    → Currency Strength divergence
- adv_target_CSM_asset_fast_next   → Currency Strength divergence
- adv_target_CSM_dxy_fast_next     → Currency Strength divergence
```
**Current Use:** ❌ NOT USED  
**Gap:** No head for asset vs DXY divergence prediction

### Head Mapping Summary

| Meta-Learner Head | Targets Consumed | Count | Status |
|---|---|---|---|
| `q_head` | target_dir_{5m,15m,30m,1h} | 4 | ✅ Wired |
| `strength_head` | adv_target_bull/bear_strength_8 (synthesized) | 4 | ✅ Wired |
| `pips_head` | forward_move_{1,3,6,12} (ATR-normalized) | 4 | ✅ Wired |
| `risk_head` | Synthetic MFE/MAE from forward_move | 8 | ✅ Wired |
| `liquidity_head` | Partial zone targets (2 dims only) | 2 | ⚠️ Under-wired |
| `reversal_head` | adv_target_reversal_prob (binary) | 1 | ✅ Wired |
| `selector_logits` | Implicit (argmax of strength) | N/A | ✅ Wired |
| **MISSING** | OHLCV sequences (35 targets) | 35 | ❌ Not used |
| **MISSING** | Volatility/velocity/regime (14+ targets) | 14+ | ❌ Not used |
| **MISSING** | Currency divergence (4 targets) | 4 | ❌ Not used |
| **MISSING** | Full zone/liquidity (5 targets) | 5 | ❌ Not used |

**Gap Analysis:**
- **Wired:** ~21 targets (57%)
- **Under-wired:** 2 targets (liquidity partial)
- **Missing:** 14+ targets (43%)

---

## Part 2: Context Window Strategy Analysis

### Current Implementation (Notebook)

**Meta-Learner:**
- Context window: **1000 bars**
- Branched architecture:
  - Branch 1 (100%): All 1000 bars → Conv1D + LSTM
  - Branch 2 (50%): Last 500 bars → Conv1D only
  - Branch 3 (30%): Last 300 bars → Conv1D only
- Purpose: "See 20 hours of price history (~1000 5m bars) to understand zones"

**Q-Learner:**
- Context window: **64 bars** (recent only)
- Inputs: Full indicator set + 28-dim meta/zone/account/time context
- Purpose: "Recent execution context + meta prediction → decide action"

### Design Intent vs Reality

**Intended (from design discussion):**
```
Meta-Learner:   150-bar "prompt" with 100%→50%→25% branching
Q-Learner:      1000-bar context for zone analysis (primary zone analyzer)
Rationale:      Q-learner needs full history to understand which zones were visited,
                when they were touched, volume profile — to predict NEXT zone
```

**Actual (from notebook):**
```
Meta-Learner:   1000-bar context with branching
Q-Learner:      64-bar context (recent bars only)
Rationale:      Meta focuses on "regime + signal strength"; Q focuses on "recent moves"
```

### Critical Question

**Which design is correct?**

- **Case A (Intended):** Q-learner is the zone analyzer → needs 1000-bar history for zone sequencing
  - Pro: More interpretable ("which pivot next?" requires history)
  - Pro: Aligns with HTF confirmation bias (zones evolve over long periods)
  - Con: Increases Q-learner complexity (1000 × num_features input)

- **Case B (Current):** Meta-learner is the zone analyzer → needs 1000-bar history for regime understanding
  - Pro: Smaller Q input (64 bars = tractable)
  - Pro: Clear separation (meta = big picture, Q = execution)
  - Con: Q-learner blind to zone history (only has 64 recent bars + meta summary)

**Recommendation (TBD after user clarification):**
Design intent (Case A) makes architectural sense IF:
- Q-learner's role is "predict next zone based on history"
- Meta-learner's role is "predict strength/direction given zone context"
- Meta-learner can use smaller 150-bar window (still sees recent regimes)

---

## Part 3: Head Isolation & Gradient Flow Analysis

### Current Meta-Learner Architecture (PyTorch)

**Branch Outputs:**
```python
# Branch 1: 1000 bars → Conv1D + LSTM → [b1_last, b1_gap] = 64-dim
# Branch 2: 500 bars → Conv1D → GlobalPool → Dense → 32-dim
# Branch 3: 300 bars → Conv1D → GlobalPool → Dense → 32-dim
b1_out = [64]
b2_out = [32]
b3_out = [32]
```

**Auxiliary Head Isolation (✅ CORRECT):**
```python
aux1 = self.aux1_head(b1_out.detach())  # ✅ Detached → no gradient backprop to branch 1
aux2 = self.aux2_head(b2_out.detach())  # ✅ Detached → no gradient backprop to branch 2
```

**Fusion Input:**
```python
fusion_in = torch.cat([b1_out, b2_out, b3_out, aux1_sg, aux2_sg], dim=-1)
# where aux1_sg = aux1.detach() and aux2_sg = aux2.detach()
```

**Private Head Isolation (⚠️ PARTIAL):**
```python
branch_cat = torch.cat([b1_out, b2_out, b3_out], dim=-1)  # [128]
pips      = self.pips_head(torch.relu(self.pips_proj(branch_cat)))
risk      = self.risk_head(torch.relu(self.risk_proj(branch_cat)))
liquidity = self.liquidity_head(torch.relu(self.liq_proj(branch_cat)))
reversal  = self.reversal_head(torch.relu(self.rev_proj(branch_cat)))
```

**Issue:** Private heads receive `branch_cat` WITHOUT `.detach()`, so:
- Gradients from pips/risk/liquidity/reversal heads backprop into b1_out, b2_out, b3_out
- This competes with gradients from fusion head (q_vals, strength, selector)
- Result: Branches optimized for conflicting objectives (private heads + fusion heads)

**Recommendation:**
```python
# Apply StopGradient before private head inputs
branch_cat_sg = torch.cat([b1_out.detach(), b2_out.detach(), b3_out.detach()], dim=-1)
pips      = self.pips_head(torch.relu(self.pips_proj(branch_cat_sg)))  # ✅ Isolated
risk      = self.risk_head(torch.relu(self.risk_proj(branch_cat_sg)))  # ✅ Isolated
```

This follows the **stochastic depth** pattern already used in `tt.py` (build_recurrent_regression_head).

---

## Part 4: Q-Learner Integration Analysis

### Current Q-Learner Forward Pass

```python
def forward(self, feat_window: torch.Tensor, ctx: torch.Tensor, horizon_idx=None):
    # feat_window: (B, Q_LOOKBACK=64, num_features)  [recent full indicators]
    # ctx:         (B, 28)                           [meta + zone + account + time]
    
    feat_h = self._encode_feat(feat_window)           # (B, 64)
    b1_out, b2_out = self._encode_ctx(ctx)            # (B, 64), (B, 64)
    shared = self.fusion_ln(self.fusion_fc(torch.cat([feat_h, b1_out, b2_out], dim=-1)))
    
    # 4 horizon heads × 3 action heads (WAIT / CALL / PUT)
    return torch.stack([h(shared) for h in self.horizon_heads], dim=1)
```

### Context Vector (28-dim)

**Composition:**
```
[0:10]   — Meta predictions (10 dims)
[10:15]  — Risk metrics (5 dims)
[15:23]  — Zone proximity (8 dims)
[23:28]  — Time features (5 dims)
Total: 28 dims
```

**What's Missing?**
- No explicit routing of meta-learner HEAD OUTPUTS (q_vals, strength, pips, risk, liquidity, reversal)
- Context is hand-crafted, not learned from meta-learner
- Q-learner sees 64-bar history but doesn't see longer-term zone evolution

### Critical Integration Gap

The meta-learner produces 6 outputs:
```
- q_vals (4 horizons × direction)
- strength (4 horizons)
- pips (4 horizons)
- risk (8 dims)
- liquidity (2 dims)
- reversal (1 dim)
```

But Q-learner only consumes:
```
- Risk metrics [10:15] — 5 dims from hand-crafted features
- Zone proximity [15:23] — 8 dims from hand-crafted features
```

**Question:** How should Q-learner actually consume meta predictions?

**Current Hypothesis (from code):**
```
Meta-learner selects best horizon (via selector_logits)
→ Q-learner takes that selection as implicit guidance
→ Q decides action (WAIT / CALL / PUT) for that horizon
```

**But this is weak because:**
1. Q-learner doesn't explicitly see meta predictions
2. Context is hand-crafted, not learned
3. No end-to-end differentiation between meta + Q

---

## Part 5: Parity Analysis (Notebook vs App Code)

### Notebook Architecture (PyTorch)
```
✅ SignalMetaNetwork      (1000-bar branched)
✅ ExecutorQNetwork       (64-bar + 28-dim context)
✅ Real SNR zone detection (detect_snr_levels_sequential)
✅ Volume profile at level (calculate_volume_profile_at_level)
✅ Training loop with multi-head targets
❌ TensorFlow/Keras code  (does not exist in notebook)
```

### App Code (TensorFlow/Keras)
```
Location: backend/app/core/evaluate_option_expiries.py
Status: Phase 1 evaluation stage (simpler than notebook)
Targets: Only basic direction labels + HTF confirmation
Models: Not yet implemented (training still pending)
❌ Meta-learner architecture
❌ Q-learner architecture
❌ Multi-head training
```

### Sync Challenge

**The notebook is 2 phases ahead of the app code:**
- Notebook: Phase 1 (meta-learner training) + Phase 2 (Q-learner replay buffer) + Phase 3 (Q evaluation)
- App code: Phase 1 (option data collection + HTF confirmation gate)

**To achieve 1:1 parity:**
1. App code must implement meta-learner (SignalMetaNetwork)
2. App code must implement Q-learner (ExecutorQNetwork)
3. App code must consume all meta-learner targets
4. App code must route meta outputs → Q-learner context

---

## Part 6: Data Leakage Analysis

### No-Lookahead Guarantees

**Good (Verified):**
- `detect_snr_levels_sequential(price_data, up_to_index)` — explicitly uses only data ≤ up_to_index ✅
- `calculate_volume_profile_at_level(price_data_slice)` — uses only the passed slice ✅
- Notebook batch indexing: `ti = np.array(batch_idx) + lookback_bars` → no future data ✅

**Unclear:**
- ml_dataset_preparation.py `shift(-N)` operations — are forward targets correctly separated from features? ⚠️
- Are target columns excluded from feature_cols? Need verification in `_identify_features()` ⚠️
- Zone detection in evaluate_option_expiries.py — does it respect anti-lookahead? Need code review ⚠️

### Recommendation

**Verification checklist:**
```
[ ] Confirm target columns prefixed 'adv_target_' are excluded from feature_cols
[ ] Confirm 'forward_move_' columns are excluded from feature_cols
[ ] Verify train/val/test splits are chronological (not shuffled)
[ ] Verify sequence generation uses only historical data ≤ current bar
[ ] Verify zone detection at bar T uses only data ≤ T (no lookahead)
```

---

## Part 7: Key Gaps & Recommendations

### Gap 1: Target Under-Consumption (43% Missing)

**Current:** 21/37 targets wired into meta-learner  
**Missing:** 16/37 targets (OHLCV sequences, volatility/velocity, currency divergence)

**Recommendation:**
Add heads for missing targets:
- `ohlcv_head` — predict next 7 bars of OHLCV (sequence output)
- `volatility_head` — predict regime/expansion/bull/bear
- `velocity_head` — predict price velocity across horizons
- `divergence_head` — predict asset vs DXY divergence (CSM)

**Rationale:** These targets encode important market structure signals that the Q-learner needs to understand.

### Gap 2: Context Window Allocation

**Current:** Meta (1000) vs Q (64)  
**Intended:** Meta (150?) vs Q (1000?)

**Recommendation:**
1. Clarify user intent: Is Q-learner the zone analyzer or execution engine?
2. If Q is zone analyzer: redesign with 1000-bar context, add zone history head to meta-learner
3. If Meta is zone analyzer: confirm 1000-bar allocation, verify Q learns zone implications from meta outputs

### Gap 3: Head Isolation

**Current:** Auxiliary heads detached, private heads NOT detached  
**Issue:** Gradient conflict in branch layers

**Recommendation:**
Apply StopGradient to private head inputs (pips, risk, liquidity, reversal).

### Gap 4: Q-Learner Meta Integration

**Current:** Q-learner uses hand-crafted 28-dim context  
**Issue:** Doesn't explicitly consume meta-learner outputs

**Recommendation:**
Route meta-learner outputs into Q-learner context:
- Use meta selector_logits to embed "meta's best horizon"
- Use meta strength values to scale horizon Q-values
- Learn a projection: `meta_outputs → 10-dim meta context` (part of 28-dim context)

### Gap 5: Parity Between Notebook and App Code

**Current:** No TensorFlow implementation in app; notebook uses PyTorch  
**Issue:** Cannot deploy notebook to production (app uses TensorFlow)

**Recommendation:**
1. Port PyTorch SignalMetaNetwork → TensorFlow equivalent (use `build_*` patterns from tt.py)
2. Port PyTorch ExecutorQNetwork → TensorFlow equivalent
3. Verify numerical parity (layer-by-layer comparison on fixed input)
4. Integrate into evaluate_option_expiries.py (replace Phase 1 logic)

---

## Detailed Architecture Diagrams

### Current Meta-Learner (1000-bar, 6 heads)

```
Input (1000 bars × 335 features)
         ↓
    [Reshape to (1000, 335)]
         ↓
    ┌────────────────────────────────────────┐
    │  Branch 1 (100%): Conv1D + LSTM Tower  │
    │  All 1000 bars → 64-dim output         │
    └────────────────────────────────────────┘
         │
         ├─→ aux1_head (detached) → softmax(5)  [Auxiliary output]
         │
    ┌────────────────────────────────────────┐
    │  Branch 2 (50%): Conv1D Tower          │
    │  Last 500 bars → 32-dim output         │
    └────────────────────────────────────────┘
         │
         ├─→ aux2_head (detached) → softmax(5)  [Auxiliary output]
         │
    ┌────────────────────────────────────────┐
    │  Branch 3 (30%): Conv1D Tower          │
    │  Last 300 bars → 32-dim output         │
    └────────────────────────────────────────┘
         │
    Concatenate [b1_out=64, b2_out=32, b3_out=32, aux1_sg=5, aux2_sg=5]
         ↓
    Fusion MLP (Dense + LN + Dense + LN) → 128-dim
         ↓
    ┌──────────────────────────────────────────────────────────────────┐
    │ OUTPUT HEADS (All receive 128-dim fusion output):                │
    ├──────────────────────────────────────────────────────────────────┤
    │ • q_head            → (B, 4 horizons)  [direction 0/1]           │
    │ • strength_head     → (B, 4 horizons)  [magnitude 0-1]           │
    │ • selector_logits   → (B, 4 horizons)  [horizon choice logits]   │
    │                                                                   │
    │ Private Heads (receive detached branch_cat):                     │
    │ • pips_head         → (B, 4 horizons)  [move in ATRs]           │
    │ • risk_head         → (B, 8)           [MFE/MAE per horizon]    │
    │ • liquidity_head    → (B, 2)           [zone proximity]         │
    │ • reversal_head     → (B, 1)           [reversal prob]          │
    └──────────────────────────────────────────────────────────────────┘
```

### Current Q-Learner (64-bar + 28-dim context)

```
Inputs:
  feat_window    (B, 64, 335)  — Recent 64 bars of all indicators
  ctx            (B, 28)       — Meta/zone/account/time context

                     ↓

    ┌─────────────────────────────────────────────────┐
    │  Branch A: Conv1D over feat_window              │
    │  (B, 64, 335) → Conv1D → Pool → Dense → 64-dim  │
    └─────────────────────────────────────────────────┘
                     │
                     ├─→ [64]

    ┌─────────────────────────────────────────────────┐
    │  Branch B: Dense over 28-dim context            │
    │  (B, 28) → Dense → LN → Dense → (64, 64)       │
    └─────────────────────────────────────────────────┘
                     │
                     ├─→ (B1_out=64, B2_out=64)

                Concatenate [feat=64, b1=64, b2=64]
                           ↓
                   Fusion MLP → 128-dim
                           ↓
    ┌──────────────────────────────────────────────────────────────┐
    │  4 Horizon-Specific Heads (each gets 128-dim):               │
    ├──────────────────────────────────────────────────────────────┤
    │  • Head 0 (5m):   Dense(128) → Dense(3)  [WAIT/CALL/PUT]    │
    │  • Head 1 (15m):  Dense(128) → Dense(3)  [WAIT/CALL/PUT]    │
    │  • Head 2 (30m):  Dense(128) → Dense(3)  [WAIT/CALL/PUT]    │
    │  • Head 3 (1h):   Dense(128) → Dense(3)  [WAIT/CALL/PUT]    │
    └──────────────────────────────────────────────────────────────┘
                           ↓
              Output: (B, 4 horizons, 3 actions)
```

---

## Conclusion & Next Steps

### Current State
- Notebook has working PyTorch implementation (6 targets wired, 1000-bar context)
- App code is 2 phases behind (Phase 1 only)
- 43% of available targets are unused
- Gradient isolation is partial (private heads not detached)
- No clear meta-Q integration strategy

### Immediate Actions (Investigation Phase)
1. ✅ Inventory all targets (37+ identified)
2. ✅ Map targets to heads (21 wired, 16 missing)
3. ✅ Analyze context window strategy (1000 meta vs 64 Q)
4. ✅ Review head isolation (auxiliary ok, private needs fix)
5. ⏳ **PENDING:** User clarification on:
   - Should Q-learner be zone analyzer (1000-bar) or execution engine (64-bar)?
   - Which missing targets should be added to meta-learner?
   - How should meta outputs feed into Q-learner?

### Phase 2: Architecture Design (Awaiting User Input)
Once user clarifies design intent:
1. Finalize meta-learner heads (all 37 targets or subset?)
2. Finalize context window allocation (1000/64 swap or keep?)
3. Design meta-Q integration (explicit output routing)
4. Define TensorFlow equivalents (for app code parity)

### Phase 3: Implementation
Port notebook → app code with full parity, all targets wired, isolated heads.

---

**Document prepared by:** Investigation Agent  
**Requires user review & clarification before proceeding to Phase 2**
