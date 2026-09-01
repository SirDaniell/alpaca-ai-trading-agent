# Investigation Summary: Meta-Learner + Q-Learner Sync

**Status:** ✅ Investigation Complete (No implementation yet)  
**Scope:** Notebook (PyTorch) vs App Code (TensorFlow), target routing, context windows, gradient isolation

---

## 🎯 Core Findings

### 1. Target Consumption: 43% Gap

```
WIRED (21 targets):
  ✅ Direction (4)           → q_head [target_dir_5m, 15m, 30m, 1h]
  ✅ Strength (4)            → strength_head [bull/bear_strength_8]
  ✅ Movement (4)            → pips_head [forward_move_1/3/6/12]
  ✅ Risk (8)                → risk_head [MFE/MAE per horizon]
  ✅ Reversal (1)            → reversal_head [reversal_prob]

PARTIAL (2 targets):
  ⚠️  Liquidity (2/7)        → liquidity_head [zone only, missing 5]

MISSING (16+ targets):
  ❌ OHLCV Sequences (35)    → No head [adv_target_Open/High/Low/Close/Volume_t1..t7]
  ❌ Volatility/Velocity (14) → No head [regime, expansion, bull/bear, speed]
  ❌ Currency Divergence (4)  → No head [CSM_hist/asset/dxy]
  ❌ Full Zone Analysis (5)   → No head [next_zone_idx, bars, distance, volume, SNR touches]
```

**Impact:** Meta-learner is only 57% of intended capacity

---

### 2. Context Window Mismatch

```
CURRENT (Notebook):
  Meta-Learner:  1000 bars  [100% + 50% + 30% branching]
  Q-Learner:     64 bars    [recent only]

INTENDED (Design):
  Meta-Learner:  150 bars?  [per-head "prompt"]
  Q-Learner:     1000 bars? [zone analyzer — needs history]

QUESTION:
  Should Q be the zone analyzer (needs 1000 history) 
  or execution engine (64 recent bars enough)?
```

**Impact:** If Q should be zone analyzer, current design won't work

---

### 3. Head Isolation: Partial

```
AUXILIARY HEADS (✅ Correct):
  aux1_head: fed b1_out.detach()     → ✅ No gradient to branch 1
  aux2_head: fed b2_out.detach()     → ✅ No gradient to branch 2

PRIVATE HEADS (❌ Problem):
  pips/risk/liquidity/reversal:      → ❌ Fed branch_cat (NOT detached)
                                         → Gradients leak into b1/b2/b3
                                         → Competes with fusion head gradients
  FIX: Use branch_cat.detach() before private heads (stochastic depth pattern)
```

**Impact:** Branch layers optimized for conflicting objectives (fusion + private heads)

---

### 4. Meta-Q Integration: Weak

```
Meta-Learner Outputs:
  • q_vals (4 horizons)
  • strength (4 horizons)
  • pips (4 horizons)
  • risk (8 dims)
  • liquidity (2 dims)
  • reversal (1 dim)
  → Total: 23 dimensions

Q-Learner Context (28 dims):
  [0:10]   Meta predictions          ← HOW ARE THESE POPULATED?
  [10:15]  Risk metrics              ← Hand-crafted
  [15:23]  Zone proximity            ← Hand-crafted
  [23:28]  Time features             ← Hand-crafted

PROBLEM: Q doesn't explicitly consume meta outputs
         Context is hard-coded, not learned from meta
```

**Impact:** Q-learner has no end-to-end differentiation with meta-learner

---

### 5. Parity Gap: Notebook 2 Phases Ahead

```
NOTEBOOK (PyTorch):
  ✅ Phase 1: Meta-learner training   (50 epochs, multi-head)
  ✅ Phase 2: Q-learner replay buffer (synthetic data)
  ✅ Phase 3: Q evaluation            (4 horizons × 3 actions)
  ✅ Real SNR zone detection          (verified 1:1 backend match)

APP CODE (TensorFlow):
  ⏳ Phase 1: Data collection only    (evaluate_option_expiries.py)
  ❌ No learners implemented yet
  ❌ No TensorFlow equivalents
  ❌ Stuck at HTF confirmation gate (manual rules, not learned)

SYNC CHALLENGE: Can't port PyTorch notebook 1:1 to TensorFlow app
```

**Impact:** Production deployment blocked (notebook ≠ app code)

---

## 📊 Recommendation Matrix

| Issue | Current | Recommended | Effort | Priority |
|-------|---------|-------------|--------|----------|
| **Target Gap** | 21/37 wired | Add heads for all 37 | Medium | Critical |
| **Context Window** | Meta=1000, Q=64 | Clarify intent (zone analysis?) | Low | Critical |
| **Head Isolation** | Partial detach | Full StopGradient | Low | High |
| **Meta-Q Integration** | Hand-crafted ctx | Learn routing layer | Medium | High |
| **Parity** | PyTorch only | Add TensorFlow equiv | High | Blocking |
| **Zone Liquidity** | Partial (2/7) | Full next-zone head | Low | High |
| **Volatility** | Missing | Add velocity/regime heads | Medium | Medium |
| **OHLCV Sequence** | Missing | Add future candle head | Low | Medium |

---

## 🔍 Data Leakage Verification

**Verified (✅):**
- `detect_snr_levels_sequential()` — uses only data ≤ up_to_index
- Notebook batching — `ti = idx + lookback_bars` (no lookahead)
- Real SNR zone detection — verified against backend

**Needs Verification (⚠️):**
- Are `adv_target_*` columns excluded from feature_cols?
- Are `forward_move_*` columns excluded from feature_cols?
- Is zone detection in app code truly no-lookahead?

---

## ✅ Action Items (Phase 2: Design)

**User Clarification Needed:**

1. **Zone Analyzer Role:** Should Q-learner be the zone analyzer?
   - If YES → Allocate 1000-bar context to Q, reduce meta to 150-bar
   - If NO → Confirm meta is zone analyzer, keep current allocation

2. **Target Scope:** Should meta-learner handle all 37 targets?
   - If YES → Add 6 new heads (ohlcv, volatility, velocity, divergence, full-zone)
   - If NO → Identify priority targets, ignore rest

3. **Meta-Q Integration:** How should meta outputs feed into Q?
   - Option A: Explicit routing (meta output → learned projection → Q context)
   - Option B: Implicit (meta selects horizon, Q refines action)
   - Option C: Keep separate (meta = strength signal, Q = execution)

4. **Production Path:** PyTorch notebook or TensorFlow app?
   - Option A: Deploy notebook as-is (requires PyTorch server)
   - Option B: Port to TensorFlow (requires full sync)
   - Option C: Keep both (notebook for research, app for prod)

---

## 📁 Investigation Documents

Location: `.kiro/specs/meta-q-learner-sync/`

- `investigation.md` — 30-page detailed analysis (targets, architecture, gaps)
- `summary.md` — This document (quick reference)
- `.config.kiro` — Spec metadata

---

**Investigation completed:** 2026-09-01  
**Awaiting:** User input on 4 design questions above  
**Next phase:** Architecture design (Phase 2)  
**Timeline:** ~2 hours for design spec once questions answered
