# Phase 2 Design Decisions Checklist

**Pre-Requisite:** Answer all 4 questions below. Answers determine architecture design.

---

## Question 1: Zone Analyzer Role

**Context Window Allocation Decision**

```
Current Allocation:
  Meta-Learner:  1000 bars
  Q-Learner:     64 bars

Intended Allocation (from design intent):
  Meta-Learner:  150 bars (recent "prompt")
  Q-Learner:     1000 bars (zone history analyzer)
```

**Your Decision:**

- [ ] **Option A: Keep Current**
  - Meta-Learner = regime/signal strength analyzer (1000-bar history)
  - Q-Learner = execution engine (64-bar recent context only)
  - Rationale: Simpler, clearer separation of concerns
  - Con: Q can't reason about zone evolution over time

- [ ] **Option B: Swap to Intended**
  - Meta-Learner = signal strength predictor (150-bar "prompt" with branching)
  - Q-Learner = zone analyzer (1000-bar history for liquidity sequencing)
  - Rationale: Q can learn "price moves zone-to-zone"
  - Con: Larger Q input (1000 × 335 features)

**Decision:** _______________________________________________

---

## Question 2: Target Scope

**Which targets should Meta-Learner handle?**

```
Current (21 targets):
  ✅ Direction (4)      [target_dir_5m/15m/30m/1h]
  ✅ Strength (4)       [bull/bear_strength]
  ✅ Pips (4)           [forward_move_1/3/6/12]
  ✅ Risk (8)           [MFE/MAE]
  ✅ Reversal (1)       [reversal_prob]

Proposed Additions (16 targets):
  ❓ OHLCV Sequences (35) [adv_target_Open/High/Low/Close/Volume_t1..t7]
      → Use case: Multi-task learning (model learns structure constraints)
      → Size: 35 targets, 1 new head
      
  ❓ Volatility/Velocity (14) [regime, expansion, speed, velocity]
      → Use case: Market regime understanding
      → Size: 14 targets, 1 new head
      
  ❓ Currency Divergence (4) [CSM metrics]
      → Use case: Detect asset vs DXY divergence
      → Size: 4 targets, 1 new head
      
  ❓ Full Liquidity Zone (5) [next_zone_idx, bars, distance, volume, SNR touches]
      → Use case: Zone sequencing (CRITICAL for Q-learner)
      → Size: 5 targets, 1 new head
      → Note: Currently only 2/7 wired
```

**Your Decision:**

- [ ] **Option A: Full Targets**
  - Add 4 new heads (ohlcv, volatility, velocity, divergence, full-zone)
  - Meta-learner handles all 37 targets
  - Pro: Most complete market understanding
  - Con: 10 heads (slow training, more parameters)

- [ ] **Option B: High-Priority Only**
  - Add full-zone head (CRITICAL for Q)
  - Add volatility/velocity head (regime understanding)
  - Skip OHLCV sequences (less critical)
  - Skip currency divergence (less critical)
  - Total: 8 heads

- [ ] **Option C: Minimal (Current)**
  - Fix partial liquidity head (7/7 instead of 2/7)
  - Keep 6 heads as-is
  - Pro: Faster training
  - Con: Misses regime + zone info

**Decision:** _______________________________________________

---

## Question 3: Meta-Q Integration

**How should Meta-learner outputs feed into Q-learner?**

```
Meta-Learner Outputs:
  - q_vals (4 horizons)           [binary direction 0/1]
  - strength (4 horizons)         [magnitude 0-1, horizon probability]
  - pips (4 horizons)             [expected move in ATRs]
  - risk (8 dims)                 [MFE/MAE per horizon]
  - liquidity (2-7 dims)          [zone proximity]
  - reversal (1 dim)              [reversal signal]
  + selector_logits (4)           [best horizon choice softmax]

Q-Learner Context (28 dims):
  [0:10]   Meta predictions        [NEEDS DESIGN]
  [10:15]  Risk metrics            [Hand-crafted]
  [15:23]  Zone proximity          [Hand-crafted]
  [23:28]  Time features           [Hand-crafted]
```

**Your Decision:**

- [ ] **Option A: Explicit Routing**
  - Learn projection layer: meta_outputs → 10-dim meta_context
  - Design: MLP(meta_outputs) → 10-dim
  - Pass 10-dim + hand-crafted 18-dim = 28-dim Q context
  - Pro: End-to-end learnable
  - Con: Adds parameters, training complexity

- [ ] **Option B: Implicit (Selector-Based)**
  - Meta's selector_logits chooses best horizon
  - Q receives horizon choice + hand-crafted context
  - Meta strength values scale Q-value outputs per horizon
  - Pro: Simpler, clear separation
  - Con: No end-to-end flow, less expressive

- [ ] **Option C: Separate Experts**
  - Meta-learner = pure signal generator (no Q coupling)
  - Q-learner = pure decision maker (consumes meta as side channel)
  - No explicit routing, learned independently
  - Pro: Modular, easier to debug
  - Con: Weak integration

**Decision:** _______________________________________________

---

## Question 4: Production Path

**Which codebase should be the source of truth?**

```
Notebook (PyTorch):
  ✅ Complete implementation (Phase 1, 2, 3)
  ✅ Multi-head training working
  ✅ Q-learner with replay buffer
  ❌ Can't deploy (PyTorch server required)

App Code (TensorFlow):
  ✅ Production framework (FastAPI + TensorFlow)
  ✅ MLOps ready (versioning, checkpoints, metrics)
  ❌ Learners not implemented yet
  ❌ 2 phases behind notebook
```

**Your Decision:**

- [ ] **Option A: Deploy Notebook**
  - Keep PyTorch as primary codebase
  - Set up PyTorch serving (TorchServe or FastAPI + torch)
  - Sync app code to Python reference (non-ML pipeline only)
  - Timeline: 1-2 weeks
  - Risk: Infrastructure change (PyTorch vs TensorFlow)

- [ ] **Option B: Port to TensorFlow**
  - Implement TensorFlow equivalents (SignalMetaNetwork → Keras)
  - Full parity testing (layer-by-layer numeric comparison)
  - Integrate into app code (backend/app/core/signal_meta_network.py)
  - Timeline: 2-3 weeks
  - Risk: Porting bugs, numerical drift

- [ ] **Option C: Dual Codebase**
  - Keep notebook for research/iteration
  - Implement TensorFlow in app for production
  - Sync after design stabilizes
  - Timeline: 3-4 weeks
  - Risk: Maintenance overhead (stay in sync)

**Decision:** _______________________________________________

---

## Post-Decision Actions (Auto-Generated After Answers)

**Once you provide answers:**

1. ✅ Generate Phase 2 Architecture Design spec
2. ✅ Create head configuration (targets per head)
3. ✅ Design context window strategy (meta + Q)
4. ✅ Plan meta-Q integration layer
5. ✅ Estimate implementation effort per decision combo
6. ✅ Create implementation plan (Phase 3)

---

## Quick Estimator (Before You Answer)

**Effort by Combination:**

| Decision A | Decision B | Decision C | Total Effort | Timeline |
|---|---|---|---|---|
| Keep (64) | Current (21) | Selector | Low | 1 week |
| Keep (64) | High-Priority (30) | Routing | Medium | 2 weeks |
| Swap (1000) | Full (37) | Routing | High | 3-4 weeks |
| Swap (1000) | Full (37) | Routing + TF | Very High | 4-6 weeks |

---

**Please fill in the 4 questions above and return to the investigation team.**  
**Estimated response time:** 10 minutes to answer  
**Estimated design phase (after answers):** 2 hours
