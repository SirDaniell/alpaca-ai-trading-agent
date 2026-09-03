# Q-Learning Architecture & Decision-Making System
## Complete Technical Analysis from notebook2398f959dc.ipynb

---

## 1. Overall Architecture Overview

The system implements a **two-phase learning framework**:

### Phase 1: Meta-Learner (Signal Strength & Target Prediction)
- **Architecture**: SignalMetaNetwork (6 primary heads + 21+ ML targets)
- **Input**: 150-bar lookback window (12h 5m data)
- **Output**: Directional signals + volatility/velocity/zone targets per 4 horizons
- **Purpose**: Provides baseline strength prediction and multi-task signal

### Phase 2: Q-Learner (Action Selection & Execution)
- **Architecture**: ExecutorQNetwork (4 horizon-independent heads, 3 actions each)
- **Input**: Feature window (300-bar) + 28-dim context state
- **Output**: Q-values for WAIT/CALL/PUT per horizon
- **Purpose**: Learns which action maximizes long-term reward per horizon

---

## 2. Q-Network Architecture

### ExecutorQNetwork Class Structure

```
Input Layer:
├─ Feature Window: (batch, 300, 92) ← 300 bars × 92 technical indicators
│  └─ Processed via dual Conv1D + LSTM tower
└─ Context State: (batch, 28) ← Static features about current market state

Dual-Branch Architecture:
├─ Branch 1: Feature Processing
│  ├─ Conv1D(filters=64, kernel=5) → ReLU → MaxPool
│  ├─ Conv1D(filters=128, kernel=3) → ReLU → MaxPool
│  └─ LSTM(256) → Dense(128) → output_f
│
├─ Branch 2: Context Processing
│  ├─ Dense(128) → ReLU → output_c
│
└─ Fusion & Q-Head Generation:
   ├─ Concatenate(output_f, output_c)
   ├─ Dense(256) → ReLU (shared trunk)
   └─ 4 Independent Q-Heads (one per horizon)
      ├─ Head 0 (5m):   Dense(3) → [Q_WAIT, Q_CALL, Q_PUT]
      ├─ Head 1 (15m):  Dense(3) → [Q_WAIT, Q_CALL, Q_PUT]
      ├─ Head 2 (30m):  Dense(3) → [Q_WAIT, Q_CALL, Q_PUT]
      └─ Head 3 (1h):   Dense(3) → [Q_WAIT, Q_CALL, Q_PUT]
```

### Key Design Decisions
1. **Independent Heads**: Each horizon has separate Q-head so losses are clean per horizon
2. **Shared Trunk**: Conv+LSTM extracts temporal patterns, reused across horizons
3. **Context Augmentation**: 28-dim state captures price level, volatility, time-of-day
4. **Fixed Lookback**: 300-bar window (Q_LOOKBACK) ensures stable reception field

---

## 3. State Vector (28-dim Context)

The system constructs a **static state vector** per bar containing:

```python
static_states[i] = [
    # Market Sentiment (6 features)
    dir_flag,                    # +1.0 = bullish, -1.0 = bearish, 0.0 = neutral
    meta_score,                  # strongest horizon's strength (0.0-1.0)
    meta_rev,                    # reversal probability (0.0-1.0)
    meta_qmax,                   # max Q-value from meta network
    meta_mfe,                    # max favorable excursion (risk metric)
    meta_mae,                    # max adverse excursion (risk metric)
    
    # Per-Horizon Strengths (4 features)
    strength[0],                 # 5m horizon strength
    strength[1],                 # 15m horizon strength
    strength[2],                 # 30m horizon strength
    strength[3],                 # 1h horizon strength
    
    # Position State (6 features)
    has_open_position,           # 1.0 if any position open, 0.0 otherwise
    0.0, 0.0, 0.0, 0.0, 0.0,   # reserved for multi-leg tracking
    
    # Price & Volatility (3 features)
    atr_normalized,              # ATR / current_price (volatility measure)
    support_distance,            # distance to nearest support level
    resistance_distance,         # distance to nearest resistance level
    
    # Volume Analysis (3 features)
    support_volume_ratio,        # volume strength at support
    resistance_volume_ratio,     # volume strength at resistance
    volume_delta_ratio,          # (buy_vol - sell_vol) / total_vol
    
    # Reserved (1 feature)
    0.0,                         # placeholder
    
    # Time Features (3 features)
    sin_hour,                    # sin(2π × hour / 24) → cyclical hour encoding
    cos_hour,                    # cos(2π × hour / 24) → cyclical hour encoding
    day_of_week_normalized,      # day_of_week / 6.0
    
    # Market Session Flags (2 features)
    is_nyse_open,                # 1.0 if 9:30-10:30 AM ET
    is_nyse_power_hour,          # 1.0 if 3:00-4:00 PM ET
]
```

**Why These Features?**
- **Meta scores**: Give Q-network prior information about signal quality
- **Zone distances**: Encode proximity to support/resistance (critical for breakout setups)
- **Time encoding**: Capture market regimes (gaps at open, power hour dynamics)
- **Volume ratios**: Indicate likely breakout vs bounce scenarios

---

## 4. Action Space & Masking

### Three Actions Per Horizon
```
H_WAIT = 0  ← Do nothing; hold cash or close position
H_CALL = 1  ← Binary CALL option (bet on price UP within horizon)
H_PUT  = 2  ← Binary PUT option (bet on price DOWN within horizon)
```

### HardActionMask Engine (Removes Impossible Actions)
The system pre-filters actions based on **market constraints**:

```python
def get_action_mask(price, atr, nearest_supp, nearest_res, buy_vol, sell_vol, 
                    has_open_position=False):
    """
    Returns [can_wait, can_call, can_put] binary mask.
    
    Rules:
    1. If position open → WAIT only (no new positions until expiry)
    2. If price TOO CLOSE to support → no PUT (would be stopped immediately)
    3. If price TOO CLOSE to resistance → no CALL (would be stopped immediately)
    4. If volume severely imbalanced → filter opposing direction
    5. Special session logic (market open → prefer CALLs if bullish)
    """
    if has_open_position:
        return [1, 0, 0]  # Only wait allowed
    
    base_mask = [1, 1, 1]  # All actions allowed by default
    
    # Price distance checks
    supp_dist = abs(price - nearest_supp["price"]) / price
    res_dist = abs(price - nearest_res["price"]) / price
    
    MIN_DISTANCE = 0.001  # 0.1% minimum safety margin
    if supp_dist < MIN_DISTANCE:
        base_mask[2] = 0  # Disable PUT (too close to support)
    if res_dist < MIN_DISTANCE:
        base_mask[1] = 0  # Disable CALL (too close to resistance)
    
    # Volume imbalance check
    if buy_vol / (sell_vol + 1e-6) > 3.0:
        base_mask[2] = 0  # Strong buy volume → no shorts
    if sell_vol / (buy_vol + 1e-6) > 3.0:
        base_mask[1] = 0  # Strong sell volume → no longs
    
    return np.array(base_mask, dtype=np.int32)
```

**Critical**: Actions from masked-out options are set to **-1e9** in the softmax, preventing any probability assignment.

---

## 5. Q-Learning Training Loop (Phase 2)

### Replay Buffer & Sampling
```python
# Separate deque per horizon (NO cross-horizon gradient contamination)
replay_buffers = [collections.deque(maxlen=50000) for _ in range(NUM_HORIZONS)]

# Each experience tuple:
(
    abs_idx_current,            # absolute index in data (for rebuilding features)
    static_state_current,       # 28-dim state vector
    action_taken,               # 0/1/2 (WAIT/CALL/PUT)
    reward_realized,            # ±1.0 (win/loss) or 0.0 (still open)
    abs_idx_next_state,         # after horizon expires
    static_state_next,          # next state after expiry
)
```

### Bellman Update (Per Horizon)
```
Training Loop (30 epochs):
├─ For each horizon h (0-3):
│  ├─ Epoch loop:
│  │  ├─ Sample mini-batch (64) from replay_buffers[h]
│  │  ├─ Current Q: q_net(s, a, h) → Q-value for action a at horizon h
│  │  ├─ Next Q: q_target(s', h) → max Q-value at next state
│  │  │  └─ Target = reward + γ × max(Q(s', a', h))
│  │  │     Where γ (gamma) = 0.99 (discount factor)
│  │  │
│  │  ├─ Epsilon-Greedy Action Selection:
│  │  │  ├─ ε starts at 1.0 (100% random)
│  │  │  ├─ ε decays step-wise to 0.05 at 50% of total steps
│  │  │  ├─ Decay formula: ε_new = ε × decay_factor ^ step
│  │  │  │  Where decay_factor = (0.05 / 1.0)^(1 / steps_to_min)
│  │  │  │
│  │  │  └─ Action selection:
│  │  │     if rand() < ε:
│  │  │         action = random(masked_actions)  ← Exploration
│  │  │     else:
│  │  │         action = argmax(Q(s,·,h))       ← Exploitation
│  │  │         respecting mask
│  │  │
│  │  ├─ Loss Computation (MSE):
│  │  │  L = 0.5 × (target_Q - current_Q)²
│  │  │
│  │  ├─ Backprop & Parameter Update (AdamW):
│  │  │  └─ lr=1e-3, weight_decay=1e-4
│  │  │     (decay regularizes to prevent overfitting)
│  │  │
│  │  └─ Periodic Target Network Sync:
│  │     └─ Every N steps: q_target.load_state_dict(q_net.state_dict())
│  │        (stabilizes targets, reduces feedback loop)
│  │
│  └─ End epoch: print progress (loss, action distribution)
│
└─ Training complete
```

### Training Hyperparameters
```python
Q_EPOCHS = 30                      # Total training passes over all data
BATCH_SIZE_Q = 64                  # Mini-batch for gradient step
BUFFER_CAPACITY = 50000            # Max experiences per horizon buffer
NUM_HORIZONS = 4                   # Separate heads for 5m/15m/30m/1h
LEARNING_RATE = 1e-3               # AdamW learning rate
WEIGHT_DECAY = 1e-4                # L2 regularization
GAMMA = 0.99                       # Reward discount factor
```

---

## 6. Decision-Making at Inference (Evaluation)

### Runtime Decision Flow
```
Input: Current bar (index, price, state)
         ↓
Step 1: Check if any horizon has an open position
        ├─ If yes → WAIT only (mask = [1,0,0])
        └─ If no → proceed to Step 2
         ↓
Step 2: Reconstruct feature window (300-bar lookback)
        └─ Flattened to (300 × 92 = 27600 features)
         ↓
Step 3: Forward pass through q_net
        ├─ Input: (feat_window, state_28dim, horizon_idx=h)
        ├─ Dual-branch processing
        └─ Output: Q-values [Q_WAIT, Q_CALL, Q_PUT] for horizon h
         ↓
Step 4: Apply action mask
        └─ Set masked actions Q-value to -1e9
         ↓
Step 5: Argmax Q-value
        └─ action = argmax(masked_q_values)
         ↓
Step 6: Execute action
        ├─ If WAIT: skip trade
        ├─ If CALL: open CALL option (betting price UP)
        │   └─ Store: entry_price = current_price, horizon_bars = [1,3,6,12]
        └─ If PUT: open PUT option (betting price DOWN)
            └─ Store: entry_price = current_price, horizon_bars = [1,3,6,12]
         ↓
Step 7: At expiry (after horizon_bars candles)
        ├─ Compare exit_price vs entry_price
        ├─ CALL wins if: exit_price > entry_price
        ├─ PUT wins if: exit_price < entry_price
        └─ Log outcome for post-analysis
```

### Inference Code Example
```python
def _get_h_logits(state, abs_idx, h, has_open):
    """Get raw Q-logits (before masking) for horizon h."""
    state = state.copy()
    state[11] = 1.0 if has_open else 0.0  # Update position flag
    state[15] = float(h) / 3.0             # Encode horizon index
    
    feat_w = build_feat_window(test_matrix, abs_idx, Q_LOOKBACK)
    with torch.no_grad():
        fw_t = torch.tensor(feat_w[None, ...], dtype=torch.float32, device=device)
        st_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return q_net(fw_t, st_t, horizon_idx=h).squeeze(0).cpu().numpy()
        # Returns [Q_WAIT, Q_CALL, Q_PUT] for this horizon

def _h_mask(cp, atr, ns, nr, bv, sv, has_open):
    """Get action mask based on market conditions."""
    if has_open:
        return np.array([1, 0, 0], dtype=np.int32)  # Only WAIT
    base = mask_engine.get_action_mask(cp, atr, ns, nr, bv, sv)
    return np.array([base[0], base[1], base[2]], dtype=np.int32)

def _pick_action(logits, mask):
    """Select action: argmax(Q) respecting mask."""
    # Set masked actions to -1e9 so they won't be argmax
    masked_logits = np.where(mask == 1, logits, -1e9)
    return int(np.argmax(masked_logits))

# Execution at bar i:
action = _pick_action(
    _get_h_logits(state, abs_idx, horizon_h, has_open_position),
    _h_mask(price, atr, nearest_supp, nearest_res, buy_vol, sell_vol, has_open)
)
```

---

## 7. Reward Signal & Experience Generation

### How Rewards Are Computed

The Q-learner learns from **actual market outcomes** (not simulated):

```python
# After position expires (at t + horizon_bars):
entry_price = price_at_t
exit_price = price_at_t_plus_horizon

if action == H_CALL:
    reward = 1.0 if exit_price > entry_price else -1.0
elif action == H_PUT:
    reward = 1.0 if exit_price < entry_price else -1.0
else:  # H_WAIT
    reward = 0.0  # No position = no P&L

# Store in replay_buffers[h]:
replay_buffers[h].append((
    abs_idx_t,                    # current index
    static_states[i],             # current 28-dim state
    action,                       # action taken
    reward,                       # +1.0 or -1.0 or 0.0
    abs_idx_t_plus_horizon,       # next state index
    static_states[i + horizon],   # next 28-dim state
))
```

**Critical Design**: Rewards are **binary** (+1/-1) per trade; the Q-network must learn:
1. **When to trade** (exploit positive expected value)
2. **When NOT to trade** (wait for better setups)
3. **Which direction** (CALL vs PUT) in current regime

---

## 8. Evaluation Modes (Phases 3 & 4)

### Phase 3a: Recommended Horizon (Strict Gate)
```
Algorithm:
├─ For each test bar:
│  ├─ Get meta_strengths[0:4] (4 horizon scores from meta-learner)
│  ├─ recommended_horizon = argmax(meta_strengths)
│  ├─ best_strength = meta_strengths[recommended_horizon]
│  ├─ margin = 1st_best_strength - 2nd_best_strength
│  │
│  ├─ Gate 1: best_strength ≥ CONFIDENCE_THRESHOLD (0.60)
│  ├─ Gate 2: margin ≥ HORIZON_MARGIN (0.05) [confidence in horizon choice]
│  │
│  ├─ If both gates pass:
│  │  ├─ Forward state through q_net for recommended_horizon
│  │  ├─ Apply action mask
│  │  └─ Pick action = argmax(Q)
│  │     ├─ CALL: enter if exit > entry
│  │     ├─ PUT: enter if exit < entry
│  │     └─ WAIT: skip
│  │
│  └─ Track: wins, losses, win_rate, streaks
│
└─ Output: Trade log, equity curves (flat 1R vs martingale)
```

**Interpretation**: The Q-network learns to **refine** the meta-learner's horizon choice by deciding WHEN to act on it.

### Phase 3b: Each Horizon Forced (Counterfactual)
```
Algorithm:
├─ For each test bar:
│  ├─ For each horizon h in [5m, 15m, 30m, 1h]:
│  │  ├─ Forward state through q_net[h] (force this head)
│  │  ├─ Apply action mask
│  │  ├─ Pick action
│  │  └─ Track outcome for horizon h
│  │
│  └─ Output: Separate stats per horizon (which one is most predictable?)
│
└─ Compare: which horizon has best Sharpe/win-rate?
```

**Purpose**: Isolate each head's performance to identify which horizons are most learnable.

### Phase 3c: Baselines (Always-CALL / Always-PUT)
```
Algorithm:
├─ For each test bar:
│  ├─ Always try to enter CALL (if mask allows)
│  ├─ Always try to enter PUT (if mask allows)
│  └─ Compare against Q-learner performance
│
└─ Sanity check: Q-learner should beat naïve baselines
```

**Expected Result**: If Q-learner is learning, it should beat 50% by carefully timing entries and exploiting per-horizon patterns.

### Phase 4: Multi-Horizon Concurrent Portfolio
```
Algorithm:
├─ For each test bar:
│  ├─ For each horizon h:
│  │  ├─ Check if position open on horizon h
│  │  ├─ If not: check strength[h] ≥ CONFIDENCE_THRESHOLD (0.60)
│  │  ├─ If sufficient strength:
│  │  │  ├─ Forward state through q_net[h]
│  │  │  ├─ Apply mask
│  │  │  └─ Pick action
│  │  │     ├─ Open position if CALL/PUT
│  │  │     └─ Hold if WAIT
│  │  │
│  │  └─ At expiry: settle position, record outcome
│  │
│  └─ Aggregate all positions → portfolio equity curve
│
└─ Track: total trades, win-rate, drawdown, Sharpe ratio
```

**Key Insight**: Multiple horizons can have open positions CONCURRENTLY (e.g., a 5m trade AND a 1h trade at same time). The Q-network must manage position state correctly (state[11] tracks "any position open").

---

## 9. Money Management & Martingale Overlay

### Flat 1R Strategy
```
Size = BASE_RISK_R (fixed, e.g., 1% equity per trade)
P&L_cumulative = sum(+1R on wins, -1R on losses)

Equity_flat[t] = Equity[t-1] + (win ? +1R : -1R)
```

### Martingale Strategy (Size Escalation After Losses)
```python
def simulate_martingale(outcomes, base_r=1.0, mult=2.0, max_steps=4):
    """
    After loss: size *= mult
    After win or max_steps consecutive losses: reset to base_r
    """
    eq = [0.0]
    size = base_r
    loss_streak = 0
    
    for outcome in outcomes:
        if outcome == 1:  # WIN
            eq.append(eq[-1] + size)
            size = base_r
            loss_streak = 0
        else:  # LOSS
            eq.append(eq[-1] - size)
            loss_streak += 1
            if loss_streak >= max_steps:
                size = base_r
                loss_streak = 0
            else:
                size = size * mult
    
    return eq
```

**Example Progression** (base=1, mult=2, max_steps=4):
```
Trade 1: LOSS   → size used = 1R,  equity -1R,  loss_streak=1
Trade 2: LOSS   → size used = 2R,  equity -3R,  loss_streak=2
Trade 3: LOSS   → size used = 4R,  equity -7R,  loss_streak=3
Trade 4: LOSS   → size used = 8R,  equity -15R, loss_streak=4 (HIT CAP)
Trade 5: ANY    → size reset = 1R  (fresh start)
```

**Decision Criteria for Using Martingale**:
```
✓ Use if:
  - Win-rate > 52%
  - Max loss streak (maxL) ≤ 4
  - Flat equity drawdown already acceptable

✗ Avoid if:
  - Win-rate < 52% (negative edge)
  - maxL >> 4 (cap resets too often, useless)
  - Portfolio already has high drawdown
```

---

## 10. Key Learnings from Evaluation

### Why Phase 3a/3b Might Take Zero Trades
```
Condition Analysis:
├─ Meta strength too low: max(strength[0:4]) < 0.60
│  └─ Solution: Relax threshold to 0.52 (diagnostic-only run)
│
├─ Margin too tight: |1st - 2nd| < 0.05
│  └─ Solution: Lower to 0.02 (less sure about horizon choice)
│
└─ Both: (max ≥ 0.60) AND (margin ≥ 0.05) rarely true
   └─ Interpretation: model is uncertain or market lacks clean signals
```

### Per-Horizon Performance Variance
```
Typical Results Observed:
├─ 5m: High trade count, ~49% WR (unreliable, noise-prone)
├─ 15m: Moderate trades, ~52% WR (learnable, sweet spot)
├─ 30m: Fewer trades, ~51% WR (clean but sparse)
└─ 1h: Rare trades, ~50-55% WR (cleanest but data-limited)

Recommendation:
└─ Focus training on 15m (best data quality & sample size tradeoff)
```

---

## 11. Code Reference Map

| Component | File | Notebook Cell | Purpose |
|-----------|------|---------|---------|
| Meta-Learner Setup | notebook2398f959dc.ipynb | Cell 9 (913-1100) | Initialize SignalMetaNetwork + targets |
| Target Extraction | notebook2398f959dc.ipynb | Cell 9 (1000-1100) | _extract_targets() function |
| Q-Network Definition | notebook2398f959dc.ipynb | Cell 6 | ExecutorQNetwork class |
| Training Loop | notebook2398f959dc.ipynb | Cell 11 (1387-1821) | Phase 2 Q-learning with replay buffers |
| Inference Setup | notebook2398f959dc.ipynb | Cell 12 (1824-2050) | Precompute meta features & zones |
| Phase 3a/3b Eval | notebook2398f959dc.ipynb | Cell 12 (2050-2300) | Decision functions + masking |
| Money Mgmt | notebook2398f959dc.ipynb | Cell 12 (1930-2000) | Martingale + Kelly helpers |
| Phase 4 Portfolio | notebook2398f959dc.ipynb | Cell 12 (2320-2387) | Concurrent multi-horizon trading |

---

## 12. Summary of Learning Mechanism

The Q-learner operates on a **learned optimal policy** πQ(a|s,h):

```
Offline RL Setting:
├─ Meta-learner generates baseline signals (strengths, targets)
├─ Q-learner refines WHEN to act on those signals
│  ├─ Training: Bellman update via replay buffers (off-policy learning)
│  ├─ Action selection: ε-greedy with action masking
│  └─ Reward: actual market outcome (+1/-1 per trade)
│
└─ Inference: Greedily follow learned Q-policy (ε=0)
   └─ Pick action = argmax(Q(s,a,h)) for recommended horizon
```

**Key Advantages**:
1. **Separate horizons**: No crosstalk between 5m/15m/30m/1h heads
2. **Action masking**: Prevents invalid trades (position limits, price proximity)
3. **Dual network**: Stabilizes learning (online q_net, target q_net)
4. **Replay buffer**: Breaks temporal correlation, improves sample efficiency
5. **Epsilon decay**: Explores early (ε=1.0), exploits late (ε=0.05)

**Limitations**:
1. **Offline learning**: Can't explore new actions; bounded by training data
2. **Binary rewards**: No reward shaping; hard to credit assignment across 12+ bar horizon
3. **Data hunger**: Needs ~6-12k experiences per horizon to converge
4. **Session risk**: Equity drawdown can exceed 8R if martingale hits cap

---

**Complete Q-Learning System Ready for Kaggle Submission**
