# Bugfix Requirements Document

## Introduction

The AXE Genesis RL training notebook (`notebook2398f959dc.ipynb`) contains seven confirmed bugs that collectively render the trained model indistinguishable from a random baseline. Phase 1 (SignalMetaNetwork) peaks at 52.89% validation win rate after 50 epochs — barely above 50% chance — and Phase 2 (DQN ExecutorQNetwork) degenerates to a WAIT-only policy. Out-of-sample win rate of 49.87% confirms the system produces no predictive signal.

The bugs span missing auxiliary targets, degenerate reward shaping, coin-flip supervision labels, a replay buffer that is destroyed every epoch, contradictory lookback constants, fully detached auxiliary heads, and un-normalized feature windows. This document captures the defective behaviors, the required correct behaviors, and the existing behaviors that must be preserved during the fix.

---

## Bug Analysis

### Current Behavior (Defect)

**BUG 1 — Missing auxiliary ML targets**

1.1 WHEN the training CSV is loaded and auxiliary target columns (`adv_target_*`, `Volatility_Regime_next`, `vel_bull_fwd_8`, etc.) are absent from the file THEN the system fills all 22 target arrays with `np.zeros`, reporting "non-zero keys: 0 / 22"

1.2 WHEN auxiliary losses `l_zone`, `l_vol`, and `l_vel` are computed against all-zero targets THEN the system trains the strength, pips, and liquidity heads to predict zero, directly opposing any real signal present in the data

1.3 WHEN zero-filled auxiliary targets consume 35% of the weighted loss budget THEN the system propagates corrupted gradient noise through the corresponding heads for the entire training run

**BUG 2 — Degenerate WAIT reward**

2.1 WHEN the agent takes a WAIT action THEN the system unconditionally returns a reward of +0.001 regardless of market conditions

2.2 WHEN CALL and PUT actions average −0.0005 reward on a coin-flip market THEN the system causes the Q-network to learn a WAIT-always policy as the mathematically optimal strategy

2.3 WHEN the WAIT-always policy converges THEN the system produces an epoch-20 action distribution of approximately 27,699 WAIT vs 3,803 CALL vs 3,327 PUT per 5-minute horizon, making the agent non-functional for trading

**BUG 3 — Coin-flip direction labels**

3.1 WHEN direction targets are computed as `df["target_dir_5m"] = (df["forward_move_1"] > 0).astype(np.float32)` THEN the system produces labels that are positive approximately 50.3% of the time, equivalent to random noise for a market where next-bar direction is unconditional

3.2 WHEN the Q-head trains on these 50/50 binary labels with MSE loss THEN the system plateaus at MSE ≈ 0.25 (the theoretical maximum entropy floor for a binary 50/50 signal), yielding no predictive improvement across epochs

**BUG 4 — Replay buffer destroyed every epoch**

4.1 WHEN `replay_buffers = [[] for _ in range(NUM_HORIZONS)]` is placed inside the epoch loop THEN the system wipes all accumulated experience at the start of each epoch, retaining 0 transitions across epoch boundaries

4.2 WHEN the buffer capacity is set to 4,000 across 34,829 steps per epoch THEN the system retains less than 3% of experience generated within a single epoch, causing severe recency bias

4.3 WHEN `pop(0)` is used to evict the oldest entry from a Python list buffer THEN the system performs an O(N) memory operation on every overflow step, making buffer management scale poorly

4.4 WHEN CALL/PUT transitions represent approximately 7% of steps and WAIT represents approximately 93% THEN the system overwhelms the replay buffer with WAIT samples, starving the Q-network of informative trading transitions

**BUG 5 — Q_LOOKBACK defined inconsistently across cells**

5.1 WHEN Cell 0 sets `Q_LOOKBACK = 150`, Cell 5 overrides it to `Q_LOOKBACK = 300`, and Cell 8 overrides it again to `Q_LOOKBACK = 64` THEN the system instantiates the network with a 300-bar architecture but trains it on 64-bar windows, creating a permanent mismatch between model design and training data shape

5.2 WHEN `AdaptiveAvgPool1d` silently accepts variable-length inputs THEN the system produces no crash or warning, allowing the mismatch to persist undetected across the entire training run and corrupting temporal feature learning

**BUG 6 — Auxiliary heads fully detached from backbone**

6.1 WHEN secondary heads (pips, risk, liq, rev) compute losses using `backbone_input.detach()` THEN the system severs the gradient path from those heads to the Conv1D and LSTM towers, so those losses never update the backbone

6.2 WHEN only `l_q` (on coin-flip labels) and `l_str` (partially corrupted by zero auxiliary targets) actually propagate gradients to the backbone THEN the system leaves the Conv1D+LSTM representation largely untrained, producing weak feature extraction regardless of epoch count

**BUG 7 — Feature windows not normalized per-window**

7.1 WHEN `build_feat_window` returns raw feature values that include absolute price levels (close, open, high, low) which vary with market level THEN the system feeds non-stationary absolute values into Conv1D layers

7.2 WHEN Conv1D filters learn from absolute price levels rather than relative price patterns THEN the system encodes market-level information instead of structural patterns, preventing generalization across different time periods and market regimes

---

### Expected Behavior (Correct)

**BUG 1 — Missing auxiliary ML targets**

2.1 WHEN the training CSV is loaded THEN the system SHALL validate that all required auxiliary target columns are present and non-zero, and SHALL raise a clear error or regenerate the targets before training begins

2.2 WHEN auxiliary losses are computed THEN the system SHALL use real, non-zero target arrays so that strength, pips, and liquidity head gradients reflect genuine market signals

2.3 WHEN auxiliary targets are valid THEN the system SHALL allow the full 35% auxiliary loss budget to provide meaningful regularization to the corresponding heads

**BUG 2 — Degenerate WAIT reward**

2.4 WHEN the agent takes a WAIT action THEN the system SHALL return a context-sensitive reward (e.g., a small cost proportional to opportunity forgone, or zero) that does not unconditionally dominate CALL/PUT expected returns

2.5 WHEN CALL or PUT actions are taken and result in profit THEN the system SHALL return a reward sufficiently larger than the WAIT reward so that the Q-network can learn to prefer profitable trades over perpetual waiting

2.6 WHEN the reward structure is corrected THEN the system SHALL produce a balanced action distribution across WAIT, CALL, and PUT, reflecting genuine market opportunities rather than policy collapse

**BUG 3 — Coin-flip direction labels**

2.7 WHEN direction targets are constructed THEN the system SHALL use zone-conditional or signal-filtered labeling (e.g., `BCEWithLogitsLoss` with zone-conditional sample weighting) so that positive and negative examples carry discriminative information

2.8 WHEN the direction head trains on corrected labels THEN the system SHALL achieve a validation MSE substantially below 0.25 as training progresses, indicating that the signal is learnable

**BUG 4 — Replay buffer destroyed every epoch**

2.9 WHEN the replay buffer is initialized THEN the system SHALL initialize it once before the epoch loop so that experience accumulates across all epochs

2.10 WHEN the buffer reaches capacity THEN the system SHALL use a `collections.deque(maxlen=N)` or equivalent O(1) structure for eviction, replacing the O(N) `list.pop(0)` pattern

2.11 WHEN CALL/PUT transitions are generated THEN the system SHALL use prioritized or stratified sampling that over-samples minority-class (CALL/PUT) transitions so the Q-network receives adequate signal from informative actions

2.12 WHEN buffer capacity is configured THEN the system SHALL set capacity large enough to hold at least one full epoch of experience (≥34,829 entries) to avoid catastrophic recency bias

**BUG 5 — Q_LOOKBACK defined inconsistently**

2.13 WHEN `Q_LOOKBACK` is defined THEN the system SHALL define it in exactly one location (or a single configuration block) and SHALL use that single value consistently for both network instantiation and training window construction

2.14 WHEN the lookback value is changed THEN the system SHALL require only one edit that propagates to all dependent cells, eliminating the multi-definition override pattern

**BUG 6 — Auxiliary heads fully detached from backbone**

2.15 WHEN auxiliary head losses are computed THEN the system SHALL use `backbone_input` directly (without `.detach()`) for all heads whose gradients should flow into the backbone, so the Conv1D+LSTM towers receive training signal from pips, risk, liq, and rev losses

2.16 WHEN the backbone receives gradient from all heads THEN the system SHALL converge to richer temporal representations that reflect multiple aspects of market structure, not just the coin-flip direction signal

**BUG 7 — Feature windows not normalized per-window**

2.17 WHEN `build_feat_window` constructs a feature window THEN the system SHALL apply per-window normalization (e.g., z-score or min-max over the window) to price-level features so that Conv1D receives stationary, relative-valued inputs

2.18 WHEN feature windows are normalized THEN the system SHALL preserve the relative relationships within each window (momentum, patterns) while removing absolute market-level information that does not generalize across time

---

### Unchanged Behavior (Regression Prevention)

3.1 WHEN auxiliary target columns ARE present and non-zero in the loaded CSV THEN the system SHALL CONTINUE TO compute auxiliary losses and apply them with their existing weights without alteration

3.2 WHEN the Q-network processes a feature window of the configured lookback length THEN the system SHALL CONTINUE TO produce a valid Q-value tensor of shape `(batch, NUM_HORIZONS, NUM_ACTIONS)` with the same output semantics

3.3 WHEN Phase 1 training completes THEN the system SHALL CONTINUE TO freeze Phase 1 weights before Phase 2 training begins, preserving the two-phase training separation

3.4 WHEN the agent takes a CALL or PUT action and the trade resolves THEN the system SHALL CONTINUE TO compute reward from the realized PnL of that trade, not from a synthetic signal

3.5 WHEN the SignalMetaNetwork forward pass runs THEN the system SHALL CONTINUE TO produce outputs for all five heads (direction, strength, pips, risk, liquidity) with the same tensor shapes and value ranges

3.6 WHEN replay sampling occurs THEN the system SHALL CONTINUE TO sample transitions as `(state, action, reward, next_state, done)` tuples compatible with the existing Q-learning update step

3.7 WHEN the strict gate in Phase 3 evaluates a prediction THEN the system SHALL CONTINUE TO apply the same confidence threshold logic, with the expectation that a correctly trained model will now fire on more than 4 trades

3.8 WHEN feature engineering runs on the raw OHLCV data THEN the system SHALL CONTINUE TO produce the same 326 feature columns in the same order, with normalization applied as a post-processing step rather than a change to feature selection

3.9 WHEN the notebook cells are executed in order THEN the system SHALL CONTINUE TO run end-to-end without requiring manual intervention between Phase 1 and Phase 2 training

3.10 WHEN `AdaptiveAvgPool1d` is used in the network architecture THEN the system SHALL CONTINUE TO use it (or an equivalent pooling mechanism) so the architecture retains flexibility for minor window-length variations during experimentation
