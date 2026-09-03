"""
q_executor.py — Tier 2 Q-Learner Trade Executor for Options.

Roles:
- Runs on LTF (5m / 15m).
- Consumes Tier 1 Meta-Learner HTF bias package, active Zone Snapshots, Account State, and Execution context.
- Applies HardActionMask to enforce no-chase entry discipline and volume delta confirmation.
- Implements Bias-Persistence Re-Entry (fixed risk sizing, NO martingale scaling).
- Evaluates delayed risk-adjusted rewards with overtrade churn penalties and training-time hindsight missed-opportunity shaping.

Architecture (Task 5 upgrade):
- Dual-input ExecutorQNetwork: Branch A (Conv1D over feature window) + Branch B (dense context encoder).
- 4 independent horizon heads: WAIT(0) / CALL(1) / PUT(2) per horizon.
- build_feat_window / build_feat_window_batch helpers for lookahead-free window construction.
"""

from __future__ import annotations

import collections
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.core.market.zone_snapshot import HardActionMask, ZoneSnapshotManager

logger = logging.getLogger(__name__)

# Action Definitions
ACTION_WAIT = 0
ACTION_BUY_CALL = 1
ACTION_BUY_PUT = 2
ACTION_TAKE_PROFIT_HALF = 3
ACTION_CLOSE_FLATTEN = 4
NUM_ACTIONS = 5
EXECUTOR_STATE_DIM = 28  # Compact representation for LTF execution context + Multi-Horizon hints
Q_LOOKBACK = 300         # bars of full indicators fed to Q-executor (zone analyzer)

HORIZON_BARS_LIST   = [1, 3, 6, 12]  # 5m, 15m, 30m, 1h
REGRET_MIN_PCT      = 0.0015         # min missed move to record as regret
REGRET_REWARD_SCALE = 5.0            # amplifier for regret signal

from collections import namedtuple

RegretTransition = namedtuple('RegretTransition', [
    'feat_window',        # (Q_LOOKBACK, F) numpy array — current bar's indicator window
    'ctx_state',          # (28,) numpy array — current 28-dim context
    'action',             # always ACTION_WAIT (0) — the missed action
    'regret_reward',      # float, negative — -abs(fwd_pct) * REGRET_REWARD_SCALE, capped -3.0
    'next_feat_window',   # (Q_LOOKBACK, F) numpy array — next bar
    'next_ctx_state',     # (28,) numpy array — next bar context
    'next_action_mask',   # (3,) numpy array
    'horizon_idx',        # int 0-3 — which horizon was the profitable one
    'counterfactual_pct', # float — what % move was missed
])


@dataclass
class HTFBiasPackage:
    """Tier 1 Meta-Learner output passed down to Tier 2 Executor."""
    direction: str = "neutral"  # "bullish", "bearish", "neutral"
    strength: float = 0.0      # [0.0, 1.0]
    reversal_prob: float = 0.0 # [0.0, 1.0]
    q_value: float = 0.0
    expected_mfe_pips: float = 0.0
    expected_mae_pips: float = 0.0
    # Multi-Horizon Hints
    horizon_strengths: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 0.5])
    optimal_horizon_idx: int = 2
    recommended_expiry: str = "30m"


@dataclass
class AccountContext:
    """Account state and trade history."""
    equity: float = 100000.0
    daily_pnl: float = 0.0
    daily_drawdown_pct: float = 0.0
    open_position_type: Optional[str] = None  # None, "CALL", "PUT"
    open_position_pnl_pct: float = 0.0
    win_streak: int = 0
    loss_streak: int = 0
    max_risk_per_trade_pct: float = 0.02  # 2% max account risk


@dataclass
class ExecutionContext:
    """LTF Market & Time Context."""
    symbol: str = "SPY"
    ltf_timeframe: str = "5m"  # "5m" or "15m" (supports dynamic switching)
    current_price: float = 500.0
    atr: float = 2.5
    buy_volume: float = 1000.0
    sell_volume: float = 800.0
    reentries_in_window: int = 0
    max_reentries_allowed: int = 3
    # Time & Session Features
    hour_of_day: float = 14.5      # [0.0 - 23.9] (e.g. 14.5 = 14:30 UTC / 09:30 EST)
    minute_of_hour: float = 30.0   # [0.0 - 59.0]
    day_of_week: int = 1           # 0=Monday ... 4=Friday
    session_phase: str = "nyse_open"  # "nyse_open", "nyse_midday", "nyse_power_hour", "extended_hours"


class ExecutorQNetwork(nn.Module):
    """
    Dual-specialist Q net (per horizon):
      Shared trunk  — Conv1D features + 28-dim context fusion
      Buy tower     — dedicated capacity for WAIT vs CALL
      Sell tower    — dedicated capacity for WAIT vs PUT
      Fusion head   — combines buy/sell embeddings → final WAIT/CALL/PUT

    Rationale: a single 3-way head mixed CALL/PUT gradients; specialists
    learn side-specific setups, fusion arbitrates to one action.
    """

    def __init__(
        self,
        num_features: int,
        input_dim: int = EXECUTOR_STATE_DIM,  # kept for backward compat
        hidden_dim: int = 128,
        num_actions: int = NUM_ACTIONS,
        ctx_dim: int = EXECUTOR_STATE_DIM,
        q_lookback: int = Q_LOOKBACK,
        num_horizons: int = 4,
        num_head_actions: int = 3,
        tower_dim: int = 96,
    ):
        super().__init__()
        self.num_features = num_features
        self.ctx_dim = ctx_dim
        self.q_lookback = q_lookback
        self.num_horizons = num_horizons
        self.tower_dim = tower_dim

        # --- Shared Branch A: full indicator window (B, T, F) → channels-first Conv1D ---
        in_channels = max(num_features, 1)
        self.feat_conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.feat_bn1   = nn.BatchNorm1d(64)
        self.feat_conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.feat_bn2   = nn.BatchNorm1d(64)
        self.feat_pool  = nn.AdaptiveAvgPool1d(1)
        self.feat_fc    = nn.Linear(64, 64)

        # --- Shared Branch B: 28-dim context dense encoder ---
        self.b1_fc1 = nn.Linear(ctx_dim, hidden_dim)
        self.b1_ln1 = nn.LayerNorm(hidden_dim)
        self.b1_fc2 = nn.Linear(hidden_dim, 64)
        self.b1_ln2 = nn.LayerNorm(64)

        self.b2_meta   = nn.Linear(10, 16)
        self.b2_risk   = nn.Linear(5, 16)
        self.b2_zone   = nn.Linear(8, 16)
        self.b2_time   = nn.Linear(5, 16)
        self.b2_fusion = nn.Linear(64, 64)
        self.b2_ln     = nn.LayerNorm(64)

        # --- Shared fusion trunk ---
        self.fusion_fc = nn.Linear(64 + 64 + 64, hidden_dim)  # feat_h + b1_out + b2_out
        self.fusion_ln = nn.LayerNorm(hidden_dim)

        # --- Specialist towers (full MLPs — real capacity, not tiny heads) ---
        def _tower():
            return nn.Sequential(
                nn.Linear(hidden_dim, tower_dim),
                nn.LayerNorm(tower_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(tower_dim, tower_dim),
                nn.LayerNorm(tower_dim),
                nn.SiLU(),
            )

        self.buy_towers = nn.ModuleList([_tower() for _ in range(num_horizons)])
        self.sell_towers = nn.ModuleList([_tower() for _ in range(num_horizons)])

        # Side logits: Buy → [WAIT, CALL], Sell → [WAIT, PUT]
        self.buy_side_heads = nn.ModuleList([
            nn.Linear(tower_dim, 2) for _ in range(num_horizons)
        ])
        self.sell_side_heads = nn.ModuleList([
            nn.Linear(tower_dim, 2) for _ in range(num_horizons)
        ])

        # Final decision fusion: [buy_emb | sell_emb | shared] → WAIT/CALL/PUT
        self.decision_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(tower_dim + tower_dim + hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 64),
                nn.SiLU(),
                nn.Linear(64, num_head_actions),
            )
            for _ in range(num_horizons)
        ])

    def _encode_feat(self, feat_window: torch.Tensor) -> torch.Tensor:
        """Encode (B, T, F) feature window → (B, 64)."""
        x = feat_window.transpose(1, 2)                              # (B, F, T)
        x = torch.nn.functional.silu(self.feat_bn1(self.feat_conv1(x)))
        x = torch.nn.functional.silu(self.feat_bn2(self.feat_conv2(x)))
        x = self.feat_pool(x).squeeze(-1)                            # (B, 64)
        return torch.nn.functional.silu(self.feat_fc(x))

    def _encode_ctx(self, ctx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode 28-dim context → (b1_out(64), b2_out(64))."""
        h      = torch.nn.functional.silu(self.b1_ln1(self.b1_fc1(ctx)))
        b1_out = torch.nn.functional.silu(self.b1_ln2(self.b1_fc2(h)))

        meta = torch.nn.functional.silu(self.b2_meta(ctx[:, 0:10]))
        risk = torch.nn.functional.silu(self.b2_risk(ctx[:, 10:15]))
        zone = torch.nn.functional.silu(self.b2_zone(ctx[:, 15:23]))
        time = torch.nn.functional.silu(self.b2_time(ctx[:, 23:28]))
        b2_out = torch.nn.functional.silu(self.b2_ln(self.b2_fusion(torch.cat([meta, risk, zone, time], dim=-1))))
        return b1_out, b2_out

    def forward(
        self,
        feat_window: torch.Tensor,
        ctx: torch.Tensor,
        horizon_idx: Optional[int] = None,
        return_aux: bool = False,
        return_sides: bool = False,
    ) -> Any:
        """
        feat_window  : (B, Q_LOOKBACK, num_features)
        ctx          : (B, 28)
        horizon_idx  : int | None  — if set returns (B, 3) for that head only
                                      otherwise returns (B, 4, 3)
        return_aux   : kept for API compat
        return_sides : bool — if True also returns buy_side (B,2) and sell_side (B,2)
        """
        feat_h         = self._encode_feat(feat_window)
        b1_out, b2_out = self._encode_ctx(ctx)
        shared = torch.nn.functional.silu(
            self.fusion_ln(
                self.fusion_fc(torch.cat([feat_h, b1_out, b2_out], dim=-1))
            )
        )

        def _one(h_idx_int):
            buy_e = self.buy_towers[h_idx_int](shared)
            sell_e = self.sell_towers[h_idx_int](shared)
            buy_side = self.buy_side_heads[h_idx_int](buy_e)    # WAIT, CALL
            sell_side = self.sell_side_heads[h_idx_int](sell_e)  # WAIT, PUT
            fused = self.decision_heads[h_idx_int](torch.cat([buy_e, sell_e, shared], dim=-1))
            return fused, buy_side, sell_side

        if horizon_idx is not None:
            h = int(horizon_idx)
            fused, buy_side, sell_side = _one(h)
            if return_sides:
                return fused, buy_side, sell_side
            return fused

        outs, buys, sells = [], [], []
        for h in range(self.num_horizons):
            f, b, s = _one(h)
            outs.append(f)
            buys.append(b)
            sells.append(s)
        stacked = torch.stack(outs, dim=1)
        if return_sides:
            return stacked, torch.stack(buys, dim=1), torch.stack(sells, dim=1)
        return stacked


# ── Module-level feature window helpers ──────────────────────────────────────

def build_feat_window(num_matrix: np.ndarray, abs_idx: int, q_lookback: int = Q_LOOKBACK) -> np.ndarray:
    """
    Build a (q_lookback, num_features) feature window for the Q-executor.
    Only uses rows <= abs_idx (no lookahead). Left-pads with zeros if needed.
    """
    start = abs_idx - q_lookback + 1
    if start >= 0:
        return num_matrix[start: abs_idx + 1].astype(np.float32)
    # Left-pad with zeros
    window = np.zeros((q_lookback, num_matrix.shape[1]), dtype=np.float32)
    available = num_matrix[: abs_idx + 1]
    window[-len(available):] = available
    return window


def build_feat_window_batch(
    num_matrix: np.ndarray,
    abs_indices: Any,
    q_lookback: int = Q_LOOKBACK,
) -> np.ndarray:
    """Vectorised version of build_feat_window for a batch of indices."""
    return np.stack([build_feat_window(num_matrix, int(i), q_lookback) for i in abs_indices])


class OptionsQExecutor:
    """
    Tier 2 Q-Learner Trade Executor for Options.
    """

    def __init__(
        self,
        num_features: int = 0,
        input_dim: int = EXECUTOR_STATE_DIM,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        device: str = "cpu",
    ):
        self.input_dim = input_dim
        self.num_features = num_features
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        self.policy_net = ExecutorQNetwork(num_features=num_features, input_dim=input_dim).to(self.device)
        self.target_net = ExecutorQNetwork(num_features=num_features, input_dim=input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.action_mask_engine = HardActionMask()
        self.replay_buffer: List[Tuple] = []
        self.buffer_capacity = buffer_capacity
        self.regret_buffer: collections.deque = collections.deque(maxlen=10_000)

    # ── State Representation Construction ────────────────────────────────────

    def build_state_vector(
        self,
        htf_bias: HTFBiasPackage,
        account: AccountContext,
        exec_ctx: ExecutionContext,
        zone_manager: ZoneSnapshotManager,
    ) -> np.ndarray:
        """
        Construct a normalized 28-dim state vector representing the full LTF execution context,
        including HTF Bias, Multi-Horizon Meta hints (5m, 15m, 30m, 1h), SNR Zone Proximity, Zonal Volume, Account State, and Time/Session Features.
        """
        nearest_supp, nearest_res = zone_manager.get_nearest_zones(exec_ctx.current_price)

        supp_dist = abs(exec_ctx.current_price - nearest_supp.price_level) / exec_ctx.current_price if nearest_supp else 1.0
        res_dist = abs(exec_ctx.current_price - nearest_res.price_level) / exec_ctx.current_price if nearest_res else 1.0

        supp_vol_ratio = nearest_supp.volume_delta_ratio if nearest_supp else 0.0
        res_vol_ratio = nearest_res.volume_delta_ratio if nearest_res else 0.0

        total_vol = exec_ctx.buy_volume + exec_ctx.sell_volume
        vol_delta_ratio = (exec_ctx.buy_volume - exec_ctx.sell_volume) / (total_vol + 1e-6)

        tf_flag = 1.0 if exec_ctx.ltf_timeframe == "15m" else 0.0
        dir_flag = 1.0 if htf_bias.direction == "bullish" else (-1.0 if htf_bias.direction == "bearish" else 0.0)

        # Multi-Horizon Strengths (5m, 15m, 30m, 1h)
        hs = htf_bias.horizon_strengths if len(htf_bias.horizon_strengths) == 4 else [0.5, 0.5, 0.5, 0.5]

        # Time & Session cyclical encodings & flags
        sin_hour = float(np.sin(2 * np.pi * exec_ctx.hour_of_day / 24.0))
        cos_hour = float(np.cos(2 * np.pi * exec_ctx.hour_of_day / 24.0))
        dow_norm = float(exec_ctx.day_of_week) / 6.0
        is_nyse_open = 1.0 if exec_ctx.session_phase == "nyse_open" else 0.0
        is_power_hour = 1.0 if exec_ctx.session_phase == "nyse_power_hour" else 0.0

        state = np.array([
            # Meta & Multi-Horizon Hints (10 features)
            dir_flag,
            float(htf_bias.strength),
            float(htf_bias.reversal_prob),
            float(htf_bias.q_value),
            float(htf_bias.expected_mfe_pips) / 100.0,
            float(htf_bias.expected_mae_pips) / 100.0,
            float(hs[0]),
            float(hs[1]),
            float(hs[2]),
            float(hs[3]),
            # Account Context (5 features)
            float(account.daily_drawdown_pct),
            1.0 if account.open_position_type == "CALL" else (-1.0 if account.open_position_type == "PUT" else 0.0),
            float(account.open_position_pnl_pct),
            float(account.win_streak) / 10.0,
            float(account.loss_streak) / 10.0,
            # Execution & Zone Context (8 features)
            tf_flag,
            float(exec_ctx.atr) / exec_ctx.current_price,
            float(supp_dist),
            float(res_dist),
            float(supp_vol_ratio),
            float(res_vol_ratio),
            float(vol_delta_ratio),
            float(exec_ctx.reentries_in_window) / float(exec_ctx.max_reentries_allowed),
            # Time & Session Context (5 features)
            sin_hour,
            cos_hour,
            dow_norm,
            is_nyse_open,
            is_power_hour,
        ], dtype=np.float32)

        return state

    # ── Action Selection with Hard Action Masking ─────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        feat_window: Optional[np.ndarray] = None,
        eval_mode: bool = False,
        eval_epsilon: float = 0.10,
        horizon_idx: int = 0,
    ) -> int:
        """
        Select action using Epsilon-Greedy with Hard Action Masking.
        Masked actions (0 in action_mask) are set to -infinity Q-value.

        feat_window : (Q_LOOKBACK, num_features) or None — zeros used if absent.
        horizon_idx : which horizon head to query (0=5m, 1=15m, 2=30m, 3=1h).

        In eval_mode, a small residual epsilon (eval_epsilon) is retained so
        that an undertrained network with WAIT-bias does not collapse to 100%
        WAIT during Phase 3 assessment.
        """
        # The horizon heads output 3 actions (WAIT/CALL/PUT); map back to full action space
        # by only masking the first 3 slots; TAKE_PROFIT / CLOSE are positional overrides.
        # For execution-gate decisions we operate on the 3-action subspace.
        valid_actions = np.where(action_mask[:3] == 1)[0] if len(action_mask) >= 3 else np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return ACTION_WAIT  # Safe fallback

        eps = eval_epsilon if eval_mode else self.epsilon
        if random.random() < eps:
            return int(random.choice(valid_actions))

        with torch.no_grad():
            t_state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            if feat_window is not None:
                t_feat = torch.tensor(feat_window[None, ...], dtype=torch.float32, device=self.device)
            else:
                nf = max(self.num_features, 1)
                t_feat = torch.zeros(1, self.policy_net.q_lookback, nf, dtype=torch.float32, device=self.device)
            q_values = self.policy_net(t_feat, t_state, horizon_idx=horizon_idx).squeeze(0).cpu().numpy()

        # Apply action mask over the 3-action subspace
        mask_3 = action_mask[:3] if len(action_mask) >= 3 else action_mask
        masked_q = np.where(mask_3 == 1, q_values, -1e9)
        return int(np.argmax(masked_q))

    # ── Replay Buffer & Training Step ─────────────────────────────────────────

    def record_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
        feat_window: Optional[np.ndarray] = None,
        next_feat_window: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record transition into replay buffer.
        If feat_window and next_feat_window are provided, stores 7-tuple:
            (feat_window, state, action, reward, next_feat_window, next_state, done)
        Otherwise falls back to legacy 6-tuple:
            (state, action, reward, next_state, done, next_action_mask)
        """
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)
        if feat_window is not None and next_feat_window is not None:
            self.replay_buffer.append((feat_window, state, action, reward, next_feat_window, next_state, done))
        else:
            self.replay_buffer.append((state, action, reward, next_state, done, next_action_mask))

    def record_regret_transition(
        self,
        feat_window: np.ndarray,
        ctx_state: np.ndarray,
        next_feat_window: np.ndarray,
        next_ctx_state: np.ndarray,
        next_action_mask: np.ndarray,
        htf_bias: "HTFBiasPackage",
        future_closes: np.ndarray,
        entry_price: float,
    ) -> None:
        """
        Record a regret transition when ACTION_WAIT was chosen but a profitable
        entry existed.  Only records for the BEST profitable horizon found.

        Called by the caller AFTER choosing ACTION_WAIT when at least one horizon
        would have produced |fwd_pct| >= REGRET_MIN_PCT in the correct direction.
        """
        for h, bars in enumerate(HORIZON_BARS_LIST):
            if len(future_closes) < bars:
                continue
            fwd_pct = float((future_closes[bars - 1] - entry_price) / (entry_price + 1e-8))
            is_bullish = htf_bias.direction == "bullish"
            is_bearish = htf_bias.direction == "bearish"
            directional = (is_bullish and fwd_pct > 0) or (is_bearish and fwd_pct < 0)
            if abs(fwd_pct) >= REGRET_MIN_PCT and directional:
                regret_reward = max(-3.0, -abs(fwd_pct) * REGRET_REWARD_SCALE)
                self.regret_buffer.append(RegretTransition(
                    feat_window=feat_window,
                    ctx_state=ctx_state,
                    action=ACTION_WAIT,
                    regret_reward=regret_reward,
                    next_feat_window=next_feat_window,
                    next_ctx_state=next_ctx_state,
                    next_action_mask=next_action_mask,
                    horizon_idx=h,
                    counterfactual_pct=fwd_pct,
                ))
                return  # Record only for the first profitable horizon found

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """
        Perform a single Q-learning gradient update step.

        Supports both 7-tuple (feat_window, ctx, action, reward, next_feat, next_ctx, done)
        and legacy 6-tuple replay buffer entries.
        Uses horizon 0 (5m) for the simplified single-step update; the full
        multi-horizon training is implemented in the notebook Cell 8.
        """
        if len(self.replay_buffer) < batch_size:
            return None

        batch = random.sample(self.replay_buffer, batch_size)

        if len(batch[0]) == 7:
            # New 7-tuple: (feat_window, ctx, action, reward, next_feat, next_ctx, done)
            feat_windows  = np.array([b[0] for b in batch], dtype=np.float32)
            states        = np.array([b[1] for b in batch], dtype=np.float32)
            actions       = np.array([b[2] for b in batch], dtype=np.int64)
            rewards       = np.array([b[3] for b in batch], dtype=np.float32)
            next_feat     = np.array([b[4] for b in batch], dtype=np.float32)
            next_states   = np.array([b[5] for b in batch], dtype=np.float32)
            dones_arr     = np.array([float(b[6]) for b in batch], dtype=np.float32)
        else:
            # Legacy 6-tuple: (state, action, reward, next_state, done, next_action_mask)
            states, actions, rewards, next_states, dones_raw, _ = zip(*batch)
            states      = np.array(states, dtype=np.float32)
            actions     = np.array(actions, dtype=np.int64)
            rewards     = np.array(rewards, dtype=np.float32)
            next_states = np.array(next_states, dtype=np.float32)
            dones_arr   = np.array([float(d) for d in dones_raw], dtype=np.float32)
            nf = max(self.num_features, 1)
            feat_windows = np.zeros((batch_size, self.policy_net.q_lookback, nf), dtype=np.float32)
            next_feat    = np.zeros_like(feat_windows)

        t_feat    = torch.tensor(feat_windows, dtype=torch.float32, device=self.device)
        t_states  = torch.tensor(states,       dtype=torch.float32, device=self.device)
        t_actions = torch.tensor(actions,      dtype=torch.int64,   device=self.device).unsqueeze(1)
        t_rewards = torch.tensor(rewards,      dtype=torch.float32, device=self.device).unsqueeze(1)
        t_nfeat   = torch.tensor(next_feat,    dtype=torch.float32, device=self.device)
        t_nstates = torch.tensor(next_states,  dtype=torch.float32, device=self.device)
        t_dones   = torch.tensor(dones_arr,    dtype=torch.float32, device=self.device).unsqueeze(1)

        # Use horizon 0 for simplified single-horizon training step
        q_vals  = self.policy_net(t_feat, t_states, horizon_idx=0)   # (B, 3)
        q_taken = q_vals.gather(1, t_actions)

        with torch.no_grad():
            next_q      = self.target_net(t_nfeat, t_nstates, horizon_idx=0)
            best_next_q = next_q.max(dim=1, keepdim=True).values
            q_target    = t_rewards + (1.0 - t_dones) * self.gamma * best_next_q

        loss = nn.SmoothL1Loss()(q_taken, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # --- Regret buffer (Track B): 20% sampling rate ---
        regret_min = max(1, batch_size // 5)
        if len(self.regret_buffer) >= regret_min and random.random() < 0.20:
            regret_sample = random.sample(list(self.regret_buffer), regret_min)
            r_feat    = torch.tensor(np.array([r.feat_window for r in regret_sample]),
                                     dtype=torch.float32, device=self.device)
            r_ctx     = torch.tensor(np.array([r.ctx_state for r in regret_sample]),
                                     dtype=torch.float32, device=self.device)
            r_nfeat   = torch.tensor(np.array([r.next_feat_window for r in regret_sample]),
                                     dtype=torch.float32, device=self.device)
            r_nctx    = torch.tensor(np.array([r.next_ctx_state for r in regret_sample]),
                                     dtype=torch.float32, device=self.device)
            r_rewards = torch.tensor([r.regret_reward for r in regret_sample],
                                     dtype=torch.float32, device=self.device).unsqueeze(1)
            r_hidx    = regret_sample[0].horizon_idx   # train each sample on its horizon

            r_q_vals  = self.policy_net(r_feat, r_ctx, horizon_idx=r_hidx)  # (B, 3)
            r_wait    = torch.zeros(regret_min, 1, dtype=torch.int64, device=self.device)  # always WAIT
            r_q_taken = r_q_vals.gather(1, r_wait)

            with torch.no_grad():
                r_next_q     = self.target_net(r_nfeat, r_nctx, horizon_idx=r_hidx)
                r_best_next  = r_next_q.max(dim=1, keepdim=True).values
                r_q_target   = r_rewards + self.gamma * r_best_next   # no done flag — always terminal

            r_loss = nn.SmoothL1Loss()(r_q_taken, r_q_target)
            self.optimizer.zero_grad()
            r_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def update_target_network(self) -> None:
        """Sync target network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Bias-Persistence Re-Entry Verification ────────────────────────────────

    def can_reenter(
        self,
        htf_bias: HTFBiasPackage,
        exec_ctx: ExecutionContext,
        zone_manager: ZoneSnapshotManager,
    ) -> bool:
        """
        Verify if Bias-Persistence Re-Entry is allowed for a closed position.
        Conditions:
        1. HTF conviction >= 0.65.
        2. HTF reversal prob <= 0.35.
        3. Nearest structural zone remains unbroken.
        4. Re-entry count within max allowed limit.
        """
        if exec_ctx.reentries_in_window >= exec_ctx.max_reentries_allowed:
            return False

        if htf_bias.strength < 0.65 or htf_bias.reversal_prob > 0.35:
            return False

        nearest_supp, nearest_res = zone_manager.get_nearest_zones(exec_ctx.current_price)
        if htf_bias.direction == "bullish" and nearest_supp and nearest_supp.is_invalidated:
            return False
        if htf_bias.direction == "bearish" and nearest_res and nearest_res.is_invalidated:
            return False

        return True

    # ── Reward Shaping & Hindsight Missed-Opportunity Penalty ──────────────────

    def calculate_executor_reward(
        self,
        action: int,
        action_mask: np.ndarray,
        pnl_pct: float,
        max_drawdown_exposed: float,
        forward_move_pct: float,
        htf_bias: HTFBiasPackage,
        is_closed: bool = False,
        is_stop_loss: bool = False,
        risk_limit_breached: bool = False,
    ) -> float:
        """
        Calculate delayed risk-adjusted reward for LTF execution adhering to options-repurposing-directives.md.

        Components:
        1. Overtrade Churn Penalty (-0.05) on BUY_CALL / BUY_PUT entry to account for option spreads.
        2. Best Price Entry Bonus (+0.15): Rewarded when entering in direction of move with low drawdown.
        3. Realized P&L & Stop Loss reward on position close.
        4. Wise Patience Reward vs Hindsight Missed-Opportunity Penalty on ACTION_WAIT:
           - Rewards ACTION_WAIT when the unmasked setup would have resulted in an adverse loss.
           - Penalizes ACTION_WAIT when all entry gates passed, HTF strength >= 0.65, and move played out.
           - Grants small discipline bonus (+0.02) when standing down in low-conviction regimes.
        """
        reward = 0.0

        # 1. Entry & Quality Sizing
        if action in (ACTION_BUY_CALL, ACTION_BUY_PUT):
            reward -= 0.05  # Churn penalty for option spread

            # Best Price Entry Bonus: minimal drawdown & strong move in setup direction
            if action == ACTION_BUY_CALL and forward_move_pct >= 0.003:
                dd_factor = max(0.0, 1.0 - (max_drawdown_exposed / 0.005))
                reward += float(np.clip(0.15 * (forward_move_pct / 0.003) * dd_factor, 0.0, 0.35))
            elif action == ACTION_BUY_PUT and forward_move_pct <= -0.003:
                dd_factor = max(0.0, 1.0 - (max_drawdown_exposed / 0.005))
                reward += float(np.clip(0.15 * (abs(forward_move_pct) / 0.003) * dd_factor, 0.0, 0.35))

        # 2. Realized P&L & Stop Loss reward on position close
        if is_closed:
            risk_denom = max(abs(max_drawdown_exposed), 0.01)
            reward += float(pnl_pct / risk_denom)

            if is_stop_loss:
                # Reinforce correct stop-loss execution vs holding losers
                reward += 0.20

        # 3. Wise Patience Reward vs Hindsight Missed-Opportunity Penalty (ACTION_WAIT)
        if action == ACTION_WAIT and not is_closed and not risk_limit_breached:
            # Bullish Setup Evaluation (BUY_CALL unmasked)
            if action_mask[ACTION_BUY_CALL] == 1 and htf_bias.strength >= 0.65:
                if forward_move_pct >= 0.003:
                    # Missed winning setup — heavy penalty scaled to move magnitude
                    # 3x multiplier vs patience bonus; cap -1.50 to outweigh Q(WAIT) accumulation
                    penalty = float(np.clip(-0.45 * (forward_move_pct / 0.003), -1.50, 0.0))
                    reward += penalty
                    logger.debug("[HindsightReward] Missed Call entry penalty applied: %.3f", penalty)
                elif forward_move_pct < 0.0:
                    # Wise patience bonus: avoided a losing bullish entry
                    patience_bonus = float(np.clip(0.15 * (abs(forward_move_pct) / 0.003), 0.0, 0.30))
                    reward += patience_bonus
                    logger.debug("[PatienceReward] Avoided Call loss bonus applied: %.3f", patience_bonus)

            # Bearish Setup Evaluation (BUY_PUT unmasked)
            elif action_mask[ACTION_BUY_PUT] == 1 and htf_bias.strength >= 0.65:
                if forward_move_pct <= -0.003:
                    # Missed winning setup — heavy penalty scaled to move magnitude
                    penalty = float(np.clip(-0.45 * (abs(forward_move_pct) / 0.003), -1.50, 0.0))
                    reward += penalty
                    logger.debug("[HindsightReward] Missed Put entry penalty applied: %.3f", penalty)
                elif forward_move_pct > 0.0:
                    # Wise patience bonus: avoided a losing bearish entry
                    patience_bonus = float(np.clip(0.15 * (forward_move_pct / 0.003), 0.0, 0.30))
                    reward += patience_bonus
                    logger.debug("[PatienceReward] Avoided Put loss bonus applied: %.3f", patience_bonus)

            else:
                # Discipline bonus for staying flat when no high-conviction setup exists
                reward += 0.02

        return float(np.clip(reward, -3.0, 3.0))
