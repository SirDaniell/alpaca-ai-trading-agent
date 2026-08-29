"""
q_executor.py — Tier 2 Q-Learner Trade Executor for Options.

Roles:
- Runs on LTF (5m / 15m).
- Consumes Tier 1 Meta-Learner HTF bias package, active Zone Snapshots, Account State, and Execution context.
- Applies HardActionMask to enforce no-chase entry discipline and volume delta confirmation.
- Implements Bias-Persistence Re-Entry (fixed risk sizing, NO martingale scaling).
- Evaluates delayed risk-adjusted rewards with overtrade churn penalties and training-time hindsight missed-opportunity shaping.
"""

from __future__ import annotations

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

EXECUTOR_STATE_DIM = 24  # Compact representation for LTF execution context


@dataclass
class HTFBiasPackage:
    """Tier 1 Meta-Learner output passed down to Tier 2 Executor."""
    direction: str = "neutral"  # "bullish", "bearish", "neutral"
    strength: float = 0.0      # [0.0, 1.0]
    reversal_prob: float = 0.0 # [0.0, 1.0]
    q_value: float = 0.0
    expected_mfe_pips: float = 0.0
    expected_mae_pips: float = 0.0


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
    """Deep Q-Network for LTF Options Execution."""

    def __init__(self, input_dim: int = EXECUTOR_STATE_DIM, hidden_dim: int = 128, num_actions: int = NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OptionsQExecutor:
    """
    Tier 2 Q-Learner Trade Executor for Options.
    """

    def __init__(
        self,
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
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        self.policy_net = ExecutorQNetwork(input_dim=input_dim).to(self.device)
        self.target_net = ExecutorQNetwork(input_dim=input_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.action_mask_engine = HardActionMask()
        self.replay_buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = []
        self.buffer_capacity = buffer_capacity

    # ── State Representation Construction ────────────────────────────────────

    def build_state_vector(
        self,
        htf_bias: HTFBiasPackage,
        account: AccountContext,
        exec_ctx: ExecutionContext,
        zone_manager: ZoneSnapshotManager,
    ) -> np.ndarray:
        """
        Construct a normalized 24-dim state vector representing the full LTF execution context,
        including HTF Bias, SNR Zone Proximity, Zonal Volume, Account State, and Time/Session Features.
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

        # Time & Session cyclical encodings & flags
        sin_hour = float(np.sin(2 * np.pi * exec_ctx.hour_of_day / 24.0))
        cos_hour = float(np.cos(2 * np.pi * exec_ctx.hour_of_day / 24.0))
        dow_norm = float(exec_ctx.day_of_week) / 6.0
        is_nyse_open = 1.0 if exec_ctx.session_phase == "nyse_open" else 0.0
        is_power_hour = 1.0 if exec_ctx.session_phase == "nyse_power_hour" else 0.0

        state = np.array([
            dir_flag,
            float(htf_bias.strength),
            float(htf_bias.reversal_prob),
            float(htf_bias.q_value),
            float(htf_bias.expected_mfe_pips) / 100.0,
            float(htf_bias.expected_mae_pips) / 100.0,
            # Account context
            float(account.daily_drawdown_pct),
            1.0 if account.open_position_type == "CALL" else (-1.0 if account.open_position_type == "PUT" else 0.0),
            float(account.open_position_pnl_pct),
            float(account.win_streak) / 10.0,
            float(account.loss_streak) / 10.0,
            # Execution & Zone Context
            tf_flag,
            float(exec_ctx.atr) / exec_ctx.current_price,
            float(supp_dist),
            float(res_dist),
            float(supp_vol_ratio),
            float(res_vol_ratio),
            float(vol_delta_ratio),
            float(exec_ctx.reentries_in_window) / float(exec_ctx.max_reentries_allowed),
            # Time & Session Context
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
        eval_mode: bool = False,
    ) -> int:
        """
        Select action using Epsilon-Greedy with Hard Action Masking.
        Masked actions (0 in action_mask) are set to -infinity Q-value.
        """
        valid_actions = np.where(action_mask == 1)[0]
        if len(valid_actions) == 0:
            return ACTION_WAIT  # Safe fallback

        if not eval_mode and random.random() < self.epsilon:
            return int(random.choice(valid_actions))

        with torch.no_grad():
            t_state = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(t_state).squeeze(0).cpu().numpy()

        # Apply action mask: set invalid actions to -infinity
        masked_q = np.where(action_mask == 1, q_values, -1e9)
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
    ) -> None:
        """Record transition into replay buffer."""
        if len(self.replay_buffer) >= self.buffer_capacity:
            self.replay_buffer.pop(0)
        self.replay_buffer.append((state, action, reward, next_state, done, next_action_mask))

    def train_step(self, batch_size: int = 32) -> Optional[float]:
        """Perform a single Q-learning gradient update step."""
        if len(self.replay_buffer) < batch_size:
            return None

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        t_states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        t_actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        t_rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        t_next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        t_dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        t_next_masks = torch.tensor(np.array(next_masks), dtype=torch.float32, device=self.device)

        # Current Q-values
        q_eval = self.policy_net(t_states).gather(1, t_actions)

        # Double Q-Learning target calculation with Action Masking
        with torch.no_grad():
            next_q_policy = self.policy_net(t_next_states)
            # Mask invalid actions
            masked_next_q = torch.where(t_next_masks == 1, next_q_policy, torch.tensor(-1e9, device=self.device))
            best_next_actions = masked_next_q.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(t_next_states).gather(1, best_next_actions)
            q_target = t_rewards + (1.0 - t_dones) * self.gamma * next_q_target

        loss = nn.SmoothL1Loss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
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
