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
EXECUTOR_STATE_DIM = 28  # Compact representation for LTF execution context + Multi-Horizon hints


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
    Two-Branch Ensemble Deep Q-Network for LTF Options Execution.
    Mirrors build_classification_ensemble_model (tt.py) architecture:
    - Branch 1 (Dense Feature Tower): Deep representation over all 28 state features with LayerNorm & SiLU.
    - Branch 2 (Multi-Scale Grouped Tower): Grouped projections over Meta-Learner (0..9), Risk (10..14), Zone (15..22), and Time (23..27) features.
    - Supervised Aux Heads (aux1_head, aux2_head): Independent aux predictions per branch before fusion.
    - StopGradient Isolation: aux1 & aux2 predictions are detached before feeding into Gated Fusion Layer.
    - Gated Ensemble Fusion Head: Combines b1_out, b2_out, aux1_detached, aux2_detached for robust Q-value estimation.
    """

    def __init__(self, input_dim: int = EXECUTOR_STATE_DIM, hidden_dim: int = 128, num_actions: int = NUM_ACTIONS):
        super().__init__()
        # Branch 1: Dense Feature Extraction Tower
        self.b1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.b1_ln1 = nn.LayerNorm(hidden_dim)
        self.b1_act1 = nn.SiLU()
        self.b1_drop = nn.Dropout(0.2)
        self.b1_fc2 = nn.Linear(hidden_dim, 64)
        self.b1_ln2 = nn.LayerNorm(64)
        self.b1_act2 = nn.SiLU()

        # Branch 2: Multi-Scale Grouped Feature Tower
        # Slice inputs into 4 semantic sub-groups: Meta (10), Risk (5), Zone (8), Time (5)
        self.b2_meta = nn.Linear(10, 32)
        self.b2_risk = nn.Linear(5, 32)
        self.b2_zone = nn.Linear(8, 32)
        self.b2_time = nn.Linear(5, 32)
        self.b2_fusion = nn.Linear(128, 64)
        self.b2_ln = nn.LayerNorm(64)
        self.b2_act = nn.SiLU()

        # Auxiliary Supervised Heads (1 per branch)
        self.aux1_head = nn.Linear(64, num_actions)
        self.aux2_head = nn.Linear(64, num_actions)

        # Gated Ensemble Fusion Head
        # Inputs: b1_out (64) + b2_out (64) + aux1_detached (5) + aux2_detached (5) = 138
        self.fusion_fc1 = nn.Linear(64 + 64 + num_actions + num_actions, hidden_dim)
        self.fusion_ln1 = nn.LayerNorm(hidden_dim)
        self.fusion_act1 = nn.SiLU()
        self.fusion_out = nn.Linear(hidden_dim, num_actions)

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ── Branch 1 ────────────────────────────────────────────────────────
        b1 = self.b1_act1(self.b1_ln1(self.b1_fc1(x)))
        b1 = self.b1_drop(b1)
        b1_out = self.b1_act2(self.b1_ln2(self.b1_fc2(b1)))

        # ── Branch 2 ────────────────────────────────────────────────────────
        meta_feats = x[:, :10]
        risk_feats = x[:, 10:15]
        zone_feats = x[:, 15:23]
        time_feats = x[:, 23:28]

        b2_m = torch.relu(self.b2_meta(meta_feats))
        b2_r = torch.relu(self.b2_risk(risk_feats))
        b2_z = torch.relu(self.b2_zone(zone_feats))
        b2_t = torch.relu(self.b2_time(time_feats))
        b2_cat = torch.cat([b2_m, b2_r, b2_z, b2_t], dim=-1)
        b2_out = self.b2_act(self.b2_ln(self.b2_fusion(b2_cat)))

        # ── Independent Auxiliary Heads ─────────────────────────────────────
        aux1_q = self.aux1_head(b1_out)
        aux2_q = self.aux2_head(b2_out)

        # ── StopGradient Isolation ──────────────────────────────────────────
        aux1_sg = aux1_q.detach()
        aux2_sg = aux2_q.detach()

        # ── Gated Fusion Layer ───────────────────────────────────────────────
        fusion_in = torch.cat([b1_out, b2_out, aux1_sg, aux2_sg], dim=-1)
        fused = self.fusion_act1(self.fusion_ln1(self.fusion_fc1(fusion_in)))
        q_final = self.fusion_out(fused)

        if return_aux:
            return q_final, aux1_q, aux2_q
        return q_final



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

        # Current Q-values with auxiliary outputs
        q_eval, aux1_eval, aux2_eval = self.policy_net(t_states, return_aux=True)
        q_eval_sel = q_eval.gather(1, t_actions)
        aux1_sel = aux1_eval.gather(1, t_actions)
        aux2_sel = aux2_eval.gather(1, t_actions)

        # Double Q-Learning target calculation with Action Masking
        with torch.no_grad():
            next_q_policy = self.policy_net(t_next_states)
            # Mask invalid actions
            masked_next_q = torch.where(t_next_masks == 1, next_q_policy, torch.tensor(-1e9, device=self.device))
            best_next_actions = masked_next_q.argmax(dim=1, keepdim=True)

            next_q_target = self.target_net(t_next_states).gather(1, best_next_actions)
            q_target = t_rewards + (1.0 - t_dones) * self.gamma * next_q_target

        smooth_l1 = nn.SmoothL1Loss()
        main_loss = smooth_l1(q_eval_sel, q_target)
        aux1_loss = smooth_l1(aux1_sel, q_target)
        aux2_loss = smooth_l1(aux2_sel, q_target)

        total_loss = main_loss + 0.2 * aux1_loss + 0.2 * aux2_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(total_loss.item())


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
                    # Missed winning setup penalty
                    penalty = float(np.clip(-0.15 * (forward_move_pct / 0.003), -0.5, 0.0))
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
                    # Missed winning setup penalty
                    penalty = float(np.clip(-0.15 * (abs(forward_move_pct) / 0.003), -0.5, 0.0))
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

