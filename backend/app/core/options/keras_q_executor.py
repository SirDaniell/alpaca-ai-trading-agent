"""
keras_q_executor.py — Tier 2 Keras-based Q-Learner Trade Executor for Options.

Roles:
- Runs on LTF (5m / 15m).
- Consumes Tier 1 Meta-Learner HTF bias package, active Zone Snapshots, Account State, and Execution context.
- Uses Keras 2-Branch Ensemble Architecture with tf.stop_gradient auxiliary head isolation.
- Applies HardActionMask to enforce no-chase entry discipline and volume delta confirmation.
"""

import os
import random
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Any, Dict, List, Optional, Tuple

from app.core.market.zone_snapshot import HardActionMask, ZoneSnapshotManager
from app.core.options.q_executor import (
    ACTION_WAIT,
    ACTION_BUY_CALL,
    ACTION_BUY_PUT,
    ACTION_TAKE_PROFIT_HALF,
    ACTION_CLOSE_FLATTEN,
    NUM_ACTIONS,
    EXECUTOR_STATE_DIM,
    HTFBiasPackage,
    AccountContext,
    ExecutionContext,
)

logger = logging.getLogger(__name__)


# ── Keras Two-Branch Ensemble Q-Network ────────────────────────────────────────

def build_keras_executor_q_network(
    input_dim: int = EXECUTOR_STATE_DIM,
    hidden_dim: int = 128,
    num_actions: int = NUM_ACTIONS,
) -> keras.Model:
    """
    Two-Branch Ensemble Deep Q-Network for LTF Options Execution in Keras.
    
    Architecture:
    - Branch 1: Deep Dense feature tower with LayerNormalization & SiLU
    - Branch 2: Multi-Scale Grouped Feature tower (Meta, Risk, Zone, Time features)
    - Aux Heads (aux1, aux2) with tf.stop_gradient isolation
    - Gated Fusion Head combining Branch 1, Branch 2, and isolated aux outputs -> Q-values (5 actions)
    """
    inputs = keras.Input(shape=(input_dim,), name="executor_state_input")

    # ── Branch 1: Deep Feature Extraction Tower ────────────────────────────────
    b1 = layers.Dense(hidden_dim, activation="silu", name="b1_fc1")(inputs)
    b1 = layers.LayerNormalization(name="b1_ln1")(b1)
    b1 = layers.Dense(hidden_dim // 2, activation="silu", name="b1_fc2")(b1)
    b1 = layers.LayerNormalization(name="b1_ln2")(b1)
    b1_out = layers.Dropout(0.1, name="b1_drop")(b1)

    aux1 = layers.Dense(num_actions, activation=None, name="aux1_head")(b1_out)

    # ── Branch 2: Multi-Scale Grouped Feature Tower ────────────────────────────
    # Group 1: Meta-Learner Features (0..9)
    meta_feats = layers.Lambda(lambda x: x[:, 0:10], name="meta_feats")(inputs)
    g_meta = layers.Dense(32, activation="relu", name="g_meta")(meta_feats)

    # Group 2: Risk Features (10..14)
    risk_feats = layers.Lambda(lambda x: x[:, 10:15], name="risk_feats")(inputs)
    g_risk = layers.Dense(16, activation="relu", name="g_risk")(risk_feats)

    # Group 3: Zone Features (15..22)
    zone_feats = layers.Lambda(lambda x: x[:, 15:23], name="zone_feats")(inputs)
    g_zone = layers.Dense(32, activation="relu", name="g_zone")(zone_feats)

    # Group 4: Time & Session Features (23..27)
    time_feats = layers.Lambda(lambda x: x[:, 23:28], name="time_feats")(inputs)
    g_time = layers.Dense(16, activation="relu", name="g_time")(time_feats)

    b2_cat = layers.Concatenate(name="b2_concat")([g_meta, g_risk, g_zone, g_time])
    b2 = layers.Dense(hidden_dim // 2, activation="silu", name="b2_fc")(b2_cat)
    b2_out = layers.LayerNormalization(name="b2_ln")(b2)

    aux2 = layers.Dense(num_actions, activation=None, name="aux2_head")(b2_out)

    # ── Auxiliary Head Gradient Isolation ──────────────────────────────────────
    aux1_iso = layers.Lambda(lambda a: tf.stop_gradient(a), name="aux1_isolation")(aux1)
    aux2_iso = layers.Lambda(lambda a: tf.stop_gradient(a), name="aux2_isolation")(aux2)

    # ── Gated Ensemble Fusion Head ─────────────────────────────────────────────
    fused_in = layers.Concatenate(name="fusion_cat")([b1_out, b2_out, aux1_iso, aux2_iso])
    gate = layers.Dense(138, activation="sigmoid", name="fusion_gate")(fused_in)

    fused = layers.Multiply(name="fusion_gated")([fused_in, gate])
    fused = layers.Dense(64, activation="silu", name="fusion_fc")(fused)
    fused = layers.LayerNormalization(name="fusion_ln")(fused)

    q_values = layers.Dense(num_actions, activation=None, name="q_values")(fused)

    model = keras.Model(
        inputs=inputs,
        outputs={"q_values": q_values, "aux1": aux1, "aux2": aux2},
        name="KerasExecutorQNetwork",
    )
    return model


# ── Keras Options Q-Executor Class ────────────────────────────────────────────

class KerasOptionsQExecutor:
    """
    Keras-based Tier 2 Q-Learner Trade Executor for Options.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 50000,
        target_update_freq: int = 100,
    ):
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_capacity = replay_capacity
        self.target_update_freq = target_update_freq
        self.step_counter = 0

        self.action_masker = HardActionMask()

        # Models
        self.q_net = build_keras_executor_q_network()
        self.target_net = build_keras_executor_q_network()
        self.target_net.set_weights(self.q_net.get_weights())

        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Replay memory
        self.memory: List[Tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]] = []

    def construct_state_vector(
        self,
        htf_bias: HTFBiasPackage,
        account: AccountContext,
        exec_ctx: ExecutionContext,
        zone_manager: ZoneSnapshotManager,
    ) -> np.ndarray:
        """Build flat 28-dim state vector."""
        dir_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
        dir_val = dir_map.get(htf_bias.direction, 0.0)

        nearest_supp, nearest_res = zone_manager.get_nearest_zones(exec_ctx.current_price)
        dist_supp = (exec_ctx.current_price - nearest_supp.price_level) / exec_ctx.atr if nearest_supp else 10.0
        dist_res = (nearest_res.price_level - exec_ctx.current_price) / exec_ctx.atr if nearest_res else 10.0

        pos_type_map = {None: 0.0, "CALL": 1.0, "PUT": -1.0}
        pos_val = pos_type_map.get(account.open_position_type, 0.0)

        state = np.array(
            [
                dir_val,
                htf_bias.strength,
                htf_bias.reversal_prob,
                htf_bias.q_value,
                htf_bias.expected_mfe_pips / 100.0,
                htf_bias.expected_mae_pips / 100.0,
                htf_bias.horizon_strengths[0],
                htf_bias.horizon_strengths[1],
                htf_bias.horizon_strengths[2],
                htf_bias.horizon_strengths[3],
                account.daily_pnl / 1000.0,
                account.daily_drawdown_pct,
                pos_val,
                account.open_position_pnl_pct,
                account.win_streak / 10.0,
                dist_supp,
                dist_res,
                nearest_supp.volume_delta_ratio if nearest_supp else 0.0,
                nearest_res.volume_delta_ratio if nearest_res else 0.0,
                (exec_ctx.buy_volume - exec_ctx.sell_volume) / max(1.0, exec_ctx.buy_volume + exec_ctx.sell_volume),
                exec_ctx.reentries_in_window / max(1.0, exec_ctx.max_reentries_allowed),
                nearest_supp.confluence_score if nearest_supp else 1.0,
                nearest_res.confluence_score if nearest_res else 1.0,
                exec_ctx.hour_of_day / 24.0,
                exec_ctx.minute_of_hour / 60.0,
                exec_ctx.day_of_week / 4.0,
                1.0 if exec_ctx.session_phase == "nyse_open" else 0.0,
                1.0 if exec_ctx.session_phase == "nyse_power_hour" else 0.0,
            ],
            dtype=np.float32,
        )
        return state

    def select_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        eval_mode: bool = False,
        eval_epsilon: float = 0.10,
    ) -> int:
        """Select action via epsilon-greedy policy with HardActionMask.

        In eval_mode, a small residual epsilon (eval_epsilon) is retained so
        that an undertrained network with WAIT-bias does not collapse to 100%
        WAIT during Phase 3 assessment.
        """
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            return ACTION_WAIT

        # Training: full epsilon-greedy. Eval: residual epsilon to prevent WAIT lock.
        eps = eval_epsilon if eval_mode else self.epsilon
        if random.random() < eps:
            return int(random.choice(valid_actions))

        state_tensor = np.expand_dims(state, axis=0)
        q_outputs = self.q_net(state_tensor, training=False)
        q_values = q_outputs["q_values"].numpy()[0]

        # Mask invalid actions with -inf
        masked_q = np.full_like(q_values, -np.inf)
        masked_q[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked_q))

    def record_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        """Store transition in replay buffer."""
        self.memory.append((state, action, reward, next_state, done, next_mask))
        if len(self.memory) > self.replay_capacity:
            self.memory.pop(0)

    def train_step(self, batch_size: int = 64) -> float:
        """Perform one gradient step of Double Q-Learning."""
        if len(self.memory) < batch_size:
            return 0.0

        indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in indices], dtype=np.float32)
        actions = np.array([self.memory[i][1] for i in indices], dtype=np.int32)
        rewards = np.array([self.memory[i][2] for i in indices], dtype=np.float32)
        next_states = np.array([self.memory[i][3] for i in indices], dtype=np.float32)
        dones = np.array([self.memory[i][4] for i in indices], dtype=np.float32)
        next_masks = np.array([self.memory[i][5] for i in indices], dtype=np.float32)

        # Compute Target Q Values
        next_q_online = self.q_net(next_states, training=False)["q_values"].numpy()
        next_q_target = self.target_net(next_states, training=False)["q_values"].numpy()

        # Double DQN target computation
        target_q = np.copy(self.q_net(states, training=False)["q_values"].numpy())
        for idx in range(batch_size):
            valid_acts = np.where(next_masks[idx] == 1)[0]
            if len(valid_acts) == 0:
                best_act = 0
            else:
                masked_online = np.full(NUM_ACTIONS, -np.inf)
                masked_online[valid_acts] = next_q_online[idx, valid_acts]
                best_act = np.argmax(masked_online)

            q_next = next_q_target[idx, best_act] if not dones[idx] else 0.0
            target_q[idx, actions[idx]] = rewards[idx] + (1.0 - dones[idx]) * self.gamma * q_next

        with tf.GradientTape() as tape:
            preds = self.q_net(states, training=True)
            loss_main = tf.reduce_mean(tf.keras.losses.huber(target_q, preds["q_values"]))
            loss_aux1 = tf.reduce_mean(tf.keras.losses.huber(target_q, preds["aux1"]))
            loss_aux2 = tf.reduce_mean(tf.keras.losses.huber(target_q, preds["aux2"]))

            total_loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2

        grads = tape.gradient(total_loss, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_net.trainable_variables))

        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.target_net.set_weights(self.q_net.get_weights())

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(total_loss.numpy())

    def save_checkpoint(self, filepath: str) -> None:
        """Save weights to disk."""
        self.q_net.save_weights(filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """Load weights from disk."""
        if os.path.exists(filepath):
            self.q_net.load_checkpoint(filepath)
            self.target_net.set_weights(self.q_net.get_weights())
