"""
keras_trade_executor.py
1:1 port of AXE Genesis (tt.py) classification ensemble for the Tier-2
Q-Learner trade executor.

Includes exact copies of:
  • build_cnn_feature_branch      — Branch A, 6-block causal CNN  (tt.py 3304)
  • build_dilated_cnn_feature_branch — Branch B, dilated causal CNN (tt.py 3354)
  • build_current_dense_branch    — Branch C, flat dense anchor     (tt.py 3418)
  • build_classification_ensemble_model — dual-branch + StopGrad aux (tt.py 3496)

The KerasTradeExecutor wraps the ensemble for Double-DQN training with
the same HardActionMask / state-vector API as the PyTorch OptionsQExecutor.
"""

import os
import random
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2 as keras_l2
from typing import Dict, List, Optional, Tuple

from app.core.market.zone_snapshot import HardActionMask, ZoneSnapshotManager
from app.core.options.q_executor import (
    ACTION_WAIT, ACTION_BUY_CALL, ACTION_BUY_PUT,
    ACTION_TAKE_PROFIT_HALF, ACTION_CLOSE_FLATTEN,
    NUM_ACTIONS, EXECUTOR_STATE_DIM,
    HTFBiasPackage, AccountContext, ExecutionContext,
)

# StopGradient must be importable for custom-objects reload
from app.core.ml.keras_signal_meta_learner import StopGradient

logger = logging.getLogger(__name__)


# ── Branch A: 6-block causal CNN (tt.py line 3304) ───────────────────────────

def build_cnn_feature_branch(inputs, kernel_size=6, dropout_rate=0.2,
                              name="cnn_branch"):
    """
    Branch A — local pattern detector.
    Filter schedule 128/128/64/64/64/64, kernel_size=6 (matched to
    research Cell 30 baseline: 0.783 test accuracy / 0.78 weighted F1).
    padding='causal' prevents look-ahead leakage.
    Single MaxPooling1D at end → Flatten → Dense(32) → BN → Dropout.
    Returns tensor of shape (batch, 32).
    """
    filter_schedule = [128, 128, 64, 64, 64, 64]
    x = inputs
    for i, f in enumerate(filter_schedule):
        x = layers.Conv1D(f, kernel_size=kernel_size, padding="causal",
                          activation="relu",
                          name=f"{name}_conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i+1}")(x)
        x = layers.LeakyReLU(name=f"{name}_lrelu{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"{name}_drop{i+1}")(x)

    x = layers.MaxPooling1D(pool_size=2, name=f"{name}_pool")(x)
    x = layers.Flatten(name=f"{name}_flatten")(x)
    x = layers.Dense(32, activation="relu", name=f"{name}_dense")(x)
    x = layers.BatchNormalization(name=f"{name}_dense_bn")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_dense_drop")(x)
    return x   # (batch, 32)


# ── Branch B: dilated causal CNN (tt.py line 3354) ───────────────────────────

def build_dilated_cnn_feature_branch(inputs, num_filters=128, kernel_size=4,
                                      dilation_rates=(4, 8, 16, 32, 64),
                                      dropout_rate=0.2, l2_lambda=0.01,
                                      name="dilated_branch"):
    """
    Branch B — multi-scale temporal feature extractor.
    dilation_rates=(4,8,16,32,64) — matched exactly to research Cell 34.
    kernel_size=4 — matches research Cell 33 training call.
    Residual skip: stem → after dilation stack (1×1 proj).
    Head Conv1D(filters*2, k=kernel_size) → GAP → Dense(128) → Dense(64).
    Returns tensor of shape (batch, 64).
    """
    # Stem
    x = layers.Conv1D(num_filters, kernel_size=kernel_size, padding="causal",
                      activation=None, name=f"{name}_stem")(inputs)
    x = layers.BatchNormalization(name=f"{name}_stem_bn")(x)
    x = layers.LeakyReLU(name=f"{name}_stem_lrelu")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_stem_drop")(x)
    residual = x

    # Dilated stack
    for i, rate in enumerate(dilation_rates):
        x = layers.Conv1D(num_filters, kernel_size=kernel_size,
                          padding="causal", dilation_rate=rate,
                          kernel_regularizer=keras_l2(l2_lambda),
                          activation=None,
                          name=f"{name}_dilconv{i}")(x)
        x = layers.BatchNormalization(name=f"{name}_dilbn{i}")(x)
        x = layers.LeakyReLU(name=f"{name}_dillrelu{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"{name}_dildrop{i}")(x)

    # Residual projection + add
    x = layers.Conv1D(num_filters, kernel_size=1, padding="causal",
                      activation=None, name=f"{name}_res_proj")(x)
    x = layers.BatchNormalization(name=f"{name}_res_bn")(x)
    x = layers.Add(name=f"{name}_add")([x, residual])

    # Head conv
    x = layers.Conv1D(num_filters * 2, kernel_size=kernel_size,
                      padding="causal", activation=None,
                      name=f"{name}_head_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_head_bn")(x)
    x = layers.LeakyReLU(name=f"{name}_head_lrelu")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_head_drop")(x)

    x = layers.GlobalAveragePooling1D(name=f"{name}_gap")(x)
    x = layers.Dense(128, activation="relu", name=f"{name}_dense1")(x)
    x = layers.Dense(64,  activation="relu", name=f"{name}_dense2")(x)
    return x   # (batch, 64)


# ── Branch C: current-bar dense anchor (tt.py line 3418) ─────────────────────

def build_current_dense_branch(inputs, embedding_dim=32, dropout_rate=0.2,
                                name="current_branch"):
    """
    Branch C — cross-sectional regime anchor.
    Flatten full window → Dense(embedding_dim) bottleneck → BN → Dropout
    → second Dense(embedding_dim) for non-linear recombination.
    Returns tensor of shape (batch, embedding_dim).
    """
    x = layers.Flatten(name=f"{name}_flatten")(inputs)
    x = layers.Dense(embedding_dim, activation="relu",
                     name=f"{name}_dense1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Dropout(dropout_rate, name=f"{name}_drop")(x)
    x = layers.Dense(embedding_dim, activation="relu",
                     name=f"{name}_dense2")(x)
    return x   # (batch, embedding_dim)


# ── Dual-branch classification ensemble (tt.py line 3496) ────────────────────

def build_classification_ensemble_model(input_shape,
                                         output_dim=5,
                                         task_specs=None,
                                         dropout_rate=0.3,
                                         binary_head=False):
    """
    1:1 port of tt.py `build_classification_ensemble_model`.

    Architecture:
        Branch A: build_cnn_feature_branch        → (batch, 32)
        Branch B: build_dilated_cnn_feature_branch → (batch, 64)
        Per-branch aux head with StopGradient before fusion.
        Fusion MLP: Dense(128)→BN→Drop → Dense(64)→BN→Drop → primary head.

    Single-head outputs:
        class_output  — primary softmax/sigmoid
        class_aux1    — Branch A independent prediction
        class_aux2    — Branch B independent prediction

    Loss weights: primary=1.0, aux1=0.5, aux2=0.5.
    """
    inp = keras.Input(shape=input_shape, name="ens_input")

    cnn_out     = build_cnn_feature_branch(
        inp, dropout_rate=dropout_rate,
        kernel_size=6, name="cnn_branch")
    dilated_out = build_dilated_cnn_feature_branch(
        inp, dropout_rate=dropout_rate,
        kernel_size=4,
        dilation_rates=(4, 8, 16, 32, 64),
        name="dilated_branch")

    # Activation / loss
    if binary_head or output_dim == 1:
        activation = "sigmoid"
        loss_str   = "binary_crossentropy"
        n_out      = 1
    else:
        activation = "softmax"
        loss_str   = "categorical_crossentropy"
        n_out      = output_dim

    if task_specs is not None:
        # Multi-task path
        aux1_outs, aux2_outs = {}, {}
        for task_name, task_dim in task_specs.items():
            task_dim = int(task_dim)
            task_act = "sigmoid" if task_dim == 1 else "softmax"
            aux1_outs[f"{task_name}_aux1"] = layers.Dense(
                task_dim, activation=task_act,
                name=f"{task_name}_aux1")(cnn_out)
            aux2_outs[f"{task_name}_aux2"] = layers.Dense(
                task_dim, activation=task_act,
                name=f"{task_name}_aux2")(dilated_out)
        fused = layers.Concatenate(name="ens_concat")([cnn_out, dilated_out])
    else:
        # Single-head path
        aux1_raw = layers.Dense(n_out, activation=activation,
                                kernel_initializer="glorot_uniform",
                                name="class_aux1")(cnn_out)
        aux2_raw = layers.Dense(n_out, activation=activation,
                                kernel_initializer="glorot_uniform",
                                name="class_aux2")(dilated_out)
        sg1   = StopGradient(name="ens_sg1")(aux1_raw)
        sg2   = StopGradient(name="ens_sg2")(aux2_raw)
        fused = layers.Concatenate(name="ens_concat")(
            [cnn_out, dilated_out, sg1, sg2])

    # Fusion MLP
    fused = layers.Dense(128, activation="relu",
                         name="ens_fusion_dense1")(fused)
    fused = layers.BatchNormalization(name="ens_fusion_bn1")(fused)
    fused = layers.Dropout(dropout_rate, name="ens_fusion_drop1")(fused)
    fused = layers.Dense(64, activation="relu",
                         name="ens_fusion_dense2")(fused)
    fused = layers.BatchNormalization(name="ens_fusion_bn2")(fused)
    fused = layers.Dropout(dropout_rate, name="ens_fusion_drop2")(fused)

    if task_specs is not None:
        primary_outs = {}
        for task_name, task_dim in task_specs.items():
            task_dim = int(task_dim)
            task_act = "sigmoid" if task_dim == 1 else "softmax"
            primary_outs[task_name] = layers.Dense(
                task_dim, activation=task_act, name=task_name)(fused)
        all_outputs = {**primary_outs, **aux1_outs, **aux2_outs}
        model = keras.Model(inputs=inp, outputs=all_outputs,
                            name="classification_ensemble")
        losses, loss_weights, metrics_dict = {}, {}, {}
        for task_name, task_dim in task_specs.items():
            task_dim = int(task_dim)
            t_loss = ("binary_crossentropy" if task_dim == 1
                      else "categorical_crossentropy")
            for out_name in [task_name, f"{task_name}_aux1",
                             f"{task_name}_aux2"]:
                losses[out_name]       = t_loss
                loss_weights[out_name] = 1.0 if out_name == task_name else 0.5
                metrics_dict[out_name] = ["accuracy"]
        model.compile(
            optimizer=keras.optimizers.RMSprop(
                learning_rate=0.001, clipvalue=1.0),
            loss=losses, loss_weights=loss_weights,
            metrics=metrics_dict)
        return model

    # Default single-head
    primary_out = layers.Dense(n_out, activation=activation,
                               name="class_output")(fused)
    model = keras.Model(
        inputs=inp,
        outputs={"class_output": primary_out,
                 "class_aux1":   aux1_raw,
                 "class_aux2":   aux2_raw},
        name="classification_ensemble")
    model.compile(
        optimizer=keras.optimizers.RMSprop(
            learning_rate=0.001, clipvalue=1.0),
        loss={"class_output": loss_str,
              "class_aux1":   loss_str,
              "class_aux2":   loss_str},
        loss_weights={"class_output": 1.0,
                      "class_aux1":   0.5,
                      "class_aux2":   0.5},
        metrics={"class_output": ["accuracy"],
                 "class_aux1":   ["accuracy"],
                 "class_aux2":   ["accuracy"]})
    return model


# ── Keras Trade Executor ──────────────────────────────────────────────────────

class KerasTradeExecutor:
    """
    Tier-2 Q-Learner trade executor using the dual-branch CNN+Dilated CNN
    classification ensemble from tt.py.

    Maintains API parity with OptionsQExecutor so the eval script can swap
    between frameworks with --framework keras / --framework pytorch.
    """

    def __init__(self,
                 seq_len: int = 48,
                 n_features: int = EXECUTOR_STATE_DIM,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.95,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.995,
                 replay_capacity: int = 50_000,
                 target_update_freq: int = 100):

        self.seq_len          = seq_len
        self.n_features       = n_features
        self.gamma            = gamma
        self.epsilon          = epsilon_start
        self.epsilon_min      = epsilon_min
        self.epsilon_decay    = epsilon_decay
        self.replay_capacity  = replay_capacity
        self.target_update_freq = target_update_freq
        self._step            = 0
        self._action_masker   = HardActionMask()

        # The CNN branches expect 3D inputs (batch, seq_len, features)
        input_shape = (seq_len, n_features)

        # Q-networks: dual-branch CNN ensemble
        self.q_net     = build_classification_ensemble_model(
            input_shape=input_shape,
            output_dim=NUM_ACTIONS,
            dropout_rate=0.2)
        self.target_net = build_classification_ensemble_model(
            input_shape=input_shape,
            output_dim=NUM_ACTIONS,
            dropout_rate=0.2)
        self.target_net.set_weights(self.q_net.get_weights())
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        # Replay buffer: (state, action, reward, next_state, done, next_mask)
        self._memory: List[Tuple] = []

    # ── state builder (matches OptionsQExecutor.build_state_vector) ──────────

    def build_state_vector(self,
                           htf_bias: HTFBiasPackage,
                           account: AccountContext,
                           exec_ctx: ExecutionContext,
                           zone_manager: ZoneSnapshotManager) -> np.ndarray:
        """Flat 28-dim state — same contract as OptionsQExecutor."""
        dir_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
        nearest_supp, nearest_res = zone_manager.get_nearest_zones(
            exec_ctx.current_price)
        dist_s = ((exec_ctx.current_price - nearest_supp.price_level)
                  / exec_ctx.atr if nearest_supp else 10.0)
        dist_r = ((nearest_res.price_level - exec_ctx.current_price)
                  / exec_ctx.atr if nearest_res else 10.0)
        pos_map = {None: 0.0, "CALL": 1.0, "PUT": -1.0}

        state = np.array([
            dir_map.get(htf_bias.direction, 0.0),
            htf_bias.strength, htf_bias.reversal_prob, htf_bias.q_value,
            htf_bias.expected_mfe_pips / 100.0,
            htf_bias.expected_mae_pips / 100.0,
            *htf_bias.horizon_strengths[:4],
            account.daily_pnl / 1000.0,
            account.daily_drawdown_pct,
            pos_map.get(account.open_position_type, 0.0),
            account.open_position_pnl_pct,
            account.win_streak / 10.0,
            dist_s, dist_r,
            nearest_supp.volume_delta_ratio if nearest_supp else 0.0,
            nearest_res.volume_delta_ratio  if nearest_res  else 0.0,
            (exec_ctx.buy_volume - exec_ctx.sell_volume)
            / max(1.0, exec_ctx.buy_volume + exec_ctx.sell_volume),
            exec_ctx.reentries_in_window
            / max(1.0, exec_ctx.max_reentries_allowed),
            nearest_supp.confluence_score if nearest_supp else 1.0,
            nearest_res.confluence_score  if nearest_res  else 1.0,
            exec_ctx.hour_of_day / 24.0,
            exec_ctx.minute_of_hour / 60.0,
            exec_ctx.day_of_week / 4.0,
            1.0 if exec_ctx.session_phase == "nyse_open" else 0.0,
            1.0 if exec_ctx.session_phase == "nyse_power_hour" else 0.0,
        ], dtype=np.float32)
        return state   # shape (28,)

    def _state_to_model_input(self, state: np.ndarray) -> np.ndarray:
        """Wrap flat state into the expected 3D input shape (1, seq_len, features)."""
        tiled = np.tile(state, (self.seq_len, 1))   # (T, F)
        return np.expand_dims(tiled, 0)             # (1, T, F)

    # ── action selection ─────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray,
                      mask: np.ndarray,
                      eval_mode: bool = False) -> int:
        valid = np.where(mask == 1)[0]
        if len(valid) == 0:
            return ACTION_WAIT

        if not eval_mode and random.random() < self.epsilon:
            return int(random.choice(valid))

        inp  = self._state_to_model_input(state)
        outs = self.q_net(inp, training=False)
        q    = outs["class_output"].numpy()[0]

        masked = np.full_like(q, -np.inf)
        masked[valid] = q[valid]
        return int(np.argmax(masked))

    def calculate_executor_reward(
        self,
        action: int,
        action_mask: np.ndarray,
        pnl_pct: float,
        max_drawdown_exposed: float,
        forward_move_pct: float,
        htf_bias,
        is_closed: bool = False,
        is_stop_loss: bool = False,
        risk_limit_breached: bool = False,
    ) -> float:
        """
        1:1 port of OptionsQExecutor.calculate_executor_reward adhering to options-repurposing-directives.md.

        Components:
          1. Overtrade churn penalty (-0.05) on BUY_CALL / BUY_PUT entry.
          2. Best Price Entry Bonus (+0.15) for high-conviction entries with low drawdown.
          3. Realized P&L / max_drawdown on close (+stop-loss bonus).
          4. Wise Patience Reward vs Hindsight Missed-Opportunity Penalty on ACTION_WAIT.
          5. Low-conviction discipline reward (+0.02) when standing down.
        """
        reward = 0.0

        # 1. Entry & Quality Sizing
        if action in (ACTION_BUY_CALL, ACTION_BUY_PUT):
            reward -= 0.05  # Churn penalty for option spread

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
                reward += 0.20

        # 3. Wise Patience Reward vs Hindsight Missed-Opportunity Penalty (ACTION_WAIT)
        if action == ACTION_WAIT and not is_closed and not risk_limit_breached:
            strength = getattr(htf_bias, "strength", 0.5)

            # Bullish Setup Evaluation
            if action_mask[ACTION_BUY_CALL] == 1 and strength >= 0.65:
                if forward_move_pct >= 0.003:
                    penalty = float(np.clip(-0.15 * (forward_move_pct / 0.003), -0.5, 0.0))
                    reward += penalty
                elif forward_move_pct < 0.0:
                    patience_bonus = float(np.clip(0.15 * (abs(forward_move_pct) / 0.003), 0.0, 0.30))
                    reward += patience_bonus

            # Bearish Setup Evaluation
            elif action_mask[ACTION_BUY_PUT] == 1 and strength >= 0.65:
                if forward_move_pct <= -0.003:
                    penalty = float(np.clip(-0.15 * (abs(forward_move_pct) / 0.003), -0.5, 0.0))
                    reward += penalty
                elif forward_move_pct > 0.0:
                    patience_bonus = float(np.clip(0.15 * (forward_move_pct / 0.003), 0.0, 0.30))
                    reward += patience_bonus

            else:
                # Discipline bonus for staying flat when no high-conviction setup exists
                reward += 0.02

        return float(np.clip(reward, -3.0, 3.0))

    # ── replay ───────────────────────────────────────────────────────────────

    def record_transition(self, state, action, reward,
                          next_state, done, next_mask):
        self._memory.append((state, action, reward, next_state, done, next_mask))
        if len(self._memory) > self.replay_capacity:
            self._memory.pop(0)

    # ── training step (Double DQN) ────────────────────────────────────────────

    def train_step(self, batch_size: int = 64) -> float:
        if len(self._memory) < batch_size:
            return 0.0

        idx        = np.random.choice(len(self._memory), size=batch_size,
                                      replace=False)
        states     = np.stack([self._memory[i][0] for i in idx]).astype(np.float32)
        actions    = np.array([self._memory[i][1] for i in idx], dtype=np.int32)
        rewards    = np.array([self._memory[i][2] for i in idx], dtype=np.float32)
        nxt_states = np.stack([self._memory[i][3] for i in idx]).astype(np.float32)
        dones      = np.array([self._memory[i][4] for i in idx], dtype=np.float32)
        nxt_masks  = np.stack([self._memory[i][5] for i in idx]).astype(np.float32)

        s_inp  = self._batch_to_model_input(states)
        ns_inp = self._batch_to_model_input(nxt_states)

        # Double DQN target
        q_online  = self.q_net(ns_inp, training=False)["class_output"].numpy()
        q_target  = self.target_net(ns_inp, training=False)["class_output"].numpy()
        q_current = self.q_net(s_inp, training=False)["class_output"].numpy()
        target_q  = q_current.copy()

        for j in range(batch_size):
            valid = np.where(nxt_masks[j] == 1)[0]
            if len(valid) == 0:
                best_a = 0
            else:
                masked = np.full(NUM_ACTIONS, -np.inf)
                masked[valid] = q_online[j, valid]
                best_a = np.argmax(masked)
            q_next = q_target[j, best_a] if not dones[j] else 0.0
            target_q[j, actions[j]] = (rewards[j]
                                       + (1.0 - dones[j]) * self.gamma * q_next)

        with tf.GradientTape() as tape:
            preds  = self.q_net(s_inp, training=True)
            loss_m = tf.reduce_mean(
                tf.keras.losses.huber(target_q, preds["class_output"]))
            loss_a1 = tf.reduce_mean(
                tf.keras.losses.huber(target_q, preds["class_aux1"]))
            loss_a2 = tf.reduce_mean(
                tf.keras.losses.huber(target_q, preds["class_aux2"]))
            total = loss_m + 0.5 * loss_a1 + 0.5 * loss_a2

        grads = tape.gradient(total, self.q_net.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.q_net.trainable_variables))

        self._step += 1
        if self._step % self.target_update_freq == 0:
            self.target_net.set_weights(self.q_net.get_weights())
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(total.numpy())

    def _batch_to_model_input(self, states: np.ndarray) -> np.ndarray:
        # states shape: (B, F) → tile to (B, T, F)
        return np.tile(states[:, None, :], (1, self.seq_len, 1))

    # ── persistence ──────────────────────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        self.q_net.save_weights(path)

    def load_checkpoint(self, path: str) -> None:
        if os.path.exists(path):
            self.q_net.load_weights(path)
            self.target_net.set_weights(self.q_net.get_weights())


# Backward-compat alias used by evaluate_option_expiries.py
KerasOptionsQExecutor = KerasTradeExecutor
