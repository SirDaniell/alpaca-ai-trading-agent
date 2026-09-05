"""
agent.py — Axe-paka-v1 Trading Agent Implementation.

Consolidates Meta-Learner (Tier 1) and Q-Learner (Tier 2) model inference,
strict 30-min & 1-hr option expiry trade gating, OCC option contract selection,
and Alpaca API/CLI execution pipeline wiring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from app.axe_paka_v1.config import AxePakaV1Config
from app.axe_paka_v1.models import (
    DEFAULT_NUM_FEATURES,
    DEFAULT_Q_LOOKBACK,
    DEFAULT_STATE_DIM,
    ExecutorQNetwork,
    SignalMetaNetwork,
    build_feat_window,
)
from app.core.market.zone_snapshot import HardActionMask, ZoneSnapshotManager
from app.core.options.alpaca_execution_engine import AlpacaExecutionEngine
from app.core.options.options_order import select_target_option_contract
from app.core.options.q_executor import AccountContext, ExecutionContext, HTFBiasPackage

from app.axe_paka_v1.money_management import MartingaleMoneyManager, MartingaleMMConfig

logger = logging.getLogger(__name__)

# Action Constants
ACTION_WAIT = 0
ACTION_BUY_CALL = 1
ACTION_BUY_PUT = 2
ACTION_TAKE_PROFIT_HALF = 3
ACTION_CLOSE_FLATTEN = 4


class AxePakaV1Agent:
    """
    Axe-paka-v1 Trading Agent.
    
    Integrates Tier 1 Meta-Learner (SignalMetaNetwork) and Tier 2 Q-Learner (ExecutorQNetwork).
    Configured with 4-Step Capped Martingale Position Sizing ($10 base, max 4 steps).
    Permits option trades across all 4 expiries (5m, 15m, 30m, 1h) for testing.
    """

    def __init__(
        self,
        config: Optional[AxePakaV1Config] = None,
        device: str = "cpu",
        use_cli: bool = True,
        auto_load_weights: bool = True,
    ):
        self.config = config or AxePakaV1Config()
        self.device = torch.device(device)
        self.use_cli = use_cli

        mm_cfg = MartingaleMMConfig(
            base_trade_dollars=self.config.base_trade_dollars,
            martingale_multiplier=self.config.martingale_multiplier,
            max_martingale_steps=self.config.max_martingale_steps,
            max_position_dollars=self.config.max_position_dollars,
        )
        self.money_manager = MartingaleMoneyManager(mm_cfg)

        # Model Architectures from notebook checkpoint (num_features=333)
        self.meta_net = SignalMetaNetwork(
            num_features=DEFAULT_NUM_FEATURES,
            hidden_dim=256,
        ).to(self.device)

        self.q_net = ExecutorQNetwork(
            num_features=DEFAULT_NUM_FEATURES,
            input_dim=DEFAULT_STATE_DIM,
            hidden_dim=128,
            num_horizons=self.config.num_horizons,
            num_head_actions=3,
        ).to(self.device)

        self.action_mask_engine = HardActionMask()
        self.execution_engine = AlpacaExecutionEngine(use_cli=self.use_cli)

        self.weights_loaded = False
        if auto_load_weights:
            self.load_checkpoints()

    def load_checkpoints(self) -> bool:
        """
        Load trained model weight checkpoints for Meta-Learner and Q-Executor.
        Searches `config.weights_dir` first, then falls back to `config.fallback_weights_dir`.
        """
        meta_loaded = self._load_single_model("Meta-Learner", self.meta_net, self.config.meta_weights_name, self.config.meta_weights_last_name)
        q_loaded = self._load_single_model("Q-Executor", self.q_net, self.config.q_weights_name, self.config.q_weights_last_name)

        self.meta_net.eval()
        self.q_net.eval()
        self.weights_loaded = meta_loaded and q_loaded
        return self.weights_loaded

    def _load_single_model(
        self,
        name: str,
        model: torch.nn.Module,
        best_filename: str,
        last_filename: str,
    ) -> bool:
        search_paths = [
            self.config.weights_dir / best_filename,
            self.config.weights_dir / last_filename,
            self.config.fallback_weights_dir / best_filename,
            self.config.fallback_weights_dir / last_filename,
        ]

        for p in search_paths:
            if p.exists() and p.is_file():
                try:
                    checkpoint = torch.load(p, map_location=self.device)
                    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
                    model.load_state_dict(state_dict)
                    logger.info("✅ Successfully loaded %s weights from: %s", name, p)
                    return True
                except Exception as e:
                    logger.warning("⚠ Failed loading %s from %s: %s", name, p, e)

        logger.warning("⚠ Could not find weight checkpoint for %s in %s or %s", name, self.config.weights_dir, self.config.fallback_weights_dir)
        return False

    def build_state_vector(
        self,
        htf_bias: HTFBiasPackage,
        account: AccountContext,
        exec_ctx: ExecutionContext,
        zone_manager: ZoneSnapshotManager,
    ) -> np.ndarray:
        """
        Construct normalized 28-dim state vector for Tier 2 Q-Learner.
        """
        nearest_supp, nearest_res = zone_manager.get_nearest_zones(exec_ctx.current_price)

        supp_dist = abs(exec_ctx.current_price - nearest_supp.price_level) / exec_ctx.current_price if nearest_supp else 1.0
        res_dist  = abs(exec_ctx.current_price - nearest_res.price_level) / exec_ctx.current_price if nearest_res else 1.0

        supp_vol_ratio = nearest_supp.volume_delta_ratio if nearest_supp else 0.0
        res_vol_ratio  = nearest_res.volume_delta_ratio if nearest_res else 0.0

        total_vol = exec_ctx.buy_volume + exec_ctx.sell_volume
        vol_delta_ratio = (exec_ctx.buy_volume - exec_ctx.sell_volume) / (total_vol + 1e-6)

        tf_flag  = 1.0 if exec_ctx.ltf_timeframe == "15m" else 0.0
        dir_flag = 1.0 if htf_bias.direction == "bullish" else (-1.0 if htf_bias.direction == "bearish" else 0.0)

        hs = htf_bias.horizon_strengths if len(htf_bias.horizon_strengths) == 4 else [0.5, 0.5, 0.5, 0.5]

        sin_hour = float(np.sin(2 * np.pi * exec_ctx.hour_of_day / 24.0))
        cos_hour = float(np.cos(2 * np.pi * exec_ctx.hour_of_day / 24.0))
        dow_norm = float(exec_ctx.day_of_week) / 6.0
        is_nyse_open  = 1.0 if exec_ctx.session_phase == "nyse_open" else 0.0
        is_power_hour = 1.0 if exec_ctx.session_phase == "nyse_power_hour" else 0.0

        return np.array([
            # Meta & Multi-Horizon Hints (10)
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
            # Account Context (5)
            float(account.daily_drawdown_pct),
            1.0 if account.open_position_type == "CALL" else (-1.0 if account.open_position_type == "PUT" else 0.0),
            float(account.open_position_pnl_pct),
            float(account.win_streak) / 10.0,
            float(account.loss_streak) / 10.0,
            # Execution & Zone Context (8)
            tf_flag,
            float(exec_ctx.atr) / exec_ctx.current_price,
            float(supp_dist),
            float(res_dist),
            float(supp_vol_ratio),
            float(res_vol_ratio),
            float(vol_delta_ratio),
            float(exec_ctx.reentries_in_window) / float(exec_ctx.max_reentries_allowed),
            # Time & Session Context (5)
            sin_hour,
            cos_hour,
            dow_norm,
            is_nyse_open,
            is_power_hour,
        ], dtype=np.float32)

    def evaluate_signal(
        self,
        feat_window: np.ndarray,
        ctx_state: np.ndarray,
        action_mask: np.ndarray,
        htf_bias: Optional[HTFBiasPackage] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate Tier 1 Meta & Tier 2 Q-Learner signals.
        STRICT EXPIRY GATING: Disallows trade signals for 5m (head 0) and 15m (head 1).
        Only permits trades on 30m (head 2) and 1h (head 3).
        """
        if feat_window.ndim == 2:
            feat_window = np.expand_dims(feat_window, axis=0)
        if ctx_state.ndim == 1:
            ctx_state = np.expand_dims(ctx_state, axis=0)

        t_feat = torch.tensor(feat_window, dtype=torch.float32, device=self.device)
        t_ctx  = torch.tensor(ctx_state,  dtype=torch.float32, device=self.device)

        with torch.no_grad():
            # Tier 1 Meta-Learner Output
            q_vals, strength_vec, pips, risk, liquidity, reversal = self.meta_net(t_feat)
            meta_strengths = strength_vec.squeeze(0).cpu().numpy()  # (4,) strengths for 5m, 15m, 30m, 1h
            meta_q_val = float(q_vals.squeeze(0).max().item())

            # Tier 2 Q-Learner Output across all 4 horizon heads
            all_q_stacked = self.q_net(t_feat, t_ctx).squeeze(0).cpu().numpy()  # (4, 3) -> [WAIT, CALL, PUT]

        # Apply Hard Action Mask (mask_3: [WAIT, CALL, PUT])
        mask_3 = action_mask[:3] if len(action_mask) >= 3 else action_mask

        horizon_evaluations: List[Dict[str, Any]] = []
        best_decision = {
            "selected_action": ACTION_WAIT,
            "action_name": "WAIT",
            "horizon_idx": None,
            "horizon_label": None,
            "q_value": 0.0,
            "q_margin": 0.0,
            "meta_strength": 0.0,
            "permitted": False,
        }

        best_q_margin = -1e9

        for h_idx in range(self.config.num_horizons):
            label = self.config.horizon_labels[h_idx]
            is_permitted = h_idx in self.config.permitted_horizon_indices
            q_head = all_q_stacked[h_idx]  # (3,)
            m_strength = float(meta_strengths[h_idx])

            # Apply mask to valid Q-values
            masked_q = np.where(mask_3 == 1, q_head, -1e9)

            if not is_permitted:
                # STRICT EXPIRY GATES: Force WAIT on disallowed horizons (5m, 15m)
                action = ACTION_WAIT
                action_name = "WAIT"
                q_val = float(q_head[ACTION_WAIT])
                q_margin = 0.0
            else:
                action = int(np.argmax(masked_q))
                action_name = "CALL" if action == ACTION_BUY_CALL else ("PUT" if action == ACTION_BUY_PUT else "WAIT")
                q_val = float(masked_q[action])
                q_wait = float(masked_q[ACTION_WAIT])
                q_margin = q_val - q_wait

            eval_entry = {
                "horizon_idx": h_idx,
                "horizon_label": label,
                "permitted": is_permitted,
                "raw_q_values": q_head.tolist(),
                "masked_q_values": masked_q.tolist(),
                "proposed_action": action,
                "action_name": action_name,
                "q_margin": q_margin,
                "meta_strength": m_strength,
            }
            horizon_evaluations.append(eval_entry)

            # Check if this permitted horizon produces a valid trade setup
            if is_permitted and action in (ACTION_BUY_CALL, ACTION_BUY_PUT):
                if q_margin > self.config.horizon_margin and m_strength >= self.config.confidence_threshold:
                    if q_margin > best_q_margin:
                        best_q_margin = q_margin
                        best_decision = {
                            "selected_action": action,
                            "action_name": action_name,
                            "horizon_idx": h_idx,
                            "horizon_label": label,
                            "q_value": q_val,
                            "q_margin": q_margin,
                            "meta_strength": m_strength,
                            "permitted": True,
                        }

        return {
            "agent_name": self.config.agent_name,
            "decision": best_decision,
            "horizon_evaluations": horizon_evaluations,
            "meta_q_value": meta_q_val,
            "permitted_expiries": list(self.config.permitted_horizon_labels),
            "weights_loaded": self.weights_loaded,
        }

    def execute_signal(
        self,
        symbol: str,
        signal_verdict: Dict[str, Any],
        current_price: float,
        dry_run: bool = False,
        override_qty: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute option trade via Alpaca API/CLI for approved signals using Martingale Money Management.
        """
        decision = signal_verdict.get("decision", {})
        action = decision.get("selected_action", ACTION_WAIT)
        horizon_idx = decision.get("horizon_idx")

        if action not in (ACTION_BUY_CALL, ACTION_BUY_PUT) or horizon_idx not in self.config.permitted_horizon_indices:
            logger.info("ℹ [AxePakaV1] No permitted trade signal to execute (Decision: %s)", decision.get("action_name", "WAIT"))
            return {
                "status": "skipped",
                "reason": "ACTION_WAIT_OR_UNPERMITTED_EXPIRY",
                "symbol": symbol,
                "decision": decision,
            }

        option_type = "CALL" if action == ACTION_BUY_CALL else "PUT"
        expiry_label = decision.get("horizon_label", "30m")
        holding_seconds = self.config.holding_seconds_map.get(horizon_idx, 1800)

        # Step 1: Select ATM Option Contract OCC Symbol
        contract_info = select_target_option_contract(
            underlying=symbol,
            current_price=current_price,
            option_type=option_type,
            days_to_expiration=7,
        )
        occ_symbol = contract_info["occ_symbol"]

        # Step 2: 4-Step Martingale Position Sizing — contract-count mode
        # Options are NOT fractional: minimum is 1 contract.
        # Use real contract close_price from Alpaca API; fall back to proxy if missing.
        api_close = contract_info.get("close_price")
        if api_close and float(api_close) > 0:
            est_contract_price = float(api_close)
        else:
            # Fallback proxy: ATM options are roughly 1-3% of underlying
            est_contract_price = max(0.50, current_price * 0.015)

        cost_per_contract = est_contract_price * 100.0  # 1 contract = 100 shares

        # Martingale steps → contract quantities (1, 2, 4, 8)
        # Base is always qty=1; step multiplier doubles contracts, not dollars
        mm_step = self.money_manager.get_step(symbol, expiry_label)
        base_contracts = 1
        calc_qty = max(1, base_contracts * (2 ** mm_step))
        target_dollars = calc_qty * cost_per_contract

        if cost_per_contract > self.config.base_trade_dollars * 10:
            logger.warning(
                "⚠ [MartingaleMM] Contract cost $%.2f/contract >> base budget $%.2f — using qty=1 minimum. "
                "Consider shorter expiry or OTM contracts for tighter Martingale sizing.",
                cost_per_contract, self.config.base_trade_dollars
            )
            calc_qty = 1  # Always at least 1 contract
            target_dollars = cost_per_contract

        trade_qty = override_qty if override_qty is not None else calc_qty

        logger.info(
            "🚀 [AxePakaV1] Executing %s option trade for %s (%s expiry, hold_timer=%ds, Martingale Step %d: $%.2f, qty=%d) -> OCC: %s",
            option_type, symbol, expiry_label, holding_seconds, mm_step + 1, target_dollars, trade_qty, occ_symbol
        )

        # Martingale callback: fires in background thread when position closes
        _sym = symbol
        _tf = expiry_label
        _mm = self.money_manager
        def _martingale_on_close(trade_result: dict) -> None:
            pnl = trade_result.get("realized_pnl", 0.0)
            is_win = pnl > 0
            _mm.record_trade_result(_sym, _tf, is_win=is_win)
            logger.info(
                "📊 [MartingaleCallback] %s_%s | PnL=$%.2f -> %s | Next step: %d",
                _sym, _tf, pnl, "WIN" if is_win else "LOSS",
                _mm.get_step(_sym, _tf) + 1,
            )

        # Step 3: Entry is SYNCHRONOUS (fills confirmed), monitor runs in background thread
        execution_report = self.execution_engine.execute_trade_with_expiry(
            symbol=occ_symbol,
            qty=trade_qty,
            side="buy",
            asset_type="option",
            expiry_seconds=holding_seconds,
            poll_interval=10.0,
            dry_run=dry_run,
            on_close_callback=_martingale_on_close,
        )

        execution_report["contract_details"] = contract_info
        execution_report["expiry_horizon"] = expiry_label
        execution_report["agent_name"] = self.config.agent_name
        execution_report["martingale_mm"] = {
            "key": f"{symbol.upper()}_{expiry_label}",
            "step": mm_step + 1,
            "target_dollars": target_dollars,
            "qty": trade_qty,
            "consecutive_losses": self.money_manager.consecutive_losses.get(
                self.money_manager._key(symbol, expiry_label), 0
            ),
        }

        return execution_report
