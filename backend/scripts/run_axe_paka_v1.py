#!/usr/bin/env python3
"""
run_axe_paka_v1.py — Run & Benchmark Axe-paka-v1 Agent on Live/Simulated Data.

Evaluates signals using Tier 1 Meta + Tier 2 Q-Learner weights,
enforces 30m and 1h options expiry gating, and tests Alpaca order formatting/dry-run execution.
"""

from pathlib import Path
import sys
import logging
import numpy as np

# Ensure backend root directory is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.axe_paka_v1.agent import AxePakaV1Agent, ACTION_BUY_CALL
from app.axe_paka_v1.config import AxePakaV1Config
from app.core.options.q_executor import HTFBiasPackage, AccountContext, ExecutionContext
from app.core.market.zone_snapshot import ZoneSnapshotManager, HardActionMask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("RunAxePakaV1")


def main():
    print("==========================================================================")
    print("          AXE-PAKA-V1 TRADING AGENT INFERENCE & EXECUTION DEMO")
    print("==========================================================================")

    config = AxePakaV1Config()
    agent = AxePakaV1Agent(config=config, use_cli=True, auto_load_weights=True)

    print(f"Agent Name: {agent.config.agent_name}")
    print(f"Weights Loaded: {agent.weights_loaded}")
    print(f"Permitted Expiries: {agent.config.permitted_horizon_labels}")

    # Build dummy market state for evaluation demo
    htf_bias = HTFBiasPackage(
        direction="bullish",
        strength=0.75,
        reversal_prob=0.15,
        q_value=1.2,
        horizon_strengths=[0.4, 0.5, 0.78, 0.85],
    )
    account = AccountContext(equity=100000.0, daily_pnl=250.0)
    exec_ctx = ExecutionContext(symbol="GLD", current_price=200.0, atr=1.2)
    zone_mgr = ZoneSnapshotManager()

    # Build 28-dim state vector and 150x333 feature window
    state_vector = agent.build_state_vector(htf_bias, account, exec_ctx, zone_mgr)
    feat_window = np.zeros((150, 333), dtype=np.float32)
    action_mask = np.array([1, 1, 1, 0, 0], dtype=np.int32)  # [WAIT, CALL, PUT] unmasked

    # Evaluate Signal
    signal_verdict = agent.evaluate_signal(feat_window, state_vector, action_mask, htf_bias)

    print("\n📊 Signal Evaluation Verdict:")
    print(f"  Decision: {signal_verdict['decision']}")
    print(f"  Meta Q-Value: {signal_verdict['meta_q_value']:.4f}")
    print("  Horizon Evaluations:")
    for h_eval in signal_verdict["horizon_evaluations"]:
        status = "PERMITTED" if h_eval["permitted"] else "DISALLOWED (BLOCKED)"
        print(f"    - [{h_eval['horizon_label']}] Status: {status:<20} | Proposed Action: {h_eval['action_name']:<5} | Q-Margin: {h_eval['q_margin']:.4f}")

    # Execute simulated dry-run trade if permitted signal was generated
    decision = signal_verdict["decision"]
    if decision.get("selected_action") != 0:
        print("\n🚀 Executing Dry-Run Option Trade...")
        exec_report = agent.execute_signal("GLD", signal_verdict, current_price=200.0, dry_run=True)
        print(f"  Status: {exec_report.get('status')}")
        print(f"  OCC Symbol: {exec_report.get('symbol')}")
        print(f"  Expiry Timer: {exec_report.get('hold_duration_sec')}s")
        print(f"  Exit Reason: {exec_report.get('exit_reason')}")
    else:
        print("\nℹ Signal evaluated to WAIT. Simulating forced CALL trade execution on 30m expiry to test Alpaca order pipeline...")
        forced_verdict = {
            "decision": {
                "selected_action": ACTION_BUY_CALL,
                "action_name": "CALL",
                "horizon_idx": 2,
                "horizon_label": "30m",
            }
        }
        exec_report = agent.execute_signal("GLD", forced_verdict, current_price=200.0, dry_run=True)
        print(f"  Status: {exec_report.get('status')}")
        print(f"  OCC Symbol: {exec_report.get('symbol')}")
        print(f"  Holding Window: {exec_report.get('hold_duration_sec')}s")
        print(f"  Exit Reason: {exec_report.get('exit_reason')}")

    print("\n✅ Axe-paka-v1 agent run completed successfully!")


if __name__ == "__main__":
    main()
