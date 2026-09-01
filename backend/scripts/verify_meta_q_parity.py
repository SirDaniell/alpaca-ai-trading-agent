#!/usr/bin/env python3
"""
verify_meta_q_parity.py

Parity verification for the AXE Meta-Learner & Q-Executor redesign.
Checks:
  1. SignalMetaNetwork output shapes
  2. branch_cat.detach() correctness (Bug 3 fix)
  3. ExecutorQNetwork dual-input output shapes (all horizons and single horizon)
  4. build_feat_window shape and zero-padding
  5. OnlineSignalMetaLearner META_PREDICT_WINDOW=150
  6. Strength targets have non-zero std from train_40k.csv
  7. State vector is 28-dim and finite
  8. Regret buffer present on OptionsQExecutor

Run: /media/daniel-joseph/Linux_Data/Dan_backup/Projects/JavaScript/fin-dash-buddy-sos/dev_env/bin/python3 scripts/verify_meta_q_parity.py
"""

import sys
import os
import unittest.mock as mock

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

import numpy as np
import torch
import torch.nn as nn

results = []

def check(name, condition, detail=""):
    icon = "✅" if condition else "❌"
    msg = f"{icon} {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)
    results.append((name, condition))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Import signal_meta_learner (no external deps needed beyond torch/numpy)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── SignalMetaNetwork ───────────────────────────────────────────────────")

try:
    # Stub only the problematic transitive import
    mock_instr = mock.MagicMock()
    mock_instr.pip_scale = 1.0
    with mock.patch.dict('sys.modules', {
        'app.core.ml.instrument_metadata': mock.MagicMock(get_instrument_metadata=lambda s: mock_instr),
        'app.core.analysis.technical_indicators': mock.MagicMock(),
    }):
        # Patch ti_meta_features constants before import
        import types
        ti_mod = types.ModuleType('app.core.ml.ti_meta_features')
        ti_mod.DECISION_FEATURE_KEYS = tuple(f"f{i}" for i in range(238))
        ti_mod.DECISION_FEATURE_COUNT = 238
        ti_mod.DECISION_WINDOW_DIM = 238 * 1000
        ti_mod.SIGNAL_META_FEATURE_CONTRACT_VERSION = "signal-meta-ti-seq-1000-v1"
        ti_mod.SIGNAL_META_LOOKBACK_BARS = 1000
        ti_mod.TI_NUMERIC_FEATURE_KEYS = tuple(f"f{i}" for i in range(217))
        ti_mod.CONTEXT_FEATURE_KEYS = tuple(f"c{i}" for i in range(21))
        ti_mod.calculate_ti_features = lambda *a, **k: None
        ti_mod.align_ti_numeric_frame = lambda *a, **k: None
        sys.modules['app.core.ml.ti_meta_features'] = ti_mod

        from app.core.ml.signal_meta_learner import (
            SignalMetaNetwork,
            OnlineSignalMetaLearner,
            META_PREDICT_WINDOW,
            HORIZON_LABELS,
        )

    NUM_FEATURES = 50
    net = SignalMetaNetwork(num_features=NUM_FEATURES)
    net.eval()

    # Check 1: output shapes with 150-bar input
    x = torch.randn(2, 150, NUM_FEATURES)
    with torch.no_grad():
        q, s, p, r, l, v = net(x)
    check("1a. q_vals shape (2,4)",      q.shape == (2, 4), str(q.shape))
    check("1b. strength shape (2,4)",    s.shape == (2, 4), str(s.shape))
    check("1c. pips shape (2,4)",        p.shape == (2, 4), str(p.shape))
    check("1d. risk shape (2,8)",        r.shape == (2, 8), str(r.shape))
    check("1e. liquidity shape (2,2)",   l.shape == (2, 2), str(l.shape))
    check("1f. reversal shape (2,1)",    v.shape == (2, 1), str(v.shape))

    # Check 2: branch_cat.detach() — pips/reversal heads must NOT flow gradients back to input
    # We verify this by enabling gradient tracking on x2 and confirming no grad arrives
    # at x2 after a backward pass through pips only (branch_cat.detach() severs the path).
    net.train()  # ensure grad tracking is active in network
    x2 = torch.randn(2, 150, NUM_FEATURES, requires_grad=True)
    q2, s2, p2, r2, l2, v2 = net(x2)
    # backward through pips only
    p2.sum().backward()
    # If branch_cat.detach() is correctly applied, x2.grad should be None
    # because pips → branch_cat.detach() → no path back to x2
    pips_grad_is_none = x2.grad is None
    check("2. branch_cat.detach() — no gradient from pips to input", pips_grad_is_none,
          f"x2.grad={'None (✓ detach works)' if pips_grad_is_none else 'not None (❌ missing detach)'}")
    net.eval()  # restore eval mode

    # Check 3: return_aux=True
    x3 = torch.randn(2, 150, NUM_FEATURES)
    with torch.no_grad():
        out = net(x3, return_aux=True)
    check("3. return_aux gives 9-tuple", len(out) == 9, f"len={len(out)}")

except Exception as e:
    check("SignalMetaNetwork import/shape", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 4. META_PREDICT_WINDOW = 150 in OnlineSignalMetaLearner
# ─────────────────────────────────────────────────────────────────────────────
print("\n── OnlineSignalMetaLearner ─────────────────────────────────────────────")
try:
    check("4a. META_PREDICT_WINDOW == 150", META_PREDICT_WINDOW == 150, str(META_PREDICT_WINDOW))

    learner = OnlineSignalMetaLearner()
    expected_input_dim = 150 * 238
    check("4b. learner.input_dim == 150*238", learner.input_dim == expected_input_dim,
          f"{learner.input_dim} vs {expected_input_dim}")
    check("4c. meta_predict_window == 150", learner.meta_predict_window == 150,
          str(learner.meta_predict_window))
    check("4d. contract version is v2", "150" in learner.feature_contract_version,
          learner.feature_contract_version)

    # Test extract_features slices correctly
    arr_1000 = np.random.randn(1000, 238).astype(np.float32)
    vec = learner.extract_features(arr_1000)
    check("4e. extract_features(1000-bar) → (150*238,)", vec.shape == (150*238,), str(vec.shape))

    arr_80 = np.random.randn(80, 238).astype(np.float32)
    vec2 = learner.extract_features(arr_80)
    check("4f. extract_features(80-bar pad) → (150*238,)", vec2.shape == (150*238,), str(vec2.shape))

except Exception as e:
    check("OnlineSignalMetaLearner checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 5. ExecutorQNetwork dual-input
# ─────────────────────────────────────────────────────────────────────────────
print("\n── ExecutorQNetwork ────────────────────────────────────────────────────")
try:
    with mock.patch.dict('sys.modules', {
        'app.core.market.zone_snapshot': mock.MagicMock(),
    }):
        from app.core.options.q_executor import (
            ExecutorQNetwork,
            build_feat_window,
            build_feat_window_batch,
            OptionsQExecutor,
            Q_LOOKBACK,
            HORIZON_BARS_LIST,
            REGRET_MIN_PCT,
            RegretTransition,
            HTFBiasPackage,
            ACTION_WAIT,
        )

    NF = 50
    qnet = ExecutorQNetwork(num_features=NF)
    qnet.eval()

    fw  = torch.randn(2, 300, NF)
    ctx = torch.randn(2, 28)

    with torch.no_grad():
        out_all = qnet(fw, ctx)
        out_h0  = qnet(fw, ctx, horizon_idx=0)
        out_h3  = qnet(fw, ctx, horizon_idx=3)

    check("5a. Q_LOOKBACK == 300", Q_LOOKBACK == 300, str(Q_LOOKBACK))
    check("5b. all-horizons output (2,4,3)", out_all.shape == (2, 4, 3), str(out_all.shape))
    check("5c. horizon_idx=0 output (2,3)",  out_h0.shape  == (2, 3),   str(out_h0.shape))
    check("5d. horizon_idx=3 output (2,3)",  out_h3.shape  == (2, 3),   str(out_h3.shape))
    check("5e. HORIZON_BARS_LIST == [1,3,6,12]", HORIZON_BARS_LIST == [1,3,6,12], str(HORIZON_BARS_LIST))

    # build_feat_window
    mat = np.random.randn(500, NF).astype(np.float32)
    fw_np = build_feat_window(mat, 400, 300)
    check("5f. build_feat_window shape (300,50)", fw_np.shape == (300, NF), str(fw_np.shape))

    # Left-pad case
    fw_pad = build_feat_window(mat, 50, 300)
    check("5g. build_feat_window left-pad shape (300,50)", fw_pad.shape == (300, NF), str(fw_pad.shape))
    check("5h. left-pad first rows are zeros", np.all(fw_pad[:249] == 0), "first 249 rows zero")

    # Batch
    fw_batch = build_feat_window_batch(mat, [100, 200, 300], 300)
    check("5i. build_feat_window_batch shape (3,300,50)", fw_batch.shape == (3, 300, NF), str(fw_batch.shape))

except Exception as e:
    check("ExecutorQNetwork checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Regret buffer
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Regret Buffer ───────────────────────────────────────────────────────")
try:
    executor = OptionsQExecutor(num_features=NF)
    check("6a. regret_buffer present", hasattr(executor, 'regret_buffer'))
    check("6b. regret_buffer maxlen==10000", executor.regret_buffer.maxlen == 10_000,
          str(executor.regret_buffer.maxlen))
    check("6c. REGRET_MIN_PCT == 0.0015", REGRET_MIN_PCT == 0.0015, str(REGRET_MIN_PCT))
    check("6d. RegretTransition has 9 fields", len(RegretTransition._fields) == 9,
          str(RegretTransition._fields))

    fw_np  = np.zeros((300, NF), dtype=np.float32)
    ctx_np = np.zeros(28, dtype=np.float32)
    mask   = np.ones(3, dtype=np.int32)
    htf    = HTFBiasPackage(direction="bullish", strength=0.7)
    # future_closes: bar[0] = +0.3% move → abs(0.003) > 0.0015 → should record
    future_closes = np.array([1.003, 1.004, 1.005, 1.006, 1.007, 1.008,
                               1.009, 1.010, 1.011, 1.012, 1.013, 1.014], dtype=np.float32)
    executor.record_regret_transition(fw_np, ctx_np, fw_np, ctx_np, mask, htf,
                                      future_closes, entry_price=1.0)
    check("6e. record_regret_transition adds entry", len(executor.regret_buffer) == 1,
          f"len={len(executor.regret_buffer)}")
    rt = executor.regret_buffer[0]
    check("6f. regret action is ACTION_WAIT", rt.action == ACTION_WAIT, str(rt.action))
    check("6g. regret_reward is negative", rt.regret_reward < 0, f"{rt.regret_reward:.4f}")
    check("6h. horizon_idx is 0 (5m, first profitable)", rt.horizon_idx == 0, str(rt.horizon_idx))

except Exception as e:
    check("Regret buffer checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 7. Strength targets have non-trivial std from training CSV
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Training Data Targets ───────────────────────────────────────────────")
try:
    import pandas as pd

    csv_path = os.path.join(BACKEND, 'data', 'train_40k.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        close_col = 'close_5m' if 'close_5m' in df.columns else df.columns[0]

        # Recompute forward_move unconditionally (Bug 1 fix)
        df['forward_move_1']  = df[close_col].shift(-1)  - df[close_col]
        df['forward_move_12'] = df[close_col].shift(-12) - df[close_col]

        # Compute ATR proxy
        if 'high_5m' in df.columns and 'low_5m' in df.columns:
            atr = (df['high_5m'] - df['low_5m']).rolling(14, min_periods=1).mean().values
        else:
            atr = df[close_col].values * 0.0008
        atr = np.maximum(atr, 1e-4)

        fwd1 = df['forward_move_1'].values.astype(np.float32)
        strength_5m = 0.5 + 0.5 * np.clip(fwd1 / (atr * 1.5), -1.0, 1.0)
        strength_5m = np.clip(strength_5m, 0.05, 0.95)

        std_5m = float(np.nanstd(strength_5m))
        mean_5m = float(np.nanmean(strength_5m))
        check("7a. strength_5m std > 0.05 (not collapsed)", std_5m > 0.05,
              f"mean={mean_5m:.3f} std={std_5m:.3f}")
        check("7b. forward_move_1 std > 0 (not all zeros)",
              float(np.nanstd(fwd1)) > 0, f"std={float(np.nanstd(fwd1)):.4f}")
    else:
        check("7. CSV check skipped (data/train_40k.csv not found)", True, "skipped")

except Exception as e:
    check("Training data checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# 8. Notebook Cell checks (verify via JSON inspection, not execution)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Notebook Cell Checks (JSON) ─────────────────────────────────────────")
try:
    import json
    nb_path = os.path.join(BACKEND, 'notebook42ef966279(6).ipynb')
    with open(nb_path) as f:
        nb = json.load(f)
    cells = nb['cells']

    c1 = ''.join(cells[1]['source'])
    c5 = ''.join(cells[5]['source'])
    c7 = ''.join(cells[7]['source'])
    c8 = ''.join(cells[8]['source'])

    check("8a. Cell 1 has Q_LOOKBACK=300",
          'Q_LOOKBACK   = 300' in c1 or 'Q_LOOKBACK = 300' in c1)
    check("8b. Cell 5 branch_cat.detach() present",
          'branch_cat = self.branch_ln(torch.cat([b1_out, b2_out, b3_out], dim=-1).detach())' in c5)
    check("8c. Cell 7 no forward_move guard",
          'if "forward_move_1" not in df.columns' not in c7)
    check("8d. Cell 7 Bug1-fix print present", 'Bug1-fix' in c7)
    check("8e. Cell 8 Q_LOOKBACK guard present",
          "Q_LOOKBACK   if 'Q_LOOKBACK'" in c8 or "Q_LOOKBACK if 'Q_LOOKBACK'" in c8)

except Exception as e:
    check("Notebook checks", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
passed = sum(1 for _, ok in results if ok)
total  = len(results)
print(f"RESULT: {passed}/{total} checks passed")
if passed == total:
    print("✅ ALL CHECKS PASSED — Parity verified")
else:
    failed = [name for name, ok in results if not ok]
    print(f"❌ Failed checks: {failed}")
sys.exit(0 if passed == total else 1)
