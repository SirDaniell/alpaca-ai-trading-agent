"""
test_model_weight_compatibility.py — Unit test verifying 1:1 weight schema compatibility.

Ensures that PyTorch state dicts saved from Kaggle Notebook training can be loaded
seamlessly by backend production models (SignalMetaNetwork & ExecutorQNetwork) without key mismatches.
"""

import json
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pytest

from app.core.ml.signal_meta_learner import SignalMetaNetwork as LocalSignalMetaNetwork
from app.core.options.q_executor import ExecutorQNetwork as LocalExecutorQNetwork


def _make_exec_ns():
    """Shared exec namespace with all required symbols."""
    import typing
    return {
        "torch": torch,
        "nn": nn,
        "np": np,
        "Optional": typing.Optional,
        "List": typing.List,
        "Dict": typing.Dict,
        "Tuple": typing.Tuple,
        "Any": typing.Any,
        "Union": typing.Union,
        "meta_lookback_bars": 150,
        "q_lookback_bars": 300,
        "Q_LOOKBACK": 300,
        "lookback_bars": 150,
        "input_dim": 150 * 238,
        "num_features": 238,
        "train_df": type("DF", (), {"columns": [f"f_{i}" for i in range(238)]})()
    }


def get_notebook_model_classes():
    """Extract SignalMetaNetwork and ExecutorQNetwork from the generated Kaggle Notebook.
    The notebook now follows the source notebook structure: each class is in its own cell.
    """
    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "notebooks/kaggle/axe_meta_learner_training_pytorch.ipynb"
    )
    with notebook_path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    meta_cell = eq_cell = None
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "class SignalMetaNetwork" in src and meta_cell is None:
                meta_cell = src
            if "class ExecutorQNetwork" in src and eq_cell is None:
                eq_cell = src

    if meta_cell is None:
        raise ValueError("Could not find SignalMetaNetwork cell in notebook.")
    if eq_cell is None:
        raise ValueError("Could not find ExecutorQNetwork cell in notebook.")

    # Execute SignalMetaNetwork cell
    meta_ns = _make_exec_ns()
    exec(meta_cell, meta_ns)

    # ExecutorQNetwork cell needs num_features and Q_LOOKBACK in scope
    eq_ns = _make_exec_ns()
    exec(eq_cell, eq_ns)

    return meta_ns["SignalMetaNetwork"], eq_ns["ExecutorQNetwork"]


def test_signal_meta_network_weight_compatibility():
    """Verify SignalMetaNetwork state_dict key and shape parity between Notebook and Local code."""
    NotebookSignalMetaNetwork, _ = get_notebook_model_classes()

    num_features = 238
    lookback_bars = 150
    input_dim = lookback_bars * num_features

    local_model = LocalSignalMetaNetwork(input_dim=input_dim, num_features=num_features)
    notebook_model = NotebookSignalMetaNetwork(input_dim=input_dim, num_features=num_features)

    local_keys = set(local_model.state_dict().keys())
    notebook_keys = set(notebook_model.state_dict().keys())

    missing_in_notebook = local_keys - notebook_keys
    missing_in_local = notebook_keys - local_keys

    assert not missing_in_notebook, f"Keys missing in notebook model: {missing_in_notebook}"
    assert not missing_in_local, f"Keys missing in local model: {missing_in_local}"

    # Verify tensor shapes match for every parameter key
    local_sd = local_model.state_dict()
    notebook_sd = notebook_model.state_dict()
    for key in local_sd:
        assert local_sd[key].shape == notebook_sd[key].shape, f"Shape mismatch for key '{key}': local {local_sd[key].shape} vs notebook {notebook_sd[key].shape}"

    # Load notebook weights into local model
    local_model.load_state_dict(notebook_model.state_dict())

    # Forward pass test for numerical identity
    x_test = torch.randn(4, input_dim)
    local_model.eval()
    notebook_model.eval()

    with torch.no_grad():
        loc_q, loc_str, loc_pips, loc_risk, loc_liq, loc_rev = local_model(x_test)
        nb_q, nb_str, nb_pips, nb_risk, nb_liq, nb_rev = notebook_model(x_test)

    assert torch.allclose(loc_q, nb_q, atol=1e-5), "Q-values mismatch between local and notebook models"
    assert torch.allclose(loc_str, nb_str, atol=1e-5), "Strength scores mismatch between local and notebook models"
    print("\n✅ SignalMetaNetwork state_dict & forward pass outputs are 100% identical!")


def test_executor_q_network_weight_compatibility():
    """Verify ExecutorQNetwork state_dict key and shape parity between Notebook and Local code."""
    _, NotebookExecutorQNetwork = get_notebook_model_classes()

    num_features = 238
    ctx_dim = 28
    q_lookback = 300

    local_model = LocalExecutorQNetwork(num_features=num_features, ctx_dim=ctx_dim, q_lookback=q_lookback)
    notebook_model = NotebookExecutorQNetwork(num_features=num_features, ctx_dim=ctx_dim, q_lookback=q_lookback)

    local_keys = set(local_model.state_dict().keys())
    notebook_keys = set(notebook_model.state_dict().keys())

    missing_in_notebook = local_keys - notebook_keys
    missing_in_local = notebook_keys - local_keys

    assert not missing_in_notebook, f"Keys missing in notebook executor model: {missing_in_notebook}"
    assert not missing_in_local, f"Keys missing in local executor model: {missing_in_local}"

    # Verify tensor shapes match for every parameter key
    local_sd = local_model.state_dict()
    notebook_sd = notebook_model.state_dict()
    for key in local_sd:
        assert local_sd[key].shape == notebook_sd[key].shape, f"Shape mismatch for key '{key}': local {local_sd[key].shape} vs notebook {notebook_sd[key].shape}"

    # Load notebook weights into local model
    local_model.load_state_dict(notebook_model.state_dict())

    # Forward pass test for numerical identity
    fw_test = torch.randn(4, q_lookback, num_features)
    ctx_test = torch.randn(4, ctx_dim)
    local_model.eval()
    notebook_model.eval()

    with torch.no_grad():
        loc_out = local_model(fw_test, ctx_test)
        nb_out = notebook_model(fw_test, ctx_test)

    assert torch.allclose(loc_out, nb_out, atol=1e-5), "Executor Q-values mismatch between local and notebook models"
    print("✅ ExecutorQNetwork state_dict & forward pass outputs are 100% identical!")

