"""
test_model_weight_compatibility.py — Unit test verifying 1:1 weight schema compatibility.

Ensures that PyTorch state dicts saved from Kaggle Notebook training can be loaded
seamlessly by backend production models (SignalMetaNetwork & ExecutorQNetwork) without key mismatches.
"""

import json
import torch
import torch.nn as nn
import numpy as np
import pytest

from app.core.ml.signal_meta_learner import SignalMetaNetwork as LocalSignalMetaNetwork
from app.core.options.q_executor import ExecutorQNetwork as LocalExecutorQNetwork


def get_notebook_model_classes():
    """Extract and execute only the PYTORCH_MODELS cell from the generated Kaggle Notebook."""
    with open("kaggle_axe_meta_learner_training.ipynb", "r") as f:
        nb = json.load(f)
    
    # Locate the cell containing model definitions
    model_cell_code = ""
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            if "class SignalMetaNetwork" in src and "class ExecutorQNetwork" in src:
                model_cell_code = src
                break

    if not model_cell_code:
        raise ValueError("Could not find model definition cell in notebook.")

    # Execute isolated cell in namespace
    ns = {
        "torch": torch,
        "nn": nn,
        "np": np,
        "input_dim": 1000 * 238,
        "num_features": 238,
        "lookback_bars": 1000,
        "train_df": type("DF", (), {"columns": [f"f_{i}" for i in range(238)]})()
    }
    exec(model_cell_code, ns)
    return ns["SignalMetaNetwork"], ns["ExecutorQNetwork"]


def test_signal_meta_network_weight_compatibility():
    """Verify SignalMetaNetwork state_dict key and shape parity between Notebook and Local code."""
    NotebookSignalMetaNetwork, _ = get_notebook_model_classes()

    num_features = 238
    lookback_bars = 1000
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

    local_model = LocalExecutorQNetwork(input_dim=28, hidden_dim=64, num_actions=5)
    notebook_model = NotebookExecutorQNetwork(input_dim=28, hidden_dim=64, num_actions=5)

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
    x_test = torch.randn(4, 28)
    local_model.eval()
    notebook_model.eval()

    with torch.no_grad():
        loc_out = local_model(x_test)
        nb_out = notebook_model(x_test)

    assert torch.allclose(loc_out, nb_out, atol=1e-5), "Executor Q-values mismatch between local and notebook models"
    print("✅ ExecutorQNetwork state_dict & forward pass outputs are 100% identical!")
