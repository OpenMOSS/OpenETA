"""Fresh-interpreter checks; no server, model inference, GPU, or download.

The synthetic checkpoint is created by the test itself. To additionally check
deployed weights, set OPENETA_CONTACT_GRASPNET_TRUSTED_CHECKPOINT to a reviewed
local checkpoint. Loading stays restricted in both cases.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("order", ["mink_first", "loader_first"])
@pytest.mark.parametrize("checkpoint_kind", ["synthetic", "trusted_local"])
def test_checkpoint_loading_after_mink_import(tmp_path, order, checkpoint_kind):
    for dependency in ("torch", "mink"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"{dependency} is not installed in this test interpreter")
    path = tmp_path / "synthetic.pt"
    if checkpoint_kind == "trusted_local":
        configured = os.environ.get("OPENETA_CONTACT_GRASPNET_TRUSTED_CHECKPOINT", "")
        if not configured:
            pytest.skip("No explicitly trusted local Contact-GraspNet checkpoint configured")
        path = Path(configured).resolve(strict=True)
        assert path.is_file()
    program = r'''
import importlib.metadata
from pathlib import Path
import sys

order, kind, checkpoint = sys.argv[1:]
if order == "mink_first":
    import mink
from tools.contact_graspnet_core import load_checkpoint_model_state
import numpy as np
import torch
if order == "loader_first":
    import mink
path = Path(checkpoint)
if kind == "synthetic":
    torch.save({"model": {"weight": torch.ones(1)}, "metric": np.float64(0.5)}, path)
state = load_checkpoint_model_state(torch=torch, np=np, checkpoint_path=path, device="cpu")
assert isinstance(state, dict)
if kind == "synthetic":
    assert torch.equal(state["weight"], torch.ones(1))
print({"numpy": np.__version__, "torch": torch.__version__,
       "mink": importlib.metadata.version("mink"), "order": order, "checkpoint": kind})
'''
    completed = subprocess.run(
        [sys.executable, "-c", program, order, checkpoint_kind, str(path)],
        cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=90,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    print(completed.stdout)
