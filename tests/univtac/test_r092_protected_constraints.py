from __future__ import annotations

import yaml
from pathlib import Path

from sim.envs.univtac.simulator_dependency_layers import packaging_layers


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/univtac/isaaclab_packaging23_bridge.yaml").read_text(encoding="utf-8"))


def test_native_and_simulator_packaging_layers_remain_distinct() -> None:
    layers = packaging_layers(CONFIG)
    assert layers["native_reference_packaging"] == "26.3"
    assert layers["simulator_integration_packaging"] == "23.0"
    assert layers["native_reference_environment"] == "UniVTAC-isaac51-sm120-r08"
    assert layers["simulator_environment"] == "UniVTAC-isaac51-sm120-r09"
    assert layers["native_invariants_unchanged"] is True
