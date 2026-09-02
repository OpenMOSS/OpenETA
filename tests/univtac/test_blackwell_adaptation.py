from __future__ import annotations

import ast
import copy
from pathlib import Path

import pytest
import yaml

from sim.envs.univtac.blackwell_adaptation import (
    EXPECTED_ADAPTATIONS,
    RUNTIME_VARIANT,
    classify_torch_probe,
    derive_libuipc_environment,
    load_and_validate_config,
    real_kernel_operations,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/univtac/blackwell_native_bridge.yaml"


def _config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_runtime_variant_and_claim_boundaries_are_fixed() -> None:
    config = load_and_validate_config(CONFIG)
    assert config["runtime_variant"] == RUNTIME_VARIANT
    assert config["official_recipe_exact"] is False
    assert config["benchmark_reproduction"] is False


def test_only_declared_blackwell_adaptations_are_allowed() -> None:
    assert _config()["authorized_adaptations"] == EXPECTED_ADAPTATIONS
    changed = _config()
    changed["authorized_adaptations"]["extra"] = "unapproved"
    with pytest.raises(ValueError, match="only the four"):
        validate_config(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("torch", "2.8.0"),
        ("torch_cuda", "12.9"),
        ("cuda_toolkit", "13.0"),
        ("cuda_arch", "89"),
        ("cuda_arch", "90"),
        ("cuda_arch", "120a"),
    ],
)
def test_forbidden_version_or_arch_fallback_is_rejected(field: str, value: str) -> None:
    changed = _config()
    changed["environment"][field] = value
    with pytest.raises(ValueError, match=f"environment.{field}"):
        validate_config(changed)


def test_derived_yaml_changes_only_cuda_and_removes_name() -> None:
    official = {
        "name": "uipc_env",
        "channels": ["main", "conda-forge", "nodefaults"],
        "dependencies": ["python=3.11", "pip", "cuda-toolkit=12.6", "cmake=3.26"],
    }
    original = copy.deepcopy(official)
    derived, diff = derive_libuipc_environment(official)
    assert official == original
    assert "name" not in derived
    assert "cuda-toolkit=12.8" in derived["dependencies"]
    assert "cuda-toolkit=12.6" not in derived["dependencies"]
    assert diff["dependency_replacements"] == [
        {"from": "cuda-toolkit=12.6", "to": "cuda-toolkit=12.8"}
    ]


def test_derived_yaml_rejects_additional_cuda126_binding() -> None:
    official = {"name": "x", "channels": [], "dependencies": ["cuda-toolkit=12.6", "other=12.6"]}
    with pytest.raises(ValueError, match="unresolved"):
        derive_libuipc_environment(official)


def test_t0_requires_two_rounds_and_every_real_kernel_operation() -> None:
    operations = {name: 0.01 for name in real_kernel_operations()}
    passing = {"torch_cuda_version": "12.8", "rounds": [{"success": True, "operations": operations}] * 2}
    assert classify_torch_probe(passing) == "passed"
    missing = copy.deepcopy(passing)
    del missing["rounds"][0]["operations"]["convolution"]
    assert classify_torch_probe(missing) == "torch_cu128_sm120_failed"


def test_blackwell_scripts_do_not_import_isaac_or_omni() -> None:
    for relative in (
        "scripts/univtac/probe_blackwell_torch.py",
        "scripts/univtac/probe_libuipc_core.py",
        "scripts/univtac/probe_blackwell_curobo.py",
        "scripts/univtac/run_blackwell_native_bridge.py",
    ):
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert imported.isdisjoint({"isaacsim", "isaaclab", "omni", "carb", "tacex_uipc", "openeta"})
