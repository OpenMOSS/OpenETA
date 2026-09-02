from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from sim.envs.univtac.simulator_install_contract import (
    EXPECTED_GATE_ORDER,
    audit_installed_versions,
    audit_pip_report,
    clone_command,
    constraints_text,
    next_stage,
    required_runtime_versions,
    validate_config,
    validate_provenance_marker,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/univtac/isaac51_blackwell_runtime.yaml"


def config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_config_freezes_claims_clone_and_gate_order() -> None:
    payload = config()
    validate_config(payload)
    assert payload["claims"] == {
        "official_univtac_recipe_exact": False,
        "legacy_ftp1_reproduction": False,
        "benchmark_version": "UniVTAC-Isaac51",
        "within_version_experiment_target": True,
        "cross_version_numeric_parity": False,
    }
    assert tuple(payload["gate_order"]) == EXPECTED_GATE_ORDER


def test_config_rejects_recipe_or_environment_drift() -> None:
    changed = copy.deepcopy(config())
    changed["claims"]["official_univtac_recipe_exact"] = True
    with pytest.raises(ValueError, match="claim"):
        validate_config(changed)
    changed = copy.deepcopy(config())
    changed["environment"]["clone_from"] = "other"
    with pytest.raises(ValueError, match="clone_from"):
        validate_config(changed)


def test_clone_command_is_exact_r08_to_r09() -> None:
    command = clone_command(Path("/conda"), "UniVTAC-isaac51-sm120-r08", "UniVTAC-isaac51-sm120-r09")
    assert command == ["/conda", "create", "--name", "UniVTAC-isaac51-sm120-r09", "--clone", "UniVTAC-isaac51-sm120-r08", "--yes"]
    with pytest.raises(ValueError, match="fixed R0.8"):
        clone_command(Path("conda"), "wrong", "UniVTAC-isaac51-sm120-r09")


def test_constraints_are_deterministic_and_do_not_pin_isaac() -> None:
    text = constraints_text(config())
    assert text == (
        "torch==2.7.0+cu128\ntorchvision==0.22.0+cu128\nwarp-lang==1.17.0\n"
        "pyuipc==0.9.0\nsetuptools==75.8.2\nsetuptools-scm==8.1.0\n"
        "wheel==0.42.0\npackaging==23.0\nfilelock==3.13.1\n"
    )
    assert "isaac" not in text


def test_pip_report_rejects_any_protected_install() -> None:
    report = {"install": [{"metadata": {"name": "Torch", "version": "2.7.0"}, "requested": False, "download_info": {"url": "https://example/torch.whl"}}]}
    result = audit_pip_report(report, ["torch", "pyuipc"])
    assert result.success is False
    assert result.protected_changes[0]["name"] == "torch"
    assert audit_pip_report({"install": [{"metadata": {"name": "isaacsim", "version": "5.1.0.0"}}]}, ["torch"]).success is True


def test_installed_version_audit_is_exact() -> None:
    required = required_runtime_versions(config())
    assert audit_installed_versions(required, required)["success"] is True
    changed = dict(required)
    changed["packaging"] = "99"
    result = audit_installed_versions(changed, required)
    assert result["success"] is False
    assert result["mismatches"] == [{"name": "packaging", "required": "23.0", "observed": "99"}]


def test_old_unvalidated_build_tool_versions_are_rejected() -> None:
    payload = config()
    assert "10.2.2" not in json.dumps(payload["protected_runtime"])
    assert '"26.3"' not in json.dumps(payload["protected_runtime"])
    assert payload["protected_runtime"]["filelock"]["version"] == "3.13.1"
    changed = copy.deepcopy(payload)
    changed["protected_runtime"]["packaging"]["version"] = "26.3"
    with pytest.raises(ValueError, match="protected runtime"):
        validate_config(changed)


def test_marker_and_order_are_fail_closed() -> None:
    marker = {
        "clone_from": "UniVTAC-isaac51-sm120-r08",
        "target_environment": "UniVTAC-isaac51-sm120-r09",
        "runtime_variant": config()["runtime_variant"],
    }
    validate_provenance_marker(marker, source=marker["clone_from"], target=marker["target_environment"])
    assert next_stage([]) == "E0"
    assert next_stage(list(EXPECTED_GATE_ORDER)) is None
    with pytest.raises(ValueError, match="ordered prefix"):
        next_stage(["P0"])


def test_config_has_no_machine_specific_absolute_paths() -> None:
    text = CONFIG.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "/tmp/" not in text
