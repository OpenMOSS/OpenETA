from __future__ import annotations

from pathlib import Path

import yaml

from sim.envs.univtac.simulator_install_contract import required_runtime_versions
from sim.envs.univtac.validated_dependency_baseline import (
    audit_changes_against_plan,
    binary_manifest,
    compare_binary_manifests,
    compare_config_to_source,
    package_diff,
    report_install_versions,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((ROOT / "configs/univtac/isaac51_blackwell_runtime.yaml").read_text(encoding="utf-8"))


def test_config_must_equal_source_and_target_actual_versions() -> None:
    versions = required_runtime_versions(CONFIG)
    assert compare_config_to_source(CONFIG, versions, versions)["success"] is True
    source = dict(versions)
    source["packaging"] = "23.0"
    assert compare_config_to_source(CONFIG, source, versions)["success"] is False


def test_flatdict_is_the_only_allowed_install_change() -> None:
    before = {"torch": "2.7.0+cu128", "flatdict": None}
    after = {"torch": "2.7.0+cu128", "flatdict": "4.0.1"}
    assert package_diff(before, after, allowed_additions={"flatdict": "4.0.1"})["success"] is True
    after["torch"] = "2.7.0+cu126"
    assert package_diff(before, after, allowed_additions={"flatdict": "4.0.1"})["success"] is False


def test_binary_manifests_detect_changes(tmp_path: Path) -> None:
    binary = tmp_path / "extension.so"
    binary.write_bytes(b"before")
    before = binary_manifest([binary], root=tmp_path)
    binary.write_bytes(b"after")
    after = binary_manifest([binary], root=tmp_path)
    assert compare_binary_manifests(before, after)["success"] is False


def test_actual_changes_must_be_in_the_reviewed_report_plan() -> None:
    report = {"install": [{"metadata": {"name": "PyBind11", "version": "3.0.2"}}]}
    plan = report_install_versions(report)
    before = {"existing": "1.0"}
    after = {"existing": "1.0", "pybind11": "3.0.2", "tacex": "0.1.0"}
    result = audit_changes_against_plan(before, after, plan, additional_allowed={"tacex": "0.1.0"})
    assert result["success"] is True
    after["existing"] = "2.0"
    assert audit_changes_against_plan(before, after, plan)["success"] is False
