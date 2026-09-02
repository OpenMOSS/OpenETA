"""Pure-Python contract for the protected R0.9 simulator installation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "openeta.univtac.isaac51_blackwell_runtime.v1"
RUNTIME_VARIANT = (
    "blackwell_compat_adaptation_v1+tinygltf_republished_archive_recovery_v1+"
    "conda_cuda_target_include_bridge_v1+curobo_warp113_public_api_backport_v1+"
    "isaac51_blackwell_simulator_integration_v1+"
    "isaaclab_setuptools_scm8_packaging23_bridge_v1+"
    "isaacsim_filelock3131_compatibility_bridge_v1"
)
EXPECTED_GATE_ORDER = (
    "E0", "P0", "I0", "I1", "I2", "N0", "S0", "S1", "G0", "L0", "C0", "H0"
)
EXPECTED_CLAIMS = {
    "official_univtac_recipe_exact": False,
    "legacy_ftp1_reproduction": False,
    "benchmark_version": "UniVTAC-Isaac51",
    "within_version_experiment_target": True,
    "cross_version_numeric_parity": False,
}
EXPECTED_ENVIRONMENTS = {
    "clone_from": "UniVTAC-isaac51-sm120-r08",
    "conda_name": "UniVTAC-isaac51-sm120-r09",
}
EXPECTED_CONSTRAINTS = {
    "torch": "2.7.0+cu128",
    "torchvision": "0.22.0+cu128",
    "warp-lang": "1.17.0",
    "pyuipc": "0.9.0",
}
EXPECTED_RUNTIME_VERSIONS = {
    "nvidia-curobo": "0.7.7.post1.dev5+dirty",
    "setuptools": "75.8.2",
    "setuptools-scm": "8.1.0",
    "wheel": "0.42.0",
    "packaging": "23.0",
    "filelock": "3.13.1",
}
PROTECTED_BASELINE_SOURCE = "r08_validated_actual_environment"
INSTALLATION_METHOD_LABEL = "flatdict_4_0_1_verified_sdist_wheel_bridge_v1"
EXPECTED_COLLECTION = {
    "G0": ("grasp_classify", 0, "official_isaac51_phase1_collection_smoke", "grasp_classify_seed0"),
    "L0": ("lift_can", 1_000_000, "legacy_ftp1_eval_seed_index_aligned", "lift_can_seed1000000"),
    "C0": ("pull_out_key", 1_000_000, "legacy_ftp1_eval_seed_index_aligned", "pull_out_key_seed1000000"),
    "H0": ("insert_hole", 1_000_000, "legacy_ftp1_eval_seed_index_aligned", "insert_hole_seed1000000"),
}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class ResolutionAudit:
    success: bool
    protected_changes: tuple[dict[str, Any], ...]
    install_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "protected_changes": list(self.protected_changes),
            "install_count": self.install_count,
        }


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("runtime_variant") != RUNTIME_VARIANT:
        raise ValueError("runtime_variant changed")
    if config.get("claims") != EXPECTED_CLAIMS:
        raise ValueError("benchmark claim boundary changed")
    environment = config.get("environment", {})
    for key, value in EXPECTED_ENVIRONMENTS.items():
        if environment.get(key) != value:
            raise ValueError(f"environment.{key} must be {value}")
    if config.get("protected_constraints") != EXPECTED_CONSTRAINTS:
        raise ValueError("protected constraints changed")
    if config.get("protected_baseline_source") != PROTECTED_BASELINE_SOURCE:
        raise ValueError("protected_baseline_source changed")
    if config.get("installation_method_label") != INSTALLATION_METHOD_LABEL:
        raise ValueError("installation_method_label changed")
    protected_runtime = config.get("protected_runtime", {})
    actual_versions = {
        canonical_name(name): str(record.get("version"))
        for name, record in protected_runtime.items()
    }
    if actual_versions != EXPECTED_RUNTIME_VERSIONS:
        raise ValueError("protected runtime versions changed")
    if tuple(config.get("gate_order", ())) != EXPECTED_GATE_ORDER:
        raise ValueError(f"gate order must be {EXPECTED_GATE_ORDER}")
    collection = config.get("collection_gates")
    if not isinstance(collection, list) or [item.get("id") for item in collection] != ["G0", "L0", "C0", "H0"]:
        raise ValueError("collection gate order changed")
    for item in collection:
        gate_id = str(item["id"])
        actual = (str(item["task"]), int(item["seed"]), str(item["label"]), str(item["output_dir"]))
        if actual != EXPECTED_COLLECTION[gate_id]:
            raise ValueError(f"{gate_id} collection contract changed")
    if collection[-1].get("conditional") is not True or collection[-1].get("requires") != "C0_runtime_and_tactile_valid":
        raise ValueError("H0 must remain conditional on C0 runtime and tactile validity")


def clone_command(conda: Path, source: str, target: str) -> list[str]:
    if source != EXPECTED_ENVIRONMENTS["clone_from"] or target != EXPECTED_ENVIRONMENTS["conda_name"]:
        raise ValueError("R0.9 must clone the fixed R0.8 environment into the fixed R0.9 name")
    return [str(conda), "create", "--name", target, "--clone", source, "--yes"]


def constraints_text(config: Mapping[str, Any]) -> str:
    constraints = required_runtime_versions(config)
    constrained_names = (
        "torch", "torchvision", "warp-lang", "pyuipc", "setuptools",
        "setuptools-scm", "wheel", "packaging", "filelock",
    )
    return "".join(f"{name}=={constraints[name]}\n" for name in constrained_names)


def required_runtime_versions(config: Mapping[str, Any]) -> dict[str, str]:
    versions = dict(config["protected_constraints"])
    versions.update(
        {name: str(record["version"]) for name, record in config["protected_runtime"].items()}
    )
    return {canonical_name(name): version for name, version in versions.items()}


def audit_installed_versions(
    observed: Mapping[str, str | None], required: Mapping[str, str]
) -> dict[str, Any]:
    normalized = {canonical_name(name): value for name, value in observed.items()}
    mismatches = [
        {"name": name, "required": version, "observed": normalized.get(name)}
        for name, version in sorted(required.items())
        if normalized.get(name) != version
    ]
    return {"success": not mismatches, "mismatches": mismatches}


def audit_pip_report(report: Mapping[str, Any], protected_names: Sequence[str]) -> ResolutionAudit:
    protected = {canonical_name(name) for name in protected_names}
    installs = report.get("install", [])
    if not isinstance(installs, list):
        raise TypeError("pip report install field must be a list")
    changes: list[dict[str, Any]] = []
    for record in installs:
        metadata = record.get("metadata", {}) if isinstance(record, Mapping) else {}
        name = canonical_name(str(metadata.get("name", "")))
        if name in protected:
            changes.append(
                {
                    "name": name,
                    "resolved_version": metadata.get("version"),
                    "requested": bool(record.get("requested")),
                    "download_url": record.get("download_info", {}).get("url")
                    if isinstance(record.get("download_info"), Mapping)
                    else None,
                }
            )
    return ResolutionAudit(success=not changes, protected_changes=tuple(changes), install_count=len(installs))


def load_pip_report(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("pip report must be a mapping")
    return payload


def validate_provenance_marker(marker: Mapping[str, Any], *, source: str, target: str) -> None:
    expected = {
        "clone_from": source,
        "target_environment": target,
        "runtime_variant": RUNTIME_VARIANT,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise ValueError(f"provenance marker {key} mismatch")


def next_stage(completed: Sequence[str]) -> str | None:
    completed_tuple = tuple(completed)
    if completed_tuple != EXPECTED_GATE_ORDER[: len(completed_tuple)]:
        raise ValueError("completed stages are not an ordered prefix")
    return EXPECTED_GATE_ORDER[len(completed_tuple)] if len(completed_tuple) < len(EXPECTED_GATE_ORDER) else None
