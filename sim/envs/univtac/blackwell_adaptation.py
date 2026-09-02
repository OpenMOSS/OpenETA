"""Contracts for the pinned RTX 5090 native compatibility adaptation."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


SCHEMA_VERSION = "openeta.univtac.blackwell_native_bridge.v1"
RUNTIME_VARIANT = "blackwell_compat_adaptation_v1"
SOURCE_COMMIT = "371fac67917307026be8f00869fcc1b61c623a9f"
PINS = {
    "tacex_commit": "f2051944e469a961241271fbf6fb60e272fc336b",
    "libuipc_commit": "1a7e93ef68765e4d3c15d5583f5c387e89af5183",
    "muda_commit": "8f9e17d8e76a658df3b6ffeeffbbc9ac47ac54bf",
    "symeigen_commit": "c72a0082e44b3b8062727b25c33ce7450f9fa933",
    "curobo_commit": "ebb71702f3f70e767f40fd8e050674af0288abe8",
    "vcpkg_commit": "dd3097e305afa53f7b4312371f62058d2e665320",
}
EXPECTED_GATE_ORDER = (
    "P0",
    "T0",
    "E0",
    "U0",
    "B0",
    "U1_RUN1",
    "U1_RUN2",
    "C0_BUILD",
    "C0_AUDIT",
    "C0_SMOKE",
)
EXPECTED_ADAPTATIONS = {
    "torch_wheel": "2.7.0+cu126 -> 2.7.0+cu128",
    "torchvision_wheel": "0.22.0+cu126 -> 0.22.0+cu128",
    "cuda_toolkit": "12.6 -> 12.8",
    "cuda_architecture": "89 -> 120",
}
FORBIDDEN_RUNTIME_IMPORTS = ("isaacsim", "isaaclab", "omni", "carb", "tacex_uipc", "openeta")


def deterministic_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_and_validate_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Blackwell bridge config must be a mapping")
    validate_config(payload)
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if config.get("runtime_variant") != RUNTIME_VARIANT:
        raise ValueError(f"runtime_variant must be {RUNTIME_VARIANT}")
    if config.get("official_recipe_exact") is not False:
        raise ValueError("Blackwell adaptation cannot claim exact official recipe")
    if config.get("benchmark_reproduction") is not False:
        raise ValueError("Blackwell adaptation cannot claim benchmark reproduction")
    source = config.get("source", {})
    if source.get("commit") != SOURCE_COMMIT:
        raise ValueError("source commit drift")
    for key, expected in PINS.items():
        if source.get(key) != expected:
            raise ValueError(f"source.{key} must remain {expected}")
    environment = config.get("environment", {})
    expected_environment = {
        "conda_name": "UniVTAC-isaac51-sm120-r08",
        "python": "3.11",
        "torch": "2.7.0",
        "torchvision": "0.22.0",
        "torch_cuda": "12.8",
        "cuda_toolkit": "12.8",
        "cuda_arch": "120",
        "torch_cuda_arch_list": "12.0",
        "build_jobs": 4,
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ValueError(f"environment.{key} must be {expected!r}")
    if "cu128" not in str(environment.get("torch_index_url")):
        raise ValueError("PyTorch index must be the cu128 index")
    if config.get("authorized_adaptations") != EXPECTED_ADAPTATIONS:
        raise ValueError("only the four declared Blackwell adaptations are allowed")
    if tuple(config.get("gate_order", ())) != EXPECTED_GATE_ORDER:
        raise ValueError(f"gate_order must be {EXPECTED_GATE_ORDER}")


def derive_libuipc_environment(official: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    derived = copy.deepcopy(dict(official))
    derived.pop("name", None)
    dependencies = derived.get("dependencies")
    if not isinstance(dependencies, list):
        raise TypeError("official env dependencies must be a list")
    replacements = 0
    output = []
    conflicts = []
    for dependency in dependencies:
        text = str(dependency)
        if text == "cuda-toolkit=12.6":
            output.append("cuda-toolkit=12.8")
            replacements += 1
        else:
            output.append(dependency)
            if "12.6" in text:
                conflicts.append(text)
    if replacements != 1:
        raise ValueError("official env must contain exactly one cuda-toolkit=12.6 dependency")
    if conflicts:
        raise ValueError(f"unresolved CUDA 12.6 package constraints: {conflicts}")
    derived["dependencies"] = output
    semantic_diff = {
        "removed_fields": {"name": official.get("name")},
        "dependency_replacements": [
            {"from": "cuda-toolkit=12.6", "to": "cuda-toolkit=12.8"}
        ],
        "unchanged_channels": official.get("channels") == derived.get("channels"),
        "unresolved_cuda_12_6_constraints": conflicts,
    }
    return derived, semantic_diff


def source_baseline_pins(text: str) -> dict[str, str]:
    result = {}
    labels = {
        "TacEx commit": "tacex_commit",
        "libuipc commit": "libuipc_commit",
        "libuipc/muda commit": "muda_commit",
        "libuipc/SymEigen commit": "symeigen_commit",
    }
    for label, key in labels.items():
        match = re.search(rf"{re.escape(label)}:\s*`([0-9a-f]{{40}})`", text)
        if match:
            result[key] = match.group(1)
    return result


def validate_source_baseline(text: str, config: Mapping[str, Any]) -> dict[str, Any]:
    observed = source_baseline_pins(text)
    expected = {key: config["source"][key] for key in PINS if key in observed}
    missing = sorted(set(("tacex_commit", "libuipc_commit", "muda_commit", "symeigen_commit")) - set(observed))
    mismatched = {key: {"expected": expected.get(key), "observed": value} for key, value in observed.items() if expected.get(key) != value}
    return {"observed": observed, "missing": missing, "mismatched": mismatched, "valid": not missing and not mismatched}


def real_kernel_operations() -> tuple[str, ...]:
    return (
        "allocation",
        "arange",
        "sort",
        "reduction",
        "matrix_multiplication",
        "convolution",
        "random_generator",
        "device_to_host_copy",
        "synchronize",
    )


def classify_torch_probe(payload: Mapping[str, Any]) -> str:
    if payload.get("torch_cuda_version") != "12.8":
        return "torch_cu128_sm120_failed"
    rounds = payload.get("rounds", [])
    if len(rounds) != 2 or not all(item.get("success") for item in rounds):
        return "torch_cu128_sm120_failed"
    if any(name not in item.get("operations", {}) for item in rounds for name in real_kernel_operations()):
        return "torch_cu128_sm120_failed"
    return "passed"


def next_gate(completed: Sequence[str], *, curobo_smoke_available: bool = True) -> str | None:
    prefix = tuple(completed)
    if prefix != EXPECTED_GATE_ORDER[: len(prefix)]:
        raise ValueError("completed gates are not an ordered prefix")
    if len(prefix) == len(EXPECTED_GATE_ORDER):
        return None
    candidate = EXPECTED_GATE_ORDER[len(prefix)]
    if candidate == "C0_SMOKE" and not curobo_smoke_available:
        return None
    return candidate
