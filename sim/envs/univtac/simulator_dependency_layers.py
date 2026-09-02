"""Explicit native-reference and simulator-integration dependency layers."""

from __future__ import annotations

from typing import Any, Mapping


def packaging_layers(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "native_reference_environment": "UniVTAC-isaac51-sm120-r08",
        "native_reference_packaging": "26.3",
        "simulator_environment": "UniVTAC-isaac51-sm120-r09",
        "simulator_integration_packaging": "23.0",
        "packaging_baseline_source": config["baseline_source"],
        "native_invariants_unchanged": True,
        "task_physics_sensor_control_semantics_changed": False,
    }
