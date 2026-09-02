"""Pure contracts for the cuRobo/Warp public-API compatibility gate."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PUBLIC_APIS = (
    "device_from_torch",
    "device_to_torch",
    "stream_from_torch",
    "from_torch",
    "to_torch",
)


def classify_warp_api(payload: Mapping[str, Any]) -> str:
    matrix = payload.get("api_matrix", {})
    if not all(bool(matrix.get(name)) for name in PUBLIC_APIS):
        return "warp_public_api_unavailable"
    if not payload.get("interop_success"):
        return "warp_public_api_unavailable"
    return "passed"


def validate_warp_kernel_witness(payload: Mapping[str, Any]) -> None:
    if payload.get("used_private_warp_torch"):
        raise ValueError("Warp kernel probe used private wp.torch API")
    if payload.get("input_witness") != list(range(16)):
        raise ValueError("unexpected Warp kernel input witness")
    if payload.get("first_output_witness") != list(range(1, 17)):
        raise ValueError("unexpected first Warp kernel output witness")
    if payload.get("second_output_witness") != list(range(2, 18)):
        raise ValueError("unexpected second Warp kernel output witness")
    if not payload.get("pointer_identity", {}).get("input"):
        raise ValueError("Torch/Warp input pointer identity failed")
    if not payload.get("pointer_identity", {}).get("output"):
        raise ValueError("Torch/Warp output pointer identity failed")
