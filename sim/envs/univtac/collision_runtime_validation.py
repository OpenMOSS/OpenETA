"""Pure validation helpers for cuRobo collision tensor witnesses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def tensor_statistics(tensor: Any) -> dict[str, Any]:
    import torch

    values = tensor.detach()
    finite = torch.isfinite(values)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "device": str(values.device),
        "numel": values.numel(),
        "min": float(values.min().item()) if values.numel() else None,
        "max": float(values.max().item()) if values.numel() else None,
        "mean": float(values.float().mean().item()) if values.numel() else None,
        "finite_count": int(finite.sum().item()),
        "nan_count": int(torch.isnan(values).sum().item()),
        "inf_count": int(torch.isinf(values).sum().item()),
    }


def validate_collision_witness(payload: Mapping[str, Any]) -> None:
    if not payload.get("collision_distance_called"):
        raise ValueError("collision distance was not called")
    if not payload.get("torch_cuda_synchronized"):
        raise ValueError("Torch CUDA synchronize did not complete")
    if not payload.get("geom_extension_loaded"):
        raise ValueError("geom_cu was not loaded from the derived checkout")
    tensors = payload.get("tensors", {})
    required = {"sphere_world_distance", "joint_world_distance", "joint_self_distance"}
    if set(tensors) != required:
        raise ValueError(f"collision tensor set mismatch: {sorted(tensors)}")
    for name, stats in tensors.items():
        if not stats.get("numel"):
            raise ValueError(f"{name} collision tensor is empty")
        if stats.get("device") != "cuda:0":
            raise ValueError(f"{name} collision tensor is not on cuda:0")
        if stats.get("finite_count") != stats.get("numel"):
            raise ValueError(f"{name} collision tensor is not finite")
        if stats.get("nan_count") or stats.get("inf_count"):
            raise ValueError(f"{name} collision tensor contains NaN or Inf")
    sphere = tensors["sphere_world_distance"]
    if sphere.get("min") == sphere.get("max"):
        raise ValueError("sphere world collision witness is degenerate")
