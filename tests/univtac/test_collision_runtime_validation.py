from __future__ import annotations

import pytest

from sim.envs.univtac.collision_runtime_validation import validate_collision_witness


def test_collision_witness_accepts_finite_cuda_values() -> None:
    payload = {
        "collision_distance_called": True,
        "torch_cuda_synchronized": True,
        "geom_extension_loaded": True,
        "tensors": {
            name: {
                "numel": 3,
                "device": "cuda:0",
                "finite_count": 3,
                "nan_count": 0,
                "inf_count": 0,
                "min": 0.0,
                "max": 1.0 if name == "sphere_world_distance" else 0.0,
            }
            for name in ("sphere_world_distance", "joint_world_distance", "joint_self_distance")
        },
    }
    validate_collision_witness(payload)


def test_collision_witness_rejects_nan_or_missing_kernel_path() -> None:
    payload = {
        "collision_distance_called": True,
        "torch_cuda_synchronized": True,
        "geom_extension_loaded": True,
        "tensors": {
            "sphere_world_distance": {
                "numel": 2,
                "device": "cuda:0",
                "finite_count": 1,
                "nan_count": 1,
                "inf_count": 0,
                "min": 0.0,
                "max": 1.0,
            },
            "joint_world_distance": {
                "numel": 1,
                "device": "cuda:0",
                "finite_count": 1,
                "nan_count": 0,
                "inf_count": 0,
                "min": 0.0,
                "max": 0.0,
            },
            "joint_self_distance": {
                "numel": 1,
                "device": "cuda:0",
                "finite_count": 1,
                "nan_count": 0,
                "inf_count": 0,
                "min": 0.0,
                "max": 0.0,
            }
        },
    }
    with pytest.raises(ValueError, match="not finite"):
        validate_collision_witness(payload)


def test_collision_witness_requires_all_three_tensor_classes() -> None:
    payload = {
        "collision_distance_called": True,
        "torch_cuda_synchronized": True,
        "geom_extension_loaded": True,
        "tensors": {},
    }
    with pytest.raises(ValueError, match="tensor set mismatch"):
        validate_collision_witness(payload)
