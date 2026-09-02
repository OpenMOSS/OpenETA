#!/usr/bin/env python3
"""Capture finite cuRobo collision tensors using the official example configuration."""

from __future__ import annotations

import argparse
import importlib.util
import json
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "sim/envs/univtac/collision_runtime_validation.py"
helper_spec = importlib.util.spec_from_file_location("univtac_collision_runtime_validation", HELPER)
if helper_spec is None or helper_spec.loader is None:
    raise ImportError(f"cannot load collision helper: {HELPER}")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_root = args.expected_root.resolve()
    payload: dict[str, object] = {
        "success": False,
        "collision_distance_called": False,
        "torch_cuda_synchronized": False,
    }
    try:
        import torch
        import warp as wp
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

        tensor_args = TensorDeviceType()
        config = RobotWorldConfig.load_from_config(
            "franka.yml", "collision_test.yml", collision_activation_distance=0.0
        )
        robot_world = RobotWorld(config)
        q_sph = torch.zeros((10, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
        q_sph[:, 0, 0, 0] = torch.linspace(-0.2, 0.7, 10, device=tensor_args.device)
        q_sph[..., 3] = 0.2
        sphere_distance = robot_world.get_collision_distance(q_sph)
        q_sample = robot_world.sample(5, mask_valid=False)
        world_distance, self_distance = robot_world.get_world_self_collision_distance_from_joints(
            q_sample
        )
        payload["collision_distance_called"] = True
        torch.cuda.synchronize()
        payload["torch_cuda_synchronized"] = True
        maps = Path("/proc/self/maps").read_text(encoding="utf-8")
        loaded = sorted(
            {line.split()[-1] for line in maps.splitlines() if "curobolib/" in line and "_cu" in line}
        )
        geom = [path for path in loaded if Path(path).name.startswith("geom_cu.")]
        payload.update(
            {
                "warp": {
                    "version": getattr(wp, "__version__", None),
                    "private_torch_attribute_present": hasattr(wp, "torch"),
                    "world_mesh_device": str(robot_world.world_model._wp_device),
                },
                "loaded_extensions": loaded,
                "geom_extension_loaded": len(geom) == 1
                and Path(geom[0]).resolve().is_relative_to(expected_root),
                "tensors": {
                    "sphere_world_distance": helper.tensor_statistics(sphere_distance),
                    "joint_world_distance": helper.tensor_statistics(world_distance),
                    "joint_self_distance": helper.tensor_statistics(self_distance),
                },
            }
        )
        helper.validate_collision_witness(payload)
        payload.update({"success": True, "classification": "passed"})
    except BaseException as exc:  # noqa: BLE001
        payload.update(
            {
                "classification": "curobo_collision_output_invalid",
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
