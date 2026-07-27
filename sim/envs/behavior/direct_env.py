"""Single-environment BEHAVIOR-1K adapter for OpenETA workers.

The existing :mod:`behavior_env` is RLinf's distributed training wrapper and
requires Ray worker metadata.  Interactive evaluation instead needs a normal
Gymnasium environment, so this module wraps OmniGibson's official
``og.Environment`` directly.  Heavy imports remain inside ``__init__`` so the
static BDDL catalog can be listed without importing Isaac Sim.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np


_TASK_FILE = Path(__file__).with_name("behavior_task.jsonl")
_IK_POSITION_SCALE_M = 0.05
_IK_ROTATION_SCALE_RAD = 0.25


def _task_description(activity_name: str) -> str:
    with _TASK_FILE.open(encoding="utf-8") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry["task_name"] == activity_name:
                return str(entry["task"])
    return activity_name.replace("_", " ")


def _challenge_task_config(data_root: Path, activity_name: str) -> dict[str, Any] | None:
    """Return the official 2026 benchmark config for one activity, if present."""
    metadata = data_root / "2026-challenge-task-instances" / "metadata" / "available_tasks.yaml"
    if not metadata.is_file():
        return None
    import yaml

    with metadata.open(encoding="utf-8") as stream:
        tasks = yaml.safe_load(stream) or {}
    variants = tasks.get(activity_name)
    if not isinstance(variants, dict) or not variants:
        return None
    # The v3.9 challenge bundle exposes one canonical scene/robot start config
    # per task under integer key 0. Keep deterministic first-key fallback for
    # future bundles with additional variants.
    selected = variants.get(0, variants.get("0"))
    if selected is None:
        selected = variants[sorted(variants, key=str)[0]]
    return dict(selected) if isinstance(selected, dict) else None


def _rgb_uint8(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0 + 1e-6:
        arr = arr * 255.0
    return np.clip(arr[..., :3], 0, 255).astype(np.uint8)


def _configure_agent_cartesian_control(robot_config: dict[str, Any]) -> None:
    """Replace R1Pro joint-position arm control with bounded delta-pose IK."""
    ik_config = {
        "name": "InverseKinematicsController",
        "command_input_limits": "default",
        "command_output_limits": [
            [-_IK_POSITION_SCALE_M] * 3 + [-_IK_ROTATION_SCALE_RAD] * 3,
            [_IK_POSITION_SCALE_M] * 3 + [_IK_ROTATION_SCALE_RAD] * 3,
        ],
        "kv": 2.0,
        "mode": "pose_delta_ori",
        "smoothing_filter_size": 2,
        "workspace_pose_limiter": None,
        "joint_range_tolerance": 0.01,
    }
    robot_config["controller_config"]["arm_left"] = dict(ik_config)
    robot_config["controller_config"]["arm_right"] = dict(ik_config)


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _canonical_camera_name(name: str) -> str:
    """Map OmniGibson sensor paths to OpenETA's stable camera names."""
    lowered = name.lower()
    if "left_realsense" in lowered or "wrist_left" in lowered:
        return "wrist_left"
    if "right_realsense" in lowered or "wrist_right" in lowered:
        return "wrist_right"
    if "zed" in lowered or "external_sensor" in lowered:
        return "zed_head"
    return lowered.rsplit("/", 1)[-1]


class BehaviorDirectEnv(gym.Env):
    """Official OmniGibson single-environment BEHAVIOR task adapter."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        activity_name: str,
        *,
        seed: int = 0,
        robot: str = "R1Pro",
        render_mode: str | None = "rgb_array",
        image_width: int = 128,
        image_height: int = 128,
        online_object_sampling: bool = False,
        scene_model: str | None = None,
        max_episode_steps: int = 1000,
        cartesian_control: bool = True,
        **_: Any,
    ) -> None:
        super().__init__()
        if robot.lower().replace("_", "") not in {"r1pro", "r1"}:
            raise ValueError(
                "OpenETA's BEHAVIOR v3.9 adapter currently supports the official "
                f"R1Pro config, got robot={robot!r}."
            )

        import torch
        import yaml
        import omnigibson as og
        from omnigibson.macros import gm

        gm.HEADLESS = render_mode != "human"
        gm.RENDER_VIEWER_CAMERA = render_mode == "human"
        gm.ENABLE_OBJECT_STATES = True
        gm.USE_GPU_DYNAMICS = False
        gm.USE_NUMPY_CONTROLLER_BACKEND = True

        config_path = Path(og.example_config_path) / "r1pro_behavior.yaml"
        with config_path.open(encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)

        challenge_cfg = _challenge_task_config(Path(gm.DATA_PATH), activity_name)
        if challenge_cfg is not None:
            # Match the official evaluator's full-scene path. The example YAML
            # only loads living_room + kitchen, which is invalid for most of
            # the 100 challenge tasks.
            cfg["scene"]["load_room_types"] = None
            cfg["scene"]["load_room_instances"] = None
            if scene_model is None:
                scene_model = challenge_cfg.get("scene_model")
            if "robot_start_position" in challenge_cfg:
                cfg["robots"][0]["position"] = challenge_cfg["robot_start_position"]
            if "robot_start_orientation" in challenge_cfg:
                cfg["robots"][0]["orientation"] = challenge_cfg["robot_start_orientation"]

        cfg["env"]["automatic_reset"] = False
        cfg["env"]["flatten_action_space"] = True
        cfg["env"]["external_sensors"][0]["sensor_kwargs"].update(
            image_height=int(image_height), image_width=int(image_width)
        )
        cfg["robots"][0]["sensor_config"]["VisionSensor"]["sensor_kwargs"].update(
            image_height=int(image_height), image_width=int(image_width)
        )
        cfg["task"].update(
            activity_name=activity_name,
            activity_definition_id=0,
            activity_instance_id=int(seed),
            online_object_sampling=bool(online_object_sampling),
            use_presampled_robot_pose=not bool(online_object_sampling),
        )
        cfg["task"]["termination_config"]["max_steps"] = int(max_episode_steps)
        if scene_model is not None:
            cfg["scene"]["scene_model"] = scene_model
        if cartesian_control:
            _configure_agent_cartesian_control(cfg["robots"][0])

        np.random.seed(seed)
        torch.manual_seed(seed)
        self._og = og
        self._env = og.Environment(configs=cfg)
        self._task_description = _task_description(activity_name)
        self.activity_name = activity_name
        self.render_mode = render_mode
        backend_action_space = self._env.action_space
        self._backend_action_is_batched = (
            len(backend_action_space.shape) == 2 and backend_action_space.shape[0] == 1
        )
        if self._backend_action_is_batched:
            self.action_space = gym.spaces.Box(
                low=np.asarray(backend_action_space.low[0]),
                high=np.asarray(backend_action_space.high[0]),
                dtype=backend_action_space.dtype,
            )
        else:
            self.action_space = backend_action_space
        self.observation_space = gym.spaces.Dict({})
        self._last_render: np.ndarray | None = None
        self._closed = False
        self._step_index = 0

    @property
    def openeta_control_spec(self) -> dict[str, Any]:
        """Describe R1Pro controller slots for the simulator-side MCP codec."""
        robot = self._env.robots[0]
        arm_names = tuple(robot.arm_names)
        arm = "right" if "right" in arm_names else robot.default_arm
        arm_indices = _as_numpy(robot.arm_action_idx[arm]).astype(int).reshape(-1).tolist()
        gripper_indices = (
            _as_numpy(robot.gripper_action_idx[arm]).astype(int).reshape(-1).tolist()
        )
        return {
            "schema_version": "openeta.sim_control.v1",
            "cartesian_delta": {
                "supported": len(arm_indices) == 6,
                "arm": arm,
                "position_indices": arm_indices[:3],
                "rotation_indices": arm_indices[3:6],
                "command_frame": "robot_base",
                "position_scale_m": _IK_POSITION_SCALE_M,
                "rotation_scale_rad": _IK_ROTATION_SCALE_RAD,
            },
            "gripper": {
                "supported": bool(gripper_indices),
                "arm": arm,
                "indices": gripper_indices,
                "open_value": 1.0,
                "close_value": -1.0,
            },
        }

    @staticmethod
    def _flatten_sensor_obs(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert nested R1Pro observations to OpenETA's compact raw form."""
        sensors: list[tuple[str, np.ndarray, np.ndarray | None]] = []
        proprio: np.ndarray | None = None

        def visit(value: Any, path: str = "") -> None:
            nonlocal proprio
            if not isinstance(value, dict):
                return
            for key, item in value.items():
                name = f"{path}/{key}" if path else str(key)
                if isinstance(item, dict):
                    if "rgb" in item:
                        depth_value = item.get("depth_linear", item.get("depth"))
                        depth = None
                        if depth_value is not None:
                            depth = _as_numpy(depth_value).astype(np.float32).squeeze()
                        sensors.append((name.lower(), _rgb_uint8(item["rgb"]), depth))
                    visit(item, name)
                elif "proprio" in str(key).lower():
                    proprio = _as_numpy(item).astype(np.float32).reshape(-1)

        visit(raw)
        if not sensors:
            raise RuntimeError("OmniGibson observation contained no RGB sensors")

        def find_sensor(
            needles: tuple[str, ...], fallback: int
        ) -> tuple[str, np.ndarray, np.ndarray | None]:
            for sensor in sensors:
                name = sensor[0]
                if any(needle in name for needle in needles):
                    return sensor
            return sensors[min(fallback, len(sensors) - 1)]

        _, main, main_depth = find_sensor(("zed", "external_sensor"), 0)
        _, left, left_depth = find_sensor(("left_realsense", "wrist_left"), 0)
        _, right, right_depth = find_sensor(("right_realsense", "wrist_right"), 1)
        result: dict[str, Any] = {
            "main_images": main,
            "wrist_images": np.stack([left, right], axis=0),
        }
        if main_depth is not None:
            result["main_depth"] = main_depth
        if left_depth is not None and right_depth is not None:
            result["wrist_depths"] = np.stack([left_depth, right_depth], axis=0)
        if proprio is not None:
            result["states"] = proprio
        return result

    def _structured_proprio(self) -> dict[str, Any]:
        """Read public OmniGibson robot state without changing agent schemas."""
        robot = self._env.robots[0]
        joint_positions = _as_numpy(robot.get_joint_positions()).astype(np.float32).reshape(-1)
        joint_velocities = _as_numpy(robot.get_joint_velocities()).astype(np.float32).reshape(-1)
        base_pos, base_quat = robot.get_position_orientation()

        arms: dict[str, Any] = {}
        for arm in tuple(getattr(robot, "arm_names", ())):
            eef_pos, eef_quat = robot.get_eef_pose(arm)
            indices = _as_numpy(robot.gripper_control_idx[arm]).astype(int).reshape(-1)
            gripper_qpos = joint_positions[indices]
            arms[str(arm)] = {
                "ee_pose": np.concatenate(
                    [_as_numpy(eef_pos).reshape(-1)[:3], _as_numpy(eef_quat).reshape(-1)[:4]]
                ).astype(np.float32),
                "gripper_joint_positions": gripper_qpos.astype(np.float32),
            }

        arm_names = tuple(getattr(robot, "arm_names", ()))
        primary_arm = "right" if "right" in arm_names else getattr(robot, "default_arm", None)
        if primary_arm not in arms and arm_names:
            primary_arm = arm_names[0]

        proprio: dict[str, Any] = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "base_pose": {
                "xyz": _as_numpy(base_pos).reshape(-1)[:3].tolist(),
                "quat_xyzw": _as_numpy(base_quat).reshape(-1)[:4].tolist(),
            },
            "metadata": {
                "primary_arm": primary_arm,
                "arms": arms,
                "pose_frame": "world",
                "quaternion_order": "xyzw",
            },
        }
        if primary_arm in arms:
            proprio["ee_pose"] = arms[primary_arm]["ee_pose"]
            indices = _as_numpy(robot.gripper_control_idx[primary_arm]).astype(int).reshape(-1)
            lower = _as_numpy(robot.joint_lower_limits).reshape(-1)[indices]
            upper = _as_numpy(robot.joint_upper_limits).reshape(-1)[indices]
            span = upper - lower
            valid = np.abs(span) > 1e-8
            if np.any(valid):
                opening = (joint_positions[indices][valid] - lower[valid]) / span[valid]
                open_fraction = float(np.clip(opening, 0.0, 1.0).mean())
                proprio["gripper_open"] = open_fraction
                proprio["gripper_state"] = {
                    "open": open_fraction > 0.5,
                    "open_fraction": open_fraction,
                }
        return proprio

    def _camera_parameters(self) -> dict[str, dict[str, Any]]:
        """Return calibrated camera packets keyed by stable OpenETA camera name."""
        robot = self._env.robots[0]
        sensor_groups = (getattr(self._env, "external_sensors", None), robot.sensors)
        parameters: dict[str, dict[str, Any]] = {}
        for group in sensor_groups:
            for sensor_name, sensor in (group or {}).items():
                if not hasattr(sensor, "intrinsic_matrix"):
                    continue
                matrix = _as_numpy(sensor.intrinsic_matrix).astype(np.float64)
                position, quaternion = sensor.get_position_orientation()
                height = int(getattr(sensor, "image_height", 0))
                width = int(getattr(sensor, "image_width", 0))
                parameters[_canonical_camera_name(str(sensor_name))] = {
                    "intrinsics": {
                        "fx": float(matrix[0, 0]),
                        "fy": float(matrix[1, 1]),
                        "cx": float(matrix[0, 2]),
                        "cy": float(matrix[1, 2]),
                        "width": width,
                        "height": height,
                    },
                    "extrinsics": {
                        "pos": _as_numpy(position).reshape(-1)[:3].tolist(),
                        "quat_xyzw": _as_numpy(quaternion).reshape(-1)[:4].tolist(),
                        "frame_transform": "camera_to_world",
                        "camera_frame": "omnigibson_usd",
                    },
                }
        return parameters

    def _annotate_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Attach sim-only structured fields consumed by UnifiedEnv."""
        warnings: list[str] = []
        try:
            obs["_openeta_proprio"] = self._structured_proprio()
        except Exception as exc:
            warnings.append(f"structured_proprio_unavailable: {type(exc).__name__}: {exc}")
        try:
            obs["_openeta_camera_params"] = self._camera_parameters()
        except Exception as exc:
            warnings.append(f"camera_calibration_unavailable: {type(exc).__name__}: {exc}")
        metadata = {
            "benchmark": "behavior-1k",
            "activity_name": self.activity_name,
            "step_index": self._step_index,
            "depth_unit": "metres",
        }
        if warnings:
            metadata["normalization_warnings"] = warnings
        obs["_openeta_metadata"] = metadata
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        raw, info = self._env.reset()
        self._step_index = 0
        obs = self._annotate_observation(self._flatten_sensor_obs(raw))
        self._last_render = obs["main_images"]
        return obs, info

    def step(self, action):
        backend_action = np.asarray(action, dtype=self.action_space.dtype)
        if self._backend_action_is_batched and backend_action.ndim == 1:
            backend_action = backend_action[None, :]
        raw, reward, terminated, truncated, info = self._env.step(backend_action)
        self._step_index += 1
        obs = self._annotate_observation(self._flatten_sensor_obs(raw))
        self._last_render = obs["main_images"]
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        rendered = self._env.render()
        if rendered is not None:
            self._last_render = _rgb_uint8(rendered)
        return self._last_render

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Environment.close() is intentionally a no-op in v3.9. Official
        # examples finish with og.shutdown(); each BEHAVIOR worker therefore
        # owns exactly one live env and is retired by BenchWorkerManager after
        # this call.
        self._env.close()
        self._og.shutdown()
