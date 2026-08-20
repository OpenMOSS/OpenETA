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

from sim.camera_conventions import (
    normalise_camera_to_world_opencv,
    quaternion_xyzw_to_rotation_matrix,
)
from sim.envs.behavior.seeding import seed_behavior_reset_rngs


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
        "mode": "pose_delta_ori",
        "smoothing_filter_size": 2,
        "workspace_pose_limiter": None,
        # `kv` and `joint_range_tolerance` were passed here but are not in
        # InverseKinematicsController's signature in OmniGibson 3.9.1, and
        # create_controller binds kwargs strictly -- so every BEHAVIOR env raised
        # TypeError before reaching a single step.  Dropped rather than remapped:
        # the nearest surviving gains (pos_damping_ratio, vel_kp) are not the same
        # quantity as the old kv, so inventing a value would silently change the
        # controller's behaviour instead of restoring it.
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


def _require_rgbd_modalities(sensor_config: dict[str, Any]) -> None:
    """Ensure VisionSensor emits RGB plus pinhole optical-Z depth."""

    # OmniGibson's create_sensor contract expects ``modalities`` alongside
    # ``sensor_kwargs``. Putting it inside ``sensor_kwargs`` would pass the
    # argument to VisionSensor twice:
    # create_sensor(..., modalities=..., **sensor_kwargs).
    configured = sensor_config.get("modalities", ["rgb"])
    if configured == "all":
        return
    if isinstance(configured, str):
        modalities = [configured]
    else:
        modalities = list(configured or [])
    for required in ("rgb", "depth_linear"):
        if required not in modalities:
            modalities.append(required)
    sensor_config["modalities"] = modalities


def _require_task_instance(data_root: Path, scene_model: str | None,
                           activity_name: str, instance_id: int) -> None:
    """Fail early, and by name, when the requested task instance is absent.

    ``seed`` becomes ``activity_instance_id``, and an activity ships only the
    instances that were actually sampled -- ``picking_up_trash`` has exactly one
    (id 0) despite 300+ sibling files, because the rest are ``-tro_state``
    variants rather than loadable templates.  OmniGibson's
    ``get_task_instance_path`` returns ``None`` for a missing one instead of
    raising, so the failure surfaces three frames later as
    ``os.path.join(None)`` -> "expected str, bytes or os.PathLike object, not
    NoneType", roughly 12 s into a boot and naming neither the activity nor the
    id.  That message cost hours to trace back to "seed 1 does not exist", and
    it looks like a path-configuration bug rather than a bad argument.

    Skipped when the scene is unknown (the caller then relies on the default
    "best" scene, which does not go through the instance directory at all).
    """
    if not scene_model:
        return
    json_dir = (data_root / "2026-challenge-task-instances" / "scenes"
                / scene_model / "json")
    if not json_dir.is_dir():
        return  # No instance directory: leave the diagnosis to OmniGibson.

    prefix = f"{scene_model}_task_{activity_name}_0_"
    wanted = json_dir / f"{prefix}{instance_id}_template.json"
    if wanted.is_file():
        return

    available = sorted(
        int(p.stem[len(prefix):-len("_template")])
        for p in json_dir.glob(f"{prefix}*_template.json")
        if p.stem[len(prefix):-len("_template")].isdigit()
    )
    raise ValueError(
        f"BEHAVIOR activity {activity_name!r} has no instance "
        f"{instance_id} (seed={instance_id}) in scene {scene_model!r}. "
        f"Available instance ids: {available or 'none'}. "
        f"Looked for {wanted.name}."
    )


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
        external_sensor_config = cfg["env"]["external_sensors"][0]
        external_sensor_kwargs = external_sensor_config["sensor_kwargs"]
        external_sensor_kwargs.update(
            image_height=int(image_height), image_width=int(image_width)
        )
        _require_rgbd_modalities(external_sensor_config)
        robot_sensor_config = cfg["robots"][0]["sensor_config"]["VisionSensor"]
        robot_sensor_kwargs = robot_sensor_config["sensor_kwargs"]
        robot_sensor_kwargs.update(
            image_height=int(image_height), image_width=int(image_width)
        )
        _require_rgbd_modalities(robot_sensor_config)
        _require_task_instance(Path(gm.DATA_PATH), scene_model, activity_name,
                               int(seed))
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

        seed_behavior_reset_rngs(seed)
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

        def slots(mapping, key) -> list[int]:
            try:
                return _as_numpy(mapping[key]).astype(int).reshape(-1).tolist()
            except Exception:
                return []

        arm_indices = slots(robot.arm_action_idx, arm)
        gripper_indices = slots(robot.gripper_action_idx, arm)

        # Per-arm slots for every arm, not just the default one.  Publishing a
        # single arm left the other one unreachable through the MCP surface even
        # though the robot is bimanual.
        per_arm: dict[str, Any] = {}
        for name in arm_names:
            a_idx = slots(robot.arm_action_idx, name)
            if len(a_idx) != 6:
                # Not IK-shaped; declaring it Cartesian would drive the wrong
                # actuators, so omit it and let the codec fail closed.
                continue
            per_arm[str(name)] = {
                "position_indices": a_idx[:3],
                "rotation_indices": a_idx[3:6],
                "gripper_indices": slots(robot.gripper_action_idx, name),
            }

        # Derived defensively: bench_worker reads this property via
        # getattr(env, "openeta_control_spec", {}), so an AttributeError raised
        # in here is swallowed into an empty dict -- which silently strips the
        # whole layout and makes every move_to fail closed.  A robot missing one
        # optional attribute must degrade to a partial spec, never to no spec.
        try:
            group_slots = {str(k): _as_numpy(v).astype(int).reshape(-1).tolist()
                           for k, v in (robot.controller_action_idx or {}).items()}
        except Exception:
            group_slots = {}

        action_dim = getattr(robot, "action_dim", None)
        if action_dim is None:
            covered = [i for idxs in group_slots.values() for i in idxs]
            action_dim = max(covered) + 1 if covered else 0

        spec: dict[str, Any] = {
            "schema_version": "openeta.sim_control.v1",
            # Morphology, read off the live robot.  The interface tier is picked
            # from these fields rather than from the backend name, because
            # R1Pro's action_dim is 21 under these IK overrides but 23 under the
            # raw r1pro_behavior.yaml joint controllers.
            "profile": "mobile_manipulation",
            "arms": [str(a) for a in arm_names],
            "default_arm": str(robot.default_arm),
            "arm_required": len(arm_names) > 1,
            "action_dim": int(action_dim),
            "cartesian_delta": {
                "supported": len(arm_indices) == 6,
                "arm": arm,
                "position_indices": arm_indices[:3],
                "rotation_indices": arm_indices[3:6],
                "command_frame": "robot_base",
                "position_scale_m": _IK_POSITION_SCALE_M,
                "rotation_scale_rad": _IK_ROTATION_SCALE_RAD,
                "per_arm": per_arm,
            },
            "gripper": {
                "supported": bool(gripper_indices),
                "arm": arm,
                "indices": gripper_indices,
                "open_value": 1.0,
                "close_value": -1.0,
            },
        }

        # Base and trunk: DOF a fixed-base Cartesian call has nowhere to put.
        # Keyed off controller_action_idx (resolved above) so a config lacking
        # them simply omits them, and no drive_base/set_trunk gets advertised.
        if group_slots.get("base"):
            spec["base"] = {
                "supported": True,
                "indices": group_slots["base"],
                "type": "holonomic",
                # HolonomicBaseJointController is velocity-mode: these are rates,
                # not positions, so motion stops when commands stop.
                "command": "velocity",
                "dims": ["vx", "vy", "wz"],
            }
        if group_slots.get("trunk"):
            spec["trunk"] = {
                "supported": True,
                "indices": group_slots["trunk"],
                "command": "position",
            }
            # Names + limits are what make "hold the trunk where it is"
            # expressible by a caller that only sees the action vector.
            #
            # JointController is position-mode with use_delta_commands=False,
            # and Controller._preprocess_command scales [-1,1] onto the joint
            # limits.  So a 0.0 in a trunk slot is *not* "no command" -- it
            # resolves to (lower+upper)/2 and drives the trunk to mid-range.
            # Holding position therefore means sending the current angle
            # re-normalised, which needs the same limits the controller used.
            #
            # Published here rather than fetched per-step because the alternative
            # is an extra RPC on every Cartesian substep, and these are constant
            # for the life of the robot.
            try:
                trunk_names = [str(n) for n in robot.trunk_joint_names]
            except Exception:
                # Derive from dof_idx when the robot exposes no trunk_joint_names.
                try:
                    j_names = list(robot.joints.keys())
                    ctrl = robot.controllers["trunk"]
                    trunk_names = [str(j_names[int(i)]) for i in ctrl.dof_idx]
                except Exception:
                    trunk_names = []
            lower: list[float] = []
            upper: list[float] = []
            for n in trunk_names:
                try:
                    j = robot.joints[n]
                    lower.append(float(j.lower_limit))
                    upper.append(float(j.upper_limit))
                except Exception:
                    lower, upper = [], []
                    break
            # All three must agree in length or a consumer cannot pair them, and
            # a half-published mapping is worse than none: it would let the
            # caller believe it can hold position while writing the wrong slot.
            if trunk_names and len(trunk_names) == len(group_slots["trunk"]) == len(lower):
                spec["trunk"]["joint_names"] = trunk_names
                spec["trunk"]["limits_lower"] = lower
                spec["trunk"]["limits_upper"] = upper
        spec["groups"] = group_slots
        return spec

    @staticmethod
    def _flatten_sensor_obs(raw: dict[str, Any]) -> dict[str, Any]:
        """Convert nested R1Pro observations to OpenETA's compact raw form."""
        sensors: list[tuple[str, str, np.ndarray, np.ndarray | None]] = []
        proprio: np.ndarray | None = None

        def visit(value: Any, path: str = "") -> None:
            nonlocal proprio
            if not isinstance(value, dict):
                return
            for key, item in value.items():
                name = f"{path}/{key}" if path else str(key)
                if isinstance(item, dict):
                    if "rgb" in item:
                        # OmniGibson `depth` is radial distance-to-camera.
                        # Pinhole deprojection and AnyGrasp require optical-Z,
                        # exposed by `depth_linear` (distance-to-image-plane).
                        depth_value = item.get("depth_linear")
                        depth = None
                        if depth_value is not None:
                            depth = _as_numpy(depth_value).astype(np.float32).squeeze()
                        sensors.append(
                            (
                                name,
                                str(key),
                                _rgb_uint8(item["rgb"]),
                                depth,
                            )
                        )
                    visit(item, name)
                elif "proprio" in str(key).lower():
                    proprio = _as_numpy(item).astype(np.float32).reshape(-1)

        visit(raw)
        if not sensors:
            raise RuntimeError("OmniGibson observation contained no RGB sensors")

        def find_sensor(
            needles: tuple[str, ...],
            *,
            semantic_name: str,
        ) -> tuple[str, str, np.ndarray, np.ndarray | None]:
            for needle in needles:
                matches = [
                    sensor
                    for sensor in sensors
                    if needle in sensor[0].lower()
                ]
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    raise RuntimeError(
                        f"OmniGibson observation has ambiguous {semantic_name} "
                        f"RGB sensors matching {needle!r}: "
                        f"{[sensor[0] for sensor in matches]}"
                    )
            raise RuntimeError(
                f"OmniGibson observation is missing the required {semantic_name} "
                f"RGB sensor; expected one name containing one of {needles}"
            )

        main_name, main_sensor_name, main, main_depth = find_sensor(
            ("zed", "external_sensor"),
            semantic_name="scene-primary",
        )
        left_name, left_sensor_name, left, left_depth = find_sensor(
            ("left_realsense", "wrist_left"),
            semantic_name="left-wrist",
        )
        right_name, right_sensor_name, right, right_depth = find_sensor(
            ("right_realsense", "wrist_right"),
            semantic_name="right-wrist",
        )
        if len({main_name, left_name, right_name}) != 3:
            raise RuntimeError(
                "OmniGibson scene and wrist camera roles must resolve to three "
                "distinct RGB sensors"
            )
        missing_linear_depth = [
            name
            for name, depth in (
                (main_name, main_depth),
                (left_name, left_depth),
                (right_name, right_depth),
            )
            if depth is None
        ]
        if missing_linear_depth:
            raise RuntimeError(
                "OmniGibson RGB-D contract requires depth_linear "
                "(distance_to_image_plane) for every selected camera; missing "
                f"{missing_linear_depth}"
            )
        for sensor_name, rgb, depth in (
            (main_name, main, main_depth),
            (left_name, left, left_depth),
            (right_name, right, right_depth),
        ):
            if depth.shape != rgb.shape[:2]:
                raise RuntimeError(
                    "OmniGibson RGB and depth_linear dimensions differ for "
                    f"{sensor_name!r}: rgb={rgb.shape[:2]}, depth={depth.shape}"
                )
        result: dict[str, Any] = {
            "main_images": main,
            "wrist_images": np.stack([left, right], axis=0),
            "main_depth": main_depth,
            "wrist_depths": np.stack([left_depth, right_depth], axis=0),
            "_openeta_camera_sources": {
                "zed_head": {
                    "observation_path": main_name,
                    "sensor_name": main_sensor_name,
                },
                "wrist_left": {
                    "observation_path": left_name,
                    "sensor_name": left_sensor_name,
                },
                "wrist_right": {
                    "observation_path": right_name,
                    "sensor_name": right_sensor_name,
                },
            },
        }
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
                ).astype(np.float32).tolist(),
                "gripper_joint_positions": gripper_qpos.astype(np.float32).tolist(),
            }

        arm_names = tuple(getattr(robot, "arm_names", ()))
        primary_arm = "right" if "right" in arm_names else getattr(robot, "default_arm", None)
        if primary_arm not in arms and arm_names:
            primary_arm = arm_names[0]

        proprio: dict[str, Any] = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            # Names in the same order as joint_positions.  R1Pro's vector is
            # base(0-5), torso(6-9), arms INTERLEAVED left/right (10-23),
            # fingers(24-27) -- so left_arm_joint2 is at index 12, not 11.
            # Any consumer that needs a subset (cuRobo wants 4 torso + 7 + 7 in
            # its own order) must permute by name; a positional slice silently
            # puts wheel and right-arm angles into left-arm slots and yields a
            # confident wrong answer.
            "joint_names": [str(n) for n in (getattr(robot, "joints", None) or {})],
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
                    # Canonical Agent-facing continuous aperture. Keep the
                    # historical open_fraction alias for experiment logs and
                    # downstream consumers that have not migrated yet.
                    "openness": open_fraction,
                    "open_fraction": open_fraction,
                }
        return proprio

    def _camera_parameters(
        self,
        camera_sources: dict[str, dict[str, str]],
    ) -> dict[str, dict[str, Any]]:
        """Return calibration for the exact sensors selected in this RGB-D frame."""
        robot = self._env.robots[0]
        sensor_groups = (getattr(self._env, "external_sensors", None), robot.sensors)
        available_sensors: list[dict[str, Any]] = []
        seen_sensor_ids: set[int] = set()
        for group in sensor_groups:
            for registry_key, sensor in (group or {}).items():
                sensor_id = id(sensor)
                if sensor_id in seen_sensor_ids:
                    continue
                seen_sensor_ids.add(sensor_id)
                sensor_name = str(getattr(sensor, "name", "") or "")
                prim_path = str(getattr(sensor, "prim_path", "") or "")
                identities = {
                    identity
                    for identity in (str(registry_key), sensor_name, prim_path)
                    if identity
                }
                available_sensors.append(
                    {
                        "registry_key": str(registry_key),
                        "sensor_name": sensor_name,
                        "prim_path": prim_path,
                        "identities": identities,
                        "sensor": sensor,
                    }
                )

        parameters: dict[str, dict[str, Any]] = {}
        for camera_name, source in camera_sources.items():
            if not isinstance(source, dict):
                raise RuntimeError(
                    f"OmniGibson camera source for {camera_name!r} must be a mapping"
                )
            observation_path = str(source.get("observation_path") or "")
            source_sensor_name = str(source.get("sensor_name") or "")
            source_role_name = (
                source_sensor_name
                or observation_path.rsplit("/", 1)[-1]
            )
            if _canonical_camera_name(source_role_name) != camera_name:
                raise RuntimeError(
                    f"OmniGibson selected source {source_role_name!r} does not "
                    f"match OpenETA camera role {camera_name!r}"
                )
            source_identities = {
                identity
                for identity in (observation_path, source_sensor_name)
                if identity
            }
            matches = [
                candidate
                for candidate in available_sensors
                if source_identities.intersection(candidate["identities"])
            ]
            if len(matches) != 1:
                match_kind = "no" if not matches else "multiple"
                raise RuntimeError(
                    f"OmniGibson selected {camera_name!r} RGB-D source "
                    f"{source_sensor_name!r} has {match_kind} exact calibration "
                    "sensor match"
                )

            selected = matches[0]
            sensor = selected["sensor"]
            try:
                matrix = _as_numpy(sensor.intrinsic_matrix).astype(np.float64)
            except AttributeError as exc:
                raise RuntimeError(
                    f"OmniGibson selected {camera_name!r} source is not a "
                    "calibrated vision sensor"
                ) from exc
            position, quaternion = sensor.get_position_orientation()
            height = int(getattr(sensor, "image_height", 0))
            width = int(getattr(sensor, "image_width", 0))
            quaternion_xyzw = _as_numpy(quaternion).reshape(-1)[:4]
            parameters[camera_name] = {
                "intrinsics": {
                    "fx": float(matrix[0, 0]),
                    "fy": float(matrix[1, 1]),
                    "cx": float(matrix[0, 2]),
                    "cy": float(matrix[1, 2]),
                    "width": width,
                    "height": height,
                    "depth_unit": "meter",
                },
                "extrinsics": normalise_camera_to_world_opencv(
                    position_xyz=_as_numpy(position).reshape(-1)[:3],
                    rotation_camera_to_world=quaternion_xyzw_to_rotation_matrix(
                        quaternion_xyzw
                    ),
                    source_camera_frame="omnigibson_usd",
                    normalized_from="omnigibson_usd",
                ),
                "calibration_source": {
                    "observation_path": observation_path,
                    "registry_key": selected["registry_key"],
                    "sensor_name": selected["sensor_name"],
                    "prim_path": selected["prim_path"],
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
        camera_sources = obs.get("_openeta_camera_sources")
        if not isinstance(camera_sources, dict) or not camera_sources:
            raise RuntimeError(
                "OmniGibson RGB-D observation is missing exact camera source identities"
            )
        camera_parameters = self._camera_parameters(camera_sources)
        required_cameras: set[str] = set()
        if obs.get("main_images") is not None:
            required_cameras.add("zed_head")
        wrist_images = obs.get("wrist_images")
        if wrist_images is not None:
            wrist_array = _as_numpy(wrist_images)
            if wrist_array.ndim == 4 and wrist_array.shape[0] >= 2:
                required_cameras.update({"wrist_left", "wrist_right"})
            elif wrist_array.ndim == 3:
                required_cameras.add("wrist_left")
        missing_calibration = sorted(required_cameras - set(camera_parameters))
        if missing_calibration:
            raise RuntimeError(
                "OmniGibson observation is missing calibrated camera packets "
                f"for {missing_calibration}"
            )
        image_shapes: dict[str, tuple[int, int]] = {}
        main_images = obs.get("main_images")
        if main_images is not None:
            main_array = _as_numpy(main_images)
            image_shapes["zed_head"] = tuple(int(v) for v in main_array.shape[:2])
        if wrist_images is not None:
            wrist_array = _as_numpy(wrist_images)
            if wrist_array.ndim == 4:
                if wrist_array.shape[0] >= 1:
                    image_shapes["wrist_left"] = tuple(
                        int(v) for v in wrist_array[0].shape[:2]
                    )
                if wrist_array.shape[0] >= 2:
                    image_shapes["wrist_right"] = tuple(
                        int(v) for v in wrist_array[1].shape[:2]
                    )
            elif wrist_array.ndim == 3:
                image_shapes["wrist_left"] = tuple(
                    int(v) for v in wrist_array.shape[:2]
                )
        for camera_name, (height, width) in image_shapes.items():
            intrinsics = camera_parameters[camera_name].get("intrinsics", {})
            if (
                int(intrinsics.get("height", -1)) != height
                or int(intrinsics.get("width", -1)) != width
            ):
                raise RuntimeError(
                    "OmniGibson calibration dimensions do not match the selected "
                    f"{camera_name!r} RGB frame"
                )
        obs["_openeta_camera_params"] = camera_parameters
        metadata = {
            "benchmark": "behavior-1k",
            "activity_name": self.activity_name,
            "step_index": self._step_index,
            "depth_unit": "meter",
        }
        if warnings:
            metadata["normalization_warnings"] = warnings
        obs["_openeta_metadata"] = metadata
        return obs

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            seed_behavior_reset_rngs(seed)
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
