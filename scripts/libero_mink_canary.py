#!/usr/bin/env python3
"""Paired LIBERO endpoint-accuracy canary for OSC and Mink.

This is intentionally an experiment runner, not a production controller.  It
replays the exact same MuJoCo initial state into robosuite's OSC_POSE and
JOINT_VELOCITY controllers.  The latter receives joint velocities from Mink.
Collision avoidance is deliberately excluded so endpoint tracking can be
measured before contact policy and scene-geometry questions are introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class TargetCase:
    name: str
    offset_xyz: tuple[float, float, float] | None = None
    absolute_xyz: tuple[float, float, float] | None = None
    world_delta_rpy_deg: tuple[float, float, float] | None = None
    absolute_rpy_deg: tuple[float, float, float] | None = None

    @property
    def controls_orientation(self) -> bool:
        return self.world_delta_rpy_deg is not None or self.absolute_rpy_deg is not None


DEFAULT_CASES = (
    TargetCase("translation_short", offset_xyz=(0.06, -0.04, 0.05)),
    TargetCase("translation_long", offset_xyz=(0.16, -0.12, 0.08)),
    TargetCase(
        "coupled_yaw_30",
        offset_xyz=(0.10, -0.08, 0.06),
        world_delta_rpy_deg=(0.0, 0.0, 30.0),
    ),
    TargetCase(
        "coupled_pitch_25_yaw_35",
        offset_xyz=(0.10, 0.07, 0.08),
        world_delta_rpy_deg=(0.0, 25.0, 35.0),
    ),
    TargetCase(
        "recorded_grasp_pose",
        absolute_xyz=(0.096651111613, -0.182900056355, 0.287612714656),
        absolute_rpy_deg=(180.0, 0.0, 0.0),
    ),
)

START_PROFILES = {
    "nominal": (0.0, 0.0, 0.0),
    "left_high": (0.08, 0.08, 0.10),
    "right_high": (0.08, -0.08, 0.10),
}


def _load_libero(libero_dir: Path):
    sys.path.insert(0, str(libero_dir))
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs.env_wrapper import ControlEnv

    return get_libero_path, get_benchmark, ControlEnv


def _make_env(ControlEnv, bddl_path: str, controller: str):
    return ControlEnv(
        bddl_file_name=bddl_path,
        controller=controller,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        camera_depths=False,
        control_freq=20,
        horizon=1000,
    )


def _eef_pose(env) -> tuple[np.ndarray, np.ndarray]:
    robot = env.env.robots[0]
    site_id = env.sim.model.site_name2id(robot.gripper.important_sites["grip_site"])
    body_quat_wxyz = np.asarray(
        env.sim.data.get_body_xquat(robot.robot_model.eef_name), dtype=np.float64
    )
    return (
        np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy(),
        body_quat_wxyz[[1, 2, 3, 0]].copy(),
    )


def _site_rotation(env) -> np.ndarray:
    robot = env.env.robots[0]
    site_id = env.sim.model.site_name2id(robot.gripper.important_sites["grip_site"])
    return np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()


def _body_rotation(env) -> np.ndarray:
    robot = env.env.robots[0]
    body_id = env.sim.model.body_name2id(robot.robot_model.eef_name)
    return np.asarray(env.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).copy()


def _target_from_case(
    case: TargetCase, start_xyz: np.ndarray, start_quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray | None]:
    if case.absolute_xyz is not None:
        target_xyz = np.asarray(case.absolute_xyz, dtype=np.float64)
    else:
        target_xyz = start_xyz + np.asarray(case.offset_xyz, dtype=np.float64)

    target_quat: np.ndarray | None = None
    if case.absolute_rpy_deg is not None:
        target_quat = Rotation.from_euler(
            "xyz", case.absolute_rpy_deg, degrees=True
        ).as_quat()
    elif case.world_delta_rpy_deg is not None:
        delta = Rotation.from_euler("xyz", case.world_delta_rpy_deg, degrees=True)
        target_quat = (delta * Rotation.from_quat(start_quat_xyzw)).as_quat()
    return target_xyz, target_quat


def _angular_error_rad(actual_xyzw: np.ndarray, target_xyzw: np.ndarray) -> float:
    relative = Rotation.from_quat(target_xyzw) * Rotation.from_quat(actual_xyzw).inv()
    return float(np.linalg.norm(relative.as_rotvec()))


def _endpoint_metrics(
    env,
    target_xyz: np.ndarray,
    target_quat: np.ndarray | None,
    *,
    position_tolerance: float,
    orientation_tolerance: float,
) -> dict[str, Any]:
    end_xyz, end_quat = _eef_pose(env)
    residual = target_xyz - end_xyz
    orientation_error = (
        _angular_error_rad(end_quat, target_quat) if target_quat is not None else None
    )
    reached = bool(
        np.max(np.abs(residual)) < position_tolerance
        and (orientation_error is None or orientation_error < orientation_tolerance)
    )
    return {
        "end_xyz": end_xyz.tolist(),
        "end_quat_xyzw": end_quat.tolist(),
        "position_error_m": float(np.linalg.norm(residual)),
        "max_axis_position_error_m": float(np.max(np.abs(residual))),
        "orientation_error_rad": orientation_error,
        "orientation_error_deg": (
            math.degrees(orientation_error) if orientation_error is not None else None
        ),
        "reached_target": reached,
    }


def _restore_state(env, initial_state: np.ndarray) -> None:
    env.reset()
    env.set_init_state(initial_state)
    robot = env.env.robots[0]
    robot.controller.update(force=True)
    robot.controller.reset_goal()


def _run_osc(
    env,
    target_xyz: np.ndarray,
    target_quat: np.ndarray | None,
    *,
    max_steps: int,
    position_tolerance: float,
    orientation_tolerance: float,
) -> tuple[int, bool]:
    position_scale = 0.05
    orientation_scale = 0.5
    recheck_every = 3 if target_quat is not None else 1
    steps = 0
    terminated = False

    while steps < max_steps:
        current_xyz, current_quat = _eef_pose(env)
        position_error = target_xyz - current_xyz
        position_ok = bool(np.max(np.abs(position_error)) < position_tolerance)
        rotation_command = np.zeros(3, dtype=np.float64)
        orientation_ok = target_quat is None
        if target_quat is not None:
            delta = Rotation.from_quat(target_quat) * Rotation.from_quat(current_quat).inv()
            rotvec = delta.as_rotvec()
            orientation_ok = bool(np.linalg.norm(rotvec) < orientation_tolerance)
            rotation_command = np.clip(
                rotvec / (recheck_every * orientation_scale), -1.0, 1.0
            )
        if position_ok and orientation_ok:
            break

        position_command = np.clip(
            position_error / (recheck_every * position_scale), -1.0, 1.0
        )
        action = np.concatenate([position_command, rotation_command, [-1.0]])
        batch_steps = min(recheck_every, max_steps - steps)
        for _ in range(batch_steps):
            _, _, done, _ = env.step(action)
            steps += 1
            if done:
                terminated = True
                break
        if terminated:
            break
    return steps, terminated


def _run_mink(
    env,
    target_xyz: np.ndarray,
    target_quat: np.ndarray | None,
    *,
    max_steps: int,
    position_tolerance: float,
    orientation_tolerance: float,
) -> tuple[int, bool]:
    import mink

    robot = env.env.robots[0]
    model = env.sim.model._model
    configuration = mink.Configuration(model, q=env.sim.data.qpos.copy())
    eef_site = robot.gripper.important_sites["grip_site"]
    frame_task = mink.FrameTask(
        frame_name=eef_site,
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0 if target_quat is not None else 0.0,
        gain=0.8,
        lm_damping=1e-6,
    )
    posture_task = mink.PostureTask(model, cost=1e-3)
    posture_task.set_target_from_configuration(configuration)

    start_site_rotation = _site_rotation(env)
    start_body_rotation = _body_rotation(env)
    if target_quat is None:
        target_site_rotation = start_site_rotation
    else:
        target_body_rotation = Rotation.from_quat(target_quat).as_matrix()
        body_to_site_rotation = start_body_rotation.T @ start_site_rotation
        target_site_rotation = target_body_rotation @ body_to_site_rotation
    frame_task.set_target(
        mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(target_site_rotation), target_xyz
        )
    )

    velocity_by_joint = {name: np.array([0.5]) for name in robot.robot_joints}
    limits = [
        mink.ConfigurationLimit(model, gain=0.95),
        mink.VelocityLimit(model, velocities=velocity_by_joint),
    ]
    qvel_indices = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    controller_velocity_scale = np.asarray(robot.controller.output_max, dtype=np.float64)
    dt = 1.0 / float(env.env.control_freq)
    steps = 0
    terminated = False

    while steps < max_steps:
        current_xyz, current_quat = _eef_pose(env)
        position_ok = bool(
            np.max(np.abs(target_xyz - current_xyz)) < position_tolerance
        )
        orientation_ok = target_quat is None or (
            _angular_error_rad(current_quat, target_quat) < orientation_tolerance
        )
        if position_ok and orientation_ok:
            break

        configuration.update(q=env.sim.data.qpos.copy())
        velocity = mink.solve_ik(
            configuration,
            [frame_task, posture_task],
            dt,
            solver="quadprog",
            damping=1e-6,
            safety_break=False,
            limits=limits,
        )
        arm_velocity = velocity[qvel_indices]
        normalized_velocity = np.divide(
            arm_velocity,
            controller_velocity_scale,
            out=np.zeros_like(arm_velocity),
            where=controller_velocity_scale != 0.0,
        )
        action = np.concatenate([np.clip(normalized_velocity, -1.0, 1.0), [-1.0]])
        _, _, done, _ = env.step(action)
        steps += 1
        if done:
            terminated = True
            break
    return steps, terminated


def _capture_initial_states(
    ControlEnv,
    bddl_path: str,
    seeds: list[int],
    *,
    max_steps: int,
    position_tolerance: float,
    orientation_tolerance: float,
) -> list[dict[str, Any]]:
    env = _make_env(ControlEnv, bddl_path, "OSC_POSE")
    states: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            env.seed(seed)
            env.reset()
            nominal_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            nominal_xyz, _ = _eef_pose(env)
            for profile, offset in START_PROFILES.items():
                _restore_state(env, nominal_state)
                if profile != "nominal":
                    target_xyz = nominal_xyz + np.asarray(offset, dtype=np.float64)
                    _run_osc(
                        env,
                        target_xyz,
                        None,
                        max_steps=max_steps,
                        position_tolerance=position_tolerance,
                        orientation_tolerance=orientation_tolerance,
                    )
                profile_xyz, _ = _eef_pose(env)
                states.append(
                    {
                        "seed": seed,
                        "profile": profile,
                        "eef_xyz": profile_xyz.tolist(),
                        "state": np.asarray(env.get_sim_state(), dtype=np.float64).copy(),
                    }
                )
    finally:
        env.close()
    return states


def _run_backend(
    ControlEnv,
    bddl_path: str,
    backend: str,
    initial_states: list[dict[str, Any]],
    cases: tuple[TargetCase, ...],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    controller = "OSC_POSE" if backend == "osc" else "JOINT_VELOCITY"
    env = _make_env(ControlEnv, bddl_path, controller)
    results: list[dict[str, Any]] = []
    try:
        for initial in initial_states:
            seed = int(initial["seed"])
            profile = str(initial["profile"])
            initial_state = initial["state"]
            for case in cases:
                _restore_state(env, initial_state)
                start_xyz, start_quat = _eef_pose(env)
                target_xyz, target_quat = _target_from_case(case, start_xyz, start_quat)
                started = time.perf_counter()
                error = ""
                steps = 0
                terminated = False
                try:
                    if backend == "osc":
                        steps, terminated = _run_osc(
                            env,
                            target_xyz,
                            target_quat,
                            max_steps=args.max_steps,
                            position_tolerance=args.position_tolerance,
                            orientation_tolerance=args.orientation_tolerance,
                        )
                    else:
                        steps, terminated = _run_mink(
                            env,
                            target_xyz,
                            target_quat,
                            max_steps=args.max_steps,
                            position_tolerance=args.position_tolerance,
                            orientation_tolerance=args.orientation_tolerance,
                        )
                except Exception as exc:  # Keep paired evidence even if one solve fails.
                    error = f"{type(exc).__name__}: {exc}"
                metrics = _endpoint_metrics(
                    env,
                    target_xyz,
                    target_quat,
                    position_tolerance=args.position_tolerance,
                    orientation_tolerance=args.orientation_tolerance,
                )
                results.append(
                    {
                        "backend": backend,
                        "controller": controller,
                        "seed": seed,
                        "start_profile": profile,
                        "case": case.name,
                        "controls_orientation": case.controls_orientation,
                        "start_xyz": start_xyz.tolist(),
                        "start_quat_xyzw": start_quat.tolist(),
                        "target_xyz": target_xyz.tolist(),
                        "target_quat_xyzw": target_quat.tolist() if target_quat is not None else None,
                        "steps_executed": steps,
                        "terminated": terminated,
                        "elapsed_s": time.perf_counter() - started,
                        "error": error or None,
                        **metrics,
                    }
                )
    finally:
        env.close()
    return results


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for backend in sorted({row["backend"] for row in rows}):
        subset = [row for row in rows if row["backend"] == backend and not row["error"]]
        oriented = [row for row in subset if row["controls_orientation"]]
        summary[backend] = {
            "runs": len(subset),
            "errors": sum(row["error"] is not None for row in rows if row["backend"] == backend),
            "reached_target_count": sum(row["reached_target"] for row in subset),
            "mean_position_error_m": float(np.mean([row["position_error_m"] for row in subset])),
            "max_position_error_m": float(np.max([row["position_error_m"] for row in subset])),
            "mean_oriented_position_error_m": float(
                np.mean([row["position_error_m"] for row in oriented])
            ),
            "mean_orientation_error_deg": float(
                np.mean([row["orientation_error_deg"] for row in oriented])
            ),
            "max_orientation_error_deg": float(
                np.max([row["orientation_error_deg"] for row in oriented])
            ),
            "mean_steps": float(np.mean([row["steps_executed"] for row in subset])),
            "elapsed_s": float(sum(row["elapsed_s"] for row in subset)),
        }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero-dir", type=Path, default=Path(os.environ.get("LIBERO_DIR", "/tmp/LIBERO")))
    parser.add_argument("--mink-path", type=Path, default=None)
    parser.add_argument("--suite", default="libero_object")
    parser.add_argument("--task-index", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--position-tolerance", type=float, default=0.002)
    parser.add_argument("--orientation-tolerance", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mink_path is not None:
        sys.path.insert(0, str(args.mink_path))
    import mink
    import mujoco
    import qpsolvers
    import robosuite

    get_libero_path, get_benchmark, ControlEnv = _load_libero(args.libero_dir)
    benchmark = get_benchmark(args.suite)()
    task = benchmark.get_task(args.task_index)
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    initial_states = _capture_initial_states(
        ControlEnv,
        bddl_path,
        args.seeds,
        max_steps=args.max_steps,
        position_tolerance=args.position_tolerance,
        orientation_tolerance=args.orientation_tolerance,
    )
    rows: list[dict[str, Any]] = []
    for backend in ("osc", "mink"):
        rows.extend(
            _run_backend(
                ControlEnv, bddl_path, backend, initial_states, DEFAULT_CASES, args
            )
        )

    payload = {
        "schema_version": "openeta.libero_mink_canary.v1",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "endpoint_accuracy_only_no_collision_avoidance",
        "task": {
            "suite": args.suite,
            "task_index": args.task_index,
            "language": task.language,
            "seeds": args.seeds,
        },
        "config": {
            "max_steps": args.max_steps,
            "position_tolerance_m": args.position_tolerance,
            "orientation_tolerance_rad": args.orientation_tolerance,
            "control_frequency_hz": 20,
            "start_profiles": START_PROFILES,
            "cases": [asdict(case) for case in DEFAULT_CASES],
        },
        "versions": {
            "mink": getattr(mink, "__version__", "0.0.6"),
            "mujoco": mujoco.__version__,
            "qpsolvers": qpsolvers.__version__,
            "robosuite": robosuite.__version__,
            "numpy": np.__version__,
        },
        "summary": _summary(rows),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
