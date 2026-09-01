"""Worker-local Mink Cartesian goal executor for LIBERO experiments."""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Callable

import numpy as np

from sim.controllers.collision_recovery import (
    project_velocity_to_joint_limits,
    verified_collision_boundary_escape,
    verified_joint_limit_escape,
)
from adapter.motion_profiles import motion_control_profile


StepCallback = Callable[[np.ndarray, bool], dict[str, Any]]


def summarize_cartesian_progress_window(
    samples: list[tuple[np.ndarray, float]],
    *,
    target_direction_xyz: np.ndarray,
) -> dict[str, float]:
    """Summarize measured EEF progress without claiming physical contact."""

    if len(samples) < 2:
        raise ValueError("at least two progress samples are required")
    start_xyz, start_error = samples[0]
    end_xyz, end_error = samples[-1]
    displacement = np.asarray(end_xyz, dtype=np.float64) - np.asarray(
        start_xyz, dtype=np.float64
    )
    direction = np.asarray(target_direction_xyz, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    direction = direction / norm if norm > 1e-12 else np.zeros(3, dtype=np.float64)
    aligned = float(np.dot(displacement, direction))
    cross_track = displacement - aligned * direction
    return {
        "window_steps": float(len(samples) - 1),
        "command_aligned_progress_m": aligned,
        "cross_track_drift_m": float(np.linalg.norm(cross_track)),
        "position_error_start_m": float(start_error),
        "position_error_end_m": float(end_error),
        "position_error_improvement_m": float(start_error - end_error),
    }


def execute_libero_mink_goal(
    env: object,
    *,
    target_xyz: list[float],
    target_quat_xyzw: list[float] | None,
    preserve_current_orientation: bool,
    max_steps: int,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    gripper_command: float,
    enable_collision_check: bool,
    contact_authorization: dict[str, Any] | None,
    attachment_proxy: dict[str, Any] | None,
    ik_execution_seed: dict[str, Any] | None,
    step_callback: StepCallback,
    motion_execution_condition: object = "A",
) -> dict[str, Any]:
    """Drive one fixed JOINT_VELOCITY environment to an EEF goal with Mink."""

    import mink

    raw, robot = _libero_runtime(env)
    motion_profile = motion_control_profile(motion_execution_condition)
    if str(getattr(getattr(env, "_env", None), "_controller", "")) != "JOINT_VELOCITY":
        raise RuntimeError("Mink goal executor requires a JOINT_VELOCITY LIBERO environment")
    target = _finite_vector(target_xyz, 3, "target_xyz")
    explicit_target = (
        _normalised_quaternion(target_quat_xyzw)
        if target_quat_xyzw is not None
        else None
    )
    start_xyz, start_quat = _eef_pose(raw, robot)
    target_quat = (
        explicit_target
        if explicit_target is not None
        else start_quat.copy()
        if preserve_current_orientation
        else None
    )

    model = raw.sim.model._model
    configuration = mink.Configuration(model, q=raw.sim.data.qpos.copy())
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

    qvel_indices = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    qpos_indices = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    seeded_joint_positions = _validated_explicit_pose_seed(
        model,
        configuration.q,
        robot,
        seed=ik_execution_seed,
        target_xyz=target,
        target_quat_xyzw=target_quat,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )
    if isinstance(seeded_joint_positions, str):
        return _seed_rejection_result(
            start_xyz=start_xyz,
            start_quat_xyzw=start_quat,
            target_xyz=target,
            target_quat_xyzw=target_quat,
            reason=seeded_joint_positions,
            seed=ik_execution_seed,
        )
    if seeded_joint_positions is not None:
        # The global IK solution selects a useful redundancy basin, but must not
        # become a raw joint-space trajectory: interpolating directly to it can
        # swing the gripper through scene objects even when the endpoint is safe.
        # Keep Cartesian error as the primary task, normalized by the actual
        # execution tolerances, and use the preview joints as a null-space posture
        # guide.  This aligns the local QP with the same acceptance contract used
        # by preview/receipt validation without inventing task-phase logic.
        frame_task.set_position_cost(1.0 / max(float(position_tolerance_m), 0.002))
        frame_task.set_orientation_cost(
            1.0 / max(float(orientation_tolerance_rad), 0.05)
        )
        seed_cost = np.zeros(model.nv, dtype=np.float64)
        # The global preview is not merely a reachability bit: it identifies a
        # redundancy basin that satisfies the exact endpoint.  A very weak
        # posture cost allowed the local QP to converge toward a different
        # basin and saturate Panda joint 6 even when the preview solution had
        # ample margin.  Keep Cartesian error dominant, but give the private
        # host-carried seed enough weight to steer null-space motion.
        seed_cost[qvel_indices] = 5.0 if motion_profile.condition == "C" else 1.0
        seeded_posture_task = mink.PostureTask(
            model,
            cost=seed_cost,
            gain=0.8,
            lm_damping=1e-6,
        )
        target_configuration = configuration.q.copy()
        target_configuration[qpos_indices] = seeded_joint_positions
        seeded_posture_task.set_target(target_configuration)
        controller_tasks: list[Any] = [frame_task, seeded_posture_task]
        execution_policy = "ik_preview_seeded_cartesian_goal"
    else:
        controller_tasks = [frame_task, posture_task]
        execution_policy = "local_differential_cartesian_ik"

    start_site_rotation = _site_rotation(raw, robot)
    if target_quat is None:
        target_site_rotation = start_site_rotation
    else:
        from scipy.spatial.transform import Rotation

        start_body_rotation = _body_rotation(raw, robot)
        target_body_rotation = Rotation.from_quat(target_quat).as_matrix()
        target_site_rotation = target_body_rotation @ (
            start_body_rotation.T @ start_site_rotation
        )
    frame_task.set_target(
        mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(target_site_rotation),
            target,
        )
    )

    carrying_object = bool(
        isinstance(attachment_proxy, dict)
        and str(attachment_proxy.get("object_name") or "")
        and str(attachment_proxy.get("status") or "") in {"tentative", "confirmed"}
    )
    joint_velocity_limit_rad_s = (
        motion_profile.carrying_joint_velocity_limit_rad_s
        if carrying_object
        else motion_profile.nominal_joint_velocity_limit_rad_s
    )
    velocity_by_joint = {
        name: np.array([joint_velocity_limit_rad_s]) for name in robot.robot_joints
    }
    configuration_limit = mink.ConfigurationLimit(model, gain=0.95)
    velocity_limit = mink.VelocityLimit(model, velocities=velocity_by_joint)
    limits: list[Any] = [configuration_limit, velocity_limit]
    noncollision_limits = list(limits)
    collision_policy: dict[str, Any] | None = None
    if enable_collision_check:
        collision_policy = _libero_collision_policy(
            raw,
            robot,
            contact_authorization=contact_authorization,
            attachment_proxy=attachment_proxy,
        )
        limits.append(collision_policy["limit"])
    fixed_nonrobot_limit = _fixed_nonrobot_velocity_limit(model, qvel_indices)
    limits.append(fixed_nonrobot_limit)
    noncollision_limits.append(fixed_nonrobot_limit)
    emergency_escape_limits = [velocity_limit, fixed_nonrobot_limit]
    controller_scale = np.asarray(robot.controller.output_max, dtype=np.float64)
    dt = 1.0 / float(raw.env.control_freq)
    max_steps = max(1, min(int(max_steps), 300))
    total_steps = 0
    final_step: dict[str, Any] = {}
    control_error = ""
    control_failure: dict[str, Any] | None = None
    terminated = False
    collision_detected = False
    collision_info: dict[str, Any] = {
        "available": bool(collision_policy),
        "detected": False,
    }
    collision_boundary_recovery_steps = 0
    attached_collision_boundary_recovery_steps = 0
    joint_velocity_projection_steps = 0
    projected_joint_indices: set[int] = set()
    start_position_error = float(np.linalg.norm(target - start_xyz))
    best_position_error = start_position_error
    consecutive_position_regressions = 0
    best_orientation_error = (
        _angular_error_rad(start_quat, target_quat)
        if target_quat is not None
        else None
    )
    orientation_stall_steps = 0
    convergence_stalled = False
    convergence_stall_kind = ""
    stable_steps_completed = 0
    final_joint_velocity_max_abs: float | None = None
    progress_samples: deque[tuple[np.ndarray, float]] = deque(
        maxlen=max(1, motion_profile.progress_window_steps + 1)
    )
    progress_diagnostics: dict[str, Any] | None = None
    target_direction_xyz = target - start_xyz
    if motion_profile.progress_stall_enabled:
        progress_samples.append((start_xyz.copy(), start_position_error))

    while total_steps < max_steps:
        current_xyz, current_quat = _eef_pose(raw, robot)
        position_ok = bool(
            np.max(np.abs(target - current_xyz)) < position_tolerance_m
        )
        orientation_error = (
            _angular_error_rad(current_quat, target_quat)
            if target_quat is not None
            else None
        )
        orientation_ok = (
            orientation_error is None
            or orientation_error < orientation_tolerance_rad
        )
        current_arm_velocity = np.asarray(
            raw.sim.data.qvel[qvel_indices], dtype=np.float64
        )
        current_joint_velocity_max_abs = (
            float(np.max(np.abs(current_arm_velocity)))
            if current_arm_velocity.size
            else 0.0
        )
        final_joint_velocity_max_abs = current_joint_velocity_max_abs
        pose_ok = position_ok and orientation_ok
        if pose_ok and not motion_profile.stable_arrival_enabled:
            break
        if (
            pose_ok
            and motion_profile.stable_arrival_enabled
            and stable_steps_completed >= motion_profile.stable_steps_required
            and current_joint_velocity_max_abs
            < motion_profile.joint_velocity_tolerance_rad_s
        ):
            break
        settling = bool(pose_ok and motion_profile.stable_arrival_enabled)
        if not pose_ok:
            stable_steps_completed = 0

        configuration.update(q=raw.sim.data.qpos.copy())
        current_pair_distances = (
            _collision_pair_distances(
                configuration,
                collision_policy["protected_pairs"],
            )
            if collision_policy is not None
            else {}
        )
        attached_pairs = (
            collision_policy.get("attached_object_pairs", [])
            if collision_policy is not None
            else []
        )
        current_attached_distances = (
            _collision_pair_distances(
                configuration,
                attached_pairs,
                refresh_contacts=False,
            )
            if attached_pairs
            else {}
        )
        current_attached_hard_violation = bool(
            collision_policy is not None
            and any(
                distance < float(collision_policy["hard_stop_distance_m"])
                for distance in current_attached_distances.values()
            )
        )
        current_recovery_boundary = bool(
            collision_policy is not None
            and any(
                distance
                < float(collision_policy["minimum_distance_from_collisions_m"])
                for distance in current_pair_distances.values()
            )
        )
        current_joints, lower_limits, upper_limits = _robot_joint_limit_state(
            model,
            configuration.q,
            robot,
        )
        current_joint_violation = any(
            value < low or value > high
            for value, low, high in zip(current_joints, lower_limits, upper_limits)
        )
        used_constraint_escape = False
        used_joint_limit_escape = False
        used_collision_qp_fallback = False
        current_projected_joint_indices: list[int] = []
        try:
            velocity = (
                np.zeros(model.nv, dtype=np.float64)
                if settling
                else mink.solve_ik(
                    configuration,
                    controller_tasks,
                    dt,
                    solver="quadprog",
                    damping=1e-6,
                    safety_break=False,
                    limits=limits,
                )
            )
        except AssertionError:
            used_collision_qp_fallback = True
            # A strict collision QP can become infeasible after external physics
            # has already placed the robot inside the hard-distance boundary.
            # Compute an unconstrained candidate, but execute it only when a
            # one-step geometric preview proves that it monotonically exits all
            # existing penetrations and creates no new one.
            try:
                velocity = mink.solve_ik(
                    configuration,
                    controller_tasks,
                    dt,
                    solver="quadprog",
                    damping=1e-6,
                    safety_break=False,
                    limits=noncollision_limits,
                )
            except AssertionError:
                try:
                    velocity = mink.solve_ik(
                        configuration,
                        controller_tasks,
                        dt,
                        solver="quadprog",
                        damping=1e-6,
                        safety_break=False,
                        limits=emergency_escape_limits,
                    )
                    used_joint_limit_escape = True
                except AssertionError:
                    control_error = (
                        "mink_qp_no_solution: collision, configuration-limit, and "
                        "verified boundary-escape QPs all returned no velocity"
                    )
                    control_failure = {
                        "schema_version": "openeta.controller_failure.v1",
                        "code": "all_qp_variants_infeasible",
                        "strict_collision_qp_solved": False,
                        "noncollision_qp_solved": False,
                        "emergency_escape_qp_solved": False,
                        "current_collision_recovery_boundary": current_recovery_boundary,
                        "current_joint_limit_violation": current_joint_violation,
                        "current_minimum_distance_m": (
                            min(current_pair_distances.values())
                            if current_pair_distances
                            else None
                        ),
                        "recovery": (
                            "Use the reported actual EEF pose and choose a materially "
                            "different orientation or a waypoint that exits the active "
                            "collision/joint boundary; replaying the same target is not useful."
                        ),
                    }
                    break
        if used_joint_limit_escape:
            projected_arm_velocity, clipped_indices = project_velocity_to_joint_limits(
                velocity[qvel_indices],
                current_joints,
                lower_limits,
                upper_limits,
                dt=dt,
            )
            velocity = np.asarray(velocity, dtype=np.float64).copy()
            velocity[qvel_indices] = np.asarray(
                projected_arm_velocity,
                dtype=np.float64,
            )
            if clipped_indices:
                joint_velocity_projection_steps += 1
                projected_joint_indices.update(clipped_indices)
                current_projected_joint_indices = list(clipped_indices)
        if collision_policy is not None:
            commanded_velocity = np.zeros_like(velocity)
            commanded_velocity[qvel_indices] = velocity[qvel_indices]
            predicted_q = configuration.integrate(commanded_velocity, dt)
            predicted = mink.Configuration(model, q=predicted_q)
            predicted_pair_distances = _collision_pair_distances(
                predicted,
                collision_policy["protected_pairs"],
            )
            preview = _collision_distance_report(
                predicted,
                collision_policy["protected_pairs"],
                distance_limit_m=collision_policy["hard_stop_distance_m"],
                pair_distances=predicted_pair_distances,
            )
            collision_policy["minimum_distance_m"] = min(
                float(collision_policy.get("minimum_distance_m", math.inf)),
                float(preview["minimum_distance_m"]),
            )
            joint_escape_safe = True
            if used_joint_limit_escape:
                predicted_joints, _, _ = _robot_joint_limit_state(
                    model,
                    predicted_q,
                    robot,
                )
                joint_escape_safe = verified_joint_limit_escape(
                    current_joints,
                    predicted_joints,
                    lower_limits,
                    upper_limits,
                )
            collision_escape_safe = (
                verified_collision_boundary_escape(
                    current_pair_distances,
                    predicted_pair_distances,
                    hard_stop_distance_m=float(
                        collision_policy["hard_stop_distance_m"]
                    ),
                    recovery_boundary_distance_m=float(
                        collision_policy["minimum_distance_from_collisions_m"]
                    ),
                )
                if current_recovery_boundary
                else all(
                    distance >= float(collision_policy["hard_stop_distance_m"])
                    for distance in predicted_pair_distances.values()
                )
            )
            attached_preview: dict[str, Any] | None = None
            attached_escape_safe = True
            if attached_pairs:
                predicted_eef_xyz, predicted_eef_quat = _configuration_eef_pose(
                    model,
                    predicted_q,
                    robot,
                )
                predicted_attached_q = _transform_attached_object_with_eef(
                    predicted_q,
                    collision_policy,
                    current_eef_xyz=current_xyz,
                    current_eef_quat_xyzw=current_quat,
                    predicted_eef_xyz=predicted_eef_xyz,
                    predicted_eef_quat_xyzw=predicted_eef_quat,
                )
                predicted_attached = mink.Configuration(
                    model,
                    q=predicted_attached_q,
                )
                predicted_attached_distances = _collision_pair_distances(
                    predicted_attached,
                    attached_pairs,
                )
                attached_preview = _collision_distance_report(
                    predicted_attached,
                    attached_pairs,
                    distance_limit_m=collision_policy["hard_stop_distance_m"],
                    pair_distances=predicted_attached_distances,
                )
                collision_policy["minimum_attached_object_distance_m"] = min(
                    float(
                        collision_policy.get(
                            "minimum_attached_object_distance_m",
                            math.inf,
                        )
                    ),
                    float(attached_preview["minimum_distance_m"]),
                )
                attached_escape_safe = (
                    verified_collision_boundary_escape(
                        current_attached_distances,
                        predicted_attached_distances,
                        hard_stop_distance_m=float(
                            collision_policy["hard_stop_distance_m"]
                        ),
                    )
                    if current_attached_hard_violation
                    else all(
                        distance
                        >= float(collision_policy["hard_stop_distance_m"])
                        for distance in predicted_attached_distances.values()
                    )
                )
            if not attached_escape_safe and attached_preview is not None:
                collision_detected = True
                collision_info = {
                    **_collision_receipt(collision_policy),
                    **attached_preview,
                    "detected": True,
                    "collision_type": "attached_object_world",
                    "attached_object": collision_policy.get(
                        "authorized_target_object"
                    ),
                    "check_mode": "pre_actuation_attached_object_configuration",
                    "message": (
                        "Mink rejected the next joint-velocity step before actuation: "
                        f"the carried object would collide with "
                        f"{attached_preview.get('geom2_name') or 'scene geometry'}. "
                        "Raise or reroute the carry waypoint. If the object is already "
                        "in contact, only a verified monotonic escape is accepted."
                    ),
                }
                break
            if used_collision_qp_fallback and collision_escape_safe and joint_escape_safe:
                used_constraint_escape = True
            elif used_collision_qp_fallback and not (
                collision_escape_safe and joint_escape_safe
            ):
                control_error = (
                    "mink_qp_no_solution: fallback velocity failed the geometric "
                    "collision/joint safety preview and was not executed"
                )
                control_failure = {
                    "schema_version": "openeta.controller_failure.v1",
                    "code": "constraint_escape_preview_rejected",
                    "strict_collision_qp_solved": False,
                    "fallback_qp_solved": True,
                    "collision_escape_safe": collision_escape_safe,
                    "joint_escape_safe": joint_escape_safe,
                    "joint_velocity_projection_applied": bool(
                        current_projected_joint_indices
                    ),
                    "projected_joint_indices": sorted(
                        current_projected_joint_indices
                    ),
                    "joint_limit_diagnostics": (
                        _joint_limit_diagnostics(
                            current_joints,
                            predicted_joints,
                            lower_limits,
                            upper_limits,
                        )
                        if used_joint_limit_escape
                        else []
                    ),
                    "current_collision_recovery_boundary": current_recovery_boundary,
                    "current_joint_limit_violation": current_joint_violation,
                    "current_minimum_distance_m": (
                        min(current_pair_distances.values())
                        if current_pair_distances
                        else None
                    ),
                    "predicted_minimum_distance_m": (
                        min(predicted_pair_distances.values())
                        if predicted_pair_distances
                        else None
                    ),
                    "hard_stop_distance_m": float(
                        collision_policy["hard_stop_distance_m"]
                    ),
                    "recovery_boundary_distance_m": float(
                        collision_policy["minimum_distance_from_collisions_m"]
                    ),
                    "recovery": (
                        "The projected fallback step still could not satisfy the active "
                        "collision/joint constraints. First use the actual EEF pose and "
                        "fresh dual-view evidence to preview a short translation that "
                        "increases separation with preserve_current_orientation=true; "
                        "execute it only with collision checking. Rotate toward a new "
                        "candidate only after leaving the boundary. If no monotonic short "
                        "retreat is visually supported, choose a materially different "
                        "orientation or waypoint; replaying the same target is not useful."
                    ),
                }
                break
            elif (
                (current_recovery_boundary or current_joint_violation)
                and collision_escape_safe
                and joint_escape_safe
            ):
                used_constraint_escape = True
            elif preview["detected"]:
                collision_detected = True
                collision_info = {
                    **_collision_receipt(collision_policy),
                    **preview,
                    "detected": True,
                    "check_mode": "pre_actuation_configuration",
                    "message": (
                        "Mink rejected the next joint-velocity step before actuation: "
                        f"protected geoms {preview.get('geom1_name')} and "
                        f"{preview.get('geom2_name')} would violate the hard collision "
                        "distance. Choose a clearance waypoint or refresh the contact anchor."
                    ),
                }
                break
        arm_velocity = velocity[qvel_indices]
        normalised = np.divide(
            arm_velocity,
            controller_scale,
            out=np.zeros_like(arm_velocity),
            where=controller_scale != 0.0,
        )
        action = np.concatenate(
            [np.clip(normalised, -1.0, 1.0), [float(gripper_command)]]
        ).astype(np.float32)
        final_step = step_callback(action, (total_steps + 1) % 15 == 0)
        total_steps += 1
        if final_step.get("error"):
            control_error = str(final_step.get("error"))
            break
        if final_step.get("terminated") or final_step.get("truncated"):
            terminated = True
            break
        post_step_xyz, post_step_quat = _eef_pose(raw, robot)
        post_step_position_error = float(np.linalg.norm(target - post_step_xyz))
        post_step_max_axis_error = float(np.max(np.abs(target - post_step_xyz)))
        post_step_orientation_error = (
            _angular_error_rad(post_step_quat, target_quat)
            if target_quat is not None
            else None
        )
        post_step_pose_ok = bool(
            post_step_max_axis_error < position_tolerance_m
            and (
                post_step_orientation_error is None
                or post_step_orientation_error < orientation_tolerance_rad
            )
        )
        post_arm_velocity = np.asarray(
            raw.sim.data.qvel[qvel_indices], dtype=np.float64
        )
        final_joint_velocity_max_abs = (
            float(np.max(np.abs(post_arm_velocity)))
            if post_arm_velocity.size
            else 0.0
        )
        if motion_profile.stable_arrival_enabled:
            if (
                settling
                and post_step_pose_ok
                and final_joint_velocity_max_abs
                < motion_profile.joint_velocity_tolerance_rad_s
            ):
                stable_steps_completed += 1
            else:
                stable_steps_completed = 0

        if motion_profile.progress_stall_enabled and not settling:
            progress_samples.append(
                (post_step_xyz.copy(), post_step_position_error)
            )
            if len(progress_samples) == progress_samples.maxlen:
                metrics = summarize_cartesian_progress_window(
                    list(progress_samples),
                    target_direction_xyz=target_direction_xyz,
                )
                insufficient_progress = (
                    metrics["command_aligned_progress_m"]
                    < motion_profile.minimum_aligned_progress_m
                )
                error_not_improving = (
                    metrics["position_error_improvement_m"]
                    < motion_profile.minimum_error_improvement_m
                )
                significant_cross_track = (
                    metrics["cross_track_drift_m"]
                    >= motion_profile.cross_track_tolerance_m
                )
                progress_diagnostics = {
                    **metrics,
                    "insufficient_aligned_progress": insufficient_progress,
                    "error_not_improving": error_not_improving,
                    "significant_cross_track_drift": significant_cross_track,
                }
                if insufficient_progress and (
                    error_not_improving or significant_cross_track
                ):
                    convergence_stalled = True
                    convergence_stall_kind = "position_progress_stalled"
        if target_quat is not None:
            if post_step_position_error < best_position_error - 1e-4:
                best_position_error = post_step_position_error
                consecutive_position_regressions = 0
            elif post_step_position_error > best_position_error + 0.05:
                consecutive_position_regressions += 1
            else:
                consecutive_position_regressions = 0
            if (
                post_step_position_error > start_position_error + 0.10
                or consecutive_position_regressions >= 5
            ):
                control_error = (
                    "mink_controller_diverged: explicit-orientation execution moved "
                    f"away from the Cartesian target (start_error={start_position_error:.4f}m, "
                    f"best_error={best_position_error:.4f}m, "
                    f"current_error={post_step_position_error:.4f}m). The controller "
                    "stopped before continuing a non-convergent motion. Re-run an exact "
                    "IK preview and execute with its bound joint seed, or choose a "
                    "different full-pose candidate."
                )
                break
            post_step_position_ok = bool(
                np.max(np.abs(target - post_step_xyz)) < position_tolerance_m
            )
            if (
                best_orientation_error is None
                or post_step_orientation_error < best_orientation_error - 5e-5
            ):
                best_orientation_error = post_step_orientation_error
                orientation_stall_steps = 0
            elif (
                post_step_position_ok
                and post_step_orientation_error >= orientation_tolerance_rad
            ):
                orientation_stall_steps += 1
            else:
                orientation_stall_steps = 0
            if orientation_stall_steps >= 30:
                convergence_stalled = True
                convergence_stall_kind = "orientation_progress_stalled"
        if collision_policy is not None:
            actual = mink.Configuration(model, q=raw.sim.data.qpos.copy())
            actual_pair_distances = _collision_pair_distances(
                actual,
                collision_policy["protected_pairs"],
            )
            post_step = _collision_distance_report(
                actual,
                collision_policy["protected_pairs"],
                distance_limit_m=collision_policy["hard_stop_distance_m"],
                pair_distances=actual_pair_distances,
            )
            collision_policy["minimum_distance_m"] = min(
                float(collision_policy.get("minimum_distance_m", math.inf)),
                float(post_step["minimum_distance_m"]),
            )
            actual_joints, _, _ = _robot_joint_limit_state(
                model,
                actual.q,
                robot,
            )
            actual_collision_escape_safe = (
                verified_collision_boundary_escape(
                    current_pair_distances,
                    actual_pair_distances,
                    hard_stop_distance_m=float(
                        collision_policy["hard_stop_distance_m"]
                    ),
                    recovery_boundary_distance_m=float(
                        collision_policy["minimum_distance_from_collisions_m"]
                    ),
                )
                if current_recovery_boundary
                else all(
                    distance >= float(collision_policy["hard_stop_distance_m"])
                    for distance in actual_pair_distances.values()
                )
            )
            actual_joint_escape_safe = verified_joint_limit_escape(
                current_joints,
                actual_joints,
                lower_limits,
                upper_limits,
            )
            actual_attached_escape_safe = True
            actual_attached_report: dict[str, Any] | None = None
            if attached_pairs:
                actual_attached_distances = _collision_pair_distances(
                    actual,
                    attached_pairs,
                    refresh_contacts=False,
                )
                actual_attached_report = _collision_distance_report(
                    actual,
                    attached_pairs,
                    distance_limit_m=collision_policy["hard_stop_distance_m"],
                    pair_distances=actual_attached_distances,
                )
                collision_policy["minimum_attached_object_distance_m"] = min(
                    float(
                        collision_policy.get(
                            "minimum_attached_object_distance_m",
                            math.inf,
                        )
                    ),
                    float(actual_attached_report["minimum_distance_m"]),
                )
                actual_attached_escape_safe = (
                    verified_collision_boundary_escape(
                        current_attached_distances,
                        actual_attached_distances,
                        hard_stop_distance_m=float(
                            collision_policy["hard_stop_distance_m"]
                        ),
                    )
                    if current_attached_hard_violation
                    else all(
                        distance
                        >= float(collision_policy["hard_stop_distance_m"])
                        for distance in actual_attached_distances.values()
                    )
                )
            if (
                not actual_attached_escape_safe
                and actual_attached_report is not None
            ):
                collision_detected = True
                collision_info = {
                    **_collision_receipt(collision_policy),
                    **actual_attached_report,
                    "detected": True,
                    "collision_type": "attached_object_world",
                    "attached_object": collision_policy.get(
                        "authorized_target_object"
                    ),
                    "check_mode": "post_step_attached_object_configuration",
                    "message": (
                        "Mink stopped after simulator feedback showed the carried "
                        f"object colliding with "
                        f"{actual_attached_report.get('geom2_name') or 'scene geometry'}. "
                        "Observe the actual pose and choose an upward or lateral "
                        "clearance waypoint; do not assume the requested target was reached."
                    ),
                }
                break
            if current_attached_hard_violation and actual_attached_escape_safe:
                attached_collision_boundary_recovery_steps += 1
                collision_policy["attached_object_boundary_recovery_steps"] = (
                    attached_collision_boundary_recovery_steps
                )
            if (
                used_constraint_escape
                and actual_collision_escape_safe
                and actual_joint_escape_safe
            ):
                collision_boundary_recovery_steps += 1
                collision_policy["collision_boundary_recovery_steps"] = (
                    collision_boundary_recovery_steps
                )
                continue
            if post_step["detected"]:
                collision_detected = True
                collision_info = {
                    **_collision_receipt(collision_policy),
                    **post_step,
                    "detected": True,
                    "check_mode": "post_step_configuration",
                    "message": (
                        "Mink stopped after actual simulator feedback reported a protected "
                        f"collision between {post_step.get('geom1_name')} and "
                        f"{post_step.get('geom2_name')}. Observe and replan from the "
                        "reported end pose; do not assume the target was reached."
                    ),
                }
                break
        if convergence_stalled:
            break

    end_xyz, end_quat = _eef_pose(raw, robot)
    position_error = float(np.linalg.norm(target - end_xyz))
    max_axis_error = float(np.max(np.abs(target - end_xyz)))
    orientation_error = (
        _angular_error_rad(end_quat, target_quat)
        if target_quat is not None
        else None
    )
    reached = bool(
        max_axis_error < position_tolerance_m
        and (
            orientation_error is None
            or orientation_error < orientation_tolerance_rad
        )
        and not terminated
        and not control_error
        and not collision_detected
        and (
            not motion_profile.stable_arrival_enabled
            or (
                stable_steps_completed >= motion_profile.stable_steps_required
                and final_joint_velocity_max_abs is not None
                and final_joint_velocity_max_abs
                < motion_profile.joint_velocity_tolerance_rad_s
            )
        )
    )
    stop_reason = (
        "target_reached"
        if reached
        else "control_step_failed"
        if control_error
        else "episode_terminated"
        if terminated
        else "collision_detected"
        if collision_detected
        else "local_convergence_stalled"
        if convergence_stalled
        else "iteration_limit"
        if total_steps >= max_steps
        else "controller_stopped"
    )
    result: dict[str, Any] = {
        "target": {"x": float(target[0]), "y": float(target[1]), "z": float(target[2])},
        "start": {"xyz": start_xyz.tolist(), "quat_xyzw": start_quat.tolist()},
        "end": {"xyz": end_xyz.tolist(), "quat_xyzw": end_quat.tolist()},
        "steps_executed": total_steps,
        "terminated": terminated,
        "reward": final_step.get("reward", 0.0),
        "reached_target": reached,
        "position_error_m": position_error,
        "max_axis_position_error_m": max_axis_error,
        "stop_reason": stop_reason,
        "collision": (
            collision_info
            if collision_detected
            else _collision_receipt(collision_policy)
            if collision_policy is not None
            else {
                "detected": False,
                "available": False,
                "reason": "collision checking was disabled for this experimental motion",
            }
        ),
        "controller_receipt": {
            "schema_version": "openeta.controller_execution_receipt.v1",
            "controller_id": "mink.robosuite_joint_velocity",
            "configured_name": "JOINT_VELOCITY",
            "command_interface": "joint_velocity",
            "goal_executor": "openeta.worker_mink_goal.v1",
            "execution_location": "bench_worker",
            "execution_policy": execution_policy,
            "joint_velocity_limit_rad_s": joint_velocity_limit_rad_s,
            "transport_profile": (
                "attached_object_gentle" if carrying_object else "nominal"
            ),
            "joint_velocity_projection_steps": joint_velocity_projection_steps,
            "projected_joint_indices": sorted(projected_joint_indices),
            "ik_execution_seed_receipt_id": (
                str(ik_execution_seed.get("receipt_id") or "")
                if seeded_joint_positions is not None
                and isinstance(ik_execution_seed, dict)
                else None
            ),
            "orientation_policy": (
                "explicit"
                if explicit_target is not None
                else "preserve_current"
                if preserve_current_orientation
                else "unconstrained"
            ),
            "iteration_budget": max_steps,
            "position_tolerance_m": float(position_tolerance_m),
            "position_tolerance_metric": "max_axis_absolute_error",
            "steps_executed": total_steps,
            "stop_reason": stop_reason,
            "reached_target": reached,
            "convergence_stalled": convergence_stalled,
            "orientation_stall_steps": orientation_stall_steps,
        },
    }
    if motion_profile.condition != "A":
        result["motion_execution_profile"] = motion_profile.receipt()
        result["controller_receipt"].update(
            {
                "motion_execution_profile": motion_profile.receipt(),
                "convergence_stall_kind": convergence_stall_kind or None,
                "stable_arrival_enabled": motion_profile.stable_arrival_enabled,
                "stable_steps_required": motion_profile.stable_steps_required,
                "stable_steps_completed": stable_steps_completed,
                "joint_velocity_tolerance_rad_s": (
                    motion_profile.joint_velocity_tolerance_rad_s
                ),
                "joint_velocity_max_abs_rad_s": final_joint_velocity_max_abs,
            }
        )
    if progress_diagnostics is not None:
        result["controller_receipt"]["progress_diagnostics"] = progress_diagnostics
    if target_quat is not None:
        result["target"]["quat_xyzw"] = target_quat.tolist()
        result["orientation_error_rad"] = orientation_error
        result["orientation_error_deg"] = math.degrees(orientation_error or 0.0)
    if convergence_stalled:
        end_joints, end_lower, end_upper = _robot_joint_limit_state(
            model,
            raw.sim.data.qpos,
            robot,
        )
        result["convergence_diagnostics"] = {
            "schema_version": "openeta.controller_convergence_diagnostics.v1",
            "code": (
                "cartesian_position_progress_stalled"
                if convergence_stall_kind == "position_progress_stalled"
                else "full_pose_local_convergence_stalled"
            ),
            "message": (
                "Measured EEF progress toward the target stalled within the bounded "
                "tracking window. More iterations on the same segment are unlikely "
                "to help; use the actual EEF pose and choose a materially different "
                "waypoint or orientation."
                if convergence_stall_kind == "position_progress_stalled"
                else "Position is within tolerance, but explicit orientation stopped "
                "improving for 30 control steps. More iterations on the same pose "
                "are unlikely to help; choose a higher-margin orientation or waypoint."
            ),
            "progress": progress_diagnostics,
            "best_orientation_error_rad": best_orientation_error,
            "final_orientation_error_rad": orientation_error,
            "orientation_tolerance_rad": float(orientation_tolerance_rad),
            "seed_max_abs_joint_error_rad": (
                float(
                    np.max(
                        np.abs(
                            np.asarray(end_joints, dtype=np.float64)
                            - np.asarray(seeded_joint_positions, dtype=np.float64)
                        )
                    )
                )
                if seeded_joint_positions is not None
                else None
            ),
            "near_joint_limits": _joint_limit_diagnostics(
                end_joints,
                end_joints,
                end_lower,
                end_upper,
                reporting_margin_rad=0.05,
            ),
            "recovery": (
                "Run a fresh exact IK preview for a materially different wrist "
                "orientation or waypoint with larger joint margin; do not replay "
                "the same full pose with only a larger iteration budget."
            ),
        }
    if control_error:
        result.update({"ok": False, "code": "control_step_failed", "error": control_error})
    if control_failure is not None:
        result["controller_failure"] = control_failure
    observation = final_step.get("observation")
    if isinstance(observation, dict):
        result["observation"] = observation
    return result


def _validated_explicit_pose_seed(
    model: Any,
    current_q: np.ndarray,
    robot: Any,
    *,
    seed: dict[str, Any] | None,
    target_xyz: np.ndarray,
    target_quat_xyzw: np.ndarray | None,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> np.ndarray | str | None:
    """Validate a host-carried orientation-bound IK solution against this goal."""

    if seed is None:
        return None
    if target_quat_xyzw is None:
        return (
            "ik_execution_seed_unexpected: joint seeds require either an explicit "
            "orientation or an epoch-bound preserve-current orientation"
        )
    if seed.get("schema_version") != "openeta.ik_execution_seed.v1":
        return "ik_execution_seed_invalid: unsupported or missing schema_version"
    try:
        joints = _finite_vector(seed.get("joint_positions"), 7, "joint_positions")
    except (TypeError, ValueError) as exc:
        return f"ik_execution_seed_invalid: {exc}"

    qpos_indices = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    candidate_q = np.asarray(current_q, dtype=np.float64).copy()
    candidate_q[qpos_indices] = joints
    candidate_xyz, candidate_quat = _configuration_eef_pose(
        model,
        candidate_q,
        robot,
    )
    max_axis_error = float(np.max(np.abs(target_xyz - candidate_xyz)))
    orientation_error = _angular_error_rad(candidate_quat, target_quat_xyzw)
    if (
        max_axis_error > float(position_tolerance_m) + 1e-6
        or orientation_error > float(orientation_tolerance_rad) + 1e-6
    ):
        return (
            "ik_execution_seed_target_mismatch: the preview joint solution does not "
            "satisfy this move_to endpoint and tolerances "
            f"(max_axis_position_error={max_axis_error:.6f}m, "
            f"orientation_error={orientation_error:.6f}rad, "
            f"position_tolerance={float(position_tolerance_m):.6f}m, "
            f"orientation_tolerance={float(orientation_tolerance_rad):.6f}rad). "
            "Preview the exact execution pose and tolerances again; the worker did "
            "not actuate the stale or incompatible seed."
        )
    return joints


def _configuration_eef_pose(
    model: Any,
    q: np.ndarray,
    robot: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import mujoco
    from scipy.spatial.transform import Rotation

    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    site_name = robot.gripper.important_sites["grip_site"]
    body_name = robot.robot_model.eef_name
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if site_id < 0 or body_id < 0:
        raise RuntimeError(f"EEF site/body unavailable: {site_name}/{body_name}")
    xyz = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
    matrix = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    quat = Rotation.from_matrix(matrix).as_quat()
    return xyz, quat


def _seed_rejection_result(
    *,
    start_xyz: np.ndarray,
    start_quat_xyzw: np.ndarray,
    target_xyz: np.ndarray,
    target_quat_xyzw: np.ndarray | None,
    reason: str,
    seed: dict[str, Any] | None,
) -> dict[str, Any]:
    target: dict[str, Any] = {
        "x": float(target_xyz[0]),
        "y": float(target_xyz[1]),
        "z": float(target_xyz[2]),
    }
    if target_quat_xyzw is not None:
        target["quat_xyzw"] = target_quat_xyzw.tolist()
    return {
        "ok": False,
        "code": "ik_execution_seed_rejected",
        "error": reason,
        "target": target,
        "start": {"xyz": start_xyz.tolist(), "quat_xyzw": start_quat_xyzw.tolist()},
        "end": {"xyz": start_xyz.tolist(), "quat_xyzw": start_quat_xyzw.tolist()},
        "steps_executed": 0,
        "terminated": False,
        "reached_target": False,
        "position_error_m": float(np.linalg.norm(target_xyz - start_xyz)),
        "stop_reason": "ik_execution_seed_rejected",
        "controller_receipt": {
            "schema_version": "openeta.controller_execution_receipt.v1",
            "controller_id": "mink.robosuite_joint_velocity",
            "execution_policy": "ik_preview_seeded_cartesian_goal",
            "ik_execution_seed_receipt_id": (
                str(seed.get("receipt_id") or "") if isinstance(seed, dict) else ""
            ),
            "steps_executed": 0,
            "stop_reason": "ik_execution_seed_rejected",
            "reached_target": False,
        },
        "recovery_options": [
            {
                "action": "repeat_exact_ik_preview",
                "reason": (
                    "Obtain a current feasible receipt for the exact xyz, orientation, "
                    "and execution tolerances before retrying move_to."
                ),
            },
            {
                "action": "select_different_full_pose_candidate",
                "reason": "Do not execute an incompatible joint seed.",
            },
        ],
    }


def _libero_collision_policy(
    raw: Any,
    robot: Any,
    *,
    contact_authorization: dict[str, Any] | None,
    attachment_proxy: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a worker-local Mink policy from actual MuJoCo body ancestry."""

    import mink
    import mujoco

    model = raw.sim.model._model
    body_parent = np.asarray(model.body_parentid, dtype=np.int64)

    def descendants(root_id: int) -> set[int]:
        selected: set[int] = set()
        for body_id in range(model.nbody):
            current = body_id
            while current > 0 and current != root_id:
                current = int(body_parent[current])
            if current == root_id:
                selected.add(body_id)
        return selected

    moving_roots: set[int] = set()
    for joint_name in robot.robot_joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id >= 0:
            moving_roots.add(int(model.jnt_bodyid[joint_id]))
    moving_bodies: set[int] = set()
    for body_id in moving_roots:
        moving_bodies.update(descendants(body_id))
    gripper_root = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        str(robot.gripper.root_body),
    )
    if gripper_root < 0:
        raise RuntimeError("Panda gripper root body is unavailable for collision policy")
    gripper_bodies = descendants(int(gripper_root))
    collision_geoms = {
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_contype[geom_id]) != 0
        or int(model.geom_conaffinity[geom_id]) != 0
    }
    robot_geoms = {
        geom_id
        for geom_id in collision_geoms
        if int(model.geom_bodyid[geom_id]) in moving_bodies
    }
    gripper_geoms = {
        geom_id
        for geom_id in robot_geoms
        if int(model.geom_bodyid[geom_id]) in gripper_bodies
    }
    arm_geoms = robot_geoms - gripper_geoms
    world_geoms = collision_geoms - robot_geoms

    contact_name = str(
        (contact_authorization or {}).get("target_object_name") or ""
    ).strip()
    attached_name = str((attachment_proxy or {}).get("object_name") or "").strip()
    if contact_name and attached_name and contact_name != attached_name:
        raise RuntimeError(
            "contact authorization and attachment proxy refer to different objects"
        )
    authorized_name = contact_name or attached_name
    target_geoms: set[int] = set()
    target_root: int | None = None
    if authorized_name:
        target_obj = next(
            (
                obj
                for obj in raw.env.objects
                if str(getattr(obj, "name", "")) == authorized_name
            ),
            None,
        )
        if target_obj is None:
            raise RuntimeError(
                f"authorized contact object {authorized_name!r} is absent from the live scene"
            )
        target_root = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            str(target_obj.root_body),
        )
        if target_root < 0:
            raise RuntimeError(
                f"authorized contact object {authorized_name!r} has no MuJoCo root body"
            )
        target_bodies = descendants(int(target_root))
        target_geoms = {
            geom_id
            for geom_id in world_geoms
            if int(model.geom_bodyid[geom_id]) in target_bodies
        }
        if not target_geoms:
            raise RuntimeError(
                f"authorized contact object {authorized_name!r} has no collision geoms"
            )

    attached_object_pairs: list[tuple[int, int]] = []
    attached_object_qpos_adr: int | None = None
    if attached_name:
        if target_root is None:
            raise RuntimeError(
                f"attached object {attached_name!r} has no resolved MuJoCo root body"
            )
        root_joint_adr = int(model.body_jntadr[target_root])
        root_joint_count = int(model.body_jntnum[target_root])
        free_joint_id = next(
            (
                joint_id
                for joint_id in range(
                    root_joint_adr,
                    root_joint_adr + root_joint_count,
                )
                if int(model.jnt_type[joint_id])
                == int(mujoco.mjtJoint.mjJNT_FREE)
            ),
            None,
        )
        if free_joint_id is None:
            raise RuntimeError(
                f"attached object {attached_name!r} has no movable free joint; "
                "carried-object trajectory prediction is unavailable"
            )
        attached_object_qpos_adr = int(model.jnt_qposadr[free_joint_id])
        other_world_geoms = world_geoms - target_geoms
        attached_object_pairs = [
            (target_geom, world_geom)
            for target_geom in sorted(target_geoms)
            for world_geom in sorted(other_world_geoms)
        ]
        if not attached_object_pairs:
            raise RuntimeError(
                "carried-object collision coverage has no object/world geometry pairs"
            )

    geom_pairs: list[tuple[list[int], list[int]]] = []
    if arm_geoms and world_geoms:
        # Arm links always avoid every world object, including the grasp target.
        geom_pairs.append((sorted(arm_geoms), sorted(world_geoms)))
    gripper_world = world_geoms - target_geoms
    if gripper_geoms and gripper_world:
        # Only the gripper subtree receives the narrow target-contact exemption.
        geom_pairs.append((sorted(gripper_geoms), sorted(gripper_world)))
    if robot_geoms:
        geom_pairs.append((sorted(robot_geoms), sorted(robot_geoms)))
    limit = mink.CollisionAvoidanceLimit(
        model,
        geom_pairs,
        gain=0.85,
        minimum_distance_from_collisions=0.003,
        collision_detection_distance=0.02,
        bound_relaxation=0.0,
    )
    return {
        "limit": limit,
        "protected_pairs": list(limit.geom_id_pairs),
        "hard_stop_distance_m": -0.001,
        "minimum_distance_from_collisions_m": 0.003,
        "minimum_distance_m": math.inf,
        "world_geom_count": len(world_geoms),
        "world_object_count": len(list(raw.env.objects)),
        "robot_geom_count": len(robot_geoms),
        "protected_pair_count": len(limit.geom_id_pairs),
        "authorized_target_object": authorized_name or None,
        "authorized_target_geom_count": len(target_geoms),
        "contact_authorization": dict(contact_authorization or {}),
        "attachment_proxy": dict(attachment_proxy or {}),
        "attached_object_pairs": attached_object_pairs,
        "attached_object_qpos_adr": attached_object_qpos_adr,
        "attached_object_geom_count": len(target_geoms) if attached_name else 0,
        "attached_object_world_geom_count": (
            len(world_geoms - target_geoms) if attached_name else 0
        ),
    }


def _transform_attached_object_with_eef(
    q: np.ndarray,
    policy: dict[str, Any],
    *,
    current_eef_xyz: np.ndarray,
    current_eef_quat_xyzw: np.ndarray,
    predicted_eef_xyz: np.ndarray,
    predicted_eef_quat_xyzw: np.ndarray,
) -> np.ndarray:
    """Rigidly carry an attached free body into a predicted EEF pose.

    MuJoCo stores free-joint orientation as WXYZ, while the EEF helpers and
    scipy use XYZW.  Reconstructing the current object-to-EEF transform from
    live qpos on every preview keeps both translation and rotation aligned
    without introducing task-stage state or trusting a stale host estimate.
    """

    qpos_adr = policy.get("attached_object_qpos_adr")
    if not isinstance(qpos_adr, int) or qpos_adr < 0:
        raise RuntimeError(
            "attached-object trajectory prediction has no valid free-joint qpos address"
        )
    transformed = np.asarray(q, dtype=np.float64).copy()
    if qpos_adr + 7 > transformed.shape[0]:
        raise RuntimeError(
            "attached-object free-joint pose lies outside the configuration"
        )

    current_eef_position = _finite_vector(
        current_eef_xyz,
        3,
        "current_eef_xyz",
    )
    predicted_eef_position = _finite_vector(
        predicted_eef_xyz,
        3,
        "predicted_eef_xyz",
    )
    current_eef_quat = _normalised_quaternion(current_eef_quat_xyzw)
    predicted_eef_quat = _normalised_quaternion(predicted_eef_quat_xyzw)

    current_object_position = transformed[qpos_adr : qpos_adr + 3].copy()
    object_quat_wxyz = transformed[qpos_adr + 3 : qpos_adr + 7].copy()
    object_quat_xyzw = object_quat_wxyz[[1, 2, 3, 0]]
    object_quat_norm = float(np.linalg.norm(object_quat_xyzw))
    if object_quat_norm <= 1e-9 or not np.isfinite(object_quat_norm):
        raise RuntimeError("attached-object free-joint quaternion is invalid")
    current_object_quat = object_quat_xyzw / object_quat_norm

    current_eef_inverse = _quaternion_conjugate_xyzw(current_eef_quat)
    eef_to_object_position = _rotate_vector_by_quaternion_xyzw(
        current_eef_inverse,
        current_object_position - current_eef_position
    )
    eef_to_object_quat = _multiply_quaternions_xyzw(
        current_eef_inverse,
        current_object_quat,
    )
    predicted_object_position = predicted_eef_position + (
        _rotate_vector_by_quaternion_xyzw(
            predicted_eef_quat,
            eef_to_object_position,
        )
    )
    predicted_object_quat_xyzw = _normalise_quaternion_xyzw(
        _multiply_quaternions_xyzw(
            predicted_eef_quat,
            eef_to_object_quat,
        ),
        "predicted_attached_object_quaternion",
    )

    transformed[qpos_adr : qpos_adr + 3] = predicted_object_position
    transformed[qpos_adr + 3 : qpos_adr + 7] = predicted_object_quat_xyzw[
        [3, 0, 1, 2]
    ]
    return transformed


def _collision_distance_report(
    configuration: Any,
    pairs: list[tuple[int, int]],
    *,
    distance_limit_m: float,
    pair_distances: dict[tuple[int, int], float] | None = None,
) -> dict[str, Any]:
    model = configuration.model
    distances = (
        pair_distances
        if pair_distances is not None
        else _collision_pair_distances(configuration, pairs)
    )
    minimum = math.inf
    worst: tuple[int, int] | None = None
    for geom1, geom2 in pairs:
        pair = (int(geom1), int(geom2))
        distance = distances[pair]
        if distance < minimum:
            minimum = distance
            worst = pair
    if not math.isfinite(minimum):
        minimum = 0.02
    report: dict[str, Any] = {
        "detected": minimum < distance_limit_m,
        "minimum_distance_m": minimum,
        "hard_stop_distance_m": distance_limit_m,
    }
    if worst is not None:
        report.update(
            {
                "geom1_id": worst[0],
                "geom2_id": worst[1],
                "geom1_name": str(model.geom(worst[0]).name or worst[0]),
                "geom2_name": str(model.geom(worst[1]).name or worst[1]),
            }
        )
    return report


def _collision_pair_distances(
    configuration: Any,
    pairs: list[tuple[int, int]],
    *,
    refresh_contacts: bool = True,
) -> dict[tuple[int, int], float]:
    """Return signed distances for deterministic penetration-recovery checks."""

    import mujoco

    model = configuration.model
    data = configuration.data

    # ``mink.Configuration.update`` intentionally runs only kinematics, centre
    # of mass and constraint preparation.  It does not populate ``data.contact``.
    # Generate contacts explicitly for this hypothetical configuration before
    # combining them with mj_geomDistance; otherwise MuJoCo 3.3.0's orthogonal
    # box degeneracy can still authorize a deeply penetrating predicted step.
    if refresh_contacts:
        mujoco.mj_collision(model, data)
    contact_distances = _contact_pair_minimum_distances(data)
    return {
        (int(geom1), int(geom2)): _signed_geom_pair_distance(
            model,
            data,
            geom1,
            geom2,
            contact_distances=contact_distances,
        )
        for geom1, geom2 in pairs
    }


def _signed_geom_pair_distance(
    model: Any,
    data: Any,
    geom1: int,
    geom2: int,
    *,
    contact_distances: dict[tuple[int, int], float] | None = None,
) -> float:
    """Return the most conservative signed distance available for one pair.

    MuJoCo 3.3.0 can return exactly zero from ``mj_geomDistance`` for deeply
    penetrating boxes near an orthogonal orientation, even though an explicit
    collision pass produces negative-distance contacts for the same geom pair.
    A zero distance is above OpenETA's -1 mm hard-stop threshold and would
    therefore authorize the colliding step.  Contact distances are an
    independent production signal, so retain the smaller of both values.
    """

    import mujoco

    first = int(geom1)
    second = int(geom2)
    fromto = np.empty(6, dtype=np.float64)
    distance = float(
        mujoco.mj_geomDistance(model, data, first, second, 0.02, fromto)
    )
    contact_distance = (contact_distances or {}).get(
        (min(first, second), max(first, second))
    )
    if contact_distance is not None:
        distance = min(distance, contact_distance)
    return distance


def _contact_pair_minimum_distances(data: Any) -> dict[tuple[int, int], float]:
    """Index finite MuJoCo contact distances once for all protected pairs."""

    distances: dict[tuple[int, int], float] = {}
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        contact_first = int(contact.geom1)
        contact_second = int(contact.geom2)
        if contact_first < 0 or contact_second < 0:
            continue
        contact_distance = float(contact.dist)
        if not math.isfinite(contact_distance):
            continue
        key = (
            min(contact_first, contact_second),
            max(contact_first, contact_second),
        )
        distances[key] = min(distances.get(key, math.inf), contact_distance)
    return distances


def _robot_joint_limit_state(
    model: Any,
    qpos: np.ndarray,
    robot: Any,
) -> tuple[list[float], list[float], list[float]]:
    """Return controllable hinge positions and hard ranges in robot-joint order."""

    import mujoco

    current: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint_name in robot.robot_joints:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            str(joint_name),
        )
        if joint_id < 0:
            raise RuntimeError(f"robot joint {joint_name!r} is unavailable")
        qpos_index = int(model.jnt_qposadr[joint_id])
        current.append(float(qpos[qpos_index]))
        lower.append(float(model.jnt_range[joint_id][0]))
        upper.append(float(model.jnt_range[joint_id][1]))
    return current, lower, upper


def _joint_limit_diagnostics(
    current: list[float],
    predicted: list[float],
    lower: list[float],
    upper: list[float],
    *,
    reporting_margin_rad: float = 0.02,
) -> list[dict[str, float | int | str]]:
    """Describe only joints near or beyond a hard range for Agent recovery."""

    diagnostics: list[dict[str, float | int | str]] = []
    for index, (now, candidate, low, high) in enumerate(
        zip(current, predicted, lower, upper)
    ):
        current_margin = min(float(now) - float(low), float(high) - float(now))
        predicted_margin = min(
            float(candidate) - float(low),
            float(high) - float(candidate),
        )
        if min(current_margin, predicted_margin) > reporting_margin_rad:
            continue
        boundary = (
            "lower"
            if float(candidate) - float(low) < float(high) - float(candidate)
            else "upper"
        )
        diagnostics.append(
            {
                "joint_index": index,
                "boundary": boundary,
                "current_rad": float(now),
                "predicted_rad": float(candidate),
                "lower_rad": float(low),
                "upper_rad": float(high),
                "current_margin_rad": current_margin,
                "predicted_margin_rad": predicted_margin,
            }
        )
    return diagnostics


def _fixed_nonrobot_velocity_limit(model: Any, robot_qvel_indices: np.ndarray) -> Any:
    """Constrain Mink's full-scene QP to the seven controllable Panda DoFs.

    LIBERO objects have free joints in the same MuJoCo model. Without this
    constraint the collision QP can satisfy an avoidance inequality by moving
    an object in its hypothetical configuration, even though the dispatched
    JOINT_VELOCITY action controls only the arm.
    """

    from mink.limits.limit import Constraint, Limit

    allowed = {int(index) for index in robot_qvel_indices.tolist()}
    fixed = [index for index in range(model.nv) if index not in allowed]

    class _FixedVelocitySubspaceLimit(Limit):
        def compute_qp_inequalities(self, configuration: Any, dt: float) -> Any:
            del configuration, dt
            if not fixed:
                return Constraint()
            projection = np.eye(model.nv, dtype=np.float64)[fixed]
            return Constraint(
                G=np.vstack([projection, -projection]),
                h=np.zeros(2 * len(fixed), dtype=np.float64),
            )

    return _FixedVelocitySubspaceLimit()


def _collision_receipt(policy: dict[str, Any] | None) -> dict[str, Any]:
    if policy is None:
        return {"available": False, "detected": False}
    minimum = float(policy.get("minimum_distance_m", math.inf))
    attached = bool(policy.get("attachment_proxy"))
    attached_pairs = list(policy.get("attached_object_pairs") or [])
    attached_trajectory_checked = not attached or bool(attached_pairs)
    minimum_attached = float(
        policy.get("minimum_attached_object_distance_m", math.inf)
    )
    return {
        "schema_version": "openeta.worker_collision_receipt.v1",
        "available": True,
        "detected": False,
        "trajectory_checked": attached_trajectory_checked,
        "robot_trajectory_checked": True,
        "endpoint_checked": True,
        "world_checked": True,
        "self_checked": True,
        "check_mode": (
            "per_step_pre_actuation_and_post_step_configuration_with_"
            "rigidly_transformed_attached_object"
            if attached
            else "per_step_pre_actuation_and_post_step_configuration"
        ),
        "world_geom_count": int(policy["world_geom_count"]),
        "world_object_count": int(policy["world_object_count"]),
        "protected_pair_count": int(policy["protected_pair_count"]),
        "minimum_distance_m": minimum if math.isfinite(minimum) else None,
        "hard_stop_distance_m": float(policy["hard_stop_distance_m"]),
        "minimum_attached_object_distance_m": (
            minimum_attached if math.isfinite(minimum_attached) else None
        ),
        "constraint_boundary_recovery": {
            "used": int(policy.get("collision_boundary_recovery_steps") or 0) > 0,
            "verified_escape_steps": int(
                policy.get("collision_boundary_recovery_steps") or 0
            ),
            "policy": (
                "monotonic_collision_and_joint_boundary_exit_without_new_hard_violation"
            ),
        },
        "contact_policy": {
            "authorized_target_object": policy.get("authorized_target_object"),
            "authorized_target_geom_count": int(
                policy.get("authorized_target_geom_count") or 0
            ),
            "scope": (
                "gripper_subtree_to_authorized_target_only"
                if policy.get("authorized_target_object")
                else "no_intentional_world_contact_authorized"
            ),
            "authorization": policy.get("contact_authorization") or None,
            "attachment": policy.get("attachment_proxy") or None,
        },
        "attached_object_coverage": (
            {
                "endpoint_aabb_checked_by_outer_server": True,
                "trajectory_checked": bool(attached_pairs),
                "predicted_step_checked": bool(attached_pairs),
                "actual_step_checked": bool(attached_pairs),
                "geometry_source": "worker_live_mujoco_collision_geoms",
                "prediction_policy": (
                    "rigid_object_to_eef_transform_per_predicted_step"
                ),
                "attached_object_geom_count": int(
                    policy.get("attached_object_geom_count") or 0
                ),
                "world_geom_count": int(
                    policy.get("attached_object_world_geom_count") or 0
                ),
                "protected_pair_count": len(attached_pairs),
                "minimum_distance_m": (
                    minimum_attached if math.isfinite(minimum_attached) else None
                ),
                "boundary_recovery": {
                    "used": int(
                        policy.get("attached_object_boundary_recovery_steps") or 0
                    )
                    > 0,
                    "verified_escape_steps": int(
                        policy.get("attached_object_boundary_recovery_steps") or 0
                    ),
                    "policy": (
                        "monotonic_signed_distance_escape_without_new_hard_violation"
                    ),
                },
                "interpretation": (
                    "The outer server performs the conservative endpoint AABB check. "
                    "The worker additionally applies the live object-to-EEF rigid "
                    "transform to the attached object's MuJoCo collision geometry and "
                    "checks both predicted and actual object/world configurations."
                ),
            }
            if attached
            else None
        ),
    }


def _libero_runtime(env: object) -> tuple[Any, Any]:
    wrapper = getattr(env, "_env", None)
    raw = getattr(wrapper, "_env", None)
    if raw is None or not getattr(raw, "env", None):
        raise RuntimeError("expected UnifiedEnv -> _LibEnvWrapper -> OffScreenRenderEnv")
    robots = getattr(raw.env, "robots", None)
    if not robots:
        raise RuntimeError("LIBERO robot is unavailable")
    return raw, robots[0]


def _eef_pose(raw: Any, robot: Any) -> tuple[np.ndarray, np.ndarray]:
    site_id = raw.sim.model.site_name2id(robot.gripper.important_sites["grip_site"])
    body_quat_wxyz = np.asarray(
        raw.sim.data.get_body_xquat(robot.robot_model.eef_name), dtype=np.float64
    )
    return (
        np.asarray(raw.sim.data.site_xpos[site_id], dtype=np.float64).copy(),
        body_quat_wxyz[[1, 2, 3, 0]].copy(),
    )


def _site_rotation(raw: Any, robot: Any) -> np.ndarray:
    site_id = raw.sim.model.site_name2id(robot.gripper.important_sites["grip_site"])
    return np.asarray(raw.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3).copy()


def _body_rotation(raw: Any, robot: Any) -> np.ndarray:
    body_id = raw.sim.model.body_name2id(robot.robot_model.eef_name)
    return np.asarray(raw.sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3).copy()


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != length or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return array


def _normalised_quaternion(value: Any) -> np.ndarray:
    return _normalise_quaternion_xyzw(value, "target_quat_xyzw")


def _normalise_quaternion_xyzw(value: Any, name: str) -> np.ndarray:
    quat = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return quat / norm


def _quaternion_conjugate_xyzw(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = _finite_vector(quat_xyzw, 4, "quat_xyzw")
    return np.asarray([-quat[0], -quat[1], -quat[2], quat[3]])


def _multiply_quaternions_xyzw(
    lhs_xyzw: np.ndarray,
    rhs_xyzw: np.ndarray,
) -> np.ndarray:
    """Compose XYZW quaternions using the Hamilton product (lhs * rhs)."""

    lx, ly, lz, lw = _finite_vector(lhs_xyzw, 4, "lhs_xyzw")
    rx, ry, rz, rw = _finite_vector(rhs_xyzw, 4, "rhs_xyzw")
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _rotate_vector_by_quaternion_xyzw(
    quat_xyzw: np.ndarray,
    vector_xyz: np.ndarray,
) -> np.ndarray:
    """Apply a unit XYZW quaternion to a 3-D vector without scipy."""

    quat = _normalise_quaternion_xyzw(quat_xyzw, "rotation_quat_xyzw")
    vector = _finite_vector(vector_xyz, 3, "vector_xyz")
    axis = quat[:3]
    return vector + 2.0 * (
        quat[3] * np.cross(axis, vector)
        + np.cross(axis, np.cross(axis, vector))
    )


def _angular_error_rad(actual_xyzw: np.ndarray, target_xyzw: np.ndarray) -> float:
    from scipy.spatial.transform import Rotation

    relative = Rotation.from_quat(target_xyzw) * Rotation.from_quat(actual_xyzw).inv()
    return float(np.linalg.norm(relative.as_rotvec()))
