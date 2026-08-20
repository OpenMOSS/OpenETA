"""Read-only endpoint reachability checks for simulator backends.

The checker deliberately answers a geometric question only: whether a target
EEF pose has a joint-limit-respecting inverse-kinematics solution.  Endpoint
collision and path feasibility remain separate layers owned by the MCP server.
"""

from __future__ import annotations

import hashlib
import math
import time
from typing import Any

import numpy as np


def check_endpoint_reachability(
    env: object,
    *,
    target_xyz: list[float],
    target_quat_xyzw: list[float] | None = None,
    preserve_current_orientation: bool = True,
    position_tolerance_m: float = 0.002,
    orientation_tolerance_rad: float = 0.05,
    max_attempts: int = 24,
    max_nfev_per_attempt: int = 300,
    timeout_s: float = 10.0,
) -> dict[str, Any]:
    """Check endpoint IK without stepping or mutating the live environment.

    A completed multi-start numerical search returns ``reachable`` or
    ``unreachable``.  Missing backend support, solver errors, and exhausted
    wall-clock budgets return ``unknown`` so callers do not confuse solver
    uncertainty with a proved geometric rejection.
    """

    backend = str(getattr(env, "_backend", "") or "")
    if backend != "libero":
        return _unknown(
            "backend_unsupported",
            f"Reachability backend is not implemented for {backend or 'unknown'}.",
            backend=backend,
        )
    try:
        problem = _libero_problem(env)
    except Exception as exc:  # noqa: BLE001 - capability discovery must be structured.
        return _unknown(
            "kinematic_model_unavailable",
            f"Could not access the LIBERO Panda kinematic model: {exc}",
            backend=backend,
        )
    try:
        return _solve_problem(
            problem,
            target_xyz=target_xyz,
            target_quat_xyzw=target_quat_xyzw,
            preserve_current_orientation=preserve_current_orientation,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            max_attempts=max_attempts,
            max_nfev_per_attempt=max_nfev_per_attempt,
            timeout_s=timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - return unknown, never a false allow/reject.
        return _unknown(
            "ik_solver_error",
            f"Reachability solver failed: {type(exc).__name__}: {exc}",
            backend=backend,
        )


def _libero_problem(env: object) -> dict[str, Any]:
    """Extract an independent MuJoCo FK problem from a UnifiedEnv LIBERO env."""

    import mujoco

    wrapper = getattr(env, "_env", None)
    raw = getattr(wrapper, "_env", None)
    if raw is None:
        raise RuntimeError("expected UnifiedEnv -> _LibEnvWrapper -> OffScreenRenderEnv")
    sim = getattr(raw, "sim", None)
    inner = getattr(raw, "env", None)
    robots = getattr(inner, "robots", None)
    if sim is None or not robots:
        raise RuntimeError("LIBERO simulator or robot is unavailable")
    robot = robots[0]
    model = getattr(getattr(sim, "model", None), "_model", None)
    if model is None:
        raise RuntimeError("native MuJoCo model is unavailable")

    joint_names = list(getattr(robot, "robot_joints", ()) or ())
    qpos_indices = np.asarray(getattr(robot, "_ref_joint_pos_indexes", ()), dtype=np.int64)
    if len(joint_names) != 7 or qpos_indices.size != 7:
        raise RuntimeError(
            f"expected a 7-DoF Panda arm, got {len(joint_names)} joints and "
            f"{qpos_indices.size} qpos indices"
        )
    lower: list[float] = []
    upper: list[float] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"joint not found in MuJoCo model: {name}")
        joint_range = np.asarray(model.jnt_range[joint_id], dtype=np.float64)
        lower.append(float(joint_range[0]))
        upper.append(float(joint_range[1]))

    site_name = robot.gripper.important_sites["grip_site"]
    body_name = robot.robot_model.eef_name
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if site_id < 0 or body_id < 0:
        raise RuntimeError(f"EEF site/body unavailable: {site_name}/{body_name}")

    current_qpos = np.asarray(sim.data.qpos, dtype=np.float64).copy()
    current_body_quat_wxyz = np.asarray(
        sim.data.get_body_xquat(body_name), dtype=np.float64
    )
    return {
        "backend": "libero",
        "model": model,
        "data": mujoco.MjData(model),
        "current_qpos": current_qpos,
        "current_arm_q": current_qpos[qpos_indices].copy(),
        "current_eef_quat_xyzw": current_body_quat_wxyz[[1, 2, 3, 0]].copy(),
        "qpos_indices": qpos_indices,
        "lower": np.asarray(lower, dtype=np.float64),
        "upper": np.asarray(upper, dtype=np.float64),
        "site_id": int(site_id),
        "body_id": int(body_id),
    }


def _solve_problem(
    problem: dict[str, Any],
    *,
    target_xyz: list[float],
    target_quat_xyzw: list[float] | None,
    preserve_current_orientation: bool,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    max_attempts: int,
    max_nfev_per_attempt: int,
    timeout_s: float,
) -> dict[str, Any]:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    target = _finite_vector(target_xyz, 3, "target_xyz")
    orientation_mode = (
        "explicit"
        if target_quat_xyzw is not None
        else "preserve_current"
        if preserve_current_orientation
        else "unconstrained"
    )
    target_quat = (
        _normalised_quaternion(target_quat_xyzw)
        if target_quat_xyzw is not None
        else np.asarray(problem["current_eef_quat_xyzw"], dtype=np.float64)
        if preserve_current_orientation
        else None
    )
    if not math.isfinite(position_tolerance_m) or position_tolerance_m <= 0:
        raise ValueError("position_tolerance_m must be positive and finite")
    if not math.isfinite(orientation_tolerance_rad) or orientation_tolerance_rad <= 0:
        raise ValueError("orientation_tolerance_rad must be positive and finite")
    max_attempts = max(1, min(int(max_attempts), 64))
    max_nfev_per_attempt = max(20, min(int(max_nfev_per_attempt), 2000))
    timeout_s = max(0.1, min(float(timeout_s), 30.0))

    model = problem["model"]
    data = problem["data"]
    current_qpos = problem["current_qpos"]
    qpos_indices = problem["qpos_indices"]
    lower = problem["lower"]
    upper = problem["upper"]
    site_id = problem["site_id"]
    body_id = problem["body_id"]
    start_time = time.monotonic()

    def fk(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import mujoco

        data.qpos[:] = current_qpos
        data.qpos[qpos_indices] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        position = np.asarray(data.site_xpos[site_id], dtype=np.float64).copy()
        rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3).copy()
        return position, rotation

    def metrics(q: np.ndarray) -> dict[str, Any]:
        position, rotation = fk(q)
        position_residual = target - position
        orientation_error = None
        if target_quat is not None:
            relative = Rotation.from_quat(target_quat) * Rotation.from_matrix(rotation).inv()
            orientation_error = float(np.linalg.norm(relative.as_rotvec()))
        max_axis = float(np.max(np.abs(position_residual)))
        norm = float(np.linalg.norm(position_residual))
        score = max_axis / position_tolerance_m
        if orientation_error is not None:
            score = max(score, orientation_error / orientation_tolerance_rad)
        return {
            "joint_positions": [float(value) for value in q],
            "position_error_m": norm,
            "max_axis_position_error_m": max_axis,
            "orientation_error_rad": orientation_error,
            "normalized_worst_constraint": float(score),
            "joint_margin_min_rad": float(np.min(np.minimum(q - lower, upper - q))),
        }

    def residual(q: np.ndarray, mode: str) -> np.ndarray:
        if time.monotonic() - start_time > timeout_s:
            raise _SearchTimeout
        position, rotation = fk(q)
        pieces: list[np.ndarray] = []
        if mode in {"full", "position"}:
            pieces.append((position - target) / position_tolerance_m)
        if mode in {"full", "orientation"} and target_quat is not None:
            relative = Rotation.from_quat(target_quat) * Rotation.from_matrix(rotation).inv()
            pieces.append(relative.as_rotvec() / orientation_tolerance_rad)
        return np.concatenate(pieces)

    seeds = _joint_seeds(problem, target, target_quat, max_attempts)

    def run(mode: str, attempts: int) -> tuple[bool, dict[str, Any], int, int]:
        best: dict[str, Any] | None = None
        completed = 0
        evaluations = 0
        for seed in seeds[:attempts]:
            solved = least_squares(
                lambda q: residual(q, mode),
                seed,
                bounds=(lower, upper),
                max_nfev=max_nfev_per_attempt,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            completed += 1
            evaluations += int(solved.nfev)
            candidate = metrics(np.asarray(solved.x, dtype=np.float64))
            if mode == "position":
                candidate_score = candidate["max_axis_position_error_m"] / position_tolerance_m
                passed = candidate["max_axis_position_error_m"] <= position_tolerance_m
            elif mode == "orientation":
                orientation_error = candidate["orientation_error_rad"]
                candidate_score = orientation_error / orientation_tolerance_rad
                passed = orientation_error <= orientation_tolerance_rad
            else:
                candidate_score = candidate["normalized_worst_constraint"]
                passed = candidate_score <= 1.0
            if best is None or candidate_score < best["_mode_score"]:
                best = {**candidate, "_mode_score": float(candidate_score)}
            if passed:
                return True, best, completed, evaluations
        assert best is not None
        return False, best, completed, evaluations

    try:
        full_ok, best, completed, evaluations = run("full", len(seeds))
        position_ok = full_ok
        orientation_ok: bool | None = full_ok if target_quat is not None else None
        component_attempts = min(8, len(seeds))
        if not full_ok:
            position_ok, _position_best, p_completed, p_evaluations = run(
                "position", component_attempts
            )
            completed += p_completed
            evaluations += p_evaluations
            if target_quat is not None:
                orientation_ok, _orientation_best, o_completed, o_evaluations = run(
                    "orientation", component_attempts
                )
                completed += o_completed
                evaluations += o_evaluations
    except _SearchTimeout:
        elapsed = time.monotonic() - start_time
        return {
            **_unknown(
                "ik_search_timeout",
                "IK search exhausted its wall-clock budget; target reachability is unknown.",
                backend=problem["backend"],
            ),
            "target": _target_payload(target, target_quat),
            "orientation_mode": orientation_mode,
            "tolerances": _tolerance_payload(
                position_tolerance_m, orientation_tolerance_rad
            ),
            "solver": {
                "method": "bounded_multistart_least_squares",
                "attempts_completed": locals().get("completed", 0),
                "function_evaluations": locals().get("evaluations", 0),
                "elapsed_s": elapsed,
                "timed_out": True,
            },
        }

    best.pop("_mode_score", None)
    elapsed = time.monotonic() - start_time
    status = "reachable" if full_ok else "unreachable"
    if full_ok:
        reason_code = "ik_solution_found"
        message = "A joint-limit-respecting IK solution was found."
        suggestions: list[str] = []
    elif position_ok and orientation_ok:
        reason_code = "full_pose_infeasible"
        message = (
            "Position and orientation are separately reachable, but the requested "
            "combined 6-DoF pose was not feasible within the search budget."
        )
        suggestions = ["relax_target_orientation", "select_another_grasp_candidate"]
    elif not position_ok:
        reason_code = "position_unreachable"
        message = "The requested EEF position was not reachable within joint limits."
        suggestions = ["move_target_toward_workspace", "select_another_grasp_candidate"]
    else:
        reason_code = "orientation_unreachable"
        message = "The requested EEF orientation was not reachable within joint limits."
        suggestions = ["relax_target_orientation", "select_another_grasp_candidate"]
    return {
        "status": status,
        "kinematic_status": status,
        "feasible": full_ok,
        "reason_code": reason_code,
        "message": message,
        "backend": problem["backend"],
        "target": _target_payload(target, target_quat),
        "orientation_mode": orientation_mode,
        "tolerances": _tolerance_payload(position_tolerance_m, orientation_tolerance_rad),
        "position_only_reachable": position_ok,
        "orientation_only_reachable": orientation_ok,
        "best_candidate": best,
        "solver": {
            "method": "bounded_multistart_least_squares",
            "attempts_completed": completed,
            "function_evaluations": evaluations,
            "elapsed_s": elapsed,
            "timed_out": False,
            "infeasibility_evidence": (
                None if full_ok else "completed_multistart_search_no_feasible_solution"
            ),
            "formal_infeasibility_proof": False,
        },
        "suggestions": suggestions,
    }


def _joint_seeds(
    problem: dict[str, Any],
    target: np.ndarray,
    target_quat: np.ndarray | None,
    count: int,
) -> list[np.ndarray]:
    lower = problem["lower"]
    upper = problem["upper"]
    current = np.clip(problem["current_arm_q"], lower, upper)
    seeds = [current, (lower + upper) / 2.0]
    digest = hashlib.sha256(
        target.tobytes() + (target_quat.tobytes() if target_quat is not None else b"")
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    while len(seeds) < count:
        seeds.append(rng.uniform(lower, upper))
    return seeds[:count]


def _finite_vector(value: list[float], length: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size != length or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return arr


def _normalised_quaternion(value: list[float]) -> np.ndarray:
    quat = _finite_vector(value, 4, "target_quat_xyzw")
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9:
        raise ValueError("target_quat_xyzw must be non-zero")
    return quat / norm


def _target_payload(target: np.ndarray, target_quat: np.ndarray | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"frame": "world", "xyz": target.tolist()}
    if target_quat is not None:
        payload["quat_xyzw"] = target_quat.tolist()
    return payload


def _tolerance_payload(position: float, orientation: float) -> dict[str, float]:
    return {
        "max_axis_position_error_m": float(position),
        "orientation_error_rad": float(orientation),
    }


def _unknown(reason_code: str, message: str, *, backend: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "kinematic_status": "unknown",
        "feasible": None,
        "reason_code": reason_code,
        "message": message,
        "backend": backend,
        "position_only_reachable": None,
        "orientation_only_reachable": None,
        "best_candidate": None,
        "suggestions": ["inspect_checker_diagnostics", "retry_or_adjust_target_conservatively"],
    }


class _SearchTimeout(RuntimeError):
    pass
