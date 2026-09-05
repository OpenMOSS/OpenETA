"""General pose control and evaluation budgets for the live UniVTAC backend.

Targets use only the Agent request and measured robot state. No task geometry
or expert action is used by the controller.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

TOOLS = ("observe", "mark_point", "move_to", "report_issue", "check_task", "finish_episode")
PROMPT = """You control one live UniVTAC Insert Hole episode through
OpenETA's general-purpose robot tools.

Start by calling observe. Use the current camera images,
bilateral tactile images, robot state, and execution feedback
to decide your motion targets, orientation, and gripper actions.

After each action, inspect the returned observations.
You may probe, retreat, adjust, and retry within this episode.
A recoverable tool failure does not end the task.

Use check_task to verify completion and finish_episode to end.
No operation demonstrations are provided.

Do not infer task success merely because a command completed.

Task goal: Insert the held object into the hole.
Coordinates: xyz_m is the absolute TCP position in world metres;
delta_mm is displacement in world or current grip_site axes, in millimetres.
The TCP is UniVTAC's measured gripper center (fixed transform from panda_hand),
not the LIBERO grip site. approach_world sets TCP +Z; jaw_world sets TCP +X.
These fields specify orientation axes, not a task-specific rotation axis.
Unspecified position/orientation/gripper remain unchanged.
Use preview=true for target resolution without physics. Use execute_preview_id
alone to execute that preview. A preview is valid until the next real request.
"""


def quaternion_matrix(q: Any) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def matrix_quaternion(r: np.ndarray) -> np.ndarray:
    # Symmetric eigenproblem avoids the singular trace formula at 180 degrees.
    k = np.array([[r[0,0]-r[1,1]-r[2,2],r[1,0]+r[0,1],r[2,0]+r[0,2],r[2,1]-r[1,2]],
                  [r[1,0]+r[0,1],r[1,1]-r[0,0]-r[2,2],r[2,1]+r[1,2],r[0,2]-r[2,0]],
                  [r[2,0]+r[0,2],r[2,1]+r[1,2],r[2,2]-r[0,0]-r[1,1],r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1],np.trace(r)]])
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    return q if q[3] >= 0 else -q


def rotvec_matrix(v: Any) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    x, y, z = v / angle
    cross = np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross@cross)


def matrix_rotvec(r: np.ndarray) -> np.ndarray:
    q = matrix_quaternion(r)
    norm = np.linalg.norm(q[:3])
    return 2*q[:3] if norm < 1e-12 else q[:3] * (2*np.arctan2(norm,q[3])/norm)


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")


def vector3(value: Any, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must contain three finite numbers")
    return v


def resolve_target(current: np.ndarray, request: dict[str, Any]) -> np.ndarray:
    """Resolve OpenETA flat move_to fields in the backend's declared TCP frame."""
    allowed = {"xyz_m", "delta_mm", "delta_frame", "approach_world", "jaw_world", "gripper", "preview", "execute_preview_id"}
    if set(request) - allowed:
        raise ValueError(f"unsupported fields: {sorted(set(request) - allowed)}")
    if request.get("gripper") not in (None, "open", "close"):
        raise ValueError("gripper must be open or close")
    if request.get("delta_frame", "world") not in ("world", "grip_site"):
        raise ValueError("delta_frame must be world or grip_site")
    if request.get("xyz_m") is not None and request.get("delta_mm") is not None:
        raise ValueError("use xyz_m or delta_mm, not both")
    if not any(request.get(k) is not None for k in ("xyz_m", "delta_mm", "approach_world", "jaw_world", "gripper")):
        raise ValueError("specify a position, orientation, or gripper command")
    target = np.array(current, copy=True)
    if request.get("xyz_m") is not None:
        target[:3, 3] = vector3(request["xyz_m"], "xyz_m")
    if request.get("delta_mm") is not None:
        delta = vector3(request["delta_mm"], "delta_mm") / 1000.0
        if request.get("delta_frame", "world") == "grip_site":
            delta = current[:3, :3] @ delta
        target[:3, 3] += delta
    if request.get("approach_world") is not None or request.get("jaw_world") is not None:
        z = vector3(request.get("approach_world", current[:3, 2]), "approach_world")
        if np.linalg.norm(z) < 1e-8:
            raise ValueError("approach_world must be nonzero")
        z = z / np.linalg.norm(z)
        x = vector3(request.get("jaw_world", current[:3, 0]), "jaw_world")
        x = x - z * np.dot(x, z)
        if np.linalg.norm(x) < 1e-8:
            raise ValueError("jaw_world must not be parallel to approach_world")
        x /= np.linalg.norm(x)
        target[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    return target


def pose_dict(matrix: np.ndarray) -> dict[str, Any]:
    return {"xyz_m": matrix[:3, 3].tolist(), "quat_xyzw": matrix_quaternion(matrix[:3, :3]).tolist(),
            "approach_world": matrix[:3, 2].tolist(), "jaw_world": matrix[:3, 0].tolist()}


def pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return target[:3, 3] - actual[:3, 3], matrix_rotvec(target[:3, :3] @ actual[:3, :3].T)


def damped_joint_delta(jacobian: np.ndarray, error: np.ndarray, max_delta: float) -> np.ndarray:
    delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 0.0025 * np.eye(6), error)
    return delta * min(1.0, max_delta / max(float(np.max(np.abs(delta))), 1e-12))


class NativeController:
    """Measured-state IK with one native qpos action per control step."""
    def __init__(self, task: Any, config: dict[str, Any], root: Path):
        self.task, self.robot, self.config, self.root = task, task._robot_manager, config, root
        self.initial_steps = int(task.step_count)
        self.initial_physics = int(task._physics_step_count)
        self.early_stop = False
        self.success_available = callable(getattr(task, "check_success", None))
        self.actual_motion_requests = 0
        self.control_steps = 0

    def tcp(self) -> np.ndarray:
        # Native get_ee_pose is base-relative; compose the actual robot root.
        robot = self.robot.robot
        base = np.eye(4)
        base[:3, 3] = robot.data.root_link_pos_w[0].detach().cpu().numpy()
        q = robot.data.root_link_quat_w[0].detach().cpu().numpy()
        base[:3, :3] = quaternion_matrix(q[[1, 2, 3, 0]])
        return base @ self.robot.get_gripper_center_pose().to_transformation_matrix()

    def state(self) -> dict[str, Any]:
        return {"tcp_frame": "univtac_gripper_center", "world_axes": "x/y/z, right-handed",
                "position_unit": "metre", "quaternion_order": "xyzw", **pose_dict(self.tcp()),
                "joint_positions_rad": self.robot.get_qpos()[0].tolist(),
                "gripper_finger_positions_m": self.robot.get_gripper_qpos_all().cpu().tolist(),
                "gripper_open_fraction": float(self.robot.get_gripper_percentage()),
                "gripper_max_finger_qpos_m": float(self.robot.gripper_max_qpos),
                "tcp_offset_from_hand_m": float(self.robot.cfg.gripper_offset)}

    def counts(self) -> dict[str, Any]:
        physics = int(self.task._physics_step_count) - self.initial_physics
        return {"actual_motion_requests": self.actual_motion_requests, "control_steps": self.control_steps,
                "native_action_count": int(self.task.take_action_cnt),
                "physics_steps": physics, "simulation_time_seconds": physics * float(self.task.cfg.sim.dt),
                "simulator_step": int(self.task.step_count), "native_step_limit": int(self.task.cfg.step_lim),
                "initialization_control_steps": self.initial_steps, "initialization_physics_steps": self.initial_physics}

    def terminal(self) -> str | None:
        if bool(self.task.eval_success):
            return "native_success"
        if self.early_stop:
            return "native_early_stop"
        if self.task.take_action_cnt >= self.task.cfg.step_lim:
            return "native_step_limit"
        return None

    def check(self) -> dict[str, Any]:
        if not self.success_available:
            return {"available": False, "success": None}
        # Read-only checks use the same native predicate and latch as take_action.
        if self.task.check_success():
            self.task.eval_success = True
        return {"available": True, "success": bool(self.task.eval_success)}

    def execute(self, target: np.ndarray, gripper: str | None) -> dict[str, Any]:
        import torch
        start = time.monotonic()
        before = self.tcp()
        before_gripper = float(self.robot.get_gripper_qpos())
        r = self.robot
        joint_ids = r._arm_ids
        finger_target = float(r.get_gripper_qpos()) if gripper is None else (float(r.gripper_max_qpos) if gripper == "open" else 0.0)
        moved = False
        steps_before = self.control_steps
        for _ in range(self.config["max_control_steps_per_move"]):
            if self.terminal():
                break
            actual = self.tcp()
            dp, dr = pose_error(actual, target)
            grip_error = finger_target - float(r.get_gripper_qpos())
            if np.linalg.norm(dp) <= self.config["position_tolerance_m"] and np.linalg.norm(dr) <= self.config["orientation_tolerance_rad"] and abs(grip_error) <= 0.001:
                break
            # PhysX Jacobians are world-frame link Jacobians. Shift to the TCP.
            jac = r.robot.root_physx_view.get_jacobians()[0, r._jacobi_body_idx][:, joint_ids].detach().cpu().numpy().copy()
            link_pos = r.robot.data.body_link_pos_w[0, r._body_idx].detach().cpu().numpy()
            offset = actual[:3, 3] - link_pos
            jac[:3] += np.cross(jac[3:].T, offset).T
            delta = damped_joint_delta(jac, np.r_[dp, dr], self.config["max_joint_delta_rad"])
            joints = r.robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
            limits = r.robot.data.soft_joint_pos_limits[0, joint_ids].detach().cpu().numpy()
            next_joints = np.clip(joints + delta, limits[:, 0], limits[:, 1])
            next_grip = float(r.get_gripper_qpos()) + float(np.clip(grip_error, -0.001, 0.001))
            action = torch.tensor([*next_joints, next_grip], dtype=torch.float32, device=self.task.device)
            # force=False sends actuator targets; it never sets simulator joint poses.
            self.task.take_action(action, action_type="qpos", force=False, is_save=False)
            self.control_steps += 1
            moved = moved or not np.allclose(self.tcp(), before, atol=1e-7) or abs(float(r.get_gripper_qpos()) - before_gripper) > 1e-7
            if not self.task.eval_success:
                self.early_stop = bool(self.task.check_early_stop())
            append_row(self.root / "control_steps.jsonl", {"index": self.control_steps, "qpos_target": action.cpu().tolist(),
                       "actual": self.state(), "counts": self.counts(), "terminal": self.terminal()})
        if moved:
            self.actual_motion_requests += 1
        actual = self.tcp()
        dp, dr = pose_error(actual, target)
        reached = bool(np.linalg.norm(dp) <= self.config["position_tolerance_m"] and np.linalg.norm(dr) <= self.config["orientation_tolerance_rad"] and abs(finger_target - float(r.get_gripper_qpos())) <= 0.001)
        return {"requested_target": pose_dict(target), "requested_gripper": gripper,
                "actual": self.state(), "reached": reached,
                "remaining_position_delta_m": dp.tolist(), "remaining_rotation_rad": dr.tolist(),
                "error": None if reached else (self.terminal() or "control_segment_not_reached"),
                "control_steps": self.control_steps - steps_before, "physical_motion": moved,
                "elapsed_seconds": time.monotonic() - start, "counts": self.counts()}
