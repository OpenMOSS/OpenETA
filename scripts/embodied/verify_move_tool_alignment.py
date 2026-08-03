#!/usr/bin/env python3
"""Verify that a simulator backend and the move tools are frame-aligned.

This is the contract every agent relies on: when the model calls
``move_to(x, y, z, roll, pitch, yaw)`` or ``gripper_close``, the simulator
must actually do that in the same world frame the observations report, and
the tool's feedback must tell the truth about what happened.  If this
contract breaks (e.g. an OpenCV/OpenGL mix-up, a flipped axis, or success
reported for an unreachable target), no agent policy above can recover.

The probe is backend-agnostic.  Run it against any locally launched sim
MCP server, for example::

    PYTHONPATH=. .venv/bin/python scripts/embodied/verify_move_tool_alignment.py \
      --simulator-url http://127.0.0.1:8769/sse \
      --env-id openeta/libero_libero_object_task0-v0 \
      --task "pick up the alphabet soup and place it in the basket"

Checks performed:

* reset/observe expose a stable end-effector pose
* a commanded translation along +X/-X/+Y/+Z actually moves the EE along
  that world axis (dominant-component direction check)
* the reported ``position_error_m`` matches the measured final distance
* the pose in the ``move_to`` result matches a fresh ``observe_env``
* commanding the current pose with orientation control converges with
  near-zero error (catches euler/quaternion convention mismatches)
* gripper close/open change the measured aperture in the right direction
* on LIBERO, a Cartesian move with no gripper argument preserves the settled
  aperture instead of merely giving an unfinished close more simulation time
* a clearly unreachable target is NOT reported as reached

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.tools.sim_mcp import SseSimulatorMcpTransport  # noqa: E402


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def line(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        return f"[{status}] {self.name}: {self.detail}"


@dataclass
class ProbeContext:
    transport: SseSimulatorMcpTransport
    handle: str = ""
    session_id: str = ""
    backend: str = ""
    results: list[CheckResult] = field(default_factory=list)

    def call(self, tool: str, arguments: dict[str, Any], timeout_s: float = 180.0) -> dict:
        payload = self.transport.call_tool(tool, arguments, timeout_s=timeout_s)
        if not isinstance(payload, dict):
            return {"error": f"non-dict payload from {tool}"}
        return payload

    def session_args(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        args = {"handle": self.handle, "session_id": self.session_id}
        if extra:
            args.update(extra)
        return args

    def record(self, name: str, ok: bool, detail: str) -> CheckResult:
        result = CheckResult(name=name, ok=bool(ok), detail=detail)
        self.results.append(result)
        print(result.line(), flush=True)
        return result


# ---------------------------------------------------------------------------
# Pose helpers (same conventions as sim/mcp_server/server.py)
# ---------------------------------------------------------------------------


def _ee_pose(observation: dict[str, Any]) -> tuple[np.ndarray, list[float] | None]:
    robot = observation.get("robot", {})
    pose = robot.get("end_effector_pose", {}) if isinstance(robot, dict) else {}
    xyz = pose.get("xyz")
    if not isinstance(xyz, list) or len(xyz) != 3:
        raise RuntimeError("observation has no robot.end_effector_pose.xyz")
    quat = pose.get("quat_xyzw")
    return np.asarray(xyz, dtype=np.float64), quat if isinstance(quat, list) else None


def _observe_xyz(ctx: ProbeContext) -> tuple[np.ndarray, list[float] | None]:
    payload = ctx.call("observe_env", ctx.session_args())
    observation = payload.get("observation", payload)
    return _ee_pose(observation)


def _quat_xyzw_to_euler_xyz(q: list[float]) -> tuple[float, float, float]:
    """Inverse of server._euler_to_quat (xyz-intrinsic, radians)."""
    x, y, z, w = q
    # R = Rx(roll) Ry(pitch) Rz(yaw)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _angular_distance(a: list[float], b: list[float]) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2.0 * math.acos(min(1.0, dot))


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_reset_pose(ctx: ProbeContext, args: argparse.Namespace) -> np.ndarray | None:
    create = ctx.call(
        "create_env",
        {
            "env_id": args.env_id,
            "task": args.task,
            "seed": args.seed,
            "render_mode": "rgb_array",
        },
    )
    if "error" in create:
        ctx.record("create_env", False, str(create["error"]))
        return None
    ctx.handle = str(create.get("handle") or "")
    ctx.session_id = str(create.get("session_id") or "")
    ctx.backend = str(create.get("backend") or "")
    ctx.record(
        "create_env",
        bool(ctx.handle),
        f"handle={ctx.handle} backend={create.get('backend')}",
    )
    reset = ctx.call("reset_env", ctx.session_args({"seed": args.seed}))
    observation = reset.get("observation", reset)
    try:
        xyz, _quat = _ee_pose(observation)
    except RuntimeError as exc:
        ctx.record("reset_exposes_ee_pose", False, str(exc))
        return None
    ctx.record(
        "reset_exposes_ee_pose",
        True,
        f"ee_xyz=({xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f})",
    )
    return xyz


def _check_observe_stable(ctx: ProbeContext) -> None:
    xyz_a, _ = _observe_xyz(ctx)
    xyz_b, _ = _observe_xyz(ctx)
    drift = float(np.linalg.norm(xyz_a - xyz_b))
    ctx.record(
        "observe_pose_stable",
        drift < 1e-3,
        f"drift between two observes = {drift*1000:.3f} mm",
    )


def _check_axis_move(
    ctx: ProbeContext,
    start_xyz: np.ndarray,
    axis: int,
    delta: float,
    tol_m: float,
) -> np.ndarray:
    axis_name = "xyz"[axis]
    target = start_xyz.copy()
    target[axis] += delta
    result = ctx.call(
        "move_to",
        ctx.session_args(
            {
                "x": float(target[0]),
                "y": float(target[1]),
                "z": float(target[2]),
                "num_steps": 300,
                "tolerance": tol_m,
            }
        ),
    )
    if "error" in result:
        ctx.record(f"move_{axis_name}{delta:+.3f}", False, str(result["error"]))
        return start_xyz
    end = result.get("end", {})
    end_xyz = np.asarray(end.get("xyz", start_xyz), dtype=np.float64)
    reached = bool(result.get("success") or result.get("reached_target"))
    final_err = float(np.linalg.norm(end_xyz - target))
    reported_err = result.get("position_error_m")
    displacement = end_xyz - start_xyz
    dominant = int(np.argmax(np.abs(displacement)))
    direction_ok = (
        dominant == axis
        and displacement[axis] * delta > 0.0
        and abs(displacement[axis]) > 0.5 * abs(delta)
    )
    feedback_ok = True
    feedback_detail = "n/a"
    if isinstance(reported_err, (int, float)):
        feedback_ok = abs(float(reported_err) - final_err) < 2e-3
        feedback_detail = (
            f"reported={float(reported_err)*1000:.2f}mm measured={final_err*1000:.2f}mm"
        )
    ok = reached and final_err < max(tol_m * 2.0, 0.01) and direction_ok and feedback_ok
    ctx.record(
        f"move_{axis_name}{delta:+.3f}",
        ok,
        "reached={} final_err={:.2f}mm dir_ok={} feedback({}) disp=({:+.1f},{:+.1f},{:+.1f})mm".format(
            reached,
            final_err * 1000,
            direction_ok,
            feedback_detail,
            displacement[0] * 1000,
            displacement[1] * 1000,
            displacement[2] * 1000,
        ),
    )
    # Cross-check: the pose reported by move_to matches a fresh observation.
    obs_xyz, _ = _observe_xyz(ctx)
    consistency = float(np.linalg.norm(obs_xyz - end_xyz))
    ctx.record(
        f"move_{axis_name}{delta:+.3f}_observe_consistent",
        consistency < 2e-3,
        f"|observe - move_result.end| = {consistency*1000:.2f} mm",
    )
    return end_xyz


def _check_orientation_hold(
    ctx: ProbeContext, xyz: np.ndarray, quat: list[float] | None
) -> None:
    if quat is None:
        ctx.record("orientation_hold_current_pose", True, "skipped: backend reports no quat")
        return
    roll, pitch, yaw = _quat_xyzw_to_euler_xyz(quat)
    result = ctx.call(
        "move_to",
        ctx.session_args(
            {
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
                "roll": math.degrees(roll),
                "pitch": math.degrees(pitch),
                "yaw": math.degrees(yaw),
                "num_steps": 60,
                "tolerance": 0.005,
                "ori_tolerance": 0.05,
            }
        ),
    )
    if "error" in result:
        ctx.record("orientation_hold_current_pose", False, str(result["error"]))
        return
    reached = bool(result.get("success") or result.get("reached_target"))
    # When the commanded pose is the current pose, the controller can return
    # immediately without a position_error_m field; prefer the measured end pose.
    reported_pos_err = result.get("position_error_m")
    end = result.get("end", {})
    end_xyz = end.get("xyz")
    if isinstance(end_xyz, list) and len(end_xyz) == 3:
        pos_err = float(np.linalg.norm(np.asarray(end_xyz, dtype=np.float64) - xyz))
    elif isinstance(reported_pos_err, (int, float)):
        pos_err = float(reported_pos_err)
    else:
        pos_err = 1.0
    ori_err = result.get("orientation_error_rad")
    ori_err_deg = (
        math.degrees(float(ori_err)) if isinstance(ori_err, (int, float)) else None
    )
    ok = reached and pos_err < 0.01 and (ori_err_deg is None or ori_err_deg < 5.0)
    ctx.record(
        "orientation_hold_current_pose",
        ok,
        f"reached={reached} pos_err={pos_err*1000:.2f}mm ori_err={ori_err_deg} deg "
        "(commanded the pose we are already at: must converge with ~0 error)",
    )


def _check_gripper(ctx: ProbeContext) -> None:
    def _aperture(payload: dict[str, Any]) -> float | None:
        obs = payload.get("observation", payload)
        robot = obs.get("robot", {}) if isinstance(obs, dict) else {}
        state = robot.get("gripper_state", {}) if isinstance(robot, dict) else {}
        value = state.get("aperture_m")
        return float(value) if isinstance(value, (int, float)) else None

    before = ctx.call("observe_env", ctx.session_args())
    before_ap = _aperture(before)
    if before_ap is None:
        ctx.record("gripper_close_open", True, "skipped: no aperture reported")
        return
    closed = ctx.call("gripper_close", ctx.session_args())
    closed_ap = _aperture(closed)

    if ctx.backend == "libero" and closed_ap is not None:
        move_start, _ = _observe_xyz(ctx)
        move_target = move_start + np.asarray([0.02, 0.0, 0.0])
        move = ctx.call(
            "move_to",
            ctx.session_args(
                {
                    "x": float(move_target[0]),
                    "y": float(move_target[1]),
                    "z": float(move_target[2]),
                    "num_steps": 300,
                    "tolerance": 0.01,
                }
            ),
        )
        after_move = ctx.call("observe_env", ctx.session_args())
        after_move_ap = _aperture(after_move)
        move_end = move.get("end", {}) if isinstance(move, dict) else {}
        move_end_xyz = move_end.get("xyz") if isinstance(move_end, dict) else None
        displacement_m = (
            float(
                np.linalg.norm(
                    np.asarray(move_end_xyz, dtype=np.float64) - move_start
                )
            )
            if isinstance(move_end_xyz, list) and len(move_end_xyz) == 3
            else 0.0
        )
        aperture_delta_mm = (
            abs(after_move_ap - closed_ap) * 1000.0
            if after_move_ap is not None
            else float("inf")
        )
        moved = bool(move.get("success") or move.get("reached_target"))
        ctx.record(
            "cartesian_move_preserves_settled_gripper",
            moved and displacement_m > 0.005 and aperture_delta_mm <= 0.2,
            "moved={} displacement={:.2f}mm closed={:.3f}mm "
            "after_move={:.3f}mm delta={:.3f}mm".format(
                moved,
                displacement_m * 1000.0,
                closed_ap * 1000.0,
                (after_move_ap or 0.0) * 1000.0,
                aperture_delta_mm,
            ),
        )

    opened = ctx.call("gripper_open", ctx.session_args())
    opened_ap = _aperture(opened)
    ok = (
        closed_ap is not None
        and opened_ap is not None
        and closed_ap < before_ap + 1e-4
        and opened_ap > closed_ap + 1e-4
    )
    ctx.record(
        "gripper_close_open",
        ok,
        f"aperture before={before_ap*1000:.1f}mm closed={closed_ap and closed_ap*1000:.1f}mm "
        f"reopened={opened_ap and opened_ap*1000:.1f}mm",
    )


def _check_unreachable_is_honest(ctx: ProbeContext, start_xyz: np.ndarray) -> None:
    target = start_xyz + np.array([0.0, 0.0, 2.0])
    result = ctx.call(
        "move_to",
        ctx.session_args(
            {
                "x": float(target[0]),
                "y": float(target[1]),
                "z": float(target[2]),
                "num_steps": 40,
                "tolerance": 0.005,
            }
        ),
    )
    if "error" in result:
        ctx.record("unreachable_target_not_reported_reached", True, "returned error (honest)")
        return
    reached = bool(result.get("success") or result.get("reached_target"))
    ctx.record(
        "unreachable_target_not_reported_reached",
        not reached,
        f"reached_target={reached} position_error_m={result.get('position_error_m')} "
        "(tool must not claim success for an unreachable pose)",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_alignment_checks(args: argparse.Namespace) -> list[CheckResult]:
    ctx = ProbeContext(transport=SseSimulatorMcpTransport(args.simulator_url))
    start_xyz = _check_reset_pose(ctx, args)
    if start_xyz is None:
        return ctx.results
    try:
        _check_observe_stable(ctx)
        _check_orientation_hold(ctx, start_xyz, _observe_xyz(ctx)[1])
        _check_gripper(ctx)

        # Each axis move gets a fresh episode: backends with small
        # max_episode_steps (e.g. ManiSkill's 200) would otherwise truncate
        # mid-sequence and later moves would silently stop moving.
        for axis, delta in ((0, args.delta_m), (0, -args.delta_m),
                            (1, args.delta_m), (1, -args.delta_m),
                            (2, args.delta_m), (2, -args.delta_m)):
            reset = ctx.call("reset_env", ctx.session_args({"seed": args.seed}))
            try:
                fresh_xyz, _quat = _ee_pose(reset.get("observation", reset))
            except RuntimeError:
                fresh_xyz = start_xyz
            _check_axis_move(ctx, fresh_xyz, axis, delta, args.tolerance_m)

        ctx.call("reset_env", ctx.session_args({"seed": args.seed}))
        try:
            honest_xyz, _ = _ee_pose(
                ctx.call("observe_env", ctx.session_args()).get("observation", {})
            )
        except RuntimeError:
            honest_xyz = np.asarray(start_xyz, dtype=np.float64)
        _check_unreachable_is_honest(ctx, honest_xyz)
    finally:
        if ctx.handle:
            try:
                ctx.call("close_env", ctx.session_args(), timeout_s=30.0)
            except Exception:  # noqa: BLE001
                pass
    return ctx.results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulator-url", default="http://127.0.0.1:8769/sse")
    parser.add_argument("--env-id", default="openeta/libero_libero_object_task0-v0")
    parser.add_argument(
        "--task", default="pick up the alphabet soup and place it in the basket"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delta-m", type=float, default=0.02)
    parser.add_argument("--tolerance-m", type=float, default=0.01)
    args = parser.parse_args()

    results = run_alignment_checks(args)
    failed = [r for r in results if not r.ok]
    print(
        f"\n{len(results) - len(failed)}/{len(results)} checks passed"
        + (f"; failures: {[r.name for r in failed]}" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
