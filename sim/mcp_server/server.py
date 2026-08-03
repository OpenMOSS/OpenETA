#!/usr/bin/env python
"""OpenETA MCP Server — FastMCP tools + Starlette ASGI + CLI entry.

The heavy lifting is delegated to sibling modules:
  session.py      — state storage & lifecycle
  worker_mgr.py   — per-bench subprocess workers & proxy helpers
  rest_api.py     — live camera-view page & SSE streaming handlers
  dashboard_html  — HTML page templates (live camera view)
"""

from __future__ import annotations

import functools
import os
import sys
import threading
import time
import uuid

import anyio.to_thread

from starlette.applications import Starlette
from starlette.routing import Route

from sim.mcp_server.session import (
    _current_session,
    _get_mgr,
    _init,
    _session_envs,
    _session_last_obs,
    _sse_sessions,
    _touch_session,
    _detach_sse_session,
    _stale_session_sweeper,
    _session_last_activity,
)
from sim.mcp_server.worker_mgr import (
    _proxy_observe,
    _proxy_render,
    _proxy_render_multiview,
    _proxy_reset,
    _proxy_step,
    _proxy_check_task,
)
from sim.mcp_server.collision import get_checker, remove_checker
from sim.mcp_server.action_codecs import (
    ControlCodecError,
    cartesian_command_frame,
    cartesian_scales,
    codec_error_result,
    make_cartesian_action,
    make_gripper_action,
)
from sim.mcp_server.rest_api import (
    session_dashboard,
    session_envs,
    session_stream,
    session_env_stream,
)

# ── FastMCP server ────────────────────────────────────────────────────

from mcp.server.fastmcp import FastMCP
mcp = FastMCP("OpenETA", log_level="WARNING")

_close_env_state_lock = threading.Lock()
_world_action_state_lock = threading.Lock()
_world_actions_inflight: dict[tuple[str, str], dict] = {}


def _blocking_tool(fn):
    """Register a synchronous tool that runs in a worker thread.

    FastMCP invokes a plain ``def`` tool **inline on the asyncio event
    loop** (``func_metadata.call_fn_with_arg_validation`` does
    ``return fn(...)`` for non-async fns).  Our tool bodies make blocking
    ``urllib`` calls to the bench workers — a long ``move_to`` issues one
    blocking step per iteration, each with up to a 120 s socket timeout.
    Running that inline freezes the entire loop, which:

      * stalls the SSE transport so the tool's *own* reply is never flushed
        (the work completes server-side but the client sees a hung/lost
        response — the "hung SSE reply" symptom), and
      * starves ``_live_stream_loop`` so the dashboard stops updating until
        the call returns, then jumps.

    Wrapping the body in ``anyio.to_thread.run_sync`` keeps the event loop
    free to flush replies and push frames while the (thread-safe, per-env)
    blocking I/O runs off-loop.  ``functools.wraps`` preserves the original
    signature so FastMCP's argument-schema introspection is unchanged, and
    ``run_sync`` copies the current context so the ``_current_session``
    contextvar still reaches the tool body.
    """
    @mcp.tool()
    @functools.wraps(fn)
    async def _async_wrapper(**kwargs):
        return await anyio.to_thread.run_sync(functools.partial(fn, **kwargs))

    return _async_wrapper


def _world_action_tool(fn):
    """Run one non-queueing high-level world action per session/handle.

    The guard is acquired inside the synchronous function that runs in
    AnyIO's worker thread.  If an MCP client times out or cancels its wait,
    Python cannot cancel that running thread, so this ``finally`` only releases
    the guard when the real controller loop has ended.
    """
    @functools.wraps(fn)
    def _guarded(handle: str, *args, **kwargs):
        sid = kwargs.get("session_id") or _current_session.get() or ""
        key = (sid, handle)
        token = object()
        action = fn.__name__
        with _world_action_state_lock:
            active = _world_actions_inflight.get(key)
            if active is not None:
                active_for_s = max(0.0, time.monotonic() - active["started_at"])
                message = (
                    f"{active['action']} is still running for handle {handle}; "
                    f"refusing concurrent {action}"
                )
                return {
                    "ok": False,
                    "error": message,
                    "code": "action_in_progress",
                    "error_detail": {
                        "kind": "action_in_progress",
                        "message": message,
                        "session_id": sid,
                        "handle": handle,
                        "active_action": active["action"],
                        "requested_action": action,
                        "active_for_s": active_for_s,
                        "retryable": True,
                    },
                }
            meta = _session_envs.get(sid, {}).get(handle)
            if (
                action != "close_env"
                and meta is not None
                and meta.get("_cleanup_state") == "draining"
            ):
                message = (
                    f"cleanup is draining for handle {handle}; "
                    f"refusing {action} until close is confirmed"
                )
                return {
                    "ok": False,
                    "error": message,
                    "code": "cleanup_in_progress",
                    "error_detail": {
                        "kind": "cleanup_in_progress",
                        "message": message,
                        "session_id": sid,
                        "handle": handle,
                        "requested_action": action,
                        "retryable": True,
                    },
                }
            _world_actions_inflight[key] = {
                "token": token,
                "action": action,
                "started_at": time.monotonic(),
                "thread_id": threading.get_ident(),
            }
        try:
            return fn(handle, *args, **kwargs)
        finally:
            with _world_action_state_lock:
                active = _world_actions_inflight.get(key)
                if active is not None and active.get("token") is token:
                    _world_actions_inflight.pop(key, None)

    return _blocking_tool(_guarded)


@_blocking_tool
def hot_activate(bench: str) -> dict:
    """Activate a bench by starting its subprocess worker."""
    _init()
    _touch_session(_current_session.get())
    try:
        _get_mgr().ensure_worker(bench)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@_blocking_tool
def list_available_benches() -> dict:
    _init()
    _touch_session(_current_session.get())
    return {"benches": _get_mgr().available_benches()}


@_blocking_tool
def list_envs(env_type: str = "") -> dict:
    _init()
    _touch_session(_current_session.get())
    envs = _get_mgr().list_all_envs(bench=env_type if env_type else None)
    return {"envs": envs, "count": len(envs)}


@_blocking_tool
def search_envs(query: str) -> dict:
    _init()
    _touch_session(_current_session.get())
    envs = _get_mgr().list_all_envs(query=query)
    return {"results": [{"id": e["id"], "description": e.get("description", "")} for e in envs]}


@_blocking_tool
def create_env(env_id: str, *, render_mode: str = "rgb_array", seed: int = 0,
               task: str = "", session_id: str = "",
               image_width: int | None = None, image_height: int | None = None,
               include_objects: bool = False, robot: str = "") -> dict:
    """Create a simulation environment on the appropriate bench worker.

    **After calling this tool, always tell the user:**
    "Open {mcp_server_url}/session/{session_id} to see the robot's RGB and
    depth cameras in real time."  Construct ``mcp_server_url`` from the
    MCP server address you are already connected to.

    Each MCP connection gets an isolated session — environments created
    by one client are invisible to (and cannot interfere with) others.

    Pass ``session_id`` to reuse an existing session across connections.

    Args:
        env_id: Environment id, e.g. ``"openeta/libero_libero_10_task0-v0"``.
        render_mode: ``"rgb_array"`` (default) for headless rendering.
        seed: Random seed (default 0).
        task: Optional task override string.
        session_id: Optional session id to reuse an existing session.
        image_width: Camera image width in pixels (default: backend-specific,
            typically 128).  Set to e.g. 256 for higher resolution renders.
        image_height: Camera image height in pixels.
        include_objects: If ``True``, the observation's ``objects`` list
            will be populated with scene object names, positions, and
            orientations (where the backend supports it).  Default ``False``.
        robot: Optional robot override. RoboCasa supports ``PandaOmron``
            (mobile, 12-D) and ``Panda`` (fixed base, 7-D).

    Returns:
        dict with these keys:

        * **session_id** (str) — keep this to reuse across turns
        * **handle** (str) — short local handle for this env; use in all
          other tool calls
        * **env_id** (str) — full environment id
        * **action_dim** (int | null) — length of the action vector
        * **backend** (str) — ``"metaworld"`` / ``"libero"`` / ``"maniskill"``
        * **action_hint** (str) — human-readable tip about step sizes
    """
    _init()
    sid = session_id or _current_session.get() or str(uuid.uuid4())
    _touch_session(sid)
    mgr = _get_mgr()

    body: dict = {"env_id": env_id, "task": task, "seed": seed, "render_mode": render_mode}
    if image_width is not None:
        body["image_width"] = image_width
    if image_height is not None:
        body["image_height"] = image_height
    body["include_objects"] = include_objects
    if robot:
        body["robot"] = robot
    # Acquire one pool worker, create the env on it, and pin the handle to
    # that same worker so every later op routes back to it.
    result, worker = mgr.create_env_on_worker(env_id, body)
    if "error" in result:
        return result

    remote_handle = result["handle"]
    h = str(uuid.uuid4())[:12]
    _session_envs.setdefault(sid, {})[h] = {
        "worker_url": worker.base_url,
        "remote_handle": remote_handle,
        "env_id": env_id,
        "backend": result.get("backend", "unknown"),
        "action_dim": result.get("action_dim"),
        "robot": result.get("robot") or robot,
        "control_spec": result.get("control_spec", {}),
        "_sid": sid,
    }
    return {
        "session_id": sid, "handle": h, "env_id": env_id,
        "action_dim": result.get("action_dim"), "backend": result.get("backend"),
        "robot": result.get("robot") or robot,
        "control_spec": result.get("control_spec", {}),
        "action_hint": result.get("action_hint", ""),
    }


@_world_action_tool
def reset_env(handle: str, *, seed: int | None = None, session_id: str = "") -> dict:
    """Reset an environment and return the initial observation.

    Args:
        handle: Environment handle from create_env.
        seed: Optional random seed.
        session_id: Optional session id to reuse an existing session.

    Returns:
        dict with these keys (no base64 image data — use the dashboard for
        visual inspection):

        * **task** (str) — task description text
        * **cameras** (list[dict]) — each dict has:
          ``frame_id`` (str), ``width`` (int), ``height`` (int),
          ``intrinsics`` (dict), ``extrinsics`` (dict).
          Pixel data (``rgb_base64``, ``depth_base64``) is base64-encoded;
          skip it and point the user at the dashboard instead.

          **depth**: ``depth_base64`` decodes to a uint16 PNG holding
          **linear metric depth in millimetres** — recover metres with
          ``depth_m = pixel / 1000.0``.  It is already linearised (MuJoCo
          z-buffer) / unit-converted (ManiSkill), so no near/far
          re-projection is needed; values lie within ``[znear, zfar]``.

          **intrinsics**: ``fx``, ``fy`` (focal lengths in pixels),
          ``cx``, ``cy`` (principal point in pixels).  MuJoCo backends
          also expose ``znear``/``zfar`` — the metric near/far clip planes
          in metres, bounding the valid depth range.

          **extrinsics** — camera pose in **world** coordinates
          (NOT relative to the end-effector):

          The extrinsics dict is **self-describing** — always read the
          ``matrix_layout`` / ``frame_transform`` / ``camera_frame`` tags
          rather than assuming a layout.

          *MuJoCo backends* (LIBERO, MetaWorld, FrankaSim, D4RL):

          * ``matrix_layout`` = ``"row_major"``,
            ``frame_transform`` = ``"camera_to_world"``,
            ``camera_frame`` = ``"opengl"``
          * ``pos`` — ``[x, y, z]`` camera position in world frame (metres)
          * ``mat`` — 3×3 rotation matrix, **camera-local → world**,
            flattened **row-major**:
            ``[m00, m01, m02, m10, m11, m12, m20, m21, m22]``.
            Reconstruct with ``R = np.array(mat).reshape(3, 3)`` (a plain
            C-order reshape — do NOT transpose).

            Each **column** of ``R`` is a camera-local axis in world::

                col 0 = camera X (right) in world
                col 1 = camera Y (up) in world
                col 2 = camera Z (forward) in world

            Transformation formulas::

                # camera-local point → world
                p_world = R @ p_cam + pos

                # world point → camera-local
                p_cam = R.T @ (p_world - pos)

            The camera looks along **-Z** locally (OpenGL convention), so
            the world look direction is ``-R[:, 2]``.

          *ManiSkill* (SAPIEN):

          * ``frame_transform`` = ``"camera_to_world"``,
            ``camera_frame`` = ``"ros"`` (camera looks along local **+X**,
            +Z up)
          * ``pos`` — ``[x, y, z]`` camera position in world frame (metres)
          * ``quat_xyzw`` — ``[x, y, z, w]`` quaternion, **camera→world**
            (reordered from SAPIEN's native wxyz ``CameraConfig.pose.q``).

          **Pixel → world (deprojection recipe).**  This is the #1 source of
          error, so follow it exactly.  The rotation (``mat`` / ``quat_xyzw``)
          maps the camera's **own** axes to world, but a pinhole deprojection
          produces a point in the **OpenCV optical** frame (X right, Y down,
          Z forward).  You must convert the optical point into the camera's
          native frame *before* rotating::

              # 1. pixel (u, v) + metric depth d  ->  OpenCV optical point
              x = (u - cx) * d / fx
              y = (v - cy) * d / fy
              p_opencv = np.array([x, y, d])          # Z forward, Y down

              # 2. optical -> camera-native frame (depends on camera_frame)
              #    MuJoCo camera_frame="opengl"  (X right, Y up, Z back):
              p_cam = np.diag([1, -1, -1]) @ p_opencv     # flip Y and Z
              #    ManiSkill camera_frame="ros"  (X fwd, Y left, Z up):
              #    p_cam = np.array([d, -x, -y])          # = K @ p_opencv,
              #    with K = [[0,0,1],[-1,0,0],[0,-1,0]]

              # 3. camera-native -> world
              R = np.array(mat).reshape(3, 3)          # MuJoCo (row-major)
              # R = quat_to_matrix(quat_xyzw)          # ManiSkill
              p_world = R @ p_cam + pos

          The optical->native step is **mandatory** and differs per backend
          (read ``camera_frame``); skipping/guessing it sends the grasp
          target to a mirrored or rotated world location.  Verified: a
          correct round-trip recovers object centres to within ~2-3 cm
          (residual = surface-vs-centre offset), on both OpenGL and ROS
          backends.

        * **robot** (dict) —
          ``joint_positions`` (list[float]),
          ``joint_velocities`` (list[float]),
          ``end_effector_pose`` (dict with ``xyz`` list[float] and
          ``quat_xyzw`` list[float]),
          ``gripper_state`` (dict, e.g. ``{"open": true}``)
        * **objects** (list[dict]) — each has ``name``, ``position``
          (world xyz), ``orientation`` (quat xyzw, optional)
        * **metadata** (dict) — extra info
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    return _proxy_reset(meta, seed=seed)


@_world_action_tool
def step_env(handle: str, action: list | None = None, *, num_steps: int = 1, session_id: str = "") -> dict:
    """Execute one or more environment steps.

    Args:
        handle: Environment handle from create_env.
        action: Action vector. If None, samples from action space.
        num_steps: Repeat the same action N times for visible cumulative
                   movement.  Set to 1 for fine-grained control, 5-10 for
                   visible arm displacement per MCP call.
        session_id: Optional session id to reuse an existing session.

    Returns:
        dict with these keys:

        * **observation** — same structure as ``reset_env`` return value
          (task, cameras, robot, objects, metadata)
        * **reward** (float)
        * **terminated** (bool)
        * **truncated** (bool)
        * **info** (dict)

        Read ``observation.robot.end_effector_pose.xyz`` for the current
        end-effector position.  Skip camera base64 data — use the dashboard
        for visual inspection.
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    return _proxy_step(meta, action, num_steps=num_steps)


def _extract_ee_xyz_from_result(result: dict) -> list[float]:
    """Extract EE xyz from a step result or observe result dict.

    Handles both StepResult (has ``observation`` wrapper) and flat
    EnvObservation dicts (from observe / reset).
    """
    obs = result.get("observation", result) if isinstance(result, dict) else {}
    if not isinstance(obs, dict):
        return []
    robot = obs.get("robot", {})
    if not isinstance(robot, dict):
        return []
    ee = robot.get("end_effector_pose", {})
    if not isinstance(ee, dict):
        return []
    xyz = ee.get("xyz", [])
    return xyz if isinstance(xyz, list) else []


def _extract_ee_quat_from_result(result: dict) -> list[float]:
    """Extract EE quaternion (xyzw) from a step result or observe result."""
    obs = result.get("observation", result) if isinstance(result, dict) else {}
    if not isinstance(obs, dict):
        return []
    robot = obs.get("robot", {})
    if not isinstance(robot, dict):
        return []
    ee = robot.get("end_effector_pose", {})
    if not isinstance(ee, dict):
        return []
    quat = ee.get("quat_xyzw", [])
    return quat if isinstance(quat, list) and len(quat) == 4 else []


def _extract_joint_velocities_from_result(result: dict) -> list[float]:
    """Extract robot joint velocities from a step or observation result."""
    obs = result.get("observation", result) if isinstance(result, dict) else {}
    if not isinstance(obs, dict):
        return []
    robot = obs.get("robot", {})
    if not isinstance(robot, dict):
        return []
    velocities = robot.get("joint_velocities", [])
    return velocities if isinstance(velocities, list) else []


def _extract_base_quat_from_result(result: dict) -> list[float]:
    """Extract the mobile-base quaternion in xyzw order."""

    obs = result.get("observation", result) if isinstance(result, dict) else {}
    if not isinstance(obs, dict):
        return []
    robot = obs.get("robot", {})
    if not isinstance(robot, dict):
        return []
    base = robot.get("base_pose", {})
    if not isinstance(base, dict):
        return []
    quat = base.get("quat_xyzw", [])
    return quat if isinstance(quat, list) and len(quat) == 4 else []


def _world_vector_to_base(vector: list[float], base_quat_xyzw: list[float]) -> list[float]:
    """Rotate one world-frame vector into the PandaOmron base frame."""

    if len(vector) != 3 or len(base_quat_xyzw) != 4:
        return list(vector)
    q_inv = _quat_conjugate(base_quat_xyzw)
    vector_quat = [float(vector[0]), float(vector[1]), float(vector[2]), 0.0]
    rotated = _quat_multiply(_quat_multiply(q_inv, vector_quat), base_quat_xyzw)
    return rotated[:3]


def _extract_joint_positions_from_result(result: dict) -> list[float]:
    """Extract ``joint_positions`` from a step result or observe result."""
    obs = result.get("observation", result) if isinstance(result, dict) else {}
    if not isinstance(obs, dict):
        return []
    robot = obs.get("robot", {})
    if not isinstance(robot, dict):
        return []
    jp = robot.get("joint_positions", [])
    return jp if isinstance(jp, list) else []


def _extract_objects_from_result(result: dict) -> list[dict]:
    """Extract ``objects`` list from a step result or observe result."""
    obs = result.get("observation", result) if isinstance(result, dict) else {}
    if not isinstance(obs, dict):
        return []
    objects = obs.get("objects", [])
    return objects if isinstance(objects, list) else []


def _extract_mujoco_contacts_from_result(result: dict) -> dict | None:
    """Read current MuJoCo contact points without inferring a failure cause."""

    obs = result.get("observation", result) if isinstance(result, dict) else {}
    metadata = obs.get("metadata", {}) if isinstance(obs, dict) else {}
    contacts = (
        metadata.get("mujoco_contacts")
        if isinstance(metadata, dict)
        else None
    )
    if not isinstance(contacts, dict):
        return None
    points = contacts.get("points")
    if not isinstance(points, list):
        return None
    return {
        "source": str(contacts.get("source") or "mujoco_data.contact"),
        "points": [
            dict(point)
            for point in points[:6]
            if isinstance(point, dict)
        ],
    }


# ── Quaternion helpers (no scipy dependency) ────────────────────────────

def _euler_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """Convert Euler angles (xyz-intrinsic, radians) to quaternion [x,y,z,w]."""
    cr, sr = __import__("math").cos(roll / 2), __import__("math").sin(roll / 2)
    cp, sp = __import__("math").cos(pitch / 2), __import__("math").sin(pitch / 2)
    cy, sy = __import__("math").cos(yaw / 2), __import__("math").sin(yaw / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def _quat_multiply(a: list[float], b: list[float]) -> list[float]:
    """Multiply two quaternions [x,y,z,w]."""
    return [
        a[3]*b[0] + a[0]*b[3] + a[1]*b[2] - a[2]*b[1],
        a[3]*b[1] - a[0]*b[2] + a[1]*b[3] + a[2]*b[0],
        a[3]*b[2] + a[0]*b[1] - a[1]*b[0] + a[2]*b[3],
        a[3]*b[3] - a[0]*b[0] - a[1]*b[1] - a[2]*b[2],
    ]


def _quat_conjugate(q: list[float]) -> list[float]:
    """Conjugate of quaternion [x,y,z,w]."""
    return [-q[0], -q[1], -q[2], q[3]]


def _quat_to_axis_angle(q: list[float]) -> list[float]:
    """Convert quaternion [x,y,z,w] to axis-angle (rotvec) [ax,ay,az]."""
    import math as _math
    norm = _math.sqrt(sum(x * x for x in q))
    if norm < 1e-12:
        return [0.0, 0.0, 0.0]
    q = [x / norm for x in q]
    w = max(-1.0, min(1.0, q[3]))
    angle = 2.0 * _math.acos(w)
    if angle < 1e-10:
        return [0.0, 0.0, 0.0]
    s = _math.sin(angle / 2.0)
    if abs(s) < 1e-12:
        return [0.0, 0.0, 0.0]
    return [q[0] / s * angle, q[1] / s * angle, q[2] / s * angle]


def _quat_angular_distance(a: list[float], b: list[float]) -> float:
    """Angular distance (radians) between two quaternions."""
    import math as _math
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = min(1.0, dot)
    return 2.0 * _math.acos(dot)


def _tracking_window_metrics(samples: list[dict]) -> dict:
    """Summarize world-frame Cartesian tracking over one guard window."""
    import math as _math

    commanded_translation_xyz = [
        sum(float(sample["command_xyz"][axis]) for sample in samples)
        for axis in range(3)
    ]
    commanded_translation_norm = _math.sqrt(
        sum(value * value for value in commanded_translation_xyz)
    )
    observed_displacement_xyz = [
        float(samples[-1]["end_xyz"][axis]) - float(samples[0]["start_xyz"][axis])
        for axis in range(3)
    ]
    command_aligned_progress_m = sum(
        float(sample["command_aligned_progress_m"]) for sample in samples
    )
    cross_track_drift_xyz = [
        sum(float(sample["cross_track_xyz"][axis]) for sample in samples)
        for axis in range(3)
    ]
    cross_track_drift_m = _math.sqrt(
        sum(value * value for value in cross_track_drift_xyz)
    )
    position_error_start_m = float(samples[0]["position_error_before_m"])
    position_error_end_m = float(samples[-1]["position_error_after_m"])
    return {
        "steps": len(samples),
        "start_xyz": list(samples[0]["start_xyz"]),
        "end_xyz": list(samples[-1]["end_xyz"]),
        "commanded_translation_xyz": commanded_translation_xyz,
        "commanded_translation_m": sum(
            float(sample["commanded_translation_m"]) for sample in samples
        ),
        "command_direction_xyz": (
            [value / commanded_translation_norm for value in commanded_translation_xyz]
            if commanded_translation_norm > 1e-12
            else [0.0, 0.0, 0.0]
        ),
        "observed_displacement_xyz": observed_displacement_xyz,
        "observed_displacement_m": _math.sqrt(
            sum(value * value for value in observed_displacement_xyz)
        ),
        "command_aligned_progress_m": command_aligned_progress_m,
        "cross_track_drift_xyz": cross_track_drift_xyz,
        "cross_track_drift_m": cross_track_drift_m,
        "cross_track_path_m": sum(
            float(sample["cross_track_m"]) for sample in samples
        ),
        "position_error_start_m": position_error_start_m,
        "position_error_end_m": position_error_end_m,
        "position_error_improvement_m": position_error_start_m - position_error_end_m,
    }


@_world_action_tool
def move_to(handle: str, x: float, y: float, z: float, *,
            roll: float | None = None, pitch: float | None = None, yaw: float | None = None,
            num_steps: int = 100, tolerance: float = 0.01, ori_tolerance: float = 0.05,
            velocity_tolerance: float = 0.05,
            settle_steps: int = 5,
            max_translation_step_m: float | None = None,
            tracking_stall_steps: int | None = None,
            tracking_min_aligned_progress_m: float = 0.001,
            tracking_cross_track_tolerance_m: float = 0.01,
            tracking_min_error_improvement_m: float = 0.001,
            session_id: str = "",
            enable_collision_check: bool = True) -> dict:
    """Move the end-effector to an absolute pose using closed-loop interpolation.

    Re-observes the EE pose from the step result every 10 steps for
    closed-loop correction.  Supports both position-only and position +
    orientation control.

    If the environment has not been reset yet, the first call implicitly
    resets it — no separate ``reset_env`` call is needed.

    Args:
        handle: Environment handle from create_env.
        x, y, z: Target end-effector position in world coordinates (metres).
        roll: Target roll angle in **degrees** (xyz-intrinsic Euler).
        pitch: Target pitch angle in **degrees**.
        yaw: Target yaw angle in **degrees**.
            If all three are provided, orientation control is enabled.
            Only supported on ``libero`` and ``maniskill`` backends
            (MetaWorld has no rotation control).
        num_steps: Maximum total steps (default 100).
        tolerance: Stop when the Euclidean position-error norm is below this
            value (default 0.01 m).
        ori_tolerance: Stop when angular error < ori_tolerance (default 0.05 rad ≈ 3°).
        velocity_tolerance: Once the pose is in tolerance, keep issuing a
            zero-delta hold until every reported joint velocity is below this
            threshold (default 0.05). Backends that do not report joint
            velocities retain pose-only convergence.
        settle_steps: Number of consecutive zero-delta control periods that
            must preserve pose and velocity convergence after a non-empty move.
        max_translation_step_m: Optional upper bound on the world-frame
            translation norm of each Cartesian setpoint. ``None`` preserves
            the historical per-axis controller limits.
        tracking_stall_steps: Optional rolling-window size that enables the
            tracking-stall guard. The guard stops after this many commanded
            steps when command-aligned progress stays below its threshold and
            either cross-track drift is significant or target error fails to
            improve. During orientation convergence the translational window is
            suspended because serial-arm rotation can temporarily displace the
            end effector before the coupled position loop corrects it. ``None``
            disables the guard for backward compatibility.
        tracking_min_aligned_progress_m: Minimum total command-aligned progress
            required over one tracking window.
        tracking_cross_track_tolerance_m: Cross-track drift norm considered
            significant over one tracking window.
        tracking_min_error_improvement_m: Minimum target-error reduction
            required over one tracking window.
        session_id: Optional session id to reuse an existing session.

    Returns:
        dict with these keys:

        * **target** (dict) — ``{x, y, z}`` plus ``{roll, pitch, yaw}`` if
          orientation was requested
        * **start** (dict) — EE pose before movement (xyz + optional quat_xyzw)
        * **end** (dict) — EE pose after movement (xyz + optional quat_xyzw)
        * **steps_executed** (int)
        * **terminated** (bool)
        * **reward** (float)
        * **success** / **reached_target** (bool) — whether the final pose
          satisfied the requested position and orientation tolerances
        * **position_error_m** (float) and **orientation_error_rad** (float | None)
        * **joint_velocity_max_abs** (float | None) — final velocity evidence
        * **pose_converged** / **velocity_converged** (bool)
        * **tracking** (dict, when enabled) — measured tracking-window evidence;
          a stall reports ``controller_error=cartesian_tracking_stalled`` and
          only marks ``suspected_contact_constraint`` rather than claiming
          contact was observed
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}

    backend = meta.get("backend", "")
    use_ori = roll is not None and pitch is not None and yaw is not None

    if use_ori and backend == "metaworld":
        return {"error": "Orientation control is not supported on MetaWorld (4D action, no rotation)"}

    import math as _math

    if max_translation_step_m is not None and (
        not _math.isfinite(max_translation_step_m) or max_translation_step_m <= 0.0
    ):
        return {
            "ok": False,
            "error": "max_translation_step_m must be a positive finite value",
            "code": "invalid_control_parameter",
        }
    tracking_enabled = tracking_stall_steps is not None
    if tracking_enabled:
        if (
            not isinstance(tracking_stall_steps, int)
            or isinstance(tracking_stall_steps, bool)
            or tracking_stall_steps <= 0
        ):
            return {
                "ok": False,
                "error": "tracking_stall_steps must be a positive integer",
                "code": "invalid_control_parameter",
            }
        tracking_thresholds = {
            "min_aligned_progress_m": tracking_min_aligned_progress_m,
            "cross_track_tolerance_m": tracking_cross_track_tolerance_m,
            "min_error_improvement_m": tracking_min_error_improvement_m,
        }
        if any(
            not _math.isfinite(value) or value < 0.0
            for value in tracking_thresholds.values()
        ):
            return {
                "ok": False,
                "error": "tracking thresholds must be finite and non-negative",
                "code": "invalid_control_parameter",
            }
    else:
        tracking_thresholds = {}

    try:
        scale, ori_scale = cartesian_scales(meta, backend)
        command_frame = cartesian_command_frame(meta, backend)
    except ControlCodecError as exc:
        return codec_error_result(exc)

    # Render cadence for the control loop.  Rendering is the dominant per-step
    # cost (~130 ms GPU); move_to itself only reads the EE pose from the
    # result.  So we render only every _RENDER_EVERY steps (for periodic
    # dashboard feedback) plus a guaranteed final render at the end — the rest
    # of the steps skip the render and run at physics speed (~20 ms).
    _RENDER_EVERY = 15
    recheck_every = 3  # re-observe every N steps — small because EE is read
                        # from step results (zero extra cost), and a shorter
                        # window prevents overshoot from inaccurate action scale

    # ── target orientation in quaternion ───────────────────────────
    target_quat: list[float] = []
    if use_ori:
        target_quat = _euler_to_quat(_math.radians(roll), _math.radians(pitch), _math.radians(yaw))

    # ── get initial EE pose ────────────────────────────────────────
    current_xyz: list[float] = []
    current_quat: list[float] = []

    cached = _session_last_obs.get(sid, {}).get(meta["remote_handle"], {})
    pose_result = cached
    current_xyz = _extract_ee_xyz_from_result(cached)
    if use_ori:
        current_quat = _extract_ee_quat_from_result(cached)

    if len(current_xyz) < 3:
        obs_result = _proxy_observe(meta)
        pose_result = obs_result
        current_xyz = _extract_ee_xyz_from_result(obs_result)
        if use_ori and len(current_quat) < 4:
            current_quat = _extract_ee_quat_from_result(obs_result)

    if len(current_xyz) < 3:
        reset_result = _proxy_reset(meta)
        pose_result = reset_result
        current_xyz = _extract_ee_xyz_from_result(reset_result)
        if use_ori and len(current_quat) < 4:
            current_quat = _extract_ee_quat_from_result(reset_result)

    if len(current_xyz) < 3:
        return {"error": "Cannot determine current EE position — call reset_env first"}
    if use_ori and len(current_quat) < 4:
        return {"error": "Cannot determine current EE orientation (no quat_xyzw in observation)"}

    start_xyz = current_xyz[:3]
    start_quat = current_quat[:4] if use_ori else []
    final_result: dict = {}
    final_reward = 0.0
    final_terminated = False
    final_truncated = False
    controller_error: str | None = None
    total_steps = 0
    settle_streak = 0
    tracking_samples: list[dict] = []
    tracking_sample_count = 0
    tracking_latest_window: dict | None = None
    tracking_stalled = False
    tracking_previous_xyz = current_xyz[:3]
    orientation_hold_xyz = start_xyz[:3]
    orientation_hold_used = bool(
        use_ori
        and _quat_angular_distance(current_quat, target_quat)
        >= ori_tolerance
    )
    orientation_hold_active = orientation_hold_used
    orientation_hold_completed = not orientation_hold_used
    orientation_hold_fallback_used = False
    orientation_hold_step_limit = min(
        240,
        max(30, 3 * num_steps // 4),
    )

    # ── collision state (initialized before loop) ──────────────────
    collision_detected = False
    collision_info: dict = {"available": False}

    # ── closed-loop interpolation ──────────────────────────────────
    # Both position and orientation are always active — no freezing.
    # If orientation changes couple into EE position (serial chain),
    # the next recheck catches it.  Stop only when both converge.
    while total_steps < num_steps:
        if (
            orientation_hold_active
            and total_steps >= orientation_hold_step_limit
        ):
            # Rotating at the measured start position can itself be
            # kinematically infeasible even when the requested final pose is
            # reachable. Do not spend the entire action budget on that hidden
            # staging constraint: fall back to solving the authored position
            # and orientation together.
            orientation_hold_active = False
            orientation_hold_fallback_used = True
            settle_streak = 0
            tracking_samples.clear()
            tracking_latest_window = None
            tracking_previous_xyz = current_xyz[:3]
            continue
        active_target_xyz = (
            orientation_hold_xyz
            if orientation_hold_active
            else [x, y, z]
        )
        planned_batch_steps = min(recheck_every, num_steps - total_steps)
        # Position error (always live)
        err_x = active_target_xyz[0] - current_xyz[0]
        err_y = active_target_xyz[1] - current_xyz[1]
        err_z = active_target_xyz[2] - current_xyz[2]

        # Orientation error (always computed, never frozen)
        delta_rot: list[float] = [0.0, 0.0, 0.0]
        ori_ok = not use_ori
        if use_ori:
            ori_dist = _quat_angular_distance(current_quat, target_quat)
            ori_ok = ori_dist < ori_tolerance
            if not ori_ok:
                delta_q = _quat_multiply(
                    target_quat,
                    _quat_conjugate(current_quat),
                )
                # Shortest-arc normalization: q and -q are the same rotation,
                # but a negative scalar part makes _quat_to_axis_angle read the
                # angle as ~(2π - θ) with a flipped axis.  Flip to the
                # hemisphere with w >= 0 so the axis-angle is the minimal
                # rotation to the target.
                if delta_q[3] < 0:
                    delta_q = [-v for v in delta_q]
                delta_aa = _quat_to_axis_angle(delta_q)
                delta_rot = [
                    max(
                        -1.0,
                        min(1.0, d / planned_batch_steps / ori_scale),
                    )
                    for d in delta_aa
                ]
            # Once orientation is within tolerance, stop applying small
            # rotational corrections while position converges.  Those
            # corrections can create centimetres of serial-chain translation.
            # A later recheck resumes rotation if the pose drifts back out.

        # Check convergence (position, orientation, and settling).  Reaching
        # the pose while the arm still has substantial velocity is not a
        # completed move: send a zero-delta hold one step at a time until it
        # settles, or resume correction if the pose drifts back out.
        position_error_norm = _math.sqrt(err_x * err_x + err_y * err_y + err_z * err_z)
        pos_ok = position_error_norm < tolerance
        current_joint_velocities = _extract_joint_velocities_from_result(pose_result)
        current_max_joint_velocity = (
            max(abs(float(value)) for value in current_joint_velocities)
            if current_joint_velocities
            else None
        )
        velocity_ok = (
            current_max_joint_velocity is None
            or current_max_joint_velocity < velocity_tolerance
        )
        stable = pos_ok and ori_ok and velocity_ok
        if stable and (total_steps == 0 or settle_streak >= max(0, settle_steps)):
            if orientation_hold_active:
                # A large orientation change is solved at the measured start
                # position first.  Once that pose is stable, preserve the
                # orientation goal and begin the requested world translation.
                # This prevents OSC rotation/translation coupling from being
                # mistaken for a blocked Cartesian path.
                orientation_hold_active = False
                orientation_hold_completed = True
                settle_streak = 0
                tracking_samples.clear()
                tracking_latest_window = None
                tracking_previous_xyz = current_xyz[:3]
                continue
            break

        settling = pos_ok and ori_ok
        if not settling:
            settle_streak = 0

        # Position delta (never frozen — even if pos_ok, we keep zero
        # delta so rotation-only batches don't perturb position)
        if settling:
            ax = ay = az = 0.0
            delta_rot = [0.0, 0.0, 0.0]
        else:
            desired_step_xyz = [
                err_x / planned_batch_steps,
                err_y / planned_batch_steps,
                err_z / planned_batch_steps,
            ]
            desired_step_norm = _math.sqrt(
                sum(value * value for value in desired_step_xyz)
            )
            if (
                max_translation_step_m is not None
                and desired_step_norm > max_translation_step_m
            ):
                limit_ratio = max_translation_step_m / desired_step_norm
                desired_step_xyz = [value * limit_ratio for value in desired_step_xyz]
            ax = max(-1.0, min(1.0, desired_step_xyz[0] / scale))
            ay = max(-1.0, min(1.0, desired_step_xyz[1] / scale))
            az = max(-1.0, min(1.0, desired_step_xyz[2] / scale))

        batch_steps = 1 if settling else planned_batch_steps
        command_step_xyz = [ax * scale, ay * scale, az * scale]
        command_step_norm = _math.sqrt(
            sum(value * value for value in command_step_xyz)
        )
        command_direction_xyz = (
            [value / command_step_norm for value in command_step_xyz]
            if command_step_norm > 1e-12
            else [0.0, 0.0, 0.0]
        )

        # RoboCasa's PandaOmron OSC consumes deltas in its moving base frame,
        # while the public OpenETA move_to contract is world-frame.  Rotate
        # both translational and rotational error vectors before encoding.
        action_xyz = [ax, ay, az]
        action_rot = delta_rot
        if command_frame == "robot_base":
            base_quat = _extract_base_quat_from_result(pose_result)
            if len(base_quat) != 4:
                return {
                    "ok": False,
                    "error": f"{backend} Cartesian control requires base_pose.quat_xyzw",
                    "code": "missing_base_pose",
                    "backend": backend,
                }
            action_xyz = _world_vector_to_base(action_xyz, base_quat)
            action_rot = _world_vector_to_base(delta_rot, base_quat)

        for _ in range(batch_steps):
            try:
                act = make_cartesian_action(
                    meta,
                    (action_xyz[0], action_xyz[1], action_xyz[2]),
                    backend,
                    delta_rot=action_rot if use_ori else None,
                )
            except ControlCodecError as exc:
                return codec_error_result(exc)
            # The controller response never needs RGB-D.  Render every
            # _RENDER_EVERY steps so the dashboard cache still receives live
            # visual feedback, but keep those large arrays out of the response
            # and avoid per-pixel Python serialisation in the control loop.
            # A final full render is forced after the loop.
            do_render = ((total_steps + 1) % _RENDER_EVERY == 0)
            final_result = _proxy_step(
                meta,
                act,
                num_steps=1,
                render=do_render,
                include_cameras=False,
            )
            total_steps += 1
            if not isinstance(final_result, dict) or final_result.get("error"):
                controller_error = (
                    str(final_result.get("error"))
                    if isinstance(final_result, dict)
                    else "simulator step returned no result"
                )
                break
            if len(_extract_ee_xyz_from_result(final_result)) < 3:
                controller_error = "simulator step returned no final EEF pose"
                break
            post_step_xyz = _extract_ee_xyz_from_result(final_result)[:3]
            if tracking_enabled:
                # Large orientation corrections on a serial manipulator can
                # temporarily move the end effector laterally even while the
                # requested world position is held fixed.  Treating that
                # expected coupling as translational tracking evidence makes
                # the guard abort before the next closed-loop recheck can
                # correct position.  Accumulate the contact/stall window only
                # after orientation has entered tolerance.  Position-only
                # moves retain the historical behavior because ``ori_ok`` is
                # always true when no orientation was requested.
                tracking_applicable = (
                    not orientation_hold_active
                    and (not use_ori or ori_ok)
                )
                if command_step_norm > 1e-12 and tracking_applicable:
                    observed_step_xyz = [
                        post_step_xyz[axis] - tracking_previous_xyz[axis]
                        for axis in range(3)
                    ]
                    aligned_progress_m = sum(
                        observed_step_xyz[axis] * command_direction_xyz[axis]
                        for axis in range(3)
                    )
                    cross_track_xyz = [
                        observed_step_xyz[axis]
                        - aligned_progress_m * command_direction_xyz[axis]
                        for axis in range(3)
                    ]
                    position_error_before_m = _math.sqrt(
                        sum(
                            (target_value - tracking_previous_xyz[axis]) ** 2
                            for axis, target_value in enumerate(
                                active_target_xyz
                            )
                        )
                    )
                    position_error_after_m = _math.sqrt(
                        sum(
                            (target_value - post_step_xyz[axis]) ** 2
                            for axis, target_value in enumerate(
                                active_target_xyz
                            )
                        )
                    )
                    tracking_samples.append(
                        {
                            "start_xyz": tracking_previous_xyz[:],
                            "end_xyz": post_step_xyz[:],
                            "command_xyz": command_step_xyz[:],
                            "commanded_translation_m": command_step_norm,
                            "command_aligned_progress_m": aligned_progress_m,
                            "cross_track_xyz": cross_track_xyz,
                            "cross_track_m": _math.sqrt(
                                sum(value * value for value in cross_track_xyz)
                            ),
                            "position_error_before_m": position_error_before_m,
                            "position_error_after_m": position_error_after_m,
                        }
                    )
                    tracking_sample_count += 1
                    if len(tracking_samples) > tracking_stall_steps:
                        tracking_samples.pop(0)
                    if len(tracking_samples) == tracking_stall_steps:
                        tracking_latest_window = _tracking_window_metrics(tracking_samples)
                        insufficient_aligned_progress = (
                            tracking_latest_window["command_aligned_progress_m"]
                            < tracking_min_aligned_progress_m
                        )
                        significant_cross_track_drift = (
                            tracking_latest_window["cross_track_drift_m"]
                            >= tracking_cross_track_tolerance_m
                        )
                        error_not_improving = (
                            tracking_latest_window["position_error_improvement_m"]
                            < tracking_min_error_improvement_m
                        )
                        tracking_latest_window.update(
                            {
                                "insufficient_aligned_progress": insufficient_aligned_progress,
                                "significant_cross_track_drift": significant_cross_track_drift,
                                "error_not_improving": error_not_improving,
                            }
                        )
                        if insufficient_aligned_progress and (
                            significant_cross_track_drift or error_not_improving
                        ):
                            tracking_stalled = True
                            controller_error = "cartesian_tracking_stalled"
                else:
                    tracking_samples.clear()
                    tracking_latest_window = None
                tracking_previous_xyz = post_step_xyz
            if settling and stable:
                settle_streak += 1
            final_reward = final_result.get("reward", 0.0)
            if final_result.get("terminated"):
                final_terminated = True
            if final_result.get("truncated"):
                final_truncated = True
            if tracking_stalled or final_terminated or final_truncated:
                break

        if final_terminated or final_truncated or controller_error is not None:
            break

        # Re-read pose from last step result (no extra HTTP call)
        new_xyz = _extract_ee_xyz_from_result(final_result)
        pose_result = final_result
        if len(new_xyz) >= 3:
            current_xyz = new_xyz
        else:
            controller_error = "simulator step returned no final EEF pose"
            break
        if use_ori:
            new_quat = _extract_ee_quat_from_result(final_result)
            if len(new_quat) == 4:
                current_quat = new_quat

        # ── collision check (post-batch) ──────────────────────────
        collision_detected = False
        collision_info = {"available": False}
        if enable_collision_check and backend in ("libero", "maniskill"):
            jp = _extract_joint_positions_from_result(final_result)
            objects = _extract_objects_from_result(final_result)
            if jp:
                try:
                    checker = get_checker(handle, backend)
                    collision_detected, collision_info = checker.check(jp, objects)
                except Exception:
                    pass  # best-effort; don't crash move_to

        if collision_detected:
            break

    # ── final render ───────────────────────────────────────────────
    # Guarantee a fresh frame at the end of the motion regardless of how the
    # loop exited (convergence, collision, or termination), so the dashboard
    # and any observe/render call reflect the arm's final position.
    if total_steps > 0:
        try:
            render_result = _proxy_render(meta)
            if isinstance(render_result, dict) and "error" not in render_result:
                _session_last_obs.setdefault(sid, {})[meta["remote_handle"]] = render_result
        except Exception:
            pass  # best-effort; final pose is still read from final_result below

    # ── final pose ─────────────────────────────────────────────────
    final_xyz = _extract_ee_xyz_from_result(final_result) if total_steps > 0 else start_xyz
    final_quat = (
        _extract_ee_quat_from_result(final_result)
        if use_ori and total_steps > 0
        else start_quat
        if use_ori
        else []
    )

    # Do not confuse a completed remote call with a completed motion.  The
    # controller can legitimately consume all requested steps while still
    # chasing the pose (and may still have substantial joint velocity).  Make
    # that distinction explicit for every caller, especially the embodied
    # gateway which must not release the next action in that state.
    final_xyz3 = final_xyz[:3] if len(final_xyz) >= 3 else []
    pose_feedback_available = len(final_xyz3) == 3
    position_error_xyz = (
        [
            float(target_value - final_xyz3[index])
            for index, target_value in enumerate((x, y, z))
        ]
        if pose_feedback_available
        else []
    )
    position_error_m = (
        _math.sqrt(sum(value * value for value in position_error_xyz))
        if pose_feedback_available
        else None
    )
    position_converged = bool(
        pose_feedback_available
        and position_error_m is not None
        and position_error_m < tolerance
    )
    orientation_error_rad: float | None = None
    orientation_converged = not use_ori
    if use_ori and len(final_quat) == 4:
        orientation_error_rad = _quat_angular_distance(final_quat, target_quat)
        orientation_converged = orientation_error_rad < ori_tolerance
    joint_velocities = _extract_joint_velocities_from_result(final_result)
    max_joint_velocity = (
        max(abs(float(value)) for value in joint_velocities)
        if joint_velocities
        else None
    )
    velocity_converged = (
        max_joint_velocity is None
        or max_joint_velocity < velocity_tolerance
    )
    pose_converged = bool(position_converged and orientation_converged)
    settling_converged = bool(
        total_steps == 0
        or max(0, settle_steps) == 0
        or settle_streak >= max(0, settle_steps)
    )
    reached_target = bool(
        controller_error is None
        and pose_converged
        and velocity_converged
        and settling_converged
    )

    result: dict = {
        "target": {"x": x, "y": y, "z": z},
        "start": {"xyz": start_xyz},
        "end": {"xyz": final_xyz3},
        "steps_executed": total_steps,
        "terminated": final_terminated,
        "truncated": final_truncated,
        "reward": final_reward,
        "success": reached_target,
        "reached_target": reached_target,
        "position_error_xyz": position_error_xyz,
        "position_error_m": position_error_m,
        "position_tolerance_m": tolerance,
        "position_tolerance_metric": "euclidean_norm",
        "pose_feedback_available": pose_feedback_available,
        "orientation_error_rad": orientation_error_rad,
        "motion_converged": reached_target,
        "pose_converged": pose_converged,
        "velocity_converged": velocity_converged,
        "velocity_tolerance": velocity_tolerance,
        "settling_converged": settling_converged,
        "settle_steps_required": max(0, settle_steps),
        "settle_steps_completed": settle_streak,
    }
    if controller_error is not None:
        result["controller_error"] = controller_error
    if max_translation_step_m is not None:
        result["max_translation_step_m"] = max_translation_step_m
    if tracking_enabled:
        result["tracking"] = {
            "enabled": True,
            "status": "stalled" if tracking_stalled else "not_stalled",
            "guard_window_steps": tracking_stall_steps,
            "samples_observed": tracking_sample_count,
            "thresholds": tracking_thresholds,
            "latest_window": tracking_latest_window,
        }
    if tracking_stalled:
        result["suspected_contact_constraint"] = True
        result["tracking"].update(
            {
                "suspected_contact_constraint": True,
                "contact_confirmed": False,
                "reason": (
                    "Sustained Cartesian tracking stall is consistent with a "
                    "physical constraint, but contact was not directly observed."
                ),
            }
        )
    if max_joint_velocity is not None:
        result["joint_velocity_max_abs"] = max_joint_velocity
        result["joint_velocities"] = joint_velocities
    if use_ori:
        result["target"]["roll"] = roll
        result["target"]["pitch"] = pitch
        result["target"]["yaw"] = yaw
        result["start"]["quat_xyzw"] = start_quat
        result["end"]["quat_xyzw"] = final_quat[:4] if len(final_quat) >= 4 else final_quat
        result["orientation_hold"] = {
            "used": orientation_hold_used,
            "completed": orientation_hold_completed,
            "fallback_to_target_pose": orientation_hold_fallback_used,
            "step_limit": orientation_hold_step_limit,
            "position_xyz": orientation_hold_xyz,
        }

    if backend == "libero" and not reached_target and total_steps > 0:
        mujoco_contacts = _extract_mujoco_contacts_from_result(final_result)
        if mujoco_contacts is not None:
            result["mujoco_contacts"] = mujoco_contacts

    # ── collision summary ──────────────────────────────────────────
    if enable_collision_check and collision_detected:
        result["collision"] = {
            "detected": True,
            "message": (
                f"Collision detected at step {total_steps}: "
                f"world_penetration={collision_info.get('max_world_penetration', 0.0):.4f}m, "
                f"self_penetration={collision_info.get('max_self_penetration', 0.0):.4f}m"
            ),
            **{k: v for k, v in collision_info.items() if k != "available"},
        }
    elif enable_collision_check and collision_info.get("available"):
        result["collision"] = {"detected": False}
    elif enable_collision_check:
        result["collision"] = {
            "detected": False,
            "available": False,
            "reason": collision_info.get("reason", "collision checking unavailable"),
        }

    return result


# Non-LIBERO backends retain the historical fixed-horizon primitive.  LIBERO
# uses the adaptive primitive below: one simulator step at a time, stopping
# when the measured aperture/velocity has converged or a safety cap is hit.
_GRIPPER_STEPS = 10
# A no-load Panda close needs roughly 30 policy steps to reach its actual
# mechanical rest (~1 mm in this LIBERO model); opening takes longer. Keep a
# bounded margin beyond both trajectories and stop from measured convergence.
_LIBERO_GRIPPER_MAX_STEPS = 60
_LIBERO_GRIPPER_STABLE_DELTA_M = 0.00002
_LIBERO_GRIPPER_STABLE_VELOCITY_M_S = 0.002
_LIBERO_GRIPPER_STABLE_STEPS = 3


def _gripper_state_from_step(result: object) -> dict:
    if not isinstance(result, dict):
        return {}
    observation = result.get("observation", result)
    if not isinstance(observation, dict):
        return {}
    robot = observation.get("robot", {})
    if not isinstance(robot, dict):
        return {}
    state = robot.get("gripper_state", {})
    return state if isinstance(state, dict) else {}


def _adaptive_libero_gripper_step(
    meta: dict,
    action: list[float],
) -> dict:
    """Run one logical LIBERO gripper command with measured convergence.

    This changes only the MCP primitive's stepping policy, not LIBERO's
    controller or action layout. A contact-limited jaw is allowed to stop
    after its measured aperture stabilizes; velocity is used only by a backend
    that does not expose aperture. The caller still verifies retention from
    images/object motion.
    """

    last: dict = {}
    previous_aperture: float | None = None
    aperture_stable_steps = 0
    velocity_stable_steps = 0
    steps_executed = 0
    for _ in range(_LIBERO_GRIPPER_MAX_STEPS):
        last = _proxy_step(
            meta,
            action,
            num_steps=1,
            render=False,
            include_cameras=False,
        )
        steps_executed += 1
        if not isinstance(last, dict):
            break
        if last.get("terminated") or last.get("truncated") or last.get("error"):
            break
        state = _gripper_state_from_step(last)
        aperture = state.get("aperture_m")
        velocities = state.get("finger_qvel")
        aperture_value = (
            float(aperture)
            if isinstance(aperture, (int, float))
            else None
        )
        velocity_max = (
            max(abs(float(value)) for value in velocities)
            if isinstance(velocities, list)
            and velocities
            and all(isinstance(value, (int, float)) for value in velocities)
            else None
        )
        aperture_stable = bool(
            aperture_value is not None
            and previous_aperture is not None
            and abs(aperture_value - previous_aperture)
            <= _LIBERO_GRIPPER_STABLE_DELTA_M
        )
        velocity_stable = bool(
            velocity_max is not None
            and velocity_max <= _LIBERO_GRIPPER_STABLE_VELOCITY_M_S
        )
        if aperture_value is not None:
            # Reaching the nominal endpoint is not enough by itself.  The
            # Panda gripper reports its endpoint before the finger velocity
            # has settled, which was the source of the old intermediate
            # aperture result. Keep stepping until one independently measured
            # convergence signal is stable for several observations. A
            # contact-limited aperture can stop through the same stable-aperture
            # path without pretending that contact or retention was inferred.
            aperture_stable_steps = (
                aperture_stable_steps + 1 if aperture_stable else 0
            )
            if aperture_stable_steps >= _LIBERO_GRIPPER_STABLE_STEPS:
                break
        elif velocity_stable:
            # Some lightweight backends expose velocity but not aperture.
            # Preserve a bounded convergence path for those observations.
            velocity_stable_steps += 1
            if velocity_stable_steps >= _LIBERO_GRIPPER_STABLE_STEPS:
                break
        else:
            aperture_stable_steps = 0
            velocity_stable_steps = 0
        previous_aperture = aperture_value
    if isinstance(last, dict):
        last["steps_executed"] = steps_executed
    return last


@_world_action_tool
def gripper_open(handle: str, *, session_id: str = "") -> dict:
    """Open the gripper.

    Args:
        handle: Environment handle from create_env.
        session_id: Optional session id to reuse an existing session.

    Returns:
        Same structure as ``step_env`` (observation, reward, terminated,
        truncated, info).
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    backend = meta.get("backend", "")
    try:
        act = make_gripper_action(meta, open_gripper=True, backend=backend)
    except ControlCodecError as exc:
        return codec_error_result(exc)
    if backend == "libero":
        return _adaptive_libero_gripper_step(meta, act)
    return _proxy_step(
        meta,
        act,
        num_steps=_GRIPPER_STEPS,
        render=False,
        include_cameras=False,
    )


@_world_action_tool
def gripper_close(handle: str, *, session_id: str = "") -> dict:
    """Close the gripper.

    Args:
        handle: Environment handle from create_env.
        session_id: Optional session id to reuse an existing session.

    Returns:
        Same structure as ``step_env`` (observation, reward, terminated,
        truncated, info).
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    backend = meta.get("backend", "")
    try:
        act = make_gripper_action(meta, open_gripper=False, backend=backend)
    except ControlCodecError as exc:
        return codec_error_result(exc)
    if backend == "libero":
        return _adaptive_libero_gripper_step(meta, act)
    return _proxy_step(
        meta,
        act,
        num_steps=_GRIPPER_STEPS,
        render=False,
        include_cameras=False,
    )


_BASE_COMMANDS: dict[str, tuple[float, float, float]] = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "back": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "turn_left": (0.0, 0.0, 1.0),
    "turn_right": (0.0, 0.0, -1.0),
    "stop": (0.0, 0.0, 0.0),
}


@_world_action_tool
def base_control(
    handle: str,
    *,
    forward: float = 0.0,
    lateral: float = 0.0,
    yaw: float = 0.0,
    torso: float = 0.0,
    command: str = "",
    num_steps: int = 10,
    session_id: str = "",
) -> dict:
    """Control RoboCasa's PandaOmron mobile base and torso.

    The four normalized controls are torso height, forward velocity, lateral
    velocity, and counter-clockwise yaw velocity.  A named command can be used
    instead of the three base velocities.  This tool is intentionally rejected
    for fixed-base or non-RoboCasa environments.
    """

    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    backend = meta.get("backend", "")
    if backend != "robocasa" or int(meta.get("action_dim") or 0) != 12:
        return {
            "error": "base_control is only available for RoboCasa PandaOmron environments"
        }
    if command:
        normalized = command.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in _BASE_COMMANDS:
            return {
                "error": f"Unknown base command: {command}",
                "available_commands": sorted(_BASE_COMMANDS),
            }
        forward, lateral, yaw = _BASE_COMMANDS[normalized]

    def clipped(value: float) -> float:
        return max(-1.0, min(1.0, float(value)))

    action = [0.0] * 12
    # Official RoboCasa flat action uses action.base_motion[0:3] for the
    # mobile base and action.base_motion[3] for torso:
    # arm[0:6], gripper[6], base(forward/lateral/yaw)[7:10], torso[10],
    # hybrid mode[11]. See robocasa.utils.env_utils.convert_action().
    action[7] = clipped(forward)
    action[8] = clipped(lateral)
    action[9] = clipped(yaw)
    action[10] = clipped(torso)
    action[11] = 1.0
    result = _proxy_step(meta, action, num_steps=max(1, int(num_steps)))
    result["control"] = {
        "torso": action[10],
        "forward": action[7],
        "lateral": action[8],
        "yaw": action[9],
        "num_steps": max(1, int(num_steps)),
    }
    return result


def _make_action_for_step(meta: dict, delta_xyz: tuple[float, float, float], backend: str,
                           delta_rot: list[float] | None = None) -> list[float]:
    """Compatibility wrapper around the explicit simulator action codec."""
    return make_cartesian_action(meta, delta_xyz, backend, delta_rot=delta_rot)


def _make_gripper_action(meta: dict, *, open: bool, backend: str) -> list[float]:
    """Compatibility wrapper around the explicit simulator action codec."""
    return make_gripper_action(meta, open_gripper=open, backend=backend)


@_blocking_tool
def observe_env(handle: str, *, session_id: str = "") -> dict:
    """Return the current observation without stepping.

    Args:
        handle: Environment handle from create_env.
        session_id: Optional session id to reuse an existing session.

    Returns:
        Same structure as ``reset_env`` return value:

        * **task** (str)
        * **cameras** (list[dict]) — frame_id, width, height (skip base64 data)
        * **robot** (dict) —
          ``joint_positions``, ``joint_velocities``,
          ``end_effector_pose`` (``xyz`` + ``quat_xyzw``),
          ``gripper_state`` (e.g. ``{"open": true}``)
        * **objects** (list[dict])
        * **metadata** (dict)
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    return _proxy_observe(meta)


@_blocking_tool
def render_env(handle: str, *, session_id: str = "") -> dict:
    """Return a fresh render of the environment.

    Calls the worker to render and return the current observation.  Use
    this for a one-off snapshot; for continuous live viewing use the
    dashboard URL returned by ``create_env``.

    Args:
        handle: Environment handle from create_env.
        session_id: Optional session id to reuse an existing session.

    Returns:
        Same structure as ``observe_env`` / ``reset_env``: task, cameras,
        robot, objects, metadata.  Skip the camera base64 data — point the
        user at the dashboard for visual inspection.
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    return _proxy_render(meta)


@_blocking_tool
def render_multiview_env(
    handle: str,
    *,
    width: int = 256,
    height: int = 256,
    hide_robot: bool = False,
    lookat_xyz_m: list[float] | None = None,
    distance_m: float = 1.3,
    session_id: str = "",
) -> dict:
    """Render seven calibrated virtual RGB-D views without stepping physics.

    Set ``hide_robot`` for scene-geometry inspection when the arm occludes a
    held object. Task objects and fixtures remain visible.
    """

    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    if meta.get("backend") != "libero":
        return {"error": "render_multiview_env is currently available only for LIBERO"}
    return _proxy_render_multiview(
        meta,
        width=max(64, min(1024, int(width))),
        height=max(64, min(1024, int(height))),
        hide_robot=bool(hide_robot),
        lookat_xyz_m=lookat_xyz_m,
        distance_m=float(distance_m),
    )


@_blocking_tool
def check_task(handle: str, *, session_id: str = "") -> dict:
    """Return the native task checker result without privileged simulator state.

    This is intentionally narrower than ``observe_env``: the embodied
    Operator may ask whether the task is complete, while the host retains
    access to richer simulator diagnostics through the worker/runtime layers.
    """

    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    return _proxy_check_task(meta)


@_world_action_tool
def close_env(handle: str, *, session_id: str = "") -> dict:
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    with _close_env_state_lock:
        meta = _session_envs.get(sid, {}).get(handle)
        if meta is None:
            # Closing is deliberately idempotent once local cleanup has been
            # fully confirmed and the handle has actually been removed.
            return {
                "ok": True,
                "already_closed": True,
                "cleanup_state": "closed",
                "cleanup_errors": [],
                "cleanup_error_details": [],
            }
        if meta.get("_cleanup_in_progress"):
            detail = {
                "stage": "remote_close",
                "kind": "cleanup_in_progress",
                "message": "cleanup is already in progress for this handle",
                "retryable": True,
            }
            return {
                "ok": False,
                "already_closed": False,
                "cleanup_state": "draining",
                "retryable": True,
                "cleanup_errors": ["remote_close: cleanup_in_progress"],
                "cleanup_error_details": [detail],
            }
        meta["_cleanup_state"] = "draining"
        meta["_cleanup_in_progress"] = True
        meta["_cleanup_attempts"] = int(meta.get("_cleanup_attempts", 0)) + 1
        attempt = meta["_cleanup_attempts"]
        remote_confirmed = bool(meta.get("_remote_close_confirmed"))

    remote_result: dict = {"ok": True, "previously_confirmed": True}
    if not remote_confirmed:
        try:
            remote_result = _get_mgr().proxy_handle_op(
                meta, f"/env/{meta['remote_handle']}", method="DELETE"
            )
        except Exception as exc:
            remote_result = {
                "ok": False,
                "error": str(exc),
                "error_detail": {
                    "kind": "transport_error",
                    "message": str(exc),
                    "error_type": type(exc).__name__,
                },
            }

        if not (
            isinstance(remote_result, dict)
            and remote_result.get("ok") is True
            and "error" not in remote_result
        ):
            source_detail = (
                remote_result.get("error_detail", {})
                if isinstance(remote_result, dict)
                else {}
            )
            detail = {
                "stage": "remote_close",
                "kind": source_detail.get("kind", "remote_cleanup_unconfirmed"),
                "message": source_detail.get(
                    "message",
                    remote_result.get("error", "worker did not confirm cleanup")
                    if isinstance(remote_result, dict)
                    else f"unexpected cleanup response: {type(remote_result).__name__}",
                ),
                "retryable": True,
                "attempt": attempt,
            }
            for key in ("error_type", "method", "path", "timeout_s", "status_code"):
                if key in source_detail:
                    detail[key] = source_detail[key]
            with _close_env_state_lock:
                meta["_cleanup_in_progress"] = False
                meta["_cleanup_error"] = detail
            return {
                "ok": False,
                "already_closed": False,
                "cleanup_state": "draining",
                "retryable": True,
                "remote": remote_result,
                "cleanup_errors": [
                    f"remote_close: {detail['kind']}: {detail['message']}"
                ],
                "cleanup_error_details": [detail],
            }

        with _close_env_state_lock:
            meta["_remote_close_confirmed"] = True

    try:
        _get_mgr().release_worker(meta.get("worker_url", ""))
    except Exception as exc:
        detail = {
            "stage": "release_worker",
            "kind": "local_release_error",
            "message": str(exc),
            "error_type": type(exc).__name__,
            "retryable": True,
            "attempt": attempt,
        }
        with _close_env_state_lock:
            meta["_cleanup_in_progress"] = False
            meta["_cleanup_error"] = detail
        return {
            "ok": False,
            "already_closed": False,
            "cleanup_state": "draining",
            "retryable": True,
            "remote": remote_result,
            "cleanup_errors": [
                f"release_worker: {type(exc).__name__}: {exc}"
            ],
            "cleanup_error_details": [detail],
        }

    with _close_env_state_lock:
        envs = _session_envs.get(sid, {})
        if envs.get(handle) is meta:
            envs.pop(handle, None)
        meta["_cleanup_state"] = "closed"
        meta["_cleanup_in_progress"] = False
        meta.pop("_cleanup_error", None)
    cache = _session_last_obs.get(sid, {})
    cache.pop(handle, None)
    cache.pop(meta.get("remote_handle", ""), None)
    remove_checker(handle)
    return {
        "ok": True,
        "already_closed": False,
        "cleanup_state": "closed",
        "remote": remote_result,
        "cleanup_errors": [],
        "cleanup_error_details": [],
    }


@_blocking_tool
def list_active_envs(*, session_id: str = "") -> dict:
    """Return all active environments in a session.

    To show the user a live camera feed, construct the URL as
    ``{mcp_server_url}/session/{session_id}`` where ``mcp_server_url``
    is the server address you are already connected to.

    Returns:
        dict with keys: **session_id**, **count**, **envs** (list of
        ``{index, handle, env_id, backend}``).
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    envs = _session_envs.get(sid, {})
    entries: list[dict] = []
    draining_count = 0
    for i, (h, meta) in enumerate(envs.items(), 1):
        cleanup_state = meta.get("_cleanup_state", "active")
        if cleanup_state == "draining":
            draining_count += 1
        entries.append({
            "index": i,
            "handle": h,
            "env_id": meta.get("env_id", "unknown"),
            "backend": meta.get("backend", "unknown"),
            "cleanup_state": cleanup_state,
        })
    return {
        "session_id": sid,
        "count": len(entries),
        "active_count": len(entries) - draining_count,
        "draining_count": draining_count,
        "envs": entries,
    }


# ══════════════════════════════════════════════════════════════════════
# Starlette app + ASGI combined
# ══════════════════════════════════════════════════════════════════════

def _build_dashboard_app() -> Starlette:
    """Build the Starlette app for the live camera view.

    This server is agent-facing: environment control happens through the MCP
    tools (``/mcp`` and ``/sse``), not over HTTP.  The only HTTP surface kept
    here is the read-only live camera view that agents point users at
    (``create_env`` returns its URL) so a human can watch the robot in real
    time.  The old clickable control GUI (``/``) and the REST control API
    (``/api/...``) were removed.
    """
    return Starlette(routes=[
        Route("/session/{sid}", session_dashboard, methods=["GET"]),
        Route("/session/{sid}/envs", session_envs, methods=["GET"]),
        Route("/session/{sid}/stream", session_stream, methods=["GET"]),
        Route("/session/{sid}/stream/{handle}", session_env_stream, methods=["GET"]),
    ])


# ══════════════════════════════════════════════════════════════════════
# CLI entry
# ══════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    import uvicorn
    from mcp.server.sse import SseServerTransport

    p = argparse.ArgumentParser(description="OpenETA MCP + Web Dashboard")
    p.add_argument("--transport", default="sse", choices=["sse", "stdio"])
    p.add_argument("--port", type=int, default=0)
    args = p.parse_args()
    port = args.port or int(os.environ.get("MCP_PORT", os.environ.get("PORT", "8765")))
    _init()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return

    # Build the dashboard/api Starlette app
    dashboard_app = _build_dashboard_app()

    # SSE transport — endpoint is the full path from server root
    sse_transport = SseServerTransport("/sse/messages/")

    # Streamable HTTP transport (the 2025 MCP transport) — a single ``/mcp``
    # endpoint that handles both directions per-request, mounted *alongside*
    # the legacy ``/sse`` transport so existing clients keep working.  Unlike
    # ``/sse`` (which mints a per-connection session_id into a contextvar),
    # streamable HTTP runs each MCP session in its own spawned task, so tools
    # cannot rely on the ``_current_session`` contextvar here — ``/mcp``
    # clients must pass the ``session_id`` returned by ``create_env`` back on
    # subsequent calls (the documented cross-connection reuse pattern).
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

    _http_manager = StreamableHTTPSessionManager(
        app=mcp._mcp_server,
        event_store=None,        # no event replay yet; session-id reconnection still works
        json_response=False,     # allow per-request SSE upgrade for progress/notifications
        stateless=False,         # keep per-session state (Mcp-Session-Id header)
    )

    import asyncio as _asyncio

    # Latch: set when SSE transport is connected and ready for post messages
    _mcp_ready: _asyncio.Event = _asyncio.Event()

    # Top-level ASGI app: intercept MCP routes, delegate rest to dashboard
    async def combined(scope, receive, send):
        # ── ASGI lifespan: drive the streamable-HTTP manager's task group ──
        if scope["type"] == "lifespan":
            async with _http_manager.run():
                message = await receive()
                assert message["type"] == "lifespan.startup"
                await send({"type": "lifespan.startup.complete"})
                while True:
                    message = await receive()
                    if message["type"] == "lifespan.shutdown":
                        await send({"type": "lifespan.shutdown.complete"})
                        return
            return
        if scope["type"] == "http":
            await _maybe_start_sweeper()
            path = scope["path"]
            # ── Streamable HTTP transport (single endpoint) ──────────
            if path == "/mcp" or path.startswith("/mcp/"):
                await _http_manager.handle_request(scope, receive, send)
                return
            if path == "/sse" and scope["method"] == "GET":
                _mcp_ready.clear()
                sid = str(uuid.uuid4())
                _sse_sessions.add(sid)
                _touch_session(sid)
                token = _current_session.set(sid)
                try:
                    async with sse_transport.connect_sse(scope, receive, send) as streams:
                        _mcp_ready.set()  # safe to accept post messages now
                        await mcp._mcp_server.run(
                            streams[0],
                            streams[1],
                            mcp._mcp_server.create_initialization_options(),
                        )
                finally:
                    _current_session.reset(token)
                    _detach_sse_session(sid)
                return
            if path.startswith("/sse/messages/") and scope["method"] == "POST":
                # Retry up to 3s waiting for SSE session to be set up
                try:
                    await _asyncio.wait_for(_mcp_ready.wait(), timeout=3.0)
                except _asyncio.TimeoutError:
                    pass
                await sse_transport.handle_post_message(scope, receive, send)
                return
        await dashboard_app(scope, receive, send)

    # Start the stale-session sweeper lazily on first HTTP request
    _sweeper_flag = [False]

    async def _maybe_start_sweeper() -> None:
        if not _sweeper_flag[0]:
            _sweeper_flag[0] = True
            _asyncio.create_task(_stale_session_sweeper())

    print(f"\n  OpenETA Dashboard:      http://0.0.0.0:{port}/")
    print(f"  MCP (Streamable HTTP):  http://0.0.0.0:{port}/mcp")
    print(f"  MCP (legacy SSE):       http://0.0.0.0:{port}/sse\n")
    uvicorn.run(combined, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    main()
