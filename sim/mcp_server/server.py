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
import uuid

import anyio.to_thread

from starlette.applications import Starlette
from starlette.routing import Route

from sim.mcp_server.session import (
    _current_session,
    _get_mgr,
    _init,
    _obs_key,
    _session_envs,
    _session_last_obs,
    _session_last_obs_lock,
    _sse_sessions,
    _touch_session,
    _detach_sse_session,
    _stale_session_sweeper,
    _session_last_activity,
)
from sim.mcp_server.worker_mgr import (
    _forget_obs_dirty,
    _proxy_observe,
    _proxy_render,
    _proxy_reset,
    _proxy_step,
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
    meta = {
        "worker_url": worker.base_url,
        "remote_handle": remote_handle,
        "env_id": env_id,
        "backend": result.get("backend", "unknown"),
        "action_dim": result.get("action_dim"),
        "robot": result.get("robot") or robot,
        "control_spec": result.get("control_spec", {}),
        "_sid": sid,
    }
    _session_envs.setdefault(sid, {})[h] = meta
    # NOTE: do NOT settle here — the worker creates the env but does NOT reset
    # it, so the MuJoCo/robosuite sim is uninitialised and stepping it produces
    # garbage frames (the "corrupted render on create" bug).  Settling happens
    # in reset_env / move_to's implicit reset, i.e. only after a real reset.
    return {
        "session_id": sid, "handle": h, "env_id": env_id,
        "action_dim": result.get("action_dim"), "backend": result.get("backend"),
        "robot": result.get("robot") or robot,
        "control_spec": result.get("control_spec", {}),
        "action_hint": result.get("action_hint", ""),
    }


@_blocking_tool
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
          ``gripper_state`` (dict with ``openness`` float in [0,1]:
          0=fully closed, 1=fully open, intermediate=partially open;
          plus a legacy ``open`` bool = openness > 0.5)
        * **objects** (list[dict]) — each has ``name``, ``position``
          (world xyz), ``orientation`` (quat xyzw, optional)
        * **metadata** (dict) — extra info
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    # A reset re-opens the gripper and starts a fresh episode, so drop any
    # latched gripper command: subsequent motion steps go back to not forcing
    # the gripper dim until the user explicitly calls gripper_open/close again.
    meta.pop("_gripper_cmd", None)
    reset_obs = _proxy_reset(meta, seed=seed)
    # Let physics settle before returning — objects can spawn hovering /
    # jittering right after reset; a few hold steps bring them to rest.
    settled = _settle_env(meta, meta.get("backend", ""))
    settled_obs = settled.get("observation") if isinstance(settled, dict) else None
    return settled_obs if isinstance(settled_obs, dict) and settled_obs else reset_obs


@_blocking_tool
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


@_blocking_tool
def move_to(handle: str, x: float, y: float, z: float, *,
            roll: float | None = None, pitch: float | None = None, yaw: float | None = None,
            num_steps: int = 100, tolerance: float = 0.002, ori_tolerance: float = 0.05,
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
        tolerance: Stop when |pos_err| < tolerance on all axes (default 0.002 m
            = 2 mm).  Measured residual at this setting is sub-mm to ~1 mm and
            it converges in ~12-15 steps.  Loosen to ~0.01 for coarse reaches.
        ori_tolerance: Stop when angular error < ori_tolerance (default 0.05 rad ≈ 3°).
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

    with _session_last_obs_lock:
        cached = _session_last_obs.get(sid, {}).get(_obs_key(meta), {})
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
        # settle physics after the implicit reset, then read the settled pose
        settled = _settle_env(meta, backend)
        pose_result = settled if isinstance(settled, dict) and settled.get("observation") else reset_result
        current_xyz = _extract_ee_xyz_from_result(pose_result)
        if use_ori and len(current_quat) < 4:
            current_quat = _extract_ee_quat_from_result(pose_result)

    if len(current_xyz) < 3:
        return {"error": "Cannot determine current EE position — call reset_env first"}
    if use_ori and len(current_quat) < 4:
        return {"error": "Cannot determine current EE orientation (no quat_xyzw in observation)"}

    start_xyz = current_xyz[:3]
    start_quat = current_quat[:4] if use_ori else []
    final_result: dict = {}
    final_reward = 0.0
    final_terminated = False
    total_steps = 0

    # ── collision state (initialized before loop) ──────────────────
    collision_detected = False
    collision_info: dict = {"available": False}

    # ── closed-loop interpolation ──────────────────────────────────
    # Both position and orientation are always active — no freezing.
    # If orientation changes couple into EE position (serial chain),
    # the next recheck catches it.  Stop only when both converge.
    for batch_start in range(0, num_steps, recheck_every):
        # Position error (always live)
        err_x = x - current_xyz[0]
        err_y = y - current_xyz[1]
        err_z = z - current_xyz[2]

        # Orientation error (always computed, never frozen)
        delta_rot: list[float] = [0.0, 0.0, 0.0]
        ori_ok = not use_ori
        if use_ori:
            ori_dist = _quat_angular_distance(current_quat, target_quat)
            ori_ok = ori_dist < ori_tolerance
            # Always compute delta even if converged — will be ~[0,0,0]
            delta_q = _quat_multiply(target_quat, _quat_conjugate(current_quat))
            # Shortest-arc normalization: q and -q are the same rotation, but
            # a negative scalar part makes _quat_to_axis_angle read the angle as
            # ~(2π - θ) with a flipped axis — e.g. a +10° yaw becomes a ~350°
            # rotation the wrong way.  Flip to the hemisphere with w >= 0 so the
            # axis-angle is always the minimal rotation to the target.
            if delta_q[3] < 0:
                delta_q = [-v for v in delta_q]
            delta_aa = _quat_to_axis_angle(delta_q)
            delta_rot = [max(-1.0, min(1.0, d / recheck_every / ori_scale)) for d in delta_aa]

        # Check convergence (both position AND orientation)
        pos_ok = abs(err_x) < tolerance and abs(err_y) < tolerance and abs(err_z) < tolerance
        if pos_ok and ori_ok:
            break

        # Position delta (never frozen — even if pos_ok, we keep zero
        # delta so rotation-only batches don't perturb position)
        ax = max(-1.0, min(1.0, err_x / recheck_every / scale))
        ay = max(-1.0, min(1.0, err_y / recheck_every / scale))
        az = max(-1.0, min(1.0, err_z / recheck_every / scale))

        batch_steps = min(recheck_every, num_steps - batch_start)

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
                # Keep the explicitly latched gripper command on every
                # Cartesian motion step.  Calling make_cartesian_action()
                # directly leaves the gripper slot at its neutral value;
                # _make_action_for_step() overlays the persistent open/closed
                # command after constructing the arm motion action.
                act = _make_action_for_step(
                    meta,
                    (action_xyz[0], action_xyz[1], action_xyz[2]),
                    backend,
                    delta_rot=action_rot if use_ori else None,
                )
            except ControlCodecError as exc:
                return codec_error_result(exc)
            # Skip the ~130 ms per-step GPU render on most steps — move_to only
            # reads the EE pose from the result.  Render every _RENDER_EVERY
            # steps so the dashboard gets periodic feedback during the motion.
            # (A final render is forced after the loop regardless of how it
            # exits — convergence, collision, or termination.)
            do_render = ((total_steps + 1) % _RENDER_EVERY == 0)
            final_result = _proxy_step(meta, act, num_steps=1, render=do_render)
            total_steps += 1
            final_reward = final_result.get("reward", 0.0)
            if final_result.get("terminated") or final_result.get("truncated"):
                final_terminated = True
                break

        if final_terminated:
            break

        # Re-read pose from last step result (no extra HTTP call)
        new_xyz = _extract_ee_xyz_from_result(final_result)
        pose_result = final_result
        if len(new_xyz) >= 3:
            current_xyz = new_xyz
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
                with _session_last_obs_lock:
                    _session_last_obs.setdefault(sid, {})[_obs_key(meta)] = render_result
        except Exception:
            pass  # best-effort; final pose is still read from final_result below

    # ── final pose ─────────────────────────────────────────────────
    final_xyz = _extract_ee_xyz_from_result(final_result) if total_steps > 0 else start_xyz
    final_quat = _extract_ee_quat_from_result(final_result) if (use_ori and total_steps > 0) else []

    result: dict = {
        "target": {"x": x, "y": y, "z": z},
        "start": {"xyz": start_xyz},
        "end": {"xyz": final_xyz[:3] if len(final_xyz) >= 3 else final_xyz},
        "steps_executed": total_steps,
        "terminated": final_terminated,
        "reward": final_reward,
    }
    if use_ori:
        result["target"]["roll"] = roll
        result["target"]["pitch"] = pitch
        result["target"]["yaw"] = yaw
        result["start"]["quat_xyzw"] = start_quat
        result["end"]["quat_xyzw"] = final_quat[:4] if len(final_quat) >= 4 else final_quat

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


# Steps issued per gripper open/close call.  The gripper closes gradually
# (position control), and the open/closed flag only flips once the fingers
# pass the halfway detection threshold.  From a fully-open start it takes ~7
# steps to cross that threshold, so 5 steps left the gripper still reporting
# "open" — a single close call must fully actuate.  10 gives margin.
_GRIPPER_STEPS = 10


@_blocking_tool
def gripper_open(handle: str, *, session_id: str = "") -> dict:
    """Open the gripper (10 steps).

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
    meta["_gripper_cmd"] = -1.0  # latch OPEN — held on every subsequent step
    try:
        act = make_gripper_action(meta, open_gripper=True, backend=backend)
    except ControlCodecError as exc:
        return codec_error_result(exc)
    return _proxy_step(meta, act, num_steps=_GRIPPER_STEPS)


@_blocking_tool
def gripper_close(handle: str, *, session_id: str = "") -> dict:
    """Close the gripper (10 steps).

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
    meta["_gripper_cmd"] = 1.0  # latch CLOSED — held (clamping) on every subsequent step
    try:
        act = make_gripper_action(meta, open_gripper=False, backend=backend)
    except ControlCodecError as exc:
        return codec_error_result(exc)
    return _proxy_step(meta, act, num_steps=_GRIPPER_STEPS)


def _gripper_cmd(meta: dict) -> float:
    """Return the persistent gripper command for this env.

    The gripper is a latched two-state actuator: once ``gripper_close`` /
    ``gripper_open`` is called, that command (``+1.0`` closed / ``-1.0`` open)
    is held on the gripper action dimension of *every* subsequent step —
    including the ``move_to`` control loop — so a grasped object stays clamped
    while the arm moves instead of the fingers relaxing to zero force.

    Defaults to ``-1.0`` (open) before any gripper call.
    """
    return meta.get("_gripper_cmd", -1.0)


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


@_blocking_tool
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
    """Build a Cartesian motion action, holding the gripper only if latched.

    Delegates slot layout to the explicit action codec (``make_cartesian_action``).
    If the user has explicitly latched a gripper command (via gripper_open/close),
    that ±1.0 command is overlaid onto the gripper slot(s) so the fingers keep
    holding their open/closed state throughout the motion instead of relaxing to
    zero force.  Until the first explicit gripper call (and after every reset)
    no gripper command is present, so the motion action leaves the gripper
    slot(s) untouched — matching each backend's plain Cartesian action contract.

    The gripper slot is backend-specific (RoboCasa slot 6, others the last slot),
    so we reuse the codec's own gripper encoder to place the value rather than
    hard-coding an index.  The latched command is already ±1.0 — exactly what
    ``make_gripper_action`` emits — so ``open_gripper=<cmd is open>`` reproduces
    the held gripper vector.
    """
    act = make_cartesian_action(meta, delta_xyz, backend, delta_rot=delta_rot)
    if "_gripper_cmd" not in meta:
        return act  # no explicit gripper command yet — don't force the dim
    try:
        held = make_gripper_action(meta, open_gripper=_gripper_cmd(meta) < 0.0, backend=backend)
    except ControlCodecError:
        return act  # backend without gripper control — nothing to hold
    for i, v in enumerate(held):
        if v != 0.0 and i < len(act):
            act[i] = v
    return act


def _make_gripper_action(meta: dict, *, open: bool, backend: str) -> list[float]:
    """Compatibility wrapper around the explicit simulator action codec."""
    return make_gripper_action(meta, open_gripper=open, backend=backend)


# Steps to run after every reset so the physics settles before the first
# observation.  Right after reset objects can be spawned slightly above their
# resting pose (or with residual velocity), so the initial frame shows them
# hovering / jittering; a few zero-motion "hold" steps let them fall and come
# to rest.  The hold action keeps the arm still and holds the latched gripper
# state, so settling never perturbs the robot or drops a held object.
_SETTLE_STEPS = 5


def _settle_env(meta: dict, backend: str) -> dict:
    """Step the env a few times with a hold action to let physics settle.

    Returns the observation from the last settle step (same structure as a
    reset/step observation), or ``{}`` if no settling was performed.  Rendered
    only on the final step so the caller gets a current frame without paying
    the per-step render cost on every settle step.
    """
    if _SETTLE_STEPS <= 0:
        return {}
    # Hold action: zero position/rotation delta, gripper held at its latched
    # state (open by default).  _make_action_for_step already fills the gripper
    # dim from the env's latched command.
    hold = _make_action_for_step(meta, (0.0, 0.0, 0.0), backend)
    last: dict = {}
    for i in range(_SETTLE_STEPS):
        render = (i == _SETTLE_STEPS - 1)  # only render the final settled frame
        res = _proxy_step(meta, hold, num_steps=1, render=render)
        last = res
        if res.get("terminated") or res.get("truncated"):
            break
    return last


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
          ``gripper_state`` (``openness`` float in [0,1] + legacy ``open`` bool)
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
def close_env(handle: str, *, session_id: str = "") -> dict:
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).pop(handle, None)
    if meta:
        remote_result: dict = {}
        cleanup_errors: list[str] = []
        # Evict the cache by the SAME composite key used to write it — the old
        # code popped by bare ``handle`` and so never actually cleared the
        # entry, leaking stale frames for a since-closed env.
        with _session_last_obs_lock:
            _session_last_obs.get(sid, {}).pop(_obs_key(meta), None)
        _forget_obs_dirty(_obs_key(meta))
        try:
            remote_result = _get_mgr().proxy_handle_op(
                meta, f"/env/{meta['remote_handle']}", method="DELETE"
            )
        except Exception as exc:
            cleanup_errors.append(f"remote_close: {type(exc).__name__}: {exc}")
        finally:
            # Always release the worker reference, including transport errors.
            try:
                _get_mgr().release_worker(meta.get("worker_url", ""))
            except Exception as exc:
                cleanup_errors.append(f"release_worker: {type(exc).__name__}: {exc}")
            remove_checker(handle)
        return {
            "ok": not cleanup_errors,
            "already_closed": False,
            "remote": remote_result,
            "cleanup_errors": cleanup_errors,
        }
    # Closing is deliberately idempotent so finally-block retries are safe.
    return {"ok": True, "already_closed": True, "cleanup_errors": []}


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
    for i, (h, meta) in enumerate(envs.items(), 1):
        entries.append({
            "index": i,
            "handle": h,
            "env_id": meta.get("env_id", "unknown"),
            "backend": meta.get("backend", "unknown"),
        })
    return {
        "session_id": sid,
        "count": len(entries),
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
