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
import hashlib
import math
import os
import sys
import threading
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
    _proxy_controller_goal,
    _proxy_reachability,
    _proxy_render,
    _proxy_reset,
    _proxy_step,
)
from sim.mcp_server.collision import (
    RECEPTACLE_CATEGORIES,
    check_attached_object_collision,
    get_checker,
    remove_checker,
    resolve_contact_authorization as resolve_contact_object_authorization,
)
from sim.mcp_server.action_codecs import (
    ControlCodecError,
    cartesian_command_frame,
    cartesian_scales,
    codec_error_result,
    make_cartesian_action,
    make_gripper_action,
    require_controller_capability,
    trunk_hold_values,
    trunk_layout,
)
from sim.mcp_server.rest_api import (
    session_dashboard,
    session_envs,
    session_stream,
    session_env_stream,
)
from sim.reachability import ROBUST_EXECUTION_JOINT_MARGIN_RAD
from adapter.motion_profiles import motion_control_profile

# ── FastMCP server ────────────────────────────────────────────────────

from mcp.server.fastmcp import FastMCP
mcp = FastMCP("OpenETA", log_level="WARNING")

_env_control_locks: dict[tuple[str, str], threading.RLock] = {}
_env_control_locks_guard = threading.Lock()


def _env_control_lock(session_id: str, handle: str) -> threading.RLock:
    key = (session_id, handle)
    with _env_control_locks_guard:
        lock = _env_control_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _env_control_locks[key] = lock
        return lock


def _serialized_env_control(fn):
    """Serialize complete control calls per env while preserving cross-env parallelism."""

    @functools.wraps(fn)
    def _wrapper(*args, **kwargs):
        handle = str(kwargs.get("handle") or (args[0] if args else ""))
        sid = str(kwargs.get("session_id") or _current_session.get() or "")
        lock = _env_control_lock(sid, handle)
        with lock:
            result = fn(*args, **kwargs)
        if fn.__name__ == "close_env":
            with _env_control_locks_guard:
                if _env_control_locks.get((sid, handle)) is lock:
                    _env_control_locks.pop((sid, handle), None)
        return result

    return _wrapper


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
    # Safety always receives privileged geometry internally.  The worker proxy
    # redacts it from public observations unless include_objects was requested.
    body["include_objects"] = True
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
        "_expose_objects": bool(include_objects),
        "_collision_objects": [],
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
@_serialized_env_control
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
    # A reset starts a fresh episode with the gripper explicitly OPEN.  LIBERO
    # uses -1 for open and +1 for close; leaving the action dimension at the
    # neutral value 0 does *not* hold the fingers open and allowed them to drift
    # closed during ordinary move_to calls before the Agent requested any
    # gripper action.  Hold OPEN through settling and later arm motions until
    # gripper_close establishes the opposite latch.
    meta["_gripper_cmd"] = -1.0
    meta.pop("_attachment_proxy", None)
    reset_obs = _proxy_reset(meta, seed=seed)
    # Let physics settle before returning — objects can spawn hovering /
    # jittering right after reset; a few hold steps bring them to rest.
    settled = _settle_env(meta, meta.get("backend", ""))
    settled_obs = settled.get("observation") if isinstance(settled, dict) else None
    return settled_obs if isinstance(settled_obs, dict) and settled_obs else reset_obs


@_blocking_tool
@_serialized_env_control
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
    result = _proxy_step(meta, action, num_steps=num_steps)
    attachment_receipt = _refresh_attachment_proxy(meta, result)
    if attachment_receipt is not None:
        result["attachment_proxy_receipt"] = attachment_receipt
    return result


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


def _extract_joint_names_from_result(result: dict) -> list[str]:
    """Extract ``joint_names`` (positional labels for ``joint_positions``).

    Needed by the collision checker for multi-arm robots, where the joints
    cuRobo wants are not a leading slice of the observation vector.  Absent for
    single-arm backends, which is fine — the checker only requires names when
    slicing would be wrong.
    """
    obs = result.get("observation", result) if isinstance(result, dict) else {}
    if not isinstance(obs, dict):
        return []
    robot = obs.get("robot", {})
    if not isinstance(robot, dict):
        return []
    names = robot.get("joint_names", [])
    return [str(n) for n in names] if isinstance(names, list) else []


def _extract_objects_from_result(result: dict) -> list[dict]:
    """Extract ``objects`` list from a step result or observe result."""
    obs = result.get("observation", result) if isinstance(result, dict) else {}
    if not isinstance(obs, dict):
        return []
    objects = obs.get("objects", [])
    return objects if isinstance(objects, list) else []


def _extract_gripper_state_from_result(result: dict) -> dict:
    obs = result.get("observation", result) if isinstance(result, dict) else {}
    robot = obs.get("robot", {}) if isinstance(obs, dict) else {}
    state = robot.get("gripper_state", {}) if isinstance(robot, dict) else {}
    return state if isinstance(state, dict) else {}


def _arm_attachment_proxy(
    meta: dict,
    result: dict,
    *,
    authorized_object: dict | None = None,
) -> dict:
    """Create a tentative held-object proxy after a non-empty close.

    When the harness supplies compiled target provenance, bind the proxy to
    that exact resolved object instead of guessing from nearest-neighbour
    geometry.  The proxy remains tentative until independent co-motion
    evidence exists; this receipt is collision-planning metadata, not proof of
    a successful grasp.
    """

    state = _extract_gripper_state_from_result(result)
    openness = state.get("openness")
    if not isinstance(openness, (int, float)) or isinstance(openness, bool):
        meta.pop("_attachment_proxy", None)
        return {
            "schema_version": "openeta.attachment_proxy_receipt.v1",
            "status": "not_armed",
            "reason": "gripper_aperture_unavailable",
            "measured_open_fraction": openness,
            "attachment_proven": False,
            "interpretation": (
                "The close response did not expose a numeric gripper aperture, so no "
                "carried-object collision proxy was armed. Inspect fresh views."
            ),
        }
    eef = _extract_ee_xyz_from_result(result)
    if len(eef) < 3:
        return {
            "schema_version": "openeta.attachment_proxy_receipt.v1",
            "status": "not_armed",
            "reason": "eef_pose_unavailable",
            "measured_open_fraction": float(openness),
            "attachment_proven": False,
        }
    nearest: tuple[float, dict] | None = None
    candidates = (
        [authorized_object]
        if isinstance(authorized_object, dict)
        else list(meta.get("_collision_objects", []))
    )
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category") or "").strip().lower()
        if category in RECEPTACLE_CATEGORIES:
            continue
        position = obj.get("position")
        if not isinstance(position, list) or len(position) < 3:
            continue
        distance = math.dist(
            [float(value) for value in eef[:3]],
            [float(value) for value in position[:3]],
        )
        if nearest is None or distance < nearest[0]:
            nearest = (distance, obj)
    if nearest is None or nearest[0] > 0.12:
        meta.pop("_attachment_proxy", None)
        return {
            "schema_version": "openeta.attachment_proxy_receipt.v1",
            "status": "not_armed",
            "reason": "authorized_target_outside_contact_envelope",
            "target_object_name": (
                str(authorized_object.get("name") or "")
                if isinstance(authorized_object, dict)
                else None
            ),
            "eef_to_target_distance_m": nearest[0] if nearest is not None else None,
            "max_contact_envelope_m": 0.12,
            "measured_open_fraction": float(openness),
            "attachment_proven": False,
            "interpretation": (
                "The close command completed, but the host-bound target was not near "
                "the EEF contact envelope. Do not infer attachment from aperture alone."
            ),
        }
    obj = nearest[1]
    position = [float(value) for value in obj.get("position", [])[:3]]
    dims = obj.get("dims")
    if not isinstance(dims, list) or len(dims) < 3:
        dims = [0.06, 0.06, 0.10]
    meta["_attachment_proxy"] = {
        "status": "tentative",
        "object_name": str(obj.get("name") or ""),
        "category": str(obj.get("category") or ""),
        "relative_xyz": [position[i] - float(eef[i]) for i in range(3)],
        "dims": [max(0.01, float(value)) for value in dims[:3]],
        "anchor_eef_xyz": [float(value) for value in eef[:3]],
        # Aperture is evidence for the independent attachment reviewer, not a
        # proxy-arming gate: thin objects may legitimately close near zero.
        "measured_open_fraction": float(openness),
        "binding_source": (
            "host_compiled_target_provenance"
            if isinstance(authorized_object, dict)
            else "nearest_object_fallback"
        ),
    }
    return {
        "schema_version": "openeta.attachment_proxy_receipt.v1",
        "status": "tentative",
        "reason": "close_near_bound_target_pending_visual_confirmation",
        "target_object_name": str(obj.get("name") or ""),
        "binding_source": meta["_attachment_proxy"]["binding_source"],
        "eef_to_target_distance_m": nearest[0],
        "measured_open_fraction": float(openness),
        "attachment_proven": False,
        "interpretation": (
            "A conservative carried-object collision proxy was armed for a lift "
            "probe. Aperture is only a hint and this is not attachment proof; "
            "require post-lift co-motion and source-vacancy evidence."
        ),
    }


def _refresh_attachment_proxy(meta: dict, result: dict) -> dict | None:
    """Retain or retire a conservative proxy without claiming attachment.

    ``meta['_collision_objects']`` is a reset-time geometry catalogue, not a
    live object-state stream.  It must therefore never be used to infer
    co-motion.  While the close command remains latched and measured aperture
    has not collapsed to the empty-close range, keep the proxy tentative for
    collision safety.  Independent visual evidence owns attachment verdicts.
    """

    proxy = meta.get("_attachment_proxy")
    if not isinstance(proxy, dict):
        return None
    state = _extract_gripper_state_from_result(result)
    openness = state.get("openness")
    if isinstance(openness, (int, float)) and not isinstance(openness, bool):
        proxy["measured_open_fraction"] = float(openness)
    eef = _extract_ee_xyz_from_result(result)
    anchor = proxy.get("anchor_eef_xyz")
    displacement = (
        math.dist([float(value) for value in eef[:3]], [float(value) for value in anchor[:3]])
        if len(eef) >= 3 and isinstance(anchor, list) and len(anchor) >= 3
        else None
    )
    proxy["status"] = "tentative"
    return {
        "schema_version": "openeta.attachment_proxy_receipt.v1",
        "status": "tentative",
        "reason": "awaiting_independent_co_motion_evidence",
        "target_object_name": str(proxy.get("object_name") or ""),
        "binding_source": proxy.get("binding_source"),
        "measured_open_fraction": (
            float(openness)
            if isinstance(openness, (int, float)) and not isinstance(openness, bool)
            else None
        ),
        "eef_displacement_since_close_m": displacement,
        "attachment_proven": False,
        "interpretation": (
            "The host keeps a conservative tentative collision proxy after close. "
            "Aperture alone neither confirms nor retires it; confirm or reject "
            "attachment from fresh dual-view co-motion and source-vacancy evidence."
        ),
    }


def _collision_objects_without_attached(meta: dict) -> list[dict]:
    proxy = meta.get("_attachment_proxy")
    attached_name = str(proxy.get("object_name") or "") if isinstance(proxy, dict) else ""
    return [
        item
        for item in meta.get("_collision_objects", [])
        if isinstance(item, dict) and str(item.get("name") or "") != attached_name
    ]


# Radius around the commanded pose within which an object is read as the
# intended manipulation target rather than an obstacle.
_APPROACH_TARGET_RADIUS_M = 0.08


def _safety_obstacles(
    meta: dict,
    *,
    approach_target_xyz: tuple[float, float, float] | list[float] | None = None,
    target_radius_m: float = _APPROACH_TARGET_RADIUS_M,
) -> list[dict]:
    """Private safety geometry, minus the held object and the approach target.

    The safety adapter always uses privileged geometry; ``_expose_objects``
    governs only what the public observation reveals.  Keeping the two coupled
    meant the arm-vs-world check ran against an empty world by default.

    The single object nearest the *commanded* pose is dropped, because
    "something sits where I am reaching" is what an intended grasp target looks
    like — treating it as an obstacle would abort every reach before contact.
    Everything else in the scene stays an obstacle.  Inferring the target from
    the commanded pose keeps this server-side and opens no new information
    channel to the Agent.
    """
    obstacles = _collision_objects_without_attached(meta)
    if approach_target_xyz is None or len(approach_target_xyz) < 3:
        return obstacles
    try:
        target = [float(value) for value in approach_target_xyz[:3]]
    except (TypeError, ValueError):
        return obstacles

    nearest_name: str | None = None
    best = float(target_radius_m)
    for obj in obstacles:
        position = obj.get("position")
        if not isinstance(position, list) or len(position) < 3:
            continue
        try:
            distance = math.dist(target, [float(value) for value in position[:3]])
        except (TypeError, ValueError):
            continue
        if distance < best:
            nearest_name, best = str(obj.get("name") or ""), distance
    if nearest_name is None:
        return obstacles
    return [obj for obj in obstacles if str(obj.get("name") or "") != nearest_name]


# Ceiling on carry-sweep samples.  Sized so the density guarantee holds across a
# Franka's full reach (~0.855 m) for the 1 cm floor that _object_aabb clamps
# dims to: 0.855 / (0.01 / 2) + 1 = 172.  At 15.7 us per sample this is ~4 ms
# worst case, so the headroom is nearly free.
_MAX_SWEEP_SAMPLES = 256


def _check_attached_object_sweep(
    attachment: dict,
    obstacles: list[dict],
    start_xyz: list[float],
    end_xyz: list[float],
) -> tuple[bool, dict]:
    """Sample the carry segment instead of testing only its endpoint.

    Testing the batch endpoint alone lets the carried object tunnel straight
    through an obstacle whenever one batch advances further than the object's
    own thickness.  Sample density is set by the smallest held dimension so no
    sample can skip past a body thinner than the object itself.  This is pure
    AABB arithmetic — no GPU work — so the extra samples are cheap.

    The sample ceiling bounds pathological spans; it should not silently trade
    away the density guarantee.  At 24 it did: past a ~0.72 m span the step
    outgrew what a 1 cm held object covers and a wall between two samples was
    missed.  That span is *not* reachable through move_to -- one batch moves at
    most ``scale * batch_steps * sqrt(3)``, i.e. 4.7 cm on LIBERO and 26 cm on
    RoboCasa -- so this was defence in depth, not a live bug.  The ceiling did
    bind on RoboCasa-scale spans (26 samples wanted) without ever approaching
    the tunnelling threshold.  Raised anyway because the cost is trivial:
    15.7 us per sample against ten obstacles, so 256 samples is 4 ms.

    When the ceiling does bind, ``swept_density_capped`` and ``swept_step_m``
    report it rather than letting a weaker check pass as an equal one.
    """
    if len(start_xyz) < 3 or len(end_xyz) < 3:
        return check_attached_object_collision(attachment, obstacles, end_xyz)

    dims = attachment.get("dims")
    smallest = 0.06
    if isinstance(dims, list) and len(dims) >= 3:
        finite = [float(v) for v in dims[:3] if isinstance(v, (int, float))]
        if finite and min(finite) > 0:
            smallest = min(finite)

    try:
        span = math.dist(
            [float(v) for v in start_xyz[:3]], [float(v) for v in end_xyz[:3]]
        )
    except (TypeError, ValueError):
        return check_attached_object_collision(attachment, obstacles, end_xyz)

    step_limit = max(0.01, smallest / 2.0)
    wanted = int(span / step_limit) + 1
    samples = max(1, min(_MAX_SWEEP_SAMPLES, wanted))
    capped = wanted > _MAX_SWEEP_SAMPLES

    last_info: dict = {"available": True, "attached_object_world_collision": False}
    for index in range(1, samples + 1):
        ratio = index / samples
        sample = [
            float(start_xyz[axis]) + (float(end_xyz[axis]) - float(start_xyz[axis])) * ratio
            for axis in range(3)
        ]
        detected, info = check_attached_object_collision(attachment, obstacles, sample)
        last_info = info
        if detected:
            info["swept_samples"] = samples
            info["swept_hit_fraction"] = ratio
            return True, info

    last_info["swept_samples"] = samples
    # Surface the achieved density so a capped sweep is never mistaken for a
    # sweep that met the guarantee.
    last_info["swept_step_m"] = span / samples if samples else 0.0
    if capped:
        last_info["swept_density_capped"] = True
        last_info["swept_samples_wanted"] = wanted
    return False, last_info


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
@_serialized_env_control
def ik_preview_check(
    handle: str,
    x: float,
    y: float,
    z: float,
    *,
    roll: float | None = None,
    pitch: float | None = None,
    yaw: float | None = None,
    position_tolerance_m: float = 0.002,
    orientation_tolerance_rad: float = 0.05,
    max_attempts: int = 24,
    max_nfev_per_attempt: int = 300,
    timeout_s: float = 10.0,
    preserve_current_orientation: bool = True,
    check_endpoint_collision: bool = False,
    include_scene_objects: bool = False,
    session_id: str = "",
) -> dict:
    """Preview endpoint IK feasibility without moving the robot.

    The result is deliberately tri-state. ``unreachable`` is a structured
    rejection, while ``unknown`` means the numerical/backend budget could not
    certify either outcome and must not be presented as a safe approval.
    Path feasibility is not checked here. Require a later motion-controller
    receipt with explicit per-step trajectory/world collision coverage.
    """

    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"ok": False, "success": False, "error": f"Unknown: {handle}"}
    orientation_values = (roll, pitch, yaw)
    if any(value is not None for value in orientation_values) and not all(
        value is not None for value in orientation_values
    ):
        return {
            "ok": False,
            "success": False,
            "error": "roll, pitch, and yaw must be provided together",
            "reason_code": "invalid_target_pose",
        }

    body: dict = {
        "target_xyz": [float(x), float(y), float(z)],
        "position_tolerance_m": float(position_tolerance_m),
        "orientation_tolerance_rad": float(orientation_tolerance_rad),
        "max_attempts": int(max_attempts),
        "max_nfev_per_attempt": int(max_nfev_per_attempt),
        "timeout_s": float(timeout_s),
        "preserve_current_orientation": bool(preserve_current_orientation),
    }
    if all(value is not None for value in orientation_values):
        body["target_euler_xyz_deg"] = [float(roll), float(pitch), float(yaw)]

    result = _proxy_reachability(meta, body)
    if not isinstance(result, dict):
        result = {
            "status": "unknown",
            "kinematic_status": "unknown",
            "feasible": None,
            "reason_code": "invalid_worker_response",
            "message": "Reachability worker returned an invalid response.",
        }
    if result.get("error"):
        return {
            "ok": False,
            "success": False,
            "error": str(result.get("error")),
            "reason_code": str(result.get("reason_code") or "reachability_backend_error"),
        }

    result = dict(result)
    result.setdefault("collision", {"checked": False})
    result["path"] = {
        "checked": False,
        "reason": (
            "endpoint IK does not check a path; require explicit per-step "
            "trajectory/world collision coverage from the motion controller"
        ),
    }
    if check_endpoint_collision and result.get("kinematic_status") == "reachable":
        candidate = result.get("best_candidate")
        joints = candidate.get("joint_positions") if isinstance(candidate, dict) else None
        objects = list(meta.get("_collision_objects") or []) if include_scene_objects else []
        if isinstance(joints, list):
            try:
                checker = get_checker(handle, str(meta.get("backend") or ""))
                in_collision, collision_info = checker.check(joints, objects)
            except Exception as exc:  # noqa: BLE001 - uncertainty must stay structured.
                in_collision = False
                collision_info = {
                    "available": False,
                    "reason": f"endpoint collision checker failed: {type(exc).__name__}: {exc}",
                }
            result["collision"] = {
                "checked": bool(collision_info.get("available")),
                "scene_objects_included": bool(include_scene_objects),
                "detected": bool(in_collision),
                **{key: value for key, value in collision_info.items() if key != "available"},
            }
            if in_collision:
                result.update(
                    {
                        "status": "unreachable",
                        "feasible": False,
                        "reason_code": "endpoint_collision",
                        "message": (
                            "A kinematic solution exists, but the requested endpoint "
                            "configuration is in collision."
                        ),
                    }
                )
                result.setdefault("suggestions", []).append("select_collision_free_target")
            elif not collision_info.get("available"):
                result.update(
                    {
                        "status": "unknown",
                        "feasible": None,
                        "reason_code": "endpoint_collision_check_unavailable",
                        "message": (
                            "IK succeeded, but the requested endpoint collision check "
                            "was unavailable; overall feasibility is unknown."
                        ),
                    }
                )
        else:
            result.update(
                {
                    "status": "unknown",
                    "feasible": None,
                    "reason_code": "ik_candidate_missing",
                    "message": "IK result did not contain a joint candidate for collision checking.",
                }
            )

    candidate = result.get("best_candidate")
    margin = candidate.get("joint_margin_min_rad") if isinstance(candidate, dict) else None
    if (
        isinstance(margin, int | float)
        and float(margin) < ROBUST_EXECUTION_JOINT_MARGIN_RAD
    ):
        margin_value = float(margin)
        risk_level = "critical" if margin_value < 0.05 else "elevated"
        solver = result.get("solver")
        solver = solver if isinstance(solver, dict) else {}
        seed_search = solver.get("execution_seed_search")
        seed_search = seed_search if isinstance(seed_search, dict) else {}
        result["execution_seed_quality"] = {
            "risk_level": risk_level,
            "selected_joint_margin_rad": margin_value,
            "robust_margin_threshold_rad": ROBUST_EXECUTION_JOINT_MARGIN_RAD,
            "robust_alternative_found": (
                seed_search.get("robust_solution_selected") is True
            ),
            "feasible_solution_count": seed_search.get("feasible_solution_count"),
            "distant_robust_solution_count": seed_search.get(
                "distant_robust_solution_count"
            ),
            "interpretation": (
                "The endpoint is kinematically feasible, but the selected IK branch "
                "remains close enough to a hard joint limit that local full-pose "
                "control may fail. This is not a positive execution recommendation. "
                "Prefer a materially different waypoint, wrist orientation, or grasp "
                "candidate when one is available; if executing anyway, inspect the "
                "motion receipt and do not replay a failed target. A mathematically "
                "robust but distant IK branch is reported as evidence, not silently "
                "used as a long local-controller posture route."
            ),
        }
        suggestions = result.setdefault("suggestions", [])
        for suggestion in (
            "select_higher_joint_margin_target_or_orientation",
            "compare_alternative_grasp_candidate_before_motion",
        ):
            if suggestion not in suggestions:
                suggestions.append(suggestion)
        result["message"] = (
            str(result.get("message") or "IK solution found.").rstrip()
            + f" Execution seed margin is only {margin_value:.6f} rad; "
            "the endpoint is feasible but execution-fragile."
        )

    if isinstance(margin, int | float) and float(margin) < 0.05:
        proximity = {
            "near_limit": True,
            "margin_rad": float(margin),
            "warning_threshold_rad": 0.05,
            "nearest_joint_limit": candidate.get("nearest_joint_limit"),
            "interpretation": (
                "The endpoint has a joint-limit-respecting IK solution, but its "
                "minimum hard-limit margin is small. Bind this exact preview receipt "
                "to execution; if local motion cannot converge, change the waypoint "
                "or wrist orientation instead of replaying the same target."
            ),
        }
        result["joint_limit_proximity"] = proximity
        suggestions = result.setdefault("suggestions", [])
        if "select_higher_joint_margin_target_or_orientation" not in suggestions:
            suggestions.append("select_higher_joint_margin_target_or_orientation")

    status = str(result.get("status") or "unknown")
    result["ok"] = status != "unreachable"
    result["success"] = status != "unreachable"
    result["content"] = str(result.get("message") or f"IK preview status: {status}")
    return result


@_blocking_tool
@_serialized_env_control
def move_to(handle: str, x: float, y: float, z: float, *,
            roll: float | None = None, pitch: float | None = None, yaw: float | None = None,
            num_steps: int = 150, tolerance: float = 0.002, ori_tolerance: float = 0.05,
            session_id: str = "",
            enable_collision_check: bool = True,
            contact_authorization: dict | None = None,
            ik_execution_seed: dict | None = None) -> dict:
    """Move the end-effector to an absolute pose using closed-loop interpolation.

    Re-observes the EE pose from step results for closed-loop correction:
    every step for position-only reaches and every three steps for full-pose
    reaches. Supports both position-only and position + orientation control.

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
        num_steps: Maximum total steps (default 150). The controller stops early
            when the requested pose tolerance is reached; the larger ceiling
            accommodates safe, continuously converging full-pose rotations.
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
        controller_capability = require_controller_capability(
            meta,
            backend,
            orientation_requested=use_ori,
        )
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
    # Every proxy step already returns EE state at no additional render cost.
    # Position-only reaches benefit from recomputing every step: reusing one
    # nominal OSC delta for three physics steps produced a centimetre-scale
    # near-target limit cycle.  Full-pose reaches retain the smaller historical
    # three-step interpolation increments; one-step full-pose commands were
    # measured to amplify translation/rotation coupling.  The explicit receipt
    # below reports when the coupled controller still cannot attain the pose.
    recheck_every = 3 if use_ori else 1

    # ── target orientation in quaternion ───────────────────────────
    target_quat: list[float] = []
    if use_ori:
        target_quat = _euler_to_quat(_math.radians(roll), _math.radians(pitch), _math.radians(yaw))

    if controller_capability.get("goal_executor") == "openeta.worker_mink_goal.v1":
        motion_profile = motion_control_profile()
        attachment = meta.get("_attachment_proxy")
        if enable_collision_check and isinstance(attachment, dict):
            with _session_last_obs_lock:
                cached = _session_last_obs.get(sid, {}).get(_obs_key(meta), {})
            baseline_eef = _extract_ee_xyz_from_result(cached)
            attached_collision, attached_info = check_attached_object_collision(
                attachment,
                list(meta.get("_collision_objects", [])),
                [float(x), float(y), float(z)],
                baseline_eef_xyz=baseline_eef,
            )
            if attached_collision:
                return {
                    "ok": False,
                    "code": "attached_object_endpoint_collision",
                    "error": str(
                        attached_info.get("message")
                        or "The carried-object endpoint would collide with scene geometry."
                    ),
                    "collision": {
                        "detected": True,
                        "endpoint_checked": True,
                        "trajectory_checked": False,
                        "world_checked": True,
                        **attached_info,
                    },
                    "steps_executed": 0,
                    "reached_target": False,
                    "stop_reason": "collision_detected",
                }
        resolved_contact: dict | None = None
        if contact_authorization is not None:
            target_object, resolution = resolve_contact_object_authorization(
                contact_authorization,
                list(meta.get("_collision_objects", [])),
            )
            if target_object is None:
                return {
                    "ok": False,
                    "code": str(
                        resolution.get("code")
                        or "contact_authorization_resolution_failed"
                    ),
                    "error": str(
                        resolution.get("message")
                        or "Host contact evidence could not be associated with scene geometry."
                    ),
                    "contact_authorization": resolution,
                    "steps_executed": 0,
                    "reached_target": False,
                    "stop_reason": "contact_authorization_unresolved",
                }
            resolved_contact = {
                **resolution,
                "target_object_name": str(target_object.get("name") or ""),
            }
        body = {
            "target_xyz": [float(x), float(y), float(z)],
            "preserve_current_orientation": not use_ori,
            "max_steps": int(num_steps),
            "position_tolerance_m": float(tolerance),
            "orientation_tolerance_rad": float(ori_tolerance),
            "gripper_command": float(_gripper_cmd(meta)) if "_gripper_cmd" in meta else 0.0,
            "enable_collision_check": bool(enable_collision_check),
            # Host-private experiment configuration.  It is selected by the
            # server process and is never copied from an Agent tool argument.
            "motion_execution_condition": motion_profile.condition,
        }
        if resolved_contact is not None:
            body["contact_authorization"] = resolved_contact
        if isinstance(ik_execution_seed, dict):
            body["ik_execution_seed"] = dict(ik_execution_seed)
        if isinstance(attachment, dict):
            body["attachment_proxy"] = {
                key: attachment.get(key)
                for key in (
                    "status",
                    "object_name",
                    "category",
                    "relative_xyz",
                    "dims",
                    "anchor_eef_xyz",
                )
            }
        if use_ori:
            body["target_quat_xyzw"] = list(target_quat)
        result = _proxy_controller_goal(meta, body)
        attachment_receipt = _refresh_attachment_proxy(meta, result)
        if attachment_receipt is not None:
            result["attachment_proxy_receipt"] = attachment_receipt
        return result

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
    # Latch the pre-motion trunk pose so every Cartesian substep re-sends it.
    # pose_result is whichever source above produced a usable EE pose, so it is
    # the freshest observation available before the arm starts moving.
    _capture_trunk_hold(meta, pose_result if isinstance(pose_result, dict) else {})
    final_result: dict = {}
    final_reward = 0.0
    final_terminated = False
    control_error = ""
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

        attachment = meta.get("_attachment_proxy")
        # A tentative proxy is checked too: its relative_xyz was measured from
        # the real object pose when armed, so it is no less accurate than a
        # confirmed one.  Waiting for confirmation left the first ~1.5 cm of
        # every post-grasp carry — the lift — entirely unguarded.
        if (
            enable_collision_check
            and isinstance(attachment, dict)
            and attachment.get("status") in {"tentative", "confirmed"}
        ):
            predicted_eef = [
                current_xyz[0] + ax * scale * batch_steps,
                current_xyz[1] + ay * scale * batch_steps,
                current_xyz[2] + az * scale * batch_steps,
            ]
            collision_detected, collision_info = _check_attached_object_sweep(
                attachment,
                _safety_obstacles(meta),
                current_xyz,
                predicted_eef,
                baseline_eef_xyz=current_xyz,
            )
            if collision_detected:
                break

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
            if final_result.get("error"):
                # A worker-side control failure is not a motion sample.  Stop
                # immediately instead of issuing the same action for the rest
                # of num_steps and eventually returning an empty end pose.
                # Keep current_xyz below as the last trustworthy pose so the
                # caller can reconcile or reset from explicit feedback.
                control_error = str(final_result.get("error"))
                final_terminated = bool(
                    final_result.get("terminated") or final_result.get("truncated")
                )
                break
            if final_result.get("terminated") or final_result.get("truncated"):
                final_terminated = True
                break

        if final_terminated or control_error:
            break

        attachment_receipt = _refresh_attachment_proxy(meta, final_result)
        if attachment_receipt is not None:
            final_result["attachment_proxy_receipt"] = attachment_receipt

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
        # BEHAVIOR is listed here even though its checker reports unavailable:
        # routing it through means move_to returns cuRobo's stated reason
        # ("no model for R1Pro") instead of a bare available:False that reads
        # identically to "scene is clear".
        if enable_collision_check and backend in ("libero", "maniskill", "behavior"):
            jp = _extract_joint_positions_from_result(final_result)
            # The arm-vs-world check always uses privileged geometry.  Gating it
            # on the public include_objects flag meant cuRobo's world was never
            # populated in the default path, so max_world_penetration was
            # structurally 0.0 and only self-collision was ever evaluated.
            # Excluding just the approach target preserves the original intent
            # (a grasp target must not read as an obstacle pre-contact) without
            # discarding the rest of the scene.
            objects = _safety_obstacles(meta, approach_target_xyz=(x, y, z))
            try:
                checker = get_checker(handle, backend)
                # Ask the checker even with no joint positions: an unsupported
                # robot must still report why, and gating that on jp would drop
                # the reason for any backend whose observation omits them.
                collision_detected, collision_info = checker.check(
                    jp or [], objects,
                    joint_names=_extract_joint_names_from_result(final_result),
                )
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
    if len(final_xyz) < 3:
        final_xyz = current_xyz
    final_quat = _extract_ee_quat_from_result(final_result) if (use_ori and total_steps > 0) else []
    if use_ori and len(final_quat) < 4:
        final_quat = current_quat

    result: dict = {
        "target": {"x": x, "y": y, "z": z},
        "start": {"xyz": start_xyz},
        "end": {"xyz": final_xyz[:3] if len(final_xyz) >= 3 else final_xyz},
        "steps_executed": total_steps,
        "terminated": final_terminated,
        "reward": final_reward,
    }
    final_position_error = (
        _math.sqrt(
            (x - final_xyz[0]) ** 2
            + (y - final_xyz[1]) ** 2
            + (z - final_xyz[2]) ** 2
        )
        if len(final_xyz) >= 3
        else None
    )
    max_axis_position_error = (
        max(abs(x - final_xyz[0]), abs(y - final_xyz[1]), abs(z - final_xyz[2]))
        if len(final_xyz) >= 3
        else None
    )
    final_orientation_error = (
        _quat_angular_distance(final_quat, target_quat)
        if use_ori and len(final_quat) >= 4
        else None
    )
    reached_target = bool(
        max_axis_position_error is not None
        and max_axis_position_error < tolerance
        and (
            not use_ori
            or (
                final_orientation_error is not None
                and final_orientation_error < ori_tolerance
            )
        )
        and not final_terminated
        and not control_error
        and not collision_detected
    )
    result["reached_target"] = reached_target
    if final_position_error is not None:
        result["position_error_m"] = final_position_error
        result["max_axis_position_error_m"] = max_axis_position_error
    if final_orientation_error is not None:
        result["orientation_error_rad"] = final_orientation_error
        result["orientation_error_deg"] = _math.degrees(final_orientation_error)
    if reached_target:
        result["stop_reason"] = "target_reached"
    elif collision_detected:
        result["stop_reason"] = "collision_detected"
    elif control_error:
        result["stop_reason"] = "control_step_failed"
    elif final_terminated:
        result["stop_reason"] = "episode_terminated"
    elif total_steps >= num_steps:
        result["stop_reason"] = "iteration_limit"
    else:
        result["stop_reason"] = "controller_stopped"
    result["controller_receipt"] = {
        "schema_version": "openeta.controller_execution_receipt.v1",
        "controller_id": str(
            controller_capability.get("controller_id") or f"{backend}.undeclared"
        ),
        "configured_name": str(controller_capability.get("configured_name") or ""),
        "command_interface": str(
            controller_capability.get("command_interface") or "backend_default"
        ),
        "goal_executor": str(
            controller_capability.get("goal_executor")
            or "openeta.outer_closed_loop_cartesian.v1"
        ),
        "execution_location": str(
            controller_capability.get("execution_location") or "mcp_server"
        ),
        "orientation_policy": "explicit" if use_ori else "preserve_current",
        "iteration_budget": int(num_steps),
        "steps_executed": int(total_steps),
        "stop_reason": result["stop_reason"],
        "reached_target": reached_target,
    }
    if control_error:
        result["ok"] = False
        result["code"] = "control_step_failed"
        result["error"] = control_error
    if use_ori:
        result["target"]["roll"] = roll
        result["target"]["pitch"] = pitch
        result["target"]["yaw"] = yaw
        result["start"]["quat_xyzw"] = start_quat
        result["end"]["quat_xyzw"] = final_quat[:4] if len(final_quat) >= 4 else final_quat

    # ── collision summary ──────────────────────────────────────────
    if enable_collision_check and collision_detected:
        collision_message = str(collision_info.get("message") or "").strip()
        result["collision"] = {
            "detected": True,
            "message": collision_message
            or (
                f"Collision detected at step {total_steps}: "
                f"world_penetration={collision_info.get('max_world_penetration', 0.0):.4f}m, "
                f"self_penetration={collision_info.get('max_self_penetration', 0.0):.4f}m"
            ),
            **{k: v for k, v in collision_info.items() if k != "available"},
        }
    elif enable_collision_check and collision_info.get("available"):
        result["collision"] = {
            "detected": False,
            **{k: v for k, v in collision_info.items() if k != "available"},
        }
        # Keep the skip reason explicit even though the complete checker receipt
        # is projected above.  This guards future receipt filtering from turning
        # "not checked" into an unexplained clear result.
        if collision_info.get("reason") and not (
            result["collision"].get("world_checked", False)
            and result["collision"].get("self_checked", False)
        ):
            result["collision"]["reason"] = collision_info["reason"]
    elif enable_collision_check:
        result["collision"] = {
            "detected": False,
            "available": False,
            "reason": collision_info.get("reason", "collision checking unavailable"),
        }

    return result


def _trajectory_pose_arguments(pose: dict, *, index: int) -> dict:
    """Convert one public world-frame trajectory pose to move_to arguments."""

    import math as _math

    if not isinstance(pose, dict) or pose.get("frame", "world") != "world":
        raise ValueError(f"trajectory[{index}] must be one world-frame pose")
    xyz = pose.get("xyz")
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError(f"trajectory[{index}].xyz must contain three finite numbers")
    values = []
    for value in xyz:
        if isinstance(value, bool):
            raise ValueError(f"trajectory[{index}].xyz must contain three finite numbers")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"trajectory[{index}].xyz must contain three finite numbers"
            ) from exc
        if not _math.isfinite(parsed):
            raise ValueError(f"trajectory[{index}].xyz must contain three finite numbers")
        values.append(parsed)
    arguments = {"x": values[0], "y": values[1], "z": values[2]}
    euler = pose.get("euler_xyz_deg")
    if isinstance(euler, (list, tuple)) and len(euler) == 3:
        parsed_euler = [float(value) for value in euler]
        if not all(_math.isfinite(value) for value in parsed_euler):
            raise ValueError(f"trajectory[{index}].euler_xyz_deg must be finite")
        arguments.update(
            {"roll": parsed_euler[0], "pitch": parsed_euler[1], "yaw": parsed_euler[2]}
        )
        return arguments
    quaternion = pose.get("quat_xyzw")
    if quaternion is not None:
        if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
            raise ValueError(f"trajectory[{index}].quat_xyzw must contain four finite numbers")
        qx, qy, qz, qw = [float(value) for value in quaternion]
        if not all(_math.isfinite(value) for value in (qx, qy, qz, qw)):
            raise ValueError(f"trajectory[{index}].quat_xyzw must contain four finite numbers")
        norm = _math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-9:
            raise ValueError(f"trajectory[{index}].quat_xyzw must be non-zero")
        qx, qy, qz, qw = [value / norm for value in (qx, qy, qz, qw)]
        roll = _math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        pitch = (
            _math.copysign(_math.pi / 2.0, sin_pitch)
            if abs(sin_pitch) >= 1.0
            else _math.asin(sin_pitch)
        )
        yaw = _math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        arguments.update(
            {
                "roll": _math.degrees(roll),
                "pitch": _math.degrees(pitch),
                "yaw": _math.degrees(yaw),
            }
        )
        return arguments
    matrix = pose.get("rotation_matrix")
    if matrix is None:
        return arguments
    if (
        not isinstance(matrix, (list, tuple))
        or len(matrix) != 3
        or any(not isinstance(row, (list, tuple)) or len(row) != 3 for row in matrix)
    ):
        raise ValueError(f"trajectory[{index}].rotation_matrix must be a finite 3x3 matrix")
    rows = [[float(value) for value in row] for row in matrix]
    if not all(_math.isfinite(value) for row in rows for value in row):
        raise ValueError(f"trajectory[{index}].rotation_matrix must be a finite 3x3 matrix")
    pitch = _math.asin(max(-1.0, min(1.0, -rows[2][0])))
    cosine_pitch = _math.cos(pitch)
    if abs(cosine_pitch) > 1e-8:
        roll = _math.atan2(rows[2][1], rows[2][2])
        yaw = _math.atan2(rows[1][0], rows[0][0])
    else:
        roll = _math.atan2(-rows[1][2], rows[1][1])
        yaw = 0.0
    arguments.update(
        {
            "roll": _math.degrees(roll),
            "pitch": _math.degrees(pitch),
            "yaw": _math.degrees(yaw),
        }
    )
    return arguments


def _trajectory_waypoint_reached(result: dict, waypoint: dict, *, tolerance: float) -> bool:
    if not isinstance(result, dict) or result.get("error"):
        return False
    # When the controller publishes an authoritative attainment verdict, the
    # route wrapper must not overwrite it with a position-only approximation.
    # This matters for stable-arrival and explicit-orientation failures.
    if "reached_target" in result and result.get("reached_target") is not True:
        return False
    collision = result.get("collision")
    if isinstance(collision, dict) and collision.get("detected") is True:
        return False
    end = result.get("end")
    end_xyz = end.get("xyz") if isinstance(end, dict) else None
    target_xyz = [waypoint.get("x"), waypoint.get("y"), waypoint.get("z")]
    if (
        not isinstance(end_xyz, (list, tuple))
        or len(end_xyz) < 3
        or any(not isinstance(value, (int, float)) for value in [*end_xyz[:3], *target_xyz])
    ):
        return False
    return all(
        abs(float(end_xyz[index]) - float(target_xyz[index])) <= tolerance
        for index in range(3)
    )


def _condition_c_route_entries(
    bundle: object,
    trajectory: list[dict],
) -> list[dict]:
    """Validate the host-private receipt bundle against the resolved public path."""

    if not isinstance(bundle, dict):
        raise ValueError("condition C requires route_execution_bundle")
    if bundle.get("schema_version") != "openeta.experimental_route_execution_bundle.v1":
        raise ValueError("route_execution_bundle has an unsupported schema_version")
    if bundle.get("condition") != "C":
        raise ValueError("route_execution_bundle is not authorized for condition C")
    if bundle.get("authority") != "host_memory_exact_receipt_resolution":
        raise ValueError("route_execution_bundle lacks host receipt authority")
    entries = bundle.get("entries")
    if not isinstance(entries, list) or len(entries) != len(trajectory):
        raise ValueError("route_execution_bundle must contain one entry per waypoint")
    validated: list[dict] = []
    for index, (entry, raw_pose) in enumerate(zip(entries, trajectory)):
        if not isinstance(entry, dict):
            raise ValueError(f"route_execution_bundle.entries[{index}] must be an object")
        receipt_id = str(entry.get("source_ik_receipt_id") or "").strip()
        if not receipt_id:
            raise ValueError(
                f"route_execution_bundle.entries[{index}] has no source receipt id"
            )
        pose = entry.get("target_pose")
        if not isinstance(pose, dict):
            raise ValueError(
                f"route_execution_bundle.entries[{index}] has no target_pose"
            )
        private_args = _trajectory_pose_arguments(pose, index=index)
        public_args = _trajectory_pose_arguments(raw_pose, index=index)
        if any(
            abs(float(private_args[axis]) - float(public_args[axis])) > 1e-9
            for axis in ("x", "y", "z")
        ):
            raise ValueError(
                f"route_execution_bundle.entries[{index}] target does not match "
                "the host-resolved trajectory"
            )
        validated.append({**entry, "execution_arguments": private_args})
    return validated


def _sequential_route_preview(
    meta: dict,
    entry: dict,
    *,
    index: int,
    tolerance: float,
    ori_tolerance: float,
) -> tuple[dict, dict | None]:
    """Re-preview one route endpoint from the actual preceding segment end."""

    arguments = dict(entry.get("execution_arguments") or {})
    preview_body: dict = {
        "target_xyz": [arguments["x"], arguments["y"], arguments["z"]],
        "position_tolerance_m": float(tolerance),
        "orientation_tolerance_rad": float(ori_tolerance),
        "preserve_current_orientation": not all(
            key in arguments for key in ("roll", "pitch", "yaw")
        ),
    }
    if all(key in arguments for key in ("roll", "pitch", "yaw")):
        preview_body["target_euler_xyz_deg"] = [
            arguments["roll"],
            arguments["pitch"],
            arguments["yaw"],
        ]
    raw = _proxy_reachability(meta, preview_body)
    raw = raw if isinstance(raw, dict) else {}
    candidate = raw.get("best_candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    joints = candidate.get("joint_positions")
    feasible = (
        raw.get("status") == "reachable"
        and raw.get("feasible") is True
        and isinstance(joints, list)
        and len(joints) == 7
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in joints
        )
    )
    source_id = str(entry.get("source_ik_receipt_id") or "")
    preview_id = hashlib.sha256(
        f"{source_id}:{index}:{raw.get('target')}:{joints}".encode("utf-8")
    ).hexdigest()[:20]
    receipt = {
        "schema_version": "openeta.sequential_route_preview_receipt.v1",
        "index": index,
        "source_ik_receipt_id": source_id,
        "preview_id": preview_id,
        "status": raw.get("status", "unknown"),
        "feasible": feasible,
        "reason_code": raw.get("reason_code"),
        "message": raw.get("message"),
        "target": raw.get("target"),
        "joint_margin_min_rad": candidate.get("joint_margin_min_rad"),
        "joint_travel_l2_rad": candidate.get("joint_travel_l2_rad"),
        "preview_state": "actual_preceding_segment_end",
        "path_collision_checked": False,
        "path_collision_authority": "controller_per_step_only",
    }
    if not feasible:
        return receipt, None
    seed = {
        "schema_version": "openeta.ik_execution_seed.v1",
        "receipt_id": preview_id,
        "pose_policy_signature": f"condition-c-route:{source_id}",
        "joint_positions": [float(value) for value in joints],
        "preview_tolerances": {
            "position_tolerance_m": float(tolerance),
            "orientation_tolerance_rad": float(ori_tolerance),
        },
    }
    return receipt, seed


@_blocking_tool
@_serialized_env_control
def follow_eef_trajectory(
    handle: str,
    trajectory: list[dict],
    *,
    session_id: str = "",
    num_steps_per_waypoint: int = 60,
    tolerance: float = 0.002,
    ori_tolerance: float = 0.05,
    enable_collision_check: bool = True,
    route_execution_bundle: dict | None = None,
) -> dict:
    """Execute 1-5 short world-frame EEF waypoints sequentially.

    The same latched gripper command is retained across every waypoint. Each
    waypoint reuses the normal closed-loop move_to controller and collision
    checks; execution stops immediately on an error, collision, or termination.
    """

    if not isinstance(trajectory, list) or not 1 <= len(trajectory) <= 5:
        return {"error": "trajectory must contain between 1 and 5 world-frame poses"}
    try:
        waypoints = [
            _trajectory_pose_arguments(pose, index=index)
            for index, pose in enumerate(trajectory)
        ]
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}
    if not isinstance(num_steps_per_waypoint, int) or not 1 <= num_steps_per_waypoint <= 100:
        return {"error": "num_steps_per_waypoint must be an integer in [1, 100]"}
    profile = motion_control_profile()
    sid = session_id or _current_session.get() or ""
    route_meta = _session_envs.get(sid, {}).get(handle)
    if profile.sequential_route_preview_enabled and not isinstance(route_meta, dict):
        return {
            "ok": False,
            "code": "sequential_route_environment_missing",
            "error": f"Unknown: {handle}",
            "reached_target": False,
            "steps_executed": 0,
            "stop_reason": "route_environment_missing",
            "motion_execution_profile": profile.receipt(),
        }
    route_entries: list[dict] = []
    if profile.sequential_route_preview_enabled:
        try:
            route_entries = _condition_c_route_entries(
                route_execution_bundle,
                trajectory,
            )
        except ValueError as exc:
            return {
                "ok": False,
                "code": "sequential_route_bundle_invalid",
                "error": str(exc),
                "reached_target": False,
                "steps_executed": 0,
                "stop_reason": "route_bundle_rejected",
                "motion_execution_profile": profile.receipt(),
            }
    move_impl = getattr(move_to, "__wrapped__", None)
    if not callable(move_impl):
        return {"error": "move_to implementation is unavailable"}
    results: list[dict] = []
    sequential_previews: list[dict] = []
    completed = 0
    for index, waypoint in enumerate(waypoints):
        execution_waypoint = waypoint
        execution_seed = None
        if profile.sequential_route_preview_enabled:
            entry = route_entries[index]
            execution_waypoint = dict(entry["execution_arguments"])
            preview, execution_seed = _sequential_route_preview(
                route_meta,
                entry,
                index=index,
                tolerance=tolerance,
                ori_tolerance=ori_tolerance,
            )
            sequential_previews.append(preview)
            if execution_seed is None:
                results.append(
                    {
                        "ok": False,
                        "code": "sequential_route_preview_rejected",
                        "error": (
                            f"waypoint {index} was not authorized from the actual "
                            f"preceding endpoint: {preview.get('message') or preview.get('reason_code')}"
                        ),
                        "reached_target": False,
                        "steps_executed": 0,
                        "stop_reason": "sequential_preview_rejected",
                    }
                )
                break
        result = move_impl(
            handle=handle,
            session_id=session_id,
            num_steps=num_steps_per_waypoint,
            tolerance=tolerance,
            ori_tolerance=ori_tolerance,
            enable_collision_check=enable_collision_check,
            ik_execution_seed=execution_seed,
            **execution_waypoint,
        )
        results.append(result)
        reached = _trajectory_waypoint_reached(
            result,
            execution_waypoint,
            tolerance=tolerance,
        )
        if reached:
            completed += 1
        if (
            not isinstance(result, dict)
            or result.get("error")
            or result.get("terminated")
            or result.get("truncated")
            or (isinstance(result.get("collision"), dict) and result["collision"].get("detected"))
            or not reached
        ):
            break
    final = results[-1] if results else {}
    final_target = waypoints[-1] if waypoints else {}
    reached_target = completed == len(waypoints)
    controller_receipt = final.get("controller_receipt")
    if isinstance(controller_receipt, dict):
        controller_receipt = {
            **controller_receipt,
            "trajectory_waypoints_requested": len(waypoints),
            "trajectory_waypoints_completed": completed,
        }
    return {
        "trajectory": trajectory,
        "waypoints_requested": len(trajectory),
        "waypoints_completed": completed,
        "start": results[0].get("start", {}) if results else {},
        "end": final.get("end", {}),
        "target": {
            key: final_target[key]
            for key in ("x", "y", "z", "roll", "pitch", "yaw")
            if key in final_target
        },
        "reached_target": reached_target,
        "steps_executed": sum(
            int(result.get("steps_executed") or 0)
            for result in results
            if isinstance(result, dict)
        ),
        "terminated": bool(final.get("terminated")),
        "truncated": bool(final.get("truncated")),
        "reward": final.get("reward", 0.0),
        "stop_reason": final.get("stop_reason"),
        "code": final.get("code"),
        "collision": final.get("collision"),
        "waypoint_results": results,
        **(
            {
                "motion_execution_profile": profile.receipt(),
                "sequential_route_preview": {
                    "schema_version": "openeta.sequential_route_preview_chain.v1",
                    "policy": "just_in_time_from_actual_segment_end",
                    "waypoints_previewed": len(sequential_previews),
                    "waypoints_authorized": sum(
                        item.get("feasible") is True for item in sequential_previews
                    ),
                    "path_collision_checked": False,
                    "path_collision_authority": "controller_per_step_only",
                    "receipts": sequential_previews,
                },
            }
            if profile.sequential_route_preview_enabled
            else {}
        ),
        **(
            {"controller_receipt": controller_receipt}
            if isinstance(controller_receipt, dict)
            else {}
        ),
        **({"error": final.get("error")} if final.get("error") else {}),
        **(
            {"controller_failure": final.get("controller_failure")}
            if isinstance(final.get("controller_failure"), dict)
            else {}
        ),
        **(
            {"convergence_diagnostics": final.get("convergence_diagnostics")}
            if isinstance(final.get("convergence_diagnostics"), dict)
            else {}
        ),
    }


# The position-controlled fingers need two different horizons.  Opening only
# needs enough time to clear the next approach.  Closing also needs stationary
# physics steps for opposing contacts and object dynamics to settle before the
# Agent receives its post-close observation.  Ten steps merely crossed the
# legacy binary threshold and could return while a marginal grasp was still
# squeezing/slipping.  These horizons match the mature LIBERO adapter used by
# CaP-X and do not prescribe any task-level action sequence.
_GRIPPER_OPEN_STEPS = 40
_GRIPPER_CLOSE_STEPS = 60


def _gripper_actuation_receipt(
    result: dict,
    *,
    command: str,
    steps_executed: int,
) -> dict:
    state = _extract_gripper_state_from_result(result)
    openness = state.get("openness")
    measured = (
        float(openness)
        if isinstance(openness, (int, float)) and not isinstance(openness, bool)
        else None
    )
    return {
        "schema_version": "openeta.gripper_actuation_receipt.v1",
        "command": command,
        "command_latched": True,
        "steps_executed": steps_executed,
        "settling_policy": (
            "stationary_continuous_position_hold"
            if command == "close"
            else "stationary_position_actuation"
        ),
        "measured_open_fraction": measured,
        "interpretation": (
            "The binary command remained applied for the reported stationary "
            "physics horizon. Aperture is contact evidence, not attachment proof."
        ),
    }


@_blocking_tool
@_serialized_env_control
def gripper_open(handle: str, *, session_id: str = "") -> dict:
    """Open the gripper and let the position actuator settle.

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
    meta.pop("_attachment_proxy", None)
    try:
        act = make_gripper_action(meta, open_gripper=True, backend=backend)
    except ControlCodecError as exc:
        return codec_error_result(exc)
    result = _proxy_step(meta, act, num_steps=_GRIPPER_OPEN_STEPS)
    result["gripper_actuation_receipt"] = _gripper_actuation_receipt(
        result,
        command="open",
        steps_executed=_GRIPPER_OPEN_STEPS,
    )
    return result


@_blocking_tool
@_serialized_env_control
def gripper_close(
    handle: str,
    *,
    session_id: str = "",
    contact_authorization: dict | None = None,
) -> dict:
    """Close the gripper and settle physical contacts before returning.

    Args:
        handle: Environment handle from create_env.
        session_id: Optional session id to reuse an existing session.
        contact_authorization: Optional host-private compiled target evidence.
            When supplied, the tentative carried-object collision proxy is
            bound to that exact scene object instead of nearest-neighbour
            inference. It never proves attachment.

    Returns:
        Same structure as ``step_env`` (observation, reward, terminated,
        truncated, info).
    """
    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    authorized_object: dict | None = None
    authorization_receipt: dict | None = None
    if contact_authorization is not None:
        authorized_object, authorization_receipt = resolve_contact_object_authorization(
            contact_authorization,
            list(meta.get("_collision_objects", [])),
        )
        if authorized_object is None:
            return {
                "ok": False,
                "code": str(
                    authorization_receipt.get("code")
                    or "contact_authorization_resolution_failed"
                ),
                "error": str(
                    authorization_receipt.get("message")
                    or "Host attachment target evidence could not be resolved."
                ),
                "contact_authorization": authorization_receipt,
                "attachment_proxy_receipt": {
                    "schema_version": "openeta.attachment_proxy_receipt.v1",
                    "status": "not_armed",
                    "reason": "host_target_resolution_failed",
                    "attachment_proven": False,
                },
            }
    backend = meta.get("backend", "")
    meta["_gripper_cmd"] = 1.0  # latch CLOSED — held (clamping) on every subsequent step
    try:
        act = make_gripper_action(meta, open_gripper=False, backend=backend)
    except ControlCodecError as exc:
        return codec_error_result(exc)
    result = _proxy_step(meta, act, num_steps=_GRIPPER_CLOSE_STEPS)
    result["gripper_actuation_receipt"] = _gripper_actuation_receipt(
        result,
        command="close",
        steps_executed=_GRIPPER_CLOSE_STEPS,
    )
    result["attachment_proxy_receipt"] = _arm_attachment_proxy(
        meta,
        result,
        authorized_object=authorized_object,
    )
    if isinstance(authorization_receipt, dict):
        result["contact_authorization"] = authorization_receipt
    return result


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
@_serialized_env_control
def base_control(
    handle: str,
    *,
    forward: float = 0.0,
    lateral: float = 0.0,
    yaw: float = 0.0,
    torso: float = 0.0,
    trunk: float | list[float] | None = None,
    command: str = "",
    num_steps: int = 10,
    session_id: str = "",
) -> dict:
    """Drive a mobile base, and command or hold the trunk.

    The base controls are normalized rates: forward, lateral, and
    counter-clockwise yaw velocity.  A named command can be used instead.
    Motion stops when the commands stop, so omitting a base control means
    "no motion on that axis".

    Trunk control differs in kind and so differs in default.  It is a
    **position** target, not a rate: on RoboCasa a single normalized height, on
    BEHAVIOR R1Pro a 4-joint torso chain.  ``0.0`` in a position slot is not
    neutral -- it scales onto the middle of the joint range -- so ``trunk`` left
    unset means *hold the current pose*, reading the trunk back from the
    observation.  Passing ``trunk`` explicitly commands it; passing ``torso``
    keeps the original RoboCasa spelling.

    Supported where the environment declares a base: RoboCasa PandaOmron and
    BEHAVIOR R1Pro.  Fixed-base environments are rejected.
    """

    sid = session_id or _current_session.get() or ""
    _touch_session(sid)
    meta = _session_envs.get(sid, {}).get(handle)
    if not meta:
        return {"error": f"Unknown: {handle}"}
    backend = meta.get("backend", "")
    if backend == "robocasa":
        if int(meta.get("action_dim") or 0) != 12:
            return {
                "error": "base_control requires the 12-dim RoboCasa PandaOmron action layout"
            }
        return _base_control_robocasa(
            meta, forward=forward, lateral=lateral, yaw=yaw,
            torso=torso, trunk=trunk, command=command, num_steps=num_steps,
        )
    if backend == "behavior":
        return _base_control_behavior(
            meta, forward=forward, lateral=lateral, yaw=yaw,
            trunk=trunk, torso=torso, command=command, num_steps=num_steps,
        )
    return {
        "error": f"base_control is not available for backend {backend!r}",
        "code": "unsupported_base_control",
    }


def _base_control_robocasa(
    meta: dict,
    *,
    forward: float,
    lateral: float,
    yaw: float,
    torso: float,
    trunk: float | list[float] | None,
    command: str,
    num_steps: int,
) -> dict:
    """RoboCasa PandaOmron: 3 base velocities plus a 1-dim torso position."""
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
    # `trunk` is the cross-backend spelling; `torso` is kept for compatibility.
    # A scalar or a 1-element list both name RoboCasa's single torso dim.
    if trunk is not None:
        torso = float(trunk[0]) if isinstance(trunk, (list, tuple)) and trunk else float(
            trunk if not isinstance(trunk, (list, tuple)) else 0.0)
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


def _base_control_behavior(
    meta: dict,
    *,
    forward: float,
    lateral: float,
    yaw: float,
    trunk: float | list[float] | None,
    torso: float,
    command: str,
    num_steps: int,
) -> dict:
    """BEHAVIOR R1Pro: 3 holonomic base rates plus a 4-joint trunk chain.

    Slots come from the declared control_spec, never from constants: R1Pro's
    action_dim is 21 under our IK overrides but 23 under the raw
    r1pro_behavior.yaml joint controllers, so hard-coded indices would drive the
    wrong actuators in one of the two configurations.
    """
    spec = meta.get("control_spec")
    base = spec.get("base") if isinstance(spec, dict) else None
    if not isinstance(base, dict) or not base.get("supported"):
        return {
            "error": "this BEHAVIOR robot declares no mobile base",
            "code": "unsupported_base_control",
        }
    base_slots = [int(i) for i in (base.get("indices") or [])]
    if len(base_slots) != 3:
        return {
            "error": f"expected 3 holonomic base slots, got {len(base_slots)}",
            "code": "unsupported_base_control",
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

    try:
        dim = int(meta.get("action_dim") or 0) or len(
            make_cartesian_action(meta, (0.0, 0.0, 0.0), "behavior"))
    except ControlCodecError as exc:
        return codec_error_result(exc)
    action = [0.0] * dim
    for slot, value in zip(base_slots, (forward, lateral, yaw)):
        if 0 <= slot < dim:
            action[slot] = clipped(value)

    # Trunk: explicit target, or hold.  These are position commands, so an
    # unset trunk cannot be left at 0.0 -- that scales to the middle of each
    # joint's range and would move the torso on a pure base command.
    tl = trunk_layout(meta)
    trunk_report: Any = None
    trunk_slots = [int(i) for i in (tl.get("indices") or [])] if tl else []
    if tl and trunk_slots:
        explicit = trunk if trunk is not None else (torso if torso else None)
        if explicit is not None:
            values = ([float(v) for v in explicit]
                      if isinstance(explicit, (list, tuple))
                      else [float(explicit)] * len(trunk_slots))
            if len(values) != len(trunk_slots):
                return {
                    "error": (f"trunk expects {len(trunk_slots)} values "
                              f"(joints {tl.get('joint_names') or trunk_slots}), "
                              f"got {len(values)}"),
                    "code": "invalid_trunk_command",
                }
            for slot, value in zip(trunk_slots, values):
                if 0 <= slot < dim:
                    action[slot] = clipped(value)
            trunk_report = {"mode": "commanded",
                            "values": [action[s] for s in trunk_slots]}
        else:
            obs = _proxy_observe(meta)
            robot = (obs.get("observation") or {}).get("robot") or {}
            slots, held = trunk_hold_values(
                meta,
                [float(v) for v in (robot.get("joint_positions") or [])],
                [str(n) for n in (robot.get("joint_names") or [])],
            )
            if slots:
                for slot, value in zip(slots, held):
                    if 0 <= slot < dim:
                        action[slot] = float(value)
                trunk_report = {"mode": "held", "values": held}
            else:
                # Say so rather than silently sending the mid-range default:
                # the caller asked for base motion and would otherwise get an
                # unexplained torso move.
                trunk_report = {
                    "mode": "unknown",
                    "reason": ("trunk pose unavailable (needs joint_names and "
                               "declared trunk limits); slots left at their "
                               "mid-range default and the torso may move"),
                }

    result = _proxy_step(meta, action, num_steps=max(1, int(num_steps)))
    result["control"] = {
        "forward": action[base_slots[0]],
        "lateral": action[base_slots[1]],
        "yaw": action[base_slots[2]],
        "command_type": "velocity",
        "num_steps": max(1, int(num_steps)),
    }
    if trunk_report is not None:
        result["control"]["trunk"] = trunk_report
    return result


def _capture_trunk_hold(meta: dict, result: dict) -> None:
    """Latch the trunk pose from *result* so motion steps can hold it.

    Captured once before a motion rather than re-read per step: the trunk is a
    position-mode target, so re-reading a still-settling angle each step would
    chase it and drift.  "Hold" means the pose the trunk had when the motion
    started.
    """
    robot = (result.get("observation") or {}).get("robot") or {}
    jp = [float(v) for v in (robot.get("joint_positions") or [])]
    jn = [str(n) for n in (robot.get("joint_names") or [])]
    slots, values = trunk_hold_values(meta, jp, jn)
    if slots:
        meta["_trunk_hold"] = (slots, values)


def _overlay_trunk_hold(meta: dict, act: list[float]) -> None:
    """Write the latched trunk hold into *act*, if one was captured.

    Without this the trunk slots stay 0.0, which position-mode scaling turns
    into a mid-range target -- so a Cartesian arm motion would drag the torso.
    """
    held = meta.get("_trunk_hold")
    if not held:
        return
    slots, values = held
    for slot, value in zip(slots, values):
        if 0 <= int(slot) < len(act):
            act[int(slot)] = float(value)


def _make_action_for_step(meta: dict, delta_xyz: tuple[float, float, float], backend: str,
                           delta_rot: list[float] | None = None) -> list[float]:
    """Build a Cartesian motion action, holding the gripper only if latched.

    Delegates slot layout to the explicit action codec (``make_cartesian_action``).
    If the user has explicitly latched a gripper command (via gripper_open/close),
    that ±1.0 command is overlaid onto the gripper slot(s) so the fingers keep
    holding their open/closed state throughout the motion instead of relaxing to
    zero force.  If a caller constructs metadata without a latch, the motion
    action leaves the gripper slot(s) untouched.  Normal environment resets
    always establish an explicit OPEN latch before settling.

    The gripper slot is backend-specific (RoboCasa slot 6, others the last slot),
    so we reuse the codec's own gripper encoder to place the value rather than
    hard-coding an index.  The latched command is already ±1.0 — exactly what
    ``make_gripper_action`` emits — so ``open_gripper=<cmd is open>`` reproduces
    the held gripper vector.
    """
    act = make_cartesian_action(meta, delta_xyz, backend, delta_rot=delta_rot)
    _overlay_trunk_hold(meta, act)
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
@_serialized_env_control
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
    import atexit as _atexit
    import uvicorn
    from mcp.server.sse import SseServerTransport

    p = argparse.ArgumentParser(description="OpenETA MCP + Web Dashboard")
    p.add_argument("--transport", default="sse", choices=["sse", "stdio"])
    p.add_argument("--host", default=os.environ.get("MCP_HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=0)
    args = p.parse_args()
    host = args.host
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

    print(f"\n  OpenETA Dashboard:      http://{host}:{port}/")
    print(f"  MCP (Streamable HTTP):  http://{host}:{port}/mcp")
    print(f"  MCP (legacy SSE):       http://{host}:{port}/sse\n")

    # Reap workers on server exit.  Without this every worker outlives the
    # server that spawned it -- for BEHAVIOR that is a ~6.5 GB VRAM process per
    # env with no owner left to close it.  uvicorn installs its own
    # SIGINT/SIGTERM handling and returns from run(), so ``finally`` covers the
    # ordinary paths and atexit covers exits that bypass it.  A SIGKILLed
    # server can run neither, which is why workers also carry PR_SET_PDEATHSIG.
    def _reap_workers() -> None:
        try:
            _get_mgr().stop_all()
        except Exception:
            pass

    _atexit.register(_reap_workers)
    try:
        uvicorn.run(combined, host=host, port=port, log_level="warning")
    finally:
        _reap_workers()


if __name__ == "__main__":
    main()
