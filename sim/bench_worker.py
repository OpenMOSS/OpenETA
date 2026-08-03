#!/usr/bin/env python
"""Per-bench subprocess worker for OpenETA sim layer.

Each bench (libero, metaworld, maniskill) runs in its own venv Python to
avoid C-extension conflicts between mujoco, robosuite, torch, etc.

Usage::

    python sim/bench_worker.py --bench libero --port 0
    # Worker writes its port to stdout on startup.

The worker exposes a subset of the REST API:
    GET  /health
    GET  /envs?type=&q=
    POST /env              {env_id, task?, seed?, render_mode?}
    DELETE /env/{handle}
    POST /env/{handle}/reset   {seed?}
    POST /env/{handle}/step    {action?, num_steps?}
    POST /env/{handle}/observe
    POST /env/{handle}/render
    POST /env/{handle}/check_task
"""

from __future__ import annotations

import argparse, asyncio, base64, io, json, math, os, queue, sys, threading, time, uuid, warnings
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
_LOCAL_LIBERO_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party",
    "LIBERO",
)
os.environ.setdefault(
    "LIBERO_DIR",
    _LOCAL_LIBERO_DIR if os.path.isdir(_LOCAL_LIBERO_DIR) else "/tmp/LIBERO",
)
os.environ.setdefault(
    "LIBERO_DATASET_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "venvs", "libero", "assets", "datasets"),
)

# ── ensure sim/ package is importable ──────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Editable installs may already add the repository later in sys.path while
# Python keeps ``sim/`` at index 0 for file-path execution.  In that order,
# ``import adapter`` incorrectly resolves to ``sim/adapter.py``.  Make the
# package root authoritative regardless of how the worker was launched.
while _REPO in sys.path:
    sys.path.remove(_REPO)
sys.path.insert(0, _REPO)


# ══════════════════════════════════════════════════════════════════════
# Env helpers (lightweight copies from mcp_server.py — no session logic)
# ══════════════════════════════════════════════════════════════════════

import numpy as np

# Per-worker storage
_envs: dict[str, object] = {}
_last_obs: dict[str, dict] = {}
_env_errors: dict[str, str] = {}  # handle → last error message
_env_locks: dict[str, threading.Lock] = {}
_env_locks_guard = threading.Lock()
_env_step_busy_until: dict[str, float] = {}
_STEP_RENDER_GRACE_S = 2.0


class _EnvUnavailableError(RuntimeError):
    """Raised when an operation waited behind a close for the same handle."""


def _env_lock(env_handle: str) -> threading.Lock:
    """Return the per-env simulator lock, creating it atomically if needed."""

    with _env_locks_guard:
        return _env_locks.setdefault(env_handle, threading.Lock())


def _run_env_locked(env_handle: str, fn, *args, _expected_env=None, **kwargs):
    """Serialize MuJoCo/EGL access for one environment."""

    with _env_lock(env_handle):
        if _expected_env is not None and _envs.get(env_handle) is not _expected_env:
            raise _EnvUnavailableError(f"Unknown handle: {env_handle}")
        return fn(*args, **kwargs)


def _try_run_env_locked(env_handle: str, fn, *args, _expected_env=None, **kwargs):
    """Run a best-effort dashboard render, or skip while the env is busy."""

    if time.monotonic() < _env_step_busy_until.get(env_handle, 0.0):
        return None
    lock = _env_lock(env_handle)
    if not lock.acquire(blocking=False):
        return None
    try:
        if time.monotonic() < _env_step_busy_until.get(env_handle, 0.0):
            return None
        if _expected_env is not None and _envs.get(env_handle) is not _expected_env:
            return None
        return fn(*args, **kwargs)
    finally:
        lock.release()


def _env_render_is_busy(env_handle: str) -> bool:
    """Check render busy state without waiting or running simulator code."""

    if time.monotonic() < _env_step_busy_until.get(env_handle, 0.0):
        return True
    lock = _env_lock(env_handle)
    if not lock.acquire(blocking=False):
        return True
    lock.release()
    return False


def _note_env_step_activity(env_handle: str) -> None:
    _env_step_busy_until[env_handle] = time.monotonic() + _STEP_RENDER_GRACE_S


def _close_env_locked(env_handle: str, env, *, close_simulator: bool) -> bool:
    """Close and forget the current env without replacing its lock object."""

    with _env_lock(env_handle):
        if _envs.get(env_handle) is not env:
            return False
        _envs.pop(env_handle, None)
        _last_obs.pop(env_handle, None)
        _env_errors.pop(env_handle, None)
        _env_step_busy_until.pop(env_handle, None)
        try:
            if close_simulator:
                env.close()
        except Exception:
            pass
        return True


class _MainThreadExecutor:
    """Marshal simulator calls from uvicorn's thread onto the process main thread."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[Future, object, tuple, dict]] = queue.Queue()
        self._shutdown = False
        self._shutdown_lock = threading.Lock()

    def submit(self, fn, *args, **kwargs) -> Future:
        with self._shutdown_lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            future: Future = Future()
            self._queue.put((future, fn, args, kwargs))
        return future

    def run_once(self, timeout: float = 0.1) -> None:
        try:
            future, fn, args, kwargs = self._queue.get(timeout=timeout)
        except queue.Empty:
            return
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(fn(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        """Reject new work and optionally cancel work not yet run by the main loop."""

        del wait  # The caller owns and joins the process main thread.
        with self._shutdown_lock:
            self._shutdown = True
        if cancel_futures:
            while True:
                try:
                    future, _, _, _ = self._queue.get_nowait()
                except queue.Empty:
                    break
                future.cancel()


def _startup_sim_executor() -> None:
    """Install the worker's one stable simulator thread before serving requests."""

    if getattr(app.state, "sim_executor", None) is None:
        app.state.sim_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="openeta-simulator",
        )


def _shutdown_sim_executor() -> None:
    """Close the simulator executor after uvicorn has drained requests."""

    executor = getattr(app.state, "sim_executor", None)
    app.state.sim_executor = None
    if executor is not None:
        executor.shutdown(wait=True, cancel_futures=True)


@asynccontextmanager
async def _sim_executor_lifespan(_app):
    _startup_sim_executor()
    try:
        yield
    finally:
        _shutdown_sim_executor()


async def _run_sim_call(fn, *args, **kwargs):
    """Run every simulator/EGL call on this worker's stable execution thread."""
    executor = getattr(app.state, "sim_executor", None)
    if executor is None:
        raise RuntimeError("simulator executor is not running")
    return await asyncio.wrap_future(executor.submit(fn, *args, **kwargs))


def _sanitize_json_payload(payload: object) -> tuple[object, list[str]]:
    """Replace non-finite numeric values before Starlette JSON encoding.

    MuJoCo controllers can transiently expose NaN/Inf values in velocities or
    camera metadata even when the environment remains step-able. Starlette's
    strict JSON encoder rejects those values and used to turn a recoverable
    observation into HTTP 500. Preserve the numeric schema with a zero
    replacement and attach explicit diagnostic paths to the response.
    """

    warnings_found: list[str] = []

    def visit(value: object, path: str) -> object:
        if isinstance(value, np.ndarray):
            return visit(value.tolist(), path)
        if isinstance(value, np.generic):
            return visit(value.item(), path)
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            warnings_found.append(path)
            return 0.0
        if isinstance(value, dict):
            return {
                str(key): visit(item, f"{path}.{key}")
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    return visit(payload, "$"), warnings_found


def _json_response(payload: object, status_code: int = 200) -> "JSONResponse":
    """Return strict JSON and surface any non-finite-value replacements."""

    sanitized, warning_paths = _sanitize_json_payload(payload)
    if warning_paths and isinstance(sanitized, dict):
        diagnostic = {
            "code": "nonfinite_values_replaced",
            "replacement": 0.0,
            "count": len(warning_paths),
            "paths": warning_paths[:32],
        }
        if isinstance(sanitized.get("observation"), dict):
            info = sanitized.setdefault("info", {})
            if isinstance(info, dict):
                info["serialization_warning"] = diagnostic
        else:
            metadata = sanitized.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["serialization_warning"] = diagnostic
    return JSONResponse(sanitized, status_code=status_code)


def _init_bench(bench: str) -> None:
    """Register all envs for *bench* via hot_activate."""
    import sim.env_registry  # noqa: F401 — triggers gym registration
    from sim.env_registry import hot_activate

    if bench not in ("dummy", "behavior"):
        ok = hot_activate(bench)
        if not ok:
            print(f"[worker:{bench}] WARNING: hot_activate returned False", flush=True)


def _auto_activate(env_id: str) -> None:
    """Auto-activate a bench from an env_id if not already registered."""
    from sim.env_registry import hot_activate
    bench = env_id.split("/")[1].split("_")[0] if "/" in env_id else ""
    mapping = {
        "metaworld": "metaworld",
        "maniskill": "maniskill",
        "libero": "libero",
        "robocasa": "robocasa",
        "genesis": "genesis",
        "d4rl": "d4rl",
    }
    target = mapping.get(bench)
    if target:
        hot_activate(target)


def _make_env(eid: str, task: str = "", seed: int = 0, render_mode: str = "rgb_array",
               image_width: int | None = None, image_height: int | None = None,
               include_objects: bool = False, robot: str | None = None):
    """Create a gym env, auto-activating the bench if needed."""
    import gymnasium as gym
    _auto_activate(eid)
    kwargs: dict = {}
    if image_width is not None:
        kwargs["image_width"] = image_width
    if image_height is not None:
        kwargs["image_height"] = image_height
    if include_objects:
        kwargs["include_objects"] = True
    if robot:
        kwargs["robot"] = robot
    return gym.make(eid, task=task, seed=seed, render_mode=render_mode, **kwargs)


def _render_frame_to_b64(env) -> tuple[str | None, int, int]:
    """Render one frame; return (base64_png, width, height) or (None, 0, 0)."""
    try:
        frame = env.render()
    except Exception:
        frame = None
    if frame is None:
        return None, 0, 0
    arr = np.asarray(frame)
    if arr.ndim < 3:
        return None, 0, 0
    enc, w, h = _encode_pixels_to_base64(arr)
    return enc, w, h


def _encode_pixels_to_base64(pixels, *, mode: str = "RGB") -> tuple[str, int, int] | tuple[None, None, None]:
    """Encode pixel array to base64 PNG."""
    from PIL import Image

    arr = np.asarray(pixels)
    if arr.ndim < 2:
        return None, None, None
    h, w = arr.shape[:2]

    if mode == "depth":
        arr = arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 1:
            arr = arr[:, :, 0]
        arr = np.nan_to_num(arr, nan=0.0, posinf=65.535, neginf=0.0)
        # uint16 millimetres — fixed scale, not per-frame normalisation
        arr = np.clip(np.round(arr * 1000.0), 0, 65535).astype(np.uint16)
    else:
        arr = arr[..., :3].astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), w, h


def _inject_render_frame(env, obs: dict) -> None:
    """Call env.render() and inject the frame into obs['cameras'] if missing.

    For MetaWorld, attempts to also extract depth via the gymnasium
    MujocoRenderer's ``rgbd_tuple`` render mode.
    """
    try:
        frame = env.render()
    except Exception:
        return
    if frame is None:
        return
    arr = np.asarray(frame)
    if arr.ndim < 3:
        return
    obs.setdefault("cameras", {})
    cam_name = "view" if not obs["cameras"] else "render"
    if cam_name not in obs["cameras"]:
        h, w = arr.shape[:2]
        cam: dict = {"rgb": arr}
        # MetaWorld: try to get depth from mujoco_renderer
        _depth = None
        try:
            from sim.unified_env import UnifiedEnv
            if isinstance(env, UnifiedEnv):
                _inner = getattr(env, "_env", None)
                if _inner is not None and hasattr(_inner, "unwrapped"):
                    _uw = _inner.unwrapped
                    _mr = getattr(_uw, "mujoco_renderer", None)
                    if _mr is not None and hasattr(_mr, "render"):
                        _, _depth = _mr.render("rgbd_tuple")
                        if _depth is not None:
                            cam["depth"] = np.flipud(np.asarray(_depth))
                cp = env._extract_camera_params(cam_name, image_width=w, image_height=h)
                cam.update(cp)
        except Exception:
            pass
        obs["cameras"][cam_name] = cam


def _env_obs_to_mcp(obs: dict) -> dict:
    """Convert a UnifiedEnv obs dict to MCP-serialisable EnvObservation."""
    from adapter.protocol import EnvObservation
    return EnvObservation.from_dict(obs).to_mcp_dict()


def _step_with_image(
    env,
    act,
    handle: str = "",
    render: bool = True,
    include_cameras: bool = True,
) -> dict:
    """Execute one env step, return StepResult as MCP dict.

    Render failure is non-fatal — we still return the step result without
    a camera frame so the user can decide to retry or reset.

    ``render`` controls whether this physics step refreshes the worker's raw
    camera cache.  ``include_cameras`` independently controls whether those
    RGB-D arrays are serialised into the HTTP response.  Controller loops only
    need proprioception, and converting every 512x512 image into nested Python
    lists can dominate the entire action latency.  They can therefore refresh
    the dashboard periodically while returning a camera-free control result.
    """
    from adapter.protocol import EnvObservation, StepResult

    obs, rew, term, trunc, info = env.step(act)
    if render:
        try:
            _inject_render_frame(env, obs)
        except Exception:
            pass  # render is best-effort; don't lose the step result
    if handle:
        _last_obs[handle] = obs
    result_obs = obs
    if not include_cameras and isinstance(obs, dict) and obs.get("cameras"):
        result_obs = dict(obs)
        result_obs["cameras"] = {}
    env_obs = EnvObservation.from_dict(result_obs)
    safe_info = _json_safe_info(info)
    return StepResult(
        observation=env_obs,
        reward=float(rew),
        terminated=bool(term),
        truncated=bool(trunc),
        info=safe_info,
    ).to_mcp_dict()


def _reset_with_image(env, seed=None, handle: str = "") -> dict:
    """Reset env and retain any serialisable initialization provenance."""
    from adapter.protocol import EnvObservation

    obs, info = env.reset(seed=seed)
    _inject_render_frame(env, obs)
    if handle:
        _last_obs[handle] = obs
    result = EnvObservation.from_dict(obs).to_mcp_dict()
    result_metadata = result.setdefault("metadata", {})
    if isinstance(result_metadata, dict):
        result_metadata["reset_info"] = _json_safe_info(info)
    return result


def _json_safe_info(info: Any) -> dict:
    """Keep simulator diagnostics serialisable without losing reset provenance."""

    if not isinstance(info, dict):
        return {"raw_info": str(info)}
    safe_info: dict = {}
    for key, value in info.items():
        try:
            json.dumps({key: value})
            safe_info[str(key)] = value
        except (TypeError, ValueError):
            safe_info[str(key)] = str(value)
    return safe_info


def _observe_with_image(env, handle: str = "") -> dict:
    """Return the last cached observation as MCP dict.

    Injects a fresh render frame into the cached raw UnifiedEnv dict.
    If the cache contains MCP-format data (e.g. from a prior render_all
    that was incorrectly written back), falls back to re-rendering only.
    """
    from adapter.protocol import EnvObservation

    obs = _last_obs.get(handle, {})
    if not obs:
        return {"error": "No observation cached — call reset_env or step_env first"}

    # Detect MCP-formatted cache (cameras is a list of frame dicts, not a
    # dict keyed by camera name).  If we see this, render-only fallback.
    cameras = obs.get("cameras")
    if isinstance(cameras, list):
        # Cache was overwritten with MCP data — can't inject render frame.
        # Return as-is; the caller will get stale frames but won't crash.
        return {"error": "Cache is MCP-formatted — call reset_env or step_env first"}

    _inject_render_frame(env, obs)
    return EnvObservation.from_dict(obs).to_mcp_dict()


def _render_to_mcp(env) -> dict:
    """Render current frame, return as MCP dict."""
    from adapter.protocol import _encode_pixels_to_base64 as _enc

    enc, w, h = _render_frame_to_b64(env)
    if enc:
        return {
            "cameras": [{
                "frame_id": "render",
                "rgb_base64": enc,
                "width": w,
                "height": h,
                "depth_base64": None,
                "intrinsics": {},
                "extrinsics": {},
            }]
        }
    return {"error": "No render frame"}


_LIBERO_MULTIVIEW_POSES = (
    (0.0, -25.0),
    (60.0, -25.0),
    (120.0, -25.0),
    (180.0, -25.0),
    (240.0, -25.0),
    (300.0, -25.0),
    (35.0, -70.0),
)


def _unwrap_libero_sim(env):
    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "sim"):
            return current
        next_value = getattr(current, "env", None)
        if next_value is None:
            next_value = getattr(current, "_env", None)
        if next_value is None:
            next_value = getattr(current, "unwrapped", None)
        if next_value is None or next_value is current:
            break
        current = next_value
    raise RuntimeError("current environment does not expose a LIBERO MuJoCo simulator")


def _unwrap_libero_task_env(env):
    """Find the LIBERO domain object that owns parsed goals and object states."""

    current = env
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if (
            hasattr(current, "parsed_problem")
            and hasattr(current, "object_states_dict")
            and hasattr(current, "sim")
        ):
            return current
        candidates = (
            getattr(current, "env", None),
            getattr(current, "_env", None),
            getattr(current, "unwrapped", None),
        )
        next_value = next(
            (
                value
                for value in candidates
                if value is not None and value is not current
            ),
            None,
        )
        if next_value is None:
            break
        current = next_value
    return None


def _libero_task_diagnostics(env) -> dict:
    """Return host-only native goal-predicate diagnostics.

    This is diagnostic truth, not an action affordance.  It lets the evaluator
    distinguish perception, contact, and placement failures without leaking
    object poses into the Operator's three-tool context.
    """

    task_env = _unwrap_libero_task_env(env)
    if task_env is None:
        return {"available": False, "reason": "libero_task_env_unavailable"}
    goal_state = list(
        getattr(task_env, "parsed_problem", {}).get("goal_state", []) or []
    )
    object_states = getattr(task_env, "object_states_dict", {})
    body_ids = getattr(task_env, "obj_body_id", {})
    data = task_env.sim.data
    predicates: list[dict] = []
    for raw_state in goal_state:
        state = list(raw_state)
        item: dict = {
            "predicate": str(state[0]) if state else "",
            "arguments": [str(value) for value in state[1:]],
        }
        try:
            item["satisfied"] = bool(task_env._eval_predicate(state))
        except Exception as exc:
            item["evaluation_error"] = str(exc)
        positions: dict[str, list[float]] = {}
        for name in state[1:]:
            body_id = body_ids.get(name)
            if body_id is None:
                continue
            try:
                positions[str(name)] = [
                    float(value) for value in data.body_xpos[int(body_id)]
                ]
            except (IndexError, TypeError, ValueError):
                continue
        if positions:
            item["body_positions_world_m"] = positions
        if len(state) == 3:
            first = positions.get(str(state[1]))
            second = positions.get(str(state[2]))
            if first is not None and second is not None:
                item["xy_center_distance_mm"] = float(
                    np.linalg.norm(
                        np.asarray(first[:2], dtype=np.float64)
                        - np.asarray(second[:2], dtype=np.float64)
                    )
                    * 1000.0
                )
            first_state = object_states.get(state[1])
            second_state = object_states.get(state[2])
            if first_state is not None and second_state is not None:
                predicate = str(state[0]).strip().lower()
                try:
                    item["contact"] = bool(second_state.check_contact(first_state))
                except Exception as exc:
                    item["contact_error"] = str(exc)
                if predicate == "in":
                    try:
                        item["contained"] = bool(
                            second_state.check_contain(first_state)
                        )
                    except Exception as exc:
                        item["containment_error"] = str(exc)
                elif predicate in {"on", "stack"} and (
                    first is not None and second is not None
                ):
                    item["support_z_lte_object_z"] = bool(second[2] <= first[2])
        predicates.append(item)
    return {"available": True, "predicates": predicates}


def _libero_robot_visual_geom_ids(holder) -> list[int]:
    """Return compiled visual geom ids owned by the robot and its gripper.

    LIBERO / robosuite places all visual meshes in MuJoCo geom group 1, so
    disabling that whole group would also hide task objects and fixtures.
    Resolve only the robot-owned visual geom names instead.  This is renderer
    bookkeeping, not simulator-state or object-pose access.
    """

    names: set[str] = set()
    for robot in list(getattr(holder, "robots", ()) or ()):
        robot_model = getattr(robot, "robot_model", None)
        if robot_model is not None:
            names.update(str(name) for name in getattr(robot_model, "visual_geoms", ()) or ())
        gripper = getattr(robot, "gripper", None)
        if gripper is not None:
            names.update(str(name) for name in getattr(gripper, "visual_geoms", ()) or ())

    model = holder.sim.model
    ids: list[int] = []
    for name in sorted(names):
        try:
            geom_id = int(model.geom_name2id(name))
        except Exception:
            continue
        if geom_id >= 0:
            ids.append(geom_id)
    return ids


def _render_libero_multiview(
    env,
    *,
    width: int,
    height: int,
    hide_robot: bool = False,
    lookat_xyz_m: list[float] | None = None,
    distance_m: float = 1.3,
) -> dict:
    """Render synchronized free-camera RGB-D views without stepping physics.

    ``hide_robot`` removes robot visual meshes from this inspection render
    while leaving task objects and fixtures untouched.  Exact grip-site,
    finger-pad, approach, and jaw geometry is added later by the Operator
    point-cloud renderer.  This makes held or occluded scene geometry
    inspectable without changing physics.
    """

    holder = _unwrap_libero_sim(env)
    context = holder.sim._render_context_offscreen
    camera = context.cam
    saved = {
        "type": int(camera.type),
        "fixedcamid": int(camera.fixedcamid),
        "lookat": np.asarray(camera.lookat, dtype=np.float64).copy(),
        "distance": float(camera.distance),
        "azimuth": float(camera.azimuth),
        "elevation": float(camera.elevation),
    }
    global_vis = holder.sim.model.vis.global_
    saved_ipd = float(global_vis.ipd)
    robot_geom_ids = (
        _libero_robot_visual_geom_ids(holder) if bool(hide_robot) else []
    )
    saved_robot_geom_groups = (
        np.asarray(holder.sim.model.geom_group[robot_geom_ids], dtype=np.int32).copy()
        if robot_geom_ids
        else np.empty((0,), dtype=np.int32)
    )
    hidden_geom_group = 2
    saved_hidden_group_visibility = int(context.vopt.geomgroup[hidden_geom_group])
    frames: list[dict[str, Any]] = []
    try:
        global_vis.ipd = 0.0
        if robot_geom_ids:
            holder.sim.model.geom_group[robot_geom_ids] = hidden_geom_group
            context.vopt.geomgroup[hidden_geom_group] = 0
        if lookat_xyz_m is None:
            lookat = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            lookat = np.asarray(lookat_xyz_m, dtype=np.float64)
            if lookat.shape != (3,) or not np.isfinite(lookat).all():
                raise ValueError("lookat_xyz_m must be a finite length-3 vector")
        distance = float(distance_m)
        if not np.isfinite(distance) or not 0.4 <= distance <= 3.0:
            raise ValueError("distance_m must be finite and between 0.4 and 3.0")
        camera.type = 0
        camera.fixedcamid = -1
        camera.lookat[:] = lookat
        camera.distance = distance
        for index, (azimuth, elevation) in enumerate(_LIBERO_MULTIVIEW_POSES):
            camera.azimuth = float(azimuth)
            camera.elevation = float(elevation)
            context.render(int(width), int(height), camera_id=-1)
            rgb_raw, depth_raw = context.read_pixels(int(width), int(height), depth=True)
            rgb = np.flipud(np.asarray(rgb_raw))[..., :3].astype(np.uint8)
            depth_buffer = np.flipud(np.asarray(depth_raw, dtype=np.float64))
            left, right_eye = context.scn.camera[0], context.scn.camera[1]
            near, far = float(left.frustum_near), float(left.frustum_far)
            depth_m = near / (1.0 - depth_buffer * (1.0 - near / far))
            position = (
                np.asarray(left.pos, dtype=np.float64)
                + np.asarray(right_eye.pos, dtype=np.float64)
            ) / 2.0
            forward = (
                np.asarray(left.forward, dtype=np.float64)
                + np.asarray(right_eye.forward, dtype=np.float64)
            ) / 2.0
            forward /= np.linalg.norm(forward)
            up = (
                np.asarray(left.up, dtype=np.float64)
                + np.asarray(right_eye.up, dtype=np.float64)
            ) / 2.0
            up /= np.linalg.norm(up)
            image_right = np.cross(forward, up)
            image_right /= np.linalg.norm(image_right)
            rotation = np.column_stack([image_right, -up, forward])
            focal = (height / 2.0) * near / max(1e-12, float(left.frustum_top))
            rgb_b64, _, _ = _encode_pixels_to_base64(rgb)
            depth_b64, _, _ = _encode_pixels_to_base64(depth_m, mode="depth")
            frames.append(
                {
                    "camera_id": f"virtual-{index:02d}",
                    "frame_id": f"virtual-{index:02d}",
                    "width": int(width),
                    "height": int(height),
                    "rgb_base64": rgb_b64,
                    "depth_base64": depth_b64,
                    "intrinsics": {
                        "fx": float(focal),
                        "fy": float(focal),
                        "cx": float(width) / 2.0,
                        "cy": float(height) / 2.0,
                        "width": int(width),
                        "height": int(height),
                        "scale": 1000.0,
                    },
                    "extrinsics": {
                        "pos": position.tolist(),
                        "mat": rotation.reshape(-1).tolist(),
                        "matrix_layout": "row_major",
                        "frame_transform": "camera_to_world",
                        "camera_frame": "opencv",
                    },
                    "virtual_camera": {
                        "azimuth_deg": float(azimuth),
                        "elevation_deg": float(elevation),
                        "distance_m": distance,
                        "lookat_xyz_m": lookat.tolist(),
                    },
                }
            )
    finally:
        if robot_geom_ids:
            holder.sim.model.geom_group[robot_geom_ids] = saved_robot_geom_groups
        context.vopt.geomgroup[hidden_geom_group] = saved_hidden_group_visibility
        global_vis.ipd = saved_ipd
        camera.type = saved["type"]
        camera.fixedcamid = saved["fixedcamid"]
        camera.lookat[:] = saved["lookat"]
        camera.distance = saved["distance"]
        camera.azimuth = saved["azimuth"]
        camera.elevation = saved["elevation"]
    return {
        "kind": "multiview_render",
        "success": True,
        "physics_stepped": False,
        "robot_visuals_hidden": bool(robot_geom_ids),
        "hidden_robot_visual_geom_count": len(robot_geom_ids),
        "camera_count": len(frames),
        "cameras": frames,
    }


# ══════════════════════════════════════════════════════════════════════
# Starlette app
# ══════════════════════════════════════════════════════════════════════

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

# Simulator calls into MuJoCo/EGL are heavy and blocking. Running them directly
# in async handlers would block uvicorn's event loop and make /health look dead.
# More importantly, an EGL context may be corrupted even when a mutex serialises
# calls if successive calls run on different OS threads. Every simulator call is
# therefore submitted to one stable per-process execution thread. BEHAVIOR uses
# the process main thread; all other benches use the single-worker executor
# installed by Starlette startup.
_egl_create_lock = threading.Lock()


def _make_env_locked(*args, **kwargs):
    with _egl_create_lock:
        return _make_env(*args, **kwargs)


async def health(request):
    return _json_response({
        "ok": True,
        "bench": request.app.state.bench,
        "python": sys.version.split()[0],
    })


async def list_envs(request):
    from sim.env_registry import list_envs as _le, search as _se

    bench_filter = request.query_params.get("type", "")
    q = request.query_params.get("q", "")
    if q:
        specs = _se(q)
        if bench_filter:
            specs = [s for s in specs if s.env_type == bench_filter]
    else:
        specs = _le(env_type=bench_filter if bench_filter else None)

    return _json_response({
        "envs": [{"id": s.id, "type": s.env_type, "description": s.task_description} for s in specs],
        "count": len(specs),
    })


async def create_env(request):
    import gymnasium as gym
    import traceback as _tb

    body = await request.json()
    eid = body["env_id"]
    try:
        make_kwargs = {
            "task": body.get("task", ""),
            "seed": body.get("seed", 0),
            "render_mode": body.get("render_mode", "rgb_array"),
            "image_width": body.get("image_width"),
            "image_height": body.get("image_height"),
            "include_objects": body.get("include_objects", False),
            "robot": body.get("robot"),
        }
        # Offload simulator builds so /health stays responsive; serialise
        # context creation via the lock. BEHAVIOR's signal registration is
        # handled by _make_env_locked's supervised-worker compatibility scope.
        env = await _run_sim_call(_make_env_locked, eid, **make_kwargs)
    except Exception as exc:
        # Surface the real traceback in the JSON body so it propagates back
        # through the proxy instead of a bare HTTP 500 with no detail.
        _tb.print_exc()
        return _json_response(
            {"error": f"create_env failed: {exc}", "traceback": _tb.format_exc()[-2000:]},
            status_code=500,
        )
    env._env_id = eid
    h = str(uuid.uuid4())[:12]
    _envs[h] = env
    _env_lock(h)

    adim, alo, ahi, be, adesc = None, None, None, "", ""
    try:
        s = env.action_space
        adim = int(s.shape[0]) if hasattr(s, "shape") and s.shape else None
        if hasattr(s, "low"):
            alo = [float(x) for x in s.low[:min(7, len(s.low))]]
            ahi = [float(x) for x in s.high[:min(7, len(s.high))]]
    except Exception:
        pass
    be = getattr(env, "_backend", "")
    if be == "metaworld":
        adesc = "xyz+gripper delta (4D)"
    elif be == "libero":
        adesc = "xyz+rpy+gripper (OSC_POSE 7D)"
    elif be == "maniskill":
        adesc = "xyz+rot+gripper delta (7D)" if adim and adim <= 7 else f"{adim}D"
    elif be == "robocasa":
        adesc = (
            "fixed Panda: xyz+rpy+gripper (7D)"
            if adim == 7
            else "PandaOmron: xyz+rpy+gripper+base_xyz+torso+base_mode (12D)"
        )
    elif be == "behavior":
        adesc = "R1Pro flattened continuous action (OmniGibson controller order)"
    elif be == "dummy":
        adesc = "dict {action_type,code}"
    hints = {
        "metaworld": "~6mm/step at action=1.0, use 5-10 steps for visible motion",
        "libero": "~50mm controller output at action=1.0 (OSC_POSE), use 3-5 steps for visible motion",
        "maniskill": "use 3-5 steps for visible motion",
        "robocasa": (
            "fixed Panda: arm 0:6, gripper 6"
            if adim == 7
            else "PandaOmron: arm 0:6, gripper 6, base 7:10, torso 10, mode 11"
        ),
        "behavior": "Use the environment action bounds; success comes from BEHAVIOR's native BDDL checker",
    }.get(be, "")

    control_spec: dict = {}
    try:
        direct_env = getattr(env, "_env", env)
        candidate = getattr(direct_env, "openeta_control_spec", {})
        if isinstance(candidate, dict):
            control_spec = candidate
    except Exception:
        control_spec = {}

    return _json_response({
        "handle": h, "env_id": eid, "action_dim": adim, "action_desc": adesc,
        "action_low": alo, "action_high": ahi, "backend": be, "action_hint": hints,
        "robot": body.get("robot"), "control_spec": control_spec,
    })


async def close_env(request):
    h = request.path_params.get("handle", "")
    env = _envs.get(h)
    is_behavior = request.app.state.bench == "behavior"
    closed = False
    if env is not None:
        # Sending og.shutdown() from inside this HTTP handler closes Kit's
        # process resources before uvicorn can flush the response. The manager
        # treats BEHAVIOR workers as single-use and terminates the process
        # immediately after receiving this acknowledgement.
        closed = await _run_sim_call(
            _close_env_locked,
            h,
            env,
            close_simulator=not is_behavior,
        )
    return _json_response({
        "ok": closed,
        "worker_retire_required": bool(closed and is_behavior),
    })


def _safe_json_body(body: bytes) -> dict:
    """Parse JSON body, returning {} for empty / unparseable input."""
    if not body or not body.strip():
        return {}
    try:
        return json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return {}


async def reset_env(request):
    import traceback as _tb

    h = request.path_params.get("handle", "")
    env = _envs.get(h)
    if env is None:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    body = _safe_json_body(await request.body()) if request.method == "POST" else {}
    try:
        result = await _run_sim_call(
            _run_env_locked,
            h,
            _reset_with_image,
            env,
            seed=body.get("seed"),
            handle=h,
            _expected_env=env,
        )
        _env_errors.pop(h, None)
        return _json_response(result)
    except _EnvUnavailableError:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    except Exception as exc:
        _tb.print_exc()
        err_msg = f"Reset failed: {exc}"
        _env_errors[h] = err_msg
        return _json_response({"error": err_msg, "handle": h, "fatal": True}, 500)


async def step_env(request):
    import gymnasium as gym
    import traceback as _tb

    h = request.path_params.get("handle", "")
    env = _envs.get(h)
    if env is None:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)

    body = _safe_json_body(await request.body()) if request.method == "POST" else {}
    act = body.get("action")
    num_steps = max(1, int(body.get("num_steps", 1)))
    render = bool(body.get("render", True))
    include_cameras = bool(body.get("include_cameras", render))

    try:
        if isinstance(act, (list, tuple)):
            act = np.array(act, dtype=np.float32)

        def _run_steps():
            step_act = act
            if step_act is None:
                space = env.action_space
                step_act = space.sample() if hasattr(space, "sample") else np.zeros(7)

            aspace = env.action_space
            if isinstance(aspace, gym.spaces.Box) and step_act.ndim == 1:
                exp = int(aspace.shape[0])
                if exp != step_act.shape[0]:
                    if step_act.shape[0] > exp:
                        step_act = step_act[:exp]
                    else:
                        step_act = np.pad(step_act, (0, exp - step_act.shape[0]))
                if hasattr(aspace, "low") and aspace.low is not None:
                    step_act = np.clip(
                        step_act,
                        aspace.low[:len(step_act)],
                        aspace.high[:len(step_act)],
                    )

            res = None
            steps_executed = 0
            try:
                for _ in range(num_steps):
                    res = _step_with_image(
                        env,
                        step_act,
                        handle=h,
                        render=render,
                        include_cameras=include_cameras,
                    )
                    steps_executed += 1
                    if res.get("terminated") or res.get("truncated"):
                        break
                if res is None:
                    res = _observe_with_image(env, handle=h)  # fallback: try observe
                if isinstance(res, dict):
                    res["steps_executed"] = steps_executed
                return res
            finally:
                _note_env_step_activity(h)

        # Offload the blocking sim/render loop so the event loop stays free.
        _note_env_step_activity(h)
        result = await _run_sim_call(
            _run_env_locked,
            h,
            _run_steps,
            _expected_env=env,
        )
        _env_errors.pop(h, None)  # clear any previous error on success
        return _json_response(result or {})
    except _EnvUnavailableError:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    except Exception as exc:
        _tb.print_exc()
        err_msg = f"Step failed: {exc}"
        _env_errors[h] = err_msg
        return _json_response({"error": err_msg, "handle": h, "fatal": False}, 500)


async def observe_env(request):
    h = request.path_params.get("handle", "")
    env = _envs.get(h)
    if env is None:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    try:
        result = await _run_sim_call(
            _run_env_locked,
            h,
            _observe_with_image,
            env,
            handle=h,
            _expected_env=env,
        )
        return _json_response(result)
    except _EnvUnavailableError:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)


async def render_env(request):
    h = request.path_params.get("handle", "")
    env = _envs.get(h)
    if env is None:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    try:
        result = await _run_sim_call(
            _run_env_locked,
            h,
            _observe_with_image,
            env,
            handle=h,
            _expected_env=env,
        )
        return _json_response(result)
    except _EnvUnavailableError:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)


async def render_multiview_env(request):
    h = request.path_params.get("handle", "")
    env = _envs.get(h)
    if env is None:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    body = _safe_json_body(await request.body()) if request.method == "POST" else {}
    width = max(64, min(1024, int(body.get("width", 256))))
    height = max(64, min(1024, int(body.get("height", 256))))
    hide_robot = bool(body.get("hide_robot", False))
    lookat_xyz_m = body.get("lookat_xyz_m")
    distance_m = body.get("distance_m", 1.3)
    try:
        result = await _run_sim_call(
            _run_env_locked,
            h,
            _render_libero_multiview,
            env,
            width=width,
            height=height,
            hide_robot=hide_robot,
            lookat_xyz_m=lookat_xyz_m,
            distance_m=distance_m,
            _expected_env=env,
        )
        return _json_response(result)
    except _EnvUnavailableError:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    except Exception as exc:
        return _json_response(
            {"error": f"Multiview render failed: {exc}", "handle": h}, 500
        )


async def check_task(request):
    """Return the backend-native task checker result and host diagnostics.

    The embodied gateway exposes only the boolean to the Operator.  Native
    predicate details are retained in host-side episode artifacts so a
    failed run can be diagnosed without guessing from pixels.
    """

    h = request.path_params.get("handle", "")
    env = _envs.get(h)
    if env is None:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)

    def _native_check() -> dict:
        checker = getattr(env, "check_success", None)
        if not callable(checker):
            return {"available": False, "success": None}
        value = checker()
        result = {"available": True, "success": bool(value)}
        try:
            result["diagnostics"] = _libero_task_diagnostics(env)
        except Exception as exc:
            result["diagnostics"] = {
                "available": False,
                "reason": "diagnostic_exception",
                "error": str(exc),
            }
        return result

    try:
        return _json_response(await _run_sim_call(
            _run_env_locked,
            h,
            _native_check,
            _expected_env=env,
        ))
    except _EnvUnavailableError:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    except Exception as exc:
        return _json_response(
            {"available": True, "success": False, "error": f"Task checker failed: {exc}"},
            500,
        )


async def render_all_envs(request):
    """Best-effort background render on the worker's simulator thread."""
    handles: list[str] = []
    if request.method == "POST":
        try:
            body = await request.json()
            handles = body.get("handles", [])
        except Exception:
            pass
    if not handles:
        handles = list(_envs.keys())

    results: dict[str, dict] = {}

    def _render_one(h: str, env) -> dict:
        try:
            # _observe_with_image() injects a render frame into the raw
            # UnifiedEnv dict in-place (via _inject_render_frame), so the
            # cache is already refreshed.  Do NOT overwrite _last_obs
            # with the MCP-format return value — that would poison the
            # cache and cause the next observe/render call to crash with
            # a TypeError when it tries to index a list like a dict.
            rendered = _try_run_env_locked(
                h,
                _observe_with_image,
                env,
                handle=h,
                _expected_env=env,
            )
            if rendered is None:
                return {"skipped": True, "reason": "env_busy"}
            return rendered
        except Exception:
            return {"error": "render failed"}

    for handle in handles:
        env = _envs.get(handle)
        if env is None:
            results[handle] = {"error": f"Unknown handle: {handle}"}
            continue
        # Do this non-blocking probe before submitting to the single simulator
        # thread. Otherwise a background render would queue behind an active
        # reset/step and render later instead of preserving the busy-skip API.
        if _env_render_is_busy(handle):
            results[handle] = {"skipped": True, "reason": "env_busy"}
            continue
        results[handle] = await _run_sim_call(_render_one, handle, env)

    return _json_response({"rendered": list(results.keys()), "by_handle": results})


app = Starlette(
    routes=[
        Route("/health", health, methods=["GET"]),
        Route("/envs", list_envs, methods=["GET"]),
        Route("/env", create_env, methods=["POST"]),
        Route("/env/{handle}", close_env, methods=["DELETE"]),
        Route("/env/{handle}/reset", reset_env, methods=["POST"]),
        Route("/env/{handle}/step", step_env, methods=["POST"]),
        Route("/env/{handle}/observe", observe_env, methods=["POST"]),
        Route("/env/{handle}/render", render_env, methods=["POST"]),
        Route("/env/{handle}/render_multiview", render_multiview_env, methods=["POST"]),
        Route("/env/{handle}/check_task", check_task, methods=["POST"]),
        Route("/render_all", render_all_envs, methods=["POST"]),
    ],
    lifespan=_sim_executor_lifespan,
)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OpenETA Bench Worker")
    p.add_argument("--bench", required=True, help="Bench name (libero, metaworld, maniskill, etc.)")
    p.add_argument("--port", type=int, default=0, help="Port (0 = random)")
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    import socket

    # Bind the listening socket up front and hand the *same* socket to uvicorn.
    #
    # The old code bound port 0, read the port, then closed the socket and only
    # rebound (via uvicorn) ~15-30s later after _init_bench().  That left a wide
    # window where (a) another concurrently-starting worker could be handed the
    # same "free" port by the OS, and (b) the parent could connect before
    # uvicorn had bound → ConnectionRefused / HTTP 500.  Both surface only under
    # concurrent spawns.  Keeping one bound, listening socket the whole time
    # reserves the port continuously and lets the OS queue early connections.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))  # args.port == 0 → OS picks a free port
    sock.listen(128)
    port = sock.getsockname()[1]

    # Import the HTTP server before benchmark activation.  Some benchmark
    # packages (notably LIBERO variants) rewrite ``sys.path`` while importing
    # their registration modules; when uvicorn was imported afterwards this
    # could make the worker lose access to its own venv site-packages and die
    # with ``ModuleNotFoundError: uvicorn`` after having already advertised a
    # port.  Keeping the module loaded up front makes startup independent of
    # those benchmark-specific path changes.
    import uvicorn

    # Register envs for this bench (suppress bench info output during init)
    import io as _io
    _old_stdout = sys.stdout
    sys.stdout = _io.StringIO()
    _init_bench(args.bench)
    sys.stdout = _old_stdout

    # Store bench name on app state
    app.state.bench = args.bench

    # Write port to stdout (single clean line) for parent process discovery.
    # Safe to announce now: the socket is already bound+listening, so the parent
    # can connect (and the OS queues the connection until uvicorn accepts).
    print(port, flush=True)

    # Start server on the pre-bound socket (fd is inherited by uvicorn).
    config = uvicorn.Config(app, fd=sock.fileno(), log_level="warning")
    server = uvicorn.Server(config)
    if args.bench == "behavior":
        # Isaac Kit / OmniGibson requires the process main thread (signal
        # handlers, render loop, and teardown). Keep HTTP responsive in a
        # server thread and execute every simulator operation here via queue.
        executor = _MainThreadExecutor()
        app.state.sim_executor = executor
        server_thread = threading.Thread(target=server.run, daemon=True)
        server_thread.start()
        try:
            while server_thread.is_alive():
                executor.run_once(timeout=0.1)
        finally:
            server.should_exit = True
            server_thread.join(timeout=5)
            executor.shutdown(wait=True, cancel_futures=True)
    else:
        server.run()
