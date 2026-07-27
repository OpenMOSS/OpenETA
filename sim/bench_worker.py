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
"""

from __future__ import annotations

import argparse, asyncio, base64, io, json, math, os, queue, sys, threading, uuid, warnings
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import contextlib

warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
os.environ.setdefault("LIBERO_DIR", "/tmp/LIBERO")
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

# ── Per-handle observation locks ────────────────────────────────────────
# ``_last_obs[handle]`` holds the MUTABLE raw UnifiedEnv obs dict.  Two
# concurrent code paths touch the SAME dict object for a given handle:
#
#   1. ``step_env`` (move_to's closed-loop stepping, on the anyio threadpool)
#      writes ``_last_obs[handle] = obs`` then iterates ``obs["cameras"]``
#      inside ``EnvObservation.from_dict``.
#   2. ``render_all_envs`` (the dashboard's background SSE refresh, on its
#      OWN 8-thread pool) calls ``_observe_with_image`` →
#      ``_inject_render_frame`` which MUTATES ``obs["cameras"]`` in place.
#
# When (2) inserts a camera key while (1) is iterating, Python raises
# ``RuntimeError: dictionary changed size during iteration`` — and even when
# it doesn't crash, (2)'s freshly injected frame bleeds into the observation
# (1) is serialising, producing the "handle matches but scene doesn't"
# garbling.  ``_gl_lock`` only guards the GPU call, not this shared dict, so
# we need a separate per-handle lock that spans "mutate/read obs + serialise".
_obs_locks: dict[str, threading.Lock] = {}
_obs_locks_guard = threading.Lock()

# Handles whose episode has already terminated/truncated.  robosuite raises
# ``ValueError("executing action in terminated episode")`` if you step a
# finished episode, so once a step reports done we refuse further stepping and
# return the last observation instead (the client must reset_env to continue).
# Guarded implicitly by the per-handle obs lock (step/reset/close serialise).
_done_handles: set[str] = set()


def _obs_lock_for(handle: str) -> threading.Lock:
    """Return (creating if needed) the per-handle observation lock."""
    with _obs_locks_guard:
        lock = _obs_locks.get(handle)
        if lock is None:
            lock = threading.Lock()
            _obs_locks[handle] = lock
        return lock

# ── Process-wide GL serialisation lock ──────────────────────────────────
# One worker process hosts MULTIPLE envs (see worker_mgr's pool) that all
# share a single EGL display / GL context.  ``eglMakeCurrent`` is thread-
# global state, so two threads touching the GPU at once corrupt each other's
# frames — the "garbled render / weird scene state under concurrency" bug.
#
# The subtlety: with robosuite/LIBERO ``use_camera_obs=True``, ``env.step()``
# ITSELF renders the camera images (``agentview_image`` etc.) inside the
# step — not just ``env.render()``.  So the lock must serialise step, reset
# AND render, not render alone.  Physics is per-env independent; only the GL
# access needs mutual exclusion, but since rendering is inseparable from the
# step here, we serialise the whole GPU-touching call.
#
# RLock (re-entrant): a locked ``step`` re-enters the lock when it calls
# ``_inject_render_frame`` on the same thread.
_gl_lock = threading.RLock()


def _patch_robosuite_render_context() -> None:
    """Force each render to bind its OWN GL context first (robosuite bug fix).

    robosuite's ``MjRenderContext.render()`` / ``read_pixels()`` issue GL calls
    against *whatever* EGL context is currently bound on the calling thread, but
    they never call ``self.gl_ctx.make_current()`` — they assume their context
    is still current.  That holds for a single env, but a worker process hosts
    MULTIPLE libero envs (see worker_mgr's pool), each with its own EGL context
    (2 per env: camera-obs + render), all sharing one EGL display.  ``eglMake-
    Current`` is *per-thread* state, and step/render run on different thread
    pools, so a render for env A frequently executes on a thread where env B's
    context (or none) is bound.  ``mjr_render`` then resolves env A's uploaded
    mesh/texture ids against env B's GL context → A's gripper is drawn with B's
    geometry: the "mesh swapped / scene weird after gripper_open" corruption.
    ``_gl_lock`` serialises the GPU call but does NOT fix which context is
    bound, so the frame recovers but shows the wrong meshes.

    We wrap ``render`` to ``make_current()`` first.  Idempotent — guarded so a
    re-import/re-activate can't double-patch.
    """
    try:
        import robosuite.utils.binding_utils as bu
    except Exception:
        return
    if getattr(bu.MjRenderContext, "_openeta_ctx_patched", False):
        return
    _orig_render = bu.MjRenderContext.render

    def _render_bound(self, *args, **kwargs):
        # Rebind THIS context to the current thread before drawing so the
        # render draws against the context that owns this env's GPU meshes.
        try:
            self.gl_ctx.make_current()
        except Exception:
            pass
        return _orig_render(self, *args, **kwargs)

    bu.MjRenderContext.render = _render_bound
    bu.MjRenderContext._openeta_ctx_patched = True


class _MainThreadExecutor:
    """Marshal simulator calls from uvicorn's thread onto the process main thread."""

    def __init__(self) -> None:
        self._queue: queue.Queue[tuple[Future, object, tuple, dict]] = queue.Queue()

    def submit(self, fn, *args, **kwargs) -> Future:
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


async def _run_sim_call(fn, *args, **kwargs):
    """Run on BEHAVIOR's main-thread executor, otherwise Starlette's pool."""
    executor = getattr(app.state, "sim_executor", None)
    if executor is not None:
        return await asyncio.wrap_future(executor.submit(fn, *args, **kwargs))
    return await run_in_threadpool(fn, *args, **kwargs)


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
    # libero renders via robosuite; fix its missing per-render context bind.
    if bench == "libero":
        _patch_robosuite_render_context()


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
        with _gl_lock:  # serialise GPU access across the worker's envs
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
        with _gl_lock:  # serialise GPU access across the worker's envs
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


def _terminated_step_result(handle: str) -> dict:
    """Build a StepResult for a handle whose episode already finished.

    Returns the last cached observation (no new step) with ``terminated``
    set, so a client that keeps calling step after done gets a clean signal
    instead of an HTTP 500 from robosuite's "terminated episode" guard.
    Caller must hold the per-handle obs lock.
    """
    from adapter.protocol import EnvObservation, StepResult

    obs = _last_obs.get(handle, {})
    try:
        env_obs = EnvObservation.from_dict(obs) if obs else EnvObservation.from_dict({})
    except Exception:
        env_obs = EnvObservation.from_dict({})
    return StepResult(
        observation=env_obs, reward=0.0, terminated=True, truncated=False,
        info={"note": "episode already terminated — call reset_env to continue"},
    ).to_mcp_dict()


def _step_with_image(env, act, handle: str = "", render: bool = True) -> dict:
    """Execute one env step, return StepResult as MCP dict.

    Render failure is non-fatal — we still return the step result without
    a camera frame so the user can decide to retry or reset.

    When ``render`` is ``False`` the (GPU-bound, ~130 ms) camera render is
    skipped entirely.  This is used by ``move_to``'s closed-loop stepping,
    which only reads the EE pose / joint positions from the result and never
    the image — the dashboard's own background ``/render_all`` refresh keeps
    the live view updated independently.  Skipping the render here makes
    each control step ~5-8x faster.
    """
    from adapter.protocol import EnvObservation, StepResult

    # LIBERO/robosuite step renders camera obs (agentview_image etc.) INSIDE
    # env.step when use_camera_obs=True, so the step touches the GL context.
    # Hold the GL lock across step AND its render injection so another env's
    # step can't interleave and corrupt the shared context mid-frame.
    # Hold the per-handle obs lock across step + cache write + serialise so a
    # concurrent render_all_envs can't mutate obs["cameras"] mid-iteration.
    # Lock order is always obs-lock → gl-lock (see _observe_with_image).
    _obs_ctx = _obs_lock_for(handle) if handle else contextlib.nullcontext()
    with _obs_ctx:
        # (#2) Episode already finished — robosuite would raise
        # ValueError("executing action in terminated episode").  Return the
        # last observation with terminated=True so the client stops stepping.
        if handle and handle in _done_handles:
            return _terminated_step_result(handle)
        # (#3) close_env may have torn this env down between the handler's
        # _envs.get() and our acquiring the lock — re-check under the lock so
        # we don't step a closed env (AttributeError: no attribute 'env').
        if handle and handle not in _envs:
            return {"error": "Env was closed — call reset_env to start again",
                    "handle": handle, "terminated": True}
        with _gl_lock:
            obs, rew, term, trunc, info = env.step(act)
            if render:
                try:
                    _inject_render_frame(env, obs)
                except Exception:
                    pass  # render is best-effort; don't lose the step result
        if handle:
            _last_obs[handle] = obs
            if term or trunc:
                _done_handles.add(handle)
        env_obs = EnvObservation.from_dict(obs)
    # Sanitise info: drop non-serialisable values
    safe_info: dict = {}
    if isinstance(info, dict):
        for k, v in info.items():
            try:
                json.dumps({k: v})
                safe_info[k] = v
            except (TypeError, ValueError):
                safe_info[k] = str(v)
    else:
        safe_info = {"raw_info": str(info)}
    return StepResult(
        observation=env_obs,
        reward=float(rew),
        terminated=bool(term),
        truncated=bool(trunc),
        info=safe_info,
    ).to_mcp_dict()


def _reset_with_image(env, seed=None, handle: str = "") -> dict:
    """Reset env, return EnvObservation as MCP dict."""
    from adapter.protocol import EnvObservation

    # reset likewise renders camera obs internally — serialise on the GL lock.
    # Guard the shared obs dict with the per-handle lock (obs-lock → gl-lock).
    _obs_ctx = _obs_lock_for(handle) if handle else contextlib.nullcontext()
    with _obs_ctx:
        with _gl_lock:
            obs, _ = env.reset(seed=seed)
            _inject_render_frame(env, obs)
        if handle:
            _last_obs[handle] = obs
            _done_handles.discard(handle)  # fresh episode — stepping allowed again
        return EnvObservation.from_dict(obs).to_mcp_dict()


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

    # This MUTATES obs["cameras"] in place and then serialises it.  A
    # concurrent step_env on the same handle iterates the same dict, so we
    # serialise both under the per-handle obs lock (obs-lock → gl-lock, the
    # same order step_env/reset use, so no deadlock).
    _obs_ctx = _obs_lock_for(handle) if handle else contextlib.nullcontext()
    with _obs_ctx:
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


# ══════════════════════════════════════════════════════════════════════
# Starlette app
# ══════════════════════════════════════════════════════════════════════

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.concurrency import run_in_threadpool


def _sanitize_floats(obj, _path: str = "", _bad: list | None = None):
    """Recursively replace non-finite floats (NaN/+-inf) with None.

    Starlette's JSONResponse serialises with ``allow_nan=False``, so a single
    NaN/inf anywhere in an observation (e.g. an EE pose or depth pixel that
    went non-finite after the physics diverged during a long move) makes the
    whole response raise ``ValueError: Out of range float values are not JSON
    compliant`` — surfacing as a bare HTTP 500 and a garbled dashboard render.

    We null out the offending values so the response still serialises, and
    collect their field paths in ``_bad`` so the caller can log *where* the
    pollution came from (to chase the physics root cause separately).
    """
    import math
    if isinstance(obj, float):
        if not math.isfinite(obj):
            if _bad is not None:
                _bad.append(_path or "<root>")
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v, f"{_path}.{k}" if _path else str(k), _bad)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_floats(v, f"{_path}[{i}]", _bad) for i, v in enumerate(obj)]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSONResponse that never dies on NaN/inf.

    First tries the normal (strict) render for speed; only on the NaN/inf
    ValueError does it sanitise + retry, logging the polluted field paths so
    the physics divergence can be traced.
    """

    def render(self, content) -> bytes:
        try:
            return super().render(content)
        except ValueError:
            bad: list[str] = []
            cleaned = _sanitize_floats(content, _bad=bad)
            if bad:
                # goes to worker stderr → OPENETA_WORKER_LOG_DIR
                print(f"[sanitize] non-finite floats nulled at: {bad[:20]}"
                      + (f" (+{len(bad) - 20} more)" if len(bad) > 20 else ""),
                      file=sys.stderr, flush=True)
            return super().render(cleaned)

# Env creation (and stepping) call into MuJoCo/EGL, which is heavy (~3s) and
# blocking.  If we ran it directly inside the async handler it would block the
# single uvicorn event-loop thread, so /health would not answer until it
# returned.  The manager polls /health with a 2s timeout to decide whether a
# worker is alive; a blocked loop looks dead and the manager terminates the
# worker mid-create — the env's later step then hits a killed process
# ("connection refused").  Offloading to a threadpool keeps the loop free so
# /health stays responsive.  EGL context creation is not reliably concurrent.
#
# CORRECTION: a previous version claimed "stepping different envs is fine in
# parallel — each has its own context".  That is FALSE for the MuJoCo EGL
# backend: all envs in a worker share one EGL display, and eglMakeCurrent is
# thread-global, so concurrent create/step/render corrupt each other's frames.
# We therefore serialise creation on the SAME process-wide GL lock used by
# step/reset/render (see _gl_lock), so a create can't race an in-flight render.
def _make_env_locked(*args, **kwargs):
    with _gl_lock:
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
        "libero": "~9mm/step at action=1.0, use 3-5 steps for visible motion",
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
    # Serialise close against any in-flight step/observe on this handle so we
    # don't yank the env / cache out from under an active serialisation.
    with _obs_lock_for(h):
        env = _envs.pop(h, None)
        _last_obs.pop(h, None)
        _done_handles.discard(h)
        if env:
            if request.app.state.bench == "behavior":
                # Sending og.shutdown() from inside this HTTP handler closes Kit's
                # process resources before uvicorn can flush the response. The
                # manager treats BEHAVIOR workers as single-use and terminates the
                # process immediately after receiving this acknowledgement.
                result = {"ok": True, "worker_retire_required": True}
            else:
                try:
                    await _run_sim_call(env.close)
                except Exception:
                    pass
                result = {"ok": True, "worker_retire_required": False}
        else:
            result = {"ok": False, "worker_retire_required": False}
    # Drop the now-unused lock (a late concurrent caller just gets a fresh one).
    with _obs_locks_guard:
        _obs_locks.pop(h, None)
    return _json_response(result)


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
            _reset_with_image, env, seed=body.get("seed"), handle=h)
        _env_errors.pop(h, None)
        return _json_response(result)
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

    try:
        if act is None:
            s = env.action_space
            act = s.sample() if hasattr(s, "sample") else np.zeros(7)
        if isinstance(act, (list, tuple)):
            act = np.array(act, dtype=np.float32)

        aspace = env.action_space
        if isinstance(aspace, gym.spaces.Box) and act.ndim == 1:
            exp = int(aspace.shape[0])
            if exp != act.shape[0]:
                if act.shape[0] > exp:
                    act = act[:exp]
                else:
                    act = np.pad(act, (0, exp - act.shape[0]))
            if hasattr(aspace, "low") and aspace.low is not None:
                act = np.clip(act, aspace.low[:len(act)], aspace.high[:len(act)])

        def _run_steps():
            res = None
            for _ in range(num_steps):
                res = _step_with_image(env, act, handle=h, render=render)
                if res.get("terminated") or res.get("truncated"):
                    break
            if res is None:
                res = _observe_with_image(env, handle=h)  # fallback: try observe
            return res

        # Offload the blocking sim/render loop so the event loop stays free.
        result = await _run_sim_call(_run_steps)
        _env_errors.pop(h, None)  # clear any previous error on success
        return _json_response(result or {})
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
    return _json_response(await _run_sim_call(_observe_with_image, env, handle=h))


async def render_env(request):
    h = request.path_params.get("handle", "")
    env = _envs.get(h)
    if env is None:
        return _json_response({"error": f"Unknown handle: {h}"}, 400)
    return _json_response(await _run_sim_call(_observe_with_image, env, handle=h))


async def render_all_envs(request):
    """Render all (or a subset of) envs in parallel using a thread pool."""
    handles: list[str] = []
    if request.method == "POST":
        try:
            body = await request.json()
            handles = body.get("handles", [])
        except Exception:
            pass
    if not handles:
        handles = list(_envs.keys())

    if getattr(request.app.state, "sim_executor", None) is not None:
        # Isaac Kit has strict main-thread affinity. Serialize BEHAVIOR renders
        # through the same process-main-thread executor as create/reset/step;
        # a ThreadPool here can self-close or corrupt the Kit render loop.
        results: dict[str, dict] = {}
        for handle in handles:
            env = _envs.get(handle)
            if env is None:
                results[handle] = {"error": f"Unknown handle: {handle}"}
                continue
            try:
                results[handle] = await _run_sim_call(_observe_with_image, env, handle=handle)
            except Exception:
                results[handle] = {"error": "render failed"}
        return _json_response({"rendered": list(results), "by_handle": results})

    # ── parallel render ──────────────────────────────────────────
    results: dict[str, dict] = {}

    def _render_one(h: str) -> tuple[str, dict]:
        env = _envs.get(h)
        if env is None:
            return h, {"error": f"Unknown handle: {h}"}
        try:
            # _observe_with_image() injects a render frame into the raw
            # UnifiedEnv dict in-place (via _inject_render_frame), so the
            # cache is already refreshed.  Do NOT overwrite _last_obs
            # with the MCP-format return value — that would poison the
            # cache and cause the next observe/render call to crash with
            # a TypeError when it tries to index a list like a dict.
            return h, _observe_with_image(env, handle=h)
        except Exception:
            return h, {"error": f"render failed"}

    max_workers = min(len(handles), 8)  # cap threads; EGL isn't fully thread-safe
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_render_one, h): h for h in handles}
        for future in as_completed(futures):
            h, data = future.result()
            results[h] = data

    return _json_response({"rendered": list(results.keys()), "by_handle": results})


app = Starlette(routes=[
    Route("/health", health, methods=["GET"]),
    Route("/envs", list_envs, methods=["GET"]),
    Route("/env", create_env, methods=["POST"]),
    Route("/env/{handle}", close_env, methods=["DELETE"]),
    Route("/env/{handle}/reset", reset_env, methods=["POST"]),
    Route("/env/{handle}/step", step_env, methods=["POST"]),
    Route("/env/{handle}/observe", observe_env, methods=["POST"]),
    Route("/env/{handle}/render", render_env, methods=["POST"]),
    Route("/render_all", render_all_envs, methods=["POST"]),
])


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
    import uvicorn
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
    else:
        server.run()
