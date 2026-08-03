"""Session state storage and lifecycle management.

Module-level dicts hold all session-scoped data.  Functions here manage
creation, activity tracking, detachment, and cleanup.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import threading
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_DIR = Path(__file__).resolve().parents[1]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

warnings = __import__("warnings")
warnings.filterwarnings("ignore")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
_LOCAL_LIBERO_DIR = str(_REPO_ROOT / "third_party" / "LIBERO")
os.environ.setdefault(
    "LIBERO_DIR",
    _LOCAL_LIBERO_DIR if os.path.isdir(_LOCAL_LIBERO_DIR) else "/tmp/LIBERO",
)
os.environ.setdefault("LIBERO_DATASET_PATH", f"{_REPO_ROOT}/sim/venvs/libero/assets/datasets")

# ── Session-scoped storage ────────────────────────────────────────────
# session_id → {handle → {"worker_url": str, "remote_handle": str, "env_id": str, ...}}
_session_envs: dict[str, dict[str, dict]] = {}
# session_id → {handle → last obs dict}
_session_last_obs: dict[str, dict[str, dict]] = {}
# session_id  or  "session_id/handle" → set of asyncio.Queue (for live-stream SSE push)
_session_streams: dict[str, set[asyncio.Queue]] = {}
# session_id  or  "session_id/handle" → asyncio.Task (live-stream render loop)
_session_stream_tasks: dict[str, asyncio.Task] = {}
# session_id → stream interval in seconds (default 0.05)
_session_stream_interval: dict[str, float] = {}
# session_id → float (monotonic timestamp of last activity)
_session_last_activity: dict[str, float] = {}

# seconds of inactivity before a non-SSE session is considered stale
_SESSION_TTL_S = 1800  # 30 min


# contextvar propagated through SSE → MCP tool calls
_current_session: contextvars.ContextVar[str] = contextvars.ContextVar("mcp_session", default="")

_initialized = False
# set of session ids that are currently connected via SSE
_sse_sessions: set[str] = set()

# BenchWorkerManager — created lazily on first use
_worker_mgr = None
_worker_mgr_lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════
# Public helpers
# ══════════════════════════════════════════════════════════════════════

def _get_mgr():
    """Return the global BenchWorkerManager singleton, creating it on first call.

    Guarded by a lock: MCP tools run in a thread pool (anyio.to_thread), so
    concurrent first calls could otherwise each build their own manager —
    fragmenting the worker pool and defeating GPU round-robin / load
    balancing (each fresh instance restarts its GPU cursor at 0).
    """
    global _worker_mgr
    if _worker_mgr is None:
        with _worker_mgr_lock:
            if _worker_mgr is None:  # double-checked under lock
                from sim.mcp_server.worker_mgr import BenchWorkerManager
                _worker_mgr = BenchWorkerManager()
    return _worker_mgr


def _init():
    """One-time initialisation: register dummy envs into gym."""
    global _initialized
    if _initialized:
        return
    import sim.env_registry  # noqa: F401 — triggers gym registration
    _initialized = True


def _touch_session(sid: str) -> None:
    """Record activity for *sid* so the TTL sweeper doesn't clean it up."""
    if sid:
        _session_last_activity[sid] = time.monotonic()


def _is_sse_session(sid: str) -> bool:
    """Return True if *sid* currently has an active SSE connection."""
    return sid in _sse_sessions


# ══════════════════════════════════════════════════════════════════════
# Session lifecycle
# ══════════════════════════════════════════════════════════════════════

def _cleanup_session(sid: str) -> None:
    """Close all env handles on their workers and remove all traces of *sid*."""
    mgr = _get_mgr()
    prefix = f"{sid}/"
    # Stop stream tasks and queues
    for sk in list(_session_stream_tasks):
        if sk == sid or sk.startswith(prefix):
            task = _session_stream_tasks.pop(sk, None)
            if task and not task.done():
                task.cancel()
    for sk in list(_session_streams):
        if sk == sid or sk.startswith(prefix):
            _session_streams.pop(sk, None)
    # Close envs via worker proxy
    envs = _session_envs.pop(sid, {})
    from sim.mcp_server.collision import remove_checker

    for handle, meta in envs.items():
        try:
            mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}", method="DELETE")
        except Exception:
            pass
        finally:
            # TTL and disconnected-session cleanup must pair every successful
            # acquire_worker with a release, just like explicit close_env.
            try:
                mgr.release_worker(meta.get("worker_url", ""))
            except Exception:
                pass
            remove_checker(handle)
    _session_last_obs.pop(sid, None)
    _session_stream_interval.pop(sid, None)
    _session_last_activity.pop(sid, None)
    _sse_sessions.discard(sid)


def _detach_sse_session(sid: str) -> None:
    """Stop streams for *sid* but keep envs alive for cross-turn reuse.

    Unlike ``_cleanup_session``, this does **not** close environments.
    The TTL sweeper will clean up idle sessions after ``_SESSION_TTL_S`` seconds.
    """
    prefix = f"{sid}/"
    # Stop stream tasks
    for sk in list(_session_stream_tasks):
        if sk == sid or sk.startswith(prefix):
            task = _session_stream_tasks.pop(sk, None)
            if task and not task.done():
                task.cancel()
    # Clear stream queues
    for sk in list(_session_streams):
        if sk == sid or sk.startswith(prefix):
            _session_streams.pop(sk, None)
    # Remove SSE marker so TTL sweeper can clean it later
    _sse_sessions.discard(sid)


async def _stale_session_sweeper(interval_s: float = 60) -> None:
    """Periodically clean up sessions that have been idle > ``_SESSION_TTL_S``."""
    _logger = __import__("logging").getLogger("openeta.sweeper")
    while True:
        await asyncio.sleep(interval_s)
        now = time.monotonic()
        stale: list[str] = []
        for sid, last_ts in list(_session_last_activity.items()):
            if sid in _sse_sessions:
                continue  # active SSE connection keeps session alive
            if now - last_ts > _SESSION_TTL_S:
                stale.append(sid)
        for sid in stale:
            _logger.info("Cleaning stale REST session %s (idle %.0fs)", sid, now - _session_last_activity[sid])
            _cleanup_session(sid)
