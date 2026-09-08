"""Process-local environment retirement shared by explicit and TTL cleanup.

Keep the handle and worker reference until remote deletion is acknowledged.
The lifecycle is resource ownership, not an Agent task-state machine.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from weakref import WeakValueDictionary

from adapter.environment_lifecycle import close_response_error

_locks: WeakValueDictionary = WeakValueDictionary()
_locks_guard = threading.Lock()


def environment_lock(session_id: str, handle: str):
    # Waiting users hold strong references too. Explicitly popping a lock after
    # close would let a third caller bypass waiters still using the original.
    with _locks_guard:
        key = (session_id, handle)
        lock = _locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _locks[key] = lock
        return lock


def close_managed_environment(
    *, session_id: str, handle: str, envs: dict, manager,
    retire: Callable[[dict], None],
) -> dict:
    with environment_lock(session_id, handle):
        meta = envs.get(session_id, {}).get(handle)
        if meta is None:
            return {"ok": True, "already_closed": True, "cleanup_errors": []}
        state = meta.setdefault("close_lifecycle", {
            "state": "active", "attempts": 0,
            "remote_confirmed": False, "worker_released": False,
        })
        state["attempts"] += 1
        state["state"] = "closing"
        stage = "remote_close"
        try:
            if not state["remote_confirmed"]:
                remote = manager.proxy_handle_op(
                    meta, f"/env/{meta['remote_handle']}", method="DELETE",
                )
                error = close_response_error(remote)
                if error is not None:
                    raise RuntimeError(error)
                state["remote"] = remote
                state["remote_confirmed"] = True
            stage = "release_worker"
            if not state["worker_released"]:
                manager.release_worker(meta.get("worker_url", ""))
                state["worker_released"] = True
            stage = "retire_local"
            retire(meta)
        except Exception as exc:
            state["state"] = "close_failed"
            state["last_error"] = f"{stage}: {type(exc).__name__}: {exc}"
            return {
                "ok": False, "already_closed": False, "close_state": "close_failed",
                "retryable": True, "cleanup_errors": [state["last_error"]],
                "error": state["last_error"], "remote": state.get("remote", {}),
            }
        state["state"] = "closed"
        # Cache/checker retirement can be retried; removing this identity is the
        # final commit, after all cleanup phases have acknowledged completion.
        envs.get(session_id, {}).pop(handle, None)
        return {
            "ok": True, "already_closed": False, "close_state": "closed",
            "remote": state.get("remote", {}), "cleanup_errors": [],
        }
