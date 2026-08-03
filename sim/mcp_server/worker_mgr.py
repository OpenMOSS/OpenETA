"""Per-bench subprocess worker management and proxy helpers.

``BenchWorkerManager`` spawns one subprocess per bench (libero, metaworld, …).
Each worker runs in its own venv Python to avoid C-extension conflicts.

Proxy helpers (``_proxy_step``, etc.) forward env operations to the correct
worker and cache observations for SSE live-streaming.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

from sim.mcp_server.session import (
    _get_mgr,
    _session_envs,
    _session_last_obs,
    _session_streams,
    _session_stream_interval,
    _SIM_DIR,
)

# ══════════════════════════════════════════════════════════════════════
# Bench → worker resolution
# ══════════════════════════════════════════════════════════════════════

_BENCH_MAP: dict[str, str] = {
    "metaworld": "metaworld", "maniskill": "maniskill",
    "libero": "libero", "robocasa": "robocasa",
    "genesis": "genesis", "d4rl": "d4rl", "behavior": "behavior",
}


def _configured_timeout(name: str, default: float) -> float:
    """Read a positive timeout from the environment with a safe fallback."""
    try:
        return max(0.1, float(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


def _bench_for_env_id(env_id: str) -> str:
    """Extract bench name from an env_id like ``openeta/libero_libero_10_task0-v0``."""
    part = env_id.split("/")[1] if "/" in env_id else env_id
    bench = part.split("_")[0]
    return _BENCH_MAP.get(bench, bench)


# ══════════════════════════════════════════════════════════════════════
# Worker-pool configuration
# ══════════════════════════════════════════════════════════════════════

def _pool_max() -> int:
    """Max workers per bench.  Override with ``OPENETA_WORKER_POOL_MAX``."""
    try:
        return max(1, int(os.environ.get("OPENETA_WORKER_POOL_MAX", "8")))
    except ValueError:
        return 8


def _detect_gpus() -> list[int]:
    """Return the list of visible GPU ordinals for round-robin binding.

    Honours ``OPENETA_WORKER_GPUS`` (comma-separated, e.g. ``"0,1"``) if set;
    otherwise queries ``nvidia-smi``.  Falls back to ``[0]`` when no GPU is
    detectable so binding still produces a valid (single-device) assignment.
    """
    override = os.environ.get("OPENETA_WORKER_GPUS", "").strip()
    if override:
        gpus = [int(x) for x in override.split(",") if x.strip().isdigit()]
        if gpus:
            return gpus
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            timeout=5, text=True,
        )
        gpus = [int(l.strip()) for l in out.splitlines() if l.strip().isdigit()]
        if gpus:
            return gpus
    except Exception:
        pass
    return [0]


# ──────────────────────────────────────────────────────────────────────
# EGL ↔ CUDA device-index calibration
#
# robosuite's EGL backend selects a render device with
# ``all_devices[MUJOCO_EGL_DEVICE_ID]`` where ``all_devices`` comes from
# ``eglQueryDevicesEXT()``.  That EGL enumeration order is NOT the CUDA ordinal
# order — on multi-GPU hosts they differ (EGL may even expose more entries than
# there are GPUs).  So ``MUJOCO_EGL_DEVICE_ID = <cuda ordinal>`` renders on the
# wrong physical GPU.  We anchor the two index spaces on the PCI bus id:
#
#   nvidia-smi:            CUDA ordinal      → PCI bus id
#   /dev/dri/by-path:      PCI bus id        → DRM card node
#   eglQueryDeviceStringEXT(EGL_DRM_DEVICE_FILE_EXT):  EGL index → DRM card node
#
# Composing these yields CUDA ordinal → EGL index, which is the value
# MUJOCO_EGL_DEVICE_ID must actually hold.  Cached per manager process.
_EGL_DRM_DEVICE_FILE_EXT = 0x3233
_egl_map_cache: dict[int, int] | None = None
_egl_map_lock = threading.Lock()


def _pci_to_drm_card() -> dict[str, str]:
    """Map normalised PCI bus id → DRM card node path via /dev/dri/by-path."""
    import glob
    out: dict[str, str] = {}
    for link in glob.glob("/dev/dri/by-path/pci-*-card"):
        try:
            target = os.path.basename(os.path.realpath(link))  # e.g. "card1"
            base = os.path.basename(link)                       # pci-0000:83:00.0-card
            pci = base[len("pci-"):-len("-card")]               # 0000:83:00.0
            out[pci.lower()] = target
        except Exception:
            continue
    return out


def _cuda_to_pci() -> dict[int, str]:
    """Map CUDA ordinal → normalised PCI bus id via nvidia-smi."""
    out: dict[int, str] = {}
    try:
        res = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            timeout=5, text=True,
        )
        for line in res.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            # nvidia-smi prints e.g. "00000000:83:00.0"; DRM by-path uses
            # "0000:83:00.0".  Normalise to the last 12 chars (dddd:bb:dd.f).
            pci = parts[1].lower()
            pci = pci[-12:] if len(pci) >= 12 else pci
            out[int(parts[0])] = pci
    except Exception:
        pass
    return out


def _egl_index_to_drm_card(bench_python: str) -> dict[int, str]:
    """Map EGL enumeration index → DRM card node, run inside the bench venv.

    EGL lives in the bench's venv (mujoco/robosuite), so we enumerate there via
    a short subprocess.  Read-only: enumerates devices and queries the DRM node
    string; it does NOT initialise a display or allocate GPU memory.
    """
    snippet = (
        "import os;os.environ.setdefault('PYOPENGL_PLATFORM','egl');"
        "os.environ.pop('CUDA_VISIBLE_DEVICES',None);"
        "from mujoco.egl import egl_ext as E;"
        "from OpenGL.EGL.EXT.device_query import eglQueryDeviceStringEXT as q;"
        "d=E.eglQueryDevicesEXT();"
        "\nfor i,dev in enumerate(d):\n"
        " s=None\n"
        " try:\n"
        f"  s=q(dev,{_EGL_DRM_DEVICE_FILE_EXT})\n"
        " except Exception:\n"
        "  pass\n"
        " s=s.decode() if isinstance(s,bytes) else ('' if s is None else str(s))\n"
        " print(f'{i}\\t{s}')\n"
    )
    out: dict[int, str] = {}
    try:
        res = subprocess.check_output(
            [bench_python, "-c", snippet], timeout=30, text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in res.splitlines():
            if "\t" not in line:
                continue
            idx_s, path = line.split("\t", 1)
            if idx_s.strip().isdigit() and path.strip():
                out[int(idx_s.strip())] = os.path.basename(path.strip())
    except Exception:
        pass
    return out


def _cuda_to_egl_index(bench_python: str) -> dict[int, int]:
    """CUDA ordinal → EGL index (for MUJOCO_EGL_DEVICE_ID).  Cached; safe."""
    global _egl_map_cache
    with _egl_map_lock:
        if _egl_map_cache is not None:
            return _egl_map_cache
        mapping: dict[int, int] = {}
        try:
            cuda_pci = _cuda_to_pci()
            pci_drm = _pci_to_drm_card()
            egl_drm = _egl_index_to_drm_card(bench_python)
            # invert egl_drm: card node → first EGL index that backs it
            drm_egl: dict[str, int] = {}
            for egl_idx, card in egl_drm.items():
                drm_egl.setdefault(card, egl_idx)
            for cuda_idx, pci in cuda_pci.items():
                card = pci_drm.get(pci)
                if card and card in drm_egl:
                    mapping[cuda_idx] = drm_egl[card]
        except Exception:
            mapping = {}
        _egl_map_cache = mapping
        return mapping


def _venv_python(bench: str) -> str | None:
    """Return the venv Python interpreter for *bench*, or None if not found."""
    import sys as _sys
    venv_dir = os.path.join(str(_SIM_DIR), "venvs", bench)
    candidates = [
        os.path.join(venv_dir, "runtime", "bin", "python3.11"),
        os.path.join(venv_dir, "runtime", "bin", "python"),
        os.path.join(venv_dir, "bin", "python3.11"),
        os.path.join(venv_dir, "bin", "python3.10"),
        os.path.join(venv_dir, "bin", "python3"),
        os.path.join(venv_dir, "bin", "python"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    # Fall back to system python for benches that work on the system interpreter
    if bench in ("metaworld", "dummy"):
        return _sys.executable
    return None


def _start_pipe_drainers(proc: "subprocess.Popen") -> None:
    """Continuously drain a worker's stdout/stderr so it can't block on write.

    Once the parent stops reading a PIPE, the worker blocks the moment the
    ~64 KiB kernel buffer fills.  We read-and-discard both streams on daemon
    threads for the life of the process.  Kept lightweight (discard, don't
    buffer) since worker logs are only useful during startup, which the caller
    already captured line-by-line before calling this.
    """
    _dbg_dir = os.environ.get("OPENETA_WORKER_LOG_DIR")

    def _drain(stream, tag) -> None:
        if stream is None:
            return
        sink = None
        if _dbg_dir:
            try:
                os.makedirs(_dbg_dir, exist_ok=True)
                sink = open(os.path.join(_dbg_dir, f"worker_{proc.pid}_{tag}.log"), "a")
            except Exception:
                sink = None
        try:
            for line in iter(stream.readline, ""):
                if sink is not None:
                    sink.write(line)
                    sink.flush()
        except Exception:
            pass
        finally:
            if sink is not None:
                sink.close()

    for stream, tag in ((proc.stdout, "out"), (proc.stderr, "err")):
        t = threading.Thread(target=_drain, args=(stream, tag), daemon=True)
        t.start()


# ══════════════════════════════════════════════════════════════════════
# BenchWorkerHandle
# ══════════════════════════════════════════════════════════════════════

@dataclass
class BenchWorkerHandle:
    bench: str
    port: int
    process: subprocess.Popen | None
    base_url: str
    gpu: int = -1              # GPU ordinal this worker is bound to (-1 = unset)
    env_count: int = 0         # live envs on this worker (for load balancing)

    def proxy(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> dict:
        """Forward an HTTP request to the worker and return parsed JSON."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            # Isaac Kit can spend several minutes compiling shaders on the
            # first BEHAVIOR create/reset. Keep this configurable while using
            # a timeout that does not kill a healthy cold start.
            if timeout_s is None:
                timeout_s = float(
                    os.environ.get("OPENETA_WORKER_HTTP_TIMEOUT_S", "600")
                )
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                message = f"HTTP {exc.code}: {exc.reason}"
                return {
                    "ok": False,
                    "error": message,
                    "error_detail": {
                        "kind": "http_error",
                        "message": message,
                        "method": method,
                        "path": path,
                        "status_code": exc.code,
                    },
                }
        except Exception as exc:
            reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
            kind = (
                "transport_timeout"
                if isinstance(reason, (TimeoutError, socket.timeout))
                else "transport_error"
            )
            message = f"Worker request failed: {exc}"
            return {
                "ok": False,
                "error": message,
                "error_detail": {
                    "kind": kind,
                    "message": message,
                    "error_type": type(exc).__name__,
                    "method": method,
                    "path": path,
                    "timeout_s": timeout_s,
                },
            }

    def stop(self, *, wait: bool = False) -> None:
        """Terminate the worker process."""
        if self.process is None:
            return
        try:
            self.process.terminate()
            if wait:
                self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════
# BenchWorkerManager
# ══════════════════════════════════════════════════════════════════════

class BenchWorkerManager:
    """Manage per-bench subprocess workers, starting them on demand."""

    def __init__(self) -> None:
        # One pool (list of workers) per bench.  Guarded by _lock for the
        # whole check→spawn→register→count lifecycle, since MCP tools run in
        # a thread pool (anyio.to_thread) and hit this concurrently.
        self._pools: dict[str, list[BenchWorkerHandle]] = {}
        self._lock = threading.RLock()
        self._gpus = _detect_gpus()
        self._next_gpu = 0  # round-robin cursor for GPU binding

    # ── low-level spawn (no locking; callers hold _lock) ───────────────
    def _spawn_worker(self, bench: str) -> BenchWorkerHandle:
        """Start one worker subprocess for *bench*, bound to the next GPU."""
        python_exe = _venv_python(bench)
        if python_exe is None:
            raise RuntimeError(f"No Python interpreter found for bench '{bench}'")

        # Round-robin GPU assignment across detected devices (CUDA ordinals).
        gpu = self._gpus[self._next_gpu % len(self._gpus)]
        self._next_gpu += 1
        child_env = dict(os.environ)

        # Pin BOTH compute and rendering to the same physical GPU.
        #   * CUDA (torch) honours CUDA_VISIBLE_DEVICES → the pinned GPU becomes
        #     the sole visible device, seen by torch as "cuda:0".
        #   * EGL (robosuite render) ignores CUDA_VISIBLE_DEVICES and selects by
        #     eglQueryDevicesEXT() index, whose order differs from the CUDA
        #     ordinal.  We translate the CUDA ordinal → EGL index via PCI-bus
        #     calibration so rendering lands on the SAME physical GPU as compute
        #     (previously MUJOCO_EGL_DEVICE_ID=<cuda ordinal> silently rendered
        #     on the wrong card).
        child_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        egl_idx = _cuda_to_egl_index(python_exe).get(gpu)
        if egl_idx is not None:
            # robosuite asserts MUJOCO_EGL_DEVICE_ID is a substring of
            # CUDA_VISIBLE_DEVICES (binding_utils.py).  When the translated EGL
            # index differs from the CUDA ordinal, that assert would fire, so we
            # widen CUDA_VISIBLE_DEVICES to include both ids while still putting
            # the pinned GPU first (→ torch "cuda:0" stays the intended card).
            visible = str(gpu) if str(egl_idx) == str(gpu) else f"{gpu},{egl_idx}"
            child_env["CUDA_VISIBLE_DEVICES"] = visible
            child_env["MUJOCO_EGL_DEVICE_ID"] = str(egl_idx)
        else:
            # Calibration unavailable (no nvidia-smi / DRM paths): fall back to
            # the CUDA ordinal.  May mis-target rendering on some hosts, but
            # keeps single-GPU setups working.
            child_env["MUJOCO_EGL_DEVICE_ID"] = str(gpu)
        # The worker adds the repo root to sys.path itself (bench_worker sets
        # _REPO), and runs as a *file path* so its own dir (sim/) is on
        # sys.path[0].  A stray PYTHONPATH=<repo> inherited from the parent
        # puts the repo root ahead of sim/ inconsistently and makes
        # ``import adapter`` resolve to sim/adapter.py ("adapter is not a
        # package").  Drop it so the worker's own path setup is authoritative.
        child_env.pop("PYTHONPATH", None)
        if bench == "behavior":
            behavior_root = os.path.join(str(_SIM_DIR), "venvs", "behavior")
            child_env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            child_env.setdefault("OMNIGIBSON_HEADLESS", "True")
            child_env.setdefault("OMNIGIBSON_GPU_ID", "0")
            child_env.setdefault(
                "OMNIGIBSON_DATA_PATH",
                os.path.join(behavior_root, "src", "BEHAVIOR-1K", "datasets"),
            )

        worker_script = os.path.join(str(_SIM_DIR), "bench_worker.py")
        proc = subprocess.Popen(
            [python_exe, "-u", worker_script, "--bench", bench, "--port", "0"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env,
        )
        # Read the port line from stdout (first non-empty digit-only line)
        port_str = ""
        for _ in range(60):  # 30s timeout
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    stderr_output = proc.stderr.read()
                    raise RuntimeError(
                        f"Worker for '{bench}' exited with code {proc.returncode}: {stderr_output[:500]}"
                    )
                continue
            stripped = line.strip()
            if stripped and stripped.isdigit():
                port_str = stripped
                break

        if not port_str:
            proc.kill()
            raise RuntimeError(f"Worker for '{bench}' did not print a port within 30s")

        port = int(port_str)
        base_url = f"http://127.0.0.1:{port}"
        handle = BenchWorkerHandle(bench=bench, port=port, process=proc,
                                   base_url=base_url, gpu=gpu)

        # Drain the worker's stdout/stderr for the rest of its life.  We only
        # read up to the port line above; after that the pipes are never read
        # again.  A worker that logs enough (MuJoCo/EGL/libero chatter under
        # concurrent env creation) fills the ~64 KiB pipe buffer and then
        # BLOCKS on write — the HTTP server thread stalls and requests get
        # "connection refused".  This surfaced only at concurrency ≥ ~6, where
        # 3 envs land on one worker.  Daemon threads keep the pipes empty.
        _start_pipe_drainers(proc)

        if not self._health_check(handle):
            handle.stop(wait=True)
            raise RuntimeError(f"Worker for '{bench}' failed health check")
        return handle

    def ensure_worker(self, bench: str) -> BenchWorkerHandle:
        """Get a healthy worker for *bench* (any pool member).

        Used for read-only fan-out (``list_all_envs``) where any live worker
        will do.  For creating an env, use ``acquire_worker`` which also does
        load balancing and reference counting.
        """
        with self._lock:
            pool = self._pools.setdefault(bench, [])
            # Prune only workers whose process has exited (not ones that are
            # merely slow to answer /health while busy), then reuse any live
            # worker.  A busy worker will simply serve the read op once free.
            for w in list(pool):
                if self._is_dead(w):
                    w.stop(wait=True)
                    pool.remove(w)
            for w in pool:
                if self._health_check_quick(w):
                    return w
            if pool:
                return pool[0]  # live process, transiently busy — reuse it
            handle = self._spawn_worker(bench)
            pool.append(handle)
            return handle

    def acquire_worker(self, bench: str) -> BenchWorkerHandle:
        """Pick (or grow) a worker for a new env, incrementing its env_count.

        Selects the healthy worker with the fewest live envs.  If the least
        loaded worker is already busy and the pool has room, spawns a new one
        (round-robin GPU) so concurrent creates fan out instead of queuing on
        a single process.  Caller must pair this with ``release_worker`` on
        close.  Thread-safe.
        """
        with self._lock:
            pool = self._pools.setdefault(bench, [])
            # Drop workers whose process has actually exited.  Do NOT prune on a
            # slow /health poll: a worker mid-create blocks its event loop and
            # would be wrongly killed, orphaning the env being created on it.
            for w in list(pool):
                if self._is_dead(w):
                    w.stop(wait=True)
                    pool.remove(w)

            pool_max = _pool_max()
            if not pool:
                chosen = self._spawn_worker(bench)
                pool.append(chosen)
            else:
                chosen = min(pool, key=lambda w: w.env_count)
                # If the least-loaded worker already has an env and we have
                # headroom, add a worker to spread the load.
                if chosen.env_count > 0 and len(pool) < pool_max:
                    chosen = self._spawn_worker(bench)
                    pool.append(chosen)

            chosen.env_count += 1
            return chosen

    def release_worker(self, base_url: str) -> None:
        """Decrement the env_count for the worker at *base_url* (on close)."""
        with self._lock:
            for bench, pool in self._pools.items():
                for w in pool:
                    if w.base_url == base_url:
                        if bench == "behavior":
                            # Isaac Kit is process-global and cannot be cleanly
                            # re-created after og.shutdown(). A BEHAVIOR worker
                            # is deliberately single-environment / single-use.
                            w.stop(wait=True)
                            pool.remove(w)
                            return
                        w.env_count = max(0, w.env_count - 1)
                        return

    def _health_check_quick(self, wh: BenchWorkerHandle) -> bool:
        """Single-shot health check (no retry) for pool pruning."""
        try:
            req = urllib.request.Request(f"{wh.base_url}/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return json.loads(resp.read().decode("utf-8")).get("ok", False)
        except Exception:
            return False

    @staticmethod
    def _is_dead(wh: BenchWorkerHandle) -> bool:
        """True only if the worker OS process has actually exited.

        We must NOT treat a slow /health response as death: env creation and
        stepping briefly block the worker's event loop, so /health can time out
        while the process is perfectly alive and mid-request.  Pruning on an
        HTTP timeout would terminate a busy worker and orphan the env created on
        it (its later step then hits a killed process — "connection refused").
        Liveness is decided by the process, not the socket.
        """
        proc = wh.process
        if proc is None:
            return False  # externally-provided handle; assume alive
        return proc.poll() is not None

    def _health_check(self, wh: BenchWorkerHandle) -> bool:
        for _ in range(10):  # retry for up to ~5s
            try:
                req = urllib.request.Request(f"{wh.base_url}/health")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("ok", False)
            except Exception:
                time.sleep(0.5)
        return False

    def list_all_envs(self, bench: str | None = None, query: str = "") -> list[dict]:
        """Aggregate env lists from one or all workers.

        Eagerly starts workers for any bench that has a venv but isn't
        running yet, so the first call shows the full catalogue.
        """
        if bench:
            benches = [bench]
        else:
            benches = [b for b, ok in self.available_benches().items() if ok and b != "dummy"]
        all_envs: list[dict] = []
        seen: set[str] = set()
        for b in benches:
            try:
                wh = self.ensure_worker(b)
                params = []
                if query:
                    params.append(f"q={query}")
                qs = f"?{'&'.join(params)}" if params else ""
                result = wh.proxy("GET", f"/envs{qs}")
                for env_dict in result.get("envs", []):
                    eid = env_dict.get("id", "")
                    if eid in seen:
                        continue
                    seen.add(eid)
                    env_dict["_bench"] = b
                    all_envs.append(env_dict)
            except Exception:
                pass
        return all_envs

    def proxy_env_op(self, env_id: str, path: str, method: str = "GET", body: dict | None = None) -> dict:
        """Resolve env_id → bench → worker, then proxy the request.

        Uses any healthy pool worker (read-only catalogue ops).  For creating
        an env, use ``create_env_on_worker`` so the env and its handle are
        pinned to the same acquired worker.
        """
        bench = _bench_for_env_id(env_id)
        wh = self.ensure_worker(bench)
        return wh.proxy(method, path, body)

    def create_env_on_worker(self, env_id: str, body: dict) -> tuple[dict, BenchWorkerHandle]:
        """Acquire a pool worker, create the env on it, return (result, worker).

        The returned worker is reference-counted (``acquire_worker``); the
        caller records ``worker.base_url`` in the session meta so every later
        op for this handle routes to the same worker.  On failure the count is
        released so a failed create doesn't leak a slot.
        """
        bench = _bench_for_env_id(env_id)
        wh = self.acquire_worker(bench)
        try:
            result = wh.proxy("POST", "/env", body)
        except Exception:
            self.release_worker(wh.base_url)
            raise
        if "error" in result:
            self.release_worker(wh.base_url)
        return result, wh

    def proxy_handle_op(self, handle_meta: dict, path: str, method: str = "GET", body: dict | None = None) -> dict:
        """Proxy a request for an already-created env handle."""
        worker_url = handle_meta["worker_url"]
        wh = BenchWorkerHandle(bench="", port=0, process=None, base_url=worker_url)
        # A slow create/reset can be healthy (shader compilation), but DELETE
        # is cleanup and must not stall the whole episode or a batch evaluator
        # for the general 10-minute worker timeout.  The server still releases
        # its worker reference when this bounded request reports an error.
        timeout_s = None
        if method.upper() == "DELETE":
            timeout_s = _configured_timeout("OPENETA_WORKER_DELETE_TIMEOUT_S", 10.0)
        return wh.proxy(method, path, body, timeout_s=timeout_s)

    def stop_all(self) -> None:
        """Stop all workers across all pools."""
        with self._lock:
            for pool in self._pools.values():
                for wh in pool:
                    wh.stop()
            self._pools.clear()

    def available_benches(self) -> dict[str, bool]:
        """Return which benches have venvs (i.e. are launchable)."""
        result = {"dummy": True}
        for bench in ["metaworld", "maniskill", "libero", "robocasa", "genesis", "d4rl", "behavior"]:
            result[bench] = _venv_python(bench) is not None
        return result


# ══════════════════════════════════════════════════════════════════════
# Proxy helpers (used by REST API and MCP tools)
# ══════════════════════════════════════════════════════════════════════

def _proxy_step(
    meta: dict,
    action,
    num_steps: int = 1,
    render: bool = True,
    include_cameras: bool | None = None,
) -> dict:
    """Proxy a step request to the worker and cache the observation.

    ``render`` controls whether the worker refreshes its raw camera cache;
    ``include_cameras`` controls whether those arrays are serialised into the
    HTTP response.  ``move_to`` periodically renders for the live dashboard
    while keeping every controller response camera-free.
    """
    mgr = _get_mgr()
    body: dict = {}
    if action is not None:
        if hasattr(action, "tolist"):
            body["action"] = action.tolist()
        elif isinstance(action, (list, tuple)):
            body["action"] = list(action)
        else:
            body["action"] = action
    body["num_steps"] = num_steps
    if not render:
        body["render"] = False
    if include_cameras is not None:
        body["include_cameras"] = bool(include_cameras)
    result = mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/step", method="POST", body=body)
    # Cache observation for streaming
    obs = result.get("observation")
    if obs:
        sid = meta.get("_sid", "")
        cache = _session_last_obs.setdefault(sid, {})
        # When the worker skipped rendering (render=False), the new obs carries
        # no camera frames.  Preserve the previously cached frames so the
        # dashboard doesn't flicker to blank between background /render_all
        # refreshes — only the robot/EE state needs to be current here.
        prev = cache.get(meta["remote_handle"])
        if isinstance(prev, dict) and isinstance(obs, dict) and not obs.get("cameras"):
            prev_cams = prev.get("cameras")
            if prev_cams:
                obs = {**obs, "cameras": prev_cams}
        cache[meta["remote_handle"]] = obs
    return result


def _proxy_reset(meta: dict, seed: int | None = None) -> dict:
    """Proxy a reset request to the worker and cache the observation."""
    mgr = _get_mgr()
    body = {"seed": seed} if seed is not None else {}
    result = mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/reset", method="POST", body=body)
    _session_last_obs.setdefault(meta.get("_sid", ""), {})[meta["remote_handle"]] = result
    return result


def _proxy_observe(meta: dict) -> dict:
    """Proxy an observe request to the worker."""
    mgr = _get_mgr()
    return mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/observe", method="POST")


def _proxy_render(meta: dict) -> dict:
    """Proxy a render request to the worker."""
    mgr = _get_mgr()
    return mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/render", method="POST")


def _proxy_render_multiview(
    meta: dict,
    *,
    width: int = 256,
    height: int = 256,
    hide_robot: bool = False,
    lookat_xyz_m: list[float] | None = None,
    distance_m: float = 1.3,
) -> dict:
    """Render synchronized virtual RGB-D cameras on the existing env handle."""

    mgr = _get_mgr()
    return mgr.proxy_handle_op(
        meta,
        f"/env/{meta['remote_handle']}/render_multiview",
        method="POST",
        body={
            "width": int(width),
            "height": int(height),
            "hide_robot": bool(hide_robot),
            **(
                {"lookat_xyz_m": list(lookat_xyz_m)}
                if lookat_xyz_m is not None
                else {}
            ),
            "distance_m": float(distance_m),
        },
    )


def _proxy_check_task(meta: dict) -> dict:
    """Proxy the narrow native task-checker result to the worker."""
    mgr = _get_mgr()
    return mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/check_task", method="POST")


def _proxy_render_all(worker_url: str, remote_handles: list[str]) -> dict:
    """Call the worker's ``/render_all`` endpoint for parallel batch rendering."""
    wh = BenchWorkerHandle(bench="", port=0, process=None, base_url=worker_url)
    timeout_s = _configured_timeout("OPENETA_WORKER_RENDER_TIMEOUT_S", 2.0)
    return wh.proxy(
        "POST",
        "/render_all",
        body={"handles": remote_handles},
        timeout_s=timeout_s,
    )


# ══════════════════════════════════════════════════════════════════════
# Live-stream helpers
# ══════════════════════════════════════════════════════════════════════

# Thread pool for background render refreshes (offloads blocking HTTP from event loop)
_render_executor = None
_render_inflight: dict[tuple[str, str], object] = {}
_render_inflight_lock = threading.Lock()

def _get_render_executor() -> ThreadPoolExecutor:
    global _render_executor
    if _render_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        _render_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sse_refresh")
    return _render_executor


def _refresh_cache_for_worker(wurl: str, remote_hs: list[str], sid: str) -> None:
    """Blocking call: ask one worker to render all its handles, update cache."""
    try:
        result = _proxy_render_all(wurl, remote_hs)
        by_handle = result.get("by_handle", {})
        for rh, obs in by_handle.items():
            if "error" not in obs and not obs.get("skipped"):
                _session_last_obs.setdefault(sid, {})[rh] = obs
    except Exception:
        pass


def _submit_render_refresh(wurl: str, remote_hs: list[str], sid: str) -> bool:
    """Submit at most one background render per session/worker pair.

    The dashboard ticks much faster than a slow or wedged worker can answer.
    Keeping the queued/running future in this keyed map prevents every 500 ms
    tick (and overlapping session/per-handle streams) from adding another copy
    of the same request to the executor's unbounded work queue.
    """
    key = (sid, wurl)
    with _render_inflight_lock:
        current = _render_inflight.get(key)
        if current is not None and not current.done():
            return False
        future = _get_render_executor().submit(
            _refresh_cache_for_worker,
            wurl,
            list(remote_hs),
            sid,
        )
        _render_inflight[key] = future

    def _forget(completed) -> None:
        with _render_inflight_lock:
            if _render_inflight.get(key) is completed:
                _render_inflight.pop(key, None)

    future.add_done_callback(_forget)
    return True


async def _live_stream_loop(sid: str, interval_s: float, stream_key: str = "", handle: str = "") -> None:
    """Push camera frames to SSE queues at *interval_s*.

    Reads from the local ``_session_last_obs`` cache directly (updated by
    ``_proxy_step`` / ``_proxy_reset`` after every env operation), so frames
    are pushed **immediately** during ``move_to`` without any blocking HTTP.

    Every ~500 ms a background thread-pool task asks each worker for a fresh
    render (``/render_all``) so the dashboard stays live even when no one is
    stepping.
    """
    sk = stream_key or sid
    tick = 0
    _REFRESH_EVERY_N = max(1, int(0.5 / max(interval_s, 0.01)))  # ~every 500 ms

    while True:
        try:
            queues = _session_streams.get(sk, set())
            if not queues:
                break

            env_dict = _session_envs.get(sid, {})
            if not env_dict:
                await asyncio.sleep(interval_s)
                continue

            # ── background refresh (non-blocking) ────────────────
            if tick % _REFRESH_EVERY_N == 0:
                if handle:
                    meta = env_dict.get(handle)
                    if meta is not None:
                        _submit_render_refresh(
                            meta["worker_url"],
                            [meta["remote_handle"]],
                            sid,
                        )
                else:
                    # Group handles by worker, fire one refresh per worker
                    by_worker: dict[str, list[str]] = {}
                    for _h, meta in list(env_dict.items()):
                        wurl = meta["worker_url"]
                        by_worker.setdefault(wurl, []).append(meta["remote_handle"])
                    for wurl, remote_hs in by_worker.items():
                        _submit_render_refresh(wurl, remote_hs, sid)
            tick += 1

            # ── build payload from local cache (instant, no I/O) ──
            if handle:
                frames = _collect_camera_frames_from_cache(sid, env_dict[handle]["remote_handle"])
                if not frames:
                    await asyncio.sleep(interval_s)
                    continue
                payload = json.dumps({"handle": handle, "cameras": frames})
            else:
                parts: list[dict] = []
                for h, meta in list(env_dict.items()):
                    frames = _collect_camera_frames_from_cache(sid, meta["remote_handle"])
                    if frames:
                        parts.append({
                            "handle": h,
                            "env_id": meta.get("env_id", "unknown"),
                            "cameras": frames,
                        })
                if not parts:
                    await asyncio.sleep(interval_s)
                    continue
                payload = json.dumps({"envs": parts})

            # ── push to all connected SSE clients ─────────────────
            dead: list[asyncio.Queue] = []
            for q in list(queues):
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    dead.append(q)
            for q in dead:
                queues.discard(q)
        except Exception:
            pass
        await asyncio.sleep(interval_s)


def _collect_camera_frames_from_cache(sid: str, handle: str) -> list[dict]:
    """Collect frames from cached MCP-formatted observation.

    Cached obs already have ``rgb_base64`` and ``depth_base64`` in their
    camera dicts.  Both are forwarded to the dashboard.
    """
    obs = _session_last_obs.get(sid, {}).get(handle, {})
    cameras = obs.get("cameras", []) if isinstance(obs, dict) else []
    if not isinstance(cameras, list):
        cameras = []

    frames: list[dict] = []
    for cam in cameras:
        if not isinstance(cam, dict):
            continue
        b64 = cam.get("rgb_base64")
        if not b64:
            continue
        f = {
            "frame_id": cam.get("frame_id", "unknown"),
            "rgb_base64": b64,
            "width": cam.get("width", 0),
            "height": cam.get("height", 0),
        }
        # Include depth if available, plus min/max for display
        depth_b64 = cam.get("depth_base64")
        if depth_b64:
            f["depth_base64"] = depth_b64
            f["depth_min"] = cam.get("depth_min", 0)
            f["depth_max"] = cam.get("depth_max", 1)
        frames.append(f)
    return frames
