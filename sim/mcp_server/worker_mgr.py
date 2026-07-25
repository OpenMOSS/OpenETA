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
import subprocess
import threading
import time
import urllib.request
import urllib.error
from dataclasses import dataclass

from sim.mcp_server.session import (
    _get_mgr,
    _obs_key,
    _session_envs,
    _session_last_obs,
    _session_last_obs_lock,
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

    def proxy(self, method: str, path: str, body: dict | None = None) -> dict:
        """Forward an HTTP request to the worker and return parsed JSON."""
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        try:
            # Isaac Kit can spend several minutes compiling shaders on the
            # first BEHAVIOR create/reset. Keep this configurable while using
            # a timeout that does not kill a healthy cold start.
            timeout_s = float(os.environ.get("OPENETA_WORKER_HTTP_TIMEOUT_S", "600"))
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            try:
                return json.loads(exc.read().decode("utf-8"))
            except Exception:
                return {"error": f"HTTP {exc.code}: {exc.reason}"}
        except Exception as exc:
            return {"error": f"Worker request failed: {exc}"}

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
        return wh.proxy(method, path, body)

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

def _proxy_step(meta: dict, action, num_steps: int = 1, render: bool = True) -> dict:
    """Proxy a step request to the worker and cache the observation.

    When ``render`` is ``False`` the worker skips its inline camera render,
    which is the dominant per-step cost (~130 ms).  ``move_to`` uses this for
    its control loop since it only needs the EE pose from the result; the
    dashboard's background ``/render_all`` refresh keeps the live view current.
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
    result = mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/step", method="POST", body=body)
    # Cache observation for streaming
    obs = result.get("observation")
    if obs:
        sid = meta.get("_sid", "")
        key = _obs_key(meta)
        with _session_last_obs_lock:
            cache = _session_last_obs.setdefault(sid, {})
            # When the worker skipped rendering (render=False), the new obs
            # carries no camera frames.  Preserve the previously cached frames
            # so the dashboard doesn't flicker to blank between background
            # /render_all refreshes — only the robot/EE state need be current.
            prev = cache.get(key)
            if isinstance(prev, dict) and isinstance(obs, dict) and not obs.get("cameras"):
                prev_cams = prev.get("cameras")
                if prev_cams:
                    obs = {**obs, "cameras": prev_cams}
            cache[key] = obs
        # Physics advanced.  If this step rendered inline, the cached frame is
        # already current for this generation; otherwise mark dirty so the SSE
        # loop refreshes it.  (render=True here means the worker rendered as
        # part of the step and obs carries fresh camera frames.)
        if render and isinstance(obs, dict) and obs.get("cameras"):
            gen = _mark_obs_dirty(key)
            _mark_obs_rendered(key, gen)
        else:
            _mark_obs_dirty(key)
    return result


def _proxy_reset(meta: dict, seed: int | None = None) -> dict:
    """Proxy a reset request to the worker and cache the observation."""
    mgr = _get_mgr()
    body = {"seed": seed} if seed is not None else {}
    result = mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/reset", method="POST", body=body)
    key = _obs_key(meta)
    with _session_last_obs_lock:
        _session_last_obs.setdefault(meta.get("_sid", ""), {})[key] = result
    # reset re-initialises physics → cached frame is stale.  If the reset
    # result already carries camera frames it's current for this generation;
    # otherwise mark dirty for the SSE loop to refresh.
    obs = result.get("observation") if isinstance(result, dict) else None
    has_cams = isinstance(obs, dict) and bool(obs.get("cameras"))
    if not has_cams and isinstance(result, dict):
        has_cams = bool(result.get("cameras"))
    if has_cams:
        gen = _mark_obs_dirty(key)
        _mark_obs_rendered(key, gen)
    else:
        _mark_obs_dirty(key)
    return result


def _proxy_observe(meta: dict) -> dict:
    """Proxy an observe request to the worker."""
    mgr = _get_mgr()
    return mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/observe", method="POST")


def _proxy_render(meta: dict) -> dict:
    """Proxy a render request to the worker."""
    mgr = _get_mgr()
    return mgr.proxy_handle_op(meta, f"/env/{meta['remote_handle']}/render", method="POST")


def _proxy_render_all(worker_url: str, remote_handles: list[str]) -> dict:
    """Call the worker's ``/render_all`` endpoint for parallel batch rendering."""
    wh = BenchWorkerHandle(bench="", port=0, process=None, base_url=worker_url)
    return wh.proxy("POST", "/render_all", body={"handles": remote_handles})


# ══════════════════════════════════════════════════════════════════════
# Live-stream helpers
# ══════════════════════════════════════════════════════════════════════

# Thread pool for background render refreshes (offloads blocking HTTP from event loop)
_render_executor = None

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
        with _session_last_obs_lock:
            cache = _session_last_obs.setdefault(sid, {})
            for rh, obs in by_handle.items():
                if "error" not in obs:
                    # key by (this worker's url, remote_handle) so a handle from
                    # another worker can't clobber this one's cache slot.
                    cache[(wurl, rh)] = obs
    except Exception:
        pass


# ── Adaptive, self-throttling background-render pacing ────────────────
#
# A single libero render costs ~1.5 s and holds the worker's GIL, so it
# blocks any concurrent observe/move_to/step on the *same* worker process.
# The old loop fired /render_all every 500 ms regardless of how long a render
# actually took, so renders piled up back-to-back and pinned the worker at
# 100 % duty cycle — starving real tool calls (observe 36 s, move_to 126 s).
#
# Two guards fix this:
#   1. in-flight dedup: never fire a second render for a worker while one is
#      still running.
#   2. adaptive gap: after a render completes, don't fire the next one until at
#      least max(_SSE_MIN_REFRESH_GAP, measured_duration) has elapsed. A 1.5 s
#      render therefore yields ~1.5 s of idle, capping background render at a
#      ~50 % duty cycle and leaving the GIL free for tool calls.
_SSE_MIN_REFRESH_GAP = 0.5  # s — never refresh one worker more often than this
_worker_render_inflight: dict[str, bool] = {}
_worker_render_next_ok: dict[str, float] = {}
_worker_render_lock = threading.Lock()


# ── Dirty tracking: only render when physics actually advanced ────────
#
# env.render() is a pure read of the current MuJoCo state — it does NOT step
# physics.  Between two steps the scene is identical, so re-rendering an idle
# env just burns the worker's GIL/GPU producing a pixel-identical frame.  We
# track, per obs-key (worker_url, remote_handle), whether the cached camera
# frame still reflects the latest physics state:
#
#   * every step / reset bumps a "dirty" generation counter for the key;
#   * a completed background render records which generation it captured.
#
# A key is dirty iff its dirty-gen != the last rendered-gen.  A snapshot of the
# dirty-gen is taken *before* the (blocking) render HTTP call and only recorded
# as rendered on success, so a step landing mid-render correctly leaves the key
# dirty and it re-renders on the next tick (no lost update).
_obs_dirty_gen: dict[tuple[str, str], int] = {}
_obs_render_gen: dict[tuple[str, str], int] = {}
_obs_dirty_lock = threading.Lock()


def _mark_obs_dirty(key: tuple[str, str]) -> int:
    """Bump the dirty generation for *key* (physics advanced).  Returns new gen."""
    with _obs_dirty_lock:
        g = _obs_dirty_gen.get(key, 1) + 1
        _obs_dirty_gen[key] = g
        return g


def _mark_obs_rendered(key: tuple[str, str], gen: int) -> None:
    """Record that generation *gen* of *key* has been rendered into the cache."""
    with _obs_dirty_lock:
        if gen > _obs_render_gen.get(key, 0):
            _obs_render_gen[key] = gen


def _obs_is_dirty(key: tuple[str, str]) -> bool:
    """True if *key*'s cached frame is stale (physics advanced since last render)."""
    with _obs_dirty_lock:
        return _obs_dirty_gen.get(key, 1) != _obs_render_gen.get(key, 0)


def _worker_any_dirty(wurl: str, remote_handles: list[str]) -> bool:
    """True if any handle on *wurl* has a stale cached frame."""
    with _obs_dirty_lock:
        for rh in remote_handles:
            key = (wurl, rh)
            if _obs_dirty_gen.get(key, 1) != _obs_render_gen.get(key, 0):
                return True
    return False


def _forget_obs_dirty(key: tuple[str, str]) -> None:
    """Drop dirty-tracking state for a closed env's key."""
    with _obs_dirty_lock:
        _obs_dirty_gen.pop(key, None)
        _obs_render_gen.pop(key, None)


def _try_reserve_worker_render(wurl: str) -> bool:
    """Atomically claim the right to render *wurl* now.

    Returns True and marks the worker in-flight iff no render is currently
    running for it AND enough time has elapsed since the last one finished.
    The caller MUST run _refresh_cache_for_worker_guarded (which clears the
    flag) if this returns True.  Doing the check-and-set under one lock closes
    the window where two consecutive SSE ticks both fire for the same worker.
    """
    with _worker_render_lock:
        if _worker_render_inflight.get(wurl):
            return False
        if time.monotonic() < _worker_render_next_ok.get(wurl, 0.0):
            return False
        _worker_render_inflight[wurl] = True
        return True


def _refresh_cache_for_worker_guarded(wurl: str, remote_hs: list[str], sid: str) -> None:
    """Self-pacing wrapper around _refresh_cache_for_worker.

    Runs inside the render thread-pool.  Assumes the caller already reserved
    the worker via _try_reserve_worker_render (in-flight flag is set).  Records
    the render duration and pushes the next-allowed time forward so the SSE
    loop paces itself to the worker's real rendering speed instead of a fixed
    500 ms assumption.

    Dirty tracking: the dirty generation of each handle is snapshotted *before*
    the blocking render.  On success those generations are recorded as
    rendered.  A step that lands mid-render bumps the gen again, so the key
    stays dirty and re-renders next tick — no lost update.
    """
    # Snapshot dirty-gens before rendering.
    with _obs_dirty_lock:
        pre_gen = {rh: _obs_dirty_gen.get((wurl, rh), 1) for rh in remote_hs}
    t0 = time.monotonic()
    try:
        _refresh_cache_for_worker(wurl, remote_hs, sid)
        for rh, g in pre_gen.items():
            _mark_obs_rendered((wurl, rh), g)
    finally:
        dt = time.monotonic() - t0
        with _worker_render_lock:
            _worker_render_inflight[wurl] = False
            _worker_render_next_ok[wurl] = time.monotonic() + max(_SSE_MIN_REFRESH_GAP, dt)


async def _live_stream_loop(sid: str, interval_s: float, stream_key: str = "", handle: str = "") -> None:
    """Push camera frames to SSE queues at *interval_s*.

    Reads from the local ``_session_last_obs`` cache directly (updated by
    ``_proxy_step`` / ``_proxy_reset`` after every env operation), so frames
    are pushed **immediately** during ``move_to`` without any blocking HTTP.

    Every ~500 ms a background thread-pool task asks each worker for a fresh
    render (``/render_all``) so the dashboard stays live even when no one is
    stepping.
    """
    import concurrent.futures

    sk = stream_key or sid
    loop = asyncio.get_running_loop()
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

            # ── background refresh (non-blocking, self-pacing) ────
            #
            # We still *poll* every _REFRESH_EVERY_N ticks, but each worker is
            # gated by _should_refresh_worker(): a render only fires if none is
            # in flight for that worker AND enough time has elapsed since the
            # last one finished (>= its measured duration).  This keeps a slow
            # worker (~1.5 s/render) from being flooded while a fast one still
            # refreshes promptly.
            if tick % _REFRESH_EVERY_N == 0:
                if handle:
                    meta = env_dict.get(handle)
                    if meta is not None:
                        wurl = meta["worker_url"]
                        rh = meta["remote_handle"]
                        # Only render if physics advanced since the last frame.
                        if _worker_any_dirty(wurl, [rh]) and _try_reserve_worker_render(wurl):
                            loop.run_in_executor(
                                _get_render_executor(),
                                _refresh_cache_for_worker_guarded,
                                wurl, [rh], sid,
                            )
                else:
                    # Group handles by worker, fire one refresh per worker
                    by_worker: dict[str, list[str]] = {}
                    for _h, meta in list(env_dict.items()):
                        wurl = meta["worker_url"]
                        by_worker.setdefault(wurl, []).append(meta["remote_handle"])
                    for wurl, remote_hs in by_worker.items():
                        # Skip workers whose frames are all still current.
                        if _worker_any_dirty(wurl, remote_hs) and _try_reserve_worker_render(wurl):
                            loop.run_in_executor(
                                _get_render_executor(),
                                _refresh_cache_for_worker_guarded,
                                wurl, remote_hs, sid,
                            )
            tick += 1

            # ── build payload from local cache (instant, no I/O) ──
            if handle:
                frames = _collect_camera_frames_from_cache(sid, _obs_key(env_dict[handle]))
                if not frames:
                    await asyncio.sleep(interval_s)
                    continue
                payload = json.dumps({"handle": handle, "cameras": frames})
            else:
                parts: list[dict] = []
                for h, meta in list(env_dict.items()):
                    frames = _collect_camera_frames_from_cache(sid, _obs_key(meta))
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


def _collect_camera_frames_from_cache(sid: str, key: tuple[str, str]) -> list[dict]:
    """Collect frames from cached MCP-formatted observation.

    ``key`` is the composite (worker_url, remote_handle) cache key.  Cached obs
    already have ``rgb_base64`` and ``depth_base64`` in their camera dicts;
    both are forwarded to the dashboard.
    """
    with _session_last_obs_lock:
        obs = _session_last_obs.get(sid, {}).get(key, {})
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
