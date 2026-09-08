"""Host supervision of a disposable, kernel-restricted Python worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import tempfile
import time


def execute_sandbox(payload: dict, *, cancel_event=None, max_output_bytes: int = 1_000_000,
                    max_result_bytes: int = 8_000_000) -> dict:
    """Bound wall time and all output streams; always kill/reap before return."""

    if cancel_event is not None and cancel_event.is_set():
        return _failure("python_exec_cancelled", "Execution cancelled before worker launch")
    started = time.monotonic()
    try:
        request = json.dumps(payload, allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        return _failure("python_exec_invalid_worker_input", str(exc))
    if len(request) > max_result_bytes:
        return _failure("python_exec_invalid_worker_input", "Worker request exceeds byte limit")
    worker = Path(__file__).with_name("python_sandbox_worker.py")
    read_fd, write_fd = os.pipe()
    os.set_inheritable(write_fd, True)
    buffers = {"stdout": bytearray(), "stderr": bytearray(), "result": bytearray()}
    process = None
    failure = None
    with tempfile.TemporaryDirectory(prefix="openeta-python-worker-") as private_tmp:
        # Never inherit API keys, proxy URLs, Python startup hooks, or host HOME.
        # Cache/temp dirs are writable only when a workspace was explicitly granted.
        workspace = payload["options"].get("workspace_root")
        env = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
               "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
               "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
               "TMPDIR": workspace or private_tmp,
               "MPLCONFIGDIR": str(Path(workspace or private_tmp) / ".matplotlib")}
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-B", str(worker), str(write_fd)],
                cwd=workspace or private_tmp, env=env, close_fds=True,
                pass_fds=(write_fd,), start_new_session=True,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            os.close(write_fd)
            write_fd = -1
            assert process.stdin and process.stdout and process.stderr
            with selectors.DefaultSelector() as selector:
                for handle, name in ((process.stdout, "stdout"), (process.stderr, "stderr"),
                                     (read_fd, "result")):
                    os.set_blocking(handle if isinstance(handle, int) else handle.fileno(), False)
                    selector.register(handle, selectors.EVENT_READ, name)
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdin, selectors.EVENT_WRITE, "input")
                sent = 0
                while selector.get_map() or process.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        failure = _failure("python_exec_cancelled", "Worker cancelled and terminated")
                        break
                    remaining = payload["timeout_s"] - (time.monotonic() - started)
                    if remaining <= 0:
                        failure = _failure("python_exec_timeout", "Worker exceeded wall-clock budget")
                        break
                    for key, _ in selector.select(min(0.025, remaining)):
                        if key.data == "input":
                            try:
                                sent += os.write(key.fd, request[sent:sent + 65536])
                            except BlockingIOError:
                                continue
                            except BrokenPipeError:
                                sent = len(request)
                            if sent >= len(request):
                                selector.unregister(key.fileobj)
                                process.stdin.close()
                            continue
                        try:
                            chunk = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        limit = max_result_bytes if key.data == "result" else max_output_bytes
                        buffer = buffers[key.data]
                        if len(buffer) + len(chunk) > limit:
                            buffer.extend(chunk[:max(0, limit - len(buffer))])
                            failure = _failure("python_exec_output_limit", f"{key.data} exceeds byte limit")
                            break
                        buffer.extend(chunk)
                    if failure:
                        break
            if failure is None:
                process.wait(timeout=max(0.01, payload["timeout_s"] - (time.monotonic() - started)))
        except (OSError, subprocess.TimeoutExpired) as exc:
            failure = _failure("python_exec_worker_failed", str(exc))
        finally:
            if process is not None:
                # Reap even on cancellation/error; no daemon handler owns a live
                # process after this function returns. Policy denies descendants.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            os.close(read_fd)
            if write_fd >= 0:
                os.close(write_fd)
    response = failure
    if response is None:
        try:
            response = json.loads(buffers["result"])
            if not isinstance(response, dict) or not isinstance(response.get("success"), bool):
                raise ValueError("Worker response lacks a boolean success value")
            if response["success"] is False and not isinstance(response.get("diagnostic"), dict):
                raise ValueError("Failed worker response lacks diagnostics")
            if process is None or process.returncode != 0:
                response = _failure("python_exec_worker_failed", "Worker exited abnormally")
        except (ValueError, UnicodeError, RecursionError) as exc:
            response = _failure("python_exec_worker_failed", f"No valid worker result: {exc}")
    response.update({"stdout": buffers["stdout"].decode(errors="replace"),
                     "stderr": buffers["stderr"].decode(errors="replace"),
                     "worker_pid": process.pid if process else None,
                     "returncode": process.returncode if process else None,
                     "elapsed_s": time.monotonic() - started})
    return response


def _failure(code: str, message: str) -> dict:
    return {"success": False, "diagnostic": {"code": code, "message": message,
                                               "error_type": "SandboxExecutionError"}}
