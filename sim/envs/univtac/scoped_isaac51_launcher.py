"""Repository-scoped launcher for Isaac51 child processes.

This control-plane module intentionally imports only the Python standard
library.  It prepares a private ``libcuda.so`` alias for the child process and
never mutates the parent environment.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ScopedIsaac51LaunchError(RuntimeError):
    """Raised when the scoped runtime cannot be prepared or launched."""


@dataclass(frozen=True)
class LibcudaDriverProbe:
    mapped_path: Path
    cu_init_returncode: int
    cu_device_count_returncode: int
    cu_device_count: int
    cu_driver_get_version_returncode: int
    cu_driver_version: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mapped_path"] = str(self.mapped_path)
        return payload


@dataclass(frozen=True)
class ScopedIsaac51LaunchSpec:
    python_executable: Path
    command: tuple[str, ...]
    cwd: Path
    output_root: Path
    timeout_seconds: float
    environment_overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopedIsaac51LaunchResult:
    alias_path: Path
    alias_target: Path
    child_environment_summary: dict[str, Any]
    command: tuple[str, ...]
    cwd: Path
    root_pid: int
    process_group_id: int
    returncode: int
    timed_out: bool
    sigterm_sent: bool
    sigkill_sent: bool
    cleanup_complete: bool
    final_process_group_members: tuple[int, ...]
    stdout_stderr_path: Path
    elapsed_seconds: float
    kit_log_path: Path | None
    mapped_libcuda_paths: tuple[str, ...]
    stub_mapped: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias_path": str(self.alias_path),
            "alias_target": str(self.alias_target),
            "child_environment_summary": self.child_environment_summary,
            "command": list(self.command),
            "cwd": str(self.cwd),
            "root_pid": self.root_pid,
            "process_group_id": self.process_group_id,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "sigterm_sent": self.sigterm_sent,
            "sigkill_sent": self.sigkill_sent,
            "cleanup_complete": self.cleanup_complete,
            "final_process_group_members": list(self.final_process_group_members),
            "stdout_stderr_path": str(self.stdout_stderr_path),
            "elapsed_seconds": self.elapsed_seconds,
            "kit_log_path": str(self.kit_log_path) if self.kit_log_path else None,
            "mapped_libcuda_paths": list(self.mapped_libcuda_paths),
            "stub_mapped": self.stub_mapped,
        }


_LOADER_PROBE = r"""
import ctypes
import json
from pathlib import Path

library = ctypes.CDLL(LIBRARY_NAME)
cu_init = library.cuInit
cu_init.argtypes = [ctypes.c_uint]
cu_init.restype = ctypes.c_int
cu_driver_get_version = library.cuDriverGetVersion
cu_driver_get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
cu_driver_get_version.restype = ctypes.c_int
cu_device_get_count = library.cuDeviceGetCount
cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
cu_device_get_count.restype = ctypes.c_int

init_rc = int(cu_init(0))
version = ctypes.c_int()
version_rc = int(cu_driver_get_version(ctypes.byref(version)))
count = ctypes.c_int()
count_rc = int(cu_device_get_count(ctypes.byref(count)))
paths = []
for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
    fields = line.split()
    if fields and fields[-1].startswith("/") and "/libcuda.so" in fields[-1]:
        paths.append(str(Path(fields[-1]).resolve()))
print(json.dumps({
    "mapped_paths": sorted(set(paths)),
    "cu_init_returncode": init_rc,
    "cu_device_count_returncode": count_rc,
    "cu_device_count": int(count.value),
    "cu_driver_get_version_returncode": version_rc,
    "cu_driver_version": int(version.value),
}, sort_keys=True))
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_real_libcuda_driver_path(
    path: str | Path,
    *,
    allowed_roots: Sequence[Path] = (Path("/usr/lib"), Path("/lib")),
    conda_prefix: str | Path | None = None,
    output_root: str | Path | None = None,
) -> Path:
    """Return a resolved system driver path or reject a non-driver candidate."""

    candidate = Path(path).expanduser().resolve()
    lowered = candidate.as_posix().lower()
    if "/stubs/" in lowered or "cuda" in lowered and "/usr/local/" in lowered:
        raise ScopedIsaac51LaunchError(f"toolkit or stub libcuda is forbidden: {candidate}")
    if conda_prefix is not None and candidate.is_relative_to(Path(conda_prefix).resolve()):
        raise ScopedIsaac51LaunchError(f"Conda libcuda is forbidden: {candidate}")
    if output_root is not None and candidate.is_relative_to(Path(output_root).resolve()):
        raise ScopedIsaac51LaunchError(f"output-local copied libcuda is forbidden: {candidate}")
    if not any(candidate.is_relative_to(root.resolve()) for root in allowed_roots):
        raise ScopedIsaac51LaunchError(f"libcuda is outside system library roots: {candidate}")
    if not candidate.is_file() or not candidate.name.startswith("libcuda.so."):
        raise ScopedIsaac51LaunchError(f"resolved libcuda driver is invalid: {candidate}")
    return candidate


def _run_loader_probe(
    *,
    python_executable: Path,
    library_name: str,
    environment: Mapping[str, str] | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> LibcudaDriverProbe:
    program = f"LIBRARY_NAME = {library_name!r}\n" + _LOADER_PROBE
    completed = run(
        [str(python_executable), "-c", program],
        env=dict(environment) if environment is not None else None,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise ScopedIsaac51LaunchError(
            f"libcuda loader probe failed ({completed.returncode}): {completed.stderr.strip()}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ScopedIsaac51LaunchError("libcuda loader probe returned invalid JSON") from exc
    mapped_paths = payload.get("mapped_paths", [])
    if len(mapped_paths) != 1:
        raise ScopedIsaac51LaunchError(f"expected one mapped libcuda driver, got {mapped_paths!r}")
    probe = LibcudaDriverProbe(
        mapped_path=Path(mapped_paths[0]).resolve(),
        cu_init_returncode=int(payload["cu_init_returncode"]),
        cu_device_count_returncode=int(payload["cu_device_count_returncode"]),
        cu_device_count=int(payload["cu_device_count"]),
        cu_driver_get_version_returncode=int(payload["cu_driver_get_version_returncode"]),
        cu_driver_version=int(payload["cu_driver_version"]),
    )
    if probe.cu_init_returncode != 0 or probe.cu_device_count_returncode != 0:
        raise ScopedIsaac51LaunchError(f"CUDA driver initialization failed: {probe.to_dict()}")
    if probe.cu_device_count < 1:
        raise ScopedIsaac51LaunchError("CUDA driver reported no visible device")
    return probe


def resolve_real_libcuda_driver(
    *,
    python_executable: Path | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> LibcudaDriverProbe:
    """Resolve ``libcuda.so.1`` through a short independent loader process."""

    probe = _run_loader_probe(
        python_executable=(python_executable or Path(sys.executable)).resolve(),
        library_name="libcuda.so.1",
        run=run,
    )
    target = validate_real_libcuda_driver_path(
        probe.mapped_path,
        conda_prefix=os.environ.get("CONDA_PREFIX"),
    )
    return LibcudaDriverProbe(
        mapped_path=target,
        cu_init_returncode=probe.cu_init_returncode,
        cu_device_count_returncode=probe.cu_device_count_returncode,
        cu_device_count=probe.cu_device_count,
        cu_driver_get_version_returncode=probe.cu_driver_get_version_returncode,
        cu_driver_version=probe.cu_driver_version,
    )


def prepare_process_local_libcuda_alias(
    output_root: Path,
    driver_target: Path,
    *,
    allowed_roots: Sequence[Path] = (Path("/usr/lib"), Path("/lib")),
) -> Path:
    """Create the sole private alias entry under one fresh runtime directory."""

    output_root = output_root.expanduser().resolve()
    target = validate_real_libcuda_driver_path(
        driver_target,
        allowed_roots=allowed_roots,
        conda_prefix=os.environ.get("CONDA_PREFIX"),
        output_root=output_root,
    )
    alias_dir = output_root / "runtime" / "libcuda_alias"
    if alias_dir.exists():
        raise ScopedIsaac51LaunchError(f"libcuda alias directory is not fresh: {alias_dir}")
    alias_dir.mkdir(parents=True, mode=0o700)
    os.chmod(alias_dir, 0o700)
    alias_path = alias_dir / "libcuda.so"
    alias_path.symlink_to(target)
    entries = list(alias_dir.iterdir())
    if entries != [alias_path] or not alias_path.is_symlink():
        raise ScopedIsaac51LaunchError("libcuda alias directory contract was not met")
    return alias_path


def build_scoped_isaac51_child_environment(
    *,
    parent_environment: Mapping[str, str],
    alias_dir: Path,
    environment_overrides: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Copy the parent environment and apply only scoped child settings."""

    parent = dict(parent_environment)
    child = parent.copy()
    if environment_overrides:
        child.update({str(key): str(value) for key, value in environment_overrides.items()})
    parent_cuda_visible_devices = parent.get("CUDA_VISIBLE_DEVICES")
    child.pop("CUDA_VISIBLE_DEVICES", None)
    child.pop("LD_PRELOAD", None)
    parent_ld_library_path = parent.get("LD_LIBRARY_PATH", "")
    child["LD_LIBRARY_PATH"] = (
        f"{alias_dir}:{parent_ld_library_path}" if parent_ld_library_path else str(alias_dir)
    )
    child.update(
        {
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTHONFAULTHANDLER": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_HOME": parent.get("CUDA_HOME", "/usr/local/cuda-12.8"),
        }
    )
    summary = {
        "parent_ld_library_path": parent_ld_library_path,
        "child_ld_library_path": child["LD_LIBRARY_PATH"],
        "parent_cuda_visible_devices": parent_cuda_visible_devices,
        "child_cuda_visible_devices": child.get("CUDA_VISIBLE_DEVICES"),
        "alias_dir": str(alias_dir),
        "ld_preload_present_in_child": "LD_PRELOAD" in child,
    }
    return child, summary


def _process_group_members(process_group_id: int) -> tuple[int, ...]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            fields = raw[raw.rfind(")") + 2 :].split()
            if int(fields[2]) == process_group_id:
                members.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            continue
    return tuple(sorted(members))


def _mapped_libcuda_paths(pid: int) -> tuple[str, ...]:
    try:
        lines = Path(f"/proc/{pid}/maps").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    paths: set[str] = set()
    for line in lines:
        fields = line.split()
        if fields and fields[-1].startswith("/") and "/libcuda.so" in fields[-1]:
            paths.add(str(Path(fields[-1]).resolve()))
    return tuple(sorted(paths))


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(dict(payload), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _discover_kit_log(text: str) -> Path | None:
    match = re.search(r"Logging to file:\s*['\"]?([^'\"\r\n]+)", text)
    return Path(match.group(1).strip()) if match else None


def _preserve_kit_log(launcher_dir: Path, stdout_path: Path) -> Path | None:
    text = stdout_path.read_text(encoding="utf-8", errors="replace")
    source = _discover_kit_log(text)
    if source is None or not source.is_file():
        return None
    destination = launcher_dir / "kit.log"
    shutil.copy2(source, destination)
    kit_text = destination.read_text(encoding="utf-8", errors="replace")
    tokens = (
        "omni.physx handle on CUDA lib",
        "GPU solver is not supported",
        "GPU BP is not supported",
        "Switching to Software mode",
        "libcuda",
        "libnvidia-ml",
        "error",
        "fatal",
    )
    relevant = [
        line for line in kit_text.splitlines() if any(t.lower() in line.lower() for t in tokens)
    ]
    (launcher_dir / "kit_relevant.txt").write_text(
        "\n".join(relevant) + ("\n" if relevant else ""), encoding="utf-8"
    )
    return destination


def build_scoped_isaac51_dry_run(
    spec: ScopedIsaac51LaunchSpec,
    driver_probe: LibcudaDriverProbe,
) -> dict[str, Any]:
    """Describe a launch without creating an alias or starting the child."""

    alias_dir = spec.output_root.expanduser().resolve() / "runtime" / "libcuda_alias"
    _, environment_summary = build_scoped_isaac51_child_environment(
        parent_environment=os.environ,
        alias_dir=alias_dir,
        environment_overrides=spec.environment_overrides,
    )
    environment_summary["alias_target"] = str(driver_probe.mapped_path)
    return {
        "dry_run": True,
        "resolved_driver": driver_probe.to_dict(),
        "alias_plan": {
            "alias_path": str(alias_dir / "libcuda.so"),
            "alias_target": str(driver_probe.mapped_path),
        },
        "child_environment_summary": environment_summary,
        "command": [str(spec.python_executable), *spec.command],
        "cwd": str(spec.cwd),
        "timeout_seconds": spec.timeout_seconds,
    }


def run_scoped_isaac51_command(spec: ScopedIsaac51LaunchSpec) -> ScopedIsaac51LaunchResult:
    """Run one child in an isolated process group with the private driver alias."""

    output_root = spec.output_root.expanduser().resolve()
    launcher_dir = output_root / "launcher"
    if launcher_dir.exists() or (output_root / "runtime" / "libcuda_alias").exists():
        raise ScopedIsaac51LaunchError("launcher output paths must be fresh")
    launcher_dir.mkdir(parents=True)
    parent_before = dict(os.environ)
    driver_probe = resolve_real_libcuda_driver(python_executable=spec.python_executable)
    alias_path = prepare_process_local_libcuda_alias(output_root, driver_probe.mapped_path)
    child_environment, environment_summary = build_scoped_isaac51_child_environment(
        parent_environment=parent_before,
        alias_dir=alias_path.parent,
        environment_overrides=spec.environment_overrides,
    )
    environment_summary["alias_target"] = str(driver_probe.mapped_path)
    alias_probe = _run_loader_probe(
        python_executable=spec.python_executable,
        library_name="libcuda.so",
        environment=child_environment,
    )
    alias_probe_target = validate_real_libcuda_driver_path(alias_probe.mapped_path)
    if alias_probe_target != driver_probe.mapped_path:
        raise ScopedIsaac51LaunchError(
            f"private alias mapped unexpected driver: {alias_probe_target}"
        )
    _write_json(
        launcher_dir / "alias_preflight.json",
        {
            "resolved_driver": driver_probe.to_dict(),
            "alias_probe": alias_probe.to_dict(),
            "alias_path": str(alias_path),
            "alias_target": str(driver_probe.mapped_path),
            "alias_directory_entries": [entry.name for entry in alias_path.parent.iterdir()],
            "alias_directory_mode": oct(alias_path.parent.stat().st_mode & 0o777),
        },
    )

    command = (str(spec.python_executable), *spec.command)
    stdout_path = launcher_dir / "stdout_stderr.log"
    samples_path = launcher_dir / "process_samples.jsonl"
    started = time.monotonic()
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    with stdout_path.open("w", encoding="utf-8", buffering=1) as stream:
        stream.write("COMMAND: " + " ".join(command) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        process = subprocess.Popen(
            command,
            cwd=spec.cwd,
            env=child_environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        process_group_id = os.getpgid(process.pid)
        deadline = started + spec.timeout_seconds
        observed_libcuda_paths: set[str] = set()
        while process.poll() is None and time.monotonic() < deadline:
            mapped = _mapped_libcuda_paths(process.pid)
            observed_libcuda_paths.update(mapped)
            _append_jsonl(
                samples_path,
                {
                    "utc_timestamp": _utc_now(),
                    "elapsed_seconds": time.monotonic() - started,
                    "root_pid": process.pid,
                    "process_group_id": process_group_id,
                    "process_group_members": list(_process_group_members(process_group_id)),
                    "mapped_libcuda_paths": list(mapped),
                    "stub_mapped": any("/stubs/" in path for path in mapped),
                },
            )
            try:
                process.wait(timeout=min(5.0, max(0.0, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
        observed_libcuda_paths.update(_mapped_libcuda_paths(process.pid))
        if process.poll() is None:
            timed_out = True
            sigterm_sent = True
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                sigkill_sent = True
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=15)

    lingering = _process_group_members(process_group_id)
    if lingering:
        sigterm_sent = True
        try:
            os.killpg(process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            pass
        cleanup_deadline = time.monotonic() + 15
        while lingering and time.monotonic() < cleanup_deadline:
            time.sleep(1)
            lingering = _process_group_members(process_group_id)
    if lingering:
        sigkill_sent = True
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(1)
    final_members = _process_group_members(process_group_id)
    kit_log_path = _preserve_kit_log(launcher_dir, stdout_path)
    if dict(os.environ) != parent_before:
        raise ScopedIsaac51LaunchError("parent environment changed during scoped launch")
    result = ScopedIsaac51LaunchResult(
        alias_path=alias_path,
        alias_target=driver_probe.mapped_path,
        child_environment_summary=environment_summary,
        command=command,
        cwd=spec.cwd.resolve(),
        root_pid=process.pid,
        process_group_id=process_group_id,
        returncode=int(process.returncode),
        timed_out=timed_out,
        sigterm_sent=sigterm_sent,
        sigkill_sent=sigkill_sent,
        cleanup_complete=not final_members,
        final_process_group_members=final_members,
        stdout_stderr_path=stdout_path,
        elapsed_seconds=time.monotonic() - started,
        kit_log_path=kit_log_path,
        mapped_libcuda_paths=tuple(sorted(observed_libcuda_paths)),
        stub_mapped=any("/stubs/" in path for path in observed_libcuda_paths),
    )
    _write_json(launcher_dir / "lifecycle.json", result.to_dict())
    return result
