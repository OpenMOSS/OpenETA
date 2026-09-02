"""Shared runtime helpers for the R0.9 Blackwell simulator gates."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from sim.envs.univtac.cuda_include_bridge import bridge_environment
from sim.envs.univtac.resource_sanitation import run_managed_process, write_json
from sim.envs.univtac.source_backport_contract import compiler_wrapper_paths, validate_compiler_wrappers


PRESERVED_ENVIRONMENT = (
    "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "PATH", "TMPDIR",
    "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
)


def clean_runtime_environment(prefix: Path, pinned_source: Path, gpu: str = "0") -> tuple[dict[str, str], dict[str, Any]]:
    wrappers = validate_compiler_wrappers(pinned_source)
    paths = compiler_wrapper_paths(pinned_source)
    environment = {key: os.environ[key] for key in PRESERVED_ENVIRONMENT if key in os.environ}
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_LAUNCH_BLOCKING": "1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PATH": f"{prefix / 'bin'}:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": str(prefix / "lib"),
            "CC": str(paths["CC"]),
            "CXX": str(paths["CXX"]),
            "CUDAHOSTCXX": str(paths["CUDAHOSTCXX"]),
        }
    )
    environment = bridge_environment(environment, prefix)
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LD_PRELOAD", "TORCH_USE_RTLD_GLOBAL", "PYTHONPATH"):
        environment.pop(key, None)
    return environment, wrappers


def managed(
    command: Sequence[str], *, cwd: Path, output_root: Path, name: str,
    timeout_seconds: float, environment: Mapping[str, str]
) -> dict[str, Any]:
    result = run_managed_process(
        list(command), cwd=cwd, log_path=output_root / "logs" / f"{name}.log",
        timeout_seconds=timeout_seconds, environment=environment,
    )
    write_json(output_root / "processes" / f"{name}.json", result)
    return result


def process_ok(result: Mapping[str, Any]) -> bool:
    return result.get("returncode") == 0 and not result.get("timed_out") and result.get("cleanup_complete") is True


def require_process(name: str, result: Mapping[str, Any]) -> None:
    if not process_ok(result):
        raise RuntimeError(f"{name} failed; see {result.get('log_path')}")


def run_capture(command: Sequence[str], *, cwd: Path | None = None, environment: Mapping[str, str] | None = None, timeout: float = 300) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(command), cwd=cwd, env=dict(environment) if environment is not None else None,
            check=False, capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": list(command), "returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"command": list(command), "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def relative_artifact(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))
