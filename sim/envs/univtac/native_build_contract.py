"""Pure helpers for native CUDA build and binary evidence."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


ARCH_TOKENS = ("sm_120", "compute_120")
FORBIDDEN_FALLBACK_TOKENS = ("sm_89", "compute_89", "sm_90", "compute_90", "sm_120a")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_native_environment(environment: Mapping[str, str], target_prefix: Path) -> None:
    prefix = target_prefix.resolve()
    expected = {
        "CUDA_HOME": prefix,
        "CUDA_PATH": prefix,
        "CUDACXX": prefix / "bin" / "nvcc",
    }
    for key, path in expected.items():
        if Path(environment.get(key, "")).resolve() != path:
            raise ValueError(f"{key} must resolve inside the target environment")
    if environment.get("CMAKE_CUDA_ARCHITECTURES") != "120":
        raise ValueError("CMAKE_CUDA_ARCHITECTURES must be 120")
    if environment.get("TORCH_CUDA_ARCH_LIST") != "12.0":
        raise ValueError("TORCH_CUDA_ARCH_LIST must be 12.0")


def architecture_tokens(text: str) -> dict[str, list[str]]:
    observed = sorted(set(re.findall(r"(?:sm|compute)_[0-9]+[a-z]?", text)))
    return {
        "observed": observed,
        "required": [token for token in ARCH_TOKENS if token in observed],
        "forbidden_fallbacks": [token for token in FORBIDDEN_FALLBACK_TOKENS if token in observed],
    }


def binary_architecture_classification(records: Iterable[Mapping[str, Any]]) -> str:
    records = list(records)
    if not records:
        return "libuipc_binary_missing_sm120_code"
    has_required = any(item.get("has_sm120_or_compute120") for item in records)
    return "passed" if has_required else "libuipc_binary_missing_sm120_code"


def build_log_classification(text: str, *, returncode: int) -> str:
    lowered = text.lower()
    if returncode == 0:
        return "passed"
    compiler_markers = ("unsupported gpu architecture", "unsupported gpu arch", "value 'sm_120' is not defined")
    if any(marker in lowered for marker in compiler_markers):
        return "libuipc_sm120_compiler_rejected"
    return "libuipc_sm120_build_failed"


def source_tracked_clean(status_porcelain: str) -> bool:
    return not any(line and not line.startswith("??") for line in status_porcelain.splitlines())
