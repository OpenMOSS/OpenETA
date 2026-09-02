"""Contract for the Conda CUDA target include bridge."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

RUNTIME_ADAPTATION = "conda_cuda_target_include_bridge_v1"
FORBIDDEN_INCLUDE_ENV = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH")


def bridge_environment(base: Mapping[str, str], prefix: Path) -> dict[str, str]:
    prefix = prefix.resolve()
    target_include = (prefix / "targets/x86_64-linux/include").resolve()
    nvcc = (prefix / "bin/nvcc").resolve()
    if not target_include.is_dir() or not nvcc.is_file():
        raise ValueError("target Conda CUDA layout is incomplete")
    environment = dict(base)
    for key in FORBIDDEN_INCLUDE_ENV:
        environment.pop(key, None)
    environment.update(
        {
            "CUDA_HOME": str(prefix),
            "CUDA_PATH": str(prefix),
            "CUDACXX": str(nvcc),
            "CUDA_INC_PATH": str(target_include),
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "CMAKE_CUDA_ARCHITECTURES": "120",
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "MAX_JOBS": "4",
            "CMAKE_BUILD_PARALLEL_LEVEL": "4",
        }
    )
    return environment


def validate_include_resolution(before: Mapping[str, Any], after: Mapping[str, Any], prefix: Path) -> None:
    expected_home = str(prefix.resolve())
    target_include = str((prefix / "targets/x86_64-linux/include").resolve())
    if before.get("cuda_home") != expected_home or after.get("cuda_home") != expected_home:
        raise ValueError("CUDA_HOME escaped the target Conda prefix")
    if target_include in before.get("include_paths", []):
        raise ValueError("target include unexpectedly present before bridge")
    if target_include not in after.get("include_paths", []):
        raise ValueError("cuda include bridge not recognized")
    if after.get("cuda_inc_path") != target_include:
        raise ValueError("CUDA_INC_PATH is not the target include")


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path.resolve()), "exists": False}
    data = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "is_symlink": path.is_symlink(),
    }
