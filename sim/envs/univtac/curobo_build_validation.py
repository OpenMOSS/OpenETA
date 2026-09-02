"""Validation helpers for the pinned cuRobo native gates."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

EXTENSIONS = (
    "lbfgs_step_cu",
    "kinematics_fused_cu",
    "line_search_cu",
    "tensor_step_cu",
    "geom_cu",
)


def classify_build_failure(log: str) -> str:
    lower = log.lower()
    missing = re.findall(r"fatal error: ([^:]+): no such file or directory", lower)
    if "cuda_runtime_api.h" in missing:
        return "curobo_same_cuda_header_failure"
    if any(name.startswith(("cuda", "crt/")) for name in missing):
        return "curobo_different_cuda_header_failure"
    if any(token in lower for token in ("cannot find -lcudart", "undefined reference", "ld returned")):
        return "curobo_link_library_layout_failure"
    if "nvcc" in lower and any(token in lower for token in ("error:", "failed:")):
        return "curobo_nvcc_sm120_compile_failure"
    if any(token in lower for token in ("failed-wheel", "editable_wheel", "packaging")):
        return "curobo_packaging_failure"
    return "curobo_host_compile_failure"


def validate_extension_names(paths: Iterable[str]) -> None:
    names = {name for path in paths for name in EXTENSIONS if name in path}
    if names != set(EXTENSIONS):
        raise ValueError(f"expected all five cuRobo extensions, found {sorted(names)}")


def find_extension_binaries(checkout: Path) -> list[Path]:
    root = checkout / "src/curobo/curobolib"
    binaries = []
    for name in EXTENSIONS:
        matches = sorted(root.glob(f"{name}.*.so"))
        if len(matches) != 1:
            raise ValueError(f"expected one binary for {name}, found {len(matches)}")
        binaries.append(matches[0].resolve())
    return binaries
