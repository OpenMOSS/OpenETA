from __future__ import annotations

from pathlib import Path

import pytest

from sim.envs.univtac.native_build_contract import (
    architecture_tokens,
    binary_architecture_classification,
    build_log_classification,
    source_tracked_clean,
    validate_native_environment,
)


def _environment(prefix: Path):
    return {
        "CUDA_HOME": str(prefix),
        "CUDA_PATH": str(prefix),
        "CUDACXX": str(prefix / "bin/nvcc"),
        "CMAKE_CUDA_ARCHITECTURES": "120",
        "TORCH_CUDA_ARCH_LIST": "12.0",
    }


def test_native_build_must_use_target_environment_nvcc(tmp_path: Path) -> None:
    validate_native_environment(_environment(tmp_path), tmp_path)
    changed = _environment(tmp_path)
    changed["CUDACXX"] = "/usr/local/cuda-12.8/bin/nvcc"
    with pytest.raises(ValueError, match="CUDACXX"):
        validate_native_environment(changed, tmp_path)


def test_architecture_audit_requires_sm120_or_compute120() -> None:
    tokens = architecture_tokens("-gencode arch=compute_120,code=sm_120")
    assert set(tokens["required"]) == {"compute_120", "sm_120"}
    assert tokens["forbidden_fallbacks"] == []
    assert binary_architecture_classification([{"has_sm120_or_compute120": True}]) == "passed"
    assert binary_architecture_classification([]) == "libuipc_binary_missing_sm120_code"


def test_arch89_and_arch90_are_reported_as_forbidden_fallbacks() -> None:
    tokens = architecture_tokens("sm_89 compute_90 sm_120a")
    assert set(tokens["forbidden_fallbacks"]) == {"compute_90", "sm_120a", "sm_89"}


def test_compiler_rejection_is_distinct_from_generic_build_failure() -> None:
    assert build_log_classification("Unsupported gpu architecture sm_120", returncode=1) == "libuipc_sm120_compiler_rejected"
    assert build_log_classification("link failed", returncode=1) == "libuipc_sm120_build_failed"
    assert build_log_classification("ok", returncode=0) == "passed"


def test_ignored_untracked_builds_do_not_count_as_tracked_source_diff() -> None:
    assert source_tracked_clean("?? ignored-build/") is True
    assert source_tracked_clean(" M tracked.py") is False
