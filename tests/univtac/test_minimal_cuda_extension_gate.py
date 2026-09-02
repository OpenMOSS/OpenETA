from __future__ import annotations

from scripts.univtac import probe_minimal_cuda_extension as probe


def test_minimal_extension_uses_no_extra_include_dirs() -> None:
    source = __import__("inspect").getsource(probe.main)
    assert "extra_include_paths" in source
    assert "extra_include_paths=" not in source
    assert "cuda_runtime_api.h" in probe.CPP
    assert "add_one_kernel" in probe.CUDA
