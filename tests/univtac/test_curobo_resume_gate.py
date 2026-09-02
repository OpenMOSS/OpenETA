from __future__ import annotations

import inspect

from scripts.univtac import resume_curobo_sm120 as runner
from sim.envs.univtac.curobo_build_validation import (
    EXTENSIONS,
    classify_build_failure,
    find_extension_binaries,
)


def test_failure_classification_separates_header_and_linker() -> None:
    assert classify_build_failure("fatal error: cuda_runtime_api.h: No such file or directory") == "curobo_same_cuda_header_failure"
    assert classify_build_failure("cannot find -lcudart") == "curobo_link_library_layout_failure"


def test_all_five_extensions_are_required() -> None:
    assert len(EXTENSIONS) == 5
    assert set(EXTENSIONS) == {"lbfgs_step_cu", "kinematics_fused_cu", "line_search_cu", "tensor_step_cu", "geom_cu"}


def test_extension_discovery_excludes_unrelated_cuda_libraries(tmp_path) -> None:
    root = tmp_path / "src/curobo/curobolib"
    root.mkdir(parents=True)
    for name in EXTENSIONS:
        (root / f"{name}.cpython.so").write_text("", encoding="utf-8")
    (root / "libtorch_cuda.so").write_text("", encoding="utf-8")
    assert len(find_extension_binaries(tmp_path)) == 5


def test_runner_gate_order_and_single_build() -> None:
    source = inspect.getsource(runner.main)
    assert source.index('manifest["gates"]["I0"]') < source.index('manifest["gates"]["X0"]')
    assert source.index('manifest["gates"]["X0"]') < source.index('manifest["gates"]["C0_BUILD"]')
    assert source.index('manifest["gates"]["C0_BUILD"]') < source.index('manifest["gates"]["C1"]')
    assert source.index('manifest["gates"]["C1"]') < source.index('manifest["gates"]["C2"]') < source.index('manifest["gates"]["C3"]') < source.index('manifest["gates"]["U2"]')
    assert source.count('name="build"') == 1
    assert '"--resume-from-passed-x0"' in source


def test_runner_does_not_start_isaac_or_rebuild_libuipc() -> None:
    source = inspect.getsource(runner)
    assert "AppLauncher" not in source
    assert "pip install --no-build-isolation -e tacex_uipc" not in source
    assert "sysctl -w" not in source and '"sudo"' not in source


def test_binary_audit_failure_generates_runtime_bundle(tmp_path) -> None:
    audit = tmp_path / "curobo_binary_audit/audit.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(
        '{"records": [{"path": "geom_cu.so"}], "missing_dependencies": ["libtorch.so => not found"]}',
        encoding="utf-8",
    )
    manifest = {"classification": "curobo_sm120_binary_audit_failed", "gates": {}}
    result = runner.finalize_summary(tmp_path, manifest)
    assert result["author_bundle_type"] == "runtime"
    assert result["c1_failure_detail"] == "raw_ldd_missing_torch_shared_libraries"
    assert result["c1_extension_record_count"] == 1
    assert (tmp_path / "author_bundle_curobo/runtime/summary.json").is_file()
