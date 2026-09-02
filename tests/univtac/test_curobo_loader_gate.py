from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

from scripts.univtac import audit_pytorch_extension_loader as probe
from scripts.univtac import probe_curobo_imports as combined_probe
from scripts.univtac import resume_curobo_loader_validation as runner


def test_probe_imports_torch_before_curobo_and_extension() -> None:
    source = inspect.getsource(probe.main)
    assert source.index("import torch") < source.index("import curobo") < source.index("importlib.import_module")


def test_probe_does_not_modify_loader_flags_or_preload_libraries() -> None:
    source = inspect.getsource(probe)
    assert "sys.setdlopenflags" not in source
    assert 'os.environ["LD_PRELOAD"]' not in source
    assert 'os.environ["TORCH_USE_RTLD_GLOBAL"]' not in source
    assert "ctypes.CDLL(args" not in source
    assert "from sim." not in source
    assert '"--target-prefix"' in source


def test_runner_gate_order_and_no_build_or_install() -> None:
    source = inspect.getsource(runner.main)
    for left, right in zip(("L0", "L1", "L2", "L3"), ("L1", "L2", "L3", "C2"), strict=True):
        assert source.index(f'["gates"]["{left}"]') < source.index(f'["gates"]["{right}"]')
    assert source.index('manifest["gates"]["C2"]') < source.index("kinematics_example.py")
    assert source.index("collision_check_example.py") < source.index('manifest["gates"]["U2"]')
    assert "pip install" not in source and "conda install" not in source
    assert "AppLauncher" not in source
    assert '"--resume-from-l0"' in source


def test_each_extension_uses_a_managed_process() -> None:
    source = inspect.getsource(runner.main)
    assert "for name, binary in zip(EXTENSIONS, binaries" in source
    assert 'root = output / "extension_loader" / name' in source


def test_combined_probe_records_provenance_memory_and_legacy_paths() -> None:
    source = inspect.getsource(combined_probe.main)
    assert 'parser.add_argument("--expected-root"' in source
    assert 'payload["gpu_memory"]' in source
    assert 'payload["loaded_objects"]' in source
    assert 'payload["legacy_paths"]' in source


def test_early_conda_failure_is_preserved_in_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "runtime_variant: test\ndiagnostic_method: test\nenvironment:\n  conda_name: missing\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "resume_curobo_loader_validation.py",
            "--config",
            str(config),
            "--r082-output",
            str(tmp_path / "r082"),
            "--source",
            str(tmp_path / "source"),
            "--curobo",
            str(tmp_path / "curobo"),
            "--output-root",
            str(output),
            "--conda-exe",
            str(tmp_path / "missing-conda"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        runner.main()
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failure"]["class"] == "FileNotFoundError"
    assert summary["status"] == "failed"
    assert {item["stage"] for item in summary["finalization_errors"]} == {"source"}
