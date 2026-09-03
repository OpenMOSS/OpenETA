from __future__ import annotations

from pathlib import Path

import pytest

from sim.envs.univtac.scoped_isaac51_launcher import (
    LibcudaDriverProbe,
    ScopedIsaac51LaunchError,
    ScopedIsaac51LaunchSpec,
    build_scoped_isaac51_child_environment,
    build_scoped_isaac51_dry_run,
    prepare_process_local_libcuda_alias,
    validate_real_libcuda_driver_path,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_SOURCE = REPO_ROOT / "sim/envs/univtac/scoped_isaac51_launcher.py"


def _fake_driver(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "usr" / "lib" / "x86_64-linux-gnu"
    root.mkdir(parents=True)
    driver = root / "libcuda.so.580.1"
    driver.write_bytes(b"driver")
    return root.parents[2], driver


def test_driver_path_rejects_stub_conda_and_output_copy(tmp_path: Path) -> None:
    allowed_root, driver = _fake_driver(tmp_path)
    assert validate_real_libcuda_driver_path(driver, allowed_roots=(allowed_root,)) == driver

    stub = allowed_root / "local" / "cuda-12.8" / "lib64" / "stubs" / "libcuda.so.1"
    stub.parent.mkdir(parents=True)
    stub.write_bytes(b"stub")
    with pytest.raises(ScopedIsaac51LaunchError, match="stub"):
        validate_real_libcuda_driver_path(stub, allowed_roots=(allowed_root,))

    conda = allowed_root / "conda" / "lib" / "libcuda.so.1"
    conda.parent.mkdir(parents=True)
    conda.write_bytes(b"conda")
    with pytest.raises(ScopedIsaac51LaunchError, match="Conda"):
        validate_real_libcuda_driver_path(
            conda, allowed_roots=(allowed_root,), conda_prefix=conda.parents[1]
        )

    with pytest.raises(ScopedIsaac51LaunchError, match="output-local"):
        validate_real_libcuda_driver_path(
            driver, allowed_roots=(allowed_root,), output_root=allowed_root
        )


def test_alias_directory_contains_only_private_libcuda_symlink(tmp_path: Path) -> None:
    allowed_root, driver = _fake_driver(tmp_path)
    output_root = tmp_path / "output"
    alias = prepare_process_local_libcuda_alias(output_root, driver, allowed_roots=(allowed_root,))
    assert alias.name == "libcuda.so"
    assert alias.is_symlink()
    assert alias.resolve() == driver
    assert [entry.name for entry in alias.parent.iterdir()] == ["libcuda.so"]
    assert alias.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ScopedIsaac51LaunchError, match="not fresh"):
        prepare_process_local_libcuda_alias(output_root, driver, allowed_roots=(allowed_root,))


def test_child_environment_is_scoped_and_parent_is_unchanged(tmp_path: Path) -> None:
    parent = {
        "CUDA_VISIBLE_DEVICES": "7",
        "LD_LIBRARY_PATH": "/one:/two",
        "KEEP_ME": "yes",
        "LD_PRELOAD": "/tmp/not-for-child.so",
    }
    original = parent.copy()
    alias_dir = tmp_path / "alias"
    child, summary = build_scoped_isaac51_child_environment(
        parent_environment=parent,
        alias_dir=alias_dir,
        environment_overrides={
            "CUDA_VISIBLE_DEVICES": "3",
            "LD_PRELOAD": "/tmp/also-forbidden.so",
            "OMNI_KIT_ACCEPT_EULA": "NO",
        },
    )
    assert parent == original
    assert "CUDA_VISIBLE_DEVICES" not in child
    assert "LD_PRELOAD" not in child
    assert child["LD_LIBRARY_PATH"] == f"{alias_dir}:/one:/two"
    assert child["KEEP_ME"] == "yes"
    assert child["OMNI_KIT_ACCEPT_EULA"] == "YES"
    assert summary["parent_cuda_visible_devices"] == "7"
    assert summary["child_cuda_visible_devices"] is None


def test_dry_run_preserves_exact_argv_without_creating_output(tmp_path: Path) -> None:
    driver = Path("/usr/lib/x86_64-linux-gnu/libcuda.so.580.1")
    probe = LibcudaDriverProbe(driver, 0, 0, 1, 0, 13000)
    spec = ScopedIsaac51LaunchSpec(
        python_executable=Path("/runtime/python"),
        command=("probe.py", "--seed", "1000000", "value with space"),
        cwd=Path("/source"),
        output_root=tmp_path / "output",
        timeout_seconds=1200,
    )
    manifest = build_scoped_isaac51_dry_run(spec, probe)
    assert manifest["command"] == [
        "/runtime/python",
        "probe.py",
        "--seed",
        "1000000",
        "value with space",
    ]
    assert not spec.output_root.exists()


def test_launcher_has_no_global_process_gate_hashing_or_shell_forwarding() -> None:
    source = LAUNCHER_SOURCE.read_text(encoding="utf-8")
    assert "no_parallel_runtime" not in source
    assert "hashlib" not in source
    assert "sha256" not in source.lower()
    assert "shell=True" not in source
    assert "start_new_session=True" in source
    assert "os.killpg" in source
    assert "pkill" not in source
    assert "killall" not in source.lower()
    assert "os.environ.clear" not in source
    assert "os.environ.update" not in source


def test_launcher_cli_does_not_start_child_in_dry_run_path() -> None:
    source = (REPO_ROOT / "scripts/univtac/run_scoped_isaac51.py").read_text(encoding="utf-8")
    dry_run_branch = source.index("if args.dry_run:")
    child_launch = source.index("run_scoped_isaac51_command(spec)")
    assert dry_run_branch < child_launch
    assert "subprocess.Popen" not in source
