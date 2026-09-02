from __future__ import annotations

import copy
import inspect
import os
from pathlib import Path

import pytest
import yaml

from scripts.univtac import apply_filelock3131_bridge as runner
from scripts.univtac import prepare_filelock3131_wheel as prepare
from scripts.univtac import resume_isaac51_r093 as continuation
from scripts.univtac import smoke_installed_filelock
from sim.envs.univtac.filelock_compatibility import (
    AFTER_VERSION,
    active_reverse_dependencies,
    exact_filelock_transition,
    private_directory_record,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/univtac/isaacsim_filelock3131_bridge.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_fixed_bridge_config_and_official_wheel_identity() -> None:
    payload = config()
    validate_config(payload)
    assert payload["before"] == {"filelock": "3.32.3"}
    assert payload["after"] == {"filelock": "3.13.1"}
    assert payload["wheel"] == {
        "version": "3.13.1",
        "filename": "filelock-3.13.1-py3-none-any.whl",
        "size": 11740,
        "sha256": "57dbda9b35157b05fb3e58ee91448612eb674172fab98ee235ccb0b5bee19a1c",
        "pypi_json_url": "https://pypi.org/pypi/filelock/3.13.1/json",
    }
    assert payload["security_exception"]["reason"] == "isaacsim-core 5.1.0.0 exact dependency"
    assert payload["security_exception"]["benchmark_semantics_changed"] is False
    changed = copy.deepcopy(payload)
    changed["wheel"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        validate_config(changed)


def test_reverse_dependency_audit_respects_markers_and_candidate() -> None:
    distributions = [
        {"name": "torch", "version": "2.7.0", "requires": ["filelock"]},
        {"name": "helper", "version": "1", "requires": ["filelock>=3.4", "filelock>=99; extra == 'test'"]},
    ]
    result = active_reverse_dependencies(distributions)
    assert result["success"] is True
    assert [item["distribution"] for item in result["active"]] == ["helper", "torch"]
    assert result["inactive"][0]["accepts_candidate"] is False
    rejected = active_reverse_dependencies([{"name": "bad", "requires": ["filelock>=99"]}])
    assert rejected["success"] is False
    assert rejected["rejected"][0]["distribution"] == "bad"


def test_only_exact_filelock_transition_is_accepted() -> None:
    before = {"filelock": "3.32.3", "torch": "2.7.0+cu128"}
    after = {"filelock": AFTER_VERSION, "torch": "2.7.0+cu128"}
    assert exact_filelock_transition(before, after)["success"] is True
    changed = dict(after, packaging="23.0")
    assert exact_filelock_transition(before, changed)["success"] is False


def test_private_directory_record_requires_mode_0700(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    private.chmod(0o700)
    record = private_directory_record(private)
    assert record["private"] is True
    assert record["owned_by_current_user"] is True
    assert record["owner_uid"] == os.getuid()
    private.chmod(0o755)
    assert private_directory_record(private)["private"] is False


def test_runner_has_one_exact_filelock_mutation_and_nfl_precedes_p0() -> None:
    source = inspect.getsource(runner.main)
    mutation = '[str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)]'
    assert source.count(mutation) == 1
    assert 'manifest["filelock_mutation_invocations"] = 1' in source
    assert 'stage="NFL"' in source
    assert source.index('manifest["stages"]["F5"] = "passed"') < source.index('stage="NFL"')
    assert source.index('stage="NFL"') < source.index('write_json(marker_path, marker_after)')
    assert "--upgrade" not in source and "--force-reinstall" not in source
    assert "sysctl" not in source and "sudo" not in source


def test_wheel_metadata_checks_active_requirements_not_optional_extras() -> None:
    source = inspect.getsource(prepare.main)
    assert 'marker_environment["extra"] = ""' in source
    assert "active_requirements" in source
    assert '"-m", "venv"' in source
    assert '"isolation_method": "fresh_venv"' in source


def test_continuation_requires_nfl_and_uses_new_conflict_classification() -> None:
    source = inspect.getsource(continuation.main)
    assert 'prior_stage="NFL" if prior_status=="filelock_bridge_validated" else "NBT"' in source
    assert '"filelock_aligned_isaac_resolution_conflict"' in source
    assert '"nonprotected_install_plan"' in source
    assert source.index('prior_stage="NFL"') < source.index('plans={"P0A"')


def test_installed_smoke_is_private_and_does_not_test_exploit() -> None:
    source = inspect.getsource(smoke_installed_filelock.main)
    assert 'mode=0o700' in source
    assert '"exploit_or_symlink_attack_tested": False' in source
