from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts.univtac import prepare_flatdict_wheel as builder
from scripts.univtac import resume_isaac51_blackwell_install as resume


def test_no_build_isolation_is_scoped_to_flatdict_wheel_build() -> None:
    build_source = inspect.getsource(builder.main)
    resume_source = inspect.getsource(resume.main)
    assert build_source.count('"--no-build-isolation"') == 1
    assert '"pip", "wheel", "--no-deps",\n            "--no-build-isolation"' in build_source
    assert "--no-build-isolation" not in resume_source


def test_pre_isaac_flatdict_builder_does_not_import_sim_package() -> None:
    source = inspect.getsource(builder)
    assert "from sim." not in source
    assert 'spec_from_file_location("univtac_flatdict_build_contract"' in source


def test_resume_never_clones_or_runs_official_installer() -> None:
    source = inspect.getsource(resume)
    assert '"clone"' not in inspect.getsource(resume.main)
    assert "scripts/install.sh" not in source
    assert "AppLauncher" not in source
    assert '"sudo"' not in source
    assert '"sysctl"' not in source


def test_corrected_plans_precede_actual_installs_and_n0() -> None:
    source = inspect.getsource(resume.main)
    positions = [
        source.index('manifest["stages"][stage] = "passed"'),
        source.index('write_json(output / "dependency_plan/summary.json"'),
        source.index('"--no-index", "--no-deps"'),
        source.index('manifest["isaac_actual_install_invocations"] = 1'),
        source.index('manifest["stages"]["I1"]'),
        source.index('manifest["stages"]["I2"]'),
        source.index('stage="N0"'),
    ]
    assert positions == sorted(positions)
    assert source.count('manifest["isaac_actual_install_invocations"] = 1') == 1


def test_prior_actual_install_detection_distinguishes_dry_run(tmp_path: Path) -> None:
    processes = tmp_path / "gate/processes"
    processes.mkdir(parents=True)
    (processes / "dry.json").write_text(json.dumps({"command": ["python", "-m", "pip", "install", "--dry-run", "x"]}), encoding="utf-8")
    assert resume.prior_actual_installs(tmp_path) == []
    (processes / "actual.json").write_text(json.dumps({"command": ["python", "-m", "pip", "install", "x"]}), encoding="utf-8")
    assert len(resume.prior_actual_installs(tmp_path)) == 1


def test_all_future_stages_start_fail_closed() -> None:
    assert resume.STAGES == (
        "F0", "F1", "F2", "F3", "P0A", "P0B", "I0A", "I0B", "I1", "I2", "N0", "S0", "S1", "G0", "L0", "C0", "H0"
    )
