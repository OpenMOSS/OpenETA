from __future__ import annotations

import inspect
from pathlib import Path

from scripts.univtac import resume_blackwell_native_bridge as runner


def test_runner_has_one_u0r_and_never_starts_isaac() -> None:
    source = inspect.getsource(runner)
    assert source.count('name="u0r"') == 1
    assert "AppLauncher" not in source
    assert "smoke_isaac51" not in source


def test_curobo_failure_summary_distinguishes_missing_include(tmp_path: Path) -> None:
    prefix = tmp_path / "env"
    header = prefix / "targets/x86_64-linux/include/cuda_runtime_api.h"
    header.parent.mkdir(parents=True)
    header.write_text("", encoding="utf-8")
    log = tmp_path / "build.log"
    log.write_text(
        "fatal error: cuda_runtime_api.h: No such file or directory\n"
        "nvcc -gencode=arch=compute_120,code=sm_120\n",
        encoding="utf-8",
    )
    result = runner.curobo_failure_summary(log, prefix)
    assert result["stage"] == "host_cxx_compile"
    assert result["sm120_nvcc_command_observed"] is True
    assert result["header_candidates_in_target_environment"]["cuda_runtime_api.h"] == [
        str(header.resolve())
    ]


def test_cleanup_summary_requires_every_managed_record(tmp_path: Path) -> None:
    process = tmp_path / "gate/processes/run.json"
    process.parent.mkdir(parents=True)
    process.write_text(
        '{"child_root_pid": 999999999, "cleanup_complete": true, "command": ["never"]}',
        encoding="utf-8",
    )
    result = runner.managed_cleanup_summary(tmp_path)
    assert result["all_records_cleanup_complete"] is True
    assert result["cleanup_residual"] is False
