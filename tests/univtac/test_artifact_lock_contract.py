from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from sim.envs.univtac.artifact_lock_contract import (
    build_artifact_lock,
    failure_classification,
    public_summary,
    validate_config,
    validate_p0a_process,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/univtac/isaac51_offline_wheelhouse.yaml"


def item(filename: str, *, name: str = "demo", version: str = "1.0", sha: str | None = "a" * 64, yanked: bool = False) -> dict:
    hashes = {"sha256": sha} if sha is not None else {}
    return {
        "download_info": {"url": f"https://files.pythonhosted.org/packages/x/{filename}", "archive_info": {"hashes": hashes}},
        "is_yanked": yanked,
        "requested": False,
        "metadata": {"name": name, "version": version},
    }


def report(*records: dict) -> dict:
    return {"version": "1", "pip_version": "26.2.1", "install": list(records)}


def test_config_freezes_attempt_and_install_accounting() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(payload)
    assert payload["max_transport_attempts_per_artifact"] == 20
    assert payload["install_accounting"] == {
        "prior_online_invocations": 1,
        "authorized_offline_retry_invocations": 1,
        "total_authorized_invocations": 2,
        "additional_install_retry_allowed": False,
    }
    changed = copy.deepcopy(payload)
    changed["max_transport_attempts_per_artifact"] = 21
    with pytest.raises(ValueError, match="attempt"):
        validate_config(changed)


def test_lock_accepts_compatible_wheel_and_redacts_url() -> None:
    lock = build_artifact_lock(report(item("demo-1.0-py3-none-any.whl")), allowed_hosts={"files.pythonhosted.org"})
    assert lock["success"] is True
    assert lock["records"][0]["wheel_tag_compatible"] is True
    summary = public_summary(lock)
    assert "url" not in summary["records"][0]
    assert summary["records"][0]["origin_host"] == "files.pythonhosted.org"


def test_nonwheel_is_a_hard_gate() -> None:
    lock = build_artifact_lock(report(item("demo-1.0.tar.gz")), allowed_hosts={"files.pythonhosted.org"})
    assert lock["success"] is False
    assert lock["failures"]["nonwheel_artifacts"] == ["demo-1.0.tar.gz"]
    assert failure_classification(lock) == "artifact_lock_contains_nonwheel"


def test_missing_sha_precedes_nonwheel_classification() -> None:
    lock = build_artifact_lock(report(item("demo-1.0.tar.gz", sha=None)), allowed_hosts={"files.pythonhosted.org"})
    assert failure_classification(lock) == "artifact_lock_missing_sha256"


def test_yanked_or_unapproved_origin_fails_validation() -> None:
    yanked = build_artifact_lock(report(item("demo-1.0-py3-none-any.whl", yanked=True)), allowed_hosts={"files.pythonhosted.org"})
    assert failure_classification(yanked) == "wheelhouse_validation_failed"
    bad = item("demo-1.0-py3-none-any.whl")
    bad["download_info"]["url"] = "https://mirror.invalid/demo-1.0-py3-none-any.whl"
    assert failure_classification(build_artifact_lock(report(bad), allowed_hosts={"files.pythonhosted.org"})) == "wheelhouse_validation_failed"


def test_duplicate_distribution_version_fails() -> None:
    record = item("demo-1.0-py3-none-any.whl")
    lock = build_artifact_lock(report(record, copy.deepcopy(record)), allowed_hosts={"files.pythonhosted.org"})
    assert lock["failures"]["duplicate_name_versions"] == ["demo==1.0"]
    assert lock["record_count"] == 2


def test_p0a_process_identity_requires_exact_report_and_constraint(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    constraints = tmp_path / "constraints.txt"
    command = [
        "python", "-m", "pip", "install", "--dry-run", "--report", str(report_path),
        "--constraint", str(constraints), "--find-links", "/wheelhouse", "--only-binary=flatdict",
        "isaaclab[isaacsim,all]==2.3.0", "--extra-index-url", "https://pypi.nvidia.com",
    ]
    process = {"command": command, "returncode": 0, "cleanup_complete": True}
    validate_p0a_process(process, report_path=report_path, constraints_path=constraints, requirement="isaaclab[isaacsim,all]==2.3.0")
    process["command"] = command + ["--no-deps"]
    with pytest.raises(ValueError, match="prohibited"):
        validate_p0a_process(process, report_path=report_path, constraints_path=constraints, requirement="isaaclab[isaacsim,all]==2.3.0")
