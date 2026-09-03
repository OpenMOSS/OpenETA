from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sim.envs.univtac.execution_revision_contract import (
    ALLOWED_CHANGED_PATHS,
    EXPECTED_COMMIT_SUBJECT,
    IMPLEMENTATION_BASE_HEAD,
    classify_delivery,
    create_and_verify_bundle,
    inspect_execution_revision,
    validate_r0952r1_evidence,
)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _commit(repo: Path, subject: str, filename: str, content: str) -> str:
    (repo / filename).parent.mkdir(parents=True, exist_ok=True)
    (repo / filename).write_text(content, encoding="utf-8")
    _git(repo, "add", filename)
    _git(
        repo,
        "-c",
        "user.name=R0952 Test",
        "-c",
        "user.email=r0952@example.invalid",
        "commit",
        "-m",
        subject,
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def revision_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    base = _commit(repo, "base", "base.txt", "base\n")
    _commit(repo, EXPECTED_COMMIT_SUBJECT, "allowed.txt", "reviewed\n")
    return repo, base


def _inspect(repo: Path, base: str) -> dict:
    return inspect_execution_revision(
        repo,
        implementation_base_head=base,
        allowed_changed_paths=frozenset({"allowed.txt"}),
        allowed_dirty_paths=frozenset({"README.md"}),
    )


def test_real_config_anchors_base_without_execution_head_self_reference() -> None:
    import yaml

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "configs/univtac/isaac51_offline_wheelhouse_r0952.yaml").read_text()
    )
    assert config["revision_contract"]["implementation_base_head"] == IMPLEMENTATION_BASE_HEAD
    assert "openeta_head" not in config
    assert "execution_head" not in config["revision_contract"]


def test_exactly_one_direct_reviewed_child_is_accepted(
    revision_repo: tuple[Path, str],
) -> None:
    repo, base = revision_repo
    result = _inspect(repo, base)
    assert result["passed"] is True
    assert result["parents"] == [base]
    assert result["ahead_count"] == 1


def test_wrong_subject_two_ahead_and_unauthorized_path_are_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    base = _commit(repo, "base", "base.txt", "base\n")
    _commit(repo, "wrong subject", "allowed.txt", "one\n")
    wrong = _inspect(repo, base)
    assert wrong["checks"]["expected_commit_subject"] is False
    _commit(repo, EXPECTED_COMMIT_SUBJECT, "forbidden.txt", "two\n")
    two_ahead = _inspect(repo, base)
    assert two_ahead["checks"]["exactly_one_commit_ahead"] is False
    assert two_ahead["checks"]["authorized_changed_paths"] is False


def test_merge_commit_and_staged_change_are_rejected(
    revision_repo: tuple[Path, str],
) -> None:
    repo, base = revision_repo
    main = _git(repo, "branch", "--show-current")
    _git(repo, "checkout", "-q", "-b", "side", base)
    _commit(repo, "side", "side.txt", "side\n")
    _git(repo, "checkout", "-q", main)
    _git(
        repo,
        "-c",
        "user.name=R0952 Test",
        "-c",
        "user.email=r0952@example.invalid",
        "merge",
        "--no-ff",
        "-m",
        EXPECTED_COMMIT_SUBJECT,
        "side",
    )
    result = _inspect(repo, base)
    assert result["checks"]["required_parent_count"] is False
    (repo / "staged.txt").write_text("staged\n", encoding="utf-8")
    _git(repo, "add", "staged.txt")
    result = _inspect(repo, base)
    assert result["checks"]["staged_empty"] is False


def test_allowed_dirty_docs_do_not_invalidate_revision(
    revision_repo: tuple[Path, str],
) -> None:
    repo, base = revision_repo
    (repo / "README.md").write_text("dirty but allowed\n", encoding="utf-8")
    assert _inspect(repo, base)["passed"] is True


def test_push_outcomes_distinguish_failure_mismatch_and_confirmation_transport() -> None:
    failed = classify_delivery(
        execution_head="a", push_returncode=1, remote_head=None, remote_error="TLS", bundle_verified=True
    )
    assert failed["status"] == "push_failed" and failed["passed"] is False
    mismatch = classify_delivery(
        execution_head="a", push_returncode=0, remote_head="b", remote_error=None, bundle_verified=True
    )
    assert mismatch["status"] == "remote_head_mismatch" and mismatch["passed"] is False
    unavailable = classify_delivery(
        execution_head="a", push_returncode=0, remote_head=None, remote_error="TLS", bundle_verified=True
    )
    assert unavailable["status"] == "push_succeeded_confirmation_transport_unavailable"
    assert unavailable["passed"] is True


def test_revision_allowlist_excludes_downloader_cache_and_outputs() -> None:
    assert "scripts/univtac/download_isaac_wheelhouse.py" not in ALLOWED_CHANGED_PATHS
    assert not any(path.startswith(("outputs/", ".cache/")) for path in ALLOWED_CHANGED_PATHS)


def test_bundle_contains_execution_tip_and_verifies(
    revision_repo: tuple[Path, str], tmp_path: Path
) -> None:
    repo, base = revision_repo
    head = _git(repo, "rev-parse", "HEAD")
    result = create_and_verify_bundle(
        repo, tmp_path / "snapshot.bundle", head, implementation_base_head=base
    )
    assert result["passed"] is True
    assert result["bundle_size"] > 0
    assert result["bundle_sha256"]


def test_r0952r1_requires_only_git_delivery_false(tmp_path: Path) -> None:
    root = tmp_path / "r1"
    (root / "resume_preflight").mkdir(parents=True)
    manifest = {
        "classification": "r0952r1_resume_precondition_failed",
        "stages": {
            stage: "not_run_due_to_gate"
            for stage in ("T0", "T1", "C0", "M0", "M1", "D0", "D1", "D2", "V0", "O0")
        },
        "actual_install_executed": False,
        "total_isaac_install_invocations": 1,
        "r08_unchanged": True,
        "r09_unchanged": True,
        "managed_cleanup_complete": True,
    }
    import json

    (root / "run_manifest.json").write_text(json.dumps(manifest))
    (root / "resume_preflight/summary.json").write_text(
        json.dumps({"conditions": {"git_delivery": False, "other": True}})
    )
    assert validate_r0952r1_evidence(root)["passed"] is True
    (root / "resume_preflight/summary.json").write_text(
        json.dumps({"conditions": {"git_delivery": False, "other": False}})
    )
    assert validate_r0952r1_evidence(root)["passed"] is False
