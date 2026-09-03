"""Pure Git revision and delivery contracts for R0.9.5.2R3."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


IMPLEMENTATION_BASE_HEAD = "2319e6587cb1c8330905313bae75e765b9b716d2"
EXECUTION_POLICY = "exactly_one_reviewed_descendant"
REQUIRED_PARENT_COUNT = 1
EXPECTED_COMMIT_SUBJECT = "fix: make R0952 execution revision self-consistent"
ALLOWED_CHANGED_PATHS = frozenset(
    {
        "configs/univtac/isaac51_offline_wheelhouse_r0952.yaml",
        "scripts/univtac/resume_isaac51_r0952.py",
        "sim/envs/univtac/execution_revision_contract.py",
        "sim/envs/univtac/offline_wheelhouse_contract.py",
        "tests/univtac/test_execution_revision_contract.py",
        "tests/univtac/test_r0952_gate_order.py",
        "tests/univtac/test_r0952_offline_wheelhouse.py",
    }
)
ALLOWED_DIRTY_PATHS = frozenset(
    {
        "README.md",
        "docs/architecture.md",
        "docs/vendor-notes.md",
        "sim/README.md",
        "sim/SETUP.md",
    }
)


def _git(repo: Path, *arguments: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _git_text(repo: Path, *arguments: str) -> str:
    result = _git(repo, *arguments)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def validate_revision_config(config: Mapping[str, Any]) -> None:
    if config != {
        "implementation_base_head": IMPLEMENTATION_BASE_HEAD,
        "execution_policy": EXECUTION_POLICY,
        "required_parent_count": REQUIRED_PARENT_COUNT,
        "expected_commit_subject": EXPECTED_COMMIT_SUBJECT,
    }:
        raise ValueError("R0.9.5.2R3 execution revision contract changed")


def inspect_execution_revision(
    repo: Path,
    *,
    implementation_base_head: str = IMPLEMENTATION_BASE_HEAD,
    expected_commit_subject: str = EXPECTED_COMMIT_SUBJECT,
    allowed_changed_paths: frozenset[str] = ALLOWED_CHANGED_PATHS,
    allowed_dirty_paths: frozenset[str] = ALLOWED_DIRTY_PATHS,
) -> dict[str, Any]:
    """Validate that HEAD is the one reviewed descendant of the fixed base."""

    execution_head = _git_text(repo, "rev-parse", "HEAD")
    parents = _git_text(repo, "show", "-s", "--format=%P", execution_head).split()
    ahead_count = int(
        _git_text(repo, "rev-list", "--count", f"{implementation_base_head}..{execution_head}")
    )
    subject = _git_text(repo, "show", "-s", "--format=%s", execution_head)
    changed_paths = tuple(
        line
        for line in _git_text(
            repo, "diff", "--name-only", implementation_base_head, execution_head
        ).splitlines()
        if line
    )
    staged_paths = tuple(
        line for line in _git_text(repo, "diff", "--cached", "--name-only").splitlines() if line
    )
    dirty_paths = tuple(
        line for line in _git_text(repo, "diff", "--name-only").splitlines() if line
    )
    unauthorized_changed_paths = tuple(sorted(set(changed_paths) - allowed_changed_paths))
    unauthorized_dirty_paths = tuple(sorted(set(dirty_paths) - allowed_dirty_paths))
    checks = {
        "execution_differs_from_base": execution_head != implementation_base_head,
        "required_parent_count": len(parents) == REQUIRED_PARENT_COUNT,
        "direct_parent": parents == [implementation_base_head],
        "exactly_one_commit_ahead": ahead_count == 1,
        "expected_commit_subject": subject == expected_commit_subject,
        "authorized_changed_paths": not unauthorized_changed_paths,
        "staged_empty": not staged_paths,
        "only_allowed_dirty_paths": not unauthorized_dirty_paths,
    }
    return {
        "implementation_base_head": implementation_base_head,
        "execution_head": execution_head,
        "execution_policy": EXECUTION_POLICY,
        "parents": parents,
        "parent_count": len(parents),
        "ahead_count": ahead_count,
        "commit_subject": subject,
        "changed_paths": list(changed_paths),
        "allowed_changed_paths": sorted(allowed_changed_paths),
        "unauthorized_changed_paths": list(unauthorized_changed_paths),
        "staged_paths": list(staged_paths),
        "dirty_paths": list(dirty_paths),
        "allowed_dirty_paths": sorted(allowed_dirty_paths),
        "unauthorized_dirty_paths": list(unauthorized_dirty_paths),
        "checks": checks,
        "passed": all(checks.values()),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_and_verify_bundle(
    repo: Path,
    bundle_path: Path,
    execution_head: str,
    *,
    implementation_base_head: str = IMPLEMENTATION_BASE_HEAD,
) -> dict[str, Any]:
    """Create a one-commit bundle and verify its prerequisite and tip objects."""

    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    create = _git(
        repo,
        "bundle",
        "create",
        str(bundle_path),
        "HEAD",
        f"^{implementation_base_head}",
    )
    verify = _git(repo, "bundle", "verify", str(bundle_path)) if create.returncode == 0 else None
    heads = _git(repo, "bundle", "list-heads", str(bundle_path)) if create.returncode == 0 else None
    base_object = _git(repo, "cat-file", "-e", f"{implementation_base_head}^{{commit}}")
    execution_object = _git(repo, "cat-file", "-e", f"{execution_head}^{{commit}}")
    listed_heads = heads.stdout.splitlines() if heads and heads.returncode == 0 else []
    checks = {
        "create_succeeded": create.returncode == 0,
        "verify_succeeded": verify is not None and verify.returncode == 0,
        "execution_head_listed": any(
            line.split(maxsplit=1)[0] == execution_head for line in listed_heads if line.strip()
        ),
        "base_object_available": base_object.returncode == 0,
        "execution_object_available": execution_object.returncode == 0,
    }
    return {
        "bundle_path": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path) if bundle_path.is_file() else None,
        "bundle_size": bundle_path.stat().st_size if bundle_path.is_file() else None,
        "create_returncode": create.returncode,
        "create_stderr": create.stderr.strip(),
        "verify_returncode": verify.returncode if verify is not None else None,
        "verify_stdout": verify.stdout.strip() if verify is not None else "",
        "verify_stderr": verify.stderr.strip() if verify is not None else "",
        "listed_heads": listed_heads,
        "implementation_base_head": implementation_base_head,
        "execution_head": execution_head,
        "checks": checks,
        "passed": all(checks.values()),
    }


def classify_delivery(
    *,
    execution_head: str,
    push_returncode: int,
    remote_head: str | None,
    remote_error: str | None,
    bundle_verified: bool,
) -> dict[str, Any]:
    if push_returncode != 0:
        status = "push_failed"
        passed = False
    elif remote_head == execution_head:
        status = "push_and_remote_confirmation_succeeded"
        passed = True
    elif remote_head is None and remote_error and bundle_verified:
        status = "push_succeeded_confirmation_transport_unavailable"
        passed = True
    elif remote_head is not None:
        status = "remote_head_mismatch"
        passed = False
    else:
        status = "remote_confirmation_unavailable_without_verified_bundle"
        passed = False
    return {
        "status": status,
        "passed": passed,
        "push_returncode": push_returncode,
        "remote_head": remote_head,
        "remote_error": remote_error,
        "bundle_verified": bundle_verified,
    }


def validate_r0952r1_evidence(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((root / "resume_preflight/summary.json").read_text(encoding="utf-8"))
    conditions = preflight.get("conditions", {})
    false_conditions = sorted(key for key, value in conditions.items() if value is not True)
    downstream = ("T0", "T1", "C0", "M0", "M1", "D0", "D1", "D2", "V0", "O0")
    checks = {
        "classification": manifest.get("classification") == "r0952r1_resume_precondition_failed",
        "only_git_delivery_false": false_conditions == ["git_delivery"],
        "downstream_not_run": all(
            manifest.get("stages", {}).get(stage) == "not_run_due_to_gate" for stage in downstream
        ),
        "actual_install_false": manifest.get("actual_install_executed") is False,
        "install_count_one": manifest.get("total_isaac_install_invocations") == 1,
        "r08_unchanged": manifest.get("r08_unchanged") is True,
        "r09_unchanged": manifest.get("r09_unchanged") is True,
        "cleanup_complete": manifest.get("managed_cleanup_complete") is True,
    }
    return {
        "root": str(root),
        "classification": manifest.get("classification"),
        "false_conditions": false_conditions,
        "observed_stages": {stage: manifest.get("stages", {}).get(stage) for stage in downstream},
        "checks": checks,
        "passed": all(checks.values()),
    }
