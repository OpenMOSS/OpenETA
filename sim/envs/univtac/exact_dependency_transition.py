"""Small helpers for exact, single-distribution environment transitions."""

from __future__ import annotations

from typing import Any, Mapping

from sim.envs.univtac.filelock_compatibility import canonical_name


def classify_install_failure(log_text: str) -> str:
    if "error: incomplete-download" in log_text and (
        "SSLEOFError" in log_text or "Connection interrupted while downloading" in log_text
    ):
        return "blocked_by_external_resources"
    return "isaac51_package_install_failed"


def distribution_changes(before: Mapping[str, str], after: Mapping[str, str]) -> list[dict[str, Any]]:
    left = {canonical_name(name): version for name, version in before.items()}
    right = {canonical_name(name): version for name, version in after.items()}
    return [
        {"name": name, "before": left.get(name), "after": right.get(name)}
        for name in sorted(set(left) | set(right))
        if left.get(name) != right.get(name)
    ]


def audit_exact_transition(
    before: Mapping[str, str], after: Mapping[str, str], *, expected: Mapping[str, tuple[str | None, str | None]]
) -> dict[str, Any]:
    changes = distribution_changes(before, after)
    allowed = [
        {"name": canonical_name(name), "before": versions[0], "after": versions[1]}
        for name, versions in sorted(expected.items(), key=lambda item: canonical_name(item[0]))
    ]
    return {"success": changes == allowed, "changes": changes, "expected_changes": allowed}
