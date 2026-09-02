"""Validated R0.8 package and binary identity contract for R0.9.1."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from sim.envs.univtac.simulator_install_contract import required_runtime_versions
from sim.envs.univtac.simulator_install_contract import canonical_name


def compare_config_to_source(
    config: Mapping[str, Any], source_distributions: Mapping[str, str | None],
    target_distributions: Mapping[str, str | None],
) -> dict[str, Any]:
    required = required_runtime_versions(config)
    mismatches = []
    for name, expected in sorted(required.items()):
        source = source_distributions.get(name)
        target = target_distributions.get(name)
        if source != expected or target != expected:
            mismatches.append({"name": name, "configured": expected, "source": source, "target": target})
    return {
        "success": not mismatches,
        "protected_baseline_source": config.get("protected_baseline_source"),
        "required": required,
        "mismatches": mismatches,
    }


def package_diff(
    before: Mapping[str, str | None], after: Mapping[str, str | None],
    *, allowed_additions: Mapping[str, str],
) -> dict[str, Any]:
    before = {canonical_name(name): version for name, version in before.items()}
    after = {canonical_name(name): version for name, version in after.items()}
    allowed_additions = {canonical_name(name): version for name, version in allowed_additions.items()}
    names = sorted(set(before) | set(after))
    changes = [
        {"name": name, "before": before.get(name), "after": after.get(name)}
        for name in names if before.get(name) != after.get(name)
    ]
    expected = [
        {"name": name, "before": None, "after": version}
        for name, version in sorted(allowed_additions.items())
    ]
    return {"success": changes == expected, "changes": changes, "allowed_changes": expected}


def report_install_versions(report: Mapping[str, Any]) -> dict[str, str]:
    return {
        canonical_name(str(item.get("metadata", {}).get("name", ""))): str(item.get("metadata", {}).get("version", ""))
        for item in report.get("install", [])
        if item.get("metadata", {}).get("name") and item.get("metadata", {}).get("version")
    }


def audit_changes_against_plan(
    before: Mapping[str, str], after: Mapping[str, str], plan: Mapping[str, str],
    *, additional_allowed: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    before_normalized = {canonical_name(name): version for name, version in before.items()}
    after_normalized = {canonical_name(name): version for name, version in after.items()}
    allowed = {canonical_name(name): version for name, version in plan.items()}
    allowed.update({canonical_name(name): version for name, version in (additional_allowed or {}).items()})
    changes = [
        {"name": name, "before": before_normalized.get(name), "after": after_normalized.get(name)}
        for name in sorted(set(before_normalized) | set(after_normalized))
        if before_normalized.get(name) != after_normalized.get(name)
    ]
    unexpected = [
        item for item in changes if allowed.get(item["name"]) != item["after"]
    ]
    return {"success": not unexpected, "changes": changes, "unexpected_changes": unexpected, "allowed_plan": allowed}


def binary_manifest(paths: Sequence[Path], *, root: Path) -> dict[str, Any]:
    records = []
    for path in sorted((item.resolve() for item in paths), key=str):
        if not path.is_file() or not path.is_relative_to(root.resolve()):
            raise ValueError(f"protected binary is missing or outside its expected root: {path}")
        records.append({"path": str(path), "size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {"root": str(root.resolve()), "records": records}


def compare_binary_manifests(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_map = {item["path"]: (item["size"], item["sha256"]) for item in before.get("records", [])}
    after_map = {item["path"]: (item["size"], item["sha256"]) for item in after.get("records", [])}
    changed = sorted(path for path in set(before_map) | set(after_map) if before_map.get(path) != after_map.get(path))
    return {"success": not changed and bool(before_map), "changed_paths": changed, "before_count": len(before_map), "after_count": len(after_map)}
