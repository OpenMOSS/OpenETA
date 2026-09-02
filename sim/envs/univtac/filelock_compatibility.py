"""Pure-Python contract for the authorized R0.9.4 filelock bridge."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path
from typing import Any, Iterable, Mapping

from packaging.markers import default_environment
from packaging.requirements import Requirement


SCHEMA_VERSION = "openeta.univtac.isaacsim_filelock3131_bridge.v1"
APPLIED_LABEL = "isaacsim_filelock3131_compatibility_bridge_v1"
BEFORE_VERSION = "3.32.3"
AFTER_VERSION = "3.13.1"
WHEEL_IDENTITY = (
    "filelock-3.13.1-py3-none-any.whl",
    11740,
    "57dbda9b35157b05fb3e58ee91448612eb674172fab98ee235ccb0b5bee19a1c",
)


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION or config.get("applied_label") != APPLIED_LABEL:
        raise ValueError("filelock bridge schema or label mismatch")
    if config.get("environment") != "UniVTAC-isaac51-sm120-r09":
        raise ValueError("filelock bridge may only target R0.9")
    if config.get("before") != {"filelock": BEFORE_VERSION}:
        raise ValueError("filelock bridge before state mismatch")
    if config.get("after") != {"filelock": AFTER_VERSION}:
        raise ValueError("filelock bridge after state mismatch")
    wheel = config.get("wheel", {})
    actual = (wheel.get("filename"), int(wheel.get("size", -1)), wheel.get("sha256"))
    if str(wheel.get("version")) != AFTER_VERSION or actual != WHEEL_IDENTITY:
        raise ValueError("fixed filelock wheel identity changed")
    if config.get("allowed_host") != "files.pythonhosted.org":
        raise ValueError("filelock wheel host changed")
    if config.get("mutation_command") != [
        "python", "-m", "pip", "install", "--no-index", "--no-deps", "VERIFIED_WHEEL"
    ]:
        raise ValueError("filelock mutation command changed")
    expected_security = {
        "isolated_research_environment": True,
        "package": "filelock",
        "version": AFTER_VERSION,
        "reason": "isaacsim-core 5.1.0.0 exact dependency",
        "scope": "isolated research simulator environment only",
        "production_use_allowed": False,
        "untrusted_code_allowed": False,
        "private_project_directories_required": True,
        "benchmark_semantics_changed": False,
    }
    if config.get("security_exception") != expected_security:
        raise ValueError("filelock security exception changed")


def active_reverse_dependencies(
    distributions: Iterable[Mapping[str, Any]], *, candidate_version: str = AFTER_VERSION
) -> dict[str, Any]:
    """Audit active installed requirements that name filelock."""
    environment = default_environment()
    environment["extra"] = ""
    active: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    distribution_records: list[dict[str, Any]] = []
    for distribution in distributions:
        owner = canonical_name(str(distribution["name"]))
        requirement_records = []
        for raw in distribution.get("requires", ()) or ():
            requirement = Requirement(str(raw))
            marker_active = requirement.marker is None or requirement.marker.evaluate(environment)
            requirement_record = {
                "requires_dist": str(requirement),
                "environment_marker": str(requirement.marker) if requirement.marker is not None else None,
                "marker_active": marker_active,
                "requirement_on_filelock": canonical_name(requirement.name) == "filelock",
            }
            if requirement_record["requirement_on_filelock"]:
                requirement_record["filelock_3_13_1_satisfies"] = requirement.specifier.contains(AFTER_VERSION, prereleases=True)
                requirement_record["filelock_3_32_3_satisfies"] = requirement.specifier.contains(BEFORE_VERSION, prereleases=True)
            requirement_records.append(requirement_record)
            if not requirement_record["requirement_on_filelock"]:
                continue
            record = {
                "distribution": owner,
                "requirement": str(requirement),
                "accepts_candidate": requirement.specifier.contains(candidate_version, prereleases=True),
            }
            if marker_active:
                active.append(record)
            else:
                inactive.append(record)
        distribution_records.append(
            {
                "distribution": owner,
                "version": str(distribution.get("version", "")),
                "requirements": requirement_records,
            }
        )
    active.sort(key=lambda item: (item["distribution"], item["requirement"]))
    inactive.sort(key=lambda item: (item["distribution"], item["requirement"]))
    rejected = [item for item in active if not item["accepts_candidate"]]
    return {
        "candidate_version": candidate_version,
        "active": active,
        "inactive": inactive,
        "rejected": rejected,
        "distributions": sorted(distribution_records, key=lambda item: item["distribution"]),
        "future_requirements": [
            {
                "distribution": "isaacsim-core",
                "version": "5.1.0.0",
                "requirement": "filelock==3.13.1",
                "filelock_3_13_1_satisfies": True,
                "filelock_3_32_3_satisfies": False,
                "source": "P0A resolver metadata",
            }
        ],
        "success": not rejected,
    }


def exact_filelock_transition(
    before: Mapping[str, str], after: Mapping[str, str]
) -> dict[str, Any]:
    """Require filelock to be the only changed installed distribution."""
    left = {canonical_name(name): value for name, value in before.items()}
    right = {canonical_name(name): value for name, value in after.items()}
    changes = [
        {"name": name, "before": left.get(name), "after": right.get(name)}
        for name in sorted(set(left) | set(right))
        if left.get(name) != right.get(name)
    ]
    expected = [{"name": "filelock", "before": BEFORE_VERSION, "after": AFTER_VERSION}]
    return {"success": changes == expected, "changes": changes, "expected_changes": expected}


def private_directory_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    info = resolved.stat()
    mode = stat.S_IMODE(info.st_mode)
    return {
        "path": str(resolved),
        "owner_uid": info.st_uid,
        "current_uid": os.getuid(),
        "mode": f"{mode:04o}",
        "owned_by_current_user": info.st_uid == os.getuid(),
        "private": mode == 0o700,
    }
