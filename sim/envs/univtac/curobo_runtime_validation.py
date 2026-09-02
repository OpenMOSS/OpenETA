"""Parsers for fixed cuRobo example evidence."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

IK_LINE = re.compile(
    r"Success, Solve Time\(s\), hz\s+([0-9.]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+tensor\(([0-9.eE+-]+).*?tensor\(([0-9.eE+-]+)"
)


def parse_ik_metrics(text: str) -> list[dict[str, float]]:
    return [
        {
            "success_ratio": float(success),
            "solve_time_seconds": float(solve),
            "hz": float(hz),
            "position_error": float(position),
            "rotation_error": float(rotation),
        }
        for success, solve, hz, position, rotation in IK_LINE.findall(text)
    ]


def classify_example_failure(message: str) -> str:
    if "module 'warp' has no attribute 'torch'" in message:
        return "curobo_official_example_packaging_failure"
    return "curobo_collision_smoke_failed"


def apply_environment_gate(
    manifest: dict[str, Any], comparisons: Mapping[str, Mapping[str, Any]]
) -> None:
    unchanged = bool(comparisons) and all(bool(item.get("unchanged")) for item in comparisons.values())
    manifest["environment_semantically_unchanged"] = unchanged
    if not unchanged and manifest.get("status") == "completed":
        manifest.update(
            {
                "status": "failed",
                "classification": "environment_fingerprint_changed",
                "author_contact_recommendation": "not_applicable",
            }
        )


def static_audit_process_usable(process: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    """Accept the auditor's raw-ldd warning exit while rejecting execution failures."""
    return bool(
        process.get("returncode") in {0, 1}
        and not process.get("timed_out")
        and process.get("cleanup_complete") is True
        and payload.get("records")
    )


def static_audit_covers_binaries(payload: Mapping[str, Any], binaries: list[Path]) -> bool:
    recorded = {str(Path(item["path"]).resolve()) for item in payload.get("records", [])}
    expected = {str(path.resolve()) for path in binaries}
    return recorded == expected
