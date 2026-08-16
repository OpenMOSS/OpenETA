"""One-way migrations for persisted runtime memory."""

from __future__ import annotations

from typing import Any


# Tombstones only: no runtime behavior may read, update, or recreate these facts.
REMOVED_TASK_POLICY_FACT_KEYS = frozenset(
    {
        "grasp_candidate_policy",
        "anygrasp_candidate_policy",
        "grasp_reestimation",
        "grasp_lift_probe",
        "grasp_execution",
        "grasp_recovery",
        "grasp_estimation_recovery",
        "placement_release",
        "completed_placement_subgoals",
        "attachment_gate",
    }
)


def purge_removed_task_policy_facts(facts: dict[str, Any]) -> list[str]:
    """Delete persisted host task-progress entries from an old memory mapping."""

    removed = [key for key in REMOVED_TASK_POLICY_FACT_KEYS if facts.pop(key, None) is not None]
    return sorted(removed)


def is_removed_task_policy_key(key: object) -> bool:
    """Return whether a key belongs to the deleted host task-policy namespace."""

    return str(key) in REMOVED_TASK_POLICY_FACT_KEYS
