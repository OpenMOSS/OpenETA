from __future__ import annotations

from pathlib import Path

from scripts.univtac.audit_ftp1_protocol import (
    CANONICAL_COMMIT,
    audit_protocol,
    classify_task_sets,
)


CANONICAL_CHECKOUT = Path(
    "/home/ubuntu/wybcode/.worktrees/ftp1-policy/r02-official-89fa681"
)


def test_task_scopes_remain_distinct() -> None:
    scopes = classify_task_sets(
        environment_tasks=["insert_hole", "insert_tube", "extra_env"],
        instruction_tasks=["insert_hole", "insert_tube", "not_an_env"],
        batch_tasks=["insert_hole"],
        checkpoint_tasks=["insert_hole"],
        single_task_defaults=["insert_tube"],
    )

    assert scopes["environment_supported_tasks"] == [
        "extra_env",
        "insert_hole",
        "insert_tube",
    ]
    assert scopes["ftp1_code_supported_tasks"] == ["insert_hole", "insert_tube"]
    assert scopes["batch_evaluated_tasks"] == ["insert_hole"]
    assert scopes["checkpoint_covered_tasks"] == ["insert_hole"]
    assert scopes["single_task_default_tasks"] == ["insert_tube"]


def test_canonical_public_protocol_is_parsed_without_simulator() -> None:
    audit, source_manifest = audit_protocol(CANONICAL_CHECKOUT)

    assert audit["canonical_commit"] == CANONICAL_COMMIT
    assert audit["default_start_seed"] == 1000000
    assert audit["batch_episode_count"] == 100
    assert audit["task_sets"]["single_task_default_tasks"] == ["insert_HDMI"]
    assert audit["paper_protocol_resolution"] == "unresolved_from_public_code"
    assert all(item["from_canonical_checkout"] for item in source_manifest["files"])


def test_batch_and_until_success_are_different_paths() -> None:
    audit, _ = audit_protocol(CANONICAL_CHECKOUT)
    paths = {entry["entry_point"]: entry for entry in audit["protocol_paths"]}

    batch = paths["UniVTAC/scripts/shell/eval_ftp1_batch.sh"]
    until = paths["UniVTAC/scripts/eval_ftp1_until_success.py"]
    assert batch["whether_until_success_semantics_is_used"] is False
    assert batch["whether_seed_manifest_is_used"] is False
    assert until["whether_until_success_semantics_is_used"] is True
