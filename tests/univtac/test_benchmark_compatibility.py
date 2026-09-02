from __future__ import annotations

import json
import subprocess
from pathlib import Path

from sim.envs.univtac.benchmark_compatibility import (
    CONTACT_RICH_TASKS,
    FTP1_TASKS,
    benchmark_compatibility,
    classify_rtx5090_recipe,
    cross_version_comparability,
    official_validation_scope,
    tactile_contract_matrix,
    task_semantic_matrix,
)


def test_all_six_ftp1_tasks_have_semantic_entries() -> None:
    matrix = task_semantic_matrix()
    assert tuple(item["task"] for item in matrix["tasks"]) == FTP1_TASKS
    required = {
        "create_actors", "actor_initial_pose", "actor_body_motion_type",
        "actor_density", "reset_noise", "reset_seed_handling", "pre_move",
        "expert_play_once", "check_success", "check_early_stop",
        "task_instruction", "task_metadata", "step_action_budget", "cleanup",
        "save_eval_behavior",
    }
    assert all(required <= set(item["audited_fields"]) for item in matrix["tasks"])


def test_contact_rich_tasks_have_detailed_field_diffs() -> None:
    by_task = {item["task"]: item for item in task_semantic_matrix()["tasks"]}
    for task in CONTACT_RICH_TASKS:
        assert len(by_task[task]["semantic_differences"]) >= 3
        assert all("field" in item and "evidence" in item for item in by_task[task]["semantic_differences"])


def test_50_series_claim_is_not_explicit_5090_recipe() -> None:
    status = classify_rtx5090_recipe(
        support_claim=True,
        explicit_5090=False,
        explicit_sm120=False,
        explicit_ptx_forward_compatibility=False,
    )
    assert status == "claimed_but_not_fully_documented"
    assert status != "explicitly_documented"
    assert status != "unsupported"


def test_missing_sm120_does_not_imply_unsupported() -> None:
    assert classify_rtx5090_recipe(
        support_claim=False,
        explicit_5090=False,
        explicit_sm120=False,
        explicit_ptx_forward_compatibility=False,
    ) == "unresolved"


def test_changed_data_or_physics_forbids_preserved_numeric_comparability() -> None:
    assert cross_version_comparability(
        {"task": "preserved", "physics": "changed", "data": "incompatible"}
    ) == "not_preserved"


def test_task_file_presence_is_not_validation_evidence() -> None:
    scope = official_validation_scope()
    for record in scope["tasks"].values():
        assert record["code_contains_task"] is True
        assert record["evaluation_validated"] == "unresolved_from_public_source"


def test_e1_e2_e3_are_computed_independently() -> None:
    tactile = tactile_contract_matrix()
    assert tactile["e1_supported"] is True
    assert tactile["e2_supported"] is True
    assert tactile["e3_supported"] is True
    assert tactile["compatibility"]["legacy_ftp1_checkpoint"] == "unresolved"


def test_final_decision_fields_obey_required_boundary() -> None:
    result = benchmark_compatibility(
        legacy_commit="89fa681d6c014cce28300946b7526db808e0b1c1",
        isaac51_commit="371fac67917307026be8f00869fcc1b61c623a9f",
    )
    assert result["rtx5090_build_recipe_status"] == "claimed_but_not_fully_documented"
    assert result["cross_version_numeric_comparability"] == "not_preserved"
    assert result["openeta_adapter_migration_scope"] == "observation_and_runtime"
    assert result["recommended_next_step"] == "clarify_rtx5090_build_recipe_then_isolated_smoke"


def test_structured_outputs_do_not_contain_credentials() -> None:
    payload = json.dumps(
        {
            "tasks": task_semantic_matrix(),
            "tactile": tactile_contract_matrix(),
            "scope": official_validation_scope(),
        }
    ).lower()
    assert "api_key" not in payload
    assert "authorization: bearer" not in payload
    assert "private key" not in payload


def test_user_dirty_documents_are_not_staged() -> None:
    repo = Path(__file__).resolve().parents[2]
    paths = [
        "README.md",
        "docs/architecture.md",
        "docs/vendor-notes.md",
        "sim/README.md",
        "sim/SETUP.md",
    ]
    completed = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", *paths],
        cwd=repo,
        check=False,
    )
    assert completed.returncode == 0
