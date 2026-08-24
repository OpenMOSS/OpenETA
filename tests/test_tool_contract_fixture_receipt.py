from __future__ import annotations

import json

import pytest

from agent.evals.tool_contract_fixture_receipt import (
    build_tool_contract_fixture_receipt,
)
from agent.evals.tool_contract_promotion import (
    build_tool_contract_promotion_dossier,
)


def test_estimate_depth_prior_fixture_receipt_covers_handler_and_gate(tmp_path) -> None:
    receipt = build_tool_contract_fixture_receipt(
        "estimate_depth_prior",
        tmp_path / "fixture",
    )

    assert receipt["schema_version"] == "openeta.tool_contract_fixture_receipt.v1"
    assert receipt["authority"] == "promotion_evidence_only"
    assert receipt["conformant"] is True
    assert receipt["coverage"]["expected_success_outcomes"] == ["completed"]
    assert receipt["coverage"]["observed_success_outcomes"] == ["completed"]
    assert receipt["coverage"]["all_success_outcomes_covered"] is True
    assert receipt["coverage"]["all_declared_outcomes_covered"] is True
    assert receipt["coverage"]["representative_failure_covered"] is True
    assert receipt["coverage"]["runtime_gate_evidence_observed"] is True
    assert receipt["coverage"]["runtime_gate_rejection_observed"] is True
    assert receipt["coverage"]["matched_gate_check_ids"] == [
        "runtime.source_packet_resolution"
    ]
    success, failure, gate = receipt["cases"]
    assert success["semantic_outcome"] == "completed"
    assert success["conformant"] is True
    assert success["artifact_paths"]
    assert failure["semantic_outcome"] == "operational_failure"
    assert failure["diagnostic_codes"] == ["mcp_call_failed"]
    assert failure["conformant"] is True
    assert gate["repair_code"] == "invalid_source_packet"
    assert gate["tool_execution_count"] == 0
    assert gate["authoritative_gate"] == "legacy_runtime"
    assert gate["enforcing"] is False
    assert gate["host_resolution_receipt"]["dispatch_authority"] == "tool_contract"
    assert gate["host_resolution_receipt"]["status"] == "rejected"
    assert gate["conformant"] is True


def test_promotion_dossier_consumes_durable_fixture_receipt(tmp_path) -> None:
    receipt = build_tool_contract_fixture_receipt(
        "estimate_depth_prior",
        tmp_path / "fixture",
    )
    receipt_path = tmp_path / "fixture-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    dossier = build_tool_contract_promotion_dossier(
        "estimate_depth_prior",
        [],
        [receipt_path],
    )

    assert dossier["evidence_checks"][
        "durable_handler_fixture_receipt_conformant"
    ] is True
    assert dossier["evidence_checks"][
        "runtime_gate_rejection_bound_to_check_id"
    ] is True
    assert dossier["fixture_evidence"]["conformant_receipt_count"] == 1
    assert all("fixture" not in item for item in dossier["unresolved_requirements"])
    assert all("gate rejection" not in item for item in dossier["unresolved_requirements"])
    assert dossier["eligible_for_review"] is False


def test_promotion_dossier_consumes_catalog_integration_canary(tmp_path) -> None:
    receipt = build_tool_contract_fixture_receipt(
        "save_memory",
        tmp_path / "fixture",
    )
    receipt_path = tmp_path / "fixture-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    canary_path = tmp_path / "integration-canary.json"
    canary_path.write_text(
        json.dumps(
            {
                "schema_version": "openeta.tool_contract_integration_canary.v1",
                "authority": "promotion_evidence_only",
                "tools": [
                    {
                        "tool": "save_memory",
                        "conformant": True,
                        "valid_request_count": 1,
                        "invalid_request_count": 1,
                        "request_acceptance_parity": True,
                        "observational_only": True,
                        "tool_result_count": 1,
                        "tool_result_violation_count": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dossier = build_tool_contract_promotion_dossier(
        "save_memory",
        [],
        [receipt_path],
        integration_canary_path=canary_path,
    )

    assert dossier["eligible_for_review"] is True
    assert dossier["eligible_for_verified_promotion"] is False
    assert dossier["integration_canary_evidence"]["conformant"] is True
    assert dossier["request_evidence"]["valid_count"] == 1
    assert dossier["request_evidence"]["invalid_count"] == 1
    assert dossier["tool_result_evidence"] == {
        "event_count": 1,
        "violation_count": 0,
        "runs": [],
    }
    assert dossier["unresolved_requirements"] == [
        "three-person review must approve the shared schema and per-tool maturity promotion",
        "a separately scoped authority canary must pass after review",
    ]


@pytest.mark.parametrize(
    "tool_name",
    [
        "python_exec",
        "save_memory",
        "get_memory",
        "delete_memory",
        "compact_memory",
        "propose_calibration_profile",
        "promote_calibration_profile",
        "propose_grasp_strategy",
        "promote_grasp_strategy",
        "register_skill",
        "update_skill",
        "web_search",
        "web_fetch",
        "select_sam3_detection",
        "reject_sam3_detections",
        "camera_pose_to_world",
        "prepare_attachment_probe",
        "compile_grasp_seed",
        "compute_wrist_alignment",
        "propose_wrist_viewpoints",
        "sam3",
        "molmopoint",
        "retrieve_asset_reference",
        "anyplace",
        "grasp_pose_estimate",
        "enhance_depth",
        "create_simulator_env",
        "close_simulator_env",
        "assess_attachment_probe",
        "observe",
        "move_to",
        "follow_eef_trajectory",
        "gripper_control",
        "ik_preview_check",
    ],
)
def test_local_runtime_fixture_receipts_cover_success_failure_and_gate(
    tmp_path,
    tool_name: str,
) -> None:
    receipt = build_tool_contract_fixture_receipt(
        tool_name,
        tmp_path / tool_name,
    )

    assert receipt["conformant"] is True
    assert receipt["coverage"]["all_success_outcomes_covered"] is True
    assert receipt["coverage"]["all_declared_outcomes_covered"] is True
    assert receipt["coverage"]["representative_failure_covered"] is True
    assert receipt["coverage"]["runtime_gate_evidence_observed"] is True
    matched_gate_ids = set(receipt["coverage"]["matched_gate_check_ids"])
    if tool_name == "observe":
        assert receipt["coverage"]["runtime_gate_rejection_observed"] is False
        assert matched_gate_ids == {"runtime.batch_boundary"}
    else:
        assert receipt["coverage"]["runtime_gate_rejection_observed"] is True
        assert "runtime.motion_reconciliation" in matched_gate_ids
    result_cases = [
        item
        for item in receipt["cases"]
        if item.get("boundary") == "ToolRegistry -> production handler"
    ]
    successes = [
        item for item in result_cases if item.get("operational_success") is True
    ]
    failure = next(
        item for item in result_cases if item.get("case") == "handler_dependency_failure"
    )
    gate_cases = [
        item
        for item in receipt["cases"]
        if item.get("boundary") != "ToolRegistry -> production handler"
    ]
    assert successes
    assert all(item["conformant"] is True for item in successes)
    assert failure["semantic_outcome"] == "operational_failure"
    assert failure["conformant"] is True
    if tool_name == "observe":
        assert len(gate_cases) == 1
        gate = gate_cases[0]
        assert gate["case"] == "batch_boundary_allow"
        assert gate["allowed"] is True
    else:
        motion_gate = next(
            item
            for item in gate_cases
            if "runtime.motion_reconciliation"
            in item.get("matched_gate_check_ids", [])
        )
        assert motion_gate["repair_code"] == "motion_reconciliation_required"
        assert motion_gate["authoritative_gate"] == "legacy_runtime"
    assert all(item["conformant"] is True for item in gate_cases)
