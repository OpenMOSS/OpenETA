from __future__ import annotations

import json

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
    assert receipt["coverage"] == {
        "expected_success_outcomes": ["completed"],
        "observed_success_outcomes": ["completed"],
        "all_success_outcomes_covered": True,
        "representative_failure_covered": True,
        "runtime_gate_rejection_observed": True,
        "matched_gate_check_ids": ["runtime.source_packet_resolution"],
    }
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
