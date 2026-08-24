from __future__ import annotations

from agent.evals.tool_contract_promotion_campaign import (
    build_tool_contract_review_request,
    build_tool_contract_promotion_campaign,
    render_tool_contract_promotion_review,
)


def test_promotion_campaign_inventories_all_remaining_tools_without_granting_authority() -> None:
    report = build_tool_contract_promotion_campaign()

    assert report["schema_version"] == "openeta.tool_contract_promotion_campaign.v1"
    assert report["authority"] == "promotion_evidence_only"
    assert report["catalog_tool_count"] == 35
    assert report["campaign_target_count"] == 34
    assert report["verified_count"] == 35
    assert report["declared_count"] == 0
    assert report["request_parity_ready_count"] == 34
    assert report["fixture_registered_count"] == 34
    assert report["resolver_binding_audit_conformant"] is True
    assert all(
        row["eligible_for_verified_promotion"] is False
        for row in report["tools"]
        if row["tool"] != "estimate_depth_prior"
    )


def test_promotion_campaign_adds_effect_specific_requirements() -> None:
    report = build_tool_contract_promotion_campaign(
        fixture_tools={"save_memory"},
        fixture_receipts=[
            {
                "tool": "save_memory",
                "conformant": True,
                "coverage": {
                    "all_success_outcomes_covered": True,
                    "all_declared_outcomes_covered": True,
                    "representative_failure_covered": True,
                    "runtime_gate_evidence_observed": True,
                    "runtime_gate_rejection_observed": True,
                    "matched_gate_check_ids": ["runtime.motion_reconciliation"],
                },
            }
        ],
    )
    rows = {row["tool"]: row for row in report["tools"]}

    assert rows["save_memory"]["risk_tier"] == "local_deterministic"
    assert rows["save_memory"]["checks"]["production_fixture_registered"] is True
    assert rows["save_memory"]["checks"]["successful_outcomes_fixture_covered"] is True
    assert rows["save_memory"]["checks"]["representative_failure_fixture_covered"] is True
    assert rows["save_memory"]["checks"]["all_declared_outcomes_fixture_covered"] is True
    assert rows["save_memory"]["checks"]["gate_repair_binding_observed"] is True
    assert "world_mutating_execution_receipt" not in rows["save_memory"]["checks"]

    assert rows["move_to"]["risk_tier"] == "world_mutating"
    assert rows["move_to"]["checks"]["world_mutating_execution_receipt"] is False
    assert "freshness_and_invalidation" in rows["move_to"]["checks"]

    assert rows["sam3"]["risk_tier"] == "evidence_sensitive"
    assert rows["sam3"]["checks"]["remote_backend_semantics"] is False


def test_promotion_campaign_consumes_shadow_only_integration_canary() -> None:
    canary_row = {
        "tool": "save_memory",
        "conformant": True,
        "valid_request_count": 1,
        "invalid_request_count": 1,
        "request_acceptance_parity": True,
        "observational_only": True,
        "tool_result_count": 1,
        "tool_result_violation_count": 0,
    }
    report = build_tool_contract_promotion_campaign(
        integration_canary={"tools": [canary_row]}
    )
    row = next(item for item in report["tools"] if item["tool"] == "save_memory")

    assert row["integration_canary_present"] is True
    assert row["checks"]["live_result_conformance"] is True
    assert row["checks"]["valid_invalid_live_shadow_parity"] is True
    assert row["checks"]["three_person_promotion_review"] is False
    assert row["checks"]["authority_canary_passed"] is False


def test_promotion_review_renderer_states_authority_boundary() -> None:
    report = build_tool_contract_promotion_campaign()
    rendered = render_tool_contract_promotion_review(report)

    assert "remaining 34 public tools" in rendered
    assert "Gate-repair-envelope authority remains empty" in rendered
    assert "| move_to | world_mutating |" in rendered


def test_review_and_batch_canary_are_explicit_evidence_not_maturity_inference() -> None:
    review = {
        "review_status": "approved",
        "decisions": {
            "remaining_tool_promotions": {
                "status": "approved",
                "approved_tools": ["save_memory"],
                "request_validation_authority_canary": True,
                "gate_repair_envelope_authority": False,
                "executable_gate_authority": "legacy_runtime",
            }
        },
    }
    canary = {
        "schema_version": "openeta.tool_contract_catalog_authority_canary.v1",
        "passed": True,
        "policy": {
            "request_validation_authority": ["save_memory"],
            "gate_repair_envelope_authority": [],
            "executable_gate_authority": "legacy_runtime",
        },
        "tool_execution_count": 0,
        "world_mutation_count": 0,
        "tools": [{"tool": "save_memory", "passed": True}],
        "authority_audit": {"conformant": True, "violation_count": 0},
    }
    report = build_tool_contract_promotion_campaign(
        review_decision=review,
        authority_canary=canary,
    )
    rows = {row["tool"]: row for row in report["tools"]}

    assert rows["save_memory"]["checks"]["three_person_promotion_review"] is True
    assert rows["save_memory"]["checks"]["authority_canary_passed"] is True
    assert rows["get_memory"]["checks"]["three_person_promotion_review"] is False
    assert rows["estimate_depth_prior"]["checks"]["three_person_promotion_review"] is False


def test_review_request_cannot_be_mistaken_for_approval() -> None:
    request = build_tool_contract_review_request(
        build_tool_contract_promotion_campaign()
    )

    assert request["status"] == "pending_three_person_review"
    assert request["tool_count"] == 34
    assert "review_status" not in request
    assert request["requested_decisions"]["gate_repair_envelope_authority"] == {
        "requested": False,
        "allowlist": [],
    }
