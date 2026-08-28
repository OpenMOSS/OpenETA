from __future__ import annotations

from pathlib import Path

from agent.evals.tool_contract_migration_status import (
    audit_tool_contract_migration_status,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_tool_contract_migration_is_complete_and_narrowly_authoritative() -> None:
    report = audit_tool_contract_migration_status(
        REPO_ROOT,
        test_passed=1353,
        test_skipped=12,
        test_warnings=36,
        harness_revision=271,
    )

    assert report["internal_conformant"] is True
    assert report["implementation_ready_for_review"] is True
    assert report["goal_complete"] is True
    assert report["structural_issues"] == []
    assert report["catalog_summary"]["tool_count"] == 35
    assert report["catalog_sha256"] == (
        "367f4ca47e1fa02f7e20fca06e31f79fa890d0e2916340ddec5a90623ef9ad9b"
    )
    assert report["catalog_summary"]["maturity_counts"] == {
        "inferred": 0,
        "declared": 0,
        "verified": 35,
    }
    assert all(
        row["current"] is True
        for row in report["generated_projections"].values()
    )
    assert all(
        row["compatible"] is True
        for row in report["typed_toolchains"].values()
    )
    statuses = {row["id"]: row["status"] for row in report["requirements"]}
    assert statuses["machine_readable_contract_catalog"] == "complete"
    assert statuses["agent_autonomy_without_task_stage_machine"] == "complete"
    assert statuses["three_person_schema_and_promotion_review"] == "complete"
    assert statuses["shared_rfc_sync"] == "complete"
    assert statuses["verified_authority_canary"] == "complete"
    assert report["external_blockers"] == []
    assert report["local_checks"]["estimate_depth_prior_authority_canary_safe"] is True
    assert report["local_checks"]["remaining_tool_review_approved"] is True
    assert report["local_checks"]["remaining_tool_authority_canary_safe"] is True
    assert report["local_checks"]["remaining_tool_promotions_complete"] is True
    assert report["local_checks"]["remaining_review_request_fulfilled"] is True
    assert report["local_checks"]["shared_rfc_synced"] is True
