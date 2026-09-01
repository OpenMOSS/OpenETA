from __future__ import annotations

from agent.evals.tool_contract_authority import audit_tool_contract_authority
from agent.evals.tool_contract_authority_baseline import (
    build_tool_contract_authority_baseline,
)


def test_no_action_authority_baseline_is_current_and_auditable(tmp_path) -> None:
    root = tmp_path / "authority-baseline"
    report = build_tool_contract_authority_baseline(
        root,
        session_id="baseline-fixture",
    )

    assert report["schema_version"] == (
        "openeta.tool_contract_authority_baseline.v1"
    )
    assert len(report["catalog_sha256"]) == 64
    assert report["policy"]["request_validation_authority"] == []
    assert report["policy"]["gate_repair_envelope_authority"] == []
    assert report["planner_pipeline_alignment"]["catalog_match"] is True
    assert report["planner_pipeline_alignment"]["policy_match"] is True
    assert report["model_call_count"] == 0
    assert report["tool_call_count"] == 0

    audit = audit_tool_contract_authority(root)
    assert audit["conformant"] is True
    assert audit["manifest_count"] == 1
    assert audit["violation_count"] == 0
