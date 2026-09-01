from __future__ import annotations

from agent.evals.tool_contract_authority_canary import (
    run_estimate_depth_prior_authority_canary,
)


def test_reviewed_estimate_depth_prior_request_authority_canary(tmp_path) -> None:
    report = run_estimate_depth_prior_authority_canary(
        tmp_path / "authority-canary",
        session_id="authority-canary-fixture",
    )

    assert report["passed"] is True
    assert report["tool"] == "estimate_depth_prior"
    assert report["policy"] == {
        "schema_version": "openeta.tool_contract_runtime_policy.v1",
        "request_validation_authority": ["estimate_depth_prior"],
        "gate_repair_envelope_authority": [],
        "executable_gate_authority": "legacy_runtime",
    }
    assert report["planner_pipeline_alignment"]["catalog_match"] is True
    assert report["planner_pipeline_alignment"]["policy_match"] is True
    assert report["model_call_count"] == 2
    assert report["successful_execution_count"] == 1
    assert report["authority_audit"]["violation_count"] == 0
    first, repaired = report["request_traces"]
    assert first["contract_maturity"] == "verified"
    assert first["enforcing"] is True
    assert first["contract_accepted"] is False
    assert repaired["contract_accepted"] is True

