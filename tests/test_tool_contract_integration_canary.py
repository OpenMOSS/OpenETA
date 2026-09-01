from agent.evals.tool_contract_integration_canary import (
    run_tool_contract_integration_canary,
)


def test_catalog_integration_canary_is_shadow_only_and_conformant(tmp_path) -> None:
    report = run_tool_contract_integration_canary(
        tmp_path / "canary",
        session_prefix="fixture",
    )

    assert report["tool_count"] == 35
    assert report["conformant_tool_count"] == 35
    assert report["all_tools_conformant"] is True
    assert report["shadow_audit"]["mismatch_count"] == 0
    assert report["shadow_audit"]["unexpected_enforcement_count"] == 0
    assert report["result_audit"] == {"event_count": 35, "violation_count": 0}
    assert all(row["valid_request_count"] >= 1 for row in report["tools"])
    assert all(row["invalid_request_count"] >= 1 for row in report["tools"])
    assert all(row["tool_result_count"] == 1 for row in report["tools"])
