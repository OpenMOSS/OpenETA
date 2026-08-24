from dataclasses import replace

import pytest

from agent.evals.tool_contract_catalog_authority_canary import (
    run_tool_contract_catalog_authority_canary,
)
from agent.tools.contracts import (
    ContractMaturity,
    ToolContractCatalog,
    build_default_tool_contract_catalog,
)
from agent.tools.registry import build_default_tool_registry


def _reviewed_catalog() -> ToolContractCatalog:
    current = build_default_tool_contract_catalog(build_default_tool_registry().list())
    return ToolContractCatalog(
        replace(contract, maturity=ContractMaturity.VERIFIED, coverage_gaps=())
        for contract in current.list()
    )


def test_catalog_authority_canary_is_request_only_and_auditable(tmp_path) -> None:
    report = run_tool_contract_catalog_authority_canary(
        tmp_path / "canary",
        session_id="reviewed-catalog-fixture",
        tool_names=["save_memory", "move_to"],
        catalog=_reviewed_catalog(),
    )

    assert report["passed"] is True
    assert report["tool_count"] == 2
    assert report["passed_tool_count"] == 2
    assert report["model_call_count"] == 4
    assert report["tool_execution_count"] == 0
    assert report["world_mutation_count"] == 0
    assert report["policy"]["request_validation_authority"] == [
        "move_to",
        "save_memory",
    ]
    assert report["policy"]["gate_repair_envelope_authority"] == []
    assert report["policy"]["executable_gate_authority"] == "legacy_runtime"
    assert report["authority_audit"]["violation_count"] == 0


def test_catalog_authority_canary_fails_closed_before_review(tmp_path) -> None:
    reviewed = _reviewed_catalog()
    unreviewed = ToolContractCatalog(
        replace(contract, maturity=ContractMaturity.DECLARED)
        if contract.name == "save_memory"
        else contract
        for contract in reviewed.list()
    )
    with pytest.raises(ValueError, match="not_verified=save_memory:declared"):
        run_tool_contract_catalog_authority_canary(
            tmp_path / "canary",
            session_id="unreviewed-catalog-fixture",
            tool_names=["save_memory"],
            catalog=unreviewed,
        )
