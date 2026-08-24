"""Run catalog-wide, non-authoritative Planner/ToolRegistry integration canaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from adapter.protocol import EnvObservation, JsonDict, RobotState
from agent.backends.planner import StaticPlannerBackend
from agent.evals.tool_contract_conformance import audit_tool_contract_conformance
from agent.evals.tool_contract_readiness import _request_cases, _schema_fixture
from agent.evals.tool_contract_shadow import audit_tool_contract_shadow
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.skills import build_default_skill_registry
from agent.tools.contracts import (
    ContractMaturity,
    ToolContract,
    build_default_tool_contract_catalog,
)
from agent.tools.registry import (
    ToolExecutionContext,
    build_default_tool_registry,
    make_tool_result,
)


SCHEMA_VERSION = "openeta.tool_contract_integration_canary.v1"


def run_tool_contract_integration_canary(
    root: str | Path,
    *,
    session_prefix: str = "toolcontract-integration",
) -> JsonDict:
    """Exercise request shadow and result recording for every explicit contract.

    The canary deliberately uses a deterministic result adapter. Production
    handler behavior is covered by the separate fixture receipt, while this run
    proves the live Planner retry, ToolRegistry normalization, and durable
    rollout boundaries without granting contract authority.
    """

    store_root = Path(root)
    registry_template = build_default_tool_registry()
    catalog = build_default_tool_contract_catalog(registry_template.list())
    explicit = [
        contract
        for contract in catalog.list()
        if contract.maturity is not ContractMaturity.INFERRED
    ]
    session_by_tool: dict[str, str] = {}
    selected_outcome_by_tool: dict[str, str] = {}
    for contract in explicit:
        request_cases = _request_cases(contract)
        valid = next(parameters for name, parameters in request_cases if name == "valid_minimal")
        invalid = next(
            parameters
            for name, parameters in request_cases
            if name != "valid_minimal"
        )
        tools = build_default_tool_registry()
        tools.bind_handler(contract.name, _canary_handler(contract))
        planner = ToolCallingPlanner(
            StaticPlannerBackend(
                [
                    {
                        "kind": "tool_call",
                        "name": contract.name,
                        "parameters": invalid,
                        "reasoning": "exercise invalid request shadow",
                    },
                    {
                        "kind": "tool_call",
                        "name": contract.name,
                        "parameters": valid,
                        "reasoning": "exercise corrected valid request shadow",
                    },
                ]
            ),
            max_validation_retries=1,
            tool_contract_catalog=catalog,
        )
        runtime = OpenEtaAgentRuntime(
            planner=planner,
            tools=tools,
            skills=build_default_skill_registry(),
            memory=AgentMemory(store=JsonMemoryStore(store_root)),
        )
        session_id = f"{session_prefix}-{contract.name.replace('_', '-')}"
        session_by_tool[contract.name] = session_id
        runtime.start_session(
            task=f"ToolContract integration canary for {contract.name}",
            session_id=session_id,
        )
        observation = EnvObservation(
            task=f"ToolContract integration canary for {contract.name}",
            cameras=[],
            robot=RobotState(),
            metadata={"step_idx": 0},
        )
        # Run the actual Planner validation/retry path. Some corrected requests
        # can still fail a later host evidence check; that is orthogonal to this
        # request-schema canary and remains covered by fixture gate receipts.
        planner.plan(
            observation,
            memory=runtime.memory,
            tools=tools,
            skills=runtime.skills,
        )
        result = tools.call(
            contract.name,
            dict(valid),
            observation=observation,
            metadata={
                "session_id": session_id,
                "integration_canary": True,
            },
        )
        selected_outcome_by_tool[contract.name] = str(
            result.details.get("semantic_outcome") or ""
        )

    shadow = audit_tool_contract_shadow(store_root)
    conformance = audit_tool_contract_conformance(store_root, catalog=catalog)
    rows: list[JsonDict] = []
    for contract in explicit:
        request_records = [
            record for record in shadow["records"] if record["tool"] == contract.name
        ]
        violations = [
            item
            for item in conformance["violations"]
            if item.get("tool") == contract.name
        ]
        result_count = int(conformance["tool_counts"].get(contract.name, 0))
        valid_count = sum(
            record.get("contract_accepted") is True for record in request_records
        )
        invalid_count = sum(
            record.get("contract_accepted") is False for record in request_records
        )
        parity = bool(request_records) and all(
            record.get("acceptance_match") is True for record in request_records
        )
        observational_only = not any(
            record.get("enforcing") is True for record in request_records
        )
        conformant = bool(
            valid_count
            and invalid_count
            and parity
            and observational_only
            and result_count == 1
            and not violations
        )
        session_id = session_by_tool[contract.name]
        rollout = store_root / "sessions" / session_id / "rollout"
        rows.append(
            {
                "tool": contract.name,
                "contract_maturity": contract.maturity.value,
                "session_id": session_id,
                "model_calls": str(rollout / "model_calls.jsonl"),
                "tool_calls": str(rollout / "tool_calls.jsonl"),
                "request_record_count": len(request_records),
                "valid_request_count": valid_count,
                "invalid_request_count": invalid_count,
                "request_acceptance_parity": parity,
                "observational_only": observational_only,
                "tool_result_count": result_count,
                "tool_result_violation_count": len(violations),
                "semantic_outcome": selected_outcome_by_tool[contract.name],
                "conformant": conformant,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "promotion_evidence_only",
        "result_adapter": "deterministic_contract_shape_not_production_backend",
        "tool_count": len(rows),
        "conformant_tool_count": sum(row["conformant"] is True for row in rows),
        "all_tools_conformant": all(row["conformant"] is True for row in rows),
        "shadow_audit": {
            "record_count": shadow["shadow_record_count"],
            "mismatch_count": shadow["acceptance_mismatch_count"],
            "unexpected_enforcement_count": shadow["unexpected_enforcement_count"],
        },
        "result_audit": {
            "event_count": conformance["tool_event_count"],
            "violation_count": conformance["violation_count"],
        },
        "tools": rows,
        "interpretation": (
            "This canary proves live Planner retry/shadow and ToolRegistry/rollout "
            "integration only. Production behavior comes from fixture receipts; "
            "runtime authority and verified maturity are unchanged."
        ),
    }


def _canary_handler(contract: ToolContract):
    outcome = next(
        item for item in contract.outcomes if item.operational_success is True
    )
    outputs_value = _schema_fixture(outcome.output_schema)
    outputs = dict(outputs_value) if isinstance(outputs_value, Mapping) else {}

    def handler(context: ToolExecutionContext):
        diagnostics = (
            [{"code": "integration_canary_diagnostic"}]
            if outcome.diagnostics_required
            else []
        )
        recovery = (
            [{"action": "integration_canary_recovery"}]
            if outcome.recovery_required
            else []
        )
        kwargs: dict[str, Any] = {}
        if contract.effect == "world_mutating":
            kwargs.update(
                state_delta={"integration_canary": True},
                environment_receipt={
                    "schema_version": "openeta.environment_receipt.v1",
                    "status": "acknowledged",
                    "integration_canary": True,
                },
            )
        return make_tool_result(
            context,
            success=True,
            content="deterministic ToolContract integration canary completed",
            outputs=outputs,
            diagnostics=diagnostics,
            semantic_outcome=outcome.semantic_outcome,
            recovery_options=recovery,
            **kwargs,
        )

    return handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--session-prefix", default="toolcontract-integration")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = run_tool_contract_integration_canary(
        args.root,
        session_prefix=args.session_prefix,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["all_tools_conformant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
