"""Run a post-review, request-validation-only authority canary for named tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter.protocol import EnvObservation, JsonDict, RobotState
from agent.backends.planner import StaticPlannerBackend
from agent.evals.tool_contract_authority import audit_tool_contract_authority
from agent.evals.tool_contract_readiness import _request_cases
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.contracts import (
    ContractMaturity,
    ToolContractCatalog,
    ToolContractRuntimePolicy,
    build_default_tool_contract_catalog,
)
from agent.tools.registry import build_default_tool_registry


SCHEMA_VERSION = "openeta.tool_contract_catalog_authority_canary.v1"


def run_tool_contract_catalog_authority_canary(
    root: str | Path,
    *,
    session_id: str,
    tool_names: list[str],
    catalog: ToolContractCatalog | None = None,
) -> JsonDict:
    """Exercise reviewed request authority without dispatching any tool call.

    Every requested tool must already be verified. The canary records one invalid
    and one corrected-valid Planner attempt per tool with ToolContract as the
    authoritative request validator. Gate-repair authority stays empty and no
    executable tool or world mutation is allowed in this canary.
    """

    names = list(dict.fromkeys(str(name).strip() for name in tool_names if str(name).strip()))
    if not names:
        raise ValueError("at least one tool is required for the authority canary")
    tools = build_default_tool_registry()
    resolved = catalog or build_default_tool_contract_catalog(tools.list())
    unknown: list[str] = []
    unverified: list[str] = []
    for name in names:
        try:
            contract = resolved.get(name)
        except KeyError:
            unknown.append(name)
            continue
        if contract.maturity is not ContractMaturity.VERIFIED:
            unverified.append(f"{name}:{contract.maturity.value}")
    if unknown or unverified:
        parts = []
        if unknown:
            parts.append("unknown=" + ",".join(sorted(unknown)))
        if unverified:
            parts.append("not_verified=" + ",".join(sorted(unverified)))
        raise ValueError(
            "authority canary requires reviewed verified contracts; " + "; ".join(parts)
        )

    policy = ToolContractRuntimePolicy(
        request_validation_authority=frozenset(names)
    )
    policy.ensure_valid(resolved)
    payloads: list[JsonDict] = []
    for name in names:
        cases = _request_cases(resolved.get(name))
        valid = next(parameters for case, parameters in cases if case == "valid_minimal")
        invalid = next(parameters for case, parameters in cases if case != "valid_minimal")
        payloads.extend(
            [
                {
                    "kind": "tool_call",
                    "name": name,
                    "parameters": invalid,
                    "reasoning": "post-review invalid request authority canary",
                },
                {
                    "kind": "tool_call",
                    "name": name,
                    "parameters": valid,
                    "reasoning": "post-review corrected request authority canary",
                },
            ]
        )

    planner = ToolCallingPlanner(
        StaticPlannerBackend(payloads),
        max_validation_retries=1,
        tool_contract_catalog=resolved,
        tool_contract_policy=policy,
    )
    pipeline = ActionPipeline(
        tool_contract_catalog=resolved,
        tool_contract_policy=policy,
    )
    store_root = Path(root)
    runtime = OpenEtaAgentRuntime(
        planner=planner,
        pipeline=pipeline,
        tools=tools,
        memory=AgentMemory(store=JsonMemoryStore(store_root)),
    )
    runtime.start_session(
        task="Reviewed ToolContract catalog request-authority canary",
        session_id=session_id,
    )
    observation = EnvObservation(
        task="Reviewed ToolContract catalog request-authority canary",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 0},
    )
    for _name in names:
        # Planner-only by design. We do not compile or dispatch the returned
        # decision, so this canary cannot mutate memory, tools, or the world.
        planner.plan(
            observation,
            memory=runtime.memory,
            tools=runtime.tools,
            skills=runtime.skills,
        )

    rollout = store_root / "sessions" / session_id / "rollout"
    model_calls = _read_jsonl(rollout / "model_calls.jsonl")
    tool_calls = _read_jsonl(rollout / "tool_calls.jsonl")
    traces_by_tool: dict[str, list[JsonDict]] = {name: [] for name in names}
    for row in model_calls:
        decision = row.get("parsed_decision")
        decision = decision if isinstance(decision, dict) else {}
        metadata = decision.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        trace = metadata.get("tool_contract_shadow_validation")
        if not isinstance(trace, dict):
            continue
        name = str(trace.get("tool") or "")
        if name in traces_by_tool:
            traces_by_tool[name].append(trace)

    rows: list[JsonDict] = []
    for name in names:
        traces = traces_by_tool[name]
        invalid = [trace for trace in traces if trace.get("contract_accepted") is False]
        valid = [trace for trace in traces if trace.get("contract_accepted") is True]
        passed = bool(
            len(traces) == 2
            and len(invalid) == 1
            and len(valid) == 1
            and all(trace.get("enforcing") is True for trace in traces)
            and all(
                trace.get("authoritative_validator") == "tool_contract"
                for trace in traces
            )
            and all(trace.get("acceptance_match") is True for trace in traces)
        )
        rows.append(
            {
                "tool": name,
                "request_trace_count": len(traces),
                "invalid_request_count": len(invalid),
                "valid_request_count": len(valid),
                "enforcing": all(trace.get("enforcing") is True for trace in traces),
                "passed": passed,
                "traces": traces,
            }
        )

    audit = audit_tool_contract_authority(store_root)
    manifest_path = rollout / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    contract_runtime = provenance.get("tool_contract_runtime")
    contract_runtime = contract_runtime if isinstance(contract_runtime, dict) else {}
    passed = bool(
        all(row["passed"] is True for row in rows)
        and not tool_calls
        and audit.get("conformant") is True
        and audit.get("violation_count") == 0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "post_review_request_validation_canary",
        "passed": passed,
        "session_id": session_id,
        "manifest": str(manifest_path),
        "catalog_sha256": contract_runtime.get("catalog_sha256"),
        "policy": contract_runtime.get("policy"),
        "planner_pipeline_alignment": contract_runtime.get("planner_pipeline_alignment"),
        "tool_count": len(rows),
        "passed_tool_count": sum(row["passed"] is True for row in rows),
        "model_call_count": len(model_calls),
        "tool_execution_count": len(tool_calls),
        "world_mutation_count": 0,
        "tools": rows,
        "authority_audit": audit,
        "interpretation": (
            "This post-review canary grants request-validation authority only. "
            "It does not compile or dispatch a tool call; gate-repair authority "
            "remains empty and executable gates remain legacy_runtime."
        ),
    }


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--tool", action="append", default=[])
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = run_tool_contract_catalog_authority_canary(
        args.root,
        session_id=args.session_id,
        tool_names=args.tool,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
