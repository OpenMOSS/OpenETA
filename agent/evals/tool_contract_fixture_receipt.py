"""Build durable, deterministic runtime fixture receipts for promotion review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from adapter.protocol import EnvObservation, JsonDict, RobotState
from agent.runtime.actions import PipelineStatus
from agent.runtime.memory import AgentMemory
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerDecision
from agent.runtime.skills import build_default_skill_registry
from agent.tools.contracts import (
    ToolContract,
    build_default_tool_contract_catalog,
    check_tool_result_conformance,
)
from agent.tools.handlers import build_depth_prior_handler
from agent.tools.registry import build_default_tool_registry


SCHEMA_VERSION = "openeta.tool_contract_fixture_receipt.v1"
FixtureBuilder = Callable[[Path], JsonDict]


def build_tool_contract_fixture_receipt(
    tool_name: str,
    artifact_root: str | Path,
) -> JsonDict:
    """Execute a registered fixture and return evidence without changing authority."""

    try:
        builder = _FIXTURE_BUILDERS[tool_name]
    except KeyError as exc:
        raise ValueError(f"no runtime fixture is registered for {tool_name}") from exc
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = builder(root)
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": tool_name,
        "authority": "promotion_evidence_only",
        "artifact_root": str(root),
        **receipt,
    }


def _estimate_depth_prior_fixture(root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    catalog = build_default_tool_contract_catalog(registry.list())
    contract = catalog.get("estimate_depth_prior")
    rgb_path = root / "fixture-rgb.png"
    pixels = np.zeros((3, 3, 3), dtype=np.uint8)
    pixels[..., 0] = 64
    pixels[..., 1] = 128
    pixels[..., 2] = 192
    Image.fromarray(pixels).save(rgb_path)
    host_parameters = {
        "rgb": str(rgb_path),
        "intrinsics": {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 1.0,
            "cy": 1.0,
            "scale": 1000.0,
        },
        "camera_id": "agentview",
        "source_packet_id": "fixture-packet",
        "camera_frame_id": "agentview",
    }

    def successful_estimate(_request: JsonDict) -> JsonDict:
        return {
            "success": True,
            "details": {
                "backend": "deterministic_fixture",
                "model": "fixture-depth-v1",
                "depth_m": [[1.0, 1.1], [1.2, 1.3]],
                "confidence": [[0.9, 0.8], [0.7, 0.6]],
            },
        }

    registry.bind_handler(
        "estimate_depth_prior",
        build_depth_prior_handler(successful_estimate, output_root=root / "success"),
    )
    success = registry.call(
        "estimate_depth_prior",
        dict(host_parameters),
        metadata={"session_id": "fixture-success"},
    )
    handler_success_case = _result_case(
        contract,
        "handler_success",
        success.details,
    )

    def failing_estimate(_request: JsonDict) -> JsonDict:
        raise TimeoutError("deterministic fixture timeout")

    registry.bind_handler(
        "estimate_depth_prior",
        build_depth_prior_handler(failing_estimate, output_root=root / "failure"),
        replace=True,
    )
    failure = registry.call(
        "estimate_depth_prior",
        dict(host_parameters),
        metadata={"session_id": "fixture-failure"},
    )
    failure_case = _result_case(contract, "handler_failure", failure.details)
    gate_case = _source_packet_gate_fixture(registry)
    expected_success_outcomes = sorted(
        outcome.semantic_outcome
        for outcome in contract.outcomes
        if outcome.operational_success
    )
    observed_success_outcomes = sorted(
        {
            str(case.get("semantic_outcome") or "")
            for case in (handler_success_case,)
            if case.get("operational_success") is True
        }
    )
    representative_failure_covered = (
        failure_case.get("semantic_outcome") == "operational_failure"
        and failure_case.get("conformant") is True
    )
    all_success_outcomes_covered = (
        observed_success_outcomes == expected_success_outcomes
    )
    conformant = bool(
        all_success_outcomes_covered
        and representative_failure_covered
        and gate_case.get("conformant") is True
    )
    return {
        "contract_schema_version": contract.schema_version,
        "contract_maturity": contract.maturity.value,
        "handler_implementation": "agent.tools.handlers.build_depth_prior_handler",
        "host_resolver_id": contract.host_resolution.resolver,
        "host_resolver_implementation": contract.host_resolution.implementation,
        "cases": [handler_success_case, failure_case, gate_case],
        "coverage": {
            "expected_success_outcomes": expected_success_outcomes,
            "observed_success_outcomes": observed_success_outcomes,
            "all_success_outcomes_covered": all_success_outcomes_covered,
            "representative_failure_covered": representative_failure_covered,
            "runtime_gate_rejection_observed": gate_case.get("blocked") is True,
            "matched_gate_check_ids": gate_case.get("matched_gate_check_ids", []),
        },
        "conformant": conformant,
    }


def _result_case(
    contract: ToolContract,
    case_name: str,
    details: JsonDict,
) -> JsonDict:
    violations = check_tool_result_conformance(contract, details)
    diagnostics = details.get("diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    artifacts = details.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, list) else []
    return {
        "case": case_name,
        "boundary": "ToolRegistry -> production handler",
        "executed": True,
        "operational_success": details.get("operational_success") is True,
        "semantic_outcome": str(details.get("semantic_outcome") or ""),
        "diagnostic_codes": sorted(
            {
                str(item.get("code") or "")
                for item in diagnostics
                if isinstance(item, dict) and item.get("code")
            }
        ),
        "artifact_paths": sorted(
            {
                str(item.get("path") or "")
                for item in artifacts
                if isinstance(item, dict) and item.get("path")
            }
        ),
        "conformant": not violations,
        "violations": [violation.to_dict() for violation in violations],
    }


def _source_packet_gate_fixture(registry) -> JsonDict:
    memory = AgentMemory()
    memory.start_session(task="ToolContract fixture")
    observation = EnvObservation(
        task="ToolContract fixture",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 0},
    )
    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="estimate_depth_prior",
            parameters={
                "source_packet_id": "missing-fixture-packet",
                "camera_frame_id": "agentview",
            },
        ),
        observation=observation,
        tools=registry,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    repair = plan.metadata.get("repair_bundle")
    repair = repair if isinstance(repair, dict) else {}
    shadow = repair.get("contract_shadow_validation")
    shadow = shadow if isinstance(shadow, dict) else {}
    resolution_receipt = plan.metadata.get("host_resolution_receipt")
    resolution_receipt = (
        resolution_receipt if isinstance(resolution_receipt, dict) else {}
    )
    matched = shadow.get("matched_gate_bindings")
    matched = matched if isinstance(matched, list) else []
    return {
        "case": "host_resolution_gate_rejection",
        "boundary": "ActionPipeline -> host resolver",
        "blocked": plan.status is PipelineStatus.BLOCKED,
        "tool_execution_count": sum(
            call.status is PipelineStatus.EXECUTED for call in plan.tool_calls
        ),
        "repair_code": str(repair.get("code") or ""),
        "matched_gate_check_ids": sorted(
            {
                str(item.get("check_id") or "")
                for item in matched
                if isinstance(item, dict) and item.get("check_id")
            }
        ),
        "authoritative_gate": str(shadow.get("authoritative_gate") or ""),
        "enforcing": shadow.get("enforcing") is True,
        "host_resolution_receipt": resolution_receipt,
        "conformant": bool(
            plan.status is PipelineStatus.BLOCKED
            and not any(
                call.status is PipelineStatus.EXECUTED for call in plan.tool_calls
            )
            and repair.get("code") == "invalid_source_packet"
            and shadow.get("conformant") is True
            and shadow.get("authoritative_gate") == "legacy_runtime"
            and resolution_receipt.get("status") == "rejected"
            and resolution_receipt.get("dispatch_authority") == "tool_contract"
            and resolution_receipt.get("resolver_id")
            == "openeta.host_resolver.estimate_depth_prior.v1"
            and any(
                isinstance(item, dict)
                and item.get("check_id") == "runtime.source_packet_resolution"
                for item in matched
            )
        ),
        "contract_shadow_validation": shadow,
    }


_FIXTURE_BUILDERS: dict[str, FixtureBuilder] = {
    "estimate_depth_prior": _estimate_depth_prior_fixture,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    receipt = build_tool_contract_fixture_receipt(args.tool, args.artifact_root)
    payload = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if receipt["conformant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
