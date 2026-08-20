"""Run the first reviewed ToolContract request-authority canary."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from adapter.protocol import JsonDict
from agent.backends.planner import StaticPlannerBackend
from agent.evals.tool_contract_authority import audit_tool_contract_authority
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.contracts import (
    ContractMaturity,
    ToolContractRuntimePolicy,
    build_default_tool_contract_catalog,
)
from agent.tools.registry import (
    build_default_tool_registry,
    make_tool_result,
)


SCHEMA_VERSION = "openeta.tool_contract_authority_canary.v1"
CANARY_TOOL = "estimate_depth_prior"


def run_estimate_depth_prior_authority_canary(
    root: str | Path,
    *,
    session_id: str,
) -> JsonDict:
    """Exercise invalid repair and valid execution under narrow request authority."""

    store_root = Path(root)
    fixture_root = store_root / "fixtures"
    fixture_root.mkdir(parents=True, exist_ok=True)
    rgb = fixture_root / "agentview-rgb.png"
    depth = fixture_root / "agentview-depth.png"
    prior = fixture_root / "prior-depth.npy"
    Image.new("RGB", (4, 4), (32, 64, 96)).save(rgb)
    Image.new("I;16", (4, 4), 1000).save(depth)
    prior.write_bytes(b"reviewed-estimate-depth-prior-authority-canary")

    tools = build_default_tool_registry()
    catalog = build_default_tool_contract_catalog(tools.list())
    contract = catalog.get(CANARY_TOOL)
    if contract.maturity is not ContractMaturity.VERIFIED:
        raise ValueError(
            f"{CANARY_TOOL} must be verified before an authority canary, "
            f"observed {contract.maturity.value}"
        )
    policy = ToolContractRuntimePolicy(
        request_validation_authority=frozenset({CANARY_TOOL})
    )
    policy.ensure_valid(catalog)

    def estimate(context):
        return make_tool_result(
            context,
            success=True,
            content="deterministic depth prior completed",
            outputs={
                "prior_depth": str(prior),
                "source_packet_id": context.parameters.get("source_packet_id"),
                "camera_frame_id": context.parameters.get("camera_frame_id"),
            },
        )

    tools.bind_handler(CANARY_TOOL, estimate)
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": CANARY_TOOL,
                    "parameters": {},
                    "reasoning": "exercise reviewed invalid-request authority",
                },
                {
                    "kind": "tool_call",
                    "name": CANARY_TOOL,
                    "parameters": {
                        "source_packet_id": "packet-authority-canary",
                        "camera_frame_id": "agentview",
                    },
                    "reasoning": "repair with the exact visible packet reference",
                },
            ]
        ),
        max_validation_retries=1,
        tool_contract_catalog=catalog,
        tool_contract_policy=policy,
    )
    pipeline = ActionPipeline(
        tool_contract_catalog=catalog,
        tool_contract_policy=policy,
    )
    runtime = OpenEtaAgentRuntime(
        planner=planner,
        pipeline=pipeline,
        tools=tools,
        memory=AgentMemory(store=JsonMemoryStore(store_root)),
    )
    runtime.start_session(
        task="reviewed ToolContract request-authority canary",
        session_id=session_id,
    )
    observation = EnvObservation(
        task="reviewed ToolContract request-authority canary",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                role="scene_primary",
                rgb=[[[32, 64, 96]]],
                depth=[[1.0]],
                intrinsics={
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 1.5,
                    "cy": 1.5,
                    "scale": 1000,
                },
            )
        ],
        robot=RobotState(),
        metadata={
            "step_idx": 0,
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": str(rgb),
                    "packet_id": "packet-authority-canary",
                },
                {
                    "kind": "depth",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": str(depth),
                    "packet_id": "packet-authority-canary",
                },
            ],
        },
    )
    action = runtime.act(observation)

    rollout = store_root / "sessions" / session_id / "rollout"
    manifest_path = rollout / "manifest.json"
    model_calls = _read_jsonl(rollout / "model_calls.jsonl")
    tool_calls = _read_jsonl(rollout / "tool_calls.jsonl")
    traces = []
    for row in model_calls:
        decision = row.get("parsed_decision")
        decision = decision if isinstance(decision, dict) else {}
        metadata = decision.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        trace = metadata.get("tool_contract_shadow_validation")
        if isinstance(trace, dict) and trace.get("tool") == CANARY_TOOL:
            traces.append(trace)
    audit = audit_tool_contract_authority(store_root)
    first = traces[0] if traces else {}
    last = traces[-1] if traces else {}
    action_calls = action.command.get("tool_calls") if isinstance(action.command, dict) else []
    action_calls = action_calls if isinstance(action_calls, list) else []
    executed = [
        call
        for call in action_calls
        if isinstance(call, dict)
        and call.get("name") == CANARY_TOOL
        and isinstance(call.get("result"), dict)
        and call["result"].get("success") is True
    ]
    passed = (
        len(traces) == 2
        and first.get("enforcing") is True
        and first.get("authoritative_validator") == "tool_contract"
        and first.get("contract_accepted") is False
        and last.get("enforcing") is True
        and last.get("authoritative_validator") == "tool_contract"
        and last.get("contract_accepted") is True
        and len(executed) == 1
        and audit.get("conformant") is True
        and audit.get("violation_count") == 0
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = manifest.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    contract_runtime = provenance.get("tool_contract_runtime")
    contract_runtime = contract_runtime if isinstance(contract_runtime, dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "passed": passed,
        "tool": CANARY_TOOL,
        "session_id": session_id,
        "manifest": str(manifest_path),
        "catalog_sha256": contract_runtime.get("catalog_sha256"),
        "policy": contract_runtime.get("policy"),
        "planner_pipeline_alignment": contract_runtime.get(
            "planner_pipeline_alignment"
        ),
        "model_call_count": len(model_calls),
        "durable_tool_event_count": len(tool_calls),
        "request_traces": traces,
        "successful_execution_count": len(executed),
        "authority_audit": audit,
        "interpretation": (
            "Only estimate_depth_prior request validation is contract-authoritative; "
            "gate-repair authority remains empty and executable gates remain legacy."
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
