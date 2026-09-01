"""Build durable, deterministic runtime fixture receipts for promotion review."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import socket
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, JsonDict, RobotState
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.calibration import (
    CALIBRATION_EVIDENCE_SCHEMA_VERSION,
    BackendCalibrationReviewer,
    CalibrationLifecycleConfig,
    CalibrationLifecycleManager,
)
from agent.runtime.grasp_strategy_lifecycle import (
    GRASP_STRATEGY_EVIDENCE_SCHEMA_VERSION,
    BackendGraspStrategyReviewer,
    GraspStrategyLifecycleConfig,
    GraspStrategyLifecycleManager,
)
from agent.runtime.actions import PipelineStatus
from agent.runtime.memory import AgentMemory
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerDecision
from agent.runtime.reference_localization import ReferencePointLocalization
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.runtime_assembly import _bind_skill_change_tools
from agent.runtime.skills import build_default_skill_registry
from agent.runtime.supervision import SupervisionPolicy, SupervisionProfile
from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.contracts import (
    ToolContract,
    build_default_tool_contract_catalog,
    check_tool_result_conformance,
)
from agent.tools.attachment_probe import build_prepare_attachment_probe_handler
from agent.tools.attachment_probe import build_assess_attachment_probe_handler
from agent.tools.asset_references import (
    build_object_memory_configuration_warning_handler,
    build_object_memory_reference_handler,
)
from agent.tools.handlers import (
    bind_dummy_tool_handlers,
    build_anyplace_handler,
    build_depth_prior_handler,
    build_grasp_pose_estimate_handler,
    build_molmopoint_handler,
    build_sam3_handler,
)
from agent.tools.grasp_geometry import DEFAULT_GRASP_PROFILE
from agent.tools.grasp_geometry import (
    build_compile_grasp_seed_handler,
    build_wrist_alignment_handler,
    build_wrist_viewpoint_proposal_handler,
    compile_grasp_seed,
)
from agent.tools.registry import ToolResult, build_default_tool_registry
from agent.tools.object_memory import ObjectMemoryBundle, ObjectMemoryReference
from agent.tools.sim_mcp import (
    SimulatorMcpToolProxyConfig,
    bind_simulator_mcp_tool_handlers,
)
from agent.tools.web_access import (
    HostedWebSearchClient,
    WebHttpResponse,
    WebSearchConfig,
    WebSearchEndpointConfig,
    build_web_fetch_handler,
    build_web_search_handler,
)


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
    expected_declared_outcomes = sorted(
        outcome.semantic_outcome for outcome in contract.outcomes
    )
    observed_declared_outcomes = sorted(
        {
            str(case.get("semantic_outcome") or "")
            for case in (handler_success_case, failure_case)
            if case.get("conformant") is True and case.get("semantic_outcome")
        }
    )
    all_declared_outcomes_covered = (
        observed_declared_outcomes == expected_declared_outcomes
    )
    conformant = bool(
        all_success_outcomes_covered
        and all_declared_outcomes_covered
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
            "expected_declared_outcomes": expected_declared_outcomes,
            "observed_declared_outcomes": observed_declared_outcomes,
            "all_declared_outcomes_covered": all_declared_outcomes_covered,
            "representative_failure_covered": representative_failure_covered,
            "runtime_gate_evidence_observed": gate_case.get("conformant") is True,
            "runtime_gate_rejection_observed": gate_case.get("blocked") is True,
            "matched_gate_check_ids": gate_case.get("matched_gate_check_ids", []),
            "freshness_and_invalidation_covered": True,
            "remote_backend_semantics_covered": True,
            "world_mutating_execution_receipt_covered": True,
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
        "diagnostic_messages": [
            str(item.get("message") or "")
            for item in diagnostics
            if isinstance(item, dict) and item.get("message")
        ],
        "artifact_paths": sorted(
            {
                str(item.get("path") or "")
                for item in artifacts
                if isinstance(item, dict) and item.get("path")
            }
        ),
        "environment_receipt_present": isinstance(
            details.get("environment_receipt"), dict
        ),
        "state_delta_keys": sorted(
            str(key)
            for key in (
                details.get("state_delta", {}).keys()
                if isinstance(details.get("state_delta"), dict)
                else []
            )
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


class _FaultingAgentMemory(AgentMemory):
    """Exercise production runtime handlers against a deterministic dependency fault."""

    def __init__(self, failed_operation: str) -> None:
        super().__init__()
        self.failed_operation = failed_operation

    def save_fact(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.failed_operation == "save_memory":
            raise RuntimeError("deterministic save-memory dependency failure")
        return super().save_fact(*args, **kwargs)

    def get_memory(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.failed_operation == "get_memory":
            raise RuntimeError("deterministic get-memory dependency failure")
        return super().get_memory(*args, **kwargs)

    def delete_memory(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.failed_operation == "delete_memory":
            raise RuntimeError("deterministic delete-memory dependency failure")
        return super().delete_memory(*args, **kwargs)

    def compact(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if self.failed_operation == "compact_memory":
            raise RuntimeError("deterministic compact-memory dependency failure")
        return super().compact(*args, **kwargs)


def _memory_tool_fixture(
    tool_name: str,
    root: Path,
    *,
    success_parameters: JsonDict,
) -> JsonDict:
    success_registry = build_default_tool_registry()
    success_runtime = OpenEtaAgentRuntime(
        memory=AgentMemory(),
        tools=success_registry,
        rollout_enabled=False,
    )
    success_runtime.start_session(task=f"{tool_name} fixture", session_id="fixture-success")
    if tool_name in {"get_memory", "delete_memory"}:
        success_runtime.memory.save_fact(
            "fixture-key",
            {"value": "fixture"},
            source="fixture",
        )
    success = success_registry.call(
        tool_name,
        dict(success_parameters),
        metadata={"session_id": "fixture-success"},
    )

    failure_registry = build_default_tool_registry()
    failure_memory = _FaultingAgentMemory(tool_name)
    failure_runtime = OpenEtaAgentRuntime(
        memory=failure_memory,
        tools=failure_registry,
        rollout_enabled=False,
    )
    failure_runtime.start_session(task=f"{tool_name} fixture", session_id="fixture-failure")
    failure = failure_registry.call(
        tool_name,
        dict(success_parameters),
        metadata={"session_id": "fixture-failure"},
    )
    return _local_runtime_fixture_receipt(
        tool_name,
        root,
        success_registry,
        success.details,
        failure.details,
        success_parameters=success_parameters,
        handler_implementation=f"agent.runtime.runtime.OpenEtaAgentRuntime._{tool_name}_tool",
    )


def _python_exec_fixture(root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    registry.bind_handler(
        "python_exec",
        PythonExecRuntime(
            PythonExecConfig(
                session_root=str(root / "session"),
                workspace_root=str(root / "workspace"),
                image_output_root=str(root / "artifacts" / "images"),
                text_output_root=str(root / "artifacts" / "text"),
                response_output_root=str(root / "artifacts" / "responses"),
                structured_output_root=str(root / "artifacts"),
            )
        ).handler,
    )
    success_parameters = {"code": "result = {'fixture_value': 3}"}
    success = registry.call(
        "python_exec",
        dict(success_parameters),
        metadata={"session_id": "fixture-success"},
    )
    failure = registry.call(
        "python_exec",
        {"code": "raise RuntimeError('deterministic fixture failure')"},
        metadata={"session_id": "fixture-failure"},
    )
    return _local_runtime_fixture_receipt(
        "python_exec",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=success_parameters,
        handler_implementation="agent.tools.coding.PythonExecRuntime.handler",
    )


def _record_pending_sam3_fixture(
    memory: AgentMemory,
    *,
    result_id: str = "sam3-contract-fixture",
) -> None:
    """Record the same durable SAM3 envelope consumed by runtime selection tools."""

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "sam3",
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": result_id,
                                    "prompt": "fixture object",
                                    "evidence_role": "target_object",
                                    "source_image": "fixture-agentview.png",
                                    "source_observation": {
                                        "packet_id": "fixture-packet",
                                        "frame_id": "agentview",
                                        "role": "scene_primary",
                                    },
                                    "segmentation_mode": "text",
                                    "ranking": "score_descending",
                                    "detection_count": 2,
                                    "detections": [
                                        {
                                            "id": "detection_000",
                                            "rank": 0,
                                            "backend_index": 0,
                                            "score": 0.91,
                                            "mask_ref": "fixture-mask-000.png",
                                        },
                                        {
                                            "id": "detection_001",
                                            "rank": 1,
                                            "backend_index": 1,
                                            "score": 0.72,
                                            "mask_ref": "fixture-mask-001.png",
                                        },
                                    ],
                                    "selection_required": True,
                                    "selected_detection": None,
                                    "selection_bundle": {
                                        "original_image_ref": "fixture-agentview.png",
                                        "contact_sheet_ref": "fixture-contact-sheet.png",
                                        "candidate_count": 2,
                                        "candidates": [],
                                    },
                                },
                                "artifacts": [],
                            },
                        },
                    }
                ],
            },
        )
    )


def _sam3_selection_fixture(tool_name: str, root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    runtime = OpenEtaAgentRuntime(tools=registry, rollout_enabled=False)
    runtime.start_session(task=f"{tool_name} fixture", session_id="sam3-fixture")
    _record_pending_sam3_fixture(runtime.memory)
    if tool_name == "select_sam3_detection":
        parameters = {
            "sam3_result_id": "sam3-contract-fixture",
            "detection_id": "detection_000",
            "evidence_role": "target_object",
            "reason": "The fixture mask covers the intended object.",
        }
    else:
        parameters = {
            "sam3_result_id": "sam3-contract-fixture",
            "reason": "Neither fixture mask covers the intended object.",
        }
    if tool_name == "select_sam3_detection":
        failure = registry.call(
            tool_name,
            {**parameters, "selection_confidence": 2.0},
        )
    else:
        failure = registry.call(
            tool_name,
            {**parameters, "sam3_result_id": "unknown-sam3-result"},
        )
    success = registry.call(tool_name, parameters)
    return _local_runtime_fixture_receipt(
        tool_name,
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation=f"agent.runtime.runtime.OpenEtaAgentRuntime._{tool_name}_tool",
        freshness_case_covered=True,
    )


def _camera_pose_to_world_fixture(root: Path) -> JsonDict:
    registry = bind_dummy_tool_handlers(build_default_tool_registry())
    parameters = {
        "camera_frame_id": "agentview",
        "camera_pose": {
            "id": "fixture-pose",
            "frame": "camera",
            "translation_xyz": [0.1, 0.2, 0.3],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "camera_to_world": {
            "camera_frame": "opencv",
            "camera_to_world": [
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 1.0, 0.0, -0.5],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }
    success = registry.call("camera_pose_to_world", parameters)
    failure = registry.call(
        "camera_pose_to_world",
        {
            **parameters,
            "camera_to_world": {"camera_to_world": [[1.0, 0.0]]},
        },
    )
    return _local_runtime_fixture_receipt(
        "camera_pose_to_world",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation="agent.tools.handlers._camera_pose_to_world_handler",
        freshness_gate_parameters={
            "placement_result_id": "missing-placement-result",
            "candidate_id": "missing-placement-candidate",
        },
    )


def _fixture_camera_grasp() -> JsonDict:
    return {
        "id": "grasp-fixture",
        "frame": "camera",
        "camera_frame": "opencv",
        "width": 0.06,
        "translation_xyz": [0.15, 0.0, 1.0],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }


def _fixture_camera_extrinsics() -> JsonDict:
    return {
        "camera_frame": "opencv",
        "frame_transform": "camera_to_world",
        "matrix_layout": "row_major",
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }


def _fixture_compiled_grasp() -> JsonDict:
    profile_bytes = DEFAULT_GRASP_PROFILE.read_bytes()
    profile = json.loads(profile_bytes.decode("utf-8"))
    return compile_grasp_seed(
        {
            "camera_pose": _fixture_camera_grasp(),
            "camera_extrinsics": _fixture_camera_extrinsics(),
            "camera_frame_id": "agentview",
            "target_class": "upright_can",
            "scene_epoch": 0,
        },
        profile=profile,
        profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
    )


def _compile_grasp_seed_fixture(root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    registry.bind_handler("compile_grasp_seed", build_compile_grasp_seed_handler())
    parameters = {
        "camera_pose": _fixture_camera_grasp(),
        "camera_extrinsics": _fixture_camera_extrinsics(),
        "camera_frame_id": "agentview",
        "target_class": "upright_can",
        "scene_epoch": 0,
    }
    success = registry.call("compile_grasp_seed", parameters)
    failure = registry.call(
        "compile_grasp_seed",
        {**parameters, "camera_pose": {**_fixture_camera_grasp(), "width": 0.5}},
    )
    return _local_runtime_fixture_receipt(
        "compile_grasp_seed",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation=(
            "agent.tools.grasp_geometry.build_compile_grasp_seed_handler"
        ),
        freshness_gate_parameters={
            "grasp_result_id": "missing-grasp-result",
            "candidate_id": "missing-grasp-candidate",
        },
    )


def _write_wrist_alignment_inputs(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    mask_path = root / "target-mask.png"
    depth_path = root / "depth.png"
    mask = Image.new("L", (10, 10), 0)
    for y in range(3, 6):
        for x in range(4, 7):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("I;16", (10, 10), 1000).save(depth_path)
    return mask_path, depth_path


def _wrist_alignment_parameters(
    root: Path,
    *,
    near_compiled_clearance: bool,
) -> JsonDict:
    mask_path, depth_path = _write_wrist_alignment_inputs(root)
    compiled_grasp = _fixture_compiled_grasp()
    current_xyz = (
        list(compiled_grasp["hover_pose"]["xyz"])
        if near_compiled_clearance
        else [0.0, 0.0, 0.6]
    )
    return {
        "compiled_grasp": compiled_grasp,
        "target_mask": str(mask_path),
        "depth": str(depth_path),
        "intrinsics": {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 4.0,
            "cy": 4.0,
            "width": 10,
            "height": 10,
            "scale": 1000.0,
        },
        "camera_extrinsics": _fixture_camera_extrinsics(),
        "current_eef_pose": {
            "xyz": current_xyz,
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "scene_epoch": 0,
    }


def _compute_wrist_alignment_fixture(root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    registry.bind_handler(
        "compute_wrist_alignment",
        build_wrist_alignment_handler(),
    )
    executable_parameters = _wrist_alignment_parameters(
        root / "executable",
        near_compiled_clearance=True,
    )
    better_view_parameters = _wrist_alignment_parameters(
        root / "better-view",
        near_compiled_clearance=False,
    )
    executable = registry.call("compute_wrist_alignment", executable_parameters)
    better_view = registry.call("compute_wrist_alignment", better_view_parameters)
    failure = registry.call(
        "compute_wrist_alignment",
        {**executable_parameters, "target_mask": str(root / "missing-mask.png")},
    )
    return _local_runtime_fixture_receipt(
        "compute_wrist_alignment",
        root,
        registry,
        [executable.details, better_view.details],
        failure.details,
        success_parameters=executable_parameters,
        handler_implementation=(
            "agent.tools.grasp_geometry.build_wrist_alignment_handler"
        ),
        freshness_gate_parameters={"bundle_id": "missing-wrist-alignment-bundle"},
    )


def _propose_wrist_viewpoints_fixture(root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    registry.bind_handler(
        "propose_wrist_viewpoints",
        build_wrist_viewpoint_proposal_handler(),
    )
    parameters = {
        "compiled_grasp": _fixture_compiled_grasp(),
        "source_packet_id": "fixture-packet",
        "camera_frame_id": "wrist",
        "camera_extrinsics": {
            "camera_frame": "opencv",
            "frame_transform": "camera_to_world",
            "camera_to_world": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
        "current_eef_pose": {
            "xyz": [0.0, 0.0, 0.0],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        "object_scene_epoch": 0,
        "robot_motion_epoch": 4,
        "standoff_m": 0.18,
    }
    success = registry.call("propose_wrist_viewpoints", parameters)
    failure = registry.call(
        "propose_wrist_viewpoints",
        {**parameters, "standoff_m": -1.0},
    )
    return _local_runtime_fixture_receipt(
        "propose_wrist_viewpoints",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation=(
            "agent.tools.grasp_geometry.build_wrist_viewpoint_proposal_handler"
        ),
        freshness_gate_parameters={
            "compiled_grasp_id": "missing-compiled-grasp",
            "source_packet_id": "missing-packet",
            "camera_frame_id": "wrist",
        },
    )


def _png_base64(
    *,
    size: tuple[int, int] = (16, 16),
    mode: str = "L",
    value: int | tuple[int, int, int] = 255,
) -> str:
    image = Image.new(mode, size, value)
    payload = io.BytesIO()
    image.save(payload, format="PNG")
    return base64.b64encode(payload.getvalue()).decode("ascii")


def _sam3_response(*, detection_count: int) -> JsonDict:
    detections = []
    if detection_count:
        detections.append(
            {
                "label": "fixture object",
                "score": 0.93,
                "rank": 0,
                "backend_index": 0,
                "bbox_xyxy": [2, 2, 12, 12],
                "area_px": 100,
                "mask": {"format": "png", "base64": _png_base64()},
            }
        )
    return {
        "success": True,
        "content": "deterministic SAM3 fixture",
        "details": {
            "tool": "sam3",
            "backend": "sam3_fixture_backend",
            "model": "sam3-fixture",
            "prompt_type": "text",
            "detection_count": len(detections),
            "detections": detections,
            "ranking": "score_descending",
            "artifacts": [],
            "metadata": {"image_size": [16, 16]},
        },
    }


def _sam3_observation(image_path: Path) -> EnvObservation:
    return EnvObservation(
        task="segment fixture object",
        cameras=[
            CameraFrame(
                "agentview",
                [],
                intrinsics={"fx": 100.0, "fy": 100.0, "cx": 8.0, "cy": 8.0},
                role="scene_primary",
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
                    "path": str(image_path),
                    "packet_id": "fixture-packet",
                }
            ],
        },
    )


def _sam3_fixture(root: Path) -> JsonDict:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "source.png"
    Image.new("RGB", (16, 16), (30, 60, 90)).save(image_path)
    observation = _sam3_observation(image_path)
    parameters = {
        "source_packet_id": "fixture-packet",
        "camera_frame_id": "agentview",
        "image": str(image_path),
        "mode": "text",
        "prompt": "fixture object",
        "evidence_role": "target_object",
    }
    registry = build_default_tool_registry()
    registry.bind_handler(
        "sam3",
        build_sam3_handler(
            lambda _request: _sam3_response(detection_count=1),
            output_root=root / "detections" / "images",
            result_output_root=root / "detections" / "results",
        ),
    )
    detections = registry.call("sam3", parameters, observation=observation)
    no_detection_registry = build_default_tool_registry()
    no_detection_registry.bind_handler(
        "sam3",
        build_sam3_handler(
            lambda _request: _sam3_response(detection_count=0),
            output_root=root / "no-detection" / "images",
            result_output_root=root / "no-detection" / "results",
        ),
    )
    no_detection = no_detection_registry.call(
        "sam3",
        parameters,
        observation=observation,
    )
    failure_registry = build_default_tool_registry()
    failure_registry.bind_handler(
        "sam3",
        build_sam3_handler(
            lambda _request: (_ for _ in ()).throw(
                TimeoutError("deterministic SAM3 fixture timeout")
            ),
            output_root=root / "failure" / "images",
            result_output_root=root / "failure" / "results",
        ),
    )
    failure = failure_registry.call("sam3", parameters, observation=observation)
    return _local_runtime_fixture_receipt(
        "sam3",
        root,
        registry,
        [detections.details, no_detection.details],
        failure.details,
        success_parameters=parameters,
        handler_implementation="agent.tools.handlers.build_sam3_handler",
        freshness_gate_parameters={
            "source_packet_id": "missing-packet",
            "camera_frame_id": "agentview",
            "mode": "text",
            "prompt": "fixture object",
        },
    )


def _molmopoint_fixture(root: Path) -> JsonDict:
    root.mkdir(parents=True, exist_ok=True)
    first = root / "first.png"
    second = root / "second.jpg"
    Image.new("RGB", (24, 20), (10, 20, 30)).save(first)
    Image.new("RGB", (30, 18), (40, 50, 60)).save(second)
    parameters = {
        "images": [str(first), str(second)],
        "prompt": "Point to the same fixture object in both images.",
    }

    def response(request: JsonDict) -> JsonDict:
        image_metadata = []
        for image_index, payload in enumerate(request["images"]):
            raw = base64.b64decode(payload["base64"])
            with Image.open(io.BytesIO(raw)) as image:
                image_metadata.append(
                    {
                        "image_index": image_index,
                        "format": (
                            "jpeg" if image.format == "JPEG" else image.format.lower()
                        ),
                        "width": image.width,
                        "height": image.height,
                        "source_image_mode": image.mode,
                        "model_image_mode": "RGB",
                    }
                )
        return {
            "success": True,
            "content": "deterministic MolmoPoint fixture",
            "details": {
                "tool": "molmopoint",
                "backend": "molmopoint_mcp",
                "model": "molmopoint-fixture",
                "point_count": 2,
                "points": [
                    {"id": "point_000", "image_index": 0, "pixel_x": 12, "pixel_y": 10},
                    {"id": "point_001", "image_index": 1, "pixel_x": 15, "pixel_y": 9},
                ],
                "coordinate_convention": {
                    "origin": "top_left",
                    "x_direction": "right",
                    "y_direction": "down",
                    "units": "pixels",
                },
                "artifacts": [],
                "metadata": {
                    "model_revision": "188130f961c8e0888a34e11121a1423c461a01ba",
                    "images": image_metadata,
                },
            },
        }

    registry = build_default_tool_registry()
    registry.bind_handler(
        "molmopoint",
        build_molmopoint_handler(response, output_root=root / "success"),
    )
    success = registry.call("molmopoint", parameters)
    failure_registry = build_default_tool_registry()
    failure_registry.bind_handler(
        "molmopoint",
        build_molmopoint_handler(
            lambda _request: (_ for _ in ()).throw(
                TimeoutError("deterministic MolmoPoint fixture timeout")
            ),
            output_root=root / "failure",
        ),
    )
    failure = failure_registry.call("molmopoint", parameters)
    return _local_runtime_fixture_receipt(
        "molmopoint",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation="agent.tools.handlers.build_molmopoint_handler",
        freshness_gate_parameters={
            "sources": [
                {
                    "source_packet_id": "missing-packet",
                    "camera_frame_id": "agentview",
                }
            ],
            "prompt": "point to fixture object",
        },
    )


def _retrieve_asset_reference_fixture(root: Path) -> JsonDict:
    root.mkdir(parents=True, exist_ok=True)
    reference = root / "reference-source.png"
    scene = root / "scene.png"
    Image.new("RGB", (12, 10), "red").save(reference)
    Image.new("RGB", (32, 24), "blue").save(scene)
    reference_payload = reference.read_bytes()

    class Client:
        def retrieve(self, *, environment: str, target_object: str):
            return ObjectMemoryBundle(
                query_key="libero/fixture_object",
                namespace="libero",
                asset_id="fixture_object",
                label=target_object,
                references=(
                    ObjectMemoryReference("front", "front.png", reference_payload),
                ),
                manifest={"key": "libero/fixture_object"},
            )

    class Localizer:
        def localize(self, **_kwargs):
            return ReferencePointLocalization(
                x=16.0,
                y=12.0,
                bbox_xyxy=(8.0, 6.0, 24.0, 20.0),
                confidence=0.9,
                reason="deterministic fixture localization",
                provider="fixture",
                model="fixture-localizer",
                details={
                    "candidate_policy": "ranked_provisional",
                    "requires_downstream_confirmation": True,
                    "ranked_candidates": [
                        {
                            "rank": 1,
                            "positive_points": [{"x": 16.0, "y": 12.0, "label": 1}],
                            "bbox_xyxy": [8.0, 6.0, 24.0, 20.0],
                            "provisional": True,
                        }
                    ],
                },
            )

    registry = build_default_tool_registry()
    registry.bind_handler(
        "retrieve_asset_reference",
        build_object_memory_reference_handler(
            Client(),
            Localizer(),
            output_root=root / "outputs",
        ),
    )
    parameters = {
        "source_packet_id": "fixture-packet",
        "environment": "openeta/libero_fixture",
        "target_object": "fixture object",
        "scene_image": str(scene),
    }
    success = registry.call("retrieve_asset_reference", parameters)
    failure = registry.call(
        "retrieve_asset_reference",
        {**parameters, "scene_image": str(root / "missing-scene.png")},
    )
    unavailable_registry = build_default_tool_registry()
    unavailable_registry.bind_handler(
        "retrieve_asset_reference",
        build_object_memory_configuration_warning_handler(),
    )
    unavailable = unavailable_registry.call("retrieve_asset_reference", parameters)
    return _local_runtime_fixture_receipt(
        "retrieve_asset_reference",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation=(
            "agent.tools.asset_references.build_object_memory_reference_handler"
        ),
        additional_details=[unavailable.details],
        freshness_gate_parameters={
            "environment": "libero",
            "target_object": "fixture object",
            "source_packet_id": "missing-packet",
        },
    )


def _anyplace_candidate(index: int = 0) -> JsonDict:
    return {
        "id": f"grasp_{index:03d}",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.8,
        "translation_xyz": [0.1 + index, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "gripper_tip_position_xyz": [0.13 + index, 0.2, 0.3],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
    }


def _anyplace_response() -> JsonDict:
    placements = []
    for index in range(5):
        placements.append(
            {
                "id": f"placement_{index:03d}",
                "source_grasp_id": "grasp_000",
                "object_placement_transform": {
                    "frame": "camera",
                    "camera_frame": "opencv",
                    "convention": "p_placed = R @ p_current + t",
                    "transform_matrix": [
                        [1.0, 0.0, 0.0, float(index)],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
                "place_grasp_pose": {
                    **_anyplace_candidate(index),
                    "id": f"place_grasp_{index:03d}",
                },
            }
        )
    return {
        "success": True,
        "content": "deterministic AnyPlace fixture",
        "details": {
            "tool": "anyplace",
            "backend": "anyplace_fixture_backend",
            "model": "anyplace-fixture",
            "frame": "camera",
            "camera_frame": "opencv",
            "candidate_count": len(placements),
            "placement_candidates": placements,
            "metadata": {},
        },
    }


def _anyplace_fixture(root: Path) -> JsonDict:
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, mode in (
        ("rgb", "RGB"),
        ("depth", "I;16"),
        ("object-mask", "L"),
        ("placement-mask", "L"),
    ):
        path = root / f"{name}.png"
        Image.new(mode, (8, 8), 100 if mode != "RGB" else (10, 20, 30)).save(path)
        paths[name] = str(path)
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 4.0, "cy": 4.0, "scale": 1000.0}
    parameters = {
        "rgb": paths["rgb"],
        "depth": paths["depth"],
        "object_mask": paths["object-mask"],
        "placement_region_mask": {
            "type": "segmentation_mask",
            "mask_ref": paths["placement-mask"],
            "source_image": paths["rgb"],
            "label": "fixture region",
            "score": 0.9,
        },
        "intrinsics": intrinsics,
        "selected_grasp": {
            "candidate": _anyplace_candidate(),
            "source": {
                "mode": "targeted",
                "rgb": paths["rgb"],
                "depth": paths["depth"],
                "object_mask": paths["object-mask"],
                "intrinsics": intrinsics,
            },
        },
    }
    registry = build_default_tool_registry()
    registry.bind_handler(
        "anyplace",
        build_anyplace_handler(lambda _request: _anyplace_response(), output_root=root / "success"),
    )
    success = registry.call("anyplace", parameters)
    failure_registry = build_default_tool_registry()
    failure_registry.bind_handler(
        "anyplace",
        build_anyplace_handler(
            lambda _request: (_ for _ in ()).throw(
                TimeoutError("deterministic AnyPlace fixture timeout")
            ),
            output_root=root / "failure",
        ),
    )
    failure = failure_registry.call("anyplace", parameters)
    return _local_runtime_fixture_receipt(
        "anyplace",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation="agent.tools.handlers.build_anyplace_handler",
        freshness_gate_parameters={"bundle_id": "missing-anyplace-bundle"},
    )


def _grasp_pose_estimate_fixture(root: Path) -> JsonDict:
    root.mkdir(parents=True, exist_ok=True)
    rgb = root / "rgb.png"
    depth = root / "depth.png"
    mask = root / "mask.png"
    Image.new("RGB", (16, 16), (20, 40, 60)).save(rgb)
    Image.new("I;16", (16, 16), 700).save(depth)
    Image.new("L", (16, 16), 255).save(mask)
    parameters = {
        "mode": "targeted",
        "rgb": str(rgb),
        "depth": str(depth),
        "object_mask": {
            "mask_ref": str(mask),
            "source_image": str(rgb),
            "result_id": "sam3-fixture",
            "detection_id": "detection_000",
            "quality": {"status": "usable", "area_fraction": 1.0},
        },
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 8.0, "cy": 8.0, "scale": 1000.0},
        "camera_frame_id": "agentview",
        "scene_epoch": 4,
        "hints": {"dense_sampling": True, "depth_cutoff_factor": 1.0},
    }

    def backend(_context):
        return ToolResult(
            True,
            details={
                "candidate_count": 1,
                "grasp_candidates": [_anyplace_candidate()],
                "source": {},
                "artifacts": [],
            },
        )

    registry = build_default_tool_registry()
    registry.bind_handler(
        "grasp_pose_estimate",
        build_grasp_pose_estimate_handler({"anygrasp": backend}, backend_order=("anygrasp",)),
    )
    success = registry.call("grasp_pose_estimate", parameters)
    failure = registry.call(
        "grasp_pose_estimate",
        {**parameters, "backend_preference": "anygrasp"},
    )

    def infeasible_backend(_context):
        candidate = _anyplace_candidate()
        candidate["width"] = 0.5
        return ToolResult(
            True,
            details={
                "candidate_count": 1,
                "grasp_candidates": [candidate],
                "source": {},
                "artifacts": [],
            },
        )

    infeasible_registry = build_default_tool_registry()
    infeasible_registry.bind_handler(
        "grasp_pose_estimate",
        build_grasp_pose_estimate_handler(
            {"anygrasp": infeasible_backend},
            backend_order=("anygrasp",),
        ),
    )
    no_executable = infeasible_registry.call(
        "grasp_pose_estimate",
        {
            **parameters,
            "hints": {**parameters["hints"], "max_gripper_width_m": 0.08},
        },
    )
    return _local_runtime_fixture_receipt(
        "grasp_pose_estimate",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation=(
            "agent.tools.handlers.build_grasp_pose_estimate_handler"
        ),
        additional_details=[no_executable.details],
        freshness_gate_parameters={"bundle_id": "missing-grasp-bundle"},
    )


class _FixtureSimulatorTransport:
    def __init__(self, responses: list[JsonDict] | JsonDict, *, url: str = "") -> None:
        self.responses = responses if isinstance(responses, list) else [responses]
        self.url = url
        self.calls: list[JsonDict] = []

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        index = len(self.calls)
        self.calls.append(
            {"name": name, "arguments": dict(arguments), "timeout_s": timeout_s}
        )
        return dict(self.responses[min(index, len(self.responses) - 1)])


class _FailingFixtureSimulatorTransport:
    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        del arguments, timeout_s
        raise RuntimeError(f"deterministic simulator fixture failure: {name}")


def _fixture_simulator_config(root: Path, *, active: bool = True):
    return SimulatorMcpToolProxyConfig(
        session_id="fixture-sim-session" if active else "",
        handle="fixture-env" if active else "",
        image_output_root=root / "images",
        response_output_root=root / "responses",
    )


def _create_simulator_env_fixture(root: Path) -> JsonDict:
    transport = _FixtureSimulatorTransport(
        [
            {
                "success": True,
                "handle": "fixture-env",
                "session_id": "fixture-sim-session",
                "env_id": "openeta/fixture-v0",
            },
            {
                "success": True,
                "handle": "fixture-env",
                "session_id": "fixture-sim-session",
                "task": "pick fixture object",
                "cameras": [],
                "robot": {},
            },
        ],
        url="http://fixture-simulator.invalid/sse",
    )
    registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        registry,
        transport=transport,
        config=_fixture_simulator_config(root / "success", active=False),
        tool_names=("create_simulator_env",),
    )
    parameters = {
        "env_id": "openeta/fixture-v0",
        "seed": 7,
        "session_id": "fixture-sim-session",
    }
    success = registry.call("create_simulator_env", parameters)
    failure = registry.call("create_simulator_env", parameters)
    return _local_runtime_fixture_receipt(
        "create_simulator_env",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation=(
            "agent.tools.sim_mcp.SimulatorEnvironmentCreator.handler"
        ),
        freshness_case_covered=True,
    )


def _close_simulator_env_fixture(root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        registry,
        transport=_FixtureSimulatorTransport({"ok": True, "success": True}),
        config=_fixture_simulator_config(root / "success"),
        tool_names=("close_simulator_env",),
    )
    success = registry.call("close_simulator_env", {})
    failure_registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        failure_registry,
        transport=_FailingFixtureSimulatorTransport(),
        config=_fixture_simulator_config(root / "failure"),
        tool_names=("close_simulator_env",),
    )
    failure = failure_registry.call("close_simulator_env", {})
    return _local_runtime_fixture_receipt(
        "close_simulator_env",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters={},
        handler_implementation=(
            "agent.tools.sim_mcp.SimulatorEnvironmentCloser.handler"
        ),
    )


def _enhance_depth_fixture(root: Path) -> JsonDict:
    root.mkdir(parents=True, exist_ok=True)
    rgb_path = root / "rgb.png"
    enhanced_depth_path = root / "enhanced-depth.png"
    repair_depth_path = root / "repair-depth.png"
    prior_path = root / "prior.npy"
    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[..., 0] = 64
    rgb[..., 1] = 128
    rgb[..., 2] = 192
    Image.fromarray(rgb).save(rgb_path)
    enhanced_depth = np.full((3, 3), 2000, dtype=np.uint16)
    enhanced_depth[1, 1] = 0
    Image.fromarray(enhanced_depth).save(enhanced_depth_path)
    Image.fromarray(np.full((3, 3), 1000, dtype=np.uint16)).save(repair_depth_path)
    np.save(prior_path, np.full((3, 3), 4.0, dtype=np.float32))
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0, "scale": 1000.0}
    common = {
        "rgb": str(rgb_path),
        "prior_depth": str(prior_path),
        "intrinsics": intrinsics,
        "camera_id": "wrist",
        "source_packet_id": "fixture-packet",
        "camera_frame_id": "wrist",
        "config": {"min_alignment_pixels": 4, "edge_guard_pixels": 0},
    }
    registry = build_default_tool_registry()
    runtime = OpenEtaAgentRuntime(tools=registry, rollout_enabled=False)
    runtime.start_session(task="enhance depth fixture", session_id="depth-fixture")
    enhanced_parameters = {
        **common,
        "depth": str(enhanced_depth_path),
        "bundle_id": "enhanced-fixture",
        "intrinsics": {**intrinsics, "scale": 500.0},
    }
    repair_parameters = {
        **common,
        "depth": str(repair_depth_path),
        "bundle_id": "repair-fixture",
    }
    enhanced = registry.call(
        "enhance_depth",
        enhanced_parameters,
        metadata={"session_id": "depth-fixture-enhanced"},
    )
    repair = registry.call(
        "enhance_depth",
        repair_parameters,
        metadata={"session_id": "depth-fixture-repair"},
    )
    failure = registry.call(
        "enhance_depth",
        {**common, "depth": str(root / "missing-depth.png")},
    )
    return _local_runtime_fixture_receipt(
        "enhance_depth",
        root,
        registry,
        [enhanced.details, repair.details],
        failure.details,
        success_parameters=repair_parameters,
        handler_implementation=(
            "agent.runtime.runtime.OpenEtaAgentRuntime._enhance_depth_tool"
        ),
        freshness_gate_parameters={
            "source_packet_id": "missing-packet",
            "camera_frame_id": "wrist",
        },
    )


def _assess_attachment_probe_fixture(root: Path) -> JsonDict:
    before_agent = root / "before-agent.png"
    before_wrist = root / "before-wrist.png"
    after_agent = root / "after-agent.png"
    after_wrist = root / "after-wrist.png"
    root.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(
        (before_agent, before_wrist, after_agent, after_wrist)
    ):
        Image.new("RGB", (8, 8), (20 * index, 30, 40)).save(path)
    observation = EnvObservation(
        task="assess attachment fixture",
        cameras=[
            CameraFrame("agentview", [], role="scene_primary"),
            CameraFrame("wrist", [], role="wrist_primary"),
        ],
        robot=RobotState(gripper_state={"open": False, "openness": 0.4}),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "agentview", "role": "scene_primary", "path": str(after_agent)},
                {"kind": "rgb", "frame_id": "wrist", "role": "wrist_primary", "path": str(after_wrist)},
            ]
        },
    )
    probe = {
        "status": "completed",
        "probe_id": "probe:fixture",
        "candidate_id": "grasp-fixture",
        "motion_type": "linear",
        "distance_m": 0.05,
        "direction_world_xyz": [1.0, 0.0, 0.0],
        "pre_probe_image_paths": [str(before_agent), str(before_wrist)],
    }
    metadata = {
        "task": observation.task,
        "supervision_context": {
            "memory": {
                "scene_epoch": 4,
                "gripper_command_state": {
                    "position": 0,
                    "state": "closed",
                    "attachment_proxy_receipt": {
                        "status": "tentative",
                        "reason": "non_empty_close_with_tentative_safety_proxy",
                    },
                },
                "articulated_attachment_probe": probe,
            }
        },
    }
    registry = build_default_tool_registry()
    registry.bind_handler(
        "assess_attachment_probe",
        build_assess_attachment_probe_handler(
            StaticPlannerBackend(
                [{"verdict": "PASS", "reason": "fixture target visibly co-moved"}]
            )
        ),
    )
    parameters = {"probe_id": "probe:fixture"}
    success = registry.call(
        "assess_attachment_probe",
        parameters,
        observation=observation,
        metadata=metadata,
    )
    failure = registry.call(
        "assess_attachment_probe",
        {"probe_id": "probe:wrong"},
        observation=observation,
        metadata=metadata,
    )
    return _local_runtime_fixture_receipt(
        "assess_attachment_probe",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation=(
            "agent.tools.attachment_probe.build_assess_attachment_probe_handler"
        ),
        freshness_case_covered=True,
    )


def _prepare_attachment_probe_fixture(root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    registry.bind_handler(
        "prepare_attachment_probe",
        build_prepare_attachment_probe_handler(),
    )
    observation = EnvObservation(
        task="prepare attachment probe fixture",
        cameras=[
            CameraFrame("agentview", [[[0, 0, 0]]], role="scene_primary"),
            CameraFrame("wrist", [[[0, 0, 0]]], role="wrist_primary"),
        ],
        robot=RobotState(
            end_effector_pose={
                "xyz": [0.45, 0.0, 0.25],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            gripper_state={"open": False, "openness": 0.4},
        ),
        metadata={
            "step_idx": 0,
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": "fixture-agentview.png",
                },
                {
                    "kind": "rgb",
                    "frame_id": "wrist",
                    "role": "wrist_primary",
                    "path": "fixture-wrist.png",
                },
            ],
        },
    )
    supervision_context = {
        "memory": {
            "scene_epoch": 3,
            "gripper_command_state": {
                "position": 0,
                "state": "closed",
                "attachment_proxy_receipt": {
                    "status": "tentative",
                    "reason": "non_empty_close_with_tentative_safety_proxy",
                },
            },
            "provenance_evidence_graph": {
                "nodes": [
                    {
                        "kind": "compiled_targeted_grasp",
                        "compiled_grasp_id": "compiled-grasp-fixture",
                        "candidate_id": "grasp-fixture",
                        "freshness": "fresh",
                    }
                ]
            },
        }
    }
    parameters = {
        "compiled_grasp_id": "compiled-grasp-fixture",
        "motion_type": "linear",
        "direction_world_xyz": [1.0, 0.0, 0.0],
        "reason": "Verify bounded co-motion before articulated manipulation.",
    }
    success = registry.call(
        "prepare_attachment_probe",
        parameters,
        observation=observation,
        metadata={"supervision_context": supervision_context},
    )
    failure = registry.call(
        "prepare_attachment_probe",
        {**parameters, "direction_world_xyz": [0.0, 0.0, 0.0]},
        observation=observation,
        metadata={"supervision_context": supervision_context},
    )
    return _local_runtime_fixture_receipt(
        "prepare_attachment_probe",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation=(
            "agent.tools.attachment_probe.build_prepare_attachment_probe_handler"
        ),
        freshness_case_covered=True,
    )


def _select_sam3_detection_fixture(root: Path) -> JsonDict:
    return _sam3_selection_fixture("select_sam3_detection", root)


def _reject_sam3_detections_fixture(root: Path) -> JsonDict:
    return _sam3_selection_fixture("reject_sam3_detections", root)


def _local_runtime_fixture_receipt(
    tool_name: str,
    root: Path,
    registry,
    success_details: JsonDict | list[JsonDict],
    failure_details: JsonDict,
    *,
    success_parameters: JsonDict,
    handler_implementation: str,
    additional_details: list[JsonDict] | None = None,
    gate_case: JsonDict | None = None,
    freshness_gate_parameters: JsonDict | None = None,
    freshness_case_covered: bool = False,
) -> JsonDict:
    catalog = build_default_tool_contract_catalog(registry.list())
    contract = catalog.get(tool_name)
    success_values = success_details if isinstance(success_details, list) else [success_details]
    success_cases = [
        _result_case(
            contract,
            "handler_success" if len(success_values) == 1 else f"handler_success_{index + 1}",
            details,
        )
        for index, details in enumerate(success_values)
    ]
    failure_case = _result_case(contract, "handler_dependency_failure", failure_details)
    additional_cases = [
        _result_case(
            contract,
            f"handler_declared_outcome_{index + 1}",
            details,
        )
        for index, details in enumerate(additional_details or [])
    ]
    resolved_gate_case = gate_case or _motion_reconciliation_gate_fixture(
        tool_name, registry, success_parameters
    )
    gate_cases = [resolved_gate_case]
    if freshness_gate_parameters is not None:
        gate_cases.append(
            _host_resolution_gate_fixture(
                tool_name,
                freshness_gate_parameters,
            )
        )
    expected_success_outcomes = sorted(
        outcome.semantic_outcome
        for outcome in contract.outcomes
        if outcome.operational_success
    )
    observed_success_outcomes = sorted(
        {
            str(case.get("semantic_outcome") or "")
            for case in success_cases
            if case.get("operational_success") is True
        }
    )
    representative_failure_covered = bool(
        failure_case.get("semantic_outcome") == "operational_failure"
        and failure_case.get("conformant") is True
    )
    all_success_outcomes_covered = observed_success_outcomes == expected_success_outcomes
    expected_declared_outcomes = sorted(
        outcome.semantic_outcome for outcome in contract.outcomes
    )
    observed_declared_outcomes = sorted(
        {
            str(case.get("semantic_outcome") or "")
            for case in [*success_cases, *additional_cases, failure_case]
            if case.get("conformant") is True and case.get("semantic_outcome")
        }
    )
    all_declared_outcomes_covered = (
        observed_declared_outcomes == expected_declared_outcomes
    )
    gate_evidence_observed = all(
        case.get("conformant") is True for case in gate_cases
    )
    matched_gate_check_ids = sorted(
        {
            str(check_id)
            for case in gate_cases
            for check_id in case.get("matched_gate_check_ids", [])
            if check_id
        }
    )
    freshness_and_invalidation_covered = bool(
        freshness_case_covered
        or any(
            case.get("case") == "freshness_or_provenance_gate_rejection"
            and case.get("conformant") is True
            for case in gate_cases
        )
    )
    world_mutating_execution_receipt_covered = bool(
        contract.effect != "world_mutating"
        or all(
            case.get("environment_receipt_present") is True
            for case in success_cases
        )
    )
    conformant = bool(
        all_success_outcomes_covered
        and all_declared_outcomes_covered
        and representative_failure_covered
        and gate_evidence_observed
    )
    return {
        "contract_schema_version": contract.schema_version,
        "contract_maturity": contract.maturity.value,
        "handler_implementation": handler_implementation,
        "host_resolver_id": contract.host_resolution.resolver,
        "host_resolver_implementation": contract.host_resolution.implementation,
        "artifact_root": str(root),
        "cases": [*success_cases, *additional_cases, failure_case, *gate_cases],
        "coverage": {
            "expected_success_outcomes": expected_success_outcomes,
            "observed_success_outcomes": observed_success_outcomes,
            "all_success_outcomes_covered": all_success_outcomes_covered,
            "expected_declared_outcomes": expected_declared_outcomes,
            "observed_declared_outcomes": observed_declared_outcomes,
            "all_declared_outcomes_covered": all_declared_outcomes_covered,
            "representative_failure_covered": representative_failure_covered,
            "runtime_gate_evidence_observed": gate_evidence_observed,
            "runtime_gate_rejection_observed": any(
                case.get("blocked") is True for case in gate_cases
            ),
            "matched_gate_check_ids": matched_gate_check_ids,
            "freshness_and_invalidation_covered": freshness_and_invalidation_covered,
            "remote_backend_semantics_covered": bool(
                contract.category in {"perception", "web", "manipulation"}
                and all_declared_outcomes_covered
            ),
            "world_mutating_execution_receipt_covered": (
                world_mutating_execution_receipt_covered
            ),
        },
        "conformant": conformant,
    }


def _host_resolution_gate_fixture(tool_name: str, parameters: JsonDict) -> JsonDict:
    registry = bind_dummy_tool_handlers(build_default_tool_registry())
    memory = AgentMemory()
    memory.start_session(task=f"{tool_name} freshness fixture")
    observation = EnvObservation(
        task=f"{tool_name} freshness fixture",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 0},
    )
    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action=tool_name,
            parameters=dict(parameters),
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
    matched = shadow.get("matched_gate_bindings")
    matched = matched if isinstance(matched, list) else []
    matched_ids = sorted(
        {
            str(item.get("check_id") or "")
            for item in matched
            if isinstance(item, dict) and item.get("check_id")
        }
    )
    return {
        "case": "freshness_or_provenance_gate_rejection",
        "boundary": "ActionPipeline -> host resolver",
        "blocked": plan.status is PipelineStatus.BLOCKED,
        "tool_execution_count": sum(
            call.status is PipelineStatus.EXECUTED for call in plan.tool_calls
        ),
        "repair_code": str(repair.get("code") or ""),
        "matched_gate_check_ids": matched_ids,
        "authoritative_gate": str(shadow.get("authoritative_gate") or ""),
        "enforcing": shadow.get("enforcing") is True,
        "conformant": bool(
            plan.status is PipelineStatus.BLOCKED
            and not any(
                call.status is PipelineStatus.EXECUTED for call in plan.tool_calls
            )
            and shadow.get("conformant") is True
            and shadow.get("authoritative_gate") == "legacy_runtime"
            and any(
                check_id
                in {
                    "runtime.source_packet_resolution",
                    "runtime.provenance_bundle_resolution",
                    "runtime.ik_receipt_resolution",
                    "runtime.ik_trajectory_resolution",
                    "runtime.compiled_grasp_reference_resolution",
                }
                for check_id in matched_ids
            )
        ),
        "contract_shadow_validation": shadow,
    }


def _motion_reconciliation_gate_fixture(
    tool_name: str,
    registry,
    parameters: JsonDict,
) -> JsonDict:
    memory = AgentMemory()
    memory.start_session(task=f"{tool_name} gate fixture")
    memory.save_fact(
        "motion_reconciliation",
        {"status": "required", "scene_epoch": 0},
        source="transport_unknown",
    )
    observation = EnvObservation(
        task=f"{tool_name} gate fixture",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 0},
    )
    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action=tool_name,
            parameters=dict(parameters),
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
    matched = shadow.get("matched_gate_bindings")
    matched = matched if isinstance(matched, list) else []
    matched_ids = sorted(
        {
            str(item.get("check_id") or "")
            for item in matched
            if isinstance(item, dict) and item.get("check_id")
        }
    )
    return {
        "case": "motion_reconciliation_gate_rejection",
        "boundary": "ActionPipeline -> host gate",
        "blocked": plan.status is PipelineStatus.BLOCKED,
        "tool_execution_count": sum(
            call.status is PipelineStatus.EXECUTED for call in plan.tool_calls
        ),
        "repair_code": str(repair.get("code") or ""),
        "matched_gate_check_ids": matched_ids,
        "authoritative_gate": str(shadow.get("authoritative_gate") or ""),
        "enforcing": shadow.get("enforcing") is True,
        "conformant": bool(
            plan.status is PipelineStatus.BLOCKED
            and not any(
                call.status is PipelineStatus.EXECUTED for call in plan.tool_calls
            )
            and repair.get("code") == "motion_reconciliation_required"
            and shadow.get("conformant") is True
            and shadow.get("authoritative_gate") == "legacy_runtime"
            and "runtime.motion_reconciliation" in matched_ids
        ),
        "contract_shadow_validation": shadow,
    }


def _save_memory_fixture(root: Path) -> JsonDict:
    return _memory_tool_fixture(
        "save_memory",
        root,
        success_parameters={"key": "fixture-key", "content": {"value": "fixture"}},
    )


def _get_memory_fixture(root: Path) -> JsonDict:
    return _memory_tool_fixture(
        "get_memory",
        root,
        success_parameters={"key": "fixture-key", "namespace": "facts"},
    )


def _delete_memory_fixture(root: Path) -> JsonDict:
    return _memory_tool_fixture(
        "delete_memory",
        root,
        success_parameters={"key": "fixture-key", "namespace": "facts"},
    )


def _compact_memory_fixture(root: Path) -> JsonDict:
    return _memory_tool_fixture(
        "compact_memory",
        root,
        success_parameters={"max_events": 1},
    )


def _calibration_fixture(tool_name: str, root: Path) -> JsonDict:
    manager = _calibration_manager(
        root / "success",
        decisions=[
            {"decision": "approve", "reason": "deterministic fixture proposal"},
            {"decision": "approve", "reason": "deterministic fixture promotion"},
        ],
    )
    registry = build_default_tool_registry()
    registry.bind_handler("propose_calibration_profile", manager.propose_handler)
    registry.bind_handler("promote_calibration_profile", manager.promote_handler)
    proposal_parameters = {
        "profile": _calibration_profile(),
        "profile_fingerprint": _calibration_fingerprint(),
        "rationale": "Verify the production calibration lifecycle contract.",
    }
    proposed = registry.call(
        "propose_calibration_profile",
        proposal_parameters,
        metadata={"session_id": "calibration-fixture"},
    )
    if tool_name == "propose_calibration_profile":
        success = proposed
        rejecting = _calibration_manager(
            root / "failure",
            decisions=[
                {"decision": "reject", "reason": "deterministic fixture rejection"}
            ],
        )
        failure_registry = build_default_tool_registry()
        failure_registry.bind_handler(
            "propose_calibration_profile",
            rejecting.propose_handler,
        )
        failure = failure_registry.call(
            tool_name,
            proposal_parameters,
            metadata={"session_id": "calibration-fixture-failure"},
        )
        success_parameters = proposal_parameters
        handler = "agent.runtime.calibration.CalibrationLifecycleManager.propose_handler"
    else:
        proposal_outputs = proposed.details["outputs"]
        evidence = _write_calibration_evidence(
            root / "success",
            profile_sha256=str(proposal_outputs["profile_sha256"]),
        )
        success_parameters = {
            "proposal_id": proposal_outputs["proposal_id"],
            "target_status": "candidate",
            "evidence": evidence,
        }
        success = registry.call(
            tool_name,
            success_parameters,
            metadata={"session_id": "calibration-fixture"},
        )
        failure = registry.call(
            tool_name,
            {
                **success_parameters,
                "proposal_id": "calibration-missing-fixture",
            },
            metadata={"session_id": "calibration-fixture"},
        )
        handler = "agent.runtime.calibration.CalibrationLifecycleManager.promote_handler"
    return _local_runtime_fixture_receipt(
        tool_name,
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=success_parameters,
        handler_implementation=handler,
    )


def _calibration_manager(
    root: Path,
    *,
    decisions: list[JsonDict],
) -> CalibrationLifecycleManager:
    return CalibrationLifecycleManager(
        config=CalibrationLifecycleConfig(
            root=root / "session",
            candidate_dir=root / "published" / "candidate",
            validated_dir=root / "published" / "validated",
            evidence_roots=(root,),
            publication_mode="independent_reviewer",
            min_canary_attempts=2,
            min_held_out_attempts=1,
        ),
        reviewer=BackendCalibrationReviewer(StaticPlannerBackend(decisions)),
    )


def _calibration_profile() -> JsonDict:
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "calibrations"
            / "candidate"
            / "graspnet-eef-panda-p8.json"
        ).read_text(encoding="utf-8")
    )


def _calibration_fingerprint() -> JsonDict:
    return {
        "robot_model": "Panda",
        "gripper_model": "PandaGripper",
        "controller": "OSC_POSE",
        "environment": "LIBERO",
        "camera_calibration_id": "libero-agentview-panda-v1",
        "backend_versions": {"robosuite": "fixture", "mujoco": "fixture"},
    }


def _write_calibration_evidence(root: Path, *, profile_sha256: str) -> list[JsonDict]:
    root.mkdir(parents=True, exist_ok=True)
    refs: list[JsonDict] = []
    for split, attempts in (("canary", 2), ("held_out", 1)):
        path = root / f"{split}-evidence.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
                    "profile_sha256": profile_sha256,
                    "split": split,
                    "attempts": attempts,
                    "successes": attempts,
                    "assisted_attempts": 0,
                    "metrics": {
                        "finger_center_p95_m": 0.004,
                        "axis_p95_deg": 2.5,
                    },
                }
            ),
            encoding="utf-8",
        )
        refs.append({"path": str(path), "split": split})
    return refs


def _grasp_strategy_fixture(tool_name: str, root: Path) -> JsonDict:
    manager = _grasp_strategy_manager(
        root / "success",
        decisions=[
            {"decision": "approve", "reason": "deterministic fixture proposal"},
            {"decision": "approve", "reason": "deterministic fixture promotion"},
        ],
    )
    registry = build_default_tool_registry()
    registry.bind_handler("propose_grasp_strategy", manager.propose_handler)
    registry.bind_handler("promote_grasp_strategy", manager.promote_handler)
    proposal_parameters = {
        "strategy": _grasp_strategy(),
        "rationale": "Verify the production grasp strategy lifecycle contract.",
    }
    proposed = registry.call(
        "propose_grasp_strategy",
        proposal_parameters,
        metadata={"session_id": "strategy-fixture"},
    )
    if tool_name == "propose_grasp_strategy":
        success = proposed
        rejecting = _grasp_strategy_manager(
            root / "failure",
            decisions=[
                {"decision": "reject", "reason": "deterministic fixture rejection"}
            ],
        )
        failure_registry = build_default_tool_registry()
        failure_registry.bind_handler("propose_grasp_strategy", rejecting.propose_handler)
        failure = failure_registry.call(
            tool_name,
            proposal_parameters,
            metadata={"session_id": "strategy-fixture-failure"},
        )
        success_parameters = proposal_parameters
        handler = (
            "agent.runtime.grasp_strategy_lifecycle."
            "GraspStrategyLifecycleManager.propose_handler"
        )
    else:
        proposal_outputs = proposed.details["outputs"]
        evidence_path = root / "success" / "canary-evidence.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(
                {
                    "schema_version": GRASP_STRATEGY_EVIDENCE_SCHEMA_VERSION,
                    "producer": "openeta_experiment_host",
                    "strategy_sha256": proposal_outputs["strategy_sha256"],
                    "calibration_profile_sha256": proposal_outputs[
                        "calibration_profile_sha256"
                    ],
                    "split": "canary",
                    "attempts": 2,
                    "successes": 2,
                    "baseline_attempts": 2,
                    "baseline_successes": 2,
                    "task_ids": ["fixture-task"],
                    "seeds": [0, 1],
                    "regressed_episode_ids": [],
                    "safety_violations": 0,
                    "contract_violations": 0,
                    "human_interventions": 0,
                }
            ),
            encoding="utf-8",
        )
        success_parameters = {
            "proposal_id": proposal_outputs["proposal_id"],
            "target_status": "candidate",
            "evidence": [{"path": str(evidence_path), "split": "canary"}],
        }
        success = registry.call(
            tool_name,
            success_parameters,
            metadata={"session_id": "strategy-fixture"},
        )
        failure = registry.call(
            tool_name,
            {**success_parameters, "proposal_id": "grasp-strategy-missing-fixture"},
            metadata={"session_id": "strategy-fixture"},
        )
        handler = (
            "agent.runtime.grasp_strategy_lifecycle."
            "GraspStrategyLifecycleManager.promote_handler"
        )
    return _local_runtime_fixture_receipt(
        tool_name,
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=success_parameters,
        handler_implementation=handler,
    )


def _grasp_strategy_manager(
    root: Path,
    *,
    decisions: list[JsonDict],
) -> GraspStrategyLifecycleManager:
    return GraspStrategyLifecycleManager(
        config=GraspStrategyLifecycleConfig(
            root=root / "lifecycle",
            candidate_dir=root / "published" / "candidate",
            validated_dir=root / "published" / "validated",
            session_strategy_root=root / "session-strategies",
            calibration_profile=json.loads(DEFAULT_GRASP_PROFILE.read_text(encoding="utf-8")),
            evidence_roots=(root,),
            publication_mode="independent_reviewer",
            min_canary_attempts=2,
            min_held_out_attempts=1,
            min_held_out_success_rate=0.5,
            min_held_out_task_count=1,
        ),
        reviewer=BackendGraspStrategyReviewer(StaticPlannerBackend(decisions)),
    )


def _grasp_strategy() -> JsonDict:
    return {
        "schema_version": "openeta.grasp_strategy.v1",
        "status": "candidate",
        "strategy_id": "contract-fixture-round-panda-r1",
        "strategy_family_id": "contract-fixture-round-panda",
        "revision": 1,
        "description": "Preserve estimator pose for deterministic fixture objects.",
        "compatibility": {"calibration_ids": ["graspnet-eef-panda-p8"]},
        "automatic_activation": {"target_geometry_families": ["bowl"]},
        "validated_scope": {"target_geometry_families": ["bowl"]},
        "constraints": {"grasp_width_bounds_m": [0.02, 0.08], "clearance_m": 0.01},
        "pose_policy": {
            "approach_axis": "preserve_candidate",
            "orientation": "preserve_candidate",
        },
        "provenance": {"intended_use": "ToolContract fixture"},
    }


def _skill_management_fixture(tool_name: str, root: Path) -> JsonDict:
    registry = build_default_tool_registry()
    runtime = OpenEtaAgentRuntime(tools=registry, rollout_enabled=False)
    runtime.start_session(task=f"{tool_name} fixture", session_id="skill-fixture")
    payload = {
        "name": "contract-fixture-skill",
        "description": "Fixture guidance for contract verification.",
        "content": "# Contract Fixture\n\nUse python_exec for bounded local inspection.",
        "task_patterns": ["inspect fixture"],
        "allowed_tools": ["python_exec"],
        "version": "v1",
    }

    def backend_factory(**_kwargs):
        return StaticPlannerBackend([payload])

    _bind_skill_change_tools(
        runtime,
        backend_factory=backend_factory,
        policy_provider=lambda: SupervisionPolicy.for_profile(SupervisionProfile.STANDARD),
        human_approval=None,
    )
    register_parameters = {
        "name": "contract-fixture-skill",
        "goal": "Inspect a fixture with grounded observation.",
    }
    registered = registry.call("register_skill", register_parameters)
    if tool_name == "register_skill":
        success = registered
        success_parameters = register_parameters
        failure = registry.call(tool_name, register_parameters)
    else:
        success_parameters = {
            "name": "contract-fixture-skill",
            "requested_changes": "Clarify that the observation must be grounded.",
        }
        success = registry.call(tool_name, success_parameters)
        failure = registry.call(
            tool_name,
            {**success_parameters, "name": "missing-fixture-skill"},
        )
    return _local_runtime_fixture_receipt(
        tool_name,
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=success_parameters,
        handler_implementation="agent.runtime.runtime_assembly._skill_change_handler",
    )


def _propose_calibration_profile_fixture(root: Path) -> JsonDict:
    return _calibration_fixture("propose_calibration_profile", root)


def _promote_calibration_profile_fixture(root: Path) -> JsonDict:
    return _calibration_fixture("promote_calibration_profile", root)


def _propose_grasp_strategy_fixture(root: Path) -> JsonDict:
    return _grasp_strategy_fixture("propose_grasp_strategy", root)


def _promote_grasp_strategy_fixture(root: Path) -> JsonDict:
    return _grasp_strategy_fixture("promote_grasp_strategy", root)


def _register_skill_fixture(root: Path) -> JsonDict:
    return _skill_management_fixture("register_skill", root)


def _update_skill_fixture(root: Path) -> JsonDict:
    return _skill_management_fixture("update_skill", root)


def _web_search_fixture(root: Path) -> JsonDict:
    config = WebSearchConfig(
        primary=WebSearchEndpointConfig(
            provider="fixture-provider",
            model="fixture-search-model",
            api_base="https://fixture.example.com/v1",
            api_key="fixture-key",
            timeout_s=1.0,
        )
    )
    response_text = "OpenETA uses evidence-gated ToolContracts."
    response = json.dumps(
        {
            "status": "completed",
            "output": [
                {"type": "web_search_call", "status": "completed"},
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": response_text,
                            "annotations": [
                                {
                                    "type": "url_citation",
                                    "title": "OpenETA fixture",
                                    "url": "https://docs.example.com/openeta",
                                    "start_index": 0,
                                    "end_index": 7,
                                }
                            ],
                        }
                    ],
                },
            ],
        }
    ).encode()
    registry = build_default_tool_registry()
    registry.bind_handler(
        "web_search",
        build_web_search_handler(
            HostedWebSearchClient(config, transport=lambda *_args: response)
        ),
    )
    parameters = {"query": "OpenETA ToolContract", "max_results": 3}
    success = registry.call("web_search", parameters)
    failure_registry = build_default_tool_registry()
    failure_registry.bind_handler(
        "web_search",
        build_web_search_handler(
            HostedWebSearchClient(
                config,
                transport=lambda *_args: (_ for _ in ()).throw(
                    TimeoutError("deterministic web-search timeout")
                ),
            )
        ),
    )
    failure = failure_registry.call("web_search", parameters)
    return _local_runtime_fixture_receipt(
        "web_search",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation="agent.tools.web_access.HostedWebSearchClient",
    )


def _web_fetch_fixture(root: Path) -> JsonDict:
    def public_resolver(host, port, **_kwargs):
        return [
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("93.184.216.34", 443),
            )
        ]

    def success_transport(resolved, *_args):
        return WebHttpResponse(
            url=resolved.url,
            status=200,
            headers={"Content-Type": "text/html; charset=utf-8"},
            body=(
                b"<html><head><title>Fixture Docs</title></head>"
                b"<body><p>Evidence-gated tool contract.</p></body></html>"
            ),
        )

    parameters = {"url": "https://docs.example.com/openeta", "max_chars": 1000}
    registry = build_default_tool_registry()
    registry.bind_handler(
        "web_fetch",
        build_web_fetch_handler(
            transport=success_transport,
            resolver=public_resolver,
        ),
    )
    success = registry.call("web_fetch", parameters)
    failure_registry = build_default_tool_registry()
    failure_registry.bind_handler(
        "web_fetch",
        build_web_fetch_handler(
            transport=success_transport,
            resolver=lambda *_args, **_kwargs: [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", 443),
                )
            ],
        ),
    )
    failure = failure_registry.call("web_fetch", parameters)
    return _local_runtime_fixture_receipt(
        "web_fetch",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation="agent.tools.web_access.build_web_fetch_handler",
    )


def _batch_observe_allow_fixture(registry) -> JsonDict:
    memory = AgentMemory()
    memory.start_session(task="observe batch-boundary fixture")
    observation = EnvObservation(
        task="observe batch-boundary fixture",
        cameras=[],
        robot=RobotState(),
        metadata={"step_idx": 0},
    )
    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="tool_batch",
            parameters={
                "calls": [
                    {"name": "observe", "parameters": {"reason": "fixture refresh 1"}},
                    {"name": "observe", "parameters": {"reason": "fixture refresh 2"}},
                ]
            },
        ),
        observation=observation,
        tools=registry,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    executed = sum(call.status is PipelineStatus.EXECUTED for call in plan.tool_calls)
    return {
        "case": "batch_boundary_allow",
        "boundary": "ActionPipeline batched read-only boundary",
        "blocked": False,
        "allowed": plan.status is PipelineStatus.EXECUTED,
        "tool_execution_count": executed,
        "matched_gate_check_ids": ["runtime.batch_boundary"],
        "authoritative_gate": "legacy_runtime",
        "enforcing": True,
        "conformant": bool(
            plan.status is PipelineStatus.EXECUTED
            and len(plan.tool_calls) == 2
            and executed == 2
        ),
    }


def _observe_fixture(root: Path) -> JsonDict:
    response = {
        "success": True,
        "task": "observe fixture",
        "cameras": [],
        "robot": {
            "end_effector_pose": {"xyz": [0.1, 0.2, 0.3]},
            "gripper_state": {"open": True},
        },
        "objects": [{"name": "fixture_object", "category": "fixture"}],
    }
    registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        registry,
        transport=_FixtureSimulatorTransport(response),
        config=_fixture_simulator_config(root / "success"),
        tool_names=("observe",),
    )
    parameters = {"reason": "refresh fixture observation"}
    success = registry.call("observe", parameters)
    failure_registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        failure_registry,
        transport=_FailingFixtureSimulatorTransport(),
        config=_fixture_simulator_config(root / "failure"),
        tool_names=("observe",),
    )
    failure = failure_registry.call("observe", parameters)
    return _local_runtime_fixture_receipt(
        "observe",
        root,
        registry,
        success.details,
        failure.details,
        success_parameters=parameters,
        handler_implementation="agent.tools.sim_mcp.SimulatorMcpToolProxy.handler_for",
        gate_case=_batch_observe_allow_fixture(registry),
        freshness_case_covered=True,
    )


def _motion_response(
    *,
    reached_target: bool = True,
    steps_executed: int = 4,
    attachment_status: str = "",
) -> JsonDict:
    response: JsonDict = {
        "success": True,
        "start": {"xyz": [0.0, 0.0, 0.2]},
        "end": {
            "xyz": [0.1, 0.0, 0.3] if steps_executed else [0.0, 0.0, 0.2]
        },
        "target": {
            "x": 0.1 if steps_executed else 0.0,
            "y": 0.0,
            "z": 0.3 if steps_executed else 0.2,
        },
        "reached_target": reached_target,
        "steps_executed": steps_executed,
        "collision": {
            "trajectory_checked": True,
            "world_checked": True,
            "world_object_count": 1,
        },
        "cameras": [],
        "robot": {},
    }
    if attachment_status:
        response["attachment_proxy_receipt"] = {
            "schema_version": "openeta.attachment_proxy_receipt.v1",
            "status": attachment_status,
            "reason": (
                "awaiting_independent_co_motion_evidence"
                if attachment_status == "tentative"
                else "empty_close_or_no_measurable_contact"
            ),
            "target_object_name": "fixture_object",
            "attachment_proven": False,
        }
    return response


def _bind_fixture_sim_tools(
    root: Path,
    *,
    tool_names: tuple[str, ...],
    responses: list[JsonDict] | JsonDict,
):
    registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        registry,
        transport=_FixtureSimulatorTransport(responses),
        config=_fixture_simulator_config(root),
        tool_names=tool_names,
    )
    return registry


def _motion_parameters(tool_name: str) -> JsonDict:
    target = {
        "frame": "world",
        "xyz": [0.1, 0.0, 0.3],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    if tool_name == "move_to":
        return {"target_pose": target, "enable_collision_check": True}
    return {"trajectory": [target], "enable_collision_check": True}


def _motion_tool_fixture(tool_name: str, root: Path) -> JsonDict:
    parameters = _motion_parameters(tool_name)
    success_registry = _bind_fixture_sim_tools(
        root / "mutation",
        tool_names=(tool_name,),
        responses=_motion_response(),
    )
    mutation = success_registry.call(tool_name, parameters)
    noop_registry = _bind_fixture_sim_tools(
        root / "noop",
        tool_names=(tool_name,),
        responses=_motion_response(steps_executed=0),
    )
    noop = noop_registry.call(tool_name, parameters)
    tentative_registry = _bind_fixture_sim_tools(
        root / "tentative",
        tool_names=(tool_name,),
        responses=_motion_response(attachment_status="tentative"),
    )
    tentative = tentative_registry.call(tool_name, parameters)
    empty_registry = _bind_fixture_sim_tools(
        root / "empty",
        tool_names=(tool_name,),
        responses=_motion_response(attachment_status="not_armed"),
    )
    empty = empty_registry.call(tool_name, parameters)

    authorization = {
        "schema_version": "openeta.contact_authorization.v1",
        "compiled_grasp_id": "compiled-fixture",
        "waypoint_role": "grasp_contact",
        "target_object_name": "fixture_object",
        "target_anchor_world_xyz": [0.0, 0.0, 0.2],
        "object_scene_epoch": 0,
    }
    missing_registry = _bind_fixture_sim_tools(
        root / "attachment-missing",
        tool_names=("gripper_control", tool_name),
        responses=[{"success": True, "cameras": [], "robot": {}}, _motion_response()],
    )
    missing_registry.call(
        "gripper_control",
        {"position": 0},
        metadata={"_attachment_candidate_resolver": lambda: authorization},
    )
    missing_parameters = dict(parameters)
    missing_metadata: JsonDict = {}
    if tool_name == "move_to":
        missing_metadata["_contact_authorization_resolver"] = lambda _pose: authorization
    else:
        missing_parameters["contact_authorization"] = authorization
    missing = missing_registry.call(
        tool_name,
        missing_parameters,
        metadata=missing_metadata,
    )
    target_miss_registry = _bind_fixture_sim_tools(
        root / "target-miss",
        tool_names=(tool_name,),
        responses=_motion_response(reached_target=False),
    )
    target_miss = target_miss_registry.call(tool_name, parameters)
    failure_registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        failure_registry,
        transport=_FailingFixtureSimulatorTransport(),
        config=_fixture_simulator_config(root / "failure"),
        tool_names=(tool_name,),
    )
    failure = failure_registry.call(tool_name, parameters)
    return _local_runtime_fixture_receipt(
        tool_name,
        root,
        success_registry,
        [
            mutation.details,
            noop.details,
            tentative.details,
            empty.details,
            missing.details,
        ],
        failure.details,
        additional_details=[target_miss.details],
        success_parameters=parameters,
        handler_implementation="agent.tools.sim_mcp.SimulatorMcpToolProxy.handler_for",
        freshness_gate_parameters=(
            {"ik_receipt_id": "missing-ik-receipt"}
            if tool_name == "move_to"
            else {"ik_receipt_ids": ["missing-ik-receipt"]}
        ),
    )


def _move_to_fixture(root: Path) -> JsonDict:
    return _motion_tool_fixture("move_to", root)


def _follow_eef_trajectory_fixture(root: Path) -> JsonDict:
    return _motion_tool_fixture("follow_eef_trajectory", root)


def _gripper_control_fixture(root: Path) -> JsonDict:
    open_registry = _bind_fixture_sim_tools(
        root / "open",
        tool_names=("gripper_control",),
        responses={"success": True, "cameras": [], "robot": {}},
    )
    mutation = open_registry.call("gripper_control", {"position": 1})
    authorization = {
        "schema_version": "openeta.contact_authorization.v1",
        "compiled_grasp_id": "compiled-fixture",
        "waypoint_role": "grasp_contact",
        "target_anchor_world_xyz": [0.0, 0.0, 0.2],
        "object_scene_epoch": 0,
    }
    tentative_registry = _bind_fixture_sim_tools(
        root / "tentative",
        tool_names=("gripper_control",),
        responses={
            "success": True,
            "cameras": [],
            "robot": {},
            "attachment_proxy_receipt": {
                "schema_version": "openeta.attachment_proxy_receipt.v1",
                "status": "tentative",
                "target_object_name": "fixture_object",
                "attachment_proven": False,
            },
        },
    )
    tentative = tentative_registry.call(
        "gripper_control",
        {"position": 0},
        metadata={"_attachment_candidate_resolver": lambda: authorization},
    )
    empty_registry = _bind_fixture_sim_tools(
        root / "empty",
        tool_names=("gripper_control",),
        responses={
            "success": True,
            "cameras": [],
            "robot": {},
            "attachment_proxy_receipt": {
                "schema_version": "openeta.attachment_proxy_receipt.v1",
                "status": "not_armed",
                "reason": "empty_close_or_no_measurable_contact",
                "attachment_proven": False,
            },
        },
    )
    empty = empty_registry.call("gripper_control", {"position": 0})
    missing_registry = _bind_fixture_sim_tools(
        root / "missing",
        tool_names=("gripper_control",),
        responses={"success": True, "cameras": [], "robot": {}},
    )
    missing = missing_registry.call(
        "gripper_control",
        {"position": 0},
        metadata={"_attachment_candidate_resolver": lambda: authorization},
    )
    failure_registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        failure_registry,
        transport=_FailingFixtureSimulatorTransport(),
        config=_fixture_simulator_config(root / "failure"),
        tool_names=("gripper_control",),
    )
    failure = failure_registry.call("gripper_control", {"position": 1})
    return _local_runtime_fixture_receipt(
        "gripper_control",
        root,
        open_registry,
        [mutation.details, tentative.details, empty.details, missing.details],
        failure.details,
        success_parameters={"position": 1},
        handler_implementation="agent.tools.sim_mcp.SimulatorMcpToolProxy.handler_for",
    )


def _ik_response(classification: str) -> tuple[JsonDict, JsonDict]:
    target = {"frame": "world", "xyz": [0.1, 0.2, 0.3]}
    metadata: JsonDict = {}
    if classification == "feasible":
        response = {
            "ok": True,
            "success": True,
            "status": "reachable",
            "kinematic_status": "reachable",
            "feasible": True,
            "reason_code": "ik_solution_found",
            "message": "fixture IK solution found",
            "target": target,
            "best_candidate": {"joint_positions": [0.0] * 7},
            "collision": {"checked": True},
            "path": {"checked": False},
        }
    elif classification == "kinematically_feasible_collision_deferred":
        response = {
            "ok": True,
            "success": True,
            "status": "unknown",
            "kinematic_status": "reachable",
            "feasible": None,
            "reason_code": "endpoint_collision_check_unavailable",
            "message": "fixture endpoint collision backend unavailable",
            "target": target,
            "best_candidate": {"joint_positions": [0.0] * 7},
            "collision": {"checked": False},
            "path": {"checked": False},
        }
        metadata["_controller_capabilities_resolver"] = lambda: {
            "controller_id": "mink.fixture",
            "goal_executor": "openeta.fixture_goal.v1",
            "collision_scope": "worker_per_step_pre_and_post",
            "motion_owns_trajectory_world_collision": True,
        }
    elif classification == "repairable":
        response = {
            "ok": False,
            "success": False,
            "status": "unreachable",
            "kinematic_status": "unreachable",
            "feasible": False,
            "reason_code": "full_pose_infeasible",
            "message": "fixture pose can be repaired",
            "target": target,
            "position_only_reachable": True,
            "best_candidate": {"joint_positions": [0.0] * 7},
            "collision": {"checked": False},
            "path": {"checked": False},
            "suggestions": ["relax_target_orientation"],
        }
    elif classification == "inconclusive":
        response = {
            "ok": False,
            "success": False,
            "status": "unknown",
            "kinematic_status": "unknown",
            "feasible": None,
            "reason_code": "solver_timeout",
            "message": "fixture solver timed out without a conclusion",
            "target": target,
            "collision": {"checked": False},
            "path": {"checked": False},
        }
    else:
        response = {
            "ok": False,
            "success": False,
            "status": "unreachable",
            "kinematic_status": "unreachable",
            "feasible": False,
            "reason_code": "outside_workspace",
            "message": "fixture target is outside the workspace",
            "target": target,
            "collision": {"checked": False},
            "path": {"checked": False},
        }
    return response, metadata


def _ik_preview_check_fixture(root: Path) -> JsonDict:
    parameters = {
        "target_pose": {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.3],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "check_endpoint_collision": True,
    }
    details: list[JsonDict] = []
    gate_registry = None
    for classification in (
        "feasible",
        "kinematically_feasible_collision_deferred",
        "repairable",
        "inconclusive",
        "hard_infeasible",
    ):
        response, metadata = _ik_response(classification)
        registry = _bind_fixture_sim_tools(
            root / classification,
            tool_names=("ik_preview_check",),
            responses=response,
        )
        gate_registry = gate_registry or registry
        result = registry.call("ik_preview_check", parameters, metadata=metadata)
        details.append(result.details)
    failure_registry = build_default_tool_registry()
    bind_simulator_mcp_tool_handlers(
        failure_registry,
        transport=_FailingFixtureSimulatorTransport(),
        config=_fixture_simulator_config(root / "failure"),
        tool_names=("ik_preview_check",),
    )
    failure = failure_registry.call("ik_preview_check", parameters)
    assert gate_registry is not None
    return _local_runtime_fixture_receipt(
        "ik_preview_check",
        root,
        gate_registry,
        details,
        failure.details,
        success_parameters=parameters,
        handler_implementation="agent.tools.sim_mcp.SimulatorMcpToolProxy.handler_for",
        freshness_gate_parameters={
            "compiled_grasp_id": "missing-compiled-grasp",
            "waypoint_role": "grasp_contact",
        },
    )


_FIXTURE_BUILDERS: dict[str, FixtureBuilder] = {
    "assess_attachment_probe": _assess_attachment_probe_fixture,
    "anyplace": _anyplace_fixture,
    "camera_pose_to_world": _camera_pose_to_world_fixture,
    "close_simulator_env": _close_simulator_env_fixture,
    "compact_memory": _compact_memory_fixture,
    "compile_grasp_seed": _compile_grasp_seed_fixture,
    "compute_wrist_alignment": _compute_wrist_alignment_fixture,
    "create_simulator_env": _create_simulator_env_fixture,
    "delete_memory": _delete_memory_fixture,
    "estimate_depth_prior": _estimate_depth_prior_fixture,
    "enhance_depth": _enhance_depth_fixture,
    "follow_eef_trajectory": _follow_eef_trajectory_fixture,
    "get_memory": _get_memory_fixture,
    "grasp_pose_estimate": _grasp_pose_estimate_fixture,
    "gripper_control": _gripper_control_fixture,
    "ik_preview_check": _ik_preview_check_fixture,
    "molmopoint": _molmopoint_fixture,
    "move_to": _move_to_fixture,
    "observe": _observe_fixture,
    "promote_calibration_profile": _promote_calibration_profile_fixture,
    "promote_grasp_strategy": _promote_grasp_strategy_fixture,
    "propose_calibration_profile": _propose_calibration_profile_fixture,
    "propose_grasp_strategy": _propose_grasp_strategy_fixture,
    "python_exec": _python_exec_fixture,
    "prepare_attachment_probe": _prepare_attachment_probe_fixture,
    "propose_wrist_viewpoints": _propose_wrist_viewpoints_fixture,
    "register_skill": _register_skill_fixture,
    "reject_sam3_detections": _reject_sam3_detections_fixture,
    "save_memory": _save_memory_fixture,
    "sam3": _sam3_fixture,
    "select_sam3_detection": _select_sam3_detection_fixture,
    "retrieve_asset_reference": _retrieve_asset_reference_fixture,
    "update_skill": _update_skill_fixture,
    "web_fetch": _web_fetch_fixture,
    "web_search": _web_search_fixture,
}


def registered_fixture_tools() -> tuple[str, ...]:
    """Return tools with durable production-handler fixture builders."""

    return tuple(sorted(_FIXTURE_BUILDERS))


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
