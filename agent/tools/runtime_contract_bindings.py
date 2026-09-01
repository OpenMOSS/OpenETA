"""Stable runtime identities for host-side ToolContract input resolvers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable

from adapter.protocol import JsonDict
from agent.tools.contracts import HostResolutionContract, ToolContractCatalog


HOST_RESOLVER_BINDING_SCHEMA_VERSION = "openeta.host_resolver_binding.v1"


@dataclass(frozen=True, slots=True)
class HostResolutionResult:
    parameters: JsonDict
    evidence: JsonDict = field(default_factory=dict)


HostResolver = Callable[[JsonDict, Any], HostResolutionResult | JsonDict]


class HostResolutionFailure(ValueError):
    """Fail-closed resolver error carrying the gate repair code."""

    def __init__(self, repair_code: str, message: str) -> None:
        super().__init__(message)
        self.repair_code = repair_code


@dataclass(frozen=True, slots=True)
class HostResolverRuntimeBinding:
    tool: str
    implementation: str
    resolution_layer: str
    resolver_id: str = ""
    schema_version: str = HOST_RESOLVER_BINDING_SCHEMA_VERSION
    handler: HostResolver | None = field(default=None, repr=False, compare=False)
    failure_repair_code: str = "invalid_source_packet"

    def __post_init__(self) -> None:
        if not self.tool or not self.implementation or not self.resolution_layer:
            raise ValueError("host resolver binding fields must be non-empty")
        if not self.resolver_id:
            object.__setattr__(
                self,
                "resolver_id",
                f"openeta.host_resolver.{self.tool}.v1",
            )

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "resolver_id": self.resolver_id,
            "tool": self.tool,
            "implementation": self.implementation,
            "resolution_layer": self.resolution_layer,
            "dispatchable": self.handler is not None,
            "failure_repair_code": self.failure_repair_code,
        }


def _binding(
    tool: str,
    implementation: str,
    layer: str,
    handler: HostResolver | None = None,
    failure_repair_code: str = "invalid_source_packet",
) -> HostResolverRuntimeBinding:
    return HostResolverRuntimeBinding(
        tool,
        implementation,
        layer,
        handler=handler,
        failure_repair_code=failure_repair_code,
    )


def resolve_host_parameters(
    *,
    tool_name: str,
    resolver_id: str,
    parameters: JsonDict,
    memory: Any,
) -> HostResolutionResult:
    """Dispatch host resolution through the stable id declared by ToolContract."""

    binding = DEFAULT_HOST_RESOLVER_BINDINGS.get(tool_name)
    if binding is None:
        raise HostResolutionFailure(
            "host_resolver_binding_missing",
            f"no host resolver binding is registered for {tool_name}",
        )
    if resolver_id != binding.resolver_id:
        raise HostResolutionFailure(
            "host_resolver_identity_mismatch",
            f"{tool_name} declares resolver {resolver_id!r}, expected {binding.resolver_id!r}",
        )
    if binding.handler is None:
        raise HostResolutionFailure(
            "host_resolver_not_dispatchable",
            f"host resolver {resolver_id} remains on its legacy runtime branch",
        )
    try:
        result = binding.handler(parameters, memory)
        if isinstance(result, HostResolutionResult):
            return result
        if isinstance(result, dict):
            return HostResolutionResult(parameters=result)
        raise TypeError("host resolver returned a non-object result")
    except HostResolutionFailure:
        raise
    except (TypeError, ValueError) as exc:
        raise HostResolutionFailure(binding.failure_repair_code, str(exc)) from exc


def _require_memory(memory: Any) -> Any:
    if memory is None:
        raise HostResolutionFailure(
            "invalid_source_packet",
            "runtime memory is unavailable",
        )
    return memory


def _resolve_sam3_input(parameters: JsonDict, memory: Any) -> JsonDict:
    memory = _require_memory(memory)
    source = memory.resolve_observation_packet(
        str(parameters.get("source_packet_id") or ""),
        str(parameters.get("camera_frame_id") or ""),
    )
    mode = str(parameters.get("mode") or "text").lower()
    if mode == "points" or parameters.get("points") is not None:
        same_view_error = memory.same_view_point_grounding_source_error([source])
        if same_view_error:
            raise HostResolutionFailure(
                "same_view_packet_mismatch",
                same_view_error,
            )
    return parameters


def _resolve_retrieve_asset_reference_input(
    parameters: JsonDict,
    memory: Any,
) -> JsonDict:
    memory = _require_memory(memory)
    source = memory.resolve_observation_packet(
        str(parameters.get("source_packet_id") or ""),
        str(parameters.get("camera_frame_id") or ""),
    )
    return {
        "environment": parameters.get("environment"),
        "target_object": parameters.get("target_object"),
        "scene_image": source.get("rgb"),
        "_source_observation": source,
    }


def _resolve_molmopoint_input(parameters: JsonDict, memory: Any) -> JsonDict:
    memory = _require_memory(memory)
    sources = parameters.get("sources")
    if not isinstance(sources, list) or not sources:
        raise HostResolutionFailure(
            "invalid_source_packet",
            "molmopoint requires one to four ordered packet sources",
        )
    if len(sources) > 4:
        raise HostResolutionFailure(
            "invalid_source_packet",
            "molmopoint accepts at most four packet sources",
        )
    resolved_sources = []
    for index, item in enumerate(sources):
        if not isinstance(item, dict):
            raise HostResolutionFailure(
                "invalid_source_packet",
                f"molmopoint sources[{index}] must be an object",
            )
        resolved_sources.append(
            memory.resolve_observation_packet(
                str(item.get("source_packet_id") or ""),
                str(item.get("camera_frame_id") or ""),
            )
        )
    same_view_error = memory.same_view_point_grounding_source_error(resolved_sources)
    if same_view_error:
        raise HostResolutionFailure(
            "same_view_packet_mismatch",
            same_view_error,
        )
    return {
        "images": [source.get("rgb") for source in resolved_sources],
        "prompt": parameters.get("prompt"),
        "_source_observations": resolved_sources,
    }


def _resolve_estimate_depth_prior_input(
    parameters: JsonDict,
    memory: Any,
) -> JsonDict:
    memory = _require_memory(memory)
    source = memory.resolve_observation_packet(
        str(parameters.get("source_packet_id") or ""),
        str(parameters.get("camera_frame_id") or ""),
    )
    intrinsics = source.get("intrinsics")
    if not isinstance(intrinsics, dict) or not intrinsics:
        raise HostResolutionFailure(
            "invalid_source_packet",
            "estimate_depth_prior source camera lacks intrinsics",
        )
    return {
        "rgb": source.get("rgb"),
        "intrinsics": dict(intrinsics),
        "camera_id": source.get("frame_id"),
        "bundle_id": f"{source.get('packet_id')}:{source.get('frame_id')}",
        "source_packet_id": source.get("packet_id"),
        "camera_frame_id": source.get("frame_id"),
        "_source_observation": source,
        **(
            {"resolution_level": parameters["resolution_level"]}
            if "resolution_level" in parameters
            else {}
        ),
    }


def _resolve_enhance_depth_input(parameters: JsonDict, memory: Any) -> JsonDict:
    memory = _require_memory(memory)
    resolution = memory.resolve_depth_enhancement_input(
        str(parameters.get("source_packet_id") or ""),
        str(parameters.get("camera_frame_id") or ""),
    )
    resolved = resolution.get("parameters")
    if not isinstance(resolved, dict):
        raise HostResolutionFailure(
            "invalid_source_packet",
            "depth enhancement resolver returned invalid parameters",
        )
    return {
        **resolved,
        **(
            {"config": parameters["config"]}
            if "config" in parameters
            else {}
        ),
    }


def _bundle_result(parameters: JsonDict, resolution: JsonDict) -> HostResolutionResult:
    resolved = resolution.get("parameters")
    if not isinstance(resolved, dict):
        raise ValueError("provenance resolver returned invalid parameters")
    return HostResolutionResult(parameters=parameters, evidence=resolution)


def _resolve_anyplace_input(
    parameters: JsonDict,
    memory: Any,
) -> HostResolutionResult:
    memory = _require_memory(memory)
    resolution = memory.resolve_anyplace_input_bundle(
        str(parameters.get("bundle_id") or "").strip()
    )
    return _bundle_result(dict(resolution.get("parameters") or {}), resolution)


def _resolve_placement_candidate_input(
    parameters: JsonDict,
    memory: Any,
) -> HostResolutionResult:
    memory = _require_memory(memory)
    resolution = memory.resolve_placement_candidate_input(
        placement_result_id=str(parameters.get("placement_result_id") or ""),
        candidate_id=str(parameters.get("candidate_id") or ""),
    )
    return _bundle_result(dict(resolution.get("parameters") or {}), resolution)


def _resolve_grasp_candidate_input(
    parameters: JsonDict,
    memory: Any,
) -> HostResolutionResult:
    memory = _require_memory(memory)
    resolution = memory.resolve_grasp_candidate_input(
        grasp_result_id=str(parameters.get("grasp_result_id") or ""),
        candidate_id=str(parameters.get("candidate_id") or ""),
    )
    resolved = dict(resolution.get("parameters") or {})
    resolved.update(
        {
            "grasp_result_id": parameters.get("grasp_result_id"),
            "candidate_id": parameters.get("candidate_id"),
            **{
                key: parameters[key]
                for key in (
                    "target_geometry_family",
                    "target_class",
                    "strategy_id",
                    "articulated_handle_options",
                    "pregrasp_distance_m",
                )
                if key in parameters
            },
        }
    )
    return _bundle_result(resolved, resolution)


def _resolve_wrist_alignment_input(
    parameters: JsonDict,
    memory: Any,
) -> HostResolutionResult:
    memory = _require_memory(memory)
    resolution = memory.resolve_wrist_alignment_bundle(
        str(parameters.get("bundle_id") or "").strip()
    )
    resolved = dict(resolution.get("parameters") or {})
    if "max_correction_m" in parameters:
        resolved["max_correction_m"] = parameters["max_correction_m"]
    return _bundle_result(resolved, resolution)


def _resolve_wrist_viewpoint_input(
    parameters: JsonDict,
    memory: Any,
) -> HostResolutionResult:
    memory = _require_memory(memory)
    resolution = memory.resolve_wrist_viewpoint_input(
        compiled_grasp_id=str(parameters.get("compiled_grasp_id") or ""),
        source_packet_id=str(parameters.get("source_packet_id") or ""),
        camera_frame_id=str(parameters.get("camera_frame_id") or ""),
    )
    return _bundle_result(dict(resolution.get("parameters") or {}), resolution)


def _resolve_grasp_input(
    parameters: JsonDict,
    memory: Any,
) -> HostResolutionResult:
    memory = _require_memory(memory)
    resolution = memory.resolve_grasp_input_bundle(
        str(parameters.get("bundle_id") or "").strip()
    )
    resolved = dict(resolution.get("parameters") or {})
    if "backend_preference" in parameters:
        resolved["backend_preference"] = parameters["backend_preference"]
    return _bundle_result(resolved, resolution)


DEFAULT_HOST_RESOLVER_BINDINGS = {
    item.tool: item
    for item in (
        _binding("observe", "agent.runtime.memory.AgentMemory.add_observation", "post_environment"),
        _binding(
            "enhance_depth",
            "agent.tools.runtime_contract_bindings._resolve_enhance_depth_input",
            "pipeline",
            _resolve_enhance_depth_input,
        ),
        _binding(
            "estimate_depth_prior",
            "agent.tools.runtime_contract_bindings._resolve_estimate_depth_prior_input",
            "pipeline",
            _resolve_estimate_depth_prior_input,
        ),
        _binding("create_simulator_env", "agent.tools.sim_mcp.SimulatorEnvironmentCreator.handler", "handler"),
        _binding("close_simulator_env", "agent.tools.sim_mcp.SimulatorEnvironmentCloser.handler", "handler"),
        _binding("python_exec", "agent.tools.coding.PythonExecRuntime.handler", "handler"),
        _binding("web_search", "agent.tools.web_access.HostedWebSearchClient", "handler"),
        _binding("web_fetch", "agent.tools.web_access.build_web_fetch_handler", "handler"),
        _binding(
            "sam3",
            "agent.tools.runtime_contract_bindings._resolve_sam3_input",
            "pipeline",
            _resolve_sam3_input,
        ),
        _binding(
            "retrieve_asset_reference",
            "agent.tools.runtime_contract_bindings._resolve_retrieve_asset_reference_input",
            "pipeline",
            _resolve_retrieve_asset_reference_input,
        ),
        _binding(
            "molmopoint",
            "agent.tools.runtime_contract_bindings._resolve_molmopoint_input",
            "pipeline",
            _resolve_molmopoint_input,
        ),
        _binding("select_sam3_detection", "agent.runtime.memory.AgentMemory.resolve_sam3_selection", "memory"),
        _binding("reject_sam3_detections", "agent.runtime.memory.AgentMemory.reject_sam3_detections", "memory"),
        _binding(
            "grasp_pose_estimate",
            "agent.tools.runtime_contract_bindings._resolve_grasp_input",
            "memory",
            _resolve_grasp_input,
            "invalid_provenance_bundle",
        ),
        _binding(
            "anyplace",
            "agent.tools.runtime_contract_bindings._resolve_anyplace_input",
            "memory",
            _resolve_anyplace_input,
            "invalid_provenance_bundle",
        ),
        _binding(
            "camera_pose_to_world",
            "agent.tools.runtime_contract_bindings._resolve_placement_candidate_input",
            "memory",
            _resolve_placement_candidate_input,
            "invalid_provenance_bundle",
        ),
        _binding("propose_calibration_profile", "agent.runtime.calibration.CalibrationLifecycleManager.propose_handler", "handler"),
        _binding("promote_calibration_profile", "agent.runtime.calibration.CalibrationLifecycleManager.promote_handler", "handler"),
        _binding("propose_grasp_strategy", "agent.runtime.grasp_strategy_lifecycle.GraspStrategyLifecycleManager.propose_handler", "handler"),
        _binding("promote_grasp_strategy", "agent.runtime.grasp_strategy_lifecycle.GraspStrategyLifecycleManager.promote_handler", "handler"),
        _binding(
            "compile_grasp_seed",
            "agent.tools.runtime_contract_bindings._resolve_grasp_candidate_input",
            "memory",
            _resolve_grasp_candidate_input,
            "invalid_provenance_bundle",
        ),
        _binding(
            "compute_wrist_alignment",
            "agent.tools.runtime_contract_bindings._resolve_wrist_alignment_input",
            "memory",
            _resolve_wrist_alignment_input,
            "invalid_provenance_bundle",
        ),
        _binding(
            "propose_wrist_viewpoints",
            "agent.tools.runtime_contract_bindings._resolve_wrist_viewpoint_input",
            "memory",
            _resolve_wrist_viewpoint_input,
            "invalid_provenance_bundle",
        ),
        _binding("prepare_attachment_probe", "agent.tools.attachment_probe.prepare_attachment_probe", "handler"),
        _binding("assess_attachment_probe", "agent.tools.attachment_probe.assess_attachment_probe", "handler"),
        _binding("move_to", "agent.runtime.memory.AgentMemory.resolve_ik_motion_reference", "memory"),
        _binding("follow_eef_trajectory", "agent.runtime.memory.AgentMemory.resolve_ik_trajectory_reference", "memory"),
        _binding("ik_preview_check", "agent.runtime.pipeline.ActionPipeline.compile", "pipeline"),
        _binding("register_skill", "agent.runtime.runtime_assembly._skill_change_handler", "handler"),
        _binding("update_skill", "agent.runtime.runtime_assembly._skill_change_handler", "handler"),
    )
}


def bind_host_resolution_contract(
    tool_name: str,
    contract: HostResolutionContract,
) -> HostResolutionContract:
    """Replace prose resolver labels with a stable, auditable runtime identity."""

    if contract.mode == "none":
        return contract
    try:
        binding = DEFAULT_HOST_RESOLVER_BINDINGS[tool_name]
    except KeyError as exc:
        raise ValueError(f"missing host resolver runtime binding for {tool_name}") from exc
    return replace(
        contract,
        resolver=binding.resolver_id,
        implementation=binding.implementation,
        resolution_layer=binding.resolution_layer,
        contract_driven_dispatch=binding.handler is not None,
    )


def audit_host_resolver_bindings(catalog: ToolContractCatalog) -> JsonDict:
    issues: list[JsonDict] = []
    bound_count = 0
    contract_driven_tools: list[str] = []
    identity_only_tools: list[str] = []
    for contract in catalog.list():
        resolution = contract.host_resolution
        if resolution.mode == "none":
            continue
        binding = DEFAULT_HOST_RESOLVER_BINDINGS.get(contract.name)
        if binding is None:
            issues.append({"tool": contract.name, "code": "binding_missing"})
            continue
        bound_count += 1
        if binding.handler is not None:
            contract_driven_tools.append(contract.name)
        else:
            identity_only_tools.append(contract.name)
        for field, observed, expected in (
            ("resolver", resolution.resolver, binding.resolver_id),
            ("implementation", resolution.implementation, binding.implementation),
            ("resolution_layer", resolution.resolution_layer, binding.resolution_layer),
            (
                "contract_driven_dispatch",
                resolution.contract_driven_dispatch,
                binding.handler is not None,
            ),
        ):
            if observed != expected:
                issues.append(
                    {
                        "tool": contract.name,
                        "code": "binding_identity_mismatch",
                        "field": field,
                        "observed": observed,
                        "expected": expected,
                    }
                )
        if binding.handler is not None:
            declared_repair_codes = {
                code
                for gate_binding in contract.gate.bindings
                for code in gate_binding.repair_codes
            }
            if binding.failure_repair_code not in declared_repair_codes:
                issues.append(
                    {
                        "tool": contract.name,
                        "code": "resolver_failure_code_unbound",
                        "observed": binding.failure_repair_code,
                    }
                )
    return {
        "schema_version": "openeta.host_resolver_binding_audit.v1",
        "bound_count": bound_count,
        "contract_driven_dispatch_count": len(contract_driven_tools),
        "contract_driven_dispatch_tools": sorted(contract_driven_tools),
        "identity_only_count": len(identity_only_tools),
        "identity_only_tools": sorted(identity_only_tools),
        "issue_count": len(issues),
        "conformant": not issues,
        "issues": issues,
    }
