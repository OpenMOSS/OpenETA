"""Shared runtime assembly for interactive and batch OpenETA entry points."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

from adapter.protocol import JsonDict
from agent.backends.planner import PlannerBackend
from agent.backends.provider_config import PlannerProviderConfig
from agent.runtime.calibration import (
    BackendCalibrationReviewer,
    CalibrationLifecycleConfig,
    CalibrationLifecycleManager,
)
from agent.runtime.checkers import CheckerSubagentConfig
from agent.runtime.grasp_strategy_lifecycle import (
    BackendGraspStrategyReviewer,
    GraspStrategyLifecycleConfig,
    GraspStrategyLifecycleManager,
)
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerContextConfig, ToolCallingPlanner
from agent.runtime.reference_localization import (
    REFERENCE_POINT_LOCALIZATION_MAX_OUTPUT_TOKENS,
    BackendReferencePointLocalizer,
)
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.self_improvement import (
    BackendReviewedSkillAutoApplier,
    SelfImprovementConfig,
    SelfImprovementReviewer,
    SkillReviewProposalStore,
)
from agent.runtime.session_workspace import SessionWorkspace
from agent.runtime.skill_authoring import (
    SKILL_AUTHORING_MAX_OUTPUT_TOKENS,
    BackendSkillAuthoringSubagent,
    BackendSkillChangeReviewer,
    SkillAuthoringRequest,
)
from agent.runtime.skills import SkillSpec
from agent.runtime.supervision import (
    BackendActionReviewer,
    SupervisionGate,
    SupervisionPolicy,
    SupervisionProfile,
)
from agent.tools.asset_references import (
    build_asset_reference_handler,
    build_object_memory_configuration_warning_handler,
    build_object_memory_reference_handler,
    load_configured_asset_reference_catalog,
)
from agent.tools.attachment_probe import (
    build_assess_attachment_probe_handler,
    build_prepare_attachment_probe_handler,
)
from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.depth_prefetch import DepthPriorPrefetchCoordinator
from agent.tools.handlers import (
    bind_dummy_tool_handlers,
    build_anygrasp_handler,
    build_anyplace_handler,
    build_depth_prior_handler,
    build_grasp_pose_estimate_handler,
    build_graspgenx_handler,
    build_molmopoint_handler,
    build_sam3_handler,
    build_sse_anygrasp_mcp_grasper,
    build_sse_anyplace_mcp_placer,
    build_sse_depth_prior_mcp_estimator,
    build_sse_graspgenx_mcp_gripper_lister,
    build_sse_graspgenx_mcp_predictor,
    build_sse_molmopoint_mcp_pointer,
    build_sse_sam3_mcp_segmenter,
)
from agent.tools.grasp_geometry import build_compile_grasp_seed_handler
from agent.tools.mcp_registry import load_mcp_server_url
from agent.tools.object_memory import (
    ObjectMemoryBankClient,
    ObjectMemoryBankConfigurationError,
    load_configured_object_memory_bank,
)
from agent.tools.registry import (
    ToolEventListener,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_default_tool_registry,
)
from agent.tools.sim_mcp import (
    SimulatorMcpResponseCallback,
    SimulatorMcpToolProxyConfig,
    SimulatorMcpTransport,
    bind_simulator_mcp_tool_handlers,
)
from agent.tools.web_access import WebAccessConfig, bind_configured_web_tool_handlers


BackendFactory = Callable[..., PlannerBackend]
McpUrlLoader = Callable[..., str]
ApprovalCallback = Callable[[ToolExecutionContext], bool]
PublicationApproval = Callable[[JsonDict], bool]
SkillApproval = Callable[[str], bool]


REMOTE_PLACEHOLDER_TOOLS = (
    "scene_detector",
    "sam3",
    "anygrasp",
    "grasp_pose_estimate",
    "contact_graspnet",
    "graspgenx",
    "list_graspgenx_grippers",
    "hand_pose_database",
    "ik_preview_check",
    "obstacle_avoidance",
    "lower_body_control_policy",
    "estimate_depth_prior",
)

ENVIRONMENT_PLACEHOLDER_TOOLS = (
    "observe",
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)


@dataclass(frozen=True, slots=True)
class RuntimeMcpEndpoints:
    """Configured remote perception backends for one runtime."""

    sam3_url: str = ""
    depth_prior_url: str = ""
    anygrasp_url: str = ""
    anyplace_url: str = ""
    graspgenx_url: str = ""
    contact_graspnet_url: str = ""
    molmopoint_url: str = ""


@dataclass(slots=True)
class RuntimeAssemblyConfig:
    """Host-owned inputs shared by TUI and batch runtime construction."""

    workspace: SessionWorkspace
    provider: PlannerProviderConfig
    backend_factory: BackendFactory
    supervision_policy: SupervisionPolicy
    endpoints: RuntimeMcpEndpoints = field(default_factory=RuntimeMcpEndpoints)
    simulator_transport: SimulatorMcpTransport | None = None
    simulator_proxy_config: SimulatorMcpToolProxyConfig | None = None
    web_access_config: WebAccessConfig | None = None
    mcp_response_callback: SimulatorMcpResponseCallback | None = None
    allow_outside_sandbox: bool = False
    approve_outside_sandbox: Callable[[ToolExecutionContext, str], bool] | None = None
    human_action_approval: ApprovalCallback | None = None
    calibration_approval: PublicationApproval | None = None
    strategy_approval: PublicationApproval | None = None
    skill_approval: SkillApproval | None = None
    supervision_policy_provider: Callable[[], SupervisionPolicy] | None = None
    pre_safety_checks: dict[str, str] = field(default_factory=dict)
    tool_listeners: tuple[ToolEventListener, ...] = ()
    max_validation_retries: int = 2


@dataclass(frozen=True, slots=True)
class RuntimeAssembly:
    """Assembled runtime plus host controls retained by its entry point."""

    runtime: OpenEtaAgentRuntime
    supervision_gate: SupervisionGate
    depth_prefetch: DepthPriorPrefetchCoordinator | None


def resolve_runtime_mcp_endpoints(
    overrides: RuntimeMcpEndpoints | None = None,
    *,
    loader: McpUrlLoader = load_mcp_server_url,
) -> RuntimeMcpEndpoints:
    """Resolve shared perception endpoint names and aliases."""

    configured = overrides or RuntimeMcpEndpoints()
    return RuntimeMcpEndpoints(
        sam3_url=configured.sam3_url
        or loader("openeta-sam3", aliases=("sam3",)),
        depth_prior_url=configured.depth_prior_url
        or loader(
            "openeta-depth-prior",
            aliases=("depth-prior", "depth_prior", "unidepth"),
        ),
        anygrasp_url=configured.anygrasp_url
        or loader("openeta-anygrasp", aliases=("anygrasp",)),
        anyplace_url=configured.anyplace_url
        or loader("openeta-anyplace", aliases=("anyplace",)),
        graspgenx_url=configured.graspgenx_url
        or loader("openeta-graspgenx", aliases=("graspgenx",)),
        contact_graspnet_url=configured.contact_graspnet_url
        or loader(
            "openeta-contact-graspnet",
            aliases=("contact-graspnet", "contact_graspnet"),
        ),
        molmopoint_url=configured.molmopoint_url
        or loader(
            "openeta-molmopoint",
            aliases=("molmopoint", "molmo-point"),
        ),
    )


def assemble_runtime(config: RuntimeAssemblyConfig) -> RuntimeAssembly:
    """Build one fail-closed runtime from shared host configuration."""

    workspace = config.workspace
    tools = bind_dummy_tool_handlers(
        build_default_tool_registry(),
        include_dummy_safety=False,
    )
    for name in REMOTE_PLACEHOLDER_TOOLS:
        tools.unbind_handler(name)
    if config.simulator_transport is None:
        for name in ENVIRONMENT_PLACEHOLDER_TOOLS:
            tools.unbind_handler(name)

    tools.bind_handler(
        "compile_grasp_seed",
        build_compile_grasp_seed_handler(
            workspace.grasp_profile_path,
            strategy_root=workspace.grasp_strategy_root,
        ),
        replace=True,
    )
    tools.bind_handler(
        "prepare_attachment_probe",
        build_prepare_attachment_probe_handler(),
        replace=True,
    )
    tools.bind_handler(
        "assess_attachment_probe",
        build_assess_attachment_probe_handler(
            config.backend_factory(max_tokens=256, max_vision_images=4)
        ),
        replace=True,
    )
    if config.simulator_transport is not None:
        bind_simulator_mcp_tool_handlers(
            tools,
            transport=config.simulator_transport,
            config=config.simulator_proxy_config,
            response_callback=config.mcp_response_callback,
            replace=True,
        )

    artifact_root = workspace.artifacts_dir
    tools.bind_handler(
        "python_exec",
        PythonExecRuntime(
            PythonExecConfig(
                mcp_transport=config.simulator_transport,
                image_output_root=str(artifact_root / "images"),
                text_output_root=str(artifact_root / "text"),
                response_output_root=str(artifact_root / "responses"),
                allow_outside_sandbox=config.allow_outside_sandbox,
                approve_outside_sandbox=config.approve_outside_sandbox,
                mcp_response_callback=config.mcp_response_callback,
                workspace_root=str(workspace.sandbox_dir),
            )
        ).handler,
        replace=True,
    )

    policy_provider = config.supervision_policy_provider or (
        lambda: config.supervision_policy
    )
    calibration_manager = CalibrationLifecycleManager(
        config=CalibrationLifecycleConfig(
            root=workspace.calibrations_dir,
            evidence_roots=(Path(".openeta_memory"), workspace.root),
            publication_mode=lambda: policy_provider().skill_change_mode,
            human_approval=config.calibration_approval,
        ),
        reviewer=BackendCalibrationReviewer(config.backend_factory()),
    )
    tools.bind_handler(
        "propose_calibration_profile",
        calibration_manager.propose_handler,
        replace=True,
    )
    tools.bind_handler(
        "promote_calibration_profile",
        calibration_manager.promote_handler,
        replace=True,
    )

    staged_profile = workspace.grasp_profile()
    strategy_lifecycle_root = workspace.working_dir / "grasp_strategy_lifecycle"
    strategy_manager = GraspStrategyLifecycleManager(
        config=GraspStrategyLifecycleConfig(
            root=strategy_lifecycle_root,
            session_strategy_root=strategy_lifecycle_root / "staged",
            calibration_profile=staged_profile,
            evidence_roots=(Path(".openeta_memory"), workspace.root),
            publication_mode=lambda: policy_provider().skill_change_mode,
            human_approval=config.strategy_approval,
        ),
        reviewer=BackendGraspStrategyReviewer(config.backend_factory()),
    )
    tools.bind_handler(
        "propose_grasp_strategy",
        strategy_manager.propose_handler,
        replace=True,
    )
    tools.bind_handler(
        "promote_grasp_strategy",
        strategy_manager.promote_handler,
        replace=True,
    )

    bind_configured_web_tool_handlers(
        tools,
        config=config.web_access_config,
        provider_config=config.provider,
    )
    depth_prefetch = bind_runtime_perception_tools(
        tools,
        endpoints=config.endpoints,
        backend_factory=config.backend_factory,
        artifact_root=artifact_root,
    )

    planner = ToolCallingPlanner(
        config.backend_factory(),
        max_validation_retries=config.max_validation_retries,
        context_config=PlannerContextConfig(
            context_window_tokens=config.provider.context_window_tokens,
            token_estimator_model=config.provider.model,
        ),
    )
    skill_review_config = SelfImprovementConfig(
        proposal_root=workspace.working_dir / "skill_reviews" / "pending",
        auto_apply_reviewed=(
            config.supervision_policy.profile == SupervisionProfile.REVIEWED_AUTONOMY
        ),
        skill_dir=workspace.skills_dir,
        task_playbook_candidate_root=workspace.working_dir / "task_playbook_reviews",
        rollout_root=workspace.memory_root,
    )
    skill_reviewer = SelfImprovementReviewer(
        config=skill_review_config,
        store=SkillReviewProposalStore(skill_review_config.proposal_root),
    )
    checker_config = _checker_config(
        tools,
        pre_safety_checks=config.pre_safety_checks,
    )
    runtime = OpenEtaAgentRuntime(
        planner=planner,
        tools=tools,
        memory=AgentMemory(store=JsonMemoryStore(root=workspace.memory_root)),
        skills=workspace.skill_registry(),
        pipeline=ActionPipeline(checker_subagents=checker_config),
        self_improvement_reviewer=skill_reviewer,
        default_session_id=workspace.session_id,
    )
    configure_runtime_self_improvement(
        runtime,
        policy=config.supervision_policy,
        backend_factory=config.backend_factory,
    )
    _bind_skill_change_tools(
        runtime,
        backend_factory=config.backend_factory,
        policy_provider=policy_provider,
        human_approval=config.skill_approval,
    )

    gate = SupervisionGate(
        config.supervision_policy,
        human_approval=config.human_action_approval,
        action_reviewer=BackendActionReviewer(config.backend_factory(max_tokens=512)),
    )
    tools.set_execution_gate(gate.authorize)
    for listener in config.tool_listeners:
        tools.add_listener(listener)
    return RuntimeAssembly(
        runtime=runtime,
        supervision_gate=gate,
        depth_prefetch=depth_prefetch,
    )


def configure_runtime_self_improvement(
    runtime: OpenEtaAgentRuntime,
    *,
    policy: SupervisionPolicy,
    backend_factory: BackendFactory,
) -> None:
    """Apply the active supervision profile to reviewed skill auto-apply."""

    reviewer = runtime.self_improvement_reviewer
    reviewed = policy.profile == SupervisionProfile.REVIEWED_AUTONOMY
    reviewer.config = replace(
        reviewer.config,
        auto_apply_reviewed=reviewed,
    )
    if not reviewed:
        reviewer.auto_applier = None
        return
    reviewer.auto_applier = BackendReviewedSkillAutoApplier(
        author=BackendSkillAuthoringSubagent(
            backend_factory(max_tokens=SKILL_AUTHORING_MAX_OUTPUT_TOKENS)
        ),
        reviewer=BackendSkillChangeReviewer(backend_factory()),
        executable_tools=_skill_authoring_tools(runtime.tools),
    )


def bind_runtime_perception_tools(
    tools: ToolRegistry,
    *,
    endpoints: RuntimeMcpEndpoints,
    backend_factory: BackendFactory,
    artifact_root: Path,
) -> DepthPriorPrefetchCoordinator | None:
    object_memory_configuration_error = ""
    try:
        object_memory_config = load_configured_object_memory_bank()
    except ObjectMemoryBankConfigurationError as exc:
        object_memory_config = None
        object_memory_configuration_error = str(exc)
    asset_catalog = load_configured_asset_reference_catalog()
    if object_memory_config is not None:
        tools.bind_handler(
            "retrieve_asset_reference",
            build_object_memory_reference_handler(
                ObjectMemoryBankClient(object_memory_config),
                BackendReferencePointLocalizer(
                    backend_factory(
                        max_tokens=REFERENCE_POINT_LOCALIZATION_MAX_OUTPUT_TOKENS,
                        max_vision_images=4,
                    )
                ),
                output_root=artifact_root / "asset_references",
            ),
            replace=True,
        )
    elif asset_catalog is not None and not object_memory_configuration_error:
        tools.bind_handler(
            "retrieve_asset_reference",
            build_asset_reference_handler(
                asset_catalog,
                output_root=artifact_root / "asset_references",
            ),
            replace=True,
        )
    else:
        tools.bind_handler(
            "retrieve_asset_reference",
            build_object_memory_configuration_warning_handler(
                configuration_error=object_memory_configuration_error,
            ),
            replace=True,
        )

    depth_prefetch: DepthPriorPrefetchCoordinator | None = None
    if endpoints.depth_prior_url:
        depth_handler = build_depth_prior_handler(
            build_sse_depth_prior_mcp_estimator(url=endpoints.depth_prior_url),
            output_root=artifact_root / "depth_prior_results",
        )
        depth_prefetch = DepthPriorPrefetchCoordinator(
            depth_handler,
            spec=tools.get("estimate_depth_prior"),
        )
        tools.bind_handler(
            "estimate_depth_prior",
            depth_prefetch.handler,
            replace=True,
        )
    if endpoints.sam3_url:
        tools.bind_handler(
            "sam3",
            build_sam3_handler(
                build_sse_sam3_mcp_segmenter(url=endpoints.sam3_url),
                segment_points=build_sse_sam3_mcp_segmenter(
                    url=endpoints.sam3_url,
                    tool_name="segment_points",
                ),
                depth_prior_prefetch=(
                    depth_prefetch.prefetch_for_sam3
                    if depth_prefetch is not None
                    else None
                ),
                output_root=artifact_root / "sam3_images",
                result_output_root=artifact_root / "sam3_results",
            ),
            replace=True,
        )
    if endpoints.molmopoint_url:
        tools.bind_handler(
            "molmopoint",
            build_molmopoint_handler(
                build_sse_molmopoint_mcp_pointer(url=endpoints.molmopoint_url),
                output_root=artifact_root / "molmopoint_results",
            ),
            replace=True,
        )
    if endpoints.anyplace_url:
        tools.bind_handler(
            "anyplace",
            build_anyplace_handler(
                build_sse_anyplace_mcp_placer(url=endpoints.anyplace_url),
                output_root=artifact_root / "anyplace_results",
            ),
            replace=True,
        )

    grasp_backends = {}
    if endpoints.anygrasp_url:
        grasp_backends["anygrasp"] = build_anygrasp_handler(
            build_sse_anygrasp_mcp_grasper(url=endpoints.anygrasp_url),
            output_root=artifact_root / "anygrasp_results",
        )
    if endpoints.graspgenx_url:
        list_grippers = build_sse_graspgenx_mcp_gripper_lister(
            url=endpoints.graspgenx_url
        )
        grasp_backends["graspgenx"] = build_graspgenx_handler(
            build_sse_graspgenx_mcp_predictor(url=endpoints.graspgenx_url),
            list_grippers,
            output_root=artifact_root / "graspgenx_results",
        )
    # Contact-GraspNet is temporarily disabled for the simulator drawer track.
    # Keep its endpoint/configuration and implementation available for a later
    # re-enable, but do not expose it as an executable grasp backend here.
    if grasp_backends:
        tools.bind_handler(
            "grasp_pose_estimate",
            build_grasp_pose_estimate_handler(grasp_backends),
            replace=True,
        )
    return depth_prefetch


def _bind_skill_change_tools(
    runtime: OpenEtaAgentRuntime,
    *,
    backend_factory: BackendFactory,
    policy_provider: Callable[[], SupervisionPolicy],
    human_approval: SkillApproval | None,
) -> None:
    runtime.tools.bind_handler(
        "register_skill",
        _skill_change_handler(
            runtime,
            operation="register",
            backend_factory=backend_factory,
            policy_provider=policy_provider,
            human_approval=human_approval,
        ),
        replace=True,
    )
    runtime.tools.bind_handler(
        "update_skill",
        _skill_change_handler(
            runtime,
            operation="update",
            backend_factory=backend_factory,
            policy_provider=policy_provider,
            human_approval=human_approval,
        ),
        replace=True,
    )


def _skill_change_handler(
    runtime: OpenEtaAgentRuntime,
    *,
    operation: str,
    backend_factory: BackendFactory,
    policy_provider: Callable[[], SupervisionPolicy],
    human_approval: SkillApproval | None,
) -> Callable[[ToolExecutionContext], ToolResult]:
    def handler(context: ToolExecutionContext) -> ToolResult:
        try:
            requested_name = str(context.parameters.get("name") or "").strip()
            if not requested_name:
                raise ValueError("skill name is required")
            if not _has_skill_authoring_instruction(
                context.parameters,
                operation=operation,
            ):
                raise ValueError(f"{operation}_skill requires authoring instructions")
            current = None
            try:
                current = runtime.skills.get(requested_name)
            except KeyError:
                if operation == "update":
                    raise ValueError(f"Unknown skill: {requested_name}") from None
            else:
                if operation == "register":
                    raise ValueError(f"Skill already registered: {requested_name}")
                if not current.editable:
                    raise ValueError(f"Skill is not editable: {requested_name}")
            request = SkillAuthoringRequest(
                operation=operation,
                parameters=context.parameters,
                current_skill=current,
                executable_tools=_skill_authoring_tools(runtime.tools),
            )
            authored = BackendSkillAuthoringSubagent(
                backend_factory(max_tokens=SKILL_AUTHORING_MAX_OUTPUT_TOKENS)
            ).author(request)
            review = _authorize_skill_change(
                request=request,
                skill=authored.skill,
                policy=policy_provider(),
                backend_factory=backend_factory,
                human_approval=human_approval,
            )
            if not review["approved"]:
                raise PermissionError(str(review.get("reason") or "skill change denied"))
            if operation == "register":
                runtime.skills.register(authored.skill)
            else:
                runtime.skills.update(authored.skill)
        except Exception as exc:  # noqa: BLE001 - provider failures stay structured.
            return ToolResult(
                False,
                content=f"skill {operation} failed: {exc}",
                details={
                    "tool": context.name,
                    "reason": "skill_authoring_failed",
                    "error_type": type(exc).__name__,
                },
            )
        action = "registered" if operation == "register" else "updated"
        return ToolResult(
            True,
            content=f"skill {action}: {authored.skill.name}",
            details={
                "tool": context.name,
                "skill": authored.skill.name,
                "source": authored.skill.source,
                "authoring": authored.details,
                "provider": authored.provider,
                "model": authored.model,
                "review": review,
            },
        )

    return handler


def _authorize_skill_change(
    *,
    request: SkillAuthoringRequest,
    skill: SkillSpec,
    policy: SupervisionPolicy,
    backend_factory: BackendFactory,
    human_approval: SkillApproval | None,
) -> JsonDict:
    skill_name = str(getattr(skill, "name", "") or "")
    if policy.profile == SupervisionProfile.HUMAN_GATED:
        approved = bool(human_approval and human_approval(skill_name))
        return {
            "approved": approved,
            "source": "human",
            "reason": "Approved by human operator." if approved else "Human approval denied.",
        }
    if policy.profile == SupervisionProfile.STANDARD:
        return {
            "approved": True,
            "source": "runtime_policy",
            "reason": "Standard profile permits session-local registry changes.",
        }
    reviewed = BackendSkillChangeReviewer(backend_factory()).review(
        request=request,
        skill=skill,
    )
    return {
        "approved": reviewed.approved,
        "source": "independent_reviewer",
        "decision": reviewed.decision,
        "reason": reviewed.reason,
        "details": reviewed.details,
    }


def _checker_config(
    tools: ToolRegistry,
    *,
    pre_safety_checks: dict[str, str],
) -> CheckerSubagentConfig:
    validated: dict[str, str] = {}
    for target, checker in pre_safety_checks.items():
        tools.get(target)
        checker_spec = tools.get(checker)
        if checker_spec.category != "safety":
            raise ValueError(f"Configured checker is not a safety tool: {checker}")
        if not tools.can_execute(checker):
            raise ValueError(f"Configured checker has no executable handler: {checker}")
        validated[target] = checker
    return CheckerSubagentConfig(
        pre_safety_checks=validated,
        post_failure_checks=tuple(spec.name for spec in tools.list()),
    )


def _skill_authoring_tools(tools: ToolRegistry) -> tuple[ToolSpec, ...]:
    return tuple(
        spec
        for spec in tools.list()
        if tools.can_execute(spec.name) and spec.category != "skill_management"
    )


def _has_skill_authoring_instruction(
    parameters: JsonDict,
    *,
    operation: str,
) -> bool:
    fields = (
        ("goal", "requirements", "description", "content")
        if operation == "register"
        else ("requested_changes", "requirements", "content")
    )
    return any(str(parameters.get(field) or "").strip() for field in fields)
