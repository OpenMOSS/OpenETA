"""Non-interactive parallel simulator evaluation entry point."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackend,
    ProviderConcurrencyLimiter,
)
from agent.backends.provider_config import load_planner_provider_config
from agent.runtime.episode import OpenEtaEpisodeRunner
from agent.runtime.interactions import (
    PausedEpisodeRecord,
    PausedEpisodeStore,
    new_interaction_id,
    question_from_episode,
)
from agent.runtime.mcp_catalog import discover_mcp_tool_catalog
from agent.runtime.parallel import (
    DEFAULT_PARALLEL_EPISODES,
    MAX_PARALLEL_EPISODES,
    ParallelEpisodeHarness,
    ParallelEpisodeOutcome,
    ParallelEpisodeSpec,
    ParallelEpisodeWorker,
    classify_episode_result,
    episode_failure_error,
)
from agent.runtime.session_workspace import DEFAULT_MEMORY_ROOT, SessionWorkspace
from agent.runtime.runtime_assembly import (
    RuntimeAssemblyConfig,
    RuntimeMcpEndpoints,
    assemble_runtime,
    resolve_runtime_mcp_endpoints,
)
from agent.runtime.skills import BUILTIN_SKILL_DIR
from agent.runtime.supervision import (
    BackendGuidanceResolver,
    SupervisionPolicy,
    SupervisionProfile,
)
from agent.tools.mcp_registry import load_mcp_server_url
from agent.tools.sim_mcp import (
    SimulatorMcpEpisodeConfig,
    SimulatorMcpEpisodeEnvironment,
    SimulatorMcpToolProxyConfig,
    SseSimulatorMcpTransport,
)
from agent.tools.web_access import (
    load_configured_web_access,
)


DEFAULT_PROVIDER_CONCURRENCY = 2
DEFAULT_PROVIDER_QUEUE_TIMEOUT_S = 180.0
DEFAULT_SIM_MCP_TIMEOUT_S = 300.0


def load_parallel_episode_manifest(path: str | Path) -> list[ParallelEpisodeSpec]:
    """Load a JSON list or `{episodes: [...]}` batch manifest."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("batch manifest must be a list or an object with `episodes`")
    specs: list[ParallelEpisodeSpec] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"episodes[{index}] must be an object")
        spec = ParallelEpisodeSpec.from_dict(row, index=index)
        if spec.episode_id in seen_ids:
            raise ValueError(f"duplicate episode_id: {spec.episode_id}")
        seen_ids.add(spec.episode_id)
        specs.append(spec)
    if not specs:
        raise ValueError("batch manifest requires at least one episode")
    return specs


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return cleaned or "episode"


def _session_memory_root(
    *,
    workspace_root: str,
    workspace_parent: str,
) -> Path:
    if workspace_root:
        requested = Path(workspace_root)
        if requested.parent.name in {"sessions", "workspaces"}:
            return requested.parent.parent
        return requested.parent
    if workspace_parent:
        requested = Path(workspace_parent)
        if requested.name in {"sessions", "workspaces"}:
            return requested.parent
        return requested
    return DEFAULT_MEMORY_ROOT


def build_mcp_episode_worker_factory(
    *,
    model_override: str = "",
    sim_url: str = "",
    sam3_url: str = "",
    depth_prior_url: str = "",
    anygrasp_url: str = "",
    anyplace_url: str = "",
    graspgenx_url: str = "",
    contact_graspnet_url: str = "",
    molmopoint_url: str = "",
    supervision_profile: SupervisionProfile | str = SupervisionProfile.STANDARD,
    provider_concurrency: int = DEFAULT_PROVIDER_CONCURRENCY,
    provider_queue_timeout_s: float = DEFAULT_PROVIDER_QUEUE_TIMEOUT_S,
):
    """Build isolated model/runtime/MCP workers for a parallel batch."""

    provider = load_planner_provider_config()
    if model_override:
        provider.model = model_override
    missing = provider.missing_fields()
    if missing:
        raise ValueError(f"planner provider config is missing: {', '.join(missing)}")
    resolved_sim_url = sim_url or load_mcp_server_url("openeta-sim", aliases=("sim",))
    endpoints = resolve_runtime_mcp_endpoints(
        RuntimeMcpEndpoints(
            sam3_url=sam3_url,
            depth_prior_url=depth_prior_url,
            anygrasp_url=anygrasp_url,
            anyplace_url=anyplace_url,
            graspgenx_url=graspgenx_url,
            contact_graspnet_url=contact_graspnet_url,
            molmopoint_url=molmopoint_url,
        ),
        loader=load_mcp_server_url,
    )
    web_access_config = load_configured_web_access(provider_config=provider)
    if not resolved_sim_url:
        raise ValueError("simulator MCP URL is required")
    policy = SupervisionPolicy.for_profile(supervision_profile)
    provider_limiter = ProviderConcurrencyLimiter(
        provider_concurrency,
        queue_timeout_s=provider_queue_timeout_s,
    )

    def new_backend(
        *,
        max_tokens: int | None = None,
        max_vision_images: int | None = None,
    ) -> PlannerBackend:
        return _new_batch_backend(
            provider,
            max_tokens=max_tokens,
            max_vision_images=max_vision_images,
            provider_limiter=provider_limiter,
        )

    def factory(spec: ParallelEpisodeSpec, batch_id: str) -> ParallelEpisodeWorker:
        requested_session_id = str(spec.metadata.get("agent_session_id") or "").strip()
        agent_session_id = requested_session_id or str(uuid4())
        requested_workspace_root = str(spec.metadata.get("workspace_root") or "").strip()
        requested_workspace_parent = str(spec.metadata.get("workspace_parent") or "").strip()
        workspace_memory_root = _session_memory_root(
            workspace_root=requested_workspace_root,
            workspace_parent=requested_workspace_parent,
        )
        source_skills = str(spec.metadata.get("source_skills_root") or "").strip()
        source_grasp_profile = str(spec.metadata.get("calibration_profile_path") or "").strip()
        source_grasp_strategies = str(
            spec.metadata.get("source_grasp_strategies_root") or ""
        ).strip()
        source_task_playbooks = str(
            spec.metadata.get("source_task_playbooks_root") or ""
        ).strip()
        workspace = SessionWorkspace.create(
            agent_session_id,
            root=workspace_memory_root,
            source_skills=source_skills or BUILTIN_SKILL_DIR,
            environment_id=spec.env_id,
            **({"source_grasp_profile": source_grasp_profile} if source_grasp_profile else {}),
            **(
                {"source_grasp_strategies": source_grasp_strategies}
                if source_grasp_strategies
                else {}
            ),
            **(
                {"source_task_playbooks": source_task_playbooks}
                if source_task_playbooks
                else {}
            ),
        )
        staged_grasp_profile = json.loads(workspace.grasp_profile_path.read_text(encoding="utf-8"))
        staged_calibration_id = (
            str(staged_grasp_profile.get("calibration_id") or "")
            if isinstance(staged_grasp_profile, dict)
            else ""
        )
        should_import_legacy = not requested_workspace_root
        if requested_workspace_root:
            should_import_legacy = (
                Path(requested_workspace_root).resolve() != workspace.root.resolve()
            )
        if should_import_legacy:
            workspace.import_legacy_roots(
                memory_root=str(spec.metadata.get("legacy_memory_root") or ""),
                artifact_root=str(spec.metadata.get("legacy_artifact_root") or ""),
            )
        artifact_root = workspace.artifacts_dir
        memory_root = workspace.memory_root
        transport = SseSimulatorMcpTransport(resolved_sim_url)
        proxy_config = SimulatorMcpToolProxyConfig(
            timeout_s=max(DEFAULT_SIM_MCP_TIMEOUT_S, provider.timeout_s),
            image_output_root=artifact_root / "images",
            text_output_root=artifact_root / "text",
            response_output_root=artifact_root / "responses",
        )
        environment = SimulatorMcpEpisodeEnvironment(
            transport=transport,
            config=SimulatorMcpEpisodeConfig(
                env_id=spec.env_id,
                seed=spec.seed,
                timeout_s=max(DEFAULT_SIM_MCP_TIMEOUT_S, provider.timeout_s),
                image_output_root=artifact_root / "images",
            ),
            tool_proxy_config=proxy_config,
        )
        assembly = assemble_runtime(
            RuntimeAssemblyConfig(
                workspace=workspace,
                provider=provider,
                backend_factory=new_backend,
                supervision_policy=policy,
                endpoints=endpoints,
                simulator_transport=transport,
                simulator_proxy_config=proxy_config,
                web_access_config=web_access_config,
                allow_outside_sandbox=False,
                max_validation_retries=2,
            )
        )
        runtime = assembly.runtime
        mcp_tool_catalog = discover_mcp_tool_catalog(
            transport,
            endpoint_url=resolved_sim_url,
            output_root=artifact_root / "responses",
        )
        if mcp_tool_catalog:
            runtime.memory.save_fact(
                "simulator_mcp_tool_catalog",
                mcp_tool_catalog,
                source="mcp.list_tools",
            )
        interaction_store = PausedEpisodeStore()

        def pause(result) -> JsonDict:
            if not result.session_id:
                raise RuntimeError("need_human episode has no session_id")
            interaction_id = new_interaction_id()
            question = question_from_episode(result)
            intervention_count = int(spec.metadata.get("human_intervention_count") or 0)
            record = PausedEpisodeRecord(
                batch_id=batch_id,
                episode_id=spec.episode_id,
                session_id=result.session_id,
                interaction_id=interaction_id,
                question=question,
                task=spec.task,
                env_id=spec.env_id,
                seed=spec.seed,
                max_turns=spec.max_turns,
                max_tool_calls=spec.max_tool_calls,
                timeout_s=spec.timeout_s,
                max_total_tokens=spec.max_total_tokens,
                tool_call_count=int(
                    (result.metadata.get("usage") or {}).get("tool_call_count") or 0
                ),
                total_tokens=int((result.metadata.get("usage") or {}).get("total_tokens") or 0),
                token_usage_sources=dict(
                    (result.metadata.get("usage") or {}).get("token_usage_sources") or {}
                ),
                turn_index=int(result.metadata.get("turn_index") or 0),
                memory_root=str(memory_root),
                artifact_root=str(artifact_root),
                workspace_root=str(workspace.root),
                skills_root=str(workspace.skills_dir),
                sandbox_root=str(workspace.sandbox_dir),
                supervision_profile=policy.profile.value,
                human_intervention_count=intervention_count,
            )
            interaction_store.save(record)
            return {
                "session_id": record.session_id,
                "interaction_id": interaction_id,
                "question": question,
                "terminal": False,
                "resume_mode": record.resume_mode,
            }

        return ParallelEpisodeWorker(
            runner=OpenEtaEpisodeRunner(
                runtime=runtime,
                environment=environment,
                interaction_resolver=(
                    BackendGuidanceResolver(new_backend())
                    if policy.profile == SupervisionProfile.REVIEWED_AUTONOMY
                    else None
                ),
                initial_session_id=agent_session_id,
            ),
            close=environment.close,
            pause=pause,
            run_metadata={
                "workspace": workspace.to_dict(),
                "supervision": policy.to_dict(),
                "planner_prompt": dict(runtime.planner.prompt_metadata),
                "calibration_profile_id": staged_calibration_id,
                "calibration_profile_sha256": workspace.grasp_profile_sha256,
                "grasp_strategy_tree_sha256": workspace.grasp_strategy_tree_sha256,
            },
        )

    setattr(factory, "provider_metrics", provider_limiter.snapshot)
    return factory


def _new_batch_backend(
    provider,
    *,
    max_tokens: int | None = None,
    max_vision_images: int | None = None,
    provider_limiter: ProviderConcurrencyLimiter | None = None,
) -> PlannerBackend:
    config = OpenAICompatiblePlannerBackendConfig.from_provider_config(provider)
    if max_tokens is not None:
        config.max_tokens = max_tokens
    if max_vision_images is not None:
        config.max_vision_images = max(config.max_vision_images, max_vision_images)
    backend = OpenAICompatiblePlannerBackend(config)
    return provider_limiter.wrap(backend) if provider_limiter is not None else backend


def _write_output(path: str, payload: JsonDict) -> None:
    output = Path(path)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("--output must be a relative path inside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def resume_paused_episode(
    *,
    session_id: str,
    interaction_id: str,
    answer: str,
    model_override: str = "",
    sim_url: str = "",
    sam3_url: str = "",
    depth_prior_url: str = "",
    anygrasp_url: str = "",
    anyplace_url: str = "",
    graspgenx_url: str = "",
    contact_graspnet_url: str = "",
    molmopoint_url: str = "",
    supervision_profile: SupervisionProfile | str | None = None,
) -> JsonDict:
    """Record one answer, rebuild the same task environment, and retry it."""

    store = PausedEpisodeStore()
    record = store.load(session_id)
    if record.interaction_id != interaction_id:
        raise ValueError("interaction_id is stale or does not match the paused session")
    if not answer.strip():
        raise ValueError("human answer must be non-empty")
    intervention_count = record.human_intervention_count + 1
    history = [
        *record.interaction_history,
        {
            "interaction_id": interaction_id,
            "question": record.question,
            "answer": answer.strip(),
            "answered_at_s": time.time(),
        },
    ]
    spec = ParallelEpisodeSpec(
        episode_id=record.episode_id,
        task=record.task,
        env_id=record.env_id,
        seed=record.seed,
        max_turns=record.max_turns,
        max_tool_calls=record.max_tool_calls,
        timeout_s=record.timeout_s,
        max_total_tokens=record.max_total_tokens,
        metadata={
            "human_intervention_count": intervention_count,
            "agent_session_id": record.session_id,
            "workspace_root": record.workspace_root,
            "legacy_memory_root": record.memory_root,
            "legacy_artifact_root": record.artifact_root,
            "resume_existing": True,
        },
    )
    worker = build_mcp_episode_worker_factory(
        model_override=model_override,
        sim_url=sim_url,
        sam3_url=sam3_url,
        depth_prior_url=depth_prior_url,
        anygrasp_url=anygrasp_url,
        anyplace_url=anyplace_url,
        graspgenx_url=graspgenx_url,
        contact_graspnet_url=contact_graspnet_url,
        molmopoint_url=molmopoint_url,
        supervision_profile=(supervision_profile or record.supervision_profile),
    )(spec, record.batch_id)
    environment = worker.runner.environment
    if not isinstance(environment, SimulatorMcpEpisodeEnvironment):
        raise RuntimeError("paused session requires SimulatorMcpEpisodeEnvironment")
    runtime = worker.runner.runtime
    runtime.resume_session(record.session_id)
    resumed_catalog = discover_mcp_tool_catalog(
        environment.transport,
        endpoint_url=str(getattr(environment.transport, "url", sim_url) or sim_url),
        output_root=Path(record.artifact_root) / "responses",
    )
    if resumed_catalog:
        runtime.memory.save_fact(
            "simulator_mcp_tool_catalog",
            resumed_catalog,
            source="mcp.list_tools",
        )
    runtime.update_memory(
        {
            "type": "human_answer",
            "session_id": record.session_id,
            "interaction_id": interaction_id,
            "answer": answer.strip(),
            "human_intervention_count": intervention_count,
            "resume_mode": record.resume_mode,
        }
    )
    started = time.monotonic()
    result = None
    cleanup: JsonDict = {"ok": True, "skipped": True}
    error: JsonDict = {}
    interaction: JsonDict = {}
    try:
        result = worker.runner.run(
            task=record.task,
            max_turns=record.max_turns,
            max_tool_calls=record.max_tool_calls,
            timeout_s=record.timeout_s,
            max_total_tokens=record.max_total_tokens,
            initial_tool_call_count=record.tool_call_count,
            initial_total_tokens=record.total_tokens,
            initial_token_usage_sources=record.token_usage_sources,
            metadata={
                "source": "resume_paused_episode",
                "batch_id": record.batch_id,
                "episode_id": record.episode_id,
                "env_id": record.env_id,
                "seed": record.seed,
                "human_answer": answer.strip(),
                "interaction_id": interaction_id,
                "human_intervention_count": intervention_count,
                "restarted_after_human": True,
                "resume_mode": record.resume_mode,
            },
        )
        status = classify_episode_result(result, env_id=record.env_id)
        if status == "fail":
            error = episode_failure_error(result)
        if status == "need_human":
            next_interaction_id = new_interaction_id()
            question = question_from_episode(result)
            store.save(
                PausedEpisodeRecord(
                    batch_id=record.batch_id,
                    episode_id=record.episode_id,
                    session_id=record.session_id,
                    interaction_id=next_interaction_id,
                    question=question,
                    task=record.task,
                    env_id=record.env_id,
                    seed=record.seed,
                    max_turns=record.max_turns,
                    max_tool_calls=record.max_tool_calls,
                    timeout_s=record.timeout_s,
                    max_total_tokens=record.max_total_tokens,
                    tool_call_count=int(
                        (result.metadata.get("usage") or {}).get("tool_call_count") or 0
                    ),
                    total_tokens=int((result.metadata.get("usage") or {}).get("total_tokens") or 0),
                    token_usage_sources=dict(
                        (result.metadata.get("usage") or {}).get("token_usage_sources") or {}
                    ),
                    turn_index=int(result.metadata.get("turn_index") or 0),
                    memory_root=record.memory_root,
                    artifact_root=record.artifact_root,
                    workspace_root=record.workspace_root,
                    skills_root=record.skills_root,
                    sandbox_root=record.sandbox_root,
                    supervision_profile=record.supervision_profile,
                    human_intervention_count=intervention_count,
                    interaction_history=history,
                )
            )
            interaction = {
                "session_id": record.session_id,
                "interaction_id": next_interaction_id,
                "question": question,
                "terminal": False,
                "resume_mode": record.resume_mode,
            }
        else:
            store.delete(record.session_id)
    except Exception as exc:  # noqa: BLE001 - resume must return structured failure.
        status = "fail"
        error = {"type": type(exc).__name__, "message": str(exc)}
        store.delete(record.session_id)
    finally:
        try:
            cleanup = worker.close()
        except Exception as exc:  # noqa: BLE001 - preserve the episode outcome.
            cleanup = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    if status != "need_human" and cleanup.get("ok") is False:
        status = "fail"
        error.setdefault("type", "CleanupError")
        error.setdefault("message", str(cleanup.get("error") or "cleanup failed"))
    outcome = ParallelEpisodeOutcome(
        index=0,
        spec=spec,
        status=status,
        duration_s=time.monotonic() - started,
        episode=result,
        cleanup=cleanup,
        error=error,
        interaction=interaction,
        human_intervention_count=intervention_count,
        guidance_intervention_count=int(
            ((result.metadata.get("assistance") or {}).get("guidance_intervention_count")) or 0
        )
        if result is not None
        else 0,
    )
    return {
        "schema_version": "openeta.parallel_episode_resume.v1",
        "outcome": outcome.to_dict(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run independent OpenETA simulator episodes concurrently."
    )
    parser.add_argument("--manifest", default="", help="JSON batch manifest path.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_PARALLEL_EPISODES,
        help=(
            f"Maximum concurrent environments (default {DEFAULT_PARALLEL_EPISODES}, "
            f"hard limit {MAX_PARALLEL_EPISODES})."
        ),
    )
    parser.add_argument(
        "--provider-concurrency",
        type=int,
        default=DEFAULT_PROVIDER_CONCURRENCY,
    )
    parser.add_argument(
        "--provider-queue-timeout-s",
        type=float,
        default=DEFAULT_PROVIDER_QUEUE_TIMEOUT_S,
    )
    parser.add_argument("--model", default="", help="Override configured planner model.")
    parser.add_argument("--sim-url", default="", help="Override simulator MCP SSE URL.")
    parser.add_argument("--sam3-url", default="", help="Override SAM3 MCP SSE URL.")
    parser.add_argument(
        "--depth-prior-url",
        default="",
        help="Override UniDepth/depth-prior MCP SSE URL.",
    )
    parser.add_argument("--anygrasp-url", default="", help="Override AnyGrasp MCP SSE URL.")
    parser.add_argument("--anyplace-url", default="", help="Override AnyPlace MCP SSE URL.")
    parser.add_argument(
        "--graspgenx-url",
        default="",
        help="Override GraspGenX MCP SSE URL.",
    )
    parser.add_argument(
        "--contact-graspnet-url",
        default="",
        help="Override Contact-GraspNet MCP SSE URL.",
    )
    parser.add_argument(
        "--molmopoint-url",
        default="",
        help="Override MolmoPoint MCP SSE URL.",
    )
    parser.add_argument(
        "--approvement",
        choices=[profile.value for profile in SupervisionProfile],
        default="",
        help="Host-selected supervision profile for this batch.",
    )
    parser.add_argument("--batch-id", default="", help="Optional stable batch identifier.")
    parser.add_argument("--output", default="", help="Optional relative JSON result path.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the manifest without contacting model or MCP services.",
    )
    parser.add_argument("--resume-session", default="", help="Paused session id.")
    parser.add_argument("--interaction-id", default="", help="Current interaction id.")
    parser.add_argument("--answer", default="", help="Human answer for a paused session.")
    args = parser.parse_args(argv)

    try:
        if args.resume_session:
            if not args.interaction_id or not args.answer:
                raise ValueError("--resume-session requires --interaction-id and --answer")
            payload = resume_paused_episode(
                session_id=args.resume_session,
                interaction_id=args.interaction_id,
                answer=args.answer,
                model_override=args.model,
                sim_url=args.sim_url,
                sam3_url=args.sam3_url,
                depth_prior_url=args.depth_prior_url,
                anygrasp_url=args.anygrasp_url,
                anyplace_url=args.anyplace_url,
                graspgenx_url=args.graspgenx_url,
                contact_graspnet_url=args.contact_graspnet_url,
                molmopoint_url=args.molmopoint_url,
                supervision_profile=args.approvement or None,
            )
            if args.output:
                _write_output(args.output, payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1 if payload["outcome"]["status"] == "fail" else 0
        if not args.manifest:
            raise ValueError("--manifest is required unless --resume-session is used")
        specs = load_parallel_episode_manifest(args.manifest)
        if not 1 <= args.concurrency <= MAX_PARALLEL_EPISODES:
            raise ValueError(f"concurrency must be between 1 and {MAX_PARALLEL_EPISODES}")
        if args.validate_only:
            payload: JsonDict = {
                "valid": True,
                "episode_count": len(specs),
                "concurrency": min(args.concurrency, len(specs)),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        worker_factory = build_mcp_episode_worker_factory(
            model_override=args.model,
            sim_url=args.sim_url,
            sam3_url=args.sam3_url,
            depth_prior_url=args.depth_prior_url,
            anygrasp_url=args.anygrasp_url,
            anyplace_url=args.anyplace_url,
            graspgenx_url=args.graspgenx_url,
            contact_graspnet_url=args.contact_graspnet_url,
            molmopoint_url=args.molmopoint_url,
            supervision_profile=(args.approvement or SupervisionProfile.STANDARD.value),
            provider_concurrency=args.provider_concurrency,
            provider_queue_timeout_s=args.provider_queue_timeout_s,
        )
        harness = ParallelEpisodeHarness(
            worker_factory,
            concurrency=args.concurrency,
        )
        result = harness.run(specs, batch_id=args.batch_id or None)
        payload = result.to_dict()
        provider_metrics = getattr(worker_factory, "provider_metrics", None)
        if callable(provider_metrics):
            payload["provider_concurrency"] = provider_metrics()
        if args.output:
            _write_output(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if result.fail_count else 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
