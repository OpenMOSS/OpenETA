"""Non-interactive parallel simulator evaluation entry point."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from adapter.protocol import JsonDict
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
)
from agent.backends.provider_config import load_planner_provider_config
from agent.runtime.checkers import CheckerSubagentConfig
from agent.runtime.episode import OpenEtaEpisodeRunner
from agent.runtime.interactions import (
    PausedEpisodeRecord,
    PausedEpisodeStore,
    new_interaction_id,
    question_from_episode,
)
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
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
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import PlannerContextConfig, ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.handlers import (
    bind_dummy_tool_handlers,
    build_anygrasp_handler,
    build_sam3_handler,
    build_sse_anygrasp_mcp_grasper,
    build_sse_sam3_mcp_segmenter,
)
from agent.tools.mcp_registry import load_mcp_server_url
from agent.tools.registry import build_default_tool_registry
from agent.tools.sim_mcp import (
    SimulatorMcpEpisodeConfig,
    SimulatorMcpEpisodeEnvironment,
    SimulatorMcpToolProxyConfig,
    SseSimulatorMcpTransport,
    bind_simulator_mcp_tool_handlers,
)


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


def build_mcp_episode_worker_factory(
    *,
    model_override: str = "",
    sim_url: str = "",
    sam3_url: str = "",
    anygrasp_url: str = "",
):
    """Build isolated model/runtime/MCP workers for a parallel batch."""

    provider = load_planner_provider_config()
    if model_override:
        provider.model = model_override
    missing = provider.missing_fields()
    if missing:
        raise ValueError(f"planner provider config is missing: {', '.join(missing)}")
    resolved_sim_url = sim_url or load_mcp_server_url("openeta-sim", aliases=("sim",))
    resolved_sam3_url = sam3_url or load_mcp_server_url(
        "openeta-sam3", aliases=("sam3",)
    )
    resolved_anygrasp_url = anygrasp_url or load_mcp_server_url(
        "openeta-anygrasp", aliases=("anygrasp",)
    )
    if not resolved_sim_url:
        raise ValueError("simulator MCP URL is required")

    def factory(spec: ParallelEpisodeSpec, batch_id: str) -> ParallelEpisodeWorker:
        episode_key = _safe_path_component(spec.episode_id)
        artifact_root = Path("tmp") / "parallel_eval" / batch_id / episode_key
        memory_root = Path(".openeta_memory") / "batches" / batch_id / episode_key
        transport = SseSimulatorMcpTransport(resolved_sim_url)
        proxy_config = SimulatorMcpToolProxyConfig(
            timeout_s=max(120.0, provider.timeout_s),
            image_output_root=artifact_root / "images",
            text_output_root=artifact_root / "text",
            response_output_root=artifact_root / "responses",
        )
        environment = SimulatorMcpEpisodeEnvironment(
            transport=transport,
            config=SimulatorMcpEpisodeConfig(
                env_id=spec.env_id,
                seed=spec.seed,
                timeout_s=max(120.0, provider.timeout_s),
                image_output_root=artifact_root / "images",
            ),
            tool_proxy_config=proxy_config,
        )
        tools = bind_dummy_tool_handlers(
            build_default_tool_registry(),
            include_dummy_safety=False,
        )
        for placeholder_name in (
            "scene_detector",
            "sam3",
            "anygrasp",
            "hand_pose_database",
            "ik_preview_check",
            "obstacle_avoidance",
            "lower_body_control_policy",
        ):
            tools.unbind_handler(placeholder_name)
        bind_simulator_mcp_tool_handlers(
            tools,
            transport=transport,
            config=proxy_config,
            replace=True,
        )
        if resolved_sam3_url:
            tools.bind_handler(
                "sam3",
                build_sam3_handler(
                    build_sse_sam3_mcp_segmenter(url=resolved_sam3_url),
                    output_root=artifact_root / "sam3_images",
                    result_output_root=artifact_root / "sam3_results",
                ),
                replace=True,
            )
        if resolved_anygrasp_url:
            tools.bind_handler(
                "anygrasp",
                build_anygrasp_handler(
                    build_sse_anygrasp_mcp_grasper(url=resolved_anygrasp_url),
                    output_root=artifact_root / "anygrasp_results",
                ),
                replace=True,
            )
        backend_config = OpenAICompatiblePlannerBackendConfig.from_provider_config(provider)
        planner = ToolCallingPlanner(
            OpenAICompatiblePlannerBackend(backend_config),
            context_config=PlannerContextConfig(
                context_window_tokens=provider.context_window_tokens,
                token_estimator_model=provider.model,
            ),
        )
        runtime = OpenEtaAgentRuntime(
            planner=planner,
            tools=tools,
            memory=AgentMemory(store=JsonMemoryStore(root=memory_root)),
            pipeline=ActionPipeline(
                checker_subagents=CheckerSubagentConfig(
                    post_failure_checks=tuple(tool_spec.name for tool_spec in tools.list())
                )
            ),
        )
        interaction_store = PausedEpisodeStore()

        def pause(result) -> JsonDict:
            if not result.session_id:
                raise RuntimeError("need_human episode has no session_id")
            interaction_id = new_interaction_id()
            question = question_from_episode(result)
            intervention_count = int(
                spec.metadata.get("human_intervention_count") or 0
            )
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
                total_tokens=int(
                    (result.metadata.get("usage") or {}).get("total_tokens") or 0
                ),
                token_usage_sources=dict(
                    (result.metadata.get("usage") or {}).get("token_usage_sources")
                    or {}
                ),
                turn_index=int(result.metadata.get("turn_index") or 0),
                memory_root=str(memory_root),
                artifact_root=str(artifact_root),
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
            runner=OpenEtaEpisodeRunner(runtime=runtime, environment=environment),
            close=environment.close,
            pause=pause,
        )

    return factory


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
    anygrasp_url: str = "",
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
        metadata={"human_intervention_count": intervention_count},
    )
    worker = build_mcp_episode_worker_factory(
        model_override=model_override,
        sim_url=sim_url,
        sam3_url=sam3_url,
        anygrasp_url=anygrasp_url,
    )(spec, record.batch_id)
    environment = worker.runner.environment
    if not isinstance(environment, SimulatorMcpEpisodeEnvironment):
        raise RuntimeError("paused session requires SimulatorMcpEpisodeEnvironment")
    runtime = worker.runner.runtime
    runtime.resume_session(record.session_id)
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
        status = classify_episode_result(result)
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
                    total_tokens=int(
                        (result.metadata.get("usage") or {}).get("total_tokens") or 0
                    ),
                    token_usage_sources=dict(
                        (result.metadata.get("usage") or {}).get(
                            "token_usage_sources"
                        )
                        or {}
                    ),
                    turn_index=int(result.metadata.get("turn_index") or 0),
                    memory_root=record.memory_root,
                    artifact_root=record.artifact_root,
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
    parser.add_argument("--model", default="", help="Override configured planner model.")
    parser.add_argument("--sim-url", default="", help="Override simulator MCP SSE URL.")
    parser.add_argument("--sam3-url", default="", help="Override SAM3 MCP SSE URL.")
    parser.add_argument("--anygrasp-url", default="", help="Override AnyGrasp MCP SSE URL.")
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
                raise ValueError(
                    "--resume-session requires --interaction-id and --answer"
                )
            payload = resume_paused_episode(
                session_id=args.resume_session,
                interaction_id=args.interaction_id,
                answer=args.answer,
                model_override=args.model,
                sim_url=args.sim_url,
                sam3_url=args.sam3_url,
                anygrasp_url=args.anygrasp_url,
            )
            if args.output:
                _write_output(args.output, payload)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 1 if payload["outcome"]["status"] == "fail" else 0
        if not args.manifest:
            raise ValueError("--manifest is required unless --resume-session is used")
        specs = load_parallel_episode_manifest(args.manifest)
        if not 1 <= args.concurrency <= MAX_PARALLEL_EPISODES:
            raise ValueError(
                f"concurrency must be between 1 and {MAX_PARALLEL_EPISODES}"
            )
        if args.validate_only:
            payload: JsonDict = {
                "valid": True,
                "episode_count": len(specs),
                "concurrency": min(args.concurrency, len(specs)),
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        harness = ParallelEpisodeHarness(
            build_mcp_episode_worker_factory(
                model_override=args.model,
                sim_url=args.sim_url,
                sam3_url=args.sam3_url,
                anygrasp_url=args.anygrasp_url,
            ),
            concurrency=args.concurrency,
        )
        result = harness.run(specs, batch_id=args.batch_id or None)
        payload = result.to_dict()
        if args.output:
            _write_output(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1 if result.fail_count else 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
