"""Non-interactive command dispatcher for parallel skill experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
)
from agent.backends.provider_config import load_planner_provider_config
from agent.cli.batch_eval import (
    DEFAULT_PROVIDER_CONCURRENCY,
    DEFAULT_PROVIDER_QUEUE_TIMEOUT_S,
    build_mcp_episode_worker_factory,
    load_parallel_episode_manifest,
)
from agent.runtime.experiments import (
    DEFAULT_EXPERIMENT_ROOT,
    ExperimentWorkspace,
    build_proposed_grasp_strategy_tree,
    build_proposed_skill_tree,
    build_proposed_task_playbook_tree,
    collect_grasp_strategy_candidates,
    collect_skill_candidates,
    collect_task_playbook_candidates,
    compact_strategy_rollout_summary,
    objective_batch_metrics,
    select_supported_grasp_strategy_candidate,
    select_supported_candidates,
    select_supported_task_playbook_candidates,
    skill_tree_hash,
    strategy_validation_has_no_regression,
    task_playbook_tree_sha256,
    validation_has_no_regression,
    write_grasp_strategy_evidence,
)
from agent.runtime.artifact_paths import safe_artifact_component
from agent.runtime.grasp_strategy_lifecycle import (
    BackendGraspStrategyAuthor,
    BackendGraspStrategyReviewer,
    GraspStrategyGateError,
    GraspStrategyLifecycleConfig,
    GraspStrategyLifecycleManager,
)
from agent.runtime.parallel import (
    DEFAULT_PARALLEL_EPISODES,
    MAX_PARALLEL_EPISODES,
    ParallelEpisodeHarness,
    ParallelEpisodeSpec,
)
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.skill_authoring import (
    BackendSkillChangeReviewer,
    SkillAuthoringRequest,
)
from agent.runtime.skills import BUILTIN_SKILL_DIR, load_skill_markdown
from agent.runtime.task_playbooks import DEFAULT_TASK_PLAYBOOK_ROOT
from agent.runtime.supervision import SupervisionProfile
from agent.tools.mcp_registry import load_mcp_server_url
from agent.tools.grasp_geometry import DEFAULT_GRASP_PROFILE
from agent.tools.grasp_strategies import (
    DEFAULT_GRASP_STRATEGY_ROOT,
    grasp_strategy_tree_sha256,
    load_grasp_strategies,
)
from agent.tools.registry import build_default_tool_registry
from agent.tools.sim_mcp import SseSimulatorMcpTransport


_UNATTENDED_PROFILE = SupervisionProfile.REVIEWED_AUTONOMY.value
_BATCH_UNBOUND_TOOLS = {
    "scene_detector",
    "hand_pose_database",
    "ik_preview_check",
    "obstacle_avoidance",
    "lower_body_control_policy",
    "anydexgrasp",
    "slam",
}
_REQUIRED_SIM_MCP_TOOLS = {
    "create_env",
    "reset_env",
    "render_env",
    "move_to",
    "close_env",
    "gripper_open",
    "gripper_close",
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            payload = preflight(args)
        elif args.command == "run":
            payload = run_generation(args)
        elif args.command == "iterate":
            payload = iterate_generations(args)
        elif args.command == "inspect":
            payload = inspect_experiment(args)
        else:  # pragma: no cover - argparse enforces the command set.
            raise ValueError(f"unsupported command: {args.command}")
        printable = (
            payload
            if getattr(args, "print_full_result", False)
            else _compact_command_output(args.command, payload)
        )
        print(json.dumps(printable, ensure_ascii=False, indent=2))
        return 1 if payload.get("ok") is False else 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


def preflight(args: argparse.Namespace) -> JsonDict:
    """Validate local inputs and remote simulator tool discovery without creating an env."""

    specs = _filter_episode_specs(
        load_parallel_episode_manifest(args.manifest),
        getattr(args, "episode_id", []),
    )
    _validate_concurrency(args.concurrency)
    _validate_provider_limits(args)
    _validate_strategy_limits(args)
    provider = load_planner_provider_config()
    if args.model:
        provider.model = args.model
    errors = (
        [f"planner provider config is missing: {', '.join(provider.missing_fields())}"]
        if provider.missing_fields()
        else []
    )
    (
        sim_url,
        sam3_url,
        anygrasp_url,
        contact_graspnet_url,
        graspgenx_url,
        anyplace_url,
        molmopoint_url,
    ) = _resolved_mcp_urls(args)
    if not sim_url:
        errors.append("simulator MCP URL is required")
    if args.require_perception and not sam3_url:
        errors.append("SAM3 MCP URL is required for embodied grasp experiments")
    if args.require_perception and not any(
        (anygrasp_url, contact_graspnet_url, graspgenx_url)
    ):
        errors.append(
            "At least one grasp estimator MCP URL is required for embodied "
            "grasp experiments"
        )
    if args.require_perception and not anyplace_url:
        errors.append("AnyPlace MCP URL is required for embodied pick-and-place experiments")
    planner = ToolCallingPlanner()
    catalog_summary: JsonDict = {"checked": False, "url": sim_url}
    if sim_url and not args.skip_mcp_check:
        if args.mcp_timeout_s <= 0:
            errors.append("mcp timeout must be positive")
        else:
            try:
                catalog = SseSimulatorMcpTransport(sim_url).list_tools(timeout_s=args.mcp_timeout_s)
                remote_names = {
                    str(tool.get("name") or "")
                    for tool in catalog.get("tools", [])
                    if isinstance(tool, dict)
                }
                missing_tools = sorted(_REQUIRED_SIM_MCP_TOOLS - remote_names)
                catalog_summary = {
                    "checked": True,
                    "url": sim_url,
                    "tool_count": len(remote_names),
                    "missing_required_tools": missing_tools,
                }
                if missing_tools:
                    errors.append(
                        "simulator MCP is missing required tools: " + ", ".join(missing_tools)
                    )
            except Exception as exc:  # noqa: BLE001 - preflight reports all checks.
                catalog_summary = {
                    "checked": True,
                    "url": sim_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                errors.append(f"simulator MCP list_tools failed: {exc}")
    baseline = Path(args.baseline_skills)
    if not any(baseline.glob("*.md")):
        errors.append(f"skill baseline contains no markdown files: {baseline}")
        baseline_hash = ""
    else:
        baseline_hash = skill_tree_hash(baseline)
    strategy_baseline = Path(
        getattr(
            args,
            "baseline_grasp_strategies",
            str(DEFAULT_GRASP_STRATEGY_ROOT),
        )
    )
    try:
        strategy_hash = grasp_strategy_tree_sha256(strategy_baseline)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        strategy_hash = ""
        errors.append(f"invalid grasp strategy baseline: {exc}")
    calibration_profile = Path(
        getattr(args, "calibration_profile", str(DEFAULT_GRASP_PROFILE))
    )
    if not calibration_profile.is_file():
        errors.append(f"calibration profile does not exist: {calibration_profile}")
    return {
        "schema_version": "openeta.command_preflight.v1",
        "ok": not errors,
        "errors": errors,
        "episode_count": len(specs),
        "concurrency": min(args.concurrency, len(specs)),
        "provider": {
            "provider": provider.provider,
            "model": provider.model,
            "concurrency": getattr(
                args, "provider_concurrency", DEFAULT_PROVIDER_CONCURRENCY
            ),
            "queue_timeout_s": getattr(
                args,
                "provider_queue_timeout_s",
                DEFAULT_PROVIDER_QUEUE_TIMEOUT_S,
            ),
        },
        "planner_prompt": dict(planner.prompt_metadata),
        "baseline_skills": str(baseline),
        "baseline_hash": baseline_hash,
        "baseline_grasp_strategies": str(strategy_baseline),
        "baseline_grasp_strategy_hash": strategy_hash,
        "calibration_profile": str(calibration_profile),
        "mcp": {
            "simulator": catalog_summary,
            "sam3": {"configured": bool(sam3_url), "url": sam3_url},
            "grasp_pose_estimate": {
                "configured": bool(
                    anygrasp_url or contact_graspnet_url or graspgenx_url
                ),
                "backends": {
                    "anygrasp": {
                        "configured": bool(anygrasp_url),
                        "url": anygrasp_url,
                    },
                    "contact_graspnet": {
                        "configured": bool(contact_graspnet_url),
                        "url": contact_graspnet_url,
                    },
                    "graspgenx": {
                        "configured": bool(graspgenx_url),
                        "url": graspgenx_url,
                    },
                },
            },
            "anyplace": {"configured": bool(anyplace_url), "url": anyplace_url},
            "molmopoint": {
                "configured": bool(molmopoint_url),
                "url": molmopoint_url,
            },
        },
    }


def run_generation(args: argparse.Namespace) -> JsonDict:
    specs = _filter_episode_specs(
        load_parallel_episode_manifest(args.manifest),
        args.episode_id,
    )
    _validate_concurrency(args.concurrency)
    _validate_provider_limits(args)
    _validate_strategy_limits(args)
    experiment_id = args.experiment_id or f"experiment-{uuid4().hex[:12]}"
    experiment = _create_experiment(args, experiment_id)
    baseline = experiment.initialize_generation(
        0,
        source_skills=args.baseline_skills,
        source_grasp_strategies=args.baseline_grasp_strategies,
        source_task_playbooks=getattr(
            args, "baseline_task_playbooks", str(DEFAULT_TASK_PLAYBOOK_ROOT)
        ),
    )
    prepared = experiment.prepare_specs(
        specs,
        generation=0,
        phase="train",
        skills_root=baseline,
        grasp_strategies_root=experiment.grasp_strategy_baseline(0),
        task_playbooks_root=experiment.task_playbook_baseline(0),
        on_need_human=args.on_need_human,
    )
    batch = _run_batch(args, prepared, batch_id=f"{experiment.experiment_id}-g000-train")
    result_path = experiment.write_phase_result(0, "train", batch)
    candidates = collect_skill_candidates(
        generation_dir=experiment.generation_dir(0),
        phase="train",
        batch_payload=batch,
    )
    playbooks = collect_task_playbook_candidates(
        generation_dir=experiment.generation_dir(0),
        phase="train",
        batch_payload=batch,
    )
    return {
        "schema_version": "openeta.command_run.v1",
        "ok": int(batch.get("fail_count") or 0) == 0,
        "experiment_id": experiment.experiment_id,
        "generation": 0,
        "result_path": str(result_path),
        "metrics": objective_batch_metrics(batch),
        "candidate_count": candidates["candidate_count"],
        "task_playbook_candidate_count": playbooks["candidate_count"],
        "batch": batch,
    }


def iterate_generations(args: argparse.Namespace) -> JsonDict:
    _validate_concurrency(args.concurrency)
    _validate_provider_limits(args)
    _validate_strategy_limits(args)
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    if args.approvement != _UNATTENDED_PROFILE:
        raise ValueError("iterate requires --approvement reviewed_autonomy")
    if args.on_need_human != "fail":
        raise ValueError("iterate requires --on-need-human fail")
    train_specs = _filter_episode_specs(
        load_parallel_episode_manifest(args.train_manifest),
        args.episode_id,
    )
    validation_specs = _filter_episode_specs(
        load_parallel_episode_manifest(args.validation_manifest),
        args.episode_id,
    )
    experiment_id = args.experiment_id or f"experiment-{uuid4().hex[:12]}"
    experiment = _create_experiment(args, experiment_id)
    source_skills = Path(args.baseline_skills)
    source_strategies = Path(args.baseline_grasp_strategies)
    source_task_playbooks = Path(
        getattr(args, "baseline_task_playbooks", str(DEFAULT_TASK_PLAYBOOK_ROOT))
    )
    rounds: list[JsonDict] = []
    for generation in range(args.rounds):
        baseline = experiment.initialize_generation(
            generation,
            source_skills=source_skills,
            source_grasp_strategies=source_strategies,
            source_task_playbooks=source_task_playbooks,
        )
        strategy_baseline = experiment.grasp_strategy_baseline(generation)
        task_playbook_baseline = experiment.task_playbook_baseline(generation)
        train_prepared = experiment.prepare_specs(
            train_specs,
            generation=generation,
            phase="train",
            skills_root=baseline,
            grasp_strategies_root=strategy_baseline,
            task_playbooks_root=task_playbook_baseline,
            on_need_human=args.on_need_human,
        )
        train = _run_batch(
            args,
            train_prepared,
            batch_id=f"{experiment.experiment_id}-g{generation:03d}-train",
        )
        experiment.write_phase_result(generation, "train", train)
        candidate_manifest = collect_skill_candidates(
            generation_dir=experiment.generation_dir(generation),
            phase="train",
            batch_payload=train,
        )
        selected = select_supported_candidates(candidate_manifest)
        reviews = _review_candidates(args, selected, baseline)
        approved = [
            candidate
            for candidate, review in zip(selected, reviews, strict=True)
            if review.get("decision") == "approve"
        ]
        strategy_manifest = collect_grasp_strategy_candidates(
            generation_dir=experiment.generation_dir(generation),
            phase="train",
            batch_payload=train,
        )
        task_playbook_manifest = collect_task_playbook_candidates(
            generation_dir=experiment.generation_dir(generation),
            phase="train",
            batch_payload=train,
        )
        selected_task_playbooks = select_supported_task_playbook_candidates(
            task_playbook_manifest
        )
        accepted_task_playbooks = task_playbook_baseline
        task_playbook_promoted = bool(selected_task_playbooks)
        if selected_task_playbooks:
            accepted_task_playbooks = build_proposed_task_playbook_tree(
                baseline_task_playbooks=task_playbook_baseline,
                destination=experiment.generation_dir(generation)
                / "proposed_task_playbooks",
                candidates=selected_task_playbooks,
            )
        accepted_strategies, strategy_iteration = _iterate_grasp_strategy(
            args,
            experiment=experiment,
            generation=generation,
            train=train,
            train_specs=train_specs,
            validation_specs=validation_specs,
            skills_root=baseline,
            baseline_strategies=strategy_baseline,
            session_candidate=select_supported_grasp_strategy_candidate(
                strategy_manifest
            ),
        )
        round_payload: JsonDict = {
            "generation": generation,
            "baseline_hash": skill_tree_hash(baseline),
            "train_metrics": objective_batch_metrics(train),
            "candidate_count": candidate_manifest["candidate_count"],
            "selected_candidate_count": len(selected),
            "unselected_candidate_count": candidate_manifest["candidate_count"]
            - len(selected),
            "reviews": reviews,
            "reviewer_error_count": sum("error_type" in review for review in reviews),
            "approved_candidate_count": len(approved),
            "grasp_strategy": strategy_iteration,
            "task_playbooks": {
                "candidate_count": task_playbook_manifest["candidate_count"],
                "selected_candidate_count": len(selected_task_playbooks),
                "accepted_for_next_generation": task_playbook_promoted,
                "proposed_hash": (
                    task_playbook_tree_sha256(accepted_task_playbooks)
                    if task_playbook_promoted
                    else None
                ),
            },
            "promoted": False,
        }
        accepted_skills = baseline
        skill_promoted = False
        if approved:
            proposed = build_proposed_skill_tree(
                baseline_skills=baseline,
                destination=experiment.generation_dir(generation)
                / "proposed_skills",
                approved_candidates=approved,
            )
            baseline_validation = _run_validation(
                args,
                experiment,
                generation,
                validation_specs,
                skills_root=baseline,
                grasp_strategies_root=accepted_strategies,
                phase="validation-baseline",
            )
            candidate_validation = _run_validation(
                args,
                experiment,
                generation,
                validation_specs,
                skills_root=proposed,
                grasp_strategies_root=accepted_strategies,
                phase="validation-candidate",
            )
            comparison = validation_has_no_regression(
                baseline_validation,
                candidate_validation,
            )
            round_payload["validation"] = comparison
            round_payload["proposed_hash"] = skill_tree_hash(proposed)
            if comparison["passed"]:
                accepted_skills = proposed
                skill_promoted = True
        strategy_promoted = bool(strategy_iteration.get("accepted_for_next_generation"))
        round_payload["promoted"] = (
            skill_promoted or strategy_promoted or task_playbook_promoted
        )
        if round_payload["promoted"]:
            source_skills = accepted_skills
            source_strategies = accepted_strategies
            source_task_playbooks = accepted_task_playbooks
            next_baseline = experiment.initialize_generation(
                generation + 1,
                source_skills=source_skills,
                source_grasp_strategies=source_strategies,
                source_task_playbooks=source_task_playbooks,
            )
            round_payload["next_baseline"] = str(next_baseline)
        elif approved or strategy_iteration.get("proposed"):
            round_payload["stop_reason"] = "validation_regression"
        else:
            round_payload["stop_reason"] = "no_approved_candidates"
        rounds.append(round_payload)
        _write_generation_result(experiment, generation, round_payload)
        if not round_payload["promoted"]:
            break
    return {
        "schema_version": "openeta.command_iterate.v1",
        "ok": all(
            (not item.get("validation") or item["validation"]["passed"])
            and not item.get("reviewer_error_count")
            and not (
                isinstance(item.get("grasp_strategy"), dict)
                and item["grasp_strategy"].get("error_count")
            )
            for item in rounds
        ),
        "experiment_id": experiment.experiment_id,
        "requested_rounds": args.rounds,
        "completed_rounds": len(rounds),
        "rounds": rounds,
    }


def inspect_experiment(args: argparse.Namespace) -> JsonDict:
    experiment_id = args.experiment_id.strip()
    if not experiment_id:
        raise ValueError("inspect requires --experiment-id")
    safe = safe_artifact_component(experiment_id, fallback="experiment")
    experiment_root = _validated_experiment_root(args.experiment_root)
    root = experiment_root / safe
    if not root.is_dir():
        raise ValueError(f"experiment does not exist: {safe}")
    generation_rows: list[JsonDict] = []
    for generation in sorted((root / "generations").glob("[0-9][0-9][0-9]")):
        row: JsonDict = {"generation": int(generation.name), "path": str(generation)}
        for name in ("generation.json", "result.json", "candidates/manifest.json"):
            path = generation / name
            if path.is_file():
                row[name] = json.loads(path.read_text(encoding="utf-8"))
        generation_rows.append(row)
    return {
        "schema_version": "openeta.command_inspect.v1",
        "ok": root.is_dir(),
        "experiment_id": safe,
        "root": str(root),
        "generations": generation_rows,
    }


def _run_validation(
    args: argparse.Namespace,
    experiment: ExperimentWorkspace,
    generation: int,
    specs: list[ParallelEpisodeSpec],
    *,
    skills_root: Path,
    grasp_strategies_root: Path,
    phase: str,
) -> JsonDict:
    prepared = experiment.prepare_specs(
        specs,
        generation=generation,
        phase=phase,
        skills_root=skills_root,
        grasp_strategies_root=grasp_strategies_root,
        on_need_human=args.on_need_human,
    )
    payload = _run_batch(
        args,
        prepared,
        batch_id=f"{experiment.experiment_id}-g{generation:03d}-{phase}",
    )
    experiment.write_phase_result(generation, phase, payload)
    return payload


def _iterate_grasp_strategy(
    args: argparse.Namespace,
    *,
    experiment: ExperimentWorkspace,
    generation: int,
    train: JsonDict,
    train_specs: list[ParallelEpisodeSpec],
    validation_specs: list[ParallelEpisodeSpec],
    skills_root: Path,
    baseline_strategies: Path,
    session_candidate: JsonDict | None,
) -> tuple[Path, JsonDict]:
    result: JsonDict = {
        "proposed": False,
        "accepted_for_next_generation": False,
        "shared_candidate_published": False,
        "shared_validated_published": False,
    }
    calibration_path = Path(args.calibration_profile)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict):
        raise ValueError("calibration profile must contain one JSON object")
    current = load_grasp_strategies(baseline_strategies)
    candidate_strategy: JsonDict | None = None
    author_details: JsonDict = {}
    if session_candidate is not None:
        candidate_strategy = json.loads(
            Path(str(session_candidate["candidate_path"])).read_text(
                encoding="utf-8"
            )
        )
        author_details = {
            "source": "reviewed_session_candidate",
            "support_count": session_candidate.get("support_count"),
            "supporting_episode_ids": session_candidate.get(
                "supporting_episode_ids"
            ),
        }
    else:
        try:
            author = BackendGraspStrategyAuthor(
                _new_experiment_backend(args, max_tokens=4096)
            )
            authored = author.author(
                current_strategies=current,
                calibration_profile=calibration,
                rollout_summary=compact_strategy_rollout_summary(train),
            )
            author_details = {
                "source": "isolated_strategy_author",
                "decision": authored.decision,
                "reason": authored.reason,
                "details": authored.details,
            }
            candidate_strategy = authored.strategy
        except Exception as exc:  # noqa: BLE001 - author failure stops this lane.
            result["author"] = {
                "decision": "error",
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
            result["error_count"] = 1
            return baseline_strategies, result
    result["author"] = author_details
    if candidate_strategy is None:
        result["stop_reason"] = "author_no_change"
        return baseline_strategies, result

    lifecycle = GraspStrategyLifecycleManager(
        config=GraspStrategyLifecycleConfig(
            root=experiment.generation_dir(generation)
            / "grasp_strategy_lifecycle",
            session_strategy_root=experiment.generation_dir(generation)
            / "authored_grasp_strategy",
            evidence_roots=(experiment.root,),
            publication_mode="independent_reviewer",
            min_canary_attempts=args.strategy_min_canary_attempts,
            min_held_out_attempts=args.strategy_min_held_out_attempts,
            min_held_out_success_rate=args.strategy_min_held_out_success_rate,
            min_held_out_task_count=args.strategy_min_held_out_tasks,
        ),
        reviewer=BackendGraspStrategyReviewer(_new_experiment_backend(args)),
    )
    try:
        proposal = lifecycle.create_proposal(
            session_id=f"strategy-g{generation:03d}-{uuid4().hex[:12]}",
            strategy=candidate_strategy,
            calibration_profile=calibration,
            rationale=str(
                author_details.get("reason")
                or "Reviewed session candidate supported by objective reward."
            ),
            rollout_summary=compact_strategy_rollout_summary(train),
        )
    except Exception as exc:  # noqa: BLE001
        result["proposal"] = {
            "status": "error",
            "reason": str(exc),
            "error_type": type(exc).__name__,
        }
        result["error_count"] = 1
        return baseline_strategies, result
    result["proposal"] = {
        key: proposal.get(key)
        for key in (
            "proposal_id",
            "status",
            "strategy_id",
            "strategy_sha256",
            "review",
            "proposal_path",
            "session_strategy_path",
        )
    }
    result["proposed"] = True
    if proposal.get("status") != "reviewed":
        result["stop_reason"] = "proposal_review_blocked"
        return baseline_strategies, result

    proposed_tree = build_proposed_grasp_strategy_tree(
        baseline_strategies=baseline_strategies,
        destination=experiment.generation_dir(generation)
        / "proposed_grasp_strategies",
        candidate_path=str(proposal["session_strategy_path"]),
    )
    proposed_tree_hash = grasp_strategy_tree_sha256(proposed_tree)
    result["proposed_tree_hash"] = proposed_tree_hash
    canary = _run_validation(
        args,
        experiment,
        generation,
        train_specs,
        skills_root=skills_root,
        grasp_strategies_root=proposed_tree,
        phase="strategy-canary-candidate",
    )
    canary_comparison = strategy_validation_has_no_regression(train, canary)
    result["canary"] = canary_comparison
    canary_evidence = write_grasp_strategy_evidence(
        experiment.generation_dir(generation)
        / "grasp_strategy_lifecycle"
        / "canary-evidence.json",
        split="canary",
        strategy_sha256=str(proposal["strategy_sha256"]),
        calibration_profile_sha256=str(
            proposal["calibration_profile_sha256"]
        ),
        baseline=train,
        candidate=canary,
        expected_strategy_tree_sha256=proposed_tree_hash,
    )
    if not canary_comparison["passed"]:
        result["stop_reason"] = "canary_regression"
        return baseline_strategies, result

    if args.publish_grasp_strategies and len(train_specs) >= int(
        args.strategy_min_canary_attempts
    ):
        try:
            receipt = lifecycle.promote(
                proposal=proposal,
                target_status="candidate",
                evidence_references=[
                    {"path": str(canary_evidence), "split": "canary"}
                ],
            )
            result["candidate_publication"] = receipt
            result["shared_candidate_published"] = True
        except GraspStrategyGateError as exc:
            result["candidate_publication"] = {
                "status": "pending_evidence",
                "reason": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            result["candidate_publication"] = {
                "status": "blocked",
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
            result["error_count"] = int(result.get("error_count") or 0) + 1

    baseline_validation = _run_validation(
        args,
        experiment,
        generation,
        validation_specs,
        skills_root=skills_root,
        grasp_strategies_root=baseline_strategies,
        phase="strategy-heldout-baseline",
    )
    candidate_validation = _run_validation(
        args,
        experiment,
        generation,
        validation_specs,
        skills_root=skills_root,
        grasp_strategies_root=proposed_tree,
        phase="strategy-heldout-candidate",
    )
    held_out_comparison = strategy_validation_has_no_regression(
        baseline_validation,
        candidate_validation,
    )
    result["held_out"] = held_out_comparison
    held_out_evidence = write_grasp_strategy_evidence(
        experiment.generation_dir(generation)
        / "grasp_strategy_lifecycle"
        / "held-out-evidence.json",
        split="held_out",
        strategy_sha256=str(proposal["strategy_sha256"]),
        calibration_profile_sha256=str(
            proposal["calibration_profile_sha256"]
        ),
        baseline=baseline_validation,
        candidate=candidate_validation,
        expected_strategy_tree_sha256=proposed_tree_hash,
    )
    if not held_out_comparison["passed"]:
        result["stop_reason"] = "held_out_regression"
        return baseline_strategies, result
    result["accepted_for_next_generation"] = True

    held_out_task_count = len(
        {
            spec.task.strip() or spec.env_id
            for spec in validation_specs
        }
    )
    if (
        args.publish_grasp_strategies
        and result["shared_candidate_published"]
        and len(validation_specs) >= int(args.strategy_min_held_out_attempts)
        and held_out_task_count >= int(args.strategy_min_held_out_tasks)
    ):
        try:
            receipt = lifecycle.promote(
                proposal=proposal,
                target_status="validated",
                evidence_references=[
                    {"path": str(canary_evidence), "split": "canary"},
                    {"path": str(held_out_evidence), "split": "held_out"},
                ],
            )
            result["validated_publication"] = receipt
            result["shared_validated_published"] = True
        except GraspStrategyGateError as exc:
            result["validated_publication"] = {
                "status": "pending_evidence",
                "reason": str(exc),
            }
        except Exception as exc:  # noqa: BLE001
            result["validated_publication"] = {
                "status": "blocked",
                "reason": str(exc),
                "error_type": type(exc).__name__,
            }
            result["error_count"] = int(result.get("error_count") or 0) + 1
    else:
        result["validated_publication"] = {
            "status": "pending_evidence",
            "held_out_attempts": len(validation_specs),
            "required_held_out_attempts": args.strategy_min_held_out_attempts,
            "held_out_task_count": held_out_task_count,
            "required_held_out_task_count": args.strategy_min_held_out_tasks,
        }
    return proposed_tree, result


def _new_experiment_backend(
    args: argparse.Namespace,
    *,
    max_tokens: int | None = None,
) -> OpenAICompatiblePlannerBackend:
    provider = load_planner_provider_config()
    if args.model:
        provider.model = args.model
    missing = provider.missing_fields()
    if missing:
        raise ValueError(
            f"planner provider config is missing: {', '.join(missing)}"
        )
    config = OpenAICompatiblePlannerBackendConfig.from_provider_config(provider)
    if max_tokens is not None:
        config.max_tokens = max_tokens
    return OpenAICompatiblePlannerBackend(config)


def _run_batch(
    args: argparse.Namespace,
    specs: list[ParallelEpisodeSpec],
    *,
    batch_id: str,
) -> JsonDict:
    worker_factory = build_mcp_episode_worker_factory(
        model_override=args.model,
        sim_url=args.sim_url,
        sam3_url=args.sam3_url,
        anygrasp_url=args.anygrasp_url,
        anyplace_url=args.anyplace_url,
        graspgenx_url=args.graspgenx_url,
        contact_graspnet_url=args.contact_graspnet_url,
        molmopoint_url=args.molmopoint_url,
        supervision_profile=args.approvement,
        provider_concurrency=args.provider_concurrency,
        provider_queue_timeout_s=args.provider_queue_timeout_s,
    )
    harness = ParallelEpisodeHarness(
        worker_factory,
        concurrency=args.concurrency,
    )
    payload = harness.run(specs, batch_id=batch_id).to_dict()
    provider_metrics = getattr(worker_factory, "provider_metrics", None)
    if callable(provider_metrics):
        payload["provider_concurrency"] = provider_metrics()
    return payload


def _review_candidates(
    args: argparse.Namespace,
    candidates: list[JsonDict],
    baseline: Path,
) -> list[JsonDict]:
    provider = load_planner_provider_config()
    if args.model:
        provider.model = args.model
    missing = provider.missing_fields()
    if missing:
        raise ValueError(f"planner provider config is missing: {', '.join(missing)}")
    reviewer = BackendSkillChangeReviewer(
        OpenAICompatiblePlannerBackend(
            OpenAICompatiblePlannerBackendConfig.from_provider_config(provider)
        )
    )
    executable_tools = tuple(
        spec
        for spec in build_default_tool_registry().list()
        if spec.name not in _BATCH_UNBOUND_TOOLS
    )
    reviews: list[JsonDict] = []
    for candidate in candidates:
        name = str(candidate.get("skill_name") or "")
        candidate_skill = load_skill_markdown(Path(str(candidate["candidate_path"])))
        baseline_path = baseline / f"{name}.md"
        current = load_skill_markdown(baseline_path) if baseline_path.exists() else None
        operation = "update" if current is not None else "register"
        request = SkillAuthoringRequest(
            operation=operation,
            parameters={
                "name": name,
                "goal": "Promote reusable guidance from objectively successful episodes.",
                "requested_changes": (
                    "Review the candidate against the immutable baseline. It is supported "
                    f"by {candidate.get('support_count', 0)} objectively successful episode(s): "
                    + ", ".join(candidate.get("supporting_episode_ids") or [])
                ),
            },
            executable_tools=executable_tools,
            current_skill=current,
        )
        try:
            decision = reviewer.review(request=request, skill=candidate_skill)
            reviews.append(
                {
                    "skill_name": name,
                    "candidate_sha256": candidate.get("sha256"),
                    "decision": decision.decision,
                    "reason": decision.reason,
                    "details": decision.details,
                }
            )
        except Exception as exc:  # noqa: BLE001 - promotion review fails closed.
            reviews.append(
                {
                    "skill_name": name,
                    "candidate_sha256": candidate.get("sha256"),
                    "decision": "reject",
                    "reason": f"reviewer failure: {exc}",
                    "error_type": type(exc).__name__,
                }
            )
    return reviews


def _write_generation_result(
    experiment: ExperimentWorkspace,
    generation: int,
    payload: JsonDict,
) -> None:
    path = experiment.generation_dir(generation) / "result.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _resolved_mcp_urls(
    args: argparse.Namespace,
) -> tuple[str, str, str, str, str, str, str]:
    return (
        args.sim_url or load_mcp_server_url("openeta-sim", aliases=("sim",)),
        args.sam3_url or load_mcp_server_url("openeta-sam3", aliases=("sam3",)),
        args.anygrasp_url or load_mcp_server_url("openeta-anygrasp", aliases=("anygrasp",)),
        getattr(args, "contact_graspnet_url", "")
        or load_mcp_server_url(
            "openeta-contact-graspnet",
            aliases=("contact-graspnet", "contact_graspnet"),
        ),
        getattr(args, "graspgenx_url", "")
        or load_mcp_server_url("openeta-graspgenx", aliases=("graspgenx",)),
        getattr(args, "anyplace_url", "")
        or load_mcp_server_url("openeta-anyplace", aliases=("anyplace",)),
        getattr(args, "molmopoint_url", "")
        or load_mcp_server_url(
            "openeta-molmopoint",
            aliases=("molmopoint", "molmo-point"),
        ),
    )


def _validate_concurrency(value: int) -> None:
    if not 1 <= value <= MAX_PARALLEL_EPISODES:
        raise ValueError(f"concurrency must be between 1 and {MAX_PARALLEL_EPISODES}")


def _validate_provider_limits(args: argparse.Namespace) -> None:
    provider_concurrency = getattr(
        args, "provider_concurrency", DEFAULT_PROVIDER_CONCURRENCY
    )
    queue_timeout_s = getattr(
        args,
        "provider_queue_timeout_s",
        DEFAULT_PROVIDER_QUEUE_TIMEOUT_S,
    )
    if provider_concurrency < 1:
        raise ValueError("provider concurrency must be positive")
    if queue_timeout_s <= 0:
        raise ValueError("provider queue timeout must be positive")


def _validate_strategy_limits(args: argparse.Namespace) -> None:
    canary = int(getattr(args, "strategy_min_canary_attempts", 2))
    held_out = int(getattr(args, "strategy_min_held_out_attempts", 20))
    task_count = int(getattr(args, "strategy_min_held_out_tasks", 2))
    success_rate = float(
        getattr(args, "strategy_min_held_out_success_rate", 0.95)
    )
    if canary < 1 or held_out < 1 or task_count < 1:
        raise ValueError("grasp strategy evidence counts must be positive")
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError("strategy held-out success rate must be in [0, 1]")


def _filter_episode_specs(
    specs: list[ParallelEpisodeSpec],
    requested_ids: list[str],
) -> list[ParallelEpisodeSpec]:
    if not requested_ids:
        return specs
    requested = list(dict.fromkeys(requested_ids))
    by_id = {spec.episode_id: spec for spec in specs}
    missing = [episode_id for episode_id in requested if episode_id not in by_id]
    if missing:
        raise ValueError("episode ids are not present in manifest: " + ", ".join(missing))
    return [by_id[episode_id] for episode_id in requested]


def _compact_command_output(command: str, payload: JsonDict) -> JsonDict:
    if command != "run" or not isinstance(payload.get("batch"), dict):
        return payload
    batch = payload["batch"]
    failures = []
    for outcome in batch.get("outcomes", []):
        if not isinstance(outcome, dict) or outcome.get("status") == "success":
            continue
        error = outcome.get("error") if isinstance(outcome.get("error"), dict) else {}
        failures.append(
            {
                "episode_id": outcome.get("episode_id"),
                "status": outcome.get("status"),
                "code": error.get("code"),
                "type": error.get("type"),
                "message": error.get("message"),
            }
        )
    return {
        key: payload[key]
        for key in (
            "schema_version",
            "ok",
            "experiment_id",
            "generation",
            "result_path",
            "metrics",
            "candidate_count",
        )
        if key in payload
    } | {
        "provider_concurrency": batch.get("provider_concurrency"),
        "failures": failures,
    }


def _validated_experiment_root(value: str) -> Path:
    root = Path(value)
    if root.is_absolute() or ".." in root.parts:
        raise ValueError("experiment root must be a relative path inside the repository")
    return root


def _create_experiment(
    args: argparse.Namespace,
    experiment_id: str,
) -> ExperimentWorkspace:
    return ExperimentWorkspace.create(
        experiment_id,
        root=_validated_experiment_root(args.experiment_root),
    )


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--concurrency", type=int, default=DEFAULT_PARALLEL_EPISODES)
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
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--print-full-result", action="store_true")
    parser.add_argument("--model", default="")
    parser.add_argument("--sim-url", default="")
    parser.add_argument("--sam3-url", default="")
    parser.add_argument("--anygrasp-url", default="")
    parser.add_argument("--contact-graspnet-url", default="")
    parser.add_argument("--graspgenx-url", default="")
    parser.add_argument("--anyplace-url", default="")
    parser.add_argument("--molmopoint-url", default="")
    parser.add_argument(
        "--approvement",
        choices=[profile.value for profile in SupervisionProfile],
        default=_UNATTENDED_PROFILE,
    )
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    parser.add_argument("--baseline-skills", default=str(BUILTIN_SKILL_DIR))
    parser.add_argument(
        "--baseline-grasp-strategies",
        default=str(DEFAULT_GRASP_STRATEGY_ROOT),
    )
    parser.add_argument(
        "--baseline-task-playbooks",
        default=str(DEFAULT_TASK_PLAYBOOK_ROOT),
    )
    parser.add_argument(
        "--calibration-profile",
        default=str(DEFAULT_GRASP_PROFILE),
    )
    parser.add_argument(
        "--publish-grasp-strategies",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--strategy-min-canary-attempts", type=int, default=2)
    parser.add_argument("--strategy-min-held-out-attempts", type=int, default=20)
    parser.add_argument("--strategy-min-held-out-success-rate", type=float, default=0.95)
    parser.add_argument("--strategy-min-held-out-tasks", type=int, default=2)
    parser.add_argument("--on-need-human", choices=("fail", "pause"), default="fail")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenETA non-interactive experiment commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    _add_runtime_arguments(preflight_parser)
    preflight_parser.add_argument("--manifest", required=True)
    preflight_parser.add_argument("--skip-mcp-check", action="store_true")
    preflight_parser.add_argument("--require-perception", action="store_true")
    preflight_parser.add_argument("--mcp-timeout-s", type=float, default=10.0)

    run_parser = subparsers.add_parser("run")
    _add_runtime_arguments(run_parser)
    run_parser.add_argument("--manifest", required=True)

    iterate_parser = subparsers.add_parser("iterate")
    _add_runtime_arguments(iterate_parser)
    iterate_parser.add_argument("--train-manifest", required=True)
    iterate_parser.add_argument("--validation-manifest", required=True)
    iterate_parser.add_argument("--rounds", type=int, default=1)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--experiment-id", required=True)
    inspect_parser.add_argument("--experiment-root", default=str(DEFAULT_EXPERIMENT_ROOT))
    return parser
