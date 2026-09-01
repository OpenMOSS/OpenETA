"""Universal durable evaluation entry point for OpenETA rollouts."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.backends.provider_config import load_planner_provider_config
from agent.cli.batch_eval import build_mcp_episode_worker_factory
from agent.evals.plan import (
    EvaluationExecution,
    compile_evaluation_plan,
    compiled_plan_payload,
    evaluation_job_from_dict,
    load_evaluation_plan,
)
from agent.evals.runner import (
    EvaluationScheduler,
    build_evaluation_report,
    inspect_evaluation_run,
)
from agent.evals.store import DEFAULT_EVALUATION_ROOT, EvaluationRunStore
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.calibration_registry import load_grasp_calibration_capabilities
from agent.runtime.mcp_catalog import simulator_mcp_contract_diagnostics
from agent.runtime.grasp_strategy_projection import (
    normalize_grasp_strategy_projection,
)
from agent.runtime.visual_history import VisualHistoryConfig
from agent.tools.anygrasp_capabilities import check_anygrasp_compatibility
from agent.tools.grasp_geometry import DEFAULT_GRASP_PROFILE
from agent.tools.mcp_registry import load_mcp_server_url
from agent.tools.object_memory import (
    load_configured_object_memory_bank,
    probe_object_memory_bank,
)
from agent.tools.sim_mcp import SseSimulatorMcpTransport


_REQUIRED_SIM_MCP_TOOLS = {
    "create_env",
    "reset_env",
    "render_env",
    "ik_preview_check",
    "move_to",
    "close_env",
    "gripper_open",
    "gripper_close",
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            payload = validate_plan(args.plan)
        elif args.command == "preflight":
            payload = preflight_plan(args)
        elif args.command == "run":
            payload = run_plan(args)
        elif args.command == "resume":
            payload = resume_run(args)
        elif args.command == "inspect":
            payload = inspect_evaluation_run(_store(args))
        elif args.command == "report":
            store = _store(args)
            jobs, _ = _compiled_jobs_and_execution(store)
            payload = build_evaluation_report(store, jobs)
            store.write_report(payload)
        else:  # pragma: no cover - argparse enforces commands.
            raise ValueError(f"unsupported eval command: {args.command}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


def validate_plan(path: str | Path) -> JsonDict:
    plan = load_evaluation_plan(path)
    jobs = compile_evaluation_plan(plan)
    for variant in plan.variants:
        _validate_runtime(variant.runtime)
    compiled = compiled_plan_payload(plan, jobs)
    return {
        "schema_version": "openeta.evaluation_validation.v1",
        "valid": True,
        "plan_id": plan.plan_id,
        "plan_sha256": compiled["plan_sha256"],
        "episode_count": len(plan.episodes),
        "variant_count": len(plan.variants),
        "repeat_count": plan.repeats,
        "job_count": len(jobs),
        "execution": plan.execution.to_dict(),
        "variant_ids": [variant.variant_id for variant in plan.variants],
    }


def run_plan(args: argparse.Namespace) -> JsonDict:
    plan = load_evaluation_plan(args.plan)
    jobs = compile_evaluation_plan(plan)
    for variant in plan.variants:
        _validate_runtime(variant.runtime)
    preflight = _remote_preflight(args)
    if not preflight["ok"]:
        raise ValueError("evaluation preflight failed: " + "; ".join(preflight["errors"]))
    compiled = compiled_plan_payload(plan, jobs)
    run_id = _safe_run_id(args.run_id or f"{plan.plan_id}-{uuid4().hex[:10]}")
    run_path = Path(args.root) / run_id / "run.json"
    if run_path.exists():
        raise ValueError(f"evaluation run already exists; use resume: {run_id}")
    store = EvaluationRunStore.create(
        run_id,
        root=args.root,
        compiled_plan=compiled,
        provenance=_provenance(args),
    )
    store.set_run_status("created", preflight=preflight)
    return _execute(store, jobs, plan.execution, args, resume=False)


def preflight_plan(args: argparse.Namespace) -> JsonDict:
    validation = validate_plan(args.plan)
    remote = _remote_preflight(args)
    return {
        "schema_version": "openeta.evaluation_preflight.v1",
        "ok": remote["ok"],
        "validation": validation,
        **remote,
    }


def resume_run(args: argparse.Namespace) -> JsonDict:
    store = _store(args)
    jobs, execution = _compiled_jobs_and_execution(store)
    _restore_execution_inputs(store, args)
    return _execute(store, jobs, execution, args, resume=True)


def _execute(
    store: EvaluationRunStore,
    jobs: tuple,
    execution: EvaluationExecution,
    args: argparse.Namespace,
    *,
    resume: bool,
) -> JsonDict:
    worker_factory = build_mcp_episode_worker_factory(
        model_override=args.model,
        sim_url=args.sim_url,
        sam3_url=args.sam3_url,
        depth_prior_url=args.depth_prior_url,
        anygrasp_url=args.anygrasp_url,
        anyplace_url=args.anyplace_url,
        graspgenx_url=args.graspgenx_url,
        molmopoint_url=args.molmopoint_url,
        supervision_profile=execution.supervision_profile,
        provider_concurrency=execution.provider_concurrency,
        provider_queue_timeout_s=execution.provider_queue_timeout_s,
    )
    scheduler = EvaluationScheduler(
        store=store,
        jobs=jobs,
        execution=execution,
        worker_factory=worker_factory,
    )
    return scheduler.run(resume=resume)


def _compiled_jobs_and_execution(
    store: EvaluationRunStore,
) -> tuple[tuple, EvaluationExecution]:
    if not store.exists():
        raise ValueError(f"unknown evaluation run: {store.run_id}")
    compiled = store.compiled_plan()
    raw_jobs = compiled.get("jobs")
    if not isinstance(raw_jobs, list):
        raise ValueError("compiled evaluation snapshot has no jobs")
    jobs = tuple(
        evaluation_job_from_dict(item)
        for item in raw_jobs
        if isinstance(item, dict)
    )
    plan = compiled.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("compiled evaluation snapshot has no plan")
    execution = EvaluationExecution.from_dict(plan.get("execution"))
    return jobs, execution


def _validate_runtime(runtime: JsonDict) -> None:
    allowed = {"visual_history", "grasp_strategies"}
    unsupported = sorted(str(key) for key in runtime if key not in allowed)
    if unsupported:
        raise ValueError("unsupported runtime section(s): " + ", ".join(unsupported))
    visual = runtime.get("visual_history")
    if visual is not None and not isinstance(visual, dict):
        raise ValueError("runtime.visual_history must be an object")
    VisualHistoryConfig.from_mapping(visual, base=VisualHistoryConfig())
    normalize_grasp_strategy_projection(runtime.get("grasp_strategies"))


def _store(args: argparse.Namespace) -> EvaluationRunStore:
    return EvaluationRunStore(_safe_run_id(args.run_id), root=args.root)


def _safe_run_id(value: str) -> str:
    result = str(value or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", result):
        raise ValueError("run-id contains unsupported characters")
    return result


def _provenance(args: argparse.Namespace) -> JsonDict:
    provider = load_planner_provider_config()
    if args.model:
        provider.model = args.model
    return {
        "captured_at_s": time.time(),
        "python": platform.python_version(),
        "git": _git_provenance(),
        "provider": provider.redacted(),
        "planner_prompt": dict(ToolCallingPlanner().prompt_metadata),
        "execution_inputs": _resolved_execution_inputs(args, model=provider.model),
    }


def _remote_preflight(args: argparse.Namespace) -> JsonDict:
    provider = load_planner_provider_config()
    if args.model:
        provider.model = args.model
    errors = [
        "planner provider config is missing: " + ", ".join(provider.missing_fields())
    ] if provider.missing_fields() else []
    warnings: list[str] = []
    inputs = _resolved_execution_inputs(args, model=provider.model)
    sim_url = str(inputs.get("sim_url") or "")
    catalog: JsonDict = {"checked": False, "url": sim_url}
    anygrasp_url = str(inputs.get("anygrasp_url") or "")
    anygrasp: JsonDict = {
        "backend": "anygrasp",
        "configured": bool(anygrasp_url),
        "url": anygrasp_url,
        "available": False,
        "compatible": False,
        "checked": False,
    }
    object_memory: JsonDict = {
        "configured": False,
        "checked": False,
        "available": False,
    }
    if not sim_url:
        errors.append("simulator MCP URL is required")
    elif not args.skip_mcp_check:
        if args.mcp_timeout_s <= 0:
            errors.append("mcp timeout must be positive")
        else:
            try:
                response = SseSimulatorMcpTransport(sim_url).list_tools(
                    timeout_s=args.mcp_timeout_s
                )
                names = {
                    str(item.get("name") or "")
                    for item in response.get("tools", [])
                    if isinstance(item, dict)
                }
                missing = sorted(_REQUIRED_SIM_MCP_TOOLS - names)
                contract_diagnostics = simulator_mcp_contract_diagnostics(
                    list(response.get("tools", []))
                )
                catalog = {
                    "checked": True,
                    "url": sim_url,
                    "tool_count": len(names),
                    "missing_required_tools": missing,
                    "contract_compatible": not contract_diagnostics,
                    "contract_diagnostics": contract_diagnostics,
                }
                if missing:
                    errors.append(
                        "simulator MCP is missing required tools: " + ", ".join(missing)
                    )
                warnings.extend(
                    str(item.get("message") or item.get("code"))
                    for item in contract_diagnostics
                )
            except Exception as exc:  # noqa: BLE001 - aggregate preflight diagnostics.
                catalog = {
                    "checked": True,
                    "url": sim_url,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                errors.append(f"simulator MCP list_tools failed: {exc}")
    if anygrasp_url and not args.skip_mcp_check:
        try:
            physical = load_grasp_calibration_capabilities(
                args.calibration_profile
            )["max_gripper_width_m"]
            anygrasp = {
                **check_anygrasp_compatibility(
                    url=anygrasp_url,
                    physical_max_gripper_width_m=float(physical),
                    timeout_s=args.mcp_timeout_s,
                ),
                "checked": True,
            }
        except Exception as exc:  # noqa: BLE001 - aggregate preflight diagnostics.
            anygrasp = {
                **anygrasp,
                "checked": True,
                "reason": "host_capability_preflight_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "message": "AnyGrasp is unavailable: host capability preflight failed.",
            }
        if anygrasp.get("compatible") is not True:
            warnings.append(
                str(anygrasp.get("message") or "AnyGrasp is unavailable")
            )
    try:
        object_memory_config = load_configured_object_memory_bank()
        object_memory = probe_object_memory_bank(
            object_memory_config,
            timeout_s=min(max(args.mcp_timeout_s, 0.25), 3.0),
        )
    except Exception as exc:  # noqa: BLE001 - aggregate sanitized diagnostics.
        object_memory = {
            "schema_version": "openeta.object_memory_health.v1",
            "configured": False,
            "checked": True,
            "available": False,
            "reason": "object_memory_preflight_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    if object_memory.get("available") is not True:
        reason = str(
            object_memory.get("reason")
            or object_memory.get("error")
            or object_memory.get("status")
            or "health check did not report available=true"
        )
        errors.append(
            "required Object Memory Bank is unavailable from the evaluation "
            f"worker: {reason}. Restore the configured service before starting "
            "the evaluation."
        )
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "provider": {
            "provider": provider.provider,
            "model": provider.model,
        },
        "mcp": {
            "simulator": catalog,
            "anygrasp": anygrasp,
            "object_memory": object_memory,
        },
        "execution_inputs": inputs,
    }


def _resolved_execution_inputs(args: argparse.Namespace, *, model: str) -> JsonDict:
    """Persist resolved endpoints so resume does not depend on changed registry state."""

    return {
        "model": model,
        "sim_url": args.sim_url
        or load_mcp_server_url("openeta-sim", aliases=("sim",)),
        "sam3_url": args.sam3_url
        or load_mcp_server_url("openeta-sam3", aliases=("sam3",)),
        "depth_prior_url": args.depth_prior_url
        or load_mcp_server_url(
            "openeta-depth-prior",
            aliases=("depth-prior", "depth_prior", "unidepth"),
        ),
        "anygrasp_url": args.anygrasp_url
        or load_mcp_server_url("openeta-anygrasp", aliases=("anygrasp",)),
        "anyplace_url": args.anyplace_url
        or load_mcp_server_url("openeta-anyplace", aliases=("anyplace",)),
        "graspgenx_url": args.graspgenx_url
        or load_mcp_server_url("openeta-graspgenx", aliases=("graspgenx",)),
        "molmopoint_url": args.molmopoint_url
        or load_mcp_server_url("openeta-molmopoint", aliases=("molmopoint",)),
        "calibration_profile": str(args.calibration_profile),
    }


def _restore_execution_inputs(
    store: EvaluationRunStore,
    args: argparse.Namespace,
) -> None:
    provenance = store.run_metadata().get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    persisted = provenance.get("execution_inputs")
    persisted = persisted if isinstance(persisted, dict) else {}
    for key in (
        "model",
        "sim_url",
        "sam3_url",
        "depth_prior_url",
        "anygrasp_url",
        "anyplace_url",
        "graspgenx_url",
        "molmopoint_url",
    ):
        if not getattr(args, key, "") and persisted.get(key):
            setattr(args, key, str(persisted[key]))


def _git_provenance() -> JsonDict:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"revision": "", "dirty": None}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run reproducible, crash-resumable OpenETA evaluations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate and compile a plan locally.")
    validate.add_argument("--plan", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="Validate a plan, provider config, and simulator catalog."
    )
    preflight.add_argument("--plan", required=True)
    _add_execution_args(preflight)

    run = subparsers.add_parser("run", help="Create and execute a new evaluation run.")
    run.add_argument("--plan", required=True)
    run.add_argument("--run-id", default="")
    _add_execution_args(run)

    resume = subparsers.add_parser("resume", help="Resume unfinished jobs in a run.")
    resume.add_argument("--run-id", required=True)
    _add_execution_args(resume)

    inspect = subparsers.add_parser("inspect", help="Inspect durable run state.")
    inspect.add_argument("--run-id", required=True)
    inspect.add_argument("--root", default=str(DEFAULT_EVALUATION_ROOT))

    report = subparsers.add_parser("report", help="Rebuild the generic report.")
    report.add_argument("--run-id", required=True)
    report.add_argument("--root", default=str(DEFAULT_EVALUATION_ROOT))
    return parser


def _add_execution_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=str(DEFAULT_EVALUATION_ROOT))
    parser.add_argument("--model", default="")
    parser.add_argument("--sim-url", default="")
    parser.add_argument("--sam3-url", default="")
    parser.add_argument("--depth-prior-url", default="")
    parser.add_argument("--anygrasp-url", default="")
    parser.add_argument("--anyplace-url", default="")
    parser.add_argument("--graspgenx-url", default="")
    parser.add_argument("--molmopoint-url", default="")
    parser.add_argument(
        "--calibration-profile",
        default=str(DEFAULT_GRASP_PROFILE),
    )
    parser.add_argument("--mcp-timeout-s", type=float, default=10.0)
    parser.add_argument("--skip-mcp-check", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
