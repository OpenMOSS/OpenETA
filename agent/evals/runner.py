"""Durable scheduling and generic reduction for OpenETA evaluations."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Callable

from adapter.protocol import JsonDict
from agent.evals.plan import EvaluationExecution, EvaluationJob
from agent.evals.store import EvaluationRunStore
from agent.runtime.parallel import (
    ParallelEpisodeHarness,
    ParallelEpisodeOutcome,
    ParallelEpisodeSpec,
    ParallelEpisodeWorker,
)


EVALUATION_REPORT_SCHEMA_VERSION = "openeta.evaluation_report.v1"
EvaluationWorkerFactory = Callable[[ParallelEpisodeSpec, str], ParallelEpisodeWorker]


class EvaluationScheduler:
    """Run compiled jobs in retry waves and commit every settled attempt."""

    def __init__(
        self,
        *,
        store: EvaluationRunStore,
        jobs: tuple[EvaluationJob, ...],
        execution: EvaluationExecution,
        worker_factory: EvaluationWorkerFactory,
    ) -> None:
        self.store = store
        self.jobs = jobs
        self.execution = execution
        self.worker_factory = worker_factory

    def run(self, *, resume: bool = False) -> JsonDict:
        if resume:
            self.store.abandon_incomplete_attempts()
        self.store.set_run_status("running", resumed=resume)
        started = time.monotonic()
        try:
            while True:
                pending = [
                    job for job in self.jobs if self.store.final_result(job.job_id) is None
                ]
                if not pending:
                    break
                scheduled: dict[str, tuple[EvaluationJob, int]] = {}
                specs: list[ParallelEpisodeSpec] = []
                for job in pending:
                    attempt = self.store.next_attempt(job.job_id)
                    if (
                        self.store.settled_attempt_count(job.job_id)
                        >= self.execution.max_attempts
                    ):
                        raise RuntimeError(
                            f"job {job.job_id} exceeded max_attempts without a final result"
                        )
                    prepared = self._attempt_spec(job, attempt)
                    self.store.start_attempt(
                        job,
                        attempt=attempt,
                        spec=_spec_to_dict(prepared),
                    )
                    scheduled[job.job_id] = (job, attempt)
                    specs.append(prepared)

                harness = ParallelEpisodeHarness(
                    self.worker_factory,
                    concurrency=self.execution.concurrency,
                )

                def commit(outcome: ParallelEpisodeOutcome) -> None:
                    job, attempt = scheduled[outcome.spec.episode_id]
                    payload = outcome.to_dict()
                    failure = classify_evaluation_failure(payload)
                    retryable = bool(failure.get("retryable"))
                    self.store.record_attempt_result(
                        job,
                        attempt=attempt,
                        outcome=payload,
                        failure=failure,
                        retryable=retryable,
                        max_attempts=self.execution.max_attempts,
                    )

                harness.run(
                    specs,
                    batch_id=f"eval-{self.store.run_id}",
                    on_outcome=commit,
                )
        except BaseException as exc:
            self.store.set_run_status(
                "interrupted",
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            raise

        scheduler_duration_s = time.monotonic() - started
        provider_metrics = getattr(self.worker_factory, "provider_metrics", None)
        provider_concurrency = (
            provider_metrics() if callable(provider_metrics) else {}
        )
        report = build_evaluation_report(
            self.store,
            self.jobs,
            scheduler_duration_s=scheduler_duration_s,
            provider_concurrency=provider_concurrency,
        )
        report_path = self.store.write_report(report)
        self.store.set_run_status(
            "complete",
            report_path=str(report_path),
            job_count=len(self.jobs),
            scheduler_duration_s=round(scheduler_duration_s, 3),
            provider_concurrency=provider_concurrency,
        )
        return report

    def _attempt_spec(self, job: EvaluationJob, attempt: int) -> ParallelEpisodeSpec:
        attempt_root = self.store.attempt_dir(job.job_id, attempt)
        return replace(
            job.spec,
            metadata={
                **job.spec.metadata,
                "workspace_parent": str(attempt_root),
                "evaluation_attempt": attempt,
                "evaluation_run_id": self.store.run_id,
            },
        )


def classify_evaluation_failure(outcome: JsonDict) -> JsonDict:
    """Map a batch-v2 outcome into stable, generic failure dimensions."""

    if outcome.get("status") == "success":
        return {
            "class": "none",
            "stage": "complete",
            "code": "",
            "retryable": False,
        }
    provider_pause = _provider_failure_pause(outcome)
    if provider_pause:
        return provider_pause
    if outcome.get("status") == "need_human":
        return {
            "class": "human_intervention",
            "stage": "agent",
            "code": "need_human",
            "retryable": False,
        }
    cleanup = outcome.get("cleanup")
    if isinstance(cleanup, dict) and cleanup.get("ok") is False:
        return {
            "class": "infrastructure",
            "stage": "cleanup",
            "code": "cleanup_failed",
            "retryable": True,
        }
    error = outcome.get("error")
    error = error if isinstance(error, dict) else {}
    episode = outcome.get("episode")
    episode = episode if isinstance(episode, dict) else {}
    metadata = episode.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    reason = metadata.get("failure_reason")
    reason = reason if isinstance(reason, dict) else {}
    code = str(error.get("code") or reason.get("code") or "").strip().lower()
    error_type = str(error.get("type") or "").strip().lower()
    searchable = f"{code} {error_type} {str(error.get('message') or '').lower()}"
    if any(marker in searchable for marker in ("provider", "rate_limit", "http_5")):
        return {
            "class": "infrastructure",
            "stage": "provider",
            "code": code or error_type or "provider_failure",
            "retryable": True,
        }
    if any(
        marker in searchable
        for marker in (
            "simulator",
            "environment_create",
            "connection",
            "mcp",
            "model_load_failed",
            "out_of_memory",
            "cuda_oom",
        )
    ):
        return {
            "class": "infrastructure",
            "stage": "environment",
            "code": code or error_type or "environment_failure",
            "retryable": True,
        }
    if code in {
        "tool_call_limit_exceeded",
        "episode_timeout",
        "token_limit_exceeded",
    }:
        return {
            "class": "resource_limit",
            "stage": "episode",
            "code": code,
            "retryable": False,
        }
    if metadata.get("stop_reason") == "task_complete":
        return {
            "class": "task_failure",
            "stage": "verdict",
            "code": code or "task_not_successful",
            "retryable": False,
        }
    return {
        "class": "agent_failure",
        "stage": "episode",
        "code": code or error_type or "episode_failed",
        "retryable": False,
    }


def _provider_failure_pause(outcome: JsonDict) -> JsonDict:
    """Distinguish provider exhaustion from a real Agent clarification request."""

    if outcome.get("status") != "need_human":
        return {}
    episode = outcome.get("episode")
    episode = episode if isinstance(episode, dict) else {}
    steps = episode.get("steps")
    steps = steps if isinstance(steps, list) else []
    if not steps or not isinstance(steps[-1], dict):
        return {}
    action = steps[-1].get("action")
    action = action if isinstance(action, dict) else {}
    parameters = action.get("request_parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    message = str(parameters.get("message") or "").strip().lower()
    error_type = str(parameters.get("error_type") or "").strip()
    provider_error_code = str(
        parameters.get("provider_error_code") or ""
    ).strip()
    provider_attempts = parameters.get("provider_attempts")
    if action.get("request_name") != "ask_human" or not (
        message == "planner provider request failed."
        and isinstance(provider_attempts, int)
        and provider_attempts > 0
    ):
        return {}
    retryable_value = parameters.get("retryable")
    retryable = retryable_value if isinstance(retryable_value, bool) else True
    external_dependency = provider_error_code in {
        "insufficient_provider_quota",
        "provider_credentials_or_access_denied",
    }
    return {
        "class": "external_dependency" if external_dependency else "infrastructure",
        "stage": "provider",
        "code": provider_error_code or "planner_provider_request_failed",
        "retryable": retryable,
        "error_type": error_type or None,
        "provider_attempts": provider_attempts,
    }


def build_evaluation_report(
    store: EvaluationRunStore,
    jobs: tuple[EvaluationJob, ...],
    *,
    scheduler_duration_s: float | None = None,
    provider_concurrency: JsonDict | None = None,
) -> JsonDict:
    """Reduce terminal jobs without adding experiment-specific interpretation."""

    by_id = {
        str(item.get("job_id") or ""): item for item in store.final_results()
    }
    finals = [by_id[job.job_id] for job in jobs if job.job_id in by_id]
    outcome_rows = [
        item.get("outcome") for item in finals if isinstance(item.get("outcome"), dict)
    ]
    status_counts = Counter(str(item.get("status") or "unknown") for item in outcome_rows)
    failure_counts = Counter(
        str((item.get("failure") or {}).get("class") or "unknown")
        for item in finals
        if (item.get("failure") or {}).get("class") != "none"
    )
    stage_counts = Counter(
        str((item.get("failure") or {}).get("stage") or "unknown")
        for item in finals
        if (item.get("failure") or {}).get("class") != "none"
    )
    objective_successes = sum(_has_objective_success(item) for item in outcome_rows)
    attempts = sum(int(item.get("attempts_used") or 0) for item in finals)
    durations = [
        float(item.get("duration_s"))
        for item in outcome_rows
        if _finite_number(item.get("duration_s"))
    ]
    variants: JsonDict = {}
    for variant_id in sorted({job.variant_id for job in jobs}):
        variant_jobs = {job.job_id for job in jobs if job.variant_id == variant_id}
        variant_finals = [item for item in finals if item.get("job_id") in variant_jobs]
        variant_outcomes = [
            item.get("outcome")
            for item in variant_finals
            if isinstance(item.get("outcome"), dict)
        ]
        variants[variant_id] = {
            "job_count": len(variant_jobs),
            "completed_count": len(variant_finals),
            "runtime_success_count": sum(
                item.get("status") == "success" for item in variant_outcomes
            ),
            "objective_success_count": sum(
                _has_objective_success(item) for item in variant_outcomes
            ),
            "failure_count": sum(item.get("status") == "fail" for item in variant_outcomes),
        }
    run_metadata = store.run_metadata()
    durable_scheduler_duration = run_metadata.get("scheduler_duration_s")
    wall = _positive_float(scheduler_duration_s)
    if wall is None:
        wall = _positive_float(durable_scheduler_duration)
    if wall is None:
        wall = _attempt_wall_clock(finals)
    if wall is None:
        wall = sum(durations)
    durable_provider_concurrency = run_metadata.get("provider_concurrency")
    if provider_concurrency is None:
        provider_concurrency = (
            dict(durable_provider_concurrency)
            if isinstance(durable_provider_concurrency, dict)
            and durable_provider_concurrency
            else _persisted_provider_concurrency(store)
        )
    return {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "run_id": store.run_id,
        "plan_id": store.run_metadata().get("plan_id"),
        "plan_sha256": store.run_metadata().get("plan_sha256"),
        "job_count": len(jobs),
        "completed_count": len(finals),
        "attempt_count": attempts,
        "retry_count": max(0, attempts - len(finals)),
        "status_counts": dict(sorted(status_counts.items())),
        "failure_class_counts": dict(sorted(failure_counts.items())),
        "failure_stage_counts": dict(sorted(stage_counts.items())),
        "objective_success_count": objective_successes,
        "objective_success_rate": _ratio(objective_successes, len(finals)),
        "wall_clock_s": round(wall, 3),
        "throughput_jobs_per_hour": round(len(finals) / wall * 3600, 6) if wall else 0.0,
        "mean_episode_duration_s": (
            round(sum(durations) / len(durations), 3) if durations else 0.0
        ),
        "provider_concurrency": provider_concurrency or {},
        "variants": variants,
        "results_root": str(store.root),
    }


def _attempt_wall_clock(finals: list[JsonDict]) -> float | None:
    starts = [
        float(item["started_at_s"])
        for item in finals
        if _finite_number(item.get("started_at_s"))
    ]
    completions = [
        float(item["completed_at_s"])
        for item in finals
        if _finite_number(item.get("completed_at_s"))
    ]
    if not starts or not completions:
        return None
    return max(0.0, max(completions) - min(starts))


def _persisted_provider_concurrency(store: EvaluationRunStore) -> JsonDict:
    """Recover the last cumulative provider snapshot from durable model calls."""

    best: JsonDict = {}
    best_key = (-1, -1.0)
    pattern = "jobs/*/attempts/*/sessions/*/rollout/model_calls.jsonl"
    for path in store.root.glob(pattern):
        try:
            stream = path.open(encoding="utf-8")
        except OSError:
            continue
        with stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict):
                    continue
                result = row.get("result")
                result = result if isinstance(result, dict) else {}
                details = result.get("details")
                details = details if isinstance(details, dict) else {}
                snapshot = details.get("provider_concurrency")
                if not isinstance(snapshot, dict):
                    continue
                request_count = snapshot.get("request_count")
                total_wait = snapshot.get("total_queue_wait_s")
                key = (
                    int(request_count) if isinstance(request_count, int) else -1,
                    float(total_wait) if _finite_number(total_wait) else -1.0,
                )
                if key > best_key:
                    best_key = key
                    best = dict(snapshot)
    if best:
        best["active"] = 0
    return best


def _positive_float(value: object) -> float | None:
    if _finite_number(value) and float(value) > 0:
        return float(value)
    return None


def inspect_evaluation_run(store: EvaluationRunStore) -> JsonDict:
    compiled = store.compiled_plan()
    jobs = compiled.get("jobs") if isinstance(compiled.get("jobs"), list) else []
    terminal = {str(item.get("job_id") or "") for item in store.final_results()}
    return {
        "schema_version": "openeta.evaluation_inspect.v1",
        "run": store.run_metadata(),
        "job_count": len(jobs),
        "terminal_count": len(terminal),
        "pending_job_ids": [
            str(item.get("job_id") or "")
            for item in jobs
            if isinstance(item, dict) and item.get("job_id") not in terminal
        ],
        "report": store.read_report(),
    }


def _has_objective_success(outcome: JsonDict) -> bool:
    episode = outcome.get("episode")
    if not isinstance(episode, dict):
        return False
    for step in episode.get("steps") or []:
        if not isinstance(step, dict):
            continue
        result = step.get("step_result")
        if not isinstance(result, dict):
            continue
        reward = result.get("reward")
        if _finite_number(reward) and float(reward) > 0:
            return True
        info = result.get("info")
        if isinstance(info, dict) and any(
            info.get(key) is True
            for key in (
                "task_success",
                "environment_success",
                "checker_success",
                "benchmark_success",
            )
        ):
            return True
    return False


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _spec_to_dict(spec: ParallelEpisodeSpec) -> JsonDict:
    return {
        "episode_id": spec.episode_id,
        "task": spec.task,
        "env_id": spec.env_id,
        "seed": spec.seed,
        "max_turns": spec.max_turns,
        "recovery_turns_per_branch": spec.recovery_turns_per_branch,
        "max_recovery_turns": spec.max_recovery_turns,
        "max_tool_calls": spec.max_tool_calls,
        "timeout_s": spec.timeout_s,
        "max_total_tokens": spec.max_total_tokens,
        "metadata": spec.metadata,
    }
