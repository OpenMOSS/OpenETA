"""Bounded parallel harness for independent OpenETA simulator episodes."""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.episode import (
    DEFAULT_EPISODE_TIMEOUT_S,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_MAX_TURNS,
    EpisodeResult,
    OpenEtaEpisodeRunner,
)


DEFAULT_PARALLEL_EPISODES = 10
MAX_PARALLEL_EPISODES = 32


@dataclass(frozen=True, slots=True)
class ParallelEpisodeSpec:
    """One independently planned simulator episode in a batch."""

    episode_id: str
    task: str
    env_id: str
    seed: int = 0
    max_turns: int = DEFAULT_MAX_TURNS
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    timeout_s: float = DEFAULT_EPISODE_TIMEOUT_S
    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    metadata: JsonDict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: JsonDict, *, index: int) -> "ParallelEpisodeSpec":
        task = str(payload.get("task") or "").strip()
        env_id = str(payload.get("env_id") or "").strip()
        if not task:
            raise ValueError(f"episodes[{index}].task is required")
        if not env_id:
            raise ValueError(f"episodes[{index}].env_id is required")
        episode_id = str(payload.get("episode_id") or f"episode-{index:03d}").strip()
        max_turns = int(payload.get("max_turns", DEFAULT_MAX_TURNS))
        if max_turns < 1:
            raise ValueError(f"episodes[{index}].max_turns must be positive")
        max_tool_calls = int(payload.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS))
        if max_tool_calls < 1:
            raise ValueError(f"episodes[{index}].max_tool_calls must be positive")
        timeout_s = float(payload.get("timeout_s", DEFAULT_EPISODE_TIMEOUT_S))
        if timeout_s <= 0:
            raise ValueError(f"episodes[{index}].timeout_s must be positive")
        max_total_tokens = int(payload.get("max_total_tokens", DEFAULT_MAX_TOTAL_TOKENS))
        if max_total_tokens < 1:
            raise ValueError(f"episodes[{index}].max_total_tokens must be positive")
        metadata = payload.get("metadata")
        return cls(
            episode_id=episode_id,
            task=task,
            env_id=env_id,
            seed=int(payload.get("seed", 0)),
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            timeout_s=timeout_s,
            max_total_tokens=max_total_tokens,
            metadata=dict(metadata) if isinstance(metadata, dict) else {},
        )


@dataclass(slots=True)
class ParallelEpisodeWorker:
    """One runner and its mandatory best-effort cleanup callback."""

    runner: OpenEtaEpisodeRunner
    close: Callable[[], JsonDict]
    pause: Callable[[EpisodeResult], JsonDict] | None = None
    run_metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ParallelEpisodeOutcome:
    """Failure-isolated result for one manifest entry."""

    index: int
    spec: ParallelEpisodeSpec
    status: str
    duration_s: float
    episode: EpisodeResult | None = None
    cleanup: JsonDict = field(default_factory=dict)
    error: JsonDict = field(default_factory=dict)
    interaction: JsonDict = field(default_factory=dict)
    human_intervention_count: int = 0
    guidance_intervention_count: int = 0

    def to_dict(self) -> JsonDict:
        return {
            "index": self.index,
            "episode_id": self.spec.episode_id,
            "env_id": self.spec.env_id,
            "seed": self.spec.seed,
            "status": self.status,
            "duration_s": round(self.duration_s, 3),
            "limits": {
                "max_turns": self.spec.max_turns,
                "max_tool_calls": self.spec.max_tool_calls,
                "timeout_s": self.spec.timeout_s,
                "max_total_tokens": self.spec.max_total_tokens,
            },
            "session_id": self.episode.session_id if self.episode else None,
            "episode": self.episode.to_dict() if self.episode else None,
            "assistance": {
                "assisted": (
                    self.human_intervention_count > 0 or self.guidance_intervention_count > 0
                ),
                "human_intervention_count": self.human_intervention_count,
                "guidance_intervention_count": self.guidance_intervention_count,
            },
            "interaction": self.interaction,
            "cleanup": self.cleanup,
            "error": self.error,
        }


@dataclass(slots=True)
class ParallelEpisodeBatchResult:
    """Ordered aggregate returned after every worker has settled."""

    batch_id: str
    concurrency: int
    duration_s: float
    outcomes: list[ParallelEpisodeOutcome]

    @property
    def failed_count(self) -> int:
        return self.fail_count

    @property
    def fail_count(self) -> int:
        return sum(outcome.status == "fail" for outcome in self.outcomes)

    @property
    def need_human_count(self) -> int:
        return sum(outcome.status == "need_human" for outcome in self.outcomes)

    @property
    def success_count(self) -> int:
        return sum(outcome.status == "success" for outcome in self.outcomes)

    @property
    def assisted_success_count(self) -> int:
        return sum(
            outcome.status == "success"
            and (outcome.human_intervention_count > 0 or outcome.guidance_intervention_count > 0)
            for outcome in self.outcomes
        )

    @property
    def agent_assisted_success_count(self) -> int:
        return sum(
            outcome.status == "success"
            and outcome.guidance_intervention_count > 0
            and outcome.human_intervention_count == 0
            for outcome in self.outcomes
        )

    @property
    def human_assisted_success_count(self) -> int:
        return sum(
            outcome.status == "success" and outcome.human_intervention_count > 0
            for outcome in self.outcomes
        )

    @property
    def intervention_count(self) -> int:
        return sum(
            outcome.human_intervention_count > 0 or outcome.guidance_intervention_count > 0
            for outcome in self.outcomes
        )

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": "openeta.parallel_episode_batch.v2",
            "batch_id": self.batch_id,
            "concurrency": self.concurrency,
            "duration_s": round(self.duration_s, 3),
            "episode_count": len(self.outcomes),
            "success_count": self.success_count,
            "autonomous_success_count": self.success_count - self.assisted_success_count,
            "assisted_success_count": self.assisted_success_count,
            "agent_assisted_success_count": self.agent_assisted_success_count,
            "human_assisted_success_count": self.human_assisted_success_count,
            "need_human_count": self.need_human_count,
            "fail_count": self.fail_count,
            "rates": {
                "success_rate": _ratio(self.success_count, len(self.outcomes)),
                "autonomous_success_rate": _ratio(
                    self.success_count - self.assisted_success_count,
                    len(self.outcomes),
                ),
                "assisted_success_rate": _ratio(self.assisted_success_count, len(self.outcomes)),
                "agent_assisted_success_rate": _ratio(
                    self.agent_assisted_success_count, len(self.outcomes)
                ),
                "human_assisted_success_rate": _ratio(
                    self.human_assisted_success_count, len(self.outcomes)
                ),
                "intervention_rate": _ratio(self.intervention_count, len(self.outcomes)),
            },
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


ParallelEpisodeWorkerFactory = Callable[[ParallelEpisodeSpec, str], ParallelEpisodeWorker]


class ParallelEpisodeHarness:
    """Run independent episodes concurrently with bounded resource usage."""

    def __init__(
        self,
        worker_factory: ParallelEpisodeWorkerFactory,
        *,
        concurrency: int = DEFAULT_PARALLEL_EPISODES,
    ) -> None:
        if not 1 <= concurrency <= MAX_PARALLEL_EPISODES:
            raise ValueError(f"concurrency must be between 1 and {MAX_PARALLEL_EPISODES}")
        self.worker_factory = worker_factory
        self.concurrency = concurrency
        self._active_workers: dict[int, ParallelEpisodeWorker] = {}
        self._active_workers_lock = threading.Lock()
        self._cancel_event = threading.Event()

    def run(
        self,
        specs: list[ParallelEpisodeSpec],
        *,
        batch_id: str | None = None,
    ) -> ParallelEpisodeBatchResult:
        if not specs:
            raise ValueError("parallel episode batch requires at least one episode")
        resolved_batch_id = batch_id or f"batch-{uuid4().hex[:12]}"
        started = time.monotonic()
        outcomes: list[ParallelEpisodeOutcome] = []
        self._cancel_event.clear()
        executor = ThreadPoolExecutor(
            max_workers=min(self.concurrency, len(specs)),
            thread_name_prefix="openeta-sim",
        )
        futures = {}
        try:
            futures = {
                executor.submit(self._run_one, index, spec, resolved_batch_id): index
                for index, spec in enumerate(specs)
            }
            for future in as_completed(futures):
                outcomes.append(future.result())
        except BaseException:
            self.interrupt()
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        outcomes.sort(key=lambda outcome: outcome.index)
        return ParallelEpisodeBatchResult(
            batch_id=resolved_batch_id,
            concurrency=min(self.concurrency, len(specs)),
            duration_s=time.monotonic() - started,
            outcomes=outcomes,
        )

    def interrupt(self) -> list[JsonDict]:
        """Propagate batch cancellation to every currently active episode runner."""

        self._cancel_event.set()
        with self._active_workers_lock:
            workers = list(self._active_workers.items())
        results: list[JsonDict] = []
        for index, worker in workers:
            interrupt = getattr(worker.runner, "interrupt", None)
            if not callable(interrupt):
                results.append({"index": index, "ok": False, "reason": "interrupt_not_supported"})
                continue
            try:
                cleanup = interrupt(code="parallel_batch_interrupted")
                results.append(
                    {
                        "index": index,
                        **(cleanup if isinstance(cleanup, dict) else {"ok": True}),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - cancellation remains best effort.
                results.append(
                    {
                        "index": index,
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        return results

    def _run_one(
        self,
        index: int,
        spec: ParallelEpisodeSpec,
        batch_id: str,
    ) -> ParallelEpisodeOutcome:
        started = time.monotonic()
        worker: ParallelEpisodeWorker | None = None
        episode: EpisodeResult | None = None
        cleanup: JsonDict = {"ok": True, "skipped": True}
        error: JsonDict = {}
        interaction: JsonDict = {}
        intervention_count = int(spec.metadata.get("human_intervention_count") or 0)
        guidance_intervention_count = int(spec.metadata.get("guidance_intervention_count") or 0)
        try:
            worker = self.worker_factory(spec, batch_id)
            with self._active_workers_lock:
                self._active_workers[index] = worker
            if self._cancel_event.is_set():
                interrupt = getattr(worker.runner, "interrupt", None)
                if callable(interrupt):
                    interrupt(code="parallel_batch_interrupted")
                raise RuntimeError("parallel batch interrupted before episode start")
            episode = worker.runner.run(
                task=spec.task,
                max_turns=spec.max_turns,
                max_tool_calls=spec.max_tool_calls,
                timeout_s=spec.timeout_s,
                max_total_tokens=spec.max_total_tokens,
                metadata={
                    "source": type(self).__name__,
                    "batch_id": batch_id,
                    "episode_id": spec.episode_id,
                    "env_id": spec.env_id,
                    "seed": spec.seed,
                    **spec.metadata,
                    **worker.run_metadata,
                },
            )
            status = classify_episode_result(
                episode,
                env_id=spec.env_id,
                require_official_reward=spec.metadata.get("require_official_reward"),
            )
            if status == "fail":
                error = episode_failure_error(episode)
            if status == "need_human":
                if spec.metadata.get("on_need_human") == "fail":
                    status = "fail"
                    error = {
                        "type": "UnattendedHumanInterventionRequired",
                        "code": "need_human_in_unattended_run",
                        "message": "Episode requested human input during an unattended run.",
                    }
                else:
                    if worker.pause is None:
                        raise RuntimeError(
                            "need_human episode requires a persistent pause callback"
                        )
                    interaction = worker.pause(episode)
            guidance_intervention_count += int(
                ((episode.metadata.get("assistance") or {}).get("guidance_intervention_count")) or 0
            )
        except Exception as exc:  # noqa: BLE001 - batch workers must be isolated.
            status = "fail"
            error = {"type": type(exc).__name__, "message": str(exc)}
            code = getattr(exc, "code", None)
            if isinstance(code, str) and code:
                error["code"] = code
        finally:
            if worker is not None:
                try:
                    cleanup = worker.close()
                except Exception as exc:  # noqa: BLE001 - preserve primary outcome.
                    cleanup = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                finally:
                    with self._active_workers_lock:
                        self._active_workers.pop(index, None)
        if status != "need_human" and cleanup.get("ok") is False:
            status = "fail"
            error.setdefault("type", "CleanupError")
            error.setdefault("message", str(cleanup.get("error") or "cleanup failed"))
        return ParallelEpisodeOutcome(
            index=index,
            spec=spec,
            status=status,
            duration_s=time.monotonic() - started,
            episode=episode,
            cleanup=cleanup,
            error=error,
            interaction=interaction,
            human_intervention_count=intervention_count,
            guidance_intervention_count=guidance_intervention_count,
        )


def classify_episode_result(
    episode: EpisodeResult,
    *,
    env_id: str = "",
    require_official_reward: object = None,
) -> str:
    if episode.metadata.get("waiting_for_human"):
        return "need_human"
    official_reward_required = _requires_official_reward(
        env_id=env_id,
        explicit=require_official_reward,
    )
    if _episode_has_objective_success(
        episode,
        require_official_reward=official_reward_required,
    ):
        return "success"
    if episode.terminated and episode.metadata.get("stop_reason") == "task_complete":
        if episode.steps:
            request = episode.steps[-1].action.command.get("request")
            parameters = request.get("parameters") if isinstance(request, dict) else {}
            if isinstance(parameters, dict) and parameters.get("success") is False:
                return "fail"
        if official_reward_required:
            return "fail"
        return "success"
    return "fail"


def _requires_official_reward(*, env_id: str, explicit: object) -> bool:
    if isinstance(explicit, bool):
        return explicit
    return "libero" in env_id.lower()


def _episode_has_objective_success(
    episode: EpisodeResult,
    *,
    require_official_reward: bool,
) -> bool:
    expected_execution_id = str(episode.metadata.get("execution_id") or "")
    for step in episode.steps:
        reward = step.step_result.reward
        reward_positive = (
            isinstance(reward, int | float)
            and not isinstance(reward, bool)
            and math.isfinite(float(reward))
            and reward > 0
        )
        if require_official_reward:
            info = step.step_result.info
            receipt = info.get("environment_receipt") if isinstance(info, dict) else None
            if (
                reward_positive
                and info.get("environment_receipt_trusted") is True
                and info.get("official_reward") is True
                and isinstance(receipt, dict)
                and receipt.get("schema_version") == "openeta.environment_receipt.v1"
                and (
                    not expected_execution_id
                    or receipt.get("execution_id") == expected_execution_id
                )
            ):
                return True
            continue
        if reward_positive:
            return True
        info = step.step_result.info
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


def episode_failure_error(episode: EpisodeResult) -> JsonDict:
    reason = episode.metadata.get("failure_reason")
    if not isinstance(reason, dict) or not reason:
        return {}
    return {
        "type": "EpisodeResourceLimit",
        "code": reason.get("code"),
        "limit": reason.get("limit"),
        "observed": reason.get("observed"),
        "unit": reason.get("unit"),
    }


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0
