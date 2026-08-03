"""Closed-loop episode runner for OpenETA agent runtime."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol
from uuid import uuid4

from adapter.protocol import (
    CameraFrame,
    EnvAction,
    EnvObservation,
    JsonDict,
    RobotState,
    StepResult,
)
from agent.runtime.memory import summarize_event_payload, summarize_observation
from agent.runtime.runtime import OpenEtaAgentRuntime


DEFAULT_MAX_TURNS = 100
DEFAULT_MAX_TOOL_CALLS = 50
DEFAULT_EPISODE_TIMEOUT_S = 600.0
DEFAULT_MAX_TOTAL_TOKENS = 5_000_000
INTERRUPT_CLOSE_GRACE_S = 0.25


class EpisodeTimeoutError(TimeoutError):
    """Raised when the runner abandons a turn at the episode deadline."""


class EpisodeEnvironment(Protocol):
    """Minimal env boundary consumed by the closed-loop agent runner."""

    def reset(self, *, task: str, metadata: JsonDict | None = None) -> EnvObservation:
        """Return the first observation for a task."""
        ...

    def step(self, action: EnvAction) -> StepResult:
        """Apply one agent action and return the next observation/result."""
        ...


@dataclass(slots=True)
class EpisodeStep:
    """One observe-plan-execute-feedback turn."""

    turn_index: int
    observation: EnvObservation
    action: EnvAction
    step_result: StepResult

    def to_dict(self) -> JsonDict:
        return {
            "turn_index": self.turn_index,
            "observation": summarize_observation(self.observation),
            "action": summarize_action(self.action),
            "step_result": summarize_step_result(self.step_result),
        }


@dataclass(slots=True)
class EpisodeResult:
    """Summary of a closed-loop episode run."""

    task: str
    session_id: str | None
    steps: list[EpisodeStep] = field(default_factory=list)
    terminated: bool = False
    truncated: bool = False
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "task": self.task,
            "session_id": self.session_id,
            "num_steps": len(self.steps),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "metadata": self.metadata,
            "steps": [step.to_dict() for step in self.steps],
        }


class OpenEtaEpisodeRunner:
    """Run the primary OpenETA loop across multiple environment feedback turns."""

    def __init__(
        self,
        *,
        runtime: OpenEtaAgentRuntime,
        environment: EpisodeEnvironment,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime = runtime
        self.environment = environment
        self.task: str = ""
        self.turn_index = 0
        self.current_observation: EnvObservation | None = None
        self.terminated = False
        self.truncated = False
        self.waiting_for_human = False
        self.max_turns = 0
        self.max_tool_calls = DEFAULT_MAX_TOOL_CALLS
        self.timeout_s = DEFAULT_EPISODE_TIMEOUT_S
        self.max_total_tokens = DEFAULT_MAX_TOTAL_TOKENS
        self.tool_call_count = 0
        self.total_tokens = 0
        self.token_usage_sources: JsonDict = {}
        self.started_at_s = 0.0
        self.failure_reason: JsonDict = {}
        self.interrupt_cleanup: JsonDict = {}
        self.stop_reason = ""
        self._clock = clock
        self.execution_id = ""
        self._cancel_event = threading.Event()
        self._active_worker: threading.Thread | None = None
        self._worker_lock = threading.Lock()

    def start(
        self,
        *,
        task: str,
        max_turns: int,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        timeout_s: float = DEFAULT_EPISODE_TIMEOUT_S,
        max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        initial_tool_call_count: int = 0,
        initial_total_tokens: int = 0,
        initial_token_usage_sources: JsonDict | None = None,
        metadata: JsonDict | None = None,
    ) -> EnvObservation:
        self.task = task
        self.turn_index = 0
        self.terminated = False
        self.truncated = False
        self.waiting_for_human = False
        self.max_turns = max(1, max_turns)
        self.max_tool_calls = max(1, max_tool_calls)
        self.timeout_s = max(0.001, timeout_s)
        self.max_total_tokens = max(1, max_total_tokens)
        self.tool_call_count = max(0, initial_tool_call_count)
        self.total_tokens = max(0, initial_total_tokens)
        self.token_usage_sources = {
            str(key): max(0, int(value))
            for key, value in dict(initial_token_usage_sources or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self.started_at_s = self._clock()
        self.failure_reason = {}
        self.interrupt_cleanup = {}
        self.stop_reason = ""
        self.execution_id = str(uuid4())
        self._cancel_event = threading.Event()
        episode_metadata = {
            "source": type(self).__name__,
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "timeout_s": self.timeout_s,
            "max_total_tokens": self.max_total_tokens,
            **dict(metadata or {}),
        }
        if self.runtime.memory.session_id is None:
            self.runtime.start_session(task=task, metadata=episode_metadata)
        reset_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
        reset_cancelled = threading.Event()

        def reset_environment() -> None:
            try:
                observation = self.environment.reset(
                    task=task,
                    metadata=episode_metadata,
                )
                if reset_cancelled.is_set():
                    self._request_environment_close()
                    return
                reset_queue.put(("result", observation))
            except BaseException as exc:  # noqa: BLE001 - re-raised in runner thread.
                if not reset_cancelled.is_set():
                    reset_queue.put(("error", exc))

        reset_worker = threading.Thread(
            target=reset_environment,
            name="openeta-episode-reset",
            daemon=True,
        )
        reset_worker.start()
        try:
            reset_worker.join(timeout=self.timeout_s)
        except BaseException:
            reset_cancelled.set()
            raise
        if reset_worker.is_alive():
            reset_cancelled.set()
            self.interrupt(
                code="episode_timeout",
                limit=self.timeout_s,
                observed=max(self.timeout_s, self.elapsed_s),
                unit="seconds",
            )
            raise EpisodeTimeoutError(
                f"environment reset exceeded {self.timeout_s:g}-second deadline"
            )
        kind, payload = reset_queue.get_nowait()
        if kind == "error":
            assert isinstance(payload, BaseException)
            raise payload
        assert isinstance(payload, EnvObservation)
        self.current_observation = payload
        self.runtime.memory.record(
            "episode_start",
            {
                "task": task,
                "environment": type(self.environment).__name__,
                "max_turns": self.max_turns,
                "max_tool_calls": self.max_tool_calls,
                "timeout_s": self.timeout_s,
                "max_total_tokens": self.max_total_tokens,
                "metadata": episode_metadata,
            },
        )
        return self.current_observation

    def step(self) -> EpisodeStep:
        if self.current_observation is None:
            raise RuntimeError("Episode has not been started.")
        if self.terminated or self.truncated:
            raise RuntimeError("Episode already finished.")
        if self.waiting_for_human:
            raise RuntimeError("Episode is waiting for human input.")

        observation = self.current_observation
        next_turn_index = self.turn_index + 1
        remaining_s = self.timeout_s - self.elapsed_s
        if remaining_s <= 0:
            self.interrupt(
                code="episode_timeout",
                limit=self.timeout_s,
                observed=self.elapsed_s,
                unit="seconds",
            )
            raise EpisodeTimeoutError("episode deadline reached before next turn")

        result_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
        turn_cancelled = threading.Event()

        def execute() -> None:
            try:
                result = self._execute_step(observation, next_turn_index)
                if turn_cancelled.is_set():
                    self._request_environment_close()
                    return
                result_queue.put(("result", result))
            except BaseException as exc:  # noqa: BLE001 - re-raised in runner thread.
                if not turn_cancelled.is_set():
                    result_queue.put(("error", exc))

        worker = threading.Thread(
            target=execute,
            name=f"openeta-episode-turn-{next_turn_index}",
            daemon=True,
        )
        with self._worker_lock:
            self._active_worker = worker
        worker.start()
        try:
            worker.join(timeout=remaining_s)
        except BaseException:
            turn_cancelled.set()
            self._cancel_event.set()
            raise
        if worker.is_alive():
            turn_cancelled.set()
            self.interrupt(
                code="episode_timeout",
                limit=self.timeout_s,
                observed=max(self.timeout_s, self.elapsed_s),
                unit="seconds",
            )
            raise EpisodeTimeoutError(
                f"episode turn exceeded {self.timeout_s:g}-second deadline"
            )
        with self._worker_lock:
            if self._active_worker is worker:
                self._active_worker = None

        kind, payload = result_queue.get_nowait()
        if kind == "error":
            assert isinstance(payload, BaseException)
            raise payload
        assert isinstance(payload, EpisodeStep)
        episode_step = payload
        self.turn_index = next_turn_index
        action = episode_step.action
        step_result = episode_step.step_result
        self.tool_call_count += count_tool_calls(action)
        action_tokens, action_sources = action_token_usage(action)
        self.total_tokens += action_tokens
        for source, count in action_sources.items():
            self.token_usage_sources[source] = int(
                self.token_usage_sources.get(source) or 0
            ) + count
        self.terminated = step_result.terminated
        self.truncated = step_result.truncated
        self.waiting_for_human = is_agent_waiting_for_human(action)
        self.stop_reason = (
            "ask_human" if self.waiting_for_human else _stop_reason_from_step_result(step_result)
        )
        self._enforce_resource_budgets()
        self.current_observation = step_result.observation
        self.runtime.memory.record("episode_step", episode_step.to_dict())
        return episode_step

    def _execute_step(
        self,
        observation: EnvObservation,
        turn_index: int,
    ) -> EpisodeStep:
        """Compute one turn without committing mutable runner state."""

        action = self.runtime.act(
            observation,
            execution_id=self.execution_id,
            cancel_event=self._cancel_event,
        )
        if is_agent_task_complete(action):
            step_result = StepResult(
                observation=observation,
                reward=0.0,
                terminated=True,
                truncated=False,
                info={
                    "termination_source": "agent",
                    "termination_reason": "task_complete",
                    "previous_action": summarize_action(action),
                },
            )
        elif is_agent_waiting_for_human(action):
            step_result = StepResult(
                observation=observation,
                reward=0.0,
                terminated=False,
                truncated=False,
                info={
                    "pause_source": "agent",
                    "pause_reason": "ask_human",
                    "previous_action": summarize_action(action),
                },
            )
        elif is_agent_terminal_response(action):
            request = action.command.get("request", {})
            response_name = str(request.get("name") or "response")
            step_result = StepResult(
                observation=observation,
                reward=0.0,
                terminated=True,
                truncated=False,
                info={
                    "termination_source": "agent",
                    "termination_reason": (
                        "status_report" if response_name == "talk" else response_name
                    ),
                    "response_name": response_name,
                    "previous_action": summarize_action(action),
                },
            )
        else:
            step_result = self.environment.step(action)
        return EpisodeStep(
            turn_index=turn_index,
            observation=observation,
            action=action,
            step_result=step_result,
        )

    def run(
        self,
        *,
        task: str,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        timeout_s: float = DEFAULT_EPISODE_TIMEOUT_S,
        max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS,
        initial_tool_call_count: int = 0,
        initial_total_tokens: int = 0,
        initial_token_usage_sources: JsonDict | None = None,
        metadata: JsonDict | None = None,
    ) -> EpisodeResult:
        try:
            self.start(
                task=task,
                max_turns=max_turns,
                max_tool_calls=max_tool_calls,
                timeout_s=timeout_s,
                max_total_tokens=max_total_tokens,
                initial_tool_call_count=initial_tool_call_count,
                initial_total_tokens=initial_total_tokens,
                initial_token_usage_sources=initial_token_usage_sources,
                metadata=metadata,
            )
        except EpisodeTimeoutError:
            pass
        return self.continue_run()

    def continue_run(self, *, max_turns: int | None = None) -> EpisodeResult:
        steps: list[EpisodeStep] = []
        turn_budget = self.remaining_turns
        if max_turns is not None:
            turn_budget = min(turn_budget, max(0, max_turns))
        for _ in range(turn_budget):
            self._enforce_resource_budgets()
            if self.terminated or self.truncated or self.waiting_for_human:
                break
            try:
                steps.append(self.step())
            except EpisodeTimeoutError:
                break
        if (
            not self.terminated
            and not self.truncated
            and self.current_observation is not None
            and self.remaining_turns <= 0
        ):
            self.truncated = True
            self.stop_reason = "max_turns"
        result = EpisodeResult(
            task=self.task,
            session_id=self.runtime.memory.session_id,
            steps=steps,
            terminated=self.terminated,
            truncated=self.truncated,
            metadata={
                "environment": type(self.environment).__name__,
                "turn_index": self.turn_index,
                "max_turns": self.max_turns,
                "remaining_turns": self.remaining_turns,
                "budget": {
                    "max_tool_calls": self.max_tool_calls,
                    "timeout_s": self.timeout_s,
                    "max_total_tokens": self.max_total_tokens,
                },
                "usage": {
                    "tool_call_count": self.tool_call_count,
                    "total_tokens": self.total_tokens,
                    "token_usage_sources": dict(self.token_usage_sources),
                    "elapsed_s": round(self.elapsed_s, 3),
                },
                "failure_reason": dict(self.failure_reason),
                "interrupt_cleanup": dict(self.interrupt_cleanup),
                "stop_reason": self.stop_reason,
                "waiting_for_human": self.waiting_for_human,
            },
        )
        self.runtime.memory.record(
            "episode_result",
            {
                "task": result.task,
                "session_id": result.session_id,
                "num_steps": len(result.steps),
                "terminated": result.terminated,
                "truncated": result.truncated,
                "metadata": result.metadata,
            },
        )
        review = self.runtime.self_improvement_reviewer.maybe_review(
            result,
            skills=self.runtime.skills,
        )
        result.metadata["self_improvement_review"] = review
        self.runtime.memory.record(
            "self_improvement_review",
            {
                "reviewed": review.get("reviewed"),
                "trigger": review.get("trigger"),
                "proposal_count": len(review.get("proposals", [])),
                "proposals": [
                    {
                        "proposal_id": proposal.get("proposal_id"),
                        "skill_name": proposal.get("skill_name"),
                        "path": proposal.get("path"),
                    }
                    for proposal in review.get("proposals", [])
                    if isinstance(proposal, dict)
                ],
            },
        )
        return result

    @property
    def remaining_turns(self) -> int:
        return max(0, self.max_turns - self.turn_index)

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self._clock() - self.started_at_s)

    def interrupt(
        self,
        *,
        code: str = "episode_interrupted",
        limit: int | float | None = None,
        observed: int | float | None = None,
        unit: str = "",
    ) -> JsonDict:
        """Stop future runner commits and request non-blocking environment cleanup."""

        self._cancel_event.set()
        reason: JsonDict = {"code": code}
        if limit is not None:
            reason["limit"] = limit
        if observed is not None:
            reason["observed"] = observed
        if unit:
            reason["unit"] = unit
        self._set_failure(reason)
        self.interrupt_cleanup = self._request_environment_close()
        self.runtime.memory.record(
            "episode_interrupt",
            {
                "task": self.task,
                "turn_index": self.turn_index,
                "failure_reason": dict(reason),
                "cleanup": dict(self.interrupt_cleanup),
            },
        )
        return dict(self.interrupt_cleanup)

    def wait_for_idle(self, *, timeout_s: float = INTERRUPT_CLOSE_GRACE_S) -> bool:
        """Wait briefly for an abandoned synchronous turn to release runtime ownership."""

        with self._worker_lock:
            worker = self._active_worker
        if worker is None or worker is threading.current_thread():
            return True
        worker.join(timeout=max(0.0, timeout_s))
        if worker.is_alive():
            return False
        with self._worker_lock:
            if self._active_worker is worker:
                self._active_worker = None
        return True

    def _request_environment_close(self) -> JsonDict:
        close = getattr(self.environment, "close", None)
        if not callable(close):
            return {"ok": True, "skipped": True, "reason": "close_not_supported"}
        result_queue: queue.Queue[JsonDict] = queue.Queue(maxsize=1)

        def close_environment() -> None:
            try:
                result = close()
                result_queue.put(result if isinstance(result, dict) else {"ok": True})
            except Exception as exc:  # noqa: BLE001 - interruption must remain bounded.
                result_queue.put(
                    {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )

        closer = threading.Thread(
            target=close_environment,
            name="openeta-episode-interrupt-close",
            daemon=True,
        )
        closer.start()
        closer.join(timeout=INTERRUPT_CLOSE_GRACE_S)
        if closer.is_alive():
            return {"ok": True, "pending": True, "requested": True}
        return result_queue.get_nowait()

    def _enforce_resource_budgets(self) -> None:
        if self.failure_reason:
            return
        reason: JsonDict = {}
        if self.elapsed_s >= self.timeout_s:
            reason = {
                "code": "episode_timeout",
                "limit": self.timeout_s,
                "observed": self.elapsed_s,
                "unit": "seconds",
            }
        elif self.total_tokens > self.max_total_tokens:
            reason = {
                "code": "token_limit_exceeded",
                "limit": self.max_total_tokens,
                "observed": self.total_tokens,
                "unit": "tokens",
            }
        elif self.tool_call_count > self.max_tool_calls:
            reason = {
                "code": "tool_call_limit_exceeded",
                "limit": self.max_tool_calls,
                "observed": self.tool_call_count,
                "unit": "tool_calls",
            }
        if not reason:
            return
        if reason.get("code") == "episode_timeout":
            self.interrupt(
                code="episode_timeout",
                limit=reason.get("limit"),
                observed=reason.get("observed"),
                unit=str(reason.get("unit") or ""),
            )
            return
        self._set_failure(reason)

    def _set_failure(self, reason: JsonDict) -> None:
        self.failure_reason = reason
        self.stop_reason = str(reason["code"])
        self.terminated = False
        self.truncated = True
        self.waiting_for_human = False

    def resume_after_human(self) -> None:
        """Allow the next turn after the CLI records a human answer."""

        if self.waiting_for_human:
            self.waiting_for_human = False
            self.stop_reason = ""


class ToolFeedbackEpisodeEnvironment:
    """Turn boundary that feeds executed tool summaries back to the planner.

    Physical state changes still happen inside bound tool handlers (for example
    simulator MCP proxies). This lightweight environment supplies the next
    observation envelope when no simulator-owned episode environment is active.
    """

    def __init__(self, *, max_steps: int | None = None) -> None:
        self.max_steps = max_steps
        self.task = ""
        self.step_idx = 0
        self.last_action_summary: JsonDict = {}

    def reset(self, *, task: str, metadata: JsonDict | None = None) -> EnvObservation:
        self.task = task
        self.step_idx = 0
        self.last_action_summary = {}
        return self._observation(metadata=dict(metadata or {}))

    def step(self, action: EnvAction) -> StepResult:
        self.step_idx += 1
        self.last_action_summary = summarize_action(action)
        terminated = self.max_steps is not None and self.step_idx >= self.max_steps
        observation = self._observation(
            metadata={
                "previous_action": self.last_action_summary,
                "source": type(self).__name__,
            }
        )
        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=terminated,
            truncated=False,
            info={
                "step_idx": self.step_idx,
                "previous_action": self.last_action_summary,
                "environment": type(self).__name__,
            },
        )

    def _observation(self, *, metadata: JsonDict | None = None) -> EnvObservation:
        merged_metadata = {
            "step_idx": self.step_idx,
            "source": type(self).__name__,
            **dict(metadata or {}),
        }
        return EnvObservation(
            task=self.task,
            cameras=[
                CameraFrame(
                    frame_id="front",
                    rgb=[[[0, 0, 0]]],
                    depth=[[1.0]],
                )
            ],
            robot=RobotState(
                end_effector_pose={"xyz": [0.0, 0.0, 0.5]},
                gripper_state={"open": 1.0},
            ),
            objects=[{"name": "cube", "position": [0.2, 0.0, 0.0]}],
            metadata=merged_metadata,
        )


class DummyEpisodeEnvironment(ToolFeedbackEpisodeEnvironment):
    """Backward-compatible deterministic test environment."""


def count_tool_calls(action: EnvAction) -> int:
    """Count concrete tool attempts represented by one planner action."""

    command = action.command if isinstance(action.command, dict) else {}
    calls = command.get("tool_calls")
    if isinstance(calls, list) and calls:
        return sum(isinstance(call, dict) for call in calls)
    request = command.get("request")
    if not isinstance(request, dict) or request.get("kind") != "tool_call":
        return 0
    if isinstance(command.get("skill_call"), dict):
        return 0
    return 1


def action_token_usage(action: EnvAction) -> tuple[int, dict[str, int]]:
    """Return charged model tokens and accounting sources for one action."""

    command = action.command if isinstance(action.command, dict) else {}
    metadata = command.get("metadata")
    if not isinstance(metadata, dict):
        return 0, {}
    planner_metadata = metadata.get("planner_metadata")
    if not isinstance(planner_metadata, dict):
        return 0, {}
    usage = planner_metadata.get("backend_usage")
    if not isinstance(usage, dict):
        backend_details = planner_metadata.get("backend_details")
        usage = backend_details.get("usage") if isinstance(backend_details, dict) else {}
    if not isinstance(usage, dict):
        return 0, {}
    sources = planner_metadata.get("backend_usage_sources")
    if not isinstance(sources, dict):
        backend_details = planner_metadata.get("backend_details")
        source = (
            str(backend_details.get("usage_source") or "unknown")
            if isinstance(backend_details, dict)
            else "unknown"
        )
        sources = {source: 1}
    normalized_sources = {
        str(key): _non_negative_int(value)
        for key, value in sources.items()
        if _non_negative_int(value) > 0
    }
    return _non_negative_int(usage.get("total_tokens")), normalized_sources


def action_total_tokens(action: EnvAction) -> int:
    """Backward-compatible scalar accessor for charged model tokens."""

    return action_token_usage(action)[0]


def summarize_action(action: EnvAction) -> JsonDict:
    request = action.command.get("request", {})
    return {
        "action_type": action.action_type,
        "request_kind": request.get("kind"),
        "request_name": request.get("name"),
        "status": action.command.get("status"),
        "tool_calls": [
            _summarize_tool_call(call)
            for call in action.command.get("tool_calls", [])
            if isinstance(call, dict)
        ],
        "skill_call": _summarize_skill_call(action.command.get("skill_call")),
        "metadata": summarize_event_payload({"metadata": action.metadata}).get("metadata", {}),
    }


def _summarize_skill_call(skill_call: object) -> object:
    if not isinstance(skill_call, dict):
        return skill_call
    result = skill_call.get("result")
    return {
        "name": skill_call.get("name"),
        "status": skill_call.get("status"),
        "result": {
            "success": result.get("success"),
            "content": _truncate_summary_text(result.get("content")),
        }
        if isinstance(result, dict)
        else None,
    }


def _summarize_tool_call(call: JsonDict) -> JsonDict:
    result = call.get("result")
    if not isinstance(result, dict):
        return {
            "name": call.get("name"),
            "status": call.get("status"),
            "result": None,
        }

    details = result.get("details")
    if not isinstance(details, dict):
        details = {}
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        outputs = {}
    artifacts = details.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
    diagnostics = details.get("diagnostics")
    if not isinstance(diagnostics, list):
        diagnostics = []

    response = outputs.get("response")
    response_path = response.get("response_path") if isinstance(response, dict) else None
    return {
        "name": call.get("name"),
        "status": call.get("status"),
        "result": {
            "success": result.get("success"),
            "content": _truncate_summary_text(result.get("content")),
            "result_type": details.get("result_type"),
            "output_keys": sorted(str(key) for key in outputs)[:12],
            "artifact_count": len(artifacts),
            "response_path": response_path,
            "diagnostic_codes": [
                diagnostic.get("code")
                for diagnostic in diagnostics[:5]
                if isinstance(diagnostic, dict)
            ],
        },
    }


def _truncate_summary_text(value: object, *, max_chars: int = 300) -> object:
    if not isinstance(value, str):
        return value
    return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def is_agent_task_complete(action: EnvAction) -> bool:
    """Return whether the planner used the current schema to end the episode."""

    request = action.command.get("request", {})
    if not isinstance(request, dict):
        return False
    kind = request.get("kind")
    if kind != "response":
        return False
    parameters = request.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    name = str(request.get("name", "")).strip().lower()
    explicit_done = bool(
        parameters.get("task_complete")
        or parameters.get("done")
        or parameters.get("success") is True
    )
    return name in {"task_complete", "done", "finish"} or explicit_done


def is_agent_waiting_for_human(action: EnvAction) -> bool:
    """Return whether the planner explicitly paused for human input."""

    request = action.command.get("request", {})
    if not isinstance(request, dict):
        return False
    return request.get("kind") == "response" and request.get("name") == "ask_human"


def is_agent_terminal_response(action: EnvAction) -> bool:
    """Return whether a non-interactive response should close this episode turn."""

    request = action.command.get("request", {})
    if not isinstance(request, dict):
        return False
    if request.get("kind") != "response":
        return False
    return str(request.get("name", "")).strip().lower() == "talk"


def summarize_step_result(step_result: StepResult) -> JsonDict:
    return {
        "reward": step_result.reward,
        "terminated": step_result.terminated,
        "truncated": step_result.truncated,
        "info": summarize_event_payload({"metadata": step_result.info}).get("metadata", {}),
        "observation": summarize_observation(step_result.observation),
    }


def _stop_reason_from_step_result(step_result: StepResult) -> str:
    if not step_result.terminated and not step_result.truncated:
        return ""
    reason = step_result.info.get("termination_reason") or step_result.info.get("truncation_reason")
    if reason:
        return str(reason)
    if step_result.terminated:
        return str(step_result.info.get("termination_source", "environment"))
    return str(step_result.info.get("truncation_source", "environment"))
