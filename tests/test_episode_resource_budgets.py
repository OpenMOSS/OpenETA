from __future__ import annotations

import threading
import time

from agent.backends.planner import (
    PlannerBackend,
    PlannerBackendRequest,
    PlannerBackendResult,
    StaticPlannerBackend,
)
from agent.runtime.actions import PipelineStatus
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.parallel import (
    ParallelEpisodeHarness,
    ParallelEpisodeSpec,
    ParallelEpisodeWorker,
    classify_episode_result,
    episode_failure_error,
)
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.registry import ToolResult, build_default_tool_registry


class UsageBackend(PlannerBackend):
    def __init__(self, usages: list[int]) -> None:
        self.usages = usages
        self.index = 0

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        del request
        usage = self.usages[min(self.index, len(self.usages) - 1)]
        self.index += 1
        return PlannerBackendResult(
            payload={
                "kind": "tool_call",
                "name": "scene_detector",
                "parameters": {"image": "front"},
            },
            status=PipelineStatus.PLANNED,
            details={"usage": {"total_tokens": usage}},
        )


class RetryUsageBackend(PlannerBackend):
    def __init__(self) -> None:
        self.index = 0

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        del request
        self.index += 1
        name = "not_a_registered_tool" if self.index == 1 else "scene_detector"
        return PlannerBackendResult(
            payload={"kind": "tool_call", "name": name, "parameters": {}},
            status=PipelineStatus.PLANNED,
            details={"usage": {"total_tokens": self.index * 10}},
        )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value


class ClosableEnvironment(DummyEpisodeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.close_count = 0
        self.closed = False

    def close(self):
        if self.closed:
            return {"ok": True, "skipped": True}
        self.closed = True
        self.close_count += 1
        return {"ok": True}


class LateHandleResetEnvironment(DummyEpisodeEnvironment):
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        super().__init__()
        self.started = started
        self.release = release
        self.has_handle = False
        self.close_count = 0

    def reset(self, *, task, metadata=None):
        self.started.set()
        self.release.wait(timeout=1.0)
        self.has_handle = True
        return super().reset(task=task, metadata=metadata)

    def close(self):
        if not self.has_handle:
            return {"ok": True, "skipped": True}
        self.has_handle = False
        self.close_count += 1
        return {"ok": True}


def _runtime(backend: PlannerBackend, *, handler=None) -> OpenEtaAgentRuntime:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "scene_detector",
        handler or (lambda context: ToolResult(True, content="objects detected")),
    )
    return OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(backend),
        tools=tools,
    )


def test_episode_fails_when_tool_calls_exceed_budget() -> None:
    runtime = _runtime(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "scene_detector",
                "parameters": {"image": "front"},
            }
        )
    )
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())

    result = runner.run(task="find cube", max_turns=10, max_tool_calls=2)

    assert len(result.steps) == 3
    assert result.metadata["usage"]["tool_call_count"] == 3
    assert result.metadata["failure_reason"] == {
        "code": "tool_call_limit_exceeded",
        "limit": 2,
        "observed": 3,
        "unit": "tool_calls",
    }
    assert classify_episode_result(result) == "fail"
    assert episode_failure_error(result)["code"] == "tool_call_limit_exceeded"


def test_episode_fails_when_cumulative_model_tokens_exceed_budget() -> None:
    runner = OpenEtaEpisodeRunner(
        runtime=_runtime(UsageBackend([3_000_000, 3_000_001])),
        environment=DummyEpisodeEnvironment(),
    )

    result = runner.run(
        task="find cube",
        max_turns=10,
        max_tool_calls=10,
        max_total_tokens=5_000_000,
    )

    assert len(result.steps) == 2
    assert result.metadata["usage"]["total_tokens"] == 6_000_001
    assert result.metadata["failure_reason"]["code"] == "token_limit_exceeded"
    assert result.metadata["failure_reason"]["limit"] == 5_000_000


def test_episode_fails_when_wall_clock_deadline_is_reached() -> None:
    clock = FakeClock()

    def advance_clock(context):
        del context
        clock.value = 600.0
        return ToolResult(True, content="objects detected")

    runner = OpenEtaEpisodeRunner(
        runtime=_runtime(
            StaticPlannerBackend(
                {"kind": "tool_call", "name": "scene_detector", "parameters": {}}
            ),
            handler=advance_clock,
        ),
        environment=DummyEpisodeEnvironment(),
        clock=clock,
    )

    result = runner.run(task="find cube", timeout_s=600.0)

    assert len(result.steps) == 1
    assert result.metadata["failure_reason"] == {
        "code": "episode_timeout",
        "limit": 600.0,
        "observed": 600.0,
        "unit": "seconds",
    }


def test_runner_actively_interrupts_blocked_turn_and_closes_environment() -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking_handler(context):
        del context
        started.set()
        release.wait(timeout=1.0)
        return ToolResult(True, content="late result")

    environment = ClosableEnvironment()
    runner = OpenEtaEpisodeRunner(
        runtime=_runtime(
            StaticPlannerBackend(
                {"kind": "tool_call", "name": "scene_detector", "parameters": {}}
            ),
            handler=blocking_handler,
        ),
        environment=environment,
    )

    before = time.monotonic()
    result = runner.run(task="find cube", timeout_s=0.05)
    duration = time.monotonic() - before
    release.set()
    time.sleep(0.02)

    assert started.is_set()
    assert duration < 0.5
    assert result.steps == []
    assert result.metadata["failure_reason"]["code"] == "episode_timeout"
    assert result.metadata["interrupt_cleanup"] == {"ok": True}
    assert environment.close_count == 1
    assert runner.turn_index == 0
    assert runner.wait_for_idle(timeout_s=0.2) is True


def test_cancelled_tool_result_is_fenced_before_next_command() -> None:
    started = threading.Event()
    release = threading.Event()
    cancel = threading.Event()
    tools = build_default_tool_registry()
    events = []

    def blocking_handler(context):
        del context
        started.set()
        release.wait(timeout=1.0)
        return ToolResult(True, content="late result")

    tools.bind_handler("scene_detector", blocking_handler)
    tools.bind_handler("observe", lambda context: ToolResult(True, content="fresh result"))
    tools.add_listener(events.append)
    old_result = {}

    def run_old_execution() -> None:
        with tools.execution_scope(
            {"execution_id": "old-execution", "_cancel_event": cancel}
        ):
            old_result["value"] = tools.call("scene_detector", {})

    worker = threading.Thread(target=run_old_execution)
    worker.start()
    assert started.wait(timeout=0.2)
    cancel.set()
    worker.join(timeout=0.3)

    assert worker.is_alive() is False
    assert old_result["value"].success is False
    assert old_result["value"].details["diagnostics"][0]["code"] == "execution_cancelled"
    fresh = tools.call("observe", {})
    assert fresh.success is True
    assert fresh.content == "fresh result"

    release.set()
    time.sleep(0.05)
    old_ends = [
        event
        for event in events
        if event.get("phase") == "end" and event.get("name") == "scene_detector"
    ]
    assert len(old_ends) == 1
    assert old_ends[0]["metadata"]["execution_id"] == "old-execution"


def test_runner_interrupts_blocked_reset_and_closes_late_handle() -> None:
    started = threading.Event()
    release = threading.Event()
    environment = LateHandleResetEnvironment(started, release)
    runner = OpenEtaEpisodeRunner(
        runtime=_runtime(StaticPlannerBackend({"kind": "response", "name": "talk"})),
        environment=environment,
    )

    before = time.monotonic()
    result = runner.run(task="wait for reset", timeout_s=0.05)
    duration = time.monotonic() - before
    release.set()
    deadline = time.monotonic() + 0.5
    while environment.close_count == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert started.is_set()
    assert duration < 0.5
    assert result.steps == []
    assert result.metadata["failure_reason"]["code"] == "episode_timeout"
    assert result.metadata["interrupt_cleanup"] == {"ok": True, "skipped": True}
    assert environment.close_count == 1
    assert environment.has_handle is False


def test_planner_token_usage_includes_schema_validation_retries() -> None:
    runtime = _runtime(RetryUsageBackend())
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())

    result = runner.run(task="find cube", max_turns=1)

    planner_metadata = result.steps[0].action.command["metadata"]["planner_metadata"]
    assert planner_metadata["backend_usage"]["total_tokens"] == 30
    assert result.metadata["usage"]["total_tokens"] == 30


def test_parallel_outcome_exposes_structured_resource_failure() -> None:
    spec = ParallelEpisodeSpec(
        episode_id="budget-fail",
        task="find cube",
        env_id="openeta/test-v0",
        max_turns=10,
        max_tool_calls=2,
    )

    result = ParallelEpisodeHarness(
        lambda spec, batch_id: ParallelEpisodeWorker(
            runner=OpenEtaEpisodeRunner(
                runtime=_runtime(
                    StaticPlannerBackend(
                        {
                            "kind": "tool_call",
                            "name": "scene_detector",
                            "parameters": {},
                        }
                    )
                ),
                environment=DummyEpisodeEnvironment(),
            ),
            close=lambda: {"ok": True},
        ),
        concurrency=1,
    ).run([spec])

    outcome = result.to_dict()["outcomes"][0]
    assert outcome["status"] == "fail"
    assert outcome["error"] == {
        "type": "EpisodeResourceLimit",
        "code": "tool_call_limit_exceeded",
        "limit": 2,
        "observed": 3,
        "unit": "tool_calls",
    }
    assert outcome["limits"]["max_tool_calls"] == 2
