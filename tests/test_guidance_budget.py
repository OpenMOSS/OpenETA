"""Guidance shares the episode deadline without late memory commits."""

from copy import deepcopy
import threading

import pytest

from agent.backends.planner import CallablePlannerBackend, PlannerBackendResult, StaticPlannerBackend
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime, RuntimeExecutionCancelled
from agent.runtime.supervision import BackendGuidanceResolver, InteractionResolution


ASK = {"kind": "response", "name": "ask_human", "parameters": {"question": "Which cube?"}}
DONE = {"kind": "response", "name": "talk", "parameters": {"message": "Guidance received."}}


def _runner(resolver, **kwargs):
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(StaticPlannerBackend([ASK, DONE])))
    return OpenEtaEpisodeRunner(
        runtime=runtime, environment=DummyEpisodeEnvironment(),
        interaction_resolver=resolver, **kwargs,
    )


class BlockingResolver:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def resolve(self, *, question, context):
        self.started.set()
        assert self.release.wait(3), "test must release the resolver"
        # Detached inputs cannot mutate Host memory, even after cancellation.
        context["memory"]["metadata"]["late_resolver_write"] = True
        return InteractionResolution(True, answer="late answer", details={"usage": {"total_tokens": 99}})


def test_episode_guidance_passes_exact_host_session_for_lineage():
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(payload={"decision": "answer", "answer": "fixture cube",
                                              "reason": "fixture"})

    runner = _runner(BackendGuidanceResolver(CallablePlannerBackend(decide)))
    result = runner.run(task="choose a cube", max_turns=2)
    assert len(requests) == 1
    assert requests[0].tool_context["request_lineage"]["parent_session_id"] == result.session_id


def test_guidance_timeout_returns_completed_step_and_rejects_late_result():
    resolver = BlockingResolver()
    runner = _runner(resolver)
    results, errors = [], []
    review_calls = []

    class Reviewer:
        def maybe_review(self, result, *, skills):
            review_calls.append(result)
            raise AssertionError("An expired episode must not start new postprocessing")

    runner.runtime.self_improvement_reviewer = Reviewer()

    def run():
        try:
            results.append(runner.run(task="choose a cube", timeout_s=0.2, max_turns=3))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert resolver.started.wait(1)
        thread.join(0.9)
        assert not thread.is_alive(), "guidance must not extend the episode to its own provider timeout"
        assert not errors
        result = results[0]
        assert len(result.steps) == 1  # The planner's ask_human action already completed.
        assert result.truncated is True
        assert result.metadata["failure_reason"]["code"] == "episode_timeout"
        assert result.metadata["waiting_for_human"] is False
        assert result.metadata["stop_reason"] == "episode_timeout"
        assert result.metadata["usage"]["total_tokens"] == 0
        assert result.metadata["assistance"]["guidance_intervention_count"] == 0
        assert not review_calls
        assert result.metadata["self_improvement_review"]["trigger"]["reason"] == (
            "episode_budget_or_interrupt_stop"
        )
        assert runner.wait_for_idle(timeout_s=0) is False
        execution_id = runner.execution_id
        with pytest.raises(RuntimeError, match="still running"):
            runner.start(task="replacement", max_turns=1)
        assert runner.execution_id == execution_id
        events = deepcopy(runner.runtime.memory.events)
        metadata = deepcopy(runner.runtime.memory.metadata)
        resolver.release.set()
        assert runner.wait_for_idle(timeout_s=1)
        assert runner.runtime.memory.events == events
        assert runner.runtime.memory.metadata == metadata
        assert runner.runtime.memory.latest_guidance_interaction() is None
        assert any(event.event_type == "guidance_resolution_timed_out" for event in events)
    finally:
        resolver.release.set()
        thread.join(2)
        runner.wait_for_idle(timeout_s=1)


def _backend_resolver(usage, *, decision="answer"):
    return BackendGuidanceResolver(CallablePlannerBackend(lambda _request: PlannerBackendResult(
        payload={"decision": decision, "answer": "the red cube", "reason": "task instruction"},
        provider="fixture", model="fixture",
        details={"usage": usage, "usage_source": "provider"},
    )))


@pytest.mark.parametrize("decision", ["answer", "abstain"])
def test_guidance_usage_is_charged_even_when_it_abstains(decision):
    runner = _runner(_backend_resolver({"total_tokens": 12}, decision=decision))
    result = runner.run(task="choose a cube", max_turns=3, max_total_tokens=10)
    assert len(result.steps) == 1
    assert result.metadata["usage"]["total_tokens"] == 12
    assert result.metadata["usage"]["token_usage_sources"] == {"guidance:provider": 1}
    assert result.metadata["failure_reason"]["code"] == "token_limit_exceeded"
    assert result.metadata["waiting_for_human"] is False


@pytest.mark.parametrize(("usage", "expected"), [
    ({"total_tokens": 0}, 0),
    ({"prompt_tokens": 3, "completion_tokens": 4}, 7),
    ({"total_tokens": True}, 0),
    ({"total_tokens": float("inf")}, 0),
    ({"total_tokens": float("nan")}, 0),
    ({"total_tokens": -1}, 0),
    ({"total_tokens": "7"}, 0),
    ({"prompt_tokens": 3}, 0),
])
def test_guidance_usage_requires_valid_receipt_counts(usage, expected):
    runner = _runner(_backend_resolver(usage))
    result = runner.run(task="choose a cube", max_turns=3)
    assert result.metadata["usage"]["total_tokens"] == expected
    assert result.metadata["assistance"]["guidance_intervention_count"] == 1


def test_provider_usage_is_preserved_without_aliasing_the_backend_result():
    usage = {"total_tokens": 12, "completion_tokens_details": {"reasoning_tokens": 3}}
    resolution = _backend_resolver(usage).resolve(question="Which?", context={})
    assert resolution.details["usage"] == usage
    assert resolution.details["usage_source"] == "provider"
    usage["completion_tokens_details"]["reasoning_tokens"] = 99
    assert resolution.details["usage"]["completion_tokens_details"]["reasoning_tokens"] == 3


def test_guidance_result_arriving_at_deadline_is_not_committed():
    clock_value = [0.0]

    class Resolver:
        def resolve(self, *, question, context):
            assert context["turn_index"] == 1
            clock_value[0] = 10.0
            return InteractionResolution(True, answer="too late")

    runner = _runner(Resolver(), clock=lambda: clock_value[0])
    result = runner.run(task="choose a cube", max_turns=3, timeout_s=10.0)
    assert len(result.steps) == 1
    assert result.metadata["failure_reason"]["code"] == "episode_timeout"
    assert runner.runtime.memory.latest_guidance_interaction() is None


def test_context_preparation_expiring_budget_does_not_start_provider(monkeypatch):
    from agent.runtime import episode

    clock_value = [0.0]
    calls = []

    class Resolver:
        def resolve(self, *, question, context):
            calls.append(question)
            return InteractionResolution(True, answer="red cube")

    def slow_copy(value):
        copied = deepcopy(value)
        if isinstance(value, dict) and "memory" in value:
            clock_value[0] = 10.0
        return copied

    monkeypatch.setattr(episode, "deepcopy", slow_copy)
    runner = _runner(Resolver(), clock=lambda: clock_value[0])
    result = runner.run(task="choose", max_turns=3, timeout_s=10.0)
    assert not calls
    assert result.metadata["failure_reason"]["code"] == "episode_timeout"
    assert runner.runtime.memory.latest_guidance_interaction() is None


def test_completed_guidance_does_not_commit_to_a_replaced_session():
    resolver = BlockingResolver()
    runner = _runner(resolver)
    runner.start(task="original", max_turns=3, timeout_s=10.0)
    errors = []

    def step():
        try:
            runner.step()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=step)
    thread.start()
    try:
        assert resolver.started.wait(1)
        runner.runtime.start_session(task="new session")
        before = deepcopy(runner.runtime.memory.events)
        resolver.release.set()
        thread.join(1)
        assert not thread.is_alive()
        assert len(errors) == 1 and isinstance(errors[0], RuntimeExecutionCancelled)
        assert runner.runtime.memory.events == before
        assert runner.runtime.memory.latest_guidance_interaction() is None
    finally:
        resolver.release.set()
        thread.join(2)
        runner.wait_for_idle(timeout_s=1)


def test_guidance_exception_preserves_step_and_falls_back_to_human():
    class Resolver:
        def resolve(self, *, question, context):
            raise ValueError("malformed guidance")

    runner = _runner(Resolver())
    result = runner.run(task="choose a cube", max_turns=3)
    assert len(result.steps) == 1
    assert result.metadata["waiting_for_human"] is True
    assert not result.metadata["failure_reason"]
    assert any(event.event_type == "guidance_resolution_failed" for event in runner.runtime.memory.events)


def test_explicit_interrupt_stops_waiting_without_accepting_late_guidance():
    resolver = BlockingResolver()
    runner = _runner(resolver)
    results = []
    thread = threading.Thread(target=lambda: results.append(
        runner.run(task="choose", max_turns=3, timeout_s=10.0)
    ))
    thread.start()
    try:
        assert resolver.started.wait(1)
        runner.interrupt(code="episode_interrupted")
        thread.join(0.7)
        assert not thread.is_alive()
        assert results[0].metadata["failure_reason"]["code"] == "episode_interrupted"
        assert results[0].metadata["waiting_for_human"] is False
        assert runner.wait_for_idle(timeout_s=0) is False
        before = deepcopy(runner.runtime.memory.events)
        resolver.release.set()
        assert runner.wait_for_idle(timeout_s=1)
        assert runner.runtime.memory.events == before
        assert runner.runtime.memory.latest_guidance_interaction() is None
    finally:
        resolver.release.set()
        thread.join(2)
        runner.wait_for_idle(timeout_s=1)


def test_expired_action_budget_does_not_dispatch_guidance():
    calls = []

    class Resolver:
        def resolve(self, *, question, context):
            calls.append(question)
            return InteractionResolution(True, answer="red cube")

    runner = _runner(Resolver())
    runner.runtime.planner = ToolCallingPlanner(CallablePlannerBackend(
        lambda _request: PlannerBackendResult(payload=ASK, details={"usage": {"total_tokens": 11}})
    ))
    result = runner.run(task="choose", max_turns=3, max_total_tokens=10)
    assert not calls
    assert len(result.steps) == 1
    assert result.metadata["failure_reason"]["code"] == "token_limit_exceeded"


@pytest.mark.parametrize("response", [ValueError("review failed"), None, {"proposals": "malformed"}])
def test_post_episode_review_failure_cannot_erase_the_episode_result(response):
    class Reviewer:
        def maybe_review(self, result, *, skills):
            if isinstance(response, Exception):
                raise response
            return response

    runner = _runner(None)
    runner.runtime.planner = ToolCallingPlanner(StaticPlannerBackend(DONE))
    runner.runtime.self_improvement_reviewer = Reviewer()
    result = runner.run(task="report", max_turns=1)
    assert result.terminated is True
    assert len(result.steps) == 1
    assert not result.metadata["failure_reason"]
    review = result.metadata["self_improvement_review"]
    assert review["reviewed"] is False
    assert review["error"]["type"] in {"ValueError", "TypeError"}
    assert review["partial_effects_possible"] is True
    event = runner.runtime.memory.events[-1]
    assert event.event_type == "self_improvement_review"
    assert event.payload["error"] == review["error"]
