"""Executable tool admission is bounded before handlers, including batches."""

from concurrent.futures import ThreadPoolExecutor
import json
import threading

import pytest

from agent.backends.planner import StaticPlannerBackend
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.call_budget import ToolCallBudget
from agent.tools.registry import (
    ToolRegistry, ToolResult, ToolSpec, build_default_tool_registry,
    cooperative_cancellation_when,
)


def _registry():
    tools = ToolRegistry()
    calls = []
    tools.register(ToolSpec(name="read", category="test", description="Read"),
                   lambda context: calls.append(context.name) or ToolResult(True))
    return tools, calls


def _episode(payload, *, limit, carried=0):
    tools = build_default_tool_registry()
    calls = []
    tools.bind_handler("get_memory", lambda context: calls.append(context.name) or ToolResult(True))
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(StaticPlannerBackend(payload)), tools=tools)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    result = runner.run(task="inspect", max_turns=5, max_tool_calls=limit, initial_tool_call_count=carried)
    return result, calls


def test_batch_cannot_dispatch_beyond_remaining_quota():
    result, calls = _episode({"kind": "tool_call", "name": "tool_batch", "parameters": {
        "calls": [{"name": "get_memory", "parameters": {}} for _ in range(5)],
    }}, limit=2)
    assert calls == ["get_memory", "get_memory"]
    assert len(result.steps) == 1
    # Keep historical request/attempt accounting; it is not dispatch evidence.
    assert result.metadata["usage"]["tool_call_count"] == 5
    assert result.metadata["usage"]["tool_admission"]["admitted_this_run"] == 2
    assert result.metadata["usage"]["tool_admission"]["denied_this_run"] == 3
    assert result.metadata["failure_reason"]["code"] == "tool_call_limit_exceeded"
    entries = result.steps[0].action.command["tool_calls"]
    assert [entry["result"]["success"] for entry in entries] == [True, True, False, False, False]
    assert entries[2]["result"]["details"]["diagnostics"][0]["code"] == "tool_call_budget_exhausted"


def test_exact_quota_still_allows_terminal_response():
    result, calls = _episode([
        {"kind": "tool_call", "name": "get_memory", "parameters": {}},
        {"kind": "response", "name": "talk", "parameters": {"message": "Finished inspecting."}},
    ], limit=1)
    assert result.terminated is True
    assert not result.metadata["failure_reason"]
    assert len(result.steps) == 2 and calls == ["get_memory"]
    assert result.metadata["usage"]["tool_admission"]["denied_this_run"] == 0


def test_carried_attempt_usage_is_conservatively_charged():
    result, calls = _episode(
        {"kind": "tool_call", "name": "get_memory", "parameters": {}}, limit=6, carried=5,
    )
    assert calls == ["get_memory"]
    assert result.metadata["usage"]["tool_call_count"] == 7
    assert result.metadata["usage"]["tool_admission"] == {
        "limit": 6, "carried_usage": 5, "admitted_this_run": 1,
        "denied_this_run": 1, "remaining": 0,
    }


def test_concurrent_callers_share_atomic_quota():
    tools, calls = _registry()
    budget = ToolCallBudget(3)

    def call(_index):
        return tools.call("read", metadata={"_tool_call_budget": budget}).success

    with ThreadPoolExecutor(max_workers=8) as executor:
        successes = list(executor.map(call, range(20)))
    assert sum(successes) == len(calls) == 3
    assert budget.snapshot()["denied_this_run"] == 17


@pytest.mark.parametrize("replacement", [None, {}, "unlimited", ToolCallBudget(100)])
def test_nested_scope_and_caller_metadata_cannot_replace_host_quota(replacement):
    tools, calls = _registry()
    budget = ToolCallBudget(1)
    with tools.execution_scope({"_tool_call_budget": budget}):
        assert tools.call("read").success
        with tools.execution_scope({"_tool_call_budget": replacement}):
            denied = tools.call("read", metadata={"_tool_call_budget": replacement})
    assert not denied.success
    assert calls == ["read"]
    assert budget.snapshot()["denied_this_run"] == 1


@pytest.mark.parametrize("worker_thread", [False, True])
def test_nested_registry_calls_in_handler_inherit_quota(worker_thread):
    tools, calls = _registry()
    nested = []

    def outer(_context):
        nested.extend([tools.call("read"), tools.call("read")])
        return ToolResult(True)

    tools.register(ToolSpec(name="outer", category="test", description="Nested calls"), outer)
    budget = ToolCallBudget(2)
    scope = {"_tool_call_budget": budget}
    if worker_thread:
        scope["_cancel_event"] = threading.Event()
    with tools.execution_scope(scope):
        assert tools.call("outer").success
    assert [result.success for result in nested] == [True, False]
    assert calls == ["read"]
    assert budget.snapshot()["admitted_this_run"] == 2


def test_admission_precedes_expensive_world_authorization():
    tools, _ = _registry()
    calls, reviews = [], []
    tools.register(ToolSpec(name="move", category="test", description="Move", effect="world_mutating"),
                   lambda context: calls.append(context.name) or ToolResult(True))
    tools.set_execution_gate(lambda context: reviews.append(context.name) or True)
    budget = ToolCallBudget(1)
    with tools.execution_scope({"_tool_call_budget": budget}):
        assert tools.call("read").success
        assert not tools.call("move").success
    assert not calls and not reviews


def test_failed_handler_is_not_refunded():
    tools, _ = _registry()
    calls = []

    def fail(_context):
        calls.append(True)
        raise RuntimeError("failed after entering handler")

    tools.bind_handler("read", fail, replace=True)
    budget = ToolCallBudget(1)
    with tools.execution_scope({"_tool_call_budget": budget}):
        assert not tools.call("read").success
        assert not tools.call("read").success
    assert calls == [True]
    assert budget.snapshot()["admitted_this_run"] == 1


def test_cooperative_cancellation_marker_survives_scoped_wrapper():
    tools, _ = _registry()
    owner = threading.current_thread()

    @cooperative_cancellation_when(lambda _context: True)
    def cooperative(_context):
        assert threading.current_thread() is owner
        return ToolResult(True)

    tools.bind_handler("read", cooperative, replace=True)
    with tools.execution_scope({"_tool_call_budget": ToolCallBudget(1), "_cancel_event": threading.Event()}):
        assert tools.call("read").success


def test_private_quota_object_does_not_leak_into_events_or_results():
    tools, _ = _registry()
    events = []
    tools.add_listener(events.append)
    with tools.execution_scope({"_tool_call_budget": ToolCallBudget(1)}):
        first, denied = tools.call("read"), tools.call("read")
    rendered = json.dumps([events, first.details, denied.details])
    assert "_tool_call_budget" not in rendered


def test_cancelled_call_is_not_admitted():
    tools, calls = _registry()
    budget = ToolCallBudget(1)
    cancel = threading.Event()
    cancel.set()
    with tools.execution_scope({"_tool_call_budget": budget, "_cancel_event": cancel}):
        assert not tools.call("read").success
    assert not calls
    assert budget.snapshot()["admitted_this_run"] == 0


def test_cancellation_during_authorization_never_enters_handler():
    tools = ToolRegistry()
    calls = []
    cancel = threading.Event()
    tools.register(ToolSpec(name="move", category="test", description="Move", effect="world_mutating"),
                   lambda _context: calls.append(True) or ToolResult(True))

    def authorize(_context):
        cancel.set()
        return True

    tools.set_execution_gate(authorize)
    budget = ToolCallBudget(1)
    with tools.execution_scope({"_tool_call_budget": budget, "_cancel_event": cancel}):
        assert not tools.call("move").success
    assert not calls
    assert budget.snapshot()["admitted_this_run"] == 1


def test_nested_admissions_survive_pause_resume_without_free_quota():
    tools = build_default_tool_registry()
    inner_calls = []
    tools.register(ToolSpec(name="inner", category="test", description="Read"),
                   lambda _context: inner_calls.append(True) or ToolResult(True))

    def outer(_context):
        assert tools.call("inner").success
        assert tools.call("inner").success
        return ToolResult(True)

    tools.bind_handler("get_memory", outer)
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(StaticPlannerBackend([
        {"kind": "tool_call", "name": "get_memory", "parameters": {}},
        {"kind": "response", "name": "ask_human", "parameters": {"question": "Continue?"}},
    ])), tools=tools)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    result = runner.run(task="inspect", max_turns=3, max_tool_calls=3)
    assert result.metadata["waiting_for_human"] is True
    assert result.metadata["usage"]["tool_call_count"] == 3
    assert result.metadata["usage"]["tool_admission"]["admitted_this_run"] == 3
    assert len(inner_calls) == 2
    resumed, calls = _episode(
        {"kind": "tool_call", "name": "get_memory", "parameters": {}},
        limit=3, carried=result.metadata["usage"]["tool_call_count"],
    )
    assert not calls
    assert resumed.metadata["failure_reason"]["code"] == "tool_call_limit_exceeded"


def test_exhausted_nested_attempt_stops_episode_even_if_outer_returns_success():
    tools = build_default_tool_registry()
    tools.register(ToolSpec(name="inner", category="test", description="Read"), lambda _context: ToolResult(True))
    tools.bind_handler("get_memory", lambda _context: tools.call("inner") and ToolResult(True))
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(StaticPlannerBackend(
        {"kind": "tool_call", "name": "get_memory", "parameters": {}},
    )), tools=tools)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    result = runner.run(task="inspect", max_turns=3, max_tool_calls=1)
    assert len(result.steps) == 1
    assert result.metadata["failure_reason"]["code"] == "tool_call_limit_exceeded"
    assert result.metadata["failure_reason"]["observed"] == 2
    assert result.metadata["usage"]["tool_admission"]["denied_this_run"] == 1


@pytest.mark.parametrize("limit,carried", [(0, 0), (True, 0), (2.0, 0), (2, -1), (2, True)])
def test_invalid_quota_is_rejected(limit, carried):
    with pytest.raises(ValueError):
        ToolCallBudget(limit, carried_usage=carried)
