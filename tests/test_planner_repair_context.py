import json
from copy import deepcopy

import pytest

from agent.backends.planner import StaticPlannerBackend, _planner_user_prompt
from agent.runtime.episode import OpenEtaEpisodeRunner, DummyEpisodeEnvironment
from agent.runtime.planner import ToolCallingPlanner, PlannerDecision, _rejected_candidate_snapshot
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.registry import build_default_tool_registry


def test_validation_retry_carries_prior_candidate_but_does_not_execute_it():
    rejected = {"kind": "tool_call", "name": "get_memory", "parameters": {"key": "target", "wrong_field": 1}}
    corrected = {"kind": "tool_call", "name": "get_memory", "parameters": {"key": "target"}}
    class Backend(StaticPlannerBackend):
        def __init__(self):
            super().__init__([rejected, corrected])
            self.requests = []
        def decide(self, request):
            self.requests.append(deepcopy(request))
            return super().decide(request)
    backend = Backend()
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(backend), tools=build_default_tool_registry())
    result = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment()).run(task="fixture", max_turns=1)
    assert len(backend.requests) == 2
    assert "previous_candidate" not in backend.requests[0].metadata
    feedback = json.loads(_planner_user_prompt(backend.requests[1]))["validation_feedback"]
    assert feedback["previous_candidate"] == rejected
    assert feedback["candidate_is_untrusted"] is True
    assert result.metadata["usage"]["tool_admission"]["admitted_this_run"] == 1
    assert result.steps[0].action.command["request"]["parameters"] == corrected["parameters"]
    backend.requests[1].metadata["isolated_context"] = True
    assert "previous_candidate" not in json.loads(_planner_user_prompt(backend.requests[1]))["validation_feedback"]


def test_snapshot_is_detached_bounded_and_not_synthesized_from_parse_fallback():
    raw = {"kind": "tool_call", "name": "sam3", "parameters": {"source_packet_id": "obs-1"}}
    decision = PlannerDecision(action_type="tool_call", action="sam3", metadata={"raw_backend_payload": raw})
    snapshot = _rejected_candidate_snapshot(decision)
    raw["parameters"]["source_packet_id"] = "changed"
    assert snapshot["parameters"]["source_packet_id"] == "obs-1"
    raw["parameters"]["huge"] = "x" * 16385
    assert _rejected_candidate_snapshot(decision) is None
    assert _rejected_candidate_snapshot(PlannerDecision(action_type="response", action="ask_human")) is None


@pytest.mark.parametrize("oversized", [False, True])
def test_unparsed_response_is_bounded_untrusted_repair_context_not_an_action(oversized):
    malformed = (
        "<decision><kind>tool_call</kind><name>get_memory</name>"
        "<parameters><key>target</key></parameters><reasoning>"
        + ("x" * 16385 if oversized else "Inspect the evidence") + "</decision>"
    )
    corrected = {"kind": "tool_call", "name": "get_memory", "parameters": {"key": "target"}}

    class Backend(StaticPlannerBackend):
        def __init__(self):
            super().__init__([malformed, corrected])
            self.requests = []

        def decide(self, request):
            self.requests.append(deepcopy(request))
            return super().decide(request)

    backend = Backend()
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(backend), tools=build_default_tool_registry())
    result = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment()).run(task="fixture", max_turns=1)
    assert len(backend.requests) == 2
    assert "previous_unparsed_response" not in backend.requests[0].metadata
    feedback = json.loads(_planner_user_prompt(backend.requests[1]))["validation_feedback"]
    assert "previous_candidate" not in feedback  # No fabricated parsed action.
    if oversized:
        assert "previous_unparsed_response" not in feedback
    else:
        assert feedback["previous_unparsed_response"] == malformed
        assert feedback["unparsed_response_is_untrusted"] is True
    assert result.metadata["usage"]["tool_admission"]["admitted_this_run"] == 1
    assert result.steps[0].action.command["request"]["parameters"] == corrected["parameters"]
    backend.requests[1].metadata["isolated_context"] = True
    assert "previous_unparsed_response" not in json.loads(_planner_user_prompt(backend.requests[1]))["validation_feedback"]
