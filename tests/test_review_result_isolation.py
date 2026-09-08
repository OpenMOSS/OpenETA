"""Review is downstream of an authoritative episode result, not its owner."""
from copy import deepcopy

import pytest

from agent.backends.planner import StaticPlannerBackend
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime


def run_with(reviewer):
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(StaticPlannerBackend({
            "kind": "response", "name": "talk", "parameters": {"message": "fixture report"},
        })), self_improvement_reviewer=reviewer,
    )
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    reviewer.runtime = runtime
    return runner.run(task="original task", max_turns=1), runtime


@pytest.mark.parametrize("outcome", ["success", "exception", "malformed"])
def test_review_cannot_rewrite_task_steps_or_outcome_even_before_failure(outcome):
    class Reviewer:
        def maybe_review(self, result, *, skills):
            result.task = "rewritten"
            result.steps[0].action.command["request"]["parameters"]["message"] = "rewritten"
            result.steps.clear()
            result.terminated = False
            result.truncated = True
            result.metadata.clear()
            if outcome == "exception":
                raise ValueError("review failed after editing its input")
            return None if outcome == "malformed" else {"reviewed": True, "proposals": []}

    result, runtime = run_with(Reviewer())
    assert result.task == "original task"
    assert len(result.steps) == 1
    assert result.steps[0].action.command["request"]["parameters"]["message"] == "fixture report"
    assert result.terminated and not result.truncated
    assert result.metadata["stop_reason"] == "status_report"
    assert result.metadata["usage"]["tool_call_count"] == 0
    episode_event = next(event for event in runtime.memory.events if event.event_type == "episode_result")
    assert episode_event.payload["num_steps"] == 1
    assert episode_event.payload["metadata"]["stop_reason"] == "status_report"


def test_retained_review_input_and_report_cannot_mutate_returned_result_or_memory_later():
    class Reviewer:
        def maybe_review(self, result, *, skills):
            self.input = result
            self.report = {
                "reviewed": True, "trigger": {"reason": "fixture"},
                "proposals": [{"proposal_id": "fixture", "skill_name": "fixture", "path": "fixture"}],
            }
            return self.report

    reviewer = Reviewer()
    result, runtime = run_with(reviewer)
    before_result = deepcopy(result.to_dict())
    before_events = deepcopy([event.payload for event in runtime.memory.events])
    reviewer.input.steps.clear()
    reviewer.input.metadata.clear()
    reviewer.report["trigger"]["reason"] = "late change"
    reviewer.report["proposals"][0]["skill_name"] = "late change"
    reviewer.report["proposals"].clear()
    assert result.to_dict() == before_result
    assert [event.payload for event in runtime.memory.events] == before_events


@pytest.mark.parametrize("same_session_id", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_review_report_is_not_published_to_replaced_session_including_aba(same_session_id, fails):
    class Reviewer:
        def maybe_review(self, result, *, skills):
            self.old_session_id = result.session_id
            self.runtime.start_session(task="replacement", session_id=(
                result.session_id if same_session_id else "replacement-session"
            ))
            if fails:
                raise ValueError("old review failed")
            return {"reviewed": True, "proposals": []}

    reviewer = Reviewer()
    result, runtime = run_with(reviewer)
    assert result.session_id == reviewer.old_session_id
    assert result.task == "original task" and len(result.steps) == 1
    assert result.metadata["self_improvement_review"]["memory_publication"] == {
        "recorded": False, "reason": "session_or_episode_changed_during_review",
    }
    assert all(event.event_type != "self_improvement_review" for event in runtime.memory.events)
