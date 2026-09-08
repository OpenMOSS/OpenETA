"""Real backend failure path with a non-network transport fixture."""
import pytest

from agent.backends.planner import OpenAICompatiblePlannerBackend, OpenAICompatiblePlannerBackendConfig, StaticPlannerBackend
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.parallel import classify_episode_result, episode_failure_error
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.self_improvement import SelfImprovementConfig, SelfImprovementReviewer
from agent.tools.registry import build_default_tool_registry


def runner_for(backend, **kwargs):
    return OpenEtaEpisodeRunner(runtime=OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(backend), tools=build_default_tool_registry(),
        self_improvement_reviewer=SelfImprovementReviewer(config=SelfImprovementConfig(enabled=False)),
    ), environment=DummyEpisodeEnvironment(), **kwargs)


def test_provider_timeout_is_failure_without_human_pause_or_guidance():
    calls = []
    def transport(*args):
        calls.append(1)
        raise TimeoutError("fixture timeout")
    backend = OpenAICompatiblePlannerBackend(OpenAICompatiblePlannerBackendConfig(
        api_base="https://fixture.invalid", api_key="fixture", model="fixture", max_attempts=1,
    ), transport=transport)
    runner = runner_for(backend, interaction_resolver=lambda **kwargs: pytest.fail("provider failure must not request guidance"))
    result = runner.run(task="fixture", max_turns=2)
    assert len(calls) == 1
    assert result.metadata["stop_reason"] == "planner_provider_failed"
    assert result.metadata["failure_reason"]["error_type"] == "TimeoutError"
    assert result.metadata["failure_reason"]["provider_error_code"] == "transient_provider_failure"
    assert result.truncated and not result.terminated
    assert result.steps[-1].step_result.truncated
    assert "pause_reason" not in result.steps[-1].step_result.info
    assert not result.metadata["waiting_for_human"]
    assert not result.metadata["assistance"]["human_assisted"]
    assert result.metadata["usage"]["human_wait_s"] == 0
    assert result.metadata["usage"]["token_usage_sources"]["unknown"] == 1
    assert classify_episode_result(result) == "fail"
    assert episode_failure_error(result)["type"] == "EpisodePlannerFailure"


ASK = {"kind": "response", "name": "ask_human", "parameters": {"question": "Which cup?"}}
TALK = {"kind": "response", "name": "talk", "parameters": {"message": "Done inspecting"}}


@pytest.mark.parametrize("answer,expected", [(None, False), ("", False), ("   ", False), ("Left cup", True)])
def test_assistance_requires_received_nonempty_answer_not_elapsed_wait(answer, expected):
    now = [0.0]
    runner = runner_for(StaticPlannerBackend([ASK, TALK]), clock=lambda: now[0])
    runner.run(task="fixture", max_turns=3)
    now[0] = 5.0
    paused = runner.continue_run(max_turns=1)
    assert paused.metadata["usage"]["human_wait_s"] == 5.0
    assert not paused.metadata["assistance"]["human_assisted"]
    if answer is not None:
        runner.runtime.update_memory({"type": "human_answer", "question": "Which cup?", "answer": answer})
    runner.resume_after_human()
    result = runner.continue_run(max_turns=1)
    assert result.metadata["assistance"]["human_assisted"] is expected


def test_received_answer_at_zero_elapsed_time_counts_and_does_not_leak_into_new_episode():
    runner = runner_for(StaticPlannerBackend([ASK, TALK, TALK]), clock=lambda: 0.0)
    runner.run(task="fixture", max_turns=3)
    runner.runtime.update_memory({"type": "human_answer", "answer": "Left cup"})
    runner.resume_after_human()
    continued = runner.continue_run(max_turns=1)
    assert continued.metadata["assistance"]["human_assisted"]
    fresh = runner.run(task="new fixture", max_turns=1)
    assert not fresh.metadata["assistance"]["human_assisted"]


def test_model_claim_of_provider_failure_is_still_an_ordinary_question():
    runner = runner_for(StaticPlannerBackend({**ASK, "parameters": {
        "question": "Which cup?", "provider_error_code": "transient_provider_failure", "error_type": "TimeoutError",
    }}))
    result = runner.run(task="fixture", max_turns=1)
    assert result.metadata["waiting_for_human"]
    assert result.metadata["failure_reason"] == {}


def test_later_blank_answer_does_not_erase_received_assistance():
    runner = runner_for(StaticPlannerBackend([ASK, TALK]), clock=lambda: 0.0)
    runner.run(task="fixture", max_turns=3)
    runner.runtime.update_memory({"type": "human_answer", "answer": "Left cup"})
    runner.runtime.update_memory({"type": "human_answer", "answer": " "})
    runner.resume_after_human()
    assert runner.continue_run(max_turns=1).metadata["assistance"]["human_assisted"]


def test_http_read_timeout_through_real_backend_is_not_human_assistance():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    arrived, release = threading.Event(), threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            arrived.set()
            release.wait(timeout=3)
            try:
                self.send_response(503)
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError):
                pass
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        backend = OpenAICompatiblePlannerBackend(OpenAICompatiblePlannerBackendConfig(
            api_base=f"http://127.0.0.1:{server.server_port}/v1", api_key="fixture", model="fixture",
            timeout_s=0.2, max_attempts=1, enable_vision=False,
        ))
        result = runner_for(backend).run(task="synthetic timeout fixture", max_turns=2)
        assert arrived.is_set()
        assert result.metadata["failure_reason"]["error_type"] == "TimeoutError"
        assert not result.metadata["waiting_for_human"]
        assert not result.metadata["assistance"]["human_assisted"]
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
