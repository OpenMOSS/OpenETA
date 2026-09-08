"""Historical defect probes; assertions confirm bugs, not implementation correctness.

Uses only local fixtures and fake transports. Each run writes to a fresh temp root.
"""
from pathlib import Path
import asyncio
import json
import os
import sys
import threading
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapter.protocol import EnvAction, EnvObservation, RobotState, StepResult
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.episode import DummyEpisodeEnvironment, EpisodeResult, EpisodeStep, OpenEtaEpisodeRunner
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.parallel import classify_episode_result
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.supervision import InteractionResolution
from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.registry import ToolExecutionContext, ToolResult, build_default_tool_registry
from agent.tools.sim_mcp import SimulatorMcpEpisodeConfig, SimulatorMcpEpisodeEnvironment, _temporary_no_proxy_for_url

ROOT = Path(tempfile.mkdtemp(prefix="openeta-review-"))
RESULTS = {}


def python_runtime(label, extra_globals=None):
    root = ROOT / label
    sandbox = root / "sandbox"
    sandbox.mkdir(parents=True, exist_ok=True)
    return PythonExecRuntime(PythonExecConfig(
        default_timeout_s=0.01, session_root=str(root), workspace_root=str(sandbox),
        image_output_root=str(root / "images"), text_output_root=str(root / "text"),
        response_output_root=str(root / "responses"), structured_output_root=str(root / "structured"),
        extra_globals=extra_globals or {},
    ))


def python_context(code):
    return ToolExecutionContext(name="python_exec", spec=build_default_tool_registry().get("python_exec"), parameters={"code": code})


def sandbox_file_boundary():
    runtime = python_runtime("sandbox-check")
    target = ROOT / "outside-session.txt"
    result = runtime.handler(python_context(f"import numpy as np\nnp.savetxt({str(target)!r}, [314159])\nresult = float(np.loadtxt({str(target)!r}))"))
    RESULTS["sandbox_boundary"] = {"tool_success": result.success, "outside_session_file_created": target.exists(), "returned": result.details.get("outputs", {}).get("result")}
    assert result.success and target.exists()


def sandbox_timeout():
    runtime = python_runtime("timeout-check")
    started = time.monotonic()
    result = runtime.handler(python_context("import datetime\nstart = datetime.datetime.now()\nwhile (datetime.datetime.now() - start).total_seconds() < 0.08:\n    pass\nresult = 'completed'"))
    duration = time.monotonic() - started
    RESULTS["python_timeout"] = {"configured_timeout_s": 0.01, "elapsed_s": round(duration, 3), "success": result.success}
    assert result.success and duration >= 0.08


def stdout_concurrency():
    # Exit in reverse entry order to restore the original stdout safely.
    entered_a, entered_b, finish_b = threading.Event(), threading.Event(), threading.Event()
    a = python_runtime("stdout-a", {"entered_a": entered_a, "entered_b": entered_b, "finish_b": finish_b})
    b = python_runtime("stdout-b", {"entered_b": entered_b, "finish_b": finish_b})
    results = {}
    def run_a():
        results["a"] = a.handler(python_context("print('A-before')\nentered_a.set()\nentered_b.wait(1)\nprint('A-during-B')\nfinish_b.set()\nresult = 'a'"))
    # Hold A before exiting redirect_stdout until B has returned.
    b_done = threading.Event()
    a.config.extra_globals["b_done"] = b_done
    def run_a_ordered():
        results["a"] = a.handler(python_context("print('A-before')\nentered_a.set()\nentered_b.wait(1)\nprint('A-during-B')\nfinish_b.set()\nb_done.wait(1)\nresult = 'a'"))
    def run_b():
        results["b"] = b.handler(python_context("print('B-before')\nentered_b.set()\nfinish_b.wait(1)\nprint('B-after')\nresult = 'b'"))
        b_done.set()
    ta = threading.Thread(target=run_a_ordered)
    tb = threading.Thread(target=run_b)
    ta.start()
    assert entered_a.wait(1)
    tb.start()
    ta.join(2)
    tb.join(2)
    assert not ta.is_alive() and not tb.is_alive()
    RESULTS["stdout_cross_session"] = {key: value.details["outputs"]["stdout"] for key, value in results.items()}
    assert "A-during-B" in RESULTS["stdout_cross_session"]["b"]


def canceled_handler_state():
    started, release, done = threading.Event(), threading.Event(), threading.Event()
    tools = build_default_tool_registry()
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(StaticPlannerBackend({"kind": "tool_call", "name": "save_memory", "parameters": {"namespace": "facts", "key": "probe", "content": "old"}})), tools=tools, rollout_enabled=False)
    original = runtime._save_memory_tool
    def slow_save(context):
        started.set()
        release.wait(2)
        result = original(context)
        done.set()
        return result
    tools.bind_handler("save_memory", slow_save, replace=True)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    runner.start(task="old task", timeout_s=5, max_turns=2)
    old_run = {}
    def execute_old():
        try:
            old_run["result"] = runner.continue_run()
        except Exception as error:
            old_run["error"] = type(error).__name__
    thread = threading.Thread(target=execute_old)
    thread.start()
    assert started.wait(1)
    runner.interrupt(code="user_interrupt")
    thread.join(0.5)
    idle = runner.wait_for_idle(timeout_s=0.3)
    runtime.start_session(task="new task", session_id="new-session")
    release.set()
    assert done.wait(1)
    RESULTS["late_handler_mutation"] = {"failure": runner.failure_reason, "runner_reports_idle": idle, "current_session": runtime.memory.session_id, "stale_fact_present": "probe" in runtime.memory.agent_working_state}
    assert RESULTS["late_handler_mutation"]["stale_fact_present"]


def guidance_timeout():
    class SlowResolver:
        def resolve(self, **kwargs):
            time.sleep(0.12)
            return InteractionResolution(resolved=False)
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(StaticPlannerBackend({"kind": "response", "name": "ask_human", "parameters": {"question": "What target?"}})), rollout_enabled=False)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment(), interaction_resolver=SlowResolver())
    start = time.monotonic()
    result = runner.run(task="select target", timeout_s=0.04)
    duration = time.monotonic() - start
    RESULTS["guidance_timeout"] = {"timeout_s": 0.04, "elapsed_s": round(duration, 3), "failure": result.metadata["failure_reason"]}
    assert duration >= 0.12


def false_success():
    obs = EnvObservation(task="not completed", cameras=[], robot=RobotState())
    action = EnvAction(action_type="tool_call")
    step = EpisodeStep(1, obs, action, StepResult(observation=obs, reward=0.1, terminated=False, truncated=False, info={"success": False, "task_success": False}))
    result = EpisodeResult(task="not completed", session_id="probe", steps=[step], truncated=True, metadata={"failure_reason": {"code": "episode_timeout"}})
    verdict = classify_episode_result(result, env_id="openeta/metaworld_50_reach-v3-v0")
    RESULTS["false_success"] = {"reward": 0.1, "task_success": False, "episode_truncated": True, "classification": verdict}
    assert verdict == "success"


def overwrite_host_memory():
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(StaticPlannerBackend({"kind": "tool_call", "name": "save_memory", "parameters": {"namespace": "facts", "key": "robot_motion_epoch", "content": {"epoch": 0}}})), rollout_enabled=False)
    runtime.start_session(task="memory probe")
    runtime.memory.save_fact("robot_motion_epoch", {"epoch": 7}, source="runtime")
    action = runtime.act(DummyEpisodeEnvironment().reset(task="memory probe"))
    RESULTS["host_memory_overwrite"] = {"before": 7, "after": runtime.memory.robot_motion_epoch(), "pipeline_status": action.command["status"], "stored_source": runtime.memory.facts["robot_motion_epoch"]["source"]}
    assert runtime.memory.robot_motion_epoch() == 0 and action.command["status"] == "executed"


def close_loses_retry():
    class FailingTransport:
        count = 0
        def call_tool(self, *args, **kwargs):
            self.count += 1
            raise TimeoutError("controlled fake close failure")
    transport = FailingTransport()
    env = SimulatorMcpEpisodeEnvironment(transport=transport, config=SimulatorMcpEpisodeConfig(env_id="fake", handle="h", session_id="s"))
    first, second = env.close(), env.close()
    RESULTS["close_retry_lost"] = {"first": first, "second": second, "remote_calls": transport.count, "remaining_handle": env.config.handle}
    assert first["ok"] is False and second["ok"] is True and transport.count == 1


def server_false_close():
    import sim.mcp_server.server as server
    class Manager:
        def proxy_handle_op(self, *args, **kwargs):
            return {"error": "controlled worker deletion failure"}
        def release_worker(self, *args):
            pass
    server._session_envs["review-only"] = {"h": {"worker_url": "fake", "remote_handle": "r"}}
    with patch.object(server, "_get_mgr", return_value=Manager()):
        result = server.close_env.__wrapped__(handle="h", session_id="review-only")
    RESULTS["server_false_close"] = result
    assert result["ok"] is True and result["remote"].get("error")


def truncated_trace():
    store = JsonMemoryStore(ROOT / "trace-check")
    memory = AgentMemory(store=store)
    memory.start_session(task="probe", session_id="trace-test")
    path = store.session_path("trace-test")
    # Model a process crash partway through the final append, in this isolated store only.
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"event_type": "partial')
    try:
        AgentMemory(store=store).resume_session("trace-test")
    except json.JSONDecodeError as error:
        RESULTS["partial_trace_resume"] = {"error": type(error).__name__, "valid_prefix_present": True}
    else:
        raise AssertionError("expected the current recovery bug")


def no_proxy_concurrency():
    original = {key: os.environ.get(key) for key in ("NO_PROXY", "no_proxy")}
    try:
        os.environ["NO_PROXY"] = "baseline.invalid"
        os.environ["no_proxy"] = "baseline.invalid"
        first = _temporary_no_proxy_for_url("http://a.invalid/sse")
        second = _temporary_no_proxy_for_url("http://b.invalid/sse")
        first.__enter__()
        second.__enter__()
        first.__exit__(None, None, None)
        during = os.environ["NO_PROXY"]
        second.__exit__(None, None, None)
        after = os.environ["NO_PROXY"]
        RESULTS["no_proxy_overlap"] = {"while_b_active": during, "after_all_calls": after}
        assert "b.invalid" not in during and "a.invalid" in after
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    for probe in (sandbox_file_boundary, sandbox_timeout, stdout_concurrency, canceled_handler_state, guidance_timeout, false_success, overwrite_host_memory, close_loses_retry, server_false_close, truncated_trace, no_proxy_concurrency):
        try:
            probe()
        except Exception as error:
            RESULTS[probe.__name__ + "_probe_error"] = {"type": type(error).__name__, "message": str(error)}
    print(json.dumps({"artifact_root": str(ROOT), "results": RESULTS}, ensure_ascii=False, indent=2))
