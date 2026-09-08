"""Host simulator RPC limits must not inherit manual/model provider waits."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from agent.backends.provider_config import PlannerProviderConfig
from agent.cli import batch_eval, experiment, openeta_cli
from agent.runtime.parallel import ParallelEpisodeSpec
from agent.tools.mcp_timeouts import DEFAULT_SIM_MCP_TIMEOUT_S, validate_simulator_timeout_s


INVALID = [0, -1, True, False, float("inf"), float("nan"), "", "invalid", None, []]


@pytest.mark.parametrize("value", INVALID)
def test_invalid_simulator_timeout_is_rejected_before_building_any_runtime(monkeypatch, value):
    monkeypatch.setattr(openeta_cli.OpenEtaCli, "_build_runtime", lambda self: pytest.fail("must not build"))
    monkeypatch.setattr(batch_eval, "load_planner_provider_config", lambda: pytest.fail("must not load provider"))
    monkeypatch.setattr(batch_eval, "PausedEpisodeStore", lambda: pytest.fail("must not touch paused record"))
    with pytest.raises(ValueError, match="positive finite"):
        openeta_cli.OpenEtaCli(simulator_timeout_s=value)
    with pytest.raises(ValueError, match="positive finite"):
        batch_eval.build_mcp_episode_worker_factory(simulator_timeout_s=value)
    with pytest.raises(ValueError, match="positive finite"):
        batch_eval.resume_paused_episode(session_id="fixture", interaction_id="fixture", answer="fixture", simulator_timeout_s=value)


@pytest.mark.parametrize("value,expected", [(45,45.0),("45.5",45.5),(0.1,0.1)])
def test_host_timeout_can_be_shorter_than_previous_floor(value, expected):
    assert validate_simulator_timeout_s(value) == expected


class Transport:
    def __init__(self, url):
        self.url = url
        self.calls = []

    def list_tools(self, *, timeout_s=None):
        return {"tools": []}

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append((name, timeout_s))
        if name == "create_env":
            return {"handle":"fixture-handle", "session_id":"fixture-session"}
        if name == "close_env":
            return {"ok":True,"already_closed":False,"cleanup_errors":[]}
        return {"success":True,"cameras":[],"robot":{},"reward":0.0}


@pytest.mark.parametrize("provider_timeout", [45.0, 86400.0])
@pytest.mark.parametrize("simulator_timeout", [None, 45.0, 10.0])
def test_cli_rebuild_and_actual_handler_keep_independent_simulator_timeout(monkeypatch, tmp_path, provider_timeout, simulator_timeout):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENETA_LLM_TIMEOUT_S", str(provider_timeout))
    made = []

    def factory(url):
        made.append(Transport(url))
        return made[-1]

    monkeypatch.setattr(openeta_cli, "SseSimulatorMcpTransport", factory)
    cli = openeta_cli.OpenEtaCli(simulator_mcp_url="http://fixture.invalid/sse",
                                **({"simulator_timeout_s":simulator_timeout} if simulator_timeout is not None else {}))
    expected = simulator_timeout if simulator_timeout is not None else DEFAULT_SIM_MCP_TIMEOUT_S
    assert cli.state.config.timeout_s == provider_timeout
    assert cli.state.simulator_mcp_config.timeout_s == expected
    cli.state.config.timeout_s = 172800.0
    cli._build_runtime()
    assert cli.state.simulator_mcp_config.timeout_s == expected
    cli.state.simulator_mcp_config.handle = "fixture-handle"
    cli.state.simulator_mcp_config.session_id = "fixture-session"
    try:
        assert cli._require_runtime().tools.call("observe", {}).success
    finally:
        assert cli.close()["ok"] is True
    assert made[0].calls == [("render_env",expected),("close_env",min(30.0,expected))]


@pytest.mark.parametrize("simulator_timeout", [None,45.0,10.0])
def test_batch_bootstrap_tools_and_close_use_same_independent_limit(monkeypatch, tmp_path, simulator_timeout):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(batch_eval, "load_planner_provider_config", lambda: PlannerProviderConfig(
        provider="manual-vlm",model="fixture",api_base="http://fixture.invalid/v1",api_key="fixture",timeout_s=86400.0,
    ))
    monkeypatch.setattr(batch_eval, "load_mcp_server_url", lambda *args, **kwargs: "")
    monkeypatch.setattr(batch_eval, "SseSimulatorMcpTransport", Transport)
    spec = ParallelEpisodeSpec(episode_id="fixture",task="fixture",env_id="openeta/test-v0",
                               metadata={"workspace_parent":str(tmp_path)})
    worker = batch_eval.build_mcp_episode_worker_factory(
        sim_url="http://fixture.invalid/sse",
        **({"simulator_timeout_s":simulator_timeout} if simulator_timeout is not None else {}),
    )(spec,"fixture-batch")
    expected = simulator_timeout if simulator_timeout is not None else DEFAULT_SIM_MCP_TIMEOUT_S
    environment = worker.runner.environment
    assert environment.config.timeout_s == environment.tool_proxy_config.timeout_s == expected
    try:
        environment.reset(task="fixture")
        assert worker.runner.runtime.tools.call("observe", {}).success
    finally:
        assert worker.close()["ok"] is True
    assert environment.transport.calls == [
        ("create_env",expected),("reset_env",expected),("render_env",expected),("close_env",min(30.0,expected)),
    ]


def test_cli_argument_reaches_host_configuration(monkeypatch):
    captured = []
    monkeypatch.setattr(openeta_cli.OpenEtaCli,"_build_runtime",lambda self: captured.append(self.state.simulator_timeout_s))
    monkeypatch.setattr(openeta_cli.OpenEtaCli,"run",lambda self: None)
    monkeypatch.setattr(openeta_cli.OpenEtaCli,"close",lambda self: None)
    assert openeta_cli.main(["--simulator-timeout-s","45"]) == 0
    assert captured == [45.0]


def test_batch_resume_flag_is_forwarded_without_running_a_session(monkeypatch):
    captured = []
    monkeypatch.setattr(batch_eval,"resume_paused_episode",lambda **kwargs: captured.append(kwargs) or {"outcome":{"status":"success"}})
    assert batch_eval.main(["--resume-session","fixture","--interaction-id","fixture","--answer","fixture",
                            "--simulator-timeout-s","45"]) == 0
    assert captured[0]["simulator_timeout_s"] == 45.0


def test_experiment_runtime_flag_is_available_to_run_and_iterate():
    parser = argparse.ArgumentParser()
    experiment._add_runtime_arguments(parser)
    assert parser.parse_args(["--simulator-timeout-s","45"]).simulator_timeout_s == 45.0


@pytest.mark.parametrize("entry", ["batch", "experiment"])
def test_fresh_batch_and_experiment_forward_timeout_to_worker_factory(monkeypatch, entry):
    module = batch_eval if entry == "batch" else experiment
    captured = []
    factory = object()
    monkeypatch.setattr(module, "build_mcp_episode_worker_factory",
                        lambda **kwargs: captured.append(kwargs) or factory)

    class Harness:
        def __init__(self, selected_factory, **kwargs):
            assert selected_factory is factory

        def run(self, specs, **kwargs):
            return SimpleNamespace(fail_count=0,to_dict=lambda: {"fixture":True})

    monkeypatch.setattr(module, "ParallelEpisodeHarness", Harness)
    if entry == "batch":
        spec = ParallelEpisodeSpec(episode_id="fixture",task="fixture",env_id="openeta/test-v0")
        monkeypatch.setattr(module, "load_parallel_episode_manifest", lambda path: [spec])
        assert module.main(["--manifest","fixture.json","--simulator-timeout-s","45"]) == 0
    else:
        parser = argparse.ArgumentParser()
        module._add_runtime_arguments(parser)
        args = parser.parse_args(["--simulator-timeout-s","45"])
        assert module._run_batch(args, [], batch_id="fixture") == {"fixture":True}
    assert captured[0]["simulator_timeout_s"] == 45.0


@pytest.mark.parametrize("entry", [openeta_cli.main,batch_eval.main])
@pytest.mark.parametrize("value", ["0","-1","nan","inf"])
def test_command_line_rejects_invalid_simulator_budget_before_dispatch(entry, value):
    with pytest.raises(SystemExit) as error:
        entry(["--simulator-timeout-s",value])
    assert error.value.code == 2
