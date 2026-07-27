from __future__ import annotations

import agent.cli.batch_eval as batch_eval
import agent.cli.openeta_cli as cli_module
from agent.backends.planner import StaticPlannerBackend
from agent.backends.provider_config import PlannerProviderConfig
from agent.cli.batch_eval import build_mcp_episode_worker_factory
from agent.cli.openeta_cli import OpenEtaCli
from agent.runtime.parallel import ParallelEpisodeSpec
from agent.runtime.runtime_assembly import (
    ENVIRONMENT_PLACEHOLDER_TOOLS,
    REMOTE_PLACEHOLDER_TOOLS,
    RuntimeAssemblyConfig,
    RuntimeMcpEndpoints,
    assemble_runtime,
    resolve_runtime_mcp_endpoints,
)
from agent.runtime.session_workspace import SessionWorkspace
from agent.runtime.supervision import SupervisionPolicy
from agent.tools.sim_mcp import SimulatorMcpToolProxyConfig
from agent.tools.web_access import WebAccessConfig


class FakeSimulatorTransport:
    def __init__(self, url: str = "") -> None:
        self.url = url

    def call_tool(self, name, arguments, *, timeout_s=None):
        del name, arguments, timeout_s
        return {"success": True}


def _backend_factory(**_kwargs):
    return StaticPlannerBackend(
        {
            "kind": "response",
            "name": "talk",
            "parameters": {"message": "fixture"},
        }
    )


def _contract_snapshot(assembly):
    tools = assembly.runtime.tools
    return {
        "specs": [
            (
                spec.name,
                spec.description,
                spec.category,
                spec.effect.value,
                spec.parameters,
            )
            for spec in tools.list()
        ],
        "executable": sorted(
            spec.name for spec in tools.list() if tools.can_execute(spec.name)
        ),
        "max_validation_retries": assembly.runtime.planner.max_validation_retries,
    }


def test_tui_and_batch_profiles_share_runtime_contracts(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    provider = PlannerProviderConfig(
        model="fixture",
        api_base="http://provider.example/v1",
        api_key="test",
    )
    endpoints = RuntimeMcpEndpoints(
        sam3_url="http://sam3.example/sse",
        depth_prior_url="http://depth.example/sse",
        anygrasp_url="http://anygrasp.example/sse",
        anyplace_url="http://anyplace.example/sse",
        graspgenx_url="http://graspgenx.example/sse",
        contact_graspnet_url="http://contact.example/sse",
        molmopoint_url="http://molmo.example/sse",
    )
    transport = FakeSimulatorTransport()
    policy = SupervisionPolicy.for_profile("standard")

    tui_workspace = SessionWorkspace.create("tui", root=tmp_path / "tui")
    batch_workspace = SessionWorkspace.create("batch", root=tmp_path / "batch")
    tui = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=tui_workspace,
            provider=provider,
            backend_factory=_backend_factory,
            supervision_policy=policy,
            endpoints=endpoints,
            simulator_transport=transport,
            simulator_proxy_config=SimulatorMcpToolProxyConfig(),
            web_access_config=WebAccessConfig(),
            allow_outside_sandbox=True,
            max_validation_retries=2,
        )
    )
    batch = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=batch_workspace,
            provider=provider,
            backend_factory=_backend_factory,
            supervision_policy=policy,
            endpoints=endpoints,
            simulator_transport=transport,
            simulator_proxy_config=SimulatorMcpToolProxyConfig(),
            web_access_config=WebAccessConfig(),
            allow_outside_sandbox=False,
            max_validation_retries=2,
        )
    )

    assert _contract_snapshot(tui) == _contract_snapshot(batch)
    assert tui.depth_prefetch is not None
    assert batch.depth_prefetch is not None
    assert tui_workspace.grasp_profile_id == batch_workspace.grasp_profile_id
    assert tui.runtime.memory.store.root == tui_workspace.memory_root
    assert batch.runtime.memory.store.root == batch_workspace.memory_root
    assert tui.runtime.memory.store.session_dir("tui") == tui_workspace.root
    assert batch.runtime.memory.store.session_dir("batch") == batch_workspace.root
    tui.runtime.start_session(task="tui task")
    batch.runtime.start_session(task="batch task")
    assert tui.runtime.memory.session_id == tui_workspace.session_id
    assert batch.runtime.memory.session_id == batch_workspace.session_id
    assert tui.runtime.memory.store.session_path("tui") == tui_workspace.root / "trace.jsonl"
    assert batch.runtime.memory.store.session_path("batch") == batch_workspace.root / "trace.jsonl"


def test_shared_runtime_fails_closed_without_remote_backends(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )
    workspace = SessionWorkspace.create("closed", root=tmp_path)
    assembly = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=workspace,
            provider=PlannerProviderConfig(
                model="fixture",
                api_base="http://provider.example/v1",
                api_key="test",
            ),
            backend_factory=_backend_factory,
            supervision_policy=SupervisionPolicy.for_profile("standard"),
            web_access_config=WebAccessConfig(),
        )
    )

    for name in (*REMOTE_PLACEHOLDER_TOOLS, *ENVIRONMENT_PLACEHOLDER_TOOLS):
        assert assembly.runtime.tools.can_execute(name) is False


def test_shared_endpoint_resolution_owns_names_aliases_and_overrides() -> None:
    calls = []

    def loader(name, *, aliases=()):
        calls.append((name, aliases))
        return f"http://{name}.example/sse"

    endpoints = resolve_runtime_mcp_endpoints(
        RuntimeMcpEndpoints(anygrasp_url="http://override.example/sse"),
        loader=loader,
    )

    assert endpoints.anygrasp_url == "http://override.example/sse"
    assert endpoints.depth_prior_url == "http://openeta-depth-prior.example/sse"
    assert (
        "openeta-depth-prior",
        ("depth-prior", "depth_prior", "unidepth"),
    ) in calls
    assert not any(name == "openeta-anygrasp" for name, _aliases in calls)


def test_real_tui_and_batch_entries_have_runtime_parity(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    provider = PlannerProviderConfig(
        model="fixture",
        api_base="http://provider.example/v1",
        api_key="test",
    )
    urls = {
        "openeta-sim": "http://sim.example/sse",
        "openeta-sam3": "http://sam3.example/sse",
        "openeta-depth-prior": "http://depth.example/sse",
        "openeta-anygrasp": "http://anygrasp.example/sse",
        "openeta-anyplace": "http://anyplace.example/sse",
        "openeta-graspgenx": "http://graspgenx.example/sse",
        "openeta-contact-graspnet": "http://contact.example/sse",
        "openeta-molmopoint": "http://molmo.example/sse",
    }

    monkeypatch.setattr(
        batch_eval,
        "load_planner_provider_config",
        lambda: provider,
    )
    monkeypatch.setattr(
        batch_eval,
        "load_mcp_server_url",
        lambda name, **_kwargs: urls.get(name, ""),
    )
    monkeypatch.setattr(
        batch_eval,
        "load_configured_web_access",
        lambda **_kwargs: WebAccessConfig(),
    )
    monkeypatch.setattr(batch_eval, "SseSimulatorMcpTransport", FakeSimulatorTransport)
    monkeypatch.setattr(
        cli_module,
        "_load_mcp_url",
        lambda name, **_kwargs: urls.get(name, ""),
    )
    monkeypatch.setattr(
        cli_module,
        "_ensure_simulator_mcp_transport",
        lambda _cli: FakeSimulatorTransport(),
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_object_memory_bank",
        lambda: None,
    )
    monkeypatch.setattr(
        "agent.runtime.runtime_assembly.load_configured_asset_reference_catalog",
        lambda: None,
    )

    tui = OpenEtaCli()
    tui.state.config = provider
    tui._build_runtime()
    batch = build_mcp_episode_worker_factory()(
        ParallelEpisodeSpec(
            episode_id="parity",
            task="inspect the scene",
            env_id="openeta/test-v0",
            metadata={"workspace_parent": str(tmp_path / "batch")},
        ),
        "parity-batch",
    )
    tui_runtime = tui._require_runtime()
    batch_runtime = batch.runner.runtime

    tui_executable = {
        spec.name for spec in tui_runtime.tools.list() if tui_runtime.tools.can_execute(spec.name)
    }
    batch_executable = {
        spec.name
        for spec in batch_runtime.tools.list()
        if batch_runtime.tools.can_execute(spec.name)
    }
    assert tui_executable == batch_executable
    assert tui_runtime.planner.max_validation_retries == 2
    assert batch_runtime.planner.max_validation_retries == 2
    assert tui.state.workspace is not None
    assert (
        tui.state.workspace.grasp_profile_id
        == batch.run_metadata["calibration_profile_id"]
    )
