from __future__ import annotations

import agent.cli.batch_eval as batch_eval
import agent.cli.openeta_cli as cli_module
import agent.runtime.runtime_assembly as runtime_assembly
import pytest
from agent.backends.planner import (
    REASONING_SUBAGENT_MAX_OUTPUT_TOKENS,
    StaticPlannerBackend,
)
from agent.backends.provider_config import PlannerProviderConfig
from agent.cli.batch_eval import build_mcp_episode_worker_factory
from agent.cli.openeta_cli import OpenEtaCli
from agent.runtime.parallel import ParallelEpisodeSpec
from agent.runtime.runtime_assembly import (
    ENVIRONMENT_PLACEHOLDER_TOOLS,
    MAIN_PLANNER_MAX_OUTPUT_TOKENS,
    VDM_MAX_OUTPUT_TOKENS,
    REMOTE_PLACEHOLDER_TOOLS,
    RuntimeAssemblyConfig,
    RuntimeMcpEndpoints,
    assemble_runtime,
    resolve_runtime_mcp_endpoints,
)
from agent.runtime.reference_localization import (
    REFERENCE_POINT_LOCALIZATION_MAX_OUTPUT_TOKENS,
)
from agent.tools.grasp_pose_advisor import GRASP_POSE_ADVISOR_MAX_OUTPUT_TOKENS
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


def _matching_anygrasp_capabilities(**_kwargs):
    return {
        "schema_version": "openeta.anygrasp_capabilities.v1",
        "backend": "anygrasp_mcp",
        "model": "anygrasp_sdk",
        "max_gripper_width_m": 0.08,
        "gripper_height_m": 0.03,
        "depth_truncation_m": 1.0,
        "max_candidates": 20,
        "geometry_change_requires_redeployment": True,
    }


@pytest.fixture(autouse=True)
def _stub_anygrasp_capability_discovery(monkeypatch):
    monkeypatch.setattr(
        runtime_assembly,
        "query_anygrasp_capabilities",
        _matching_anygrasp_capabilities,
    )


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
        "visual_history": assembly.runtime.visual_history.descriptor(),
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
    assert (
        tui.runtime.planner.tool_contract_policy
        is tui.runtime.pipeline.tool_contract_policy
    )
    assert (
        tui.runtime.planner.tool_contract_catalog
        is tui.runtime.pipeline.tool_contract_catalog
    )
    assert tui.runtime.planner.tool_contract_policy.to_dict() == {
        "schema_version": "openeta.tool_contract_runtime_policy.v1",
        "request_validation_authority": [],
        "gate_repair_envelope_authority": [],
        "executable_gate_authority": "legacy_runtime",
    }
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
    assert tui.runtime.visual_history is not batch.runtime.visual_history
    assert tui.runtime.visual_history.config == batch.runtime.visual_history.config
    assert tui.runtime.planner.context_config.visual_history == (
        batch.runtime.planner.context_config.visual_history
    )


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

    assert assembly.runtime.tools.can_execute("prepare_attachment_probe") is True
    assert assembly.runtime.tools.can_execute("assess_attachment_probe") is True


def test_shared_assembly_reserves_visual_window_and_isolates_vdm_backend(
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
    calls = []

    def backend_factory(**kwargs):
        calls.append(dict(kwargs))
        return StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "ok"}}
        )

    assembly = assemble_runtime(
        RuntimeAssemblyConfig(
            workspace=SessionWorkspace.create("visual-history", root=tmp_path),
            provider=PlannerProviderConfig(
                model="fixture",
                api_base="http://provider.example/v1",
                api_key="test",
            ),
            backend_factory=backend_factory,
            supervision_policy=SupervisionPolicy.for_profile("standard"),
            web_access_config=WebAccessConfig(),
        )
    )

    assert {
        "max_tokens": MAIN_PLANNER_MAX_OUTPUT_TOKENS,
        "max_vision_images": 9,
    } in calls
    assert {
        "max_tokens": VDM_MAX_OUTPUT_TOKENS,
        "max_vision_images": 2,
        "enable_thinking": False,
    } in calls
    assert {
        "max_tokens": REASONING_SUBAGENT_MAX_OUTPUT_TOKENS,
        "max_vision_images": 4,
    } in calls
    assert {"max_tokens": REASONING_SUBAGENT_MAX_OUTPUT_TOKENS} in calls
    assert REASONING_SUBAGENT_MAX_OUTPUT_TOKENS >= 8192
    assert GRASP_POSE_ADVISOR_MAX_OUTPUT_TOKENS >= 8192
    assert REFERENCE_POINT_LOCALIZATION_MAX_OUTPUT_TOKENS >= 8192
    assert VDM_MAX_OUTPUT_TOKENS >= 4096
    assert (
        assembly.runtime.planner.context_config.reserved_output_tokens
        == MAIN_PLANNER_MAX_OUTPUT_TOKENS
    )
    assert assembly.runtime.visual_history.backend is not assembly.runtime.planner.backend


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


def test_contact_graspnet_is_absent_from_runtime_registry(tmp_path) -> None:
    workspace = SessionWorkspace.create("contact-disabled", root=tmp_path)
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
            endpoints=RuntimeMcpEndpoints(
                anygrasp_url="http://anygrasp.example/sse",
                graspgenx_url="http://graspgenx.example/sse",
            ),
            web_access_config=WebAccessConfig(),
        )
    )

    assert "contact_graspnet" not in {
        tool.name for tool in assembly.runtime.tools.list()
    }
    assert assembly.runtime.tools.can_execute("grasp_pose_estimate") is True


def test_shared_runtime_disables_anygrasp_when_deployment_width_mismatches(
    tmp_path,
) -> None:
    workspace = SessionWorkspace.create("anygrasp-mismatch", root=tmp_path)
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
            endpoints=RuntimeMcpEndpoints(
                anygrasp_url="http://anygrasp.example/sse",
            ),
            anygrasp_capability_query=lambda **_kwargs: {
                **_matching_anygrasp_capabilities(),
                "max_gripper_width_m": 0.1,
            },
            web_access_config=WebAccessConfig(),
        )
    )

    report = assembly.perception_capabilities["backends"]["anygrasp"]
    assert report["reason"] == "gripper_width_mismatch"
    assert report["available"] is False
    assert assembly.runtime.tools.can_execute("grasp_pose_estimate") is False

    assembly.runtime.start_session(task="pick the milk")
    context = assembly.runtime.memory.planning_context()
    assert context["working_memory"]["facts"]["perception_backend_capabilities"]["value"][
        "backends"
    ]["anygrasp"]["reason"] == "gripper_width_mismatch"


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
