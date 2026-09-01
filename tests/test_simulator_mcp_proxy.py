"""Tests for simulator MCP tool proxy handlers."""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path

import pytest

import agent.tools.sim_mcp as sim_mcp
from adapter.protocol import EnvAction, JsonDict
from agent.tools.contracts import (
    build_default_tool_contract_catalog,
    check_tool_result_conformance,
)
from agent.tools.sim_mcp import (
    DEFAULT_SIMULATOR_MCP_TOOL_NAMES,
    SimulatorMcpEpisodeConfig,
    SimulatorMcpEpisodeEnvironment,
    SimulatorMcpToolProxyConfig,
    SseSimulatorMcpTransport,
    bind_simulator_mcp_tool_handlers,
    close_simulator_mcp_env,
    _parse_mcp_tool_result,
    _parse_mcp_tools_result,
)
from agent.tools.registry import build_default_tool_registry


PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000100ffff03000006000557bfab0d000000"
        "0049454e44ae426082"
    )
).decode("ascii")


class FakeSimulatorMcpTransport:
    def __init__(self, response: JsonDict) -> None:
        self.response = response
        self.calls: list[JsonDict] = []

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        self.calls.append(
            {
                "name": name,
                "arguments": dict(arguments),
                "timeout_s": timeout_s,
            }
        )
        return self.response


class SequencedSimulatorMcpTransport:
    def __init__(self, responses: list[JsonDict], *, url: str = "") -> None:
        self.responses = responses
        self.url = url
        self.calls: list[JsonDict] = []

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append({"name": name, "arguments": dict(arguments), "timeout_s": timeout_s})
        return self.responses[len(self.calls) - 1]


class FailingSimulatorMcpTransport:
    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        raise RuntimeError(f"{name} unavailable")


class UnknownHandleOnceTransport:
    def __init__(self) -> None:
        self.calls: list[JsonDict] = []
        self.create_count = 0
        self.reset_count = 0

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append({"name": name, "arguments": dict(arguments)})
        if name == "create_env":
            self.create_count += 1
            return {
                "success": True,
                "handle": f"env-{self.create_count}",
                "session_id": "session-retry",
            }
        if name == "reset_env":
            self.reset_count += 1
            if self.reset_count == 1:
                return {"success": False, "error": "Unknown handle: env-1"}
            return {"success": True, "cameras": [], "robot": {}}
        if name == "close_env":
            return {"ok": True}
        raise AssertionError(name)


class CreateConnectionRefusedOnceTransport(UnknownHandleOnceTransport):
    def __init__(self) -> None:
        super().__init__()
        self.reset_count = 1

    def call_tool(self, name, arguments, *, timeout_s=None):
        if name == "create_env" and self.create_count == 0:
            self.create_count += 1
            self.calls.append({"name": name, "arguments": dict(arguments)})
            return {
                "success": False,
                "error": "Worker request failed: connection refused",
            }
        return super().call_tool(name, arguments, timeout_s=timeout_s)


class RenderConnectionRefusedOnceTransport:
    def __init__(self) -> None:
        self.calls: list[JsonDict] = []

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append({"name": name, "arguments": dict(arguments)})
        if len(self.calls) == 1:
            return {"success": False, "error": "Worker request failed: connection refused"}
        return {"success": True, "cameras": [], "robot": {}}


class GroupedTransportError(RuntimeError):
    def __init__(self, message: str, exceptions: tuple[BaseException, ...]) -> None:
        super().__init__(message)
        self.exceptions = exceptions


class RemoteProtocolErrorOnceTransport:
    def __init__(self, failing_tool: str) -> None:
        self.failing_tool = failing_tool
        self.failed = False
        self.calls: list[JsonDict] = []

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append({"name": name, "arguments": dict(arguments)})
        if name == self.failing_tool and not self.failed:
            self.failed = True
            raise GroupedTransportError(
                "unhandled errors in a TaskGroup",
                (
                    RuntimeError(
                        "peer closed connection without sending complete message body "
                        "(incomplete chunked read)"
                    ),
                ),
            )
        if name == "create_env":
            return {
                "success": True,
                "handle": "env-protocol-retry",
                "session_id": "session-protocol-retry",
            }
        return {"success": True, "cameras": [], "robot": {}}


class FakeMcpResult:
    def __init__(
        self,
        *,
        content: list[JsonDict] | None = None,
        is_error: bool = False,
    ) -> None:
        self.content = content or []
        self.isError = is_error


class FakeToolListResult:
    def __init__(self, tools: list[JsonDict]) -> None:
        self.tools = tools


def test_close_simulator_mcp_env_calls_close_env() -> None:
    transport = FakeSimulatorMcpTransport({"ok": True})

    result = close_simulator_mcp_env(
        transport,
        handle="env-close",
        session_id="session-close",
        timeout_s=3.0,
    )

    assert result == {"ok": True}
    assert transport.calls == [
        {
            "name": "close_env",
            "arguments": {
                "handle": "env-close",
                "session_id": "session-close",
            },
            "timeout_s": 3.0,
        }
    ]


def test_episode_environment_close_claims_handle_once_across_threads() -> None:
    started = threading.Event()
    release = threading.Event()

    class BlockingCloseTransport(FakeSimulatorMcpTransport):
        def call_tool(self, name, arguments, *, timeout_s=None):
            self.calls.append({"name": name, "arguments": dict(arguments), "timeout_s": timeout_s})
            started.set()
            release.wait(timeout=1.0)
            return {"ok": True}

    transport = BlockingCloseTransport({"ok": True})
    environment = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/test-v0",
            session_id="session-close",
            handle="handle-close",
        ),
    )
    first_result = {}
    first = threading.Thread(
        target=lambda: first_result.update(environment.close()),
        daemon=True,
    )
    first.start()
    assert started.wait(timeout=0.5)

    second_result = environment.close()
    release.set()
    first.join(timeout=0.5)

    assert first_result == {"ok": True}
    assert second_result == {"ok": True, "skipped": True}
    assert len(transport.calls) == 1


def test_close_simulator_mcp_env_returns_structured_cleanup_error() -> None:
    result = close_simulator_mcp_env(
        FailingSimulatorMcpTransport(),
        handle="env-close",
        session_id="session-close",
    )

    assert result["ok"] is False
    assert result["error_type"] == "RuntimeError"
    assert result["handle"] == "env-close"
    assert result["session_id"] == "session-close"


def test_sse_transport_temporarily_bypasses_proxy_for_mcp_host(monkeypatch) -> None:
    observed: dict[str, JsonDict] = {}

    async def fake_list_tools(*, url: str, timeout_s: float | None) -> JsonDict:
        observed["list_tools"] = {
            "url": url,
            "timeout_s": timeout_s,
            "NO_PROXY": os.environ.get("NO_PROXY", ""),
            "no_proxy": os.environ.get("no_proxy", ""),
        }
        return {"tools": [], "tool_count": 0}

    async def fake_call_tool(
        *,
        url: str,
        tool_name: str,
        arguments: JsonDict,
        timeout_s: float | None,
    ) -> JsonDict:
        observed["call_tool"] = {
            "url": url,
            "tool_name": tool_name,
            "arguments": dict(arguments),
            "timeout_s": timeout_s,
            "NO_PROXY": os.environ.get("NO_PROXY", ""),
            "no_proxy": os.environ.get("no_proxy", ""),
        }
        return {"success": True}

    monkeypatch.setenv("NO_PROXY", "localhost")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.setattr(sim_mcp, "_list_sse_mcp_tools", fake_list_tools)
    monkeypatch.setattr(sim_mcp, "_call_sse_mcp_tool", fake_call_tool)

    transport = SseSimulatorMcpTransport("http://127.0.0.1:8773/sse")

    assert transport.list_tools(timeout_s=3.0)["tool_count"] == 0
    assert transport.call_tool("segment", {"prompt": "cube"}, timeout_s=4.0)["success"] is True

    for record in observed.values():
        assert "localhost" in record["NO_PROXY"]
        assert "127.0.0.1" in record["NO_PROXY"]
        assert "127.0.0.1:8773" in record["NO_PROXY"]
        assert record["NO_PROXY"] == record["no_proxy"]
    assert os.environ["NO_PROXY"] == "localhost"
    assert "no_proxy" not in os.environ


def test_sse_read_timeout_covers_long_tool_deadline() -> None:
    assert sim_mcp._sse_read_timeout_s(None) == 300.0
    assert sim_mcp._sse_read_timeout_s(30.0) == 300.0
    assert sim_mcp._sse_read_timeout_s(1200.0) == 1205.0


def test_sse_transport_unwraps_grouped_timeout(monkeypatch) -> None:
    class GroupedTransportError(RuntimeError):
        def __init__(self, nested: BaseException) -> None:
            super().__init__("SDK task group failed")
            self.exceptions = (nested,)

    async def fail_list_tools(*, url: str, timeout_s: float | None) -> JsonDict:
        del url, timeout_s
        raise GroupedTransportError(TimeoutError("connect timed out"))

    monkeypatch.setattr(sim_mcp, "_list_sse_mcp_tools", fail_list_tools)
    transport = SseSimulatorMcpTransport("http://sim.example/sse")

    with pytest.raises(sim_mcp.SimulatorMcpTransportError) as raised:
        transport.list_tools(timeout_s=15.0)

    assert raised.value.code == "simulator_mcp_transport_timeout"
    assert raised.value.operation == "list_tools"
    assert raised.value.cause_type == "TimeoutError"
    assert str(raised.value) == "list_tools failed: TimeoutError: connect timed out"


def test_default_simulator_mcp_binding_uses_remote_stable_tools() -> None:
    transport = FakeSimulatorMcpTransport({"cameras": [], "robot": {}})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
    )

    assert DEFAULT_SIMULATOR_MCP_TOOL_NAMES == (
        "create_simulator_env",
        "close_simulator_env",
        "observe",
        "ik_preview_check",
        "move_to",
        "follow_eef_trajectory",
        "gripper_control",
    )
    assert tools.can_execute("create_simulator_env")
    assert tools.can_execute("close_simulator_env")
    assert tools.can_execute("observe")
    assert tools.can_execute("ik_preview_check")
    assert tools.can_execute("move_to")
    assert tools.can_execute("follow_eef_trajectory")
    assert tools.can_execute("gripper_control")


def test_create_simulator_env_is_atomic_create_reset_and_state_sync(tmp_path: Path) -> None:
    transport = SequencedSimulatorMcpTransport(
        [
            {
                "success": True,
                "handle": "env-1",
                "session_id": "session-1",
                "env_id": "openeta/demo-v0",
            },
            {
                "success": True,
                "handle": "env-1",
                "session_id": "session-1",
                "task": "pick up alphabet soup and place it into basket",
                "cameras": [
                    {
                        "frame_id": "agentview",
                        "rgb_base64": PNG_1X1,
                        "depth_base64": PNG_1X1,
                        "intrinsics": {"fx": 618, "fy": 618, "cx": 256, "cy": 256},
                    }
                ],
                "robot": {},
            },
        ],
        url="http://sim.example/sse",
    )
    config = SimulatorMcpToolProxyConfig(
        image_output_root=tmp_path / "images",
        response_output_root=tmp_path / "responses",
    )
    callbacks: list[JsonDict] = []
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=config,
        tool_names=("create_simulator_env",),
        response_callback=lambda name, arguments, response: callbacks.append(
            {"name": name, "arguments": arguments, "response": response}
        ),
    )

    result = tools.call(
        "create_simulator_env",
        {
            "env_id": "openeta/demo-v0",
            "seed": 7,
            "session_id": "session-1",
            "include_objects": True,
        },
        metadata={"session_id": "agent-session-1"},
    )

    assert result.success is True
    assert [call["name"] for call in transport.calls] == ["create_env", "reset_env"]
    assert transport.calls[0]["arguments"] == {
        "env_id": "openeta/demo-v0",
        "render_mode": "rgb_array",
        "seed": 7,
        "image_width": 512,
        "image_height": 512,
        "session_id": "session-1",
        "include_objects": True,
    }
    assert transport.calls[1]["arguments"] == {
        "handle": "env-1",
        "seed": 7,
        "session_id": "session-1",
    }
    assert config.handle == "env-1"
    assert config.session_id == "session-1"
    assert result.content.endswith("Assigned task: pick up alphabet soup and place it into basket")
    assert result.details["outputs"]["assigned_task"] == (
        "pick up alphabet soup and place it into basket"
    )
    environment = result.details["outputs"]["environment"]
    assert environment["assigned_task"] == ("pick up alphabet soup and place it into basket")
    assert environment["dashboard_url"] == "http://sim.example/session/session-1"
    camera = result.details["outputs"]["initial_observation"]["cameras"][0]
    assert camera["anygrasp_intrinsics"]["scale"] == 1000.0
    assert Path(camera["rgb_path"]).exists()
    assert Path(camera["rgb_path"]).relative_to(tmp_path / "images").parts[0] == ("agent-session-1")
    assert (
        Path(result.details["outputs"]["create_response"]["response_path"])
        .relative_to(tmp_path / "responses")
        .parts[0]
        == "agent-session-1"
    )
    assert result.details["state_delta"]["simulator_environment"]["handle"] == "env-1"
    assert [callback["name"] for callback in callbacks] == ["create_env", "reset_env"]


def test_create_simulator_env_requires_env_id() -> None:
    transport = FakeSimulatorMcpTransport({"success": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        tool_names=("create_simulator_env",),
    )

    result = tools.call("create_simulator_env", {})

    assert result.success is False
    assert result.details["diagnostics"][0]["code"] == "missing_env_id"
    assert transport.calls == []


def test_close_simulator_env_closes_and_clears_bound_handle() -> None:
    transport = FakeSimulatorMcpTransport({"ok": True})
    config = SimulatorMcpToolProxyConfig(
        session_id="session-close",
        handle="env-close",
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=config,
        tool_names=("close_simulator_env",),
    )

    result = tools.call("close_simulator_env")

    assert result.success is True
    assert result.details["outputs"]["closed"] is True
    assert config.handle == ""
    assert transport.calls == [
        {
            "name": "close_env",
            "arguments": {
                "handle": "env-close",
                "session_id": "session-close",
            },
            "timeout_s": 30.0,
        }
    ]


def test_create_simulator_env_rejects_invalid_dimensions() -> None:
    transport = FakeSimulatorMcpTransport({"success": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        tool_names=("create_simulator_env",),
    )

    result = tools.call(
        "create_simulator_env",
        {"env_id": "openeta/demo-v0", "image_width": -1},
    )

    assert result.success is False
    assert result.details["diagnostics"][0]["code"] == "simulator_mcp_argument_error"
    assert transport.calls == []


def test_observe_proxy_uses_remote_render_env_tool() -> None:
    transport = FakeSimulatorMcpTransport({"cameras": [], "robot": {}})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-0", handle="env-0"),
        tool_names=("observe",),
    )

    result = tools.call("observe", {"reason": "refresh"})

    assert result.success is True
    assert transport.calls == [
        {
            "name": "render_env",
            "arguments": {
                "handle": "env-0",
                "session_id": "session-0",
            },
            "timeout_s": 120.0,
        }
    ]


def test_observe_proxy_exposes_metric_intrinsics_for_depth_png(tmp_path: Path) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "cameras": [
                {
                    "frame_id": "agentview",
                    "rgb_base64": PNG_1X1,
                    "depth_base64": PNG_1X1,
                    "width": 512,
                    "height": 512,
                    "intrinsics": {
                        "fx": 618.0386719675123,
                        "fy": 618.0386719675123,
                        "cx": 256,
                        "cy": 256,
                    },
                }
            ],
            "robot": {
                "end_effector_pose": {
                    "xyz": [0.1, 0.2, 0.3],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "gripper_state": {"open": False},
            },
            "objects": [
                {
                    "name": "alphabet_soup_1",
                    "category": "alphabet_soup",
                    "position": [-0.1, -0.2, 0.47],
                }
            ],
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-0",
            handle="env-0",
            image_output_root=tmp_path / "image",
            response_output_root=tmp_path / "tool_result",
        ),
        tool_names=("observe",),
    )

    result = tools.call("observe", {"reason": "refresh"})

    assert result.success is True
    camera = result.details["outputs"]["response"]["cameras"][0]
    assert camera["intrinsics"]["scale"] == 1000.0
    assert camera["anygrasp_intrinsics"]["scale"] == 1000.0
    response_path = Path(result.details["outputs"]["response"]["response_path"])
    payload = json.loads(response_path.read_text(encoding="utf-8"))
    assert payload["cameras"][0]["intrinsics"]["scale"] == 1000.0
    assert payload["cameras"][0]["anygrasp_intrinsics"] == {
        "fx": 618.0386719675123,
        "fy": 618.0386719675123,
        "cx": 256,
        "cy": 256,
        "scale": 1000.0,
    }
    observation = result.details["state_delta"]["observation"]
    assert observation["robot"]["end_effector_pose"]["xyz"] == [0.1, 0.2, 0.3]
    assert observation["robot"]["gripper_state"]["open"] is False
    assert observation["objects"] == []
    assert payload.get("objects") == []
    assert "alphabet_soup_1" not in response_path.read_text(encoding="utf-8")
    response_summary = result.details["outputs"]["response"]["observation_summary"]
    assert response_summary["objects"] == []


def test_control_tool_proxy_forwards_to_simulator_mcp_and_materializes_images(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "observation": {
                "task": "pick cube",
                "cameras": [
                    {
                        "frame_id": "agentview",
                        "rgb_base64": PNG_1X1,
                        "width": 1,
                        "height": 1,
                    }
                ],
                "robot": {},
            },
            "reward": 0.2,
            "terminated": False,
            "truncated": False,
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-1",
            handle="env-1",
            image_output_root=tmp_path,
        ),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {
            "target_pose": {"xyz": [0.1, 0.2, 0.3]},
        },
    )

    assert result.success is True
    assert transport.calls == [
        {
            "name": "move_to",
            "arguments": {
                "x": 0.1,
                "y": 0.2,
                "z": 0.3,
                "handle": "env-1",
                "session_id": "session-1",
            },
            "timeout_s": 120.0,
        }
    ]
    assert result.details["result_type"] == "world_mutating"
    assert result.details["outputs"]["mcp"]["tool"] == "move_to"
    response = result.details["outputs"]["response"]
    camera = response["cameras"][0]
    assert "rgb_base64" not in camera
    assert camera["rgb_ref"] == "observation.cameras.0.agentview.rgb"
    assert Path(camera["rgb_path"]).exists()
    assert Path(result.details["artifacts"][0]["path"]).exists()
    assert Path(response["response_path"]).exists()
    assert result.details["state_delta"]["reward"] == 0.2
    assert (
        result.details["state_delta"]["observation"]["cameras"][0]["rgb_path"]
        == (camera["rgb_path"])
    )
    assert PNG_1X1 not in json.dumps(result.details)


def test_control_tool_proxy_promotes_explicit_remote_terminal_error(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "ok": False,
            "error": (
                "Worker-local controller goal failed: ValueError: "
                "executing action in terminated episode"
            ),
            "motion_summary": {
                "reached_target": False,
                "steps_executed": 0,
                "stop_reason": "controller_error",
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-terminated",
            handle="env-terminated",
            response_output_root=tmp_path,
        ),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.2, 0.3]}})

    assert result.success is False
    assert result.details["environment_receipt"]["terminated"] is True
    assert result.details["state_delta"]["terminated"] is True
    assert result.details["diagnostics"][0]["code"] == "remote_episode_terminated"
    assert result.details["diagnostics"][0]["failure_class"] == "environment_terminal"
    assert "already terminated" in result.content
    assert "do not retry" in result.content


def test_move_to_proxy_converts_world_rotation_matrix_to_mcp_euler_angles() -> None:
    transport = FakeSimulatorMcpTransport({"success": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {
            "target_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        },
    )

    assert result.success is True
    arguments = transport.calls[0]["arguments"]
    assert arguments["roll"] == 0.0
    assert arguments["pitch"] == 0.0
    assert arguments["yaw"] == 0.0


def test_move_to_wrist_viewpoint_returns_fresh_evidence_handoff() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "start": {"xyz": [0.0, 0.0, 0.3]},
            "end": {"xyz": [0.1, 0.2, 0.3]},
            "target": {"x": 0.1, "y": 0.2, "z": 0.3},
            "steps_executed": 4,
            "reached_target": True,
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {
            "target_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.3],
                "waypoint_role": "wrist_observation_viewpoint",
                "viewpoint_candidate_id": "wrist_view_00",
                "camera_frame_id": "robot0_eye_in_hand",
                "compiled_grasp_id": "compiled-1",
            }
        },
    )

    handoff = result.details["outputs"]["post_motion_evidence_handoff"]
    assert handoff["status"] == "fresh_wrist_packet_expected"
    assert handoff["materially_new_view"] is True
    assert handoff["camera_frame_id"] == "robot0_eye_in_hand"
    assert handoff["compiled_grasp_id"] == "compiled-1"
    assert handoff["viewpoint_candidate_id"] == "wrist_view_00"
    assert handoff["fresh_packet_source"] == (
        "current_observation.source_packet_id in the next planner context"
    )
    assert handoff == result.details["outputs"]["response"][
        "post_motion_evidence_handoff"
    ]
    assert "did not refine the older contact pose" in result.content
    assert "current_observation.source_packet_id" in result.content


def test_ik_preview_proxy_preserves_actionable_reachability_summary() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "ok": False,
            "success": False,
            "status": "unreachable",
            "kinematic_status": "unreachable",
            "feasible": False,
            "reason_code": "full_pose_infeasible",
            "message": "Position and orientation cannot be satisfied together.",
            "position_only_reachable": True,
            "orientation_only_reachable": True,
            "target": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
            "tolerances": {
                "max_axis_position_error_m": 0.002,
                "orientation_error_rad": 0.05,
            },
            "best_candidate": {
                "joint_positions": [0.0] * 7,
                "max_axis_position_error_m": 0.011,
                "orientation_error_rad": 0.05,
            },
            "collision": {"checked": False},
            "path": {"checked": False},
            "suggestions": ["relax_target_orientation"],
            # A legacy backend is not authoritative for execution permission.
            "motion_execution_ref": {
                "tool": "move_to",
                "ik_receipt_id": "backend-invented",
            },
            "execution_authorization": {"authorized_for_move_to": True},
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("ik_preview_check",),
    )

    result = tools.call(
        "ik_preview_check",
        {
            "target_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.3],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        },
    )

    assert result.success is False
    assert result.details["operational_success"] is True
    assert transport.calls[0]["name"] == "ik_preview_check"
    assert transport.calls[0]["arguments"]["handle"] == "env-1"
    reachability = result.details["outputs"]["reachability"]
    assert result.details["outputs"]["feasible"] is False
    assert reachability["status"] == "unreachable"
    assert reachability["reason_code"] == "full_pose_infeasible"
    assert reachability["position_only_reachable"] is True
    assert reachability["best_candidate"]["max_axis_position_error_m"] == 0.011
    receipt = result.details["outputs"]["ik_preview_receipt"]
    assert receipt["classification"] == "repairable"
    assert receipt["orientation_policy"] == "explicit_orientation"
    assert len(receipt["target_signature"]) == 24
    assert len(receipt["pose_policy_signature"]) == 24
    assert result.details["diagnostics"][0]["candidate_rejection"] is True
    assert result.details["semantic_outcome"] == "ik_repairable"
    authorization = result.details["outputs"]["execution_authorization"]
    assert authorization["authorized_for_move_to"] is False
    assert authorization["same_pose_retry_disposition"] == (
        "requires_materially_changed_pose_or_policy"
    )
    assert "motion_execution_ref" not in result.details["outputs"]
    assert "motion_execution_ref" not in result.details["outputs"]["response"]
    assert "motion_execution_ref" not in result.details["outputs"]["mcp"]
    assert "Do not pass ik_receipt_id=" in result.content
    assert "Execution reference:" not in result.content
    assert any(
        option["action"] == "preview_modified_pose"
        for option in result.details["recovery_options"]
    )
    assert "cannot be satisfied" in result.content
    contract = build_default_tool_contract_catalog(tools.list()).get("ik_preview_check")
    assert check_tool_result_conformance(contract, result.details) == ()


def test_ik_preview_proxy_preserves_execution_seed_quality_for_agent() -> None:
    quality = {
        "risk_level": "critical",
        "selected_joint_margin_rad": 0.0017,
        "robust_margin_threshold_rad": 0.1,
        "distant_robust_solution_count": 5,
        "interpretation": "Compare another grasp candidate before motion.",
    }
    transport = FakeSimulatorMcpTransport(
        {
            "ok": True,
            "success": True,
            "status": "reachable",
            "kinematic_status": "reachable",
            "feasible": True,
            "reason_code": "ik_solution_found",
            "message": "Endpoint feasible but execution-fragile.",
            "target": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
            "best_candidate": {
                "joint_positions": [0.0] * 7,
                "joint_margin_min_rad": 0.0017,
            },
            "execution_seed_quality": quality,
            "collision": {"checked": True},
            "path": {"checked": False},
            "suggestions": ["compare_alternative_grasp_candidate_before_motion"],
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("ik_preview_check",),
    )

    result = tools.call(
        "ik_preview_check",
        {"target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]}},
    )

    assert result.success is True
    assert result.details["outputs"]["reachability"][
        "execution_seed_quality"
    ] == quality
    assert result.details["outputs"]["ik_preview_receipt"]["reachability"][
        "execution_seed_quality"
    ] == quality
    assert result.details["outputs"]["execution_authorization"][
        "authorized_for_move_to"
    ] is True
    assert result.details["outputs"]["motion_execution_ref"]["ik_receipt_id"]
    world_effect = result.details["outputs"]["world_effect"]
    assert world_effect == result.details["outputs"]["response"]["world_effect"]
    assert world_effect["world_mutated"] is False
    assert world_effect["eef_pose_unchanged"] is True
    assert world_effect["robot_motion_epoch_unchanged"] is True
    assert world_effect["object_scene_epoch_unchanged"] is True
    assert "did not move the robot" in world_effect["interpretation"]
    assert "did not move the robot" in result.details["outputs"][
        "motion_execution_ref"
    ]["instruction"]
    assert "Execution reference:" in result.content
    assert "did not move the robot" in result.content
    assert "execution-fragile" in result.content


def test_ik_preview_collision_backend_gap_returns_exact_downgrade_recovery() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "ok": True,
            "success": True,
            "status": "unknown",
            "kinematic_status": "reachable",
            "feasible": None,
            "reason_code": "endpoint_collision_check_unavailable",
            "message": (
                "IK succeeded, but the requested endpoint collision check was unavailable."
            ),
            "position_only_reachable": True,
            "orientation_only_reachable": True,
            "target": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
            "best_candidate": {
                "joint_positions": [0.0] * 7,
                "max_axis_position_error_m": 0.0,
                "orientation_error_rad": 0.0,
            },
            "collision": {
                "checked": False,
                "reason": "cuRobo not installed or CUDA unavailable",
            },
            "path": {"checked": False},
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("ik_preview_check",),
    )

    result = tools.call(
        "ik_preview_check",
        {
            "target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
            "preserve_current_orientation": True,
            "check_endpoint_collision": True,
        },
        metadata={
            "_controller_capabilities_resolver": lambda: {
                "controller_id": "mink.robosuite_joint_velocity",
                "goal_executor": "openeta.worker_mink_goal.v1",
                "collision_scope": (
                    "worker_per_step_pre_actuation_and_post_step_configuration"
                ),
                "motion_owns_trajectory_world_collision": True,
            }
        },
    )

    assert result.success is True
    assert (
        result.details["semantic_outcome"]
        == "ik_kinematically_feasible_collision_deferred"
    )
    delegation = result.details["outputs"]["motion_collision_delegation"]
    assert delegation["available_for_matching_move"] is True
    assert delegation["controller_id"] == "mink.robosuite_joint_velocity"
    assert result.details["outputs"]["ik_preview_receipt"][
        "motion_collision_delegation"
    ] == delegation
    assert result.details["outputs"]["execution_authorization"][
        "authorized_for_move_to"
    ] is True
    assert "motion_execution_ref" in result.details["outputs"]
    actions = {
        option["action"]: option for option in result.details["recovery_options"]
    }
    assert actions["execute_exact_pose_with_verified_motion_collision"][
        "parameters"
    ] == {"enable_collision_check": True}
    assert "repeat_exact_pose_with_kinematics_only" not in actions
    assert "inspect_fresh_observation" not in actions
    assert "preview_modified_pose" not in actions
    assert "current environment controller explicitly owns" in result.content


def test_ik_preview_collision_backend_gap_does_not_invent_controller_coverage() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "ok": True,
            "success": True,
            "status": "unknown",
            "kinematic_status": "reachable",
            "feasible": None,
            "reason_code": "endpoint_collision_check_unavailable",
            "message": "IK succeeded, but endpoint collision checking was unavailable.",
            "target": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
            "best_candidate": {"joint_positions": [0.0] * 7},
            "collision": {"checked": False},
            "path": {"checked": False},
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("ik_preview_check",),
    )

    result = tools.call(
        "ik_preview_check",
        {
            "target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
            "preserve_current_orientation": True,
            "check_endpoint_collision": True,
        },
    )

    delegation = result.details["outputs"]["motion_collision_delegation"]
    assert delegation["applicable"] is True
    assert delegation["available_for_matching_move"] is False
    actions = {
        option["action"] for option in result.details["recovery_options"]
    }
    assert "execute_exact_pose_with_verified_motion_collision" not in actions
    assert "delegate_collision_to_verified_motion_controller" in actions
    assert result.details["outputs"]["execution_authorization"][
        "authorized_for_move_to"
    ] is False
    assert "motion_execution_ref" not in result.details["outputs"]
    assert "Do not pass ik_receipt_id=" in result.content
    assert "does not declare the required" in result.content


def test_ik_preview_matches_move_to_uncalibrated_grasp_orientation_rule() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "status": "reachable",
            "kinematic_status": "reachable",
            "feasible": True,
            "reason_code": "ik_solution_found",
            "message": "IK solution found.",
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("ik_preview_check",),
    )

    result = tools.call(
        "ik_preview_check",
        {
            "target_pose": {
                "id": "grasp_003",
                "rank": 3,
                "frame": "world",
                "translation_xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        },
    )

    assert result.success is True
    assert {"roll", "pitch", "yaw"}.isdisjoint(transport.calls[0]["arguments"])


def test_move_to_proxy_rejects_unsupported_speed_parameter() -> None:
    transport = FakeSimulatorMcpTransport({"success": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {"target_pose": {"xyz": [0.1, 0.2, 0.3]}, "speed": "slow"},
    )

    assert result.success is False
    assert transport.calls == []
    assert result.details["diagnostics"][0]["code"] == "simulator_mcp_argument_error"
    assert "speed" in result.details["diagnostics"][0]["message"]


def test_move_to_preserves_orientation_for_uncalibrated_grasp_deployment() -> None:
    transport = FakeSimulatorMcpTransport({"success": True, "reached_target": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {
            "target_pose": {
                "id": "grasp_003",
                "frame": "world",
                "translation_xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "gripper_tip_position_xyz": [0.11, 0.22, 0.33],
                "depth": 0.04,
            },
        },
    )

    assert result.success is True
    arguments = transport.calls[0]["arguments"]
    assert transport.calls[0]["name"] == "move_to"
    assert [arguments["x"], arguments["y"], arguments["z"]] == [0.1, 0.2, 0.3]
    assert {"roll", "pitch", "yaw"}.isdisjoint(arguments)
    assert result.details["outputs"]["mcp"]["target_orientation_mode"] == ("preserve_current")
    assert set(arguments).issubset(
        {
            "handle",
            "session_id",
            "x",
            "y",
            "z",
            "roll",
            "pitch",
            "yaw",
            "num_steps",
            "tolerance",
            "ori_tolerance",
            "enable_collision_check",
        }
    )
    assert {
        "candidate_id",
        "approach_x",
        "approach_y",
        "approach_z",
        "target_reference",
        "grasp_phase",
    }.isdisjoint(arguments)


def test_move_to_privately_forwards_host_resolved_contact_authorization() -> None:
    transport = FakeSimulatorMcpTransport({"success": True, "reached_target": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )
    target_pose = {
        "frame": "world",
        "xyz": [0.1, 0.2, 0.3],
        "compiled_grasp_id": "compiled-1",
        "waypoint_role": "grasp_contact",
    }
    authorization = {
        "schema_version": "openeta.contact_authorization.v1",
        "compiled_grasp_id": "compiled-1",
        "waypoint_role": "grasp_contact",
        "target_anchor_world_xyz": [0.1, 0.2, 0.25],
        "object_scene_epoch": 2,
    }

    result = tools.call(
        "move_to",
        {"target_pose": target_pose},
        metadata={"_contact_authorization_resolver": lambda pose: authorization},
    )

    assert result.success is True
    assert transport.calls[0]["arguments"]["contact_authorization"] == authorization
    assert "contact_authorization" not in result.details["parameters"]


def test_move_to_privately_forwards_host_resolved_ik_execution_seed() -> None:
    transport = FakeSimulatorMcpTransport({"success": True, "reached_target": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )
    parameters = {
        "target_pose": {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.3],
            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
    }
    seed = {
        "schema_version": "openeta.ik_execution_seed.v1",
        "receipt_id": "ik-1",
        "joint_positions": [0.1] * 7,
    }

    result = tools.call(
        "move_to",
        parameters,
        metadata={"_ik_execution_seed_resolver": lambda request: seed},
    )

    assert result.success is True
    assert transport.calls[0]["arguments"]["ik_execution_seed"] == seed
    assert "ik_execution_seed" not in result.details["parameters"]


def test_move_to_can_map_anygrasp_world_pose_to_panda_eef_orientation() -> None:
    transport = FakeSimulatorMcpTransport({"success": True, "reached_target": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-1",
            handle="env-1",
            forward_grasp_candidate_orientation=True,
        ),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {
            "target_pose": {
                "id": "grasp_003",
                "rank": 3,
                "frame": "world",
                "translation_xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        },
    )

    assert result.success is True
    arguments = transport.calls[0]["arguments"]
    assert [arguments[axis] for axis in ("roll", "pitch", "yaw")] == [
        90.0,
        0.0,
        90.0,
    ]
    assert result.details["outputs"]["mcp"]["target_orientation_mode"] == ("graspnet_to_panda_eef")


def test_move_to_preserves_orientation_for_anyplace_pose_by_default() -> None:
    transport = FakeSimulatorMcpTransport({"success": True, "reached_target": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {
            "target_pose": {
                "id": "place_grasp_000",
                "source_grasp_id": "grasp_003",
                "frame": "world",
                "translation_xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        },
    )

    assert result.success is True
    assert {"roll", "pitch", "yaw"}.isdisjoint(transport.calls[0]["arguments"])
    assert result.details["outputs"]["mcp"]["target_orientation_mode"] == ("preserve_current")


def test_move_to_preserves_anyplace_orientation_when_grasp_forwarding_enabled() -> None:
    transport = FakeSimulatorMcpTransport({"success": True, "reached_target": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-1",
            handle="env-1",
            forward_grasp_candidate_orientation=True,
        ),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {
            "target_pose": {
                "id": "place_grasp_000",
                "source_grasp_id": "grasp_003",
                "frame": "world",
                "translation_xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            }
        },
    )

    assert result.success is True
    assert {"roll", "pitch", "yaw"}.isdisjoint(transport.calls[0]["arguments"])
    assert result.details["outputs"]["mcp"]["target_orientation_mode"] == ("preserve_current")


def test_move_to_proxy_preserves_transport_success_but_marks_target_miss_operationally(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "collision": {
                "detected": True,
                "world_collision": True,
                "message": "Collision detected at step 3",
            },
            "start": {"xyz": [0.0, 0.0, 0.5]},
            "end": {"xyz": [0.02, 0.0, 0.5]},
            "target": {"x": 0.2, "y": 0.0, "z": 0.5},
            "steps_executed": 3,
            "reached_target": False,
            "position_error_m": 0.18,
            "max_axis_position_error_m": 0.18,
            "orientation_error_deg": 12.5,
            "stop_reason": "collision_detected",
            "reward": 0.0,
            "terminated": False,
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-collision",
            handle="env-collision",
            response_output_root=tmp_path,
        ),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.2, 0.0, 0.5]}})

    assert result.success is True
    assert result.details["operational_success"] is False
    assert result.details["semantic_outcome"] == "target_not_reached"
    assert result.details["diagnostics"][0]["code"] == "simulator_mcp_collision"
    assert result.details["diagnostics"][0]["position_error_m"] == pytest.approx(0.18)
    assert {item["action"] for item in result.details["recovery_options"]} == {
        "classify_collision_visually_before_replanning",
        "plan_ik_checked_raised_or_lateral_detour",
        "replan_from_actual_pose",
    }
    assert "stopped for collision" in result.content
    assert "host-private checker" in result.content
    motion = result.details["outputs"]["response"]["motion_summary"]
    assert motion["collision"]["detected"] is True
    assert motion["reached_target"] is False
    assert motion["position_error_m"] == pytest.approx(0.18)
    assert motion["max_axis_position_error_m"] == pytest.approx(0.18)
    assert motion["orientation_error_deg"] == pytest.approx(12.5)
    assert motion["stop_reason"] == "collision_detected"
    assert result.details["state_delta"]["motion"] == motion
    pose_feedback = result.details["outputs"]["pose_feedback"]
    assert pose_feedback["schema_version"] == "openeta.eef_pose_feedback.v1"
    assert pose_feedback["requested_xyz"] == [0.2, 0.0, 0.5]
    assert pose_feedback["actual_xyz"] == [0.02, 0.0, 0.5]
    assert pose_feedback["position_error_m"] == pytest.approx(0.18)
    assert "object-relative contact" in pose_feedback["interpretation"]
    execution_receipt = result.details["host_execution_receipt"]
    assert execution_receipt["schema_version"] == (
        "openeta.resolved_tool_execution.v1"
    )
    assert execution_receipt["tool"] == "move_to"
    assert execution_receipt["dispatch_status"] == "response_received"
    assert execution_receipt["parameters"]["target_pose"] == {
        "xyz": [0.2, 0.0, 0.5]
    }
    assert result.details["host_provenance"]["authority"] == "environment"


def test_move_to_promotes_structured_controller_boundary_failure() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "error": "mink_qp_no_solution",
            "start": {"xyz": [0.0, 0.0, 0.3]},
            "end": {"xyz": [0.05, 0.0, 0.25]},
            "target": {"x": 0.1, "y": 0.0, "z": 0.2},
            "steps_executed": 53,
            "reached_target": False,
            "controller_failure": {
                "schema_version": "openeta.controller_failure.v1",
                "code": "constraint_escape_preview_rejected",
                "current_minimum_distance_m": 0.012,
                "predicted_minimum_distance_m": 0.011,
                "recovery": "Choose a waypoint that increases clearance.",
            },
            "collision": {
                "detected": False,
                "trajectory_checked": True,
                "world_checked": True,
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-qp", handle="env-qp"),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.0, 0.2]}})

    failure = result.details["outputs"]["motion_summary"]["controller_failure"]
    assert failure["code"] == "constraint_escape_preview_rejected"
    assert "controller_failure=constraint_escape_preview_rejected" in result.content
    assert {item["action"] for item in result.details["recovery_options"]} == {
        "change_wrist_orientation_or_candidate",
        "exit_reported_controller_boundary",
    }
    assert "Recommended recovery: exit_reported_controller_boundary" in result.content
    assert '"preserve_current_orientation":true' in result.content
    serialized = json.dumps(result.details)
    assert "current_minimum_distance_m" not in serialized
    assert "predicted_minimum_distance_m" not in serialized


def test_move_to_exposes_tentative_attachment_refresh_to_agent() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "start": {"xyz": [0.0, 0.0, 0.2]},
            "end": {"xyz": [0.0, 0.0, 0.26]},
            "target": {"x": 0.0, "y": 0.0, "z": 0.26},
            "reached_target": True,
            "attachment_proxy_receipt": {
                "schema_version": "openeta.attachment_proxy_receipt.v1",
                "status": "tentative",
                "reason": "awaiting_independent_co_motion_evidence",
                "target_object_name": "salad_dressing_1",
                "attachment_proven": False,
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-lift", handle="env-lift"),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.0, 0.0, 0.26]}})

    receipt = result.details["outputs"]["attachment_proxy_receipt"]
    assert receipt["status"] == "tentative"
    assert receipt["attachment_proven"] is False
    assert result.details["semantic_outcome"] == "requires_attachment_confirmation"
    assert {item["action"] for item in result.details["recovery_options"]} == {
        "inspect_fresh_dual_view",
        "withhold_transport_until_visual_confirmation",
    }
    assert "Carried-object proxy feedback" in result.content
    assert "awaiting_independent_co_motion_evidence" in result.content
    assert "attachment_proven=false" in result.content


def test_move_to_proxy_allows_unchanged_baseline_contact() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "reached_target": True,
            "collision": {
                "available": True,
                "detected": True,
                "new_or_worsened": False,
                "trajectory_checked": True,
                "world_checked": True,
                "world_object_count": 3,
                "pairs": [],
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.2, 0.3]}})

    assert result.success is True
    assert result.details["diagnostics"] == []
    assert result.details["outputs"]["collision_coverage"]["coverage_complete"] is True
    motion = result.details["outputs"]["response"]["motion_summary"]
    assert motion["collision"]["detected"] is True
    assert motion["reached_target"] is True
    assert result.details["state_delta"]["motion"] == motion


def test_move_to_proxy_exposes_unknown_collision_scope_without_claiming_safety(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "reached_target": True,
            "collision": {
                "detected": False,
                "trajectory_checked": False,
                "world_checked": False,
                "world_object_count": 0,
            },
            "end": {"xyz": [0.1, 0.2, 0.3]},
            "target": {"xyz": [0.1, 0.2, 0.3]},
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-coverage",
            handle="env-coverage",
            response_output_root=tmp_path,
        ),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.2, 0.3]}})

    coverage = result.details["outputs"]["collision_coverage"]
    assert coverage["coverage_status"] == "remote_collision_result_without_coverage"
    assert coverage["coverage_complete"] is False
    assert coverage["collision_detected"] is False
    assert "does not prove" in coverage["interpretation"]
    assert result.details["diagnostics"][-1]["code"] == "collision_coverage_incomplete"


def test_move_to_target_not_reached_is_not_operational_success() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "reached_target": False,
            "collision": {"trajectory_checked": False, "world_checked": False},
            "start": {"xyz": [0.0, 0.0, 0.3]},
            "end": {"xyz": [0.02, 0.0, 0.3]},
            "target": {"xyz": [0.1, 0.0, 0.3]},
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-miss", handle="env-miss"),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.0, 0.3]}})

    assert result.success is True
    assert result.details["semantic_outcome"] == "target_not_reached"
    assert result.details["operational_success"] is False
    assert result.details["recovery_options"]


def test_move_to_zero_step_target_hit_reports_unchanged_physical_view() -> None:
    pose = {"xyz": [0.1, 0.0, 0.3]}
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "reached_target": True,
            "steps_executed": 0,
            "stop_reason": "target_reached",
            "start": pose,
            "end": pose,
            "target": pose,
            "collision": {
                "trajectory_checked": True,
                "world_checked": True,
                "world_object_count": 4,
            },
            "controller_receipt": {
                "schema_version": "openeta.controller_execution_receipt.v1",
                "controller_id": "mink.robosuite_joint_velocity",
                "command_interface": "joint_velocity",
                "goal_executor": "openeta.worker_mink_goal.v1",
                "execution_location": "bench_worker",
                "orientation_policy": "preserve_current",
                "iteration_budget": 100,
                "steps_executed": 0,
                "stop_reason": "target_reached",
                "reached_target": True,
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-noop",
            handle="env-noop",
        ),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {"target_pose": pose, "enable_collision_check": True},
    )

    assert result.success is True
    assert result.details["operational_success"] is True
    assert result.details["semantic_outcome"] == "target_already_within_tolerance"
    assert result.details["outputs"]["motion_outcome"] == "no_state_change"
    assert "physical camera viewpoint did NOT change" in result.content
    assert {
        item["action"] for item in result.details["recovery_options"]
    } == {
        "consume_existing_visual_evidence",
        "propose_materially_distinct_checked_endpoint",
    }


def test_zero_step_attached_collision_is_promoted_to_top_level_content(tmp_path: Path) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "reached_target": False,
            "steps_executed": 0,
            "collision": {
                "detected": True,
                "collision_type": "attached_object_world",
                "attached_object": "bottle_1",
                "obstacle": "box_1",
                "message": "Attached object bottle_1 would collide with box_1.",
            },
            "start": {"xyz": [0.0, 0.0, 0.15]},
            "end": {"xyz": [0.0, 0.0, 0.15]},
            "target": {"x": 0.0, "y": 0.0, "z": 0.25},
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-deadlock",
            handle="env-deadlock",
            response_output_root=tmp_path,
        ),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.0, 0.0, 0.25]}})

    assert "host-private checker" in result.content
    assert "No controller step executed" in result.content
    assert "bottle_1" not in json.dumps(result.details)
    assert "box_1" not in json.dumps(result.details)
    response_path = Path(result.details["outputs"]["response"]["response_path"])
    artifact_text = response_path.read_text(encoding="utf-8")
    assert "bottle_1" not in artifact_text
    assert "box_1" not in artifact_text
    assert not any(
        item.get("code") == "collision_coverage_incomplete"
        for item in result.details["diagnostics"]
    )
    recovery = {
        item["action"]: item for item in result.details["recovery_options"]
    }
    escape = recovery["escape_current_collision_boundary"]
    assert escape["parameters"] == {
        "actual_eef_xyz": [0.0, 0.0, 0.15],
        "preserve_current_orientation": True,
        "enable_collision_check": True,
    }
    assert escape["evidence"] == {
        "collision_detected": True,
        "collision_class": "attached_object_world",
        "steps_executed": 0,
        "feedback_scope": "host_private_geometry_withheld",
    }
    assert "Re-segmenting the same object does not move the robot" in recovery[
        "consume_returned_motion_evidence"
    ]["reason"]


def test_partial_motion_collision_recommends_checked_detour_from_actual_pose(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "reached_target": False,
            "steps_executed": 23,
            "collision": {
                "detected": True,
                "geom1_name": "robot0_link7_collision",
                "geom2_name": "tomato_sauce_1_g4",
                "trajectory_checked": True,
                "world_checked": True,
                "world_object_count": 4,
            },
            "start": {"xyz": [0.0, 0.0, 0.30]},
            "end": {"xyz": [0.08, -0.04, 0.24]},
            "target": {"xyz": [0.12, -0.08, 0.18]},
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-partial-collision",
            handle="env-partial-collision",
            response_output_root=tmp_path,
        ),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.12, -0.08, 0.18]}})

    recovery = {item["action"]: item for item in result.details["recovery_options"]}
    classify = recovery["classify_collision_visually_before_replanning"]
    assert classify["evidence"]["collision_detected"] is True
    assert classify["evidence"]["feedback_scope"] == (
        "host_private_geometry_withheld"
    )
    assert classify["evidence"]["actual_eef_xyz"] == [0.08, -0.04, 0.24]
    serialized = json.dumps(result.details)
    assert "robot0_link7_collision" not in serialized
    assert "tomato_sauce_1_g4" not in serialized
    response_path = Path(result.details["outputs"]["response"]["response_path"])
    artifact_text = response_path.read_text(encoding="utf-8")
    assert "robot0_link7_collision" not in artifact_text
    assert "tomato_sauce_1_g4" not in artifact_text
    detour = recovery["plan_ik_checked_raised_or_lateral_detour"]
    assert detour["parameters"] == {
        "start_from_actual_eef_xyz": [0.08, -0.04, 0.24],
        "preserve_current_orientation_for_clearance": True,
        "preview_each_waypoint_separately": True,
        "execute_with": "follow_eef_trajectory",
        "enable_collision_check": True,
    }
    assert "arithmetic midpoint" in detour["reason"]


def test_ik_proxy_preserves_configuration_collision_scope() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "status": "reachable",
            "feasible": True,
            "collision": {
                "checked": True,
                "detected": False,
                "self_checked": True,
                "trajectory_checked": False,
                "world_checked": False,
                "world_object_count": 0,
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-ik", handle="env-ik"),
        tool_names=("ik_preview_check",),
    )

    result = tools.call(
        "ik_preview_check",
        {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
    )

    coverage = result.details["outputs"]["collision_coverage"]
    assert coverage["coverage_status"] == "endpoint_only"
    assert coverage["endpoint_checked"] is True
    assert coverage["world_checked"] is False
    assert coverage["coverage_complete"] is False
    assert result.details["diagnostics"][-1]["code"] == "collision_coverage_incomplete"


def test_ik_proxy_fails_world_coverage_closed_on_world_update_error() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "status": "reachable",
            "feasible": True,
            "collision": {
                "available": True,
                "checked": True,
                "detected": False,
                "world_checked": True,
                "world_update_error": "synthetic cuRobo update failure",
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-ik-world-error",
            handle="env-ik-world-error",
        ),
        tool_names=("ik_preview_check",),
    )

    result = tools.call(
        "ik_preview_check",
        {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
    )

    coverage = result.details["outputs"]["collision_coverage"]
    assert coverage["endpoint_checked"] is True
    assert coverage["world_checked"] is False
    assert coverage["coverage_complete"] is False
    assert coverage["coverage_status"] == "endpoint_only"
    assert result.details["diagnostics"][-1]["code"] == "collision_coverage_incomplete"


@pytest.mark.parametrize(
    "failure_fields",
    [
        {"error": "synthetic collision backend error"},
        {"available": False},
        {"collision": {"available": False}},
        {"success": False},
    ],
)
def test_collision_coverage_fails_closed_on_failed_receipt(
    failure_fields: JsonDict,
) -> None:
    response: JsonDict = {
        "success": True,
        "collision": {
            "available": True,
            "checked": True,
            "endpoint_checked": True,
            "trajectory_checked": True,
            "world_checked": True,
            "detected": False,
        },
    }
    for key, value in failure_fields.items():
        if key == "collision" and isinstance(value, dict):
            response["collision"].update(value)
        else:
            response[key] = value

    coverage = sim_mcp._collision_coverage_receipt(
        response,
        agent_tool="move_to",
        requested_collision_check=True,
    )

    assert coverage["endpoint_checked"] is False
    assert coverage["trajectory_checked"] is False
    assert coverage["world_checked"] is False
    assert coverage["coverage_complete"] is False


def test_simulator_proxy_uses_immutable_artifact_paths_per_call(tmp_path: Path) -> None:
    transport = FakeSimulatorMcpTransport({"task": "first", "cameras": [], "robot": {}})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-artifacts",
            handle="env-artifacts",
            response_output_root=tmp_path,
        ),
        tool_names=("observe",),
    )

    first = tools.call("observe", {})
    transport.response = {"task": "second", "cameras": [], "robot": {}}
    second = tools.call("observe", {})

    first_path = Path(first.details["outputs"]["response"]["response_path"])
    second_path = Path(second.details["outputs"]["response"]["response_path"])
    assert first_path != second_path
    assert json.loads(first_path.read_text(encoding="utf-8"))["task"] == "first"
    assert json.loads(second_path.read_text(encoding="utf-8"))["task"] == "second"


def test_move_to_proxy_rejects_camera_frame_target_pose() -> None:
    transport = FakeSimulatorMcpTransport({"success": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {"target_pose": {"frame": "camera", "translation_xyz": [0.1, 0.2, 0.3]}},
    )

    assert result.success is False
    assert transport.calls == []
    diagnostic = result.details["diagnostics"][0]
    assert diagnostic["code"] == "simulator_mcp_argument_error"
    assert "target_pose.frame must be 'world'" in diagnostic["message"]


def test_control_tool_proxy_fails_fast_without_active_handle() -> None:
    transport = FakeSimulatorMcpTransport({"success": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.2, 0.3]}})

    assert result.success is False
    assert transport.calls == []
    diagnostic = result.details["diagnostics"][0]
    assert diagnostic["code"] == "simulator_mcp_argument_error"
    assert "No active simulator MCP environment handle" in diagnostic["message"]


def test_follow_eef_trajectory_proxy_forwards_to_simulator_mcp() -> None:
    transport = FakeSimulatorMcpTransport({"success": True, "cameras": [], "robot": {}})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("follow_eef_trajectory",),
    )

    result = tools.call(
        "follow_eef_trajectory",
        {
            "trajectory": [{"xyz": [0.0, 0.0, 0.4]}],
            "ik_receipt_ids": ["ik-host-only"],
        },
    )

    assert result.success is True
    assert transport.calls == [
        {
            "name": "follow_eef_trajectory",
            "arguments": {
                "trajectory": [{"xyz": [0.0, 0.0, 0.4]}],
                "handle": "env-1",
                "session_id": "session-1",
            },
            "timeout_s": 120.0,
        }
    ]
    execution_receipt = result.details["host_execution_receipt"]
    assert execution_receipt["reference_kind"] == "ik_trajectory_receipts"
    assert execution_receipt["parameters"] == {
        "trajectory": [{"xyz": [0.0, 0.0, 0.4]}],
        "ik_receipt_ids": ["ik-host-only"],
    }


def test_condition_c_privately_forwards_sequential_route_bundle(monkeypatch) -> None:
    monkeypatch.setenv("OPENETA_MOTION_EXPERIMENT_CONDITION", "C")
    transport = FakeSimulatorMcpTransport(
        {"success": True, "reached_target": True, "cameras": [], "robot": {}}
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("follow_eef_trajectory",),
    )
    parameters = {
        "trajectory": [{"frame": "world", "xyz": [0.0, 0.0, 0.4]}],
        "ik_receipt_ids": ["ik-host-only"],
    }
    bundle = {
        "schema_version": "openeta.experimental_route_execution_bundle.v1",
        "condition": "C",
        "authority": "host_memory_exact_receipt_resolution",
        "entries": [
            {
                "source_ik_receipt_id": "ik-host-only",
                "target_pose": parameters["trajectory"][0],
            }
        ],
    }

    result = tools.call(
        "follow_eef_trajectory",
        parameters,
        metadata={
            "_ik_trajectory_execution_bundle_resolver": lambda request: bundle
        },
    )

    assert result.success is True
    forwarded = transport.calls[0]["arguments"]
    assert forwarded["route_execution_bundle"] == bundle
    assert "ik_receipt_ids" not in forwarded
    assert "route_execution_bundle" not in result.details["parameters"]
    assert result.details["parameters"] == parameters


def test_follow_eef_trajectory_incomplete_receipt_requires_reconciliation() -> None:
    transport = FakeSimulatorMcpTransport(
        {"steps_executed": 4, "waypoints_requested": 2, "waypoints_completed": 1}
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-1", handle="env-1"),
        tool_names=("follow_eef_trajectory",),
    )

    result = tools.call(
        "follow_eef_trajectory",
        {"trajectory": [{"xyz": [0.0, 0.0, 0.4]}, {"xyz": [0.01, 0.0, 0.4]}]},
    )

    assert result.success is False
    assert result.details["outputs"]["motion_outcome"] == "unknown"
    assert result.details["outputs"]["reconciliation_required"] is True


def test_control_tool_proxy_materializes_long_text_response(tmp_path: Path) -> None:
    long_log = "start\n" + ("important simulator log line\n" * 300)
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "content": long_log,
            "reward": 0.0,
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-1",
            handle="env-1",
            response_output_root=tmp_path,
            max_inline_text_chars=100,
        ),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.2, 0.3]}})

    response = result.details["outputs"]["response"]
    assert result.success is True
    assert response["response_omitted"] is True
    assert Path(response["response_path"]).exists()
    assert "important simulator log line" in Path(response["response_path"]).read_text(
        encoding="utf-8"
    )
    assert "important simulator log line" * 20 not in json.dumps(result.details)
    assert result.details["artifacts"][0]["type"] == "json"


def test_mcp_episode_previous_action_summary_omits_large_tool_details() -> None:
    huge_payload = "x" * 10000
    transport = FakeSimulatorMcpTransport({"cameras": [], "robot": {}, "reward": 0.0})
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            session_id="session-previous-action",
            handle="env-previous-action",
        ),
    )
    action = EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": "python_exec"},
            "status": "executed",
            "tool_calls": [
                {
                    "name": "python_exec",
                    "status": "executed",
                    "result": {
                        "success": True,
                        "content": huge_payload,
                        "details": {"outputs": {"result": huge_payload}},
                    },
                }
            ],
        },
    )

    step = env.step(action)

    metadata_json = json.dumps(step.observation.metadata)
    info_json = json.dumps(step.info)
    assert huge_payload not in metadata_json
    assert huge_payload not in info_json
    assert step.observation.metadata["previous_action"]["request_name"] == "python_exec"
    assert len(metadata_json) < 1000


def test_mcp_episode_stops_after_explicit_remote_termination_error() -> None:
    transport = FakeSimulatorMcpTransport({"cameras": [], "robot": {}, "reward": 0.0})
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            session_id="session-terminated",
            handle="env-terminated",
        ),
    )
    action = EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": "gripper_control"},
            "status": "failed",
            "tool_calls": [
                {
                    "name": "gripper_control",
                    "status": "failed",
                    "result": {
                        "success": False,
                        "content": "Step failed: executing action in terminated episode",
                        "details": {
                            "diagnostics": [
                                {
                                    "code": "simulator_mcp_error",
                                    "message": (
                                        "Step failed: executing action in terminated episode"
                                    ),
                                }
                            ]
                        },
                    },
                }
            ],
        },
    )

    step = env.step(action)

    assert step.terminated is True
    assert step.info["termination_source"] == "simulator_mcp"
    assert step.info["termination_reason"] == "remote_episode_terminated"


def test_mcp_episode_keeps_first_terminal_receipt_reward() -> None:
    def trusted_call(name: str, *, reward: float) -> dict:
        return {
            "name": name,
            "result": {
                "success": True,
                "details": {
                    "host_provenance": {"authority": "environment"},
                    "environment_receipt": {
                        "schema_version": "openeta.environment_receipt.v1",
                        "reward_present": True,
                        "reward": reward,
                        "terminated": True,
                        "truncated": False,
                    },
                },
            },
        }

    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "cameras": [],
            "robot": {},
            "objects": [],
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }
    )
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/libero_probe-v0",
            session_id="sim-session",
            handle="env-1",
        ),
    )
    action = EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": "move_to"},
            "tool_calls": [
                trusted_call("move_to", reward=1.0),
                trusted_call("repeated_step", reward=0.0),
            ],
        },
    )

    step = env.step(action)

    assert step.reward == 1.0
    assert step.terminated is True
    assert step.info["official_reward"] is True


def test_mcp_episode_does_not_treat_regular_tool_failure_as_termination() -> None:
    transport = FakeSimulatorMcpTransport({"cameras": [], "robot": {}, "reward": 0.0})
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            session_id="session-active",
            handle="env-active",
        ),
    )
    action = EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": "move_to"},
            "status": "failed",
            "tool_calls": [
                {
                    "name": "move_to",
                    "status": "failed",
                    "result": {
                        "success": False,
                        "content": "IK failed",
                        "details": {
                            "diagnostics": [{"code": "simulator_mcp_error", "message": "IK failed"}]
                        },
                    },
                }
            ],
        },
    )

    step = env.step(action)

    assert step.terminated is False
    assert "termination_reason" not in step.info


def test_mcp_episode_retries_transient_render_connection_refusal() -> None:
    transport = RenderConnectionRefusedOnceTransport()
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            session_id="session-render-retry",
            handle="env-render-retry",
            startup_attempts=2,
            startup_retry_delay_s=0,
        ),
    )

    step = env.step(EnvAction(action_type="tool_call", command={}))

    assert step.observation.task == ""
    assert [call["name"] for call in transport.calls] == ["render_env", "render_env"]


def test_mcp_episode_retries_grouped_remote_protocol_render_failure() -> None:
    transport = RemoteProtocolErrorOnceTransport("render_env")
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            session_id="session-render-retry",
            handle="env-render-retry",
            startup_attempts=2,
            startup_retry_delay_s=0,
        ),
    )

    step = env.step(EnvAction(action_type="tool_call", command={}))

    assert step.observation.task == ""
    assert [call["name"] for call in transport.calls] == ["render_env", "render_env"]


def test_mcp_episode_retries_grouped_remote_protocol_create_failure() -> None:
    transport = RemoteProtocolErrorOnceTransport("create_env")
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            startup_attempts=2,
            startup_retry_delay_s=0,
        ),
    )

    observation = env.reset(task="inspect scene")

    assert observation.task == "inspect scene"
    assert [call["name"] for call in transport.calls] == [
        "create_env",
        "create_env",
        "reset_env",
    ]


def test_mcp_episode_observation_artifacts_are_session_scoped_and_unique(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "cameras": [
                {
                    "frame_id": "front",
                    "rgb_base64": PNG_1X1,
                    "depth_base64": PNG_1X1,
                    "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5},
                }
            ],
            "robot": {},
        }
    )
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            session_id="remote-sim-session",
            handle="env-observe",
            image_output_root=tmp_path,
        ),
    )

    reset_observation = env.reset(
        task="inspect scene",
        metadata={"agent_session_id": "agent-session"},
    )
    step = env.step(EnvAction(action_type="tool_call", command={}))
    reset_path = Path(reset_observation.metadata["image_artifacts"][0]["path"])
    step_path = Path(step.observation.metadata["image_artifacts"][0]["path"])

    assert reset_path != step_path
    assert reset_path.relative_to(tmp_path).parts[0] == "agent-session"
    assert step_path.relative_to(tmp_path).parts[0] == "agent-session"
    assert reset_observation.cameras[0].intrinsics["scale"] == 1000.0
    assert step.observation.cameras[0].intrinsics["scale"] == 1000.0


def test_mcp_episode_create_env_defaults_to_high_resolution() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "handle": "env-high-res",
            "session_id": "session-high-res",
            "cameras": [],
            "robot": {},
        }
    )
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(env_id="openeta/dummy_sim-v0"),
    )

    env.reset(task="inspect scene")

    assert transport.calls[0]["name"] == "create_env"
    assert transport.calls[0]["arguments"]["image_width"] == 512
    assert transport.calls[0]["arguments"]["image_height"] == 512
    assert transport.calls[1]["name"] == "reset_env"


def test_mcp_episode_prefers_simulator_assigned_task_over_manifest_task() -> None:
    assigned_task = (
        "pick up the black bowl between the plate and the ramekin and place it on the plate"
    )
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "handle": "env-assigned-task",
            "session_id": "session-assigned-task",
            "task": assigned_task,
            "cameras": [],
            "robot": {},
        }
    )
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(env_id="openeta/libero_spatial_task0-v0"),
    )

    observation = env.reset(task="complete the task assigned by the simulator")

    assert observation.task == assigned_task
    assert env.task == assigned_task
    assert observation.metadata["assigned_task"] == assigned_task
    assert observation.metadata["assigned_task_source"] == "simulator_observation"
    assert transport.calls[0]["arguments"]["task"] == (
        "complete the task assigned by the simulator"
    )


def test_mcp_episode_recreates_once_after_transient_unknown_handle() -> None:
    transport = UnknownHandleOnceTransport()
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            startup_attempts=2,
            startup_retry_delay_s=0,
        ),
    )

    observation = env.reset(task="inspect scene")

    assert observation.task == "inspect scene"
    assert [call["name"] for call in transport.calls] == [
        "create_env",
        "reset_env",
        "close_env",
        "create_env",
        "reset_env",
    ]
    assert env.config.handle == "env-2"
    assert observation.metadata["startup_attempt_count"] == 2


def test_mcp_episode_retries_transient_create_connection_refusal() -> None:
    transport = CreateConnectionRefusedOnceTransport()
    env = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(
            env_id="openeta/dummy_sim-v0",
            startup_attempts=2,
            startup_retry_delay_s=0,
        ),
    )

    observation = env.reset(task="inspect scene")

    assert observation.task == "inspect scene"
    assert [call["name"] for call in transport.calls] == [
        "create_env",
        "create_env",
        "reset_env",
    ]
    assert observation.metadata["startup_attempt_count"] == 2


def test_gripper_control_selects_open_or_close_mcp_tool(tmp_path: Path) -> None:
    transport = FakeSimulatorMcpTransport({"cameras": [], "robot": {}})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-2",
            handle="env-2",
            image_output_root=tmp_path,
        ),
        tool_names=("gripper_control",),
    )

    open_result = tools.call("gripper_control", {"position": 1.0})
    close_result = tools.call("gripper_control", {"position": 0.0})

    assert open_result.success is True
    assert close_result.success is True
    assert transport.calls[0]["name"] == "gripper_open"
    assert transport.calls[0]["arguments"] == {
        "handle": "env-2",
        "session_id": "session-2",
    }
    assert transport.calls[1]["name"] == "gripper_close"
    close_receipt = close_result.details["outputs"]["attachment_proxy_receipt"]
    assert close_receipt["status"] == "not_armed"
    assert close_receipt["reason"] == "no_active_contact_authorization"


def test_gripper_close_exposes_stationary_settle_receipt(tmp_path: Path) -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "cameras": [],
            "robot": {"gripper_state": {"openness": 0.42}},
            "gripper_actuation_receipt": {
                "schema_version": "openeta.gripper_actuation_receipt.v1",
                "command": "close",
                "command_latched": True,
                "steps_executed": 60,
                "settling_policy": "stationary_continuous_position_hold",
                "measured_open_fraction": 0.42,
            },
            "attachment_proxy_receipt": {
                "schema_version": "openeta.attachment_proxy_receipt.v1",
                "status": "tentative",
                "reason": "non_empty_close_near_bound_target",
                "target_object_name": "milk_1",
                "measured_open_fraction": 0.42,
                "attachment_proven": False,
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-settle",
            handle="env-settle",
            image_output_root=tmp_path,
        ),
        tool_names=("gripper_control",),
    )

    result = tools.call("gripper_control", {"position": 0})

    receipt = result.details["outputs"]["gripper_actuation_receipt"]
    assert receipt["steps_executed"] == 60
    assert receipt["command_latched"] is True
    assert result.details["outputs"]["attachment_proxy_receipt"]["reason"] == (
        "non_empty_close_with_tentative_safety_proxy"
    )
    assert "stationary_settle_steps=60" in result.content
    assert "milk_1" not in result.content
    assert "milk_1" not in json.dumps(result.details)
    response_path = Path(result.details["outputs"]["response"]["response_path"])
    assert "milk_1" not in response_path.read_text(encoding="utf-8")


def test_missing_remote_attachment_receipt_is_explicit_and_refreshes_on_lift() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "cameras": [],
            "robot": {},
            "start": {"xyz": [0.0, 0.0, 0.12]},
            "end": {"xyz": [0.0, 0.0, 0.2]},
            "target": {"x": 0.0, "y": 0.0, "z": 0.2},
            "reached_target": True,
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-old", handle="env-old"),
        tool_names=("gripper_control", "move_to"),
    )
    authorization = {
        "schema_version": "openeta.contact_authorization.v1",
        "compiled_grasp_id": "compiled-old",
        "waypoint_role": "grasp_contact",
        "target_object_name": "milk_1",
        "target_anchor_world_xyz": [0.0, 0.0, 0.12],
        "object_scene_epoch": 0,
    }

    close_result = tools.call(
        "gripper_control",
        {"position": 0},
        metadata={"_attachment_candidate_resolver": lambda: authorization},
    )
    lift_result = tools.call(
        "move_to",
        {"target_pose": {"xyz": [0.0, 0.0, 0.2]}},
        metadata={"_contact_authorization_resolver": lambda _pose: authorization},
    )

    close_receipt = close_result.details["outputs"]["attachment_proxy_receipt"]
    assert close_receipt["status"] == "backend_contract_missing"
    assert close_receipt["contact_authorization_forwarded"] is True
    assert close_receipt["collision_proxy_active"] is None
    assert close_result.details["semantic_outcome"] == "attachment_contract_unavailable"
    assert {item["action"] for item in close_result.details["recovery_options"]} == {
        "inspect_fresh_dual_view",
        "small_guarded_lift_probe",
        "upgrade_or_restart_simulator_service",
    }
    assert any(
        item["code"] == "attachment_proxy_backend_contract_missing"
        for item in close_result.details["diagnostics"]
    )
    assert "Physical attachment is unknown" in close_result.content

    lift_receipt = lift_result.details["outputs"]["attachment_proxy_receipt"]
    assert lift_receipt["status"] == "backend_contract_missing"
    assert lift_receipt["reason"] == "remote_attachment_proxy_refresh_receipt_missing"
    assert lift_result.details["semantic_outcome"] == "attachment_contract_unavailable"


def test_gripper_close_privately_forwards_active_attachment_target() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "cameras": [],
            "robot": {},
            "attachment_proxy_receipt": {
                "schema_version": "openeta.attachment_proxy_receipt.v1",
                "status": "tentative",
                "target_object_name": "salad_dressing_1",
                "attachment_proven": False,
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-2", handle="env-2"),
        tool_names=("gripper_control",),
    )
    authorization = {
        "schema_version": "openeta.contact_authorization.v1",
        "compiled_grasp_id": "compiled-1",
        "waypoint_role": "grasp_contact",
        "target_anchor_world_xyz": [0.1, 0.2, 0.12],
        "target_evidence_id": "sam3:result:target",
        "object_scene_epoch": 0,
    }

    result = tools.call(
        "gripper_control",
        {"position": 0},
        metadata={"_attachment_candidate_resolver": lambda: authorization},
    )

    assert result.success is True
    assert transport.calls[0]["arguments"]["contact_authorization"] == authorization
    assert "contact_authorization" not in result.details["parameters"]
    assert result.details["outputs"]["attachment_proxy_receipt"]["status"] == (
        "tentative"
    )
    assert result.details["semantic_outcome"] == "requires_attachment_probe"
    assert "attachment_proven=false" in result.content
    assert "2-5 cm" in result.content
    assert "do not reuse a prior grasp_clearance" in result.content
    recovery = next(
        item
        for item in result.details["recovery_options"]
        if item["action"] == "small_lift_probe"
    )
    assert recovery["distance_range_m"] == [0.02, 0.05]
    assert recovery["pose_source"] == "fresh_current_eef_pose"


def test_gripper_empty_close_feedback_does_not_recommend_lift_probe() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "cameras": [],
            "robot": {},
            "attachment_proxy_receipt": {
                "schema_version": "openeta.attachment_proxy_receipt.v1",
                "status": "not_armed",
                "reason": "empty_close_or_no_measurable_contact",
                "measured_open_fraction": 0.03,
                "attachment_proven": False,
            },
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-empty", handle="env-empty"),
        tool_names=("gripper_control",),
    )

    result = tools.call("gripper_control", {"position": 0})

    assert result.success is True
    assert result.details["semantic_outcome"] == "no_attachment_evidence"
    assert {item["action"] for item in result.details["recovery_options"]} == {
        "inspect_fresh_dual_view",
        "reopen_and_repair_contact",
    }
    assert "Do NOT treat a lift as an attachment probe" in result.content
    assert "reopen the gripper" in result.content


def test_gripper_control_rejects_fractional_command(tmp_path: Path) -> None:
    transport = FakeSimulatorMcpTransport({"cameras": [], "robot": {}})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-binary-gripper",
            handle="env-binary-gripper",
            image_output_root=tmp_path,
        ),
        tool_names=("gripper_control",),
    )

    result = tools.call("gripper_control", {"position": 0.5})

    assert result.success is False
    assert "exactly 0 or 1" in result.content
    assert transport.calls == []


def test_proxy_can_override_agent_tool_name_to_simulator_mcp_tool(tmp_path: Path) -> None:
    transport = FakeSimulatorMcpTransport({"cameras": [], "robot": {}})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="session-2",
            handle="env-2",
            tool_name_map={"gripper_control": "control_gripper"},
            image_output_root=tmp_path,
        ),
        tool_names=("gripper_control",),
    )

    result = tools.call("gripper_control", {"position": 0.0})

    assert result.success is True
    assert transport.calls[0]["name"] == "control_gripper"
    assert transport.calls[0]["arguments"]["position"] == 0.0
    assert transport.calls[0]["arguments"]["handle"] == "env-2"
    assert result.details["outputs"]["mcp"]["agent_tool"] == "gripper_control"
    assert result.details["outputs"]["mcp"]["tool"] == "control_gripper"


def test_proxy_structures_simulator_mcp_errors() -> None:
    transport = FakeSimulatorMcpTransport({"error": "IK failed"})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-3", handle="env-3"),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [99, 99, 99]}})

    assert result.success is False
    assert result.details["diagnostics"][0]["code"] == "simulator_mcp_error"
    assert result.details["outputs"]["response"]["error"] == "IK failed"


def test_proxy_reconciles_world_mutation_after_grouped_remote_protocol_failure() -> None:
    transport = RemoteProtocolErrorOnceTransport("move_to")
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-3", handle="env-3"),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.2, 0.3]}})

    assert result.success is False
    assert result.details["outputs"]["motion_outcome"] == "unknown"
    assert result.details["outputs"]["reconciliation_required"] is True
    assert result.details["diagnostics"][0]["code"] == ("simulator_mcp_transport_connection_lost")
    execution_receipt = result.details["host_execution_receipt"]
    assert execution_receipt["dispatch_status"] == "outcome_unknown"
    assert execution_receipt["parameters"]["target_pose"] == {
        "xyz": [0.1, 0.2, 0.3]
    }


def test_proxy_reconciles_world_mutation_when_remote_action_receipt_is_not_json() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "error": "Step failed: Out of range float values are not JSON compliant",
            "fatal": False,
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-3", handle="env-3"),
        tool_names=("gripper_control",),
    )

    result = tools.call("gripper_control", {"position": 1.0})

    assert result.success is False
    assert result.details["outputs"]["motion_outcome"] == "unknown"
    assert result.details["outputs"]["reconciliation_required"] is True
    assert result.details["diagnostics"] == [
        {
            "code": "simulator_mcp_action_receipt_unavailable",
            "message": "Step failed: Out of range float values are not JSON compliant",
            "candidate_rejection": False,
            "failure_class": "action_outcome_unknown",
        }
    ]


def test_move_to_requires_observation_when_controller_receipt_omits_end_pose() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "reward": 0,
            "terminated": False,
            "steps_executed": 100,
            "start": {"xyz": [0.0, 0.0, 0.5]},
            "end": {"xyz": []},
            "target": {"x": 0.1, "y": 0.2, "z": 0.3},
        }
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(session_id="session-3", handle="env-3"),
        tool_names=("move_to",),
    )

    result = tools.call("move_to", {"target_pose": {"xyz": [0.1, 0.2, 0.3]}})

    assert result.success is False
    assert result.details["outputs"]["motion_outcome"] == "unknown"
    assert result.details["outputs"]["reconciliation_required"] is True
    assert result.details["diagnostics"][0]["code"] == ("simulator_mcp_motion_receipt_incomplete")


def test_parse_mcp_tool_result_accepts_structured_content() -> None:
    result = FakeMcpResult(content=[{"type": "text", "json": {"success": True, "reward": 1.0}}])

    assert _parse_mcp_tool_result(result) == {"success": True, "reward": 1.0}


def test_parse_mcp_tool_result_preserves_mcp_error_text() -> None:
    result = FakeMcpResult(
        content=[{"type": "text", "text": "Unknown tool: legacy_macro"}],
        is_error=True,
    )

    parsed = _parse_mcp_tool_result(result)

    assert parsed["success"] is False
    assert parsed["error"] == "Unknown tool: legacy_macro"
    assert parsed["content"] == "Unknown tool: legacy_macro"
    assert parsed["failure_class"] == "remote_capability_missing"
    assert parsed["candidate_rejection"] is False
    assert parsed["details"]["mcp_is_error"] is True


def test_parse_mcp_tool_result_overrides_success_for_mcp_error() -> None:
    result = FakeMcpResult(
        content=[{"type": "text", "json": {"success": True, "reward": 1.0}}],
        is_error=True,
    )

    parsed = _parse_mcp_tool_result(result)

    assert parsed["success"] is False
    assert parsed["failure_class"] == "mcp_tool_error"
    assert parsed["details"]["mcp_is_error"] is True


def test_binding_does_not_require_a_remote_tool_catalog() -> None:
    transport = FakeSimulatorMcpTransport({"success": True})
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(handle="env-1"),
        tool_names=("move_to",),
    )

    result = tools.call(
        "move_to",
        {
            "target_pose": {
                "id": "candidate-0",
                "translation_xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
        },
    )

    assert result.success is True
    assert transport.calls[0]["name"] == "move_to"


def test_parse_mcp_tools_result_accepts_tool_docs() -> None:
    result = FakeToolListResult(
        [
            {
                "name": "create_env",
                "description": "Create env",
                "inputSchema": {
                    "type": "object",
                    "required": ["env_id"],
                    "properties": {"env_id": {"type": "string"}},
                },
            }
        ]
    )

    parsed = _parse_mcp_tools_result(result)

    assert parsed["tool_count"] == 1
    assert parsed["tools"][0]["name"] == "create_env"
    assert parsed["tools"][0]["input_schema"]["required"] == ["env_id"]
