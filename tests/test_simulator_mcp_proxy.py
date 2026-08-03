"""Tests for simulator MCP tool proxy handlers."""

from __future__ import annotations

import base64
import json
import os
import threading
from pathlib import Path

import agent.tools.sim_mcp as sim_mcp
from adapter.protocol import EnvAction, JsonDict
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
        self.calls.append(
            {"name": name, "arguments": dict(arguments), "timeout_s": timeout_s}
        )
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
            self.calls.append(
                {"name": name, "arguments": dict(arguments), "timeout_s": timeout_s}
            )
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

    transport = SseSimulatorMcpTransport("http://203.0.113.10:8773/sse")

    assert transport.list_tools(timeout_s=3.0)["tool_count"] == 0
    assert transport.call_tool("segment", {"prompt": "cube"}, timeout_s=4.0)["success"] is True

    for record in observed.values():
        assert "localhost" in record["NO_PROXY"]
        assert "203.0.113.10" in record["NO_PROXY"]
        assert "203.0.113.10:8773" in record["NO_PROXY"]
        assert record["NO_PROXY"] == record["no_proxy"]
    assert os.environ["NO_PROXY"] == "localhost"
    assert "no_proxy" not in os.environ


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
        "move_to",
        "follow_eef_trajectory",
        "gripper_control",
    )
    assert tools.can_execute("create_simulator_env")
    assert tools.can_execute("close_simulator_env")
    assert tools.can_execute("observe")
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
    environment = result.details["outputs"]["environment"]
    assert environment["dashboard_url"] == "http://sim.example/session/session-1"
    camera = result.details["outputs"]["initial_observation"]["cameras"][0]
    assert camera["anygrasp_intrinsics"]["scale"] == 1000.0
    assert Path(camera["rgb_path"]).exists()
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
    assert observation["objects"][0]["category"] == "alphabet_soup"
    response_summary = result.details["outputs"]["response"]["observation_summary"]
    assert response_summary["objects"][0]["position"] == [-0.1, -0.2, 0.47]


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
    assert result.details["state_delta"]["observation"]["cameras"][0]["rgb_path"] == (
        camera["rgb_path"]
    )
    assert PNG_1X1 not in json.dumps(result.details)


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


def test_move_to_uses_standard_world_pose_contract_for_anygrasp_pose() -> None:
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


def test_move_to_proxy_preserves_motion_summary_without_overriding_remote_outcome(
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
    assert result.details["diagnostics"] == []
    motion = result.details["outputs"]["response"]["motion_summary"]
    assert motion["collision"]["detected"] is True
    assert motion["reached_target"] is False
    assert result.details["state_delta"]["motion"] == motion


def test_move_to_proxy_allows_unchanged_baseline_contact() -> None:
    transport = FakeSimulatorMcpTransport(
        {
            "success": True,
            "reached_target": True,
            "collision": {
                "available": True,
                "detected": True,
                "new_or_worsened": False,
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
    motion = result.details["outputs"]["response"]["motion_summary"]
    assert motion["collision"]["detected"] is True
    assert motion["reached_target"] is True
    assert result.details["state_delta"]["motion"] == motion


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
        {"trajectory": [{"xyz": [0.0, 0.0, 0.4]}]},
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
