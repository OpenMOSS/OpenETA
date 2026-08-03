from __future__ import annotations

import base64
import json

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.keys import Keys

import agent.cli.openeta_cli as cli_module
from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState, StepResult
from agent.cli.openeta_cli import (
    OpenEtaCli,
    SLASH_COMMANDS,
    SlashCommandCompleter,
    _cancel_active_completion,
    _build_cli_checker_config,
    _load_mcp_url,
    _load_sim_mcp_url,
    _parse_promote_memory_args,
    _prompt_html,
)
from agent.runtime.episode import EpisodeResult
from agent.runtime.episode import EpisodeStep
from agent.runtime.memory_store import JsonMemoryStore
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry


PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000100ffff03000006000557bfab0d000000"
        "0049454e44ae426082"
    )
).decode("ascii")


@pytest.fixture(autouse=True)
def isolate_cli_memory_store(monkeypatch, tmp_path):
    class IsolatedJsonMemoryStore(JsonMemoryStore):
        def __init__(self, root=None) -> None:
            super().__init__(root=tmp_path / ".openeta_memory")

    monkeypatch.setattr(
        "agent.cli.openeta_cli.JsonMemoryStore",
        IsolatedJsonMemoryStore,
    )


def _completion_texts(text: str) -> list[str]:
    completer = SlashCommandCompleter(SLASH_COMMANDS)
    return [item.text for item in completer.get_completions(Document(text), None)]


def test_cli_checker_config_keeps_pre_safety_gates_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("OPENETA_PRE_SAFETY_CHECKS", raising=False)
    tools = bind_dummy_tool_handlers(build_default_tool_registry())

    config = _build_cli_checker_config(tools)

    assert config.pre_safety_checks == {}
    assert "move_to" in config.post_failure_checks


def test_cli_checker_config_accepts_explicit_safety_mapping(monkeypatch) -> None:
    monkeypatch.setenv(
        "OPENETA_PRE_SAFETY_CHECKS",
        '{"move_to": "ik_preview_check"}',
    )
    tools = bind_dummy_tool_handlers(build_default_tool_registry())

    config = _build_cli_checker_config(tools)

    assert config.pre_safety_checks == {"move_to": "ik_preview_check"}


def test_slash_completer_shows_default_commands_for_bare_slash() -> None:
    completions = _completion_texts("/")

    assert completions[:3] == ["/provider ", "/model ", "/models "]
    assert "/quit " in completions
    assert "/exit " not in completions


def test_slash_completer_filters_by_command_prefix() -> None:
    assert _completion_texts("/mo") == ["/model ", "/models "]
    assert _completion_texts("/mem") == ["/memory "]
    assert _completion_texts("/ne") == ["/new "]
    assert _completion_texts("/ses") == ["/sessions ", "/session "]
    assert _completion_texts("/skill") == ["/skill-reviews ", "/skill-review "]


def test_slash_completer_stops_after_command_arguments() -> None:
    assert _completion_texts("/run inspect cube") == []


def test_escape_completion_helper_cancels_active_completion() -> None:
    class FakeBuffer:
        complete_state = object()

        def __init__(self) -> None:
            self.cancelled = False

        def cancel_completion(self) -> None:
            self.cancelled = True

    buffer = FakeBuffer()

    assert _cancel_active_completion(buffer) is True
    assert buffer.cancelled is True


def test_escape_completion_helper_ignores_plain_input() -> None:
    class FakeBuffer:
        complete_state = None

        def cancel_completion(self) -> None:
            raise AssertionError("cancel_completion should not be called")

    assert _cancel_active_completion(FakeBuffer()) is False


def test_escape_completion_binding_is_eager() -> None:
    cli = OpenEtaCli()
    bindings = cli._key_bindings().bindings

    escape_binding = next(binding for binding in bindings if binding.keys == (Keys.Escape,))

    assert escape_binding.eager()


def test_promote_memory_args_parse_target_and_note() -> None:
    args = _parse_promote_memory_args(
        ["facts", "target", "--target", "tool_lessons.md", "--note", "reviewed"]
    )

    assert args["namespace"] == "facts"
    assert args["key"] == "target"
    assert args["target"] == "tool_lessons.md"
    assert args["note"] == "reviewed"


def test_load_sim_mcp_url_prefers_named_openeta_server(tmp_path) -> None:
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        """
        {
          "mcpServers": {
            "other": {"url": "http://127.0.0.1:1/sse"},
            "openeta": {"url": "http://203.0.113.10:8765/sse"}
          }
        }
        """,
        encoding="utf-8",
    )

    assert _load_sim_mcp_url(config_path) == "http://203.0.113.10:8765/sse"


def test_load_mcp_url_reads_named_perception_services(tmp_path) -> None:
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        """
        {
          "mcpServers": {
            "openeta-sim": {"url": "http://203.0.113.10:8765/sse"},
            "openeta-sam3": {"url": "http://203.0.113.10:8773/sse"},
            "openeta-anygrasp": {"url": "http://203.0.113.10:8774/sse"},
            "openeta-anyplace": {"url": "http://203.0.113.10:8775/sse"},
            "openeta-contact-graspnet": {"url": "http://203.0.113.10:8776/sse"}
          }
        }
        """,
        encoding="utf-8",
    )

    assert _load_mcp_url("openeta-sam3", path=config_path) == "http://203.0.113.10:8773/sse"
    assert _load_mcp_url("openeta-anygrasp", path=config_path) == "http://203.0.113.10:8774/sse"
    assert _load_mcp_url("openeta-anyplace", path=config_path) == "http://203.0.113.10:8775/sse"
    assert _load_mcp_url("openeta-contact-graspnet", path=config_path) == (
        "http://203.0.113.10:8776/sse"
    )


def test_load_contact_graspnet_mcp_url_accepts_alias(tmp_path) -> None:
    config_path = tmp_path / ".mcp.json"
    config_path.write_text(
        '{"mcpServers":{"contact_graspnet":{"url":"http://contact.example/sse"}}}',
        encoding="utf-8",
    )

    assert _load_mcp_url(
        "openeta-contact-graspnet",
        aliases=("contact-graspnet", "contact_graspnet"),
        path=config_path,
    ) == "http://contact.example/sse"


def test_cli_binds_perception_mcp_handlers_from_registry(monkeypatch, tmp_path) -> None:
    urls = {
        "openeta-sam3": "http://sam3.example/sse",
        "openeta-anygrasp": "http://anygrasp.example/sse",
        "openeta-anyplace": "http://anyplace.example/sse",
        "openeta-contact-graspnet": "http://contact.example/sse",
    }
    calls = {
        "sam3_urls": [],
        "anygrasp_urls": [],
        "anyplace_urls": [],
        "contact_urls": [],
        "sam3": [],
        "anygrasp": [],
        "anyplace": [],
        "contact": [],
    }

    def fake_load_mcp_url(name, *, aliases=(), path=".mcp.json"):
        del aliases, path
        return urls.get(name, "")

    def fake_sam3_segmenter(*, url, tool_name="segment", timeout_seconds=600.0):
        calls["sam3_urls"].append((url, tool_name, timeout_seconds))

        def segment(request):
            calls["sam3"].append(request)
            return {
                "success": True,
                "content": "SAM3 segmentation completed.",
                "details": {
                    "tool": "sam3",
                    "backend": "sam3_mcp",
                    "model": "sam3",
                    "prompt": request["prompt"],
                    "source_image": "server-side-value",
                    "raw_output_ref": "raw.json",
                    "detection_count": 1,
                    "detections": [
                        {
                            "label": request["prompt"],
                            "score": 0.7,
                            "bbox_xyxy": [0, 0, 1, 1],
                            "mask": {"format": "png", "base64": PNG_1X1},
                            "area_px": 1,
                        }
                    ],
                    "artifacts": [],
                    "metadata": {},
                },
            }

        return segment

    def fake_anygrasp_grasper(*, url, tool_name="detect_grasps", timeout_seconds=600.0):
        calls["anygrasp_urls"].append((url, tool_name, timeout_seconds))

        def detect_grasps(request):
            calls["anygrasp"].append(request)
            return {
                "success": True,
                "content": "AnyGrasp grasp detection completed.",
                "details": {
                    "tool": "anygrasp",
                    "backend": "anygrasp_mcp",
                    "model": "anygrasp_sdk",
                    "mode": request["mode"],
                    "candidate_count": 1,
                    "grasp_candidates": [
                        {
                            "id": "grasp_000",
                            "frame": "camera",
                            "score": 0.5,
                            "translation_xyz": [0.1, 0.2, 0.3],
                            "rotation_matrix": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "depth": 0.03,
                            "width": 0.06,
                            "height": 0.03,
                            "gripper_tip_position_xyz": [0.13, 0.2, 0.3],
                        }
                    ],
                    "artifacts": [],
                    "metadata": {},
                },
            }

        return detect_grasps

    def fake_anyplace_placer(*, url, tool_name="predict_placement", timeout_seconds=600.0):
        calls["anyplace_urls"].append((url, tool_name, timeout_seconds))

        def predict_placement(request):
            calls["anyplace"].append(request)
            grasp = request["selected_grasp"]
            candidates = []
            for index in range(5):
                candidates.append(
                    {
                        "id": f"placement_{index:03d}",
                        "source_grasp_id": grasp["id"],
                        "object_placement_transform": {
                            "frame": "camera",
                            "camera_frame": "opencv",
                            "convention": "p_placed = R @ p_current + t",
                            "transform_matrix": [
                                [1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0],
                            ],
                        },
                        "place_grasp_pose": {**grasp, "id": f"place_grasp_{index:03d}"},
                    }
                )
            return {
                "success": True,
                "details": {
                    "tool": "anyplace",
                    "backend": "anyplace_mcp",
                    "model": "anyplace_multitask",
                    "frame": "camera",
                    "camera_frame": "opencv",
                    "candidate_count": 5,
                    "placement_candidates": candidates,
                    "metadata": {},
                },
            }

        return predict_placement

    def fake_contact_predictor(*, url, tool_name="predict_grasps", timeout_seconds=600.0):
        calls["contact_urls"].append((url, tool_name, timeout_seconds))

        def predict_grasps(request):
            calls["contact"].append(request)
            return {
                "success": True,
                "details": {
                    "tool": "contact_graspnet",
                    "backend": "contact_graspnet_mcp",
                    "model": "contact_graspnet_pytorch_unofficial",
                    "mode": "targeted",
                    "frame": "camera",
                    "camera_frame": "opencv",
                    "grasp_frame": "graspnet",
                    "candidate_count": 1,
                    "grasp_candidates": [
                        {
                            "id": "grasp_000",
                            "frame": "camera",
                            "camera_frame": "opencv",
                            "grasp_frame": "graspnet",
                            "source_model": "contact_graspnet",
                            "gripper_model": "panda",
                            "score": 0.7,
                            "translation_xyz": [0.1, 0.2, 0.3],
                            "rotation_matrix": [
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            "gripper_depth": 0.1034,
                            "width": 0.04,
                            "gripper_tip_position_xyz": [0.2034, 0.2, 0.3],
                            "contact_point_xyz": [0.2034, 0.18, 0.3],
                        }
                    ],
                    "artifacts": [],
                    "metadata": {"max_gripper_width": 0.08},
                },
            }

        return predict_grasps

    monkeypatch.setattr(cli_module, "_load_mcp_url", fake_load_mcp_url)
    monkeypatch.setattr(cli_module, "build_sse_sam3_mcp_segmenter", fake_sam3_segmenter)
    monkeypatch.setattr(cli_module, "build_sse_anygrasp_mcp_grasper", fake_anygrasp_grasper)
    monkeypatch.setattr(cli_module, "build_sse_anyplace_mcp_placer", fake_anyplace_placer)
    monkeypatch.setattr(
        cli_module,
        "build_sse_contact_graspnet_mcp_predictor",
        fake_contact_predictor,
    )

    image = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    placement_mask = tmp_path / "placement-mask.png"
    for path in (image, depth, mask, placement_mask):
        path.write_bytes(base64.b64decode(PNG_1X1))

    runtime = OpenEtaCli()._require_runtime()

    sam3 = runtime.tools.call("sam3", {"image": str(image), "prompt": "cube"})
    anygrasp = runtime.tools.call(
        "anygrasp",
        {
            "mode": "targeted",
            "rgb": str(image),
            "depth": str(depth),
            "target_mask": str(mask),
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "scale": 1000.0},
        },
    )
    anygrasp_outputs = anygrasp.details["outputs"]
    contact = runtime.tools.call(
        "contact_graspnet",
        {
            "rgb": str(image),
            "depth": str(depth),
            "object_mask": {
                "mask_ref": str(mask),
                "source_image": str(image),
                "label": "cube",
            },
            "intrinsics": {
                "fx": 1.0,
                "fy": 1.0,
                "cx": 0.5,
                "cy": 0.5,
                "scale": 1000.0,
            },
        },
    )
    anyplace = runtime.tools.call(
        "anyplace",
        {
            "rgb": str(image),
            "depth": str(depth),
            "object_mask": str(mask),
            "placement_region_mask": {
                "mask_ref": str(placement_mask),
                "source_image": str(image),
                "label": "target region",
            },
            "intrinsics": {
                "fx": 1.0,
                "fy": 1.0,
                "cx": 0.5,
                "cy": 0.5,
                "scale": 1000.0,
            },
            "selected_grasp": {
                "candidate": anygrasp_outputs["grasp_candidates"][0],
                "source": anygrasp_outputs["source"],
            },
        },
    )

    assert calls["sam3_urls"] == [("http://sam3.example/sse", "segment", 600.0)]
    assert calls["anygrasp_urls"] == [
        ("http://anygrasp.example/sse", "detect_grasps", 600.0)
    ]
    assert calls["anyplace_urls"] == [
        ("http://anyplace.example/sse", "predict_placement", 600.0)
    ]
    assert calls["contact_urls"] == [
        ("http://contact.example/sse", "predict_grasps", 600.0)
    ]
    assert calls["sam3"][0]["prompt"] == "cube"
    assert calls["anygrasp"][0]["mode"] == "targeted"
    assert sam3.success is True
    assert anygrasp.success is True
    assert contact.success is True
    assert set(calls["contact"][0]) == {"depth", "object_mask", "intrinsics"}
    assert calls["anyplace"][0]["selected_grasp"]["id"] == "grasp_000"
    assert anyplace.success is True
    assert anyplace.details["outputs"]["candidate_count"] == 5


def test_cli_reuses_simulator_mcp_state_for_stable_control_tools(monkeypatch) -> None:
    class FakeTransport:
        def __init__(self, url: str) -> None:
            self.url = url
            self.calls = []

        def call_tool(self, name, arguments, *, timeout_s=None):
            self.calls.append(
                {
                    "name": name,
                    "arguments": dict(arguments),
                    "timeout_s": timeout_s,
                }
            )
            return {"success": True, "reward": 0.0, "cameras": [], "robot": {}}

    transports = []

    def fake_transport(url: str) -> FakeTransport:
        transport = FakeTransport(url)
        transports.append(transport)
        return transport

    monkeypatch.setattr("agent.cli.openeta_cli._load_sim_mcp_url", lambda: "http://sim/mcp")
    monkeypatch.setattr("agent.cli.openeta_cli.SseSimulatorMcpTransport", fake_transport)
    cli = OpenEtaCli()
    cli._sync_simulator_mcp_response(
        "create_env",
        {"env_id": "openeta/libero_libero_10_task0-v0"},
        {"handle": "env-1", "session_id": "session-1"},
    )

    result = cli._require_runtime().tools.call(
        "move_to",
        {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
    )

    assert result.success is True
    assert transports
    assert transports[0].calls[-1]["name"] == "move_to"
    assert transports[0].calls[-1]["arguments"] == {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "handle": "env-1",
        "session_id": "session-1",
    }


def test_cli_does_not_sync_failed_simulator_mcp_response(tmp_path: Path) -> None:
    response_path = tmp_path / "response.json"
    response_path.write_text(
        '{"success": false, "handle": "bad-env", "session_id": "bad-session"}',
        encoding="utf-8",
    )
    cli = OpenEtaCli()

    cli._sync_simulator_mcp_response(
        "create_env",
        {"env_id": "openeta/libero_libero_10_task0-v0"},
        {
            "success": False,
            "response_path": str(response_path),
            "response_omitted": True,
        },
    )

    assert cli.state.simulator_mcp_config.handle == ""
    assert cli.state.simulator_mcp_config.session_id == ""


def test_cli_syncs_simulator_mcp_state_from_response_artifact(tmp_path: Path) -> None:
    response_path = tmp_path / "response.json"
    response_path.write_text(
        '{"success": true, "handle": "env-from-file", "session_id": "session-from-file"}',
        encoding="utf-8",
    )
    cli = OpenEtaCli()
    cli.state.simulator_mcp_url = "http://sim.example/sse"

    cli._sync_simulator_mcp_response(
        "create_env",
        {"env_id": "openeta/libero_libero_10_task0-v0"},
        {
            "response_path": str(response_path),
            "response_omitted": True,
        },
    )

    assert cli.state.simulator_mcp_config.handle == "env-from-file"
    assert cli.state.simulator_mcp_config.session_id == "session-from-file"
    stored_fact = cli._require_runtime().memory.facts["simulator_mcp_state"]["value"]
    stored_artifact = cli._require_runtime().memory.artifacts["simulator_mcp_state"]["value"]
    assert stored_fact["handle"] == "env-from-file"
    assert stored_fact["session_id"] == "session-from-file"
    assert stored_fact["mcp_server_url"] == "http://sim.example"
    assert stored_fact["dashboard_url"] == "http://sim.example/session/session-from-file"
    assert stored_artifact["dashboard_url"] == stored_fact["dashboard_url"]


def test_cli_caches_simulator_mcp_tool_catalog(monkeypatch) -> None:
    class FakeTransport:
        def __init__(self, url: str) -> None:
            self.url = url

        def list_tools(self, *, timeout_s=None):
            return {
                "tools": [
                    {
                        "name": "create_env",
                        "description": "Create a simulator environment.",
                        "input_schema": {
                            "type": "object",
                            "required": ["env_id"],
                            "properties": {
                                "env_id": {
                                    "type": "string",
                                    "description": "OpenETA env id",
                                }
                            },
                        },
                    }
                ],
                "tool_count": 1,
            }

        def call_tool(self, name, arguments, *, timeout_s=None):
            return {"success": True}

    monkeypatch.setattr("agent.cli.openeta_cli._load_sim_mcp_url", lambda: "http://sim/mcp")
    monkeypatch.setattr("agent.cli.openeta_cli.SseSimulatorMcpTransport", FakeTransport)

    cli = OpenEtaCli()

    catalog = cli.state.simulator_mcp_tool_catalog
    assert catalog["available"] is True
    assert catalog["tool_count"] == 1
    assert catalog["tools"][0]["name"] == "create_env"
    assert catalog["tools"][0]["required"] == ["env_id"]
    assert "response_path" in catalog
    assert "simulator_mcp_tool_catalog" in cli._require_runtime().memory.facts


def test_prompt_html_strips_ansi_and_escapes_markup() -> None:
    rendered = _prompt_html("\033[33mpermission\033[0m {'target': '<left>'}: ")

    text = str(rendered)
    assert "\033" not in text
    assert "&lt;left&gt;" in text


def test_cli_confirm_denies_without_tty(monkeypatch, capsys) -> None:
    class NonTty:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr("agent.cli.openeta_cli.sys.stdin", NonTty())
    cli = OpenEtaCli()

    assert cli.confirm("move_to {'target': '<left>'}?") is False
    assert "permission required" in capsys.readouterr().out


def test_cli_ask_human_prints_question(monkeypatch, capsys) -> None:
    cli = OpenEtaCli()
    monkeypatch.setattr(cli, "_prompt_text", lambda prompt: "openeta/libero_libero_10_task0-v0")
    action = EnvAction(
        action_type="response",
        command={
            "request": {
                "kind": "response",
                "name": "ask_human",
                "parameters": {"question": "Which LIBERO environment should I create?"},
            }
        },
    )

    cli._handle_interactive_action(action)

    assert "Which LIBERO environment should I create?" in capsys.readouterr().out
    assert cli.state.continue_after_human is True


def test_cli_ask_human_trace_prints_question_without_parameter_json(capsys) -> None:
    cli = OpenEtaCli()
    action = EnvAction(
        action_type="response",
        command={
            "status": "pending",
            "request": {
                "kind": "response",
                "name": "ask_human",
                "parameters": {
                    "question": "Which LIBERO environment should I create?",
                    "debug": {"large": "x" * 1000},
                },
                "reasoning": "Need operator choice.",
            },
        },
    )

    cli._print_action_trace(action)

    output = capsys.readouterr().out
    assert "request response::ask_human -> pending" in output
    assert "Which LIBERO environment should I create?" in output
    assert "ask_human" in output
    assert "parameters" not in output
    assert '"debug"' not in output
    assert "x" * 100 not in output


def test_cli_response_trace_prints_message_without_parameter_json(capsys) -> None:
    cli = OpenEtaCli()
    action = EnvAction(
        action_type="response",
        command={
            "status": "executed",
            "request": {
                "kind": "response",
                "name": "task_complete",
                "parameters": {
                    "message": "已查到 280 个可用环境。\n完整列表在 tmp/tool_result/response.json",
                    "debug": {"large": "x" * 1000},
                },
                "reasoning": "Results are ready for the user.",
            },
            "metadata": {
                "planner_metadata": {
                    "backend": {"provider": "openai-compatible", "model": "gpt-5.5"},
                    "backend_details": {
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                            "total_tokens": 12,
                        }
                    },
                }
            },
        },
    )

    cli._print_action_trace(action)

    output = capsys.readouterr().out
    assert "request response::task_complete -> executed" in output
    assert "已查到 280 个可用环境。" in output
    assert "完整列表在 tmp/tool_result/response.json" in output
    assert "assistant" in output
    assert "parameters" not in output
    assert '"debug"' not in output
    assert "x" * 100 not in output


def test_cli_tool_trace_collapses_large_result_to_key_facts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_module, "_tool_detail_width", lambda: 100)
    cli = OpenEtaCli()
    huge = "x" * 20_000
    action = EnvAction(
        action_type="tool_call",
        command={
            "status": "executed",
            "request": {
                "kind": "tool_call",
                "name": "create_simulator_env",
                "parameters": {
                    "env_id": "openeta/libero_libero_10_task0-v0",
                    "debug": huge,
                },
                "reasoning": "Create one simulator environment.",
            },
            "tool_calls": [
                {
                    "name": "create_simulator_env",
                    "status": "executed",
                    "result": {
                        "success": True,
                        "content": "Simulator environment created and reset.",
                        "details": {
                            "outputs": {
                                "environment": {
                                    "env_id": "openeta/libero_libero_10_task0-v0",
                                    "handle": "env-1",
                                    "session_id": "session-1",
                                    "dashboard_url": "http://sim/session/session-1",
                                },
                                "initial_observation": {
                                    "response_path": "tmp/tool_result/reset-response.json",
                                    "cameras": [{"rgb_base64": huge}],
                                },
                                "raw_backend_payload": huge,
                            },
                            "artifacts": [
                                {"path": "tmp/image/agentview-rgb.png"},
                                {"path": "tmp/tool_result/create-response.json"},
                            ],
                            "state_delta": {
                                "simulator_environment": {"handle": "env-1"},
                                "observation": {"raw": huge},
                            },
                            "diagnostics": [],
                        },
                    },
                }
            ],
            "safety_checks": [],
        },
    )

    cli._print_action_trace(action)

    output = capsys.readouterr().out
    result_lines = output.split("    result ", 1)[1].splitlines()
    assert len(result_lines) <= cli_module.TOOL_RESULT_MAX_LINES
    assert "Simulator environment created and reset." in output
    assert "artifacts=2" in output
    assert "env_id" in output
    assert "full result retained in session trace" in output
    assert huge not in output
    assert '"raw_backend_payload"' not in output
    assert '"debug"' not in output


def test_cli_tool_trace_prioritizes_failure_diagnostic(capsys) -> None:
    cli = OpenEtaCli()
    action = EnvAction(
        action_type="tool_call",
        command={
            "status": "failed",
            "request": {
                "kind": "tool_call",
                "name": "move_to",
                "parameters": {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
            },
            "tool_calls": [
                {
                    "name": "move_to",
                    "status": "failed",
                    "result": {
                        "success": False,
                        "content": "Simulator move failed.",
                        "details": {
                            "outputs": {"response": {"error": "IK failed"}},
                            "diagnostics": [
                                {
                                    "code": "simulator_mcp_target_not_reached",
                                    "message": "Target was not reached.",
                                }
                            ],
                            "artifacts": [],
                            "state_delta": {},
                        },
                    },
                }
            ],
            "safety_checks": [],
        },
    )

    cli._print_action_trace(action)

    output = capsys.readouterr().out
    assert "simulator_mcp_target_not_reached" in output
    assert "Target was not reached." in output
    assert "IK failed" in output


def test_cli_episode_steps_show_user_and_observation_only_on_first_turn(capsys) -> None:
    cli = OpenEtaCli()

    def step(turn_index: int) -> EpisodeStep:
        observation = EnvObservation(
            task="close the simulator",
            cameras=[CameraFrame(frame_id="front", rgb=[[[0, 0, 0]]])],
            robot=RobotState(),
            objects=[{"name": "cube"}],
            metadata={"step_idx": turn_index - 1},
        )
        return EpisodeStep(
            turn_index=turn_index,
            observation=observation,
            action=EnvAction(
                action_type="response",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "response",
                        "name": "task_complete",
                        "parameters": {"message": "done"},
                    },
                },
            ),
            step_result=StepResult(
                observation=observation,
                reward=0.0,
                terminated=turn_index == 2,
                truncated=False,
            ),
        )

    cli._print_episode_step(step(1))
    cli._print_episode_step(step(2))

    output = capsys.readouterr().out
    assert output.count("user close the simulator") == 1
    assert output.count("observation step=") == 1
    assert "turn 1" in output
    assert "turn 2" in output


def test_cli_prints_status_report_as_stopped(capsys) -> None:
    cli = OpenEtaCli()
    observation = EnvObservation(
        task="find image path",
        cameras=[CameraFrame(frame_id="front", rgb=[[[0, 0, 0]]])],
        robot=RobotState(),
        metadata={"step_idx": 0},
    )
    step = EpisodeStep(
        turn_index=1,
        observation=observation,
        action=EnvAction(
            action_type="response",
            command={
                "status": "executed",
                "request": {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {"message": "No image path is available."},
                },
            },
        ),
        step_result=StepResult(
            observation=observation,
            reward=0.0,
            terminated=True,
            truncated=False,
            info={"termination_reason": "status_report"},
        ),
    )
    cli._print_episode_result(
        EpisodeResult(
            task="find image path",
            session_id="session-0",
            steps=[step],
            terminated=True,
            metadata={"stop_reason": "status_report"},
        )
    )

    output = capsys.readouterr().out
    assert "episode stopped: status_report" in output
    assert "episode terminated: status_report" not in output


def test_empty_line_after_human_answer_continues_remaining_turns(monkeypatch) -> None:
    cli = OpenEtaCli()
    calls = []
    monkeypatch.setattr(cli, "continue_task", lambda *, max_turns: calls.append(max_turns))
    cli.state.continue_after_human = True

    cli._handle_empty_line()

    assert calls == [None]
    assert cli.state.continue_after_human is False


def test_cli_run_auto_continues_after_human_answer_without_reading_new_task(
    monkeypatch,
) -> None:
    class TtyInput:
        def isatty(self) -> bool:
            return True

    class FakeSession:
        def __init__(self) -> None:
            self.prompts = 0

        def prompt(self, *args, **kwargs):
            del args, kwargs
            self.prompts += 1
            if self.prompts == 1:
                return "pick ketchup"
            raise EOFError

    cli = OpenEtaCli()
    fake_session = FakeSession()
    cli.session = fake_session  # type: ignore[assignment]
    run_tasks = []
    continued = []

    def pause_for_human(task, *, max_turns):
        run_tasks.append((task, max_turns))
        cli.state.continue_after_human = True

    monkeypatch.setattr("agent.cli.openeta_cli.sys.stdin", TtyInput())
    monkeypatch.setattr(cli, "_print_header", lambda: None)
    monkeypatch.setattr(cli, "run_task", pause_for_human)
    monkeypatch.setattr(
        cli,
        "continue_task",
        lambda *, max_turns: continued.append(max_turns),
    )

    cli.run()

    assert run_tasks == [("pick ketchup", cli_module.DEFAULT_MAX_TURNS)]
    assert continued == [None]
    assert fake_session.prompts == 2


def test_step_command_still_continues_one_turn(monkeypatch) -> None:
    cli = OpenEtaCli()
    calls = []
    monkeypatch.setattr(cli, "continue_task", lambda *, max_turns: calls.append(max_turns))
    cli.state.continue_after_human = True

    assert cli._handle_command("/step") is True

    assert calls == [1]
    assert cli.state.continue_after_human is False


def test_new_session_resets_active_episode_and_starts_empty_working_memory() -> None:
    cli = OpenEtaCli()
    runtime = cli._require_runtime()
    runtime.memory.save_fact("target", {"name": "milk"}, source="unit")
    cli.state.current_task = "pick milk"
    cli.state.step_idx = 3
    cli.state.continue_after_human = True
    cli.state.episode_runner = object()  # type: ignore[assignment]

    assert cli._handle_command("/new") is True

    runtime = cli._require_runtime()
    assert cli.state.current_task == ""
    assert cli.state.step_idx == 0
    assert cli.state.continue_after_human is False
    assert cli.state.episode_runner is None
    assert "target" not in runtime.memory.facts


def test_new_session_can_clear_current_session_working_memory(monkeypatch) -> None:
    cli = OpenEtaCli()
    runtime = cli._require_runtime()
    runtime.start_session(task="pick milk")
    runtime.memory.save_fact("target", {"name": "milk"}, source="unit")
    session_id = runtime.memory.session_id
    assert session_id is not None
    store = runtime.memory.store
    assert isinstance(store, JsonMemoryStore)
    monkeypatch.setattr(cli, "confirm", lambda _message: True)

    assert cli._handle_command("/new --clear-working-memory") is True

    facts_path = store.working_dir_for(session_id) / "facts.json"
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    assert "target" not in facts
    runtime = cli._require_runtime()
    assert "target" not in runtime.memory.facts


def test_sessions_and_resume_load_session_scoped_working_memory(capsys) -> None:
    cli = OpenEtaCli()
    runtime = cli._require_runtime()
    runtime.start_session(task="pick milk")
    runtime.memory.save_fact("target", {"name": "milk"}, source="unit")
    session_id = runtime.memory.session_id
    assert session_id is not None
    cli._handle_command("/new")

    assert cli._handle_command(f"/resume {session_id[:12]}") is True

    runtime = cli._require_runtime()
    assert runtime.memory.session_id == session_id
    assert runtime.memory.facts["target"]["value"]["name"] == "milk"

    assert cli._handle_command("/sessions") is True
    output = capsys.readouterr().out
    assert session_id[:12] in output


def test_resume_without_args_lists_sessions_before_selection(capsys) -> None:
    cli = OpenEtaCli()
    runtime = cli._require_runtime()
    runtime.start_session(task="pick milk")
    session_id = runtime.memory.session_id
    assert session_id is not None
    cli._handle_command("/new")

    assert cli._handle_command("/resume") is True

    runtime = cli._require_runtime()
    assert runtime.memory.session_id is None
    output = capsys.readouterr().out
    assert "resumable sessions" in output
    assert session_id[:12] in output
    assert "resume cancelled" in output


def test_resume_picker_selection_loads_session(monkeypatch, capsys) -> None:
    cli = OpenEtaCli()
    runtime = cli._require_runtime()
    runtime.start_session(task="pick milk")
    runtime.memory.save_fact("target", {"name": "milk"}, source="unit")
    session_id = runtime.memory.session_id
    assert session_id is not None
    cli._handle_command("/new")
    monkeypatch.setattr(cli, "_prompt_resume_selection", lambda: "1")

    assert cli._handle_command("/resume") is True

    runtime = cli._require_runtime()
    assert runtime.memory.session_id == session_id
    assert runtime.memory.facts["target"]["value"]["name"] == "milk"
    output = capsys.readouterr().out
    assert "resumable sessions" in output
    assert f"resumed session: {session_id}" in output


def test_resume_last_loads_latest_session() -> None:
    cli = OpenEtaCli()
    runtime = cli._require_runtime()
    runtime.start_session(task="pick milk")
    first_session_id = runtime.memory.session_id
    assert first_session_id is not None
    cli._handle_command("/new")
    runtime = cli._require_runtime()
    runtime.start_session(task="place cup")
    second_session_id = runtime.memory.session_id
    assert second_session_id is not None
    cli._handle_command("/new")

    assert cli._handle_command("/resume --last") is True

    runtime = cli._require_runtime()
    assert runtime.memory.session_id == second_session_id
    assert runtime.memory.task == "place cup"


def test_cli_prints_codex_style_realtime_status(monkeypatch, capsys) -> None:
    clock = iter([0.0, 0.2, 1.7, 3.9])
    monkeypatch.setattr("agent.cli.openeta_cli._load_sim_mcp_url", lambda: "")
    monkeypatch.setattr("agent.cli.openeta_cli.time.monotonic", lambda: next(clock))
    cli = OpenEtaCli()

    cli._begin_agent_activity()
    cli._print_tool_event(
        {
            "phase": "start",
            "name": "python_exec",
            "effect": "world_mutating",
            "parameters": {"sandbox": "outside_sandbox", "code": "result = {'ok': True}"},
        }
    )
    cli._print_tool_event(
        {
            "phase": "end",
            "name": "python_exec",
            "success": True,
            "content": "python_exec completed",
        }
    )
    cli._finish_agent_activity("worked")

    output = capsys.readouterr().out
    assert "thinking" in output
    assert "working" in output
    assert "→" in output
    assert "✓" in output
    assert "python_exec" in output
    assert "world_mutating" in output
    assert "sandbox=outside_sandbox" in output
    assert "python_exec completed" in output
    assert "worked for 3.9s" in output
