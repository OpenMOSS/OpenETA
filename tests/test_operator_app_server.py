from __future__ import annotations

import json
from pathlib import Path

from scripts.embodied.operator_app_server import (
    _compact_event,
    _continuation_prompt,
    _episode_projection,
    _operator_tool_count,
    _scrub_event,
    _thread_start_result,
    _write_persistent_event,
)
from scripts.embodied.operator_provider_config import render_provider_config


def test_operator_provider_config_copies_only_selected_provider(
    tmp_path: Path,
) -> None:
    source = tmp_path / "config.toml"
    source.write_text(
        """
model = "host-model"
model_provider = "other"

[features]
memories = true

[model_providers.other]
name = "Other"
base_url = "https://other.invalid/v1"
wire_api = "responses"

[model_providers.example]
name = "OpenAI"
base_url = "https://example.invalid/v1"
wire_api = "responses"
request_max_retries = 10
auth = { command = "/bin/sh", args = ["-c", "printf token"], timeout_ms = 5000 }
""".lstrip(),
        encoding="utf-8",
    )

    rendered = render_provider_config(source, "example")

    assert 'model_provider = "example"' in rendered
    assert "[model_providers.example]" in rendered
    assert 'base_url = "https://example.invalid/v1"' in rendered
    assert 'auth = { command = "/bin/sh"' in rendered
    assert "host-model" not in rendered
    assert "memories" not in rendered
    assert "other.invalid" not in rendered


def test_operator_provider_config_supports_builtin_openai(
    tmp_path: Path,
) -> None:
    rendered = render_provider_config(tmp_path / "not-needed.toml", "openai")

    assert rendered == 'model_provider = "openai"\n'


def test_thread_start_result_exposes_effective_reasoning_effort() -> None:
    message = {
        "id": 7,
        "result": {
            "thread": {"id": "thread-1"},
            "reasoningEffort": "medium",
        },
    }

    assert _thread_start_result(message, 7) == ("thread-1", "medium")
    assert _thread_start_result(message, 8) == (None, None)


def test_persistent_operator_continuation_uses_live_episode_state(tmp_path: Path) -> None:
    (tmp_path / "current.json").write_text(
        json.dumps(
            {
                "status": "running",
                "latest_issue": {
                    "code": "pose_execution_mismatch",
                    "message": "target not reached",
                },
            }
        )
    )
    (tmp_path / "operator_context.jsonl").write_text("{}\n{}\n{}\n")

    prompt = _continuation_prompt(tmp_path, previous_count=1, turn_number=2)

    assert _episode_projection(tmp_path)["status"] == "running"
    assert _operator_tool_count(tmp_path) == 3
    assert "previous turn made 2 tool call" in prompt
    assert "pose_execution_mismatch" in prompt
    assert "finish_episode" in prompt


def test_episode_projection_falls_back_to_episode_record(tmp_path: Path) -> None:
    (tmp_path / "episode.json").write_text(json.dumps({"status": "failed"}))

    assert _episode_projection(tmp_path)["status"] == "failed"


def test_scrub_event_omits_image_bytes_but_retains_metadata() -> None:
    event = {
        "method": "item/completed",
        "params": {
            "item": {
                "type": "mcpToolCall",
                "tool": "observe",
                "result": {
                    "content": [
                        {"type": "text", "text": '{"kind":"observation"}'},
                        {"type": "image", "mimeType": "image/png", "data": "abc123"},
                    ]
                },
            }
        },
    }

    scrubbed = _scrub_event(event)

    content = scrubbed["params"]["item"]["result"]["content"]
    assert content[0]["text"] == '{"kind":"observation"}'
    assert content[1]["mimeType"] == "image/png"
    assert content[1]["data"] == "<omitted 6 chars>"


def test_scrub_event_omits_images_nested_inside_serialized_json_text() -> None:
    nested = json.dumps(
        {
            "content": [
                {"type": "text", "text": "semantic result"},
                {"type": "image", "mimeType": "image/png", "data": "x" * 5000},
            ]
        }
    )
    event = {"type": "text", "text": nested}

    scrubbed = _scrub_event(event)
    decoded = json.loads(scrubbed["text"])

    assert decoded["content"][0]["text"] == "semantic result"
    assert decoded["content"][1]["data"] == "<omitted 5000 chars>"


def test_compact_event_hides_deltas_and_summarizes_mcp_result() -> None:
    assert _compact_event({"method": "item/agentMessage/delta", "params": {"delta": "x"}}) is None

    rendered = _compact_event(
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "type": "mcpToolCall",
                    "tool": "propose_grasps",
                    "status": "completed",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "kind": "grasp_candidates",
                                        "success": True,
                                        "observation_id": "obs-2",
                                        "candidate_ids": ["g0", "g1", "g2"],
                                        "message": "inspect in Viser",
                                    }
                                ),
                            },
                            {"type": "image", "data": "large"},
                        ]
                    },
                }
            },
        }
    )

    assert rendered is not None
    assert "propose_grasps -> completed" in rendered
    assert '"candidate_count":3' in rendered
    assert "large" not in rendered


def test_persistent_log_filters_deltas_and_compacts_scrubbed_items(tmp_path: Path) -> None:
    path = tmp_path / "operator_app_server.jsonl"
    image_base64 = "iVBORw0KGgo" + "A" * 64
    messages = [
        {
            "method": "item/agentMessage/delta",
            "params": {"delta": "token", "nested": {"image": image_base64}},
        },
        {
            "method": "item/started",
            "params": {"item": {"id": "agent-1", "type": "agentMessage", "text": "partial"}},
        },
        {
            "method": "item/completed",
            "params": {
                "turnId": "turn-1",
                "item": {
                    "id": "agent-1",
                    "type": "agentMessage",
                    "status": "completed",
                    "text": "final answer",
                },
            },
        },
        {
            "method": "item/started",
            "params": {
                "item": {
                    "id": "tool-1",
                    "type": "mcpToolCall",
                    "server": "openeta",
                    "tool": "observe",
                    "status": "inProgress",
                    "arguments": {"nested": {"image_base64": image_base64}},
                }
            },
        },
        {
            "method": "item/completed",
            "params": {
                "item": {
                    "id": "tool-1",
                    "type": "mcpToolCall",
                    "server": "openeta",
                    "tool": "observe",
                    "status": "completed",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "kind": "observation",
                                        "success": True,
                                        "nested": {"image": {"base64": image_base64}},
                                    }
                                ),
                            },
                            {"type": "image", "data": image_base64},
                        ]
                    },
                }
            },
        },
        {
            "method": "turn/completed",
            "params": {
                "turn": {"id": "turn-1", "status": "completed"},
                "metadata": json.dumps(
                    {"nested": {"image": {"base64": image_base64}}}
                ),
            },
        },
    ]

    with path.open("w", encoding="utf-8") as stream:
        written = [_write_persistent_event(stream, message) for message in messages]

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    serialized = path.read_text(encoding="utf-8")

    assert written == [False, False, True, True, True, True]
    assert [record["method"] for record in records] == [
        "item/completed",
        "item/started",
        "item/completed",
        "turn/completed",
    ]
    assert records[0]["params"]["item"]["text"] == "final answer"
    assert records[1]["params"]["item"] == {
        "id": "tool-1",
        "type": "mcpToolCall",
        "server": "openeta",
        "tool": "observe",
        "status": "inProgress",
    }
    assert records[2]["params"]["item"]["resultSummary"] == (
        '{"kind":"observation","success":true}'
    )
    assert "arguments" not in records[1]["params"]["item"]
    assert "result" not in records[2]["params"]["item"]
    assert "<omitted" in records[3]["params"]["metadata"]
    assert image_base64 not in serialized
    assert "token" not in serialized
    assert "partial" not in serialized
