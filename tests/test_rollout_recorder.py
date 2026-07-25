from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest,
    StaticPlannerBackend,
)
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.planner import PlannerDecision, ToolCallingPlanner
from agent.runtime.rollout import RolloutRecorder, validate_rollout_bundle
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.registry import ToolResult, build_default_tool_registry


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _tools():
    tools = build_default_tool_registry()
    tools.bind_handler(
        "scene_detector",
        lambda context: ToolResult(
            True,
            content="detected",
            details={"outputs": {"objects": [{"name": "cube"}]}},
        ),
    )
    return tools


def test_rollout_records_rejected_and_accepted_planner_attempts(tmp_path: Path) -> None:
    root = tmp_path / ".openeta_memory"
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "missing_tool",
                    "parameters": {"value": 1},
                    "reasoning": "invalid candidate",
                },
                {
                    "kind": "tool_call",
                    "name": "scene_detector",
                    "parameters": {"image": "front"},
                    "reasoning": "valid candidate",
                },
            ]
        ),
        max_validation_retries=1,
    )
    runtime = OpenEtaAgentRuntime(
        planner=planner,
        tools=_tools(),
        memory=AgentMemory(store=JsonMemoryStore(root)),
    )
    runtime.start_session(task="find the cube", session_id="rollout-retry")
    observation = DummyEpisodeEnvironment().reset(task="find the cube")

    runtime.act(observation)

    rollout = root / "sessions" / "rollout-retry" / "rollout"
    calls = _rows(rollout / "model_calls.jsonl")
    assert len(calls) == 2
    assert calls[0]["result"]["payload"]["name"] == "missing_tool"
    assert calls[0]["parsed_decision"]["parameters"] == {"value": 1}
    assert calls[0]["validation"]["accepted"] is False
    assert calls[0]["validation"]["errors"]
    assert calls[1]["result"]["payload"]["name"] == "scene_detector"
    assert calls[1]["validation"] == {"accepted": True, "errors": []}
    assert [call["seq"] for call in calls] == [1, 2]

    trace = (
        root / "sessions" / "rollout-retry" / "trace.jsonl"
    ).read_text(encoding="utf-8")
    assert "_rollout_exchange" not in trace


def test_rollout_transition_preserves_media_state_action_and_reward(tmp_path: Path) -> None:
    root = tmp_path / ".openeta_memory"
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(
            StaticPlannerBackend(
                {
                    "kind": "tool_call",
                    "name": "scene_detector",
                    "parameters": {"image": "front"},
                }
            )
        ),
        tools=_tools(),
        memory=AgentMemory(store=JsonMemoryStore(root)),
    )
    runner = OpenEtaEpisodeRunner(
        runtime=runtime,
        environment=DummyEpisodeEnvironment(max_steps=1),
    )

    result = runner.run(task="find the cube", max_turns=1)

    session_id = result.session_id
    assert session_id is not None
    rollout = root / "sessions" / session_id / "rollout"
    manifest = json.loads((rollout / "manifest.json").read_text(encoding="utf-8"))
    transitions = _rows(rollout / "transitions.jsonl")
    episodes = _rows(rollout / "episodes.jsonl")
    artifacts = _rows(rollout / "artifacts.jsonl")

    assert manifest["schema_version"] == "openeta.rollout.v1"
    assert manifest["provenance"]["git"]["commit"]
    assert any(tool["name"] == "scene_detector" for tool in manifest["provenance"]["tools"])
    assert [event["event"] for event in episodes] == ["start", "result"]
    assert len(transitions) == 1
    transition = transitions[0]
    assert transition["reward"] == 0.0
    assert transition["terminated"] is True
    assert transition["action"]["command"]["request"]["name"] == "scene_detector"
    assert transition["observation"]["objects"] == [
        {"name": "cube", "position": [0.2, 0.0, 0.0]}
    ]

    rgb = transition["observation"]["cameras"][0]["rgb"]
    depth = transition["observation"]["cameras"][0]["depth"]
    assert (rollout / rgb["bundle_path"]).is_file()
    assert (rollout / depth["bundle_path"]).is_file()
    assert Image.open(rollout / rgb["bundle_path"]).getpixel((0, 0)) == (0, 0, 0)
    np.testing.assert_array_equal(np.load(rollout / depth["bundle_path"]), [[1.0]])
    assert {entry["mime_type"] for entry in artifacts} >= {
        "image/png",
        "application/x-npy",
    }
    assert all(entry["artifact_id"] == f"sha256:{entry['sha256']}" for entry in artifacts)
    validation = validate_rollout_bundle(rollout)
    assert validation["valid"] is True
    assert validation["streams"]["transitions"] == 1
    assert validation["artifact_count"] == 2


def test_rollout_externalizes_provider_images_and_redacts_secrets(tmp_path: Path) -> None:
    image_path = tmp_path / "scene.png"
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(image_path)
    captured: dict = {}

    def transport(url, body, headers, timeout_s):
        captured.update({"url": url, "body": body, "headers": headers, "timeout_s": timeout_s})
        return {
            "id": "response-1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"kind":"response","name":"talk",'
                            '"parameters":{"message":"done"}}'
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="vlm",
            api_base="https://provider.example",
            api_key="test-key",
            enable_vision=True,
        ),
        transport=transport,
    )
    request = PlannerBackendRequest(
        tool_context={"vision_image_paths": [str(image_path)]},
        system_prompt="Use Bearer top-secret-token only in this test.",
    )
    result = backend.decide(request)
    recorder = RolloutRecorder(tmp_path / "memory")
    recorder.start_session(
        session_id="vision",
        task="inspect",
        provenance={"api_key": "test-key"},
    )
    now = time.time()
    recorder.record_model_call(
        request=request,
        result=result,
        decision=PlannerDecision(
            action_type="response",
            action="talk",
            parameters={"message": "done"},
        ),
        validation_errors=[],
        backend=backend.descriptor(),
        started_at_s=now,
        completed_at_s=now + 0.1,
    )

    rollout = tmp_path / "memory" / "sessions" / "vision" / "rollout"
    text = (rollout / "model_calls.jsonl").read_text(encoding="utf-8")
    assert "test-key" not in text
    assert "top-secret-token" not in text
    assert "data:image/png;base64" not in text
    assert "<redacted-api-key>" in text or "<redacted>" in text

    call = _rows(rollout / "model_calls.jsonl")[0]
    exchange = call["provider_exchange"]["attempts"][0]
    image_content = exchange["request_body"]["messages"][-1]["content"][1]
    assert image_content["image_url"]["url"]["artifact_id"].startswith("sha256:")
    assert call["artifact_refs"]
    assert captured["headers"]["Authorization"].endswith("test-key")
