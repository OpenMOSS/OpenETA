from __future__ import annotations

import json

from adapter.protocol import EnvAction
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest,
)
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore


def _action(name: str, *, index: int = 0, payload: str = "") -> EnvAction:
    return EnvAction(
        action_type="tool_call",
        command={
            "status": "executed",
            "request": {
                "kind": "tool_call",
                "name": name,
                "parameters": {
                    "index": index,
                    **({"image": payload} if payload else {}),
                },
            },
            "tool_calls": [
                {
                    "name": name,
                    "status": "executed",
                    "result": {
                        "success": True,
                        "content": "ok",
                        "details": {
                            "outputs": {"index": index},
                            "artifacts": [{"path": f"/tmp/artifact-{index}.json"}],
                        },
                    },
                }
            ],
        },
    )


def _environment_action(
    task: str,
    *,
    env_id: str = "openeta/libero-task0-v0",
    handle: str = "env-1",
    session_id: str = "sim-session-1",
) -> EnvAction:
    return EnvAction(
        action_type="tool_call",
        command={
            "status": "executed",
            "request": {
                "kind": "tool_call",
                "name": "create_simulator_env",
                "parameters": {"env_id": env_id},
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
                                "assigned_task": task,
                                "environment": {
                                    "env_id": env_id,
                                    "handle": handle,
                                    "session_id": session_id,
                                },
                                "observation_summary": {"task": task},
                            },
                            "state_delta": {"observation": {"task": task}},
                        },
                    },
                }
            ],
        },
    )


def _close_environment_action() -> EnvAction:
    return EnvAction(
        action_type="tool_call",
        command={
            "status": "executed",
            "request": {
                "kind": "tool_call",
                "name": "close_simulator_env",
                "parameters": {},
            },
            "tool_calls": [
                {
                    "name": "close_simulator_env",
                    "status": "executed",
                    "result": {
                        "success": True,
                        "content": "Simulator environment closed.",
                        "details": {"outputs": {"closed": True}},
                    },
                }
            ],
        },
    )


def test_user_constraint_survives_many_operational_events() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick milk")
    memory.begin_user_turn(
        "You may pick the cube, but keep the gripper closed until after the lift.",
        source="episode_start",
    )

    for index in range(30):
        memory.add_action(_action("move_to", index=index))
        memory.record("diagnostic", {"index": index})

    messages = memory.model_conversation_messages()
    context = memory.planning_context(max_events=8)

    assert messages[0] == {"role": "user", "content": "pick milk"}
    assert any("keep the gripper closed" in message["content"] for message in messages)
    assert [message["role"] for message in messages[-2:]] == ["assistant", "user"]
    assert "OpenETA host execution evidence" in messages[-1]["content"]
    assert context["current_user_request"].startswith("You may pick the cube")
    assert context["task"] == context["current_user_request"]
    assert all(event["type"] != "user_message" for event in context["recent_events"])


def test_environment_assigned_task_survives_many_tool_calls() -> None:
    memory = AgentMemory()
    memory.start_session(task="Create an environment and complete its assigned task.")
    assigned_task = "pick up alphabet soup and place it into basket"
    memory.add_action(_environment_action(assigned_task))

    for index in range(30):
        memory.add_action(_action("move_to", index=index))
        memory.record("environment_receipt", {"index": index})

    active = memory.active_environment_task()
    context = memory.planning_context(max_events=8)

    assert active is not None
    assert active["task"] == assigned_task
    assert active["env_id"] == "openeta/libero-task0-v0"
    assert context["active_environment_task"] == active
    assert assigned_task not in json.dumps(context["recent_events"])


def test_environment_assigned_task_replaces_and_clears_with_environment_lifecycle() -> None:
    memory = AgentMemory()
    memory.start_session(task="Run simulator tasks.")
    memory.add_action(_environment_action("pick milk", handle="env-1"))
    memory.add_action(_close_environment_action())

    assert memory.active_environment_task() is None

    memory.add_action(
        _environment_action(
            "pick cube",
            env_id="openeta/libero-task1-v0",
            handle="env-2",
            session_id="sim-session-2",
        )
    )

    active = memory.active_environment_task()
    assert active is not None
    assert active["task"] == "pick cube"
    assert active["env_id"] == "openeta/libero-task1-v0"
    assert active["handle"] == "env-2"
    assert active["session_id"] == "sim-session-2"
    assert active["source_tool"] == "create_simulator_env"
    assert active["source_field"] == "outputs.assigned_task"
    assert active["scene_epoch"] == 0


def test_environment_assigned_task_survives_compaction_and_resume(tmp_path) -> None:
    root = tmp_path / ".openeta_memory"
    memory = AgentMemory(store=JsonMemoryStore(root))
    memory.start_session(task="Create an environment and complete its assigned task.")
    memory.add_action(_environment_action("pick up alphabet soup"))
    for index in range(20):
        memory.add_action(_action("observe", index=index))
    memory.compact(max_events=2)
    session_id = memory.session_id
    assert session_id is not None

    resumed = AgentMemory(store=JsonMemoryStore(root))
    resumed.resume_session(session_id, max_events=1)

    assert resumed.active_environment_task()["task"] == "pick up alphabet soup"
    assert resumed.planning_context()["active_environment_task"]["handle"] == "env-1"


def test_latest_user_request_supersedes_initial_task_for_planning() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick milk")
    memory.begin_user_turn("Pick the cube instead.", source="episode_start")
    memory.begin_user_turn("Close the simulator environment.", source="episode_start")

    context = memory.planning_context()

    assert memory.task == "pick milk"
    assert memory.current_user_request == "Close the simulator environment."
    assert context["session_initial_task"] == "pick milk"
    assert context["task"] == "Close the simulator environment."
    assert [message["content"] for message in memory.model_conversation_messages()] == [
        "pick milk",
        "Pick the cube instead.",
        "Close the simulator environment.",
    ]


def test_compaction_checkpoint_and_resume_rebuild_canonical_history(tmp_path) -> None:
    root = tmp_path / ".openeta_memory"
    memory = AgentMemory(store=JsonMemoryStore(root))
    memory.start_session(task="pick alphabet soup")
    for index in range(20):
        memory.add_action(_action("move_to", index=index))
    memory.begin_user_turn(
        "Do not release the object; lift once more and then verify attachment.",
        source="episode_start",
    )
    memory.record("episode_interrupted", {"code": "keyboard_interrupt"})

    memory.compact(max_events=4)
    session_id = memory.session_id
    assert session_id is not None

    resumed = AgentMemory(store=JsonMemoryStore(root))
    resumed.resume_session(session_id, max_events=1)

    assert resumed.current_user_request.startswith("Do not release")
    assert any(
        message["content"].startswith("Do not release")
        for message in resumed.model_conversation_messages()
    )
    assert sum(item.kind == "action" for item in resumed.conversation.items) == 12
    assert resumed.conversation.checkpoint["dropped_item_count"] == 16
    assert "assistant action: tool_call::move_to" in resumed.conversation_checkpoint_summary()

    records = JsonMemoryStore(root).load_conversation_records(session_id)
    assert records[-1]["record_type"] == "checkpoint"
    assert records[-1]["replacement_items"]


def test_conversation_action_envelope_omits_inline_image_payload(tmp_path) -> None:
    root = tmp_path / ".openeta_memory"
    memory = AgentMemory(store=JsonMemoryStore(root))
    memory.start_session(task="inspect image")
    memory.add_action(_action("observe", payload="data:image/png;base64," + "A" * 50_000))
    session_id = memory.session_id
    assert session_id is not None

    text = JsonMemoryStore(root).conversation_path(session_id).read_text(encoding="utf-8")

    assert "data:image/png;base64" not in text
    assert "<inline_image_omitted>" in text
    assert len(text) < 20_000


def test_backend_orders_stable_history_before_dynamic_tool_context() -> None:
    captured = {}

    def fake_transport(url, body, headers, timeout_s):
        del url, headers, timeout_s
        captured["body"] = body
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"kind":"response","name":"talk"}'},
                }
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 5,
                "total_tokens": 125,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="test-model",
            api_base="https://api.example.test",
            api_key="secret-key",
            enable_vision=False,
        ),
        transport=fake_transport,
    )
    result = backend.decide(
        PlannerBackendRequest(
            tool_context={"task": "close simulator", "scene_epoch": 4},
            system_prompt="return json",
            conversation_summary="Earlier move_to calls completed.",
            conversation_messages=[
                {"role": "user", "content": "pick milk"},
                {"role": "assistant", "content": '{"openeta_action":{"name":"observe"}}'},
                {"role": "user", "content": "close simulator"},
            ],
        )
    )

    messages = captured["body"]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "system",
        "user",
        "assistant",
        "user",
        "user",
    ]
    assert messages[2]["content"] == "pick milk"
    assert json.loads(messages[-1]["content"])["tool_context"]["scene_epoch"] == 4
    assert result.details["usage"]["cached_tokens"] == 80
