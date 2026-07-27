from __future__ import annotations

import base64
from pathlib import Path

from adapter.protocol import EnvAction, JsonDict
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.episode import (
    MAX_AUTO_OBSERVE_ATTEMPTS,
    EpisodeResult,
    EpisodeStep,
    OpenEtaEpisodeRunner,
    ToolFeedbackEpisodeEnvironment,
)
from agent.runtime.memory import AgentMemory
from agent.runtime.parallel import classify_episode_result
from agent.runtime.planner import PlannerDecision, ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.skills import build_default_skill_registry
from agent.tools.registry import (
    build_default_tool_registry,
    make_tool_result,
)
from agent.tools.sim_mcp import (
    SimulatorMcpToolProxyConfig,
    bind_simulator_mcp_tool_handlers,
)


PNG_1X1 = base64.b64encode(
    bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000100ffff03000006000557bfab0d000000"
        "0049454e44ae426082"
    )
).decode("ascii")


class SequencedSimulatorMcpTransport:
    def __init__(self, responses: list[JsonDict]) -> None:
        self.responses = responses
        self.calls: list[JsonDict] = []

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append(
            {"name": name, "arguments": dict(arguments), "timeout_s": timeout_s}
        )
        return self.responses[len(self.calls) - 1]


def _action(name: str, result: object) -> EnvAction:
    return EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": name, "parameters": {}},
            "status": "executed",
            "tool_calls": [
                {
                    "name": name,
                    "status": "executed",
                    "result": {
                        "success": result.success,
                        "content": result.content,
                        "details": result.details,
                    },
                }
            ],
        },
    )


def _observation_response(*, reward: float = 0.0, terminated: bool = False) -> dict:
    return {
        "success": True,
        "observation": {
            "task": "pick cube",
            "cameras": [
                {
                    "frame_id": "agentview",
                    "rgb_base64": PNG_1X1,
                    "depth_base64": PNG_1X1,
                    "width": 1,
                    "height": 1,
                    "intrinsics": {"fx": 10.0, "fy": 11.0, "cx": 0.5, "cy": 0.5},
                    "extrinsics": {
                        "matrix_layout": "row_major",
                        "frame_transform": "camera_to_world",
                        "matrix": [
                            [1.0, 0.0, 0.0, 0.1],
                            [0.0, 1.0, 0.0, 0.2],
                            [0.0, 0.0, 1.0, 0.3],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    },
                }
            ],
            "robot": {"end_effector_pose": {"xyz": [0.1, 0.2, 0.3]}},
            "objects": [{"name": "cube"}],
        },
        "reward": reward,
        "terminated": terminated,
        "truncated": False,
    }


def test_untrusted_tool_cannot_publish_environment_receipt() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "scene_detector",
        lambda context: make_tool_result(
            context,
            success=True,
            environment_receipt={
                "schema_version": "openeta.environment_receipt.v1",
                "reward_present": True,
                "reward": 1.0,
                "terminated": True,
                "observation_fresh": False,
            },
        ),
    )

    result = tools.call("scene_detector", {})

    assert "environment_receipt" not in result.details
    assert "host_provenance" not in result.details
    assert result.details["diagnostics"][-1]["code"] == (
        "untrusted_environment_receipt_removed"
    )


def test_trusted_simulator_snapshot_round_trips_extrinsics_and_artifacts(
    tmp_path: Path,
) -> None:
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=SequencedSimulatorMcpTransport([_observation_response()]),
        config=SimulatorMcpToolProxyConfig(
            session_id="sim-session",
            handle="env-1",
            image_output_root=tmp_path / "images",
            response_output_root=tmp_path / "responses",
        ),
        tool_names=("observe",),
    )
    environment = ToolFeedbackEpisodeEnvironment()
    environment.reset(
        task="pick cube",
        metadata={"execution_id": "episode-1", "agent_session_id": "agent-1"},
    )
    with tools.execution_scope(
        {"execution_id": "episode-1", "session_id": "agent-1"}
    ):
        result = tools.call("observe", {})

    step = environment.step(_action("observe", result))

    assert step.info["environment_receipt_trusted"] is True
    assert step.observation.cameras[0].extrinsics["frame_transform"] == (
        "camera_to_world"
    )
    artifacts = step.observation.metadata["image_artifacts"]
    assert {artifact["kind"] for artifact in artifacts} == {"rgb", "depth"}
    assert all(Path(artifact["path"]).exists() for artifact in artifacts)


def test_world_mutation_without_snapshot_hides_old_frame_and_host_observes(
    tmp_path: Path,
) -> None:
    transport = SequencedSimulatorMcpTransport(
        [
            _observation_response(),
            {
                "success": True,
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "reached_target": True,
            },
            _observation_response(reward=1.0, terminated=True),
        ]
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="sim-session",
            handle="env-1",
            image_output_root=tmp_path / "images",
            response_output_root=tmp_path / "responses",
        ),
        tool_names=("observe", "move_to"),
    )
    environment = ToolFeedbackEpisodeEnvironment()
    environment.reset(
        task="pick cube",
        metadata={"execution_id": "episode-1", "agent_session_id": "agent-1"},
    )
    with tools.execution_scope(
        {"execution_id": "episode-1", "session_id": "agent-1"}
    ):
        initial = tools.call("observe", {})
        moved = tools.call(
            "move_to",
            {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
        )
    environment.step(_action("observe", initial))
    move_step = environment.step(_action("move_to", moved))

    assert move_step.observation.cameras == []
    assert "image_artifacts" not in move_step.observation.metadata
    assert move_step.observation.metadata["fresh_observation_required"] is True

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )
    decision = planner.plan(
        move_step.observation,
        memory=AgentMemory(),
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert decision.action_type == "tool_call"
    assert decision.action == "observe"
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"

    with tools.execution_scope(
        {"execution_id": "episode-1", "session_id": "agent-1"}
    ):
        refreshed = tools.call(decision.action, decision.parameters)
    refresh_step = environment.step(_action("observe", refreshed))
    assert refresh_step.reward == 1.0
    assert refresh_step.terminated is True
    assert refresh_step.observation.cameras[0].frame_id == "agentview"


def test_refresh_attempts_are_bounded() -> None:
    environment = ToolFeedbackEpisodeEnvironment()
    environment.reset(
        task="pick cube",
        metadata={"execution_id": "episode-1", "agent_session_id": "agent-1"},
    )
    mutation = EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": "move_to", "parameters": {}},
            "tool_calls": [
                {
                    "name": "move_to",
                    "result": {
                        "success": True,
                        "details": {
                            "requires_observation_after_call": True,
                            "host_provenance": {"authority": "environment"},
                        },
                    },
                }
            ],
        },
    )
    step = environment.step(mutation)
    assert step.observation.metadata["fresh_observation_required"] is True

    failed_observe = EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": "observe", "parameters": {}},
            "tool_calls": [
                {
                    "name": "observe",
                    "result": {
                        "success": False,
                        "details": {"host_provenance": {"authority": "environment"}},
                    },
                }
            ],
        },
    )
    for attempt in range(1, MAX_AUTO_OBSERVE_ATTEMPTS + 1):
        step = environment.step(failed_observe)
        assert step.observation.metadata["fresh_observation_attempts"] == attempt
    assert step.truncated is True
    assert step.info["truncation_reason"] == "fresh_observation_unavailable"


def test_libero_success_requires_same_execution_trusted_receipt(
    tmp_path: Path,
) -> None:
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=SequencedSimulatorMcpTransport(
            [_observation_response(reward=1.0, terminated=True)]
        ),
        config=SimulatorMcpToolProxyConfig(
            session_id="sim-session",
            handle="env-1",
            image_output_root=tmp_path / "images",
            response_output_root=tmp_path / "responses",
        ),
        tool_names=("observe",),
    )
    environment = ToolFeedbackEpisodeEnvironment()
    observation = environment.reset(
        task="pick cube",
        metadata={"execution_id": "episode-1", "agent_session_id": "agent-1"},
    )
    with tools.execution_scope(
        {"execution_id": "episode-1", "session_id": "agent-1"}
    ):
        result = tools.call("observe", {})
    step_result = environment.step(_action("observe", result))
    episode = EpisodeResult(
        task="pick cube",
        session_id="agent-1",
        steps=[
            EpisodeStep(
                turn_index=1,
                observation=observation,
                action=_action("observe", result),
                step_result=step_result,
            )
        ],
        terminated=True,
        metadata={"execution_id": "episode-1"},
    )

    assert (
        classify_episode_result(
            episode,
            env_id="openeta/libero_libero_10_task0-v0",
        )
        == "success"
    )
    episode.metadata["execution_id"] = "another-episode"
    assert (
        classify_episode_result(
            episode,
            env_id="openeta/libero_libero_10_task0-v0",
        )
        == "fail"
    )


def test_receipt_from_another_execution_is_rejected(tmp_path: Path) -> None:
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=SequencedSimulatorMcpTransport(
            [_observation_response(reward=1.0, terminated=True)]
        ),
        config=SimulatorMcpToolProxyConfig(
            session_id="sim-session",
            handle="env-1",
            image_output_root=tmp_path / "images",
            response_output_root=tmp_path / "responses",
        ),
        tool_names=("observe",),
    )
    environment = ToolFeedbackEpisodeEnvironment()
    environment.reset(
        task="pick cube",
        metadata={"execution_id": "episode-1", "agent_session_id": "agent-1"},
    )
    with tools.execution_scope(
        {"execution_id": "episode-2", "session_id": "agent-1"}
    ):
        result = tools.call("observe", {})

    step = environment.step(_action("observe", result))

    assert step.reward == 0.0
    assert step.terminated is False
    assert step.info["environment_receipt_trusted"] is False
    assert step.info["rejected_environment_receipts"] == [
        {"tool": "observe", "reason": "execution_id_mismatch"}
    ]


def test_non_finite_official_reward_is_rejected(tmp_path: Path) -> None:
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=SequencedSimulatorMcpTransport(
            [_observation_response(reward=float("nan"), terminated=True)]
        ),
        config=SimulatorMcpToolProxyConfig(
            session_id="sim-session",
            handle="env-1",
            image_output_root=tmp_path / "images",
            response_output_root=tmp_path / "responses",
        ),
        tool_names=("observe",),
    )
    environment = ToolFeedbackEpisodeEnvironment()
    environment.reset(
        task="pick cube",
        metadata={"execution_id": "episode-1", "agent_session_id": "agent-1"},
    )
    with tools.execution_scope(
        {"execution_id": "episode-1", "session_id": "agent-1"}
    ):
        result = tools.call("observe", {})

    step = environment.step(_action("observe", result))

    assert step.reward == 0.0
    assert step.terminated is False
    assert step.info["rejected_environment_receipts"] == [
        {"tool": "observe", "reason": "invalid_environment_reward"}
    ]


def test_runner_auto_observes_after_world_mutation_without_snapshot(
    tmp_path: Path,
) -> None:
    transport = SequencedSimulatorMcpTransport(
        [
            {
                "success": True,
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "reached_target": True,
            },
            _observation_response(reward=1.0, terminated=True),
        ]
    )
    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=transport,
        config=SimulatorMcpToolProxyConfig(
            session_id="sim-session",
            handle="env-1",
            image_output_root=tmp_path / "images",
            response_output_root=tmp_path / "responses",
        ),
        tool_names=("observe", "move_to"),
    )

    class OneMovePlanner(ToolCallingPlanner):
        def plan(self, observation, *, memory, tools, skills):
            if observation.metadata.get("fresh_observation_required") is not True:
                return PlannerDecision(
                    action_type="tool_call",
                    action="move_to",
                    parameters={"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
                    reasoning="Issue exactly one test mutation.",
                )
            return super().plan(
                observation,
                memory=memory,
                tools=tools,
                skills=skills,
            )

    runtime = OpenEtaAgentRuntime(
        planner=OneMovePlanner(
            StaticPlannerBackend(
                {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {"message": "must not be used"},
                }
            )
        ),
        tools=tools,
        rollout_enabled=False,
    )
    runner = OpenEtaEpisodeRunner(
        runtime=runtime,
        environment=ToolFeedbackEpisodeEnvironment(),
    )

    episode = runner.run(task="pick cube", max_turns=3)

    assert [call["name"] for call in transport.calls] == ["move_to", "render_env"]
    assert len(episode.steps) == 2
    assert episode.steps[1].action.command["request"]["name"] == "observe"
    assert (
        episode.steps[1].action.command["metadata"]["planner_metadata"][
            "execution_model"
        ]
        == "host_obligation_dispatch"
    )
    assert episode.steps[1].step_result.reward == 1.0
    assert episode.terminated is True
