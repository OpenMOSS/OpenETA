from __future__ import annotations

import json
from pathlib import Path

from adapter.agent import AgentAdapter
from adapter.bridge import AgentSimBridge
from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState, StepResult
from adapter.sim import SimulatorAdapter
from logger.episode_logger import EPISODE_SCHEMA_VERSION, STEP_SCHEMA_VERSION, EpisodeLogger
from logger.replay import build_timeline, export_video


def _observation(task: str, step: int) -> EnvObservation:
    value = min(255, 20 + step * 40)
    rgb = [[[value, x * 8, y * 8] for x in range(16)] for y in range(16)]
    depth = [[0.5 + step * 0.01 for _ in range(16)] for _ in range(16)]
    return EnvObservation(
        task=task,
        cameras=[
            CameraFrame(
                frame_id="front",
                rgb=rgb,
                depth=depth,
                intrinsics={"fx": 10.0, "fy": 10.0, "cx": 7.5, "cy": 7.5},
                extrinsics={"frame_transform": "camera_to_world"},
            )
        ],
        robot=RobotState(
            joint_positions=[float(step)],
            end_effector_pose={"xyz": [0.0, 0.0, 0.5]},
            gripper_state={"open": True},
        ),
        objects=[{"name": "cube", "position": [0.1, 0.0, 0.0]}],
        metadata={"step_index": step},
    )


class RecordingSimulator(SimulatorAdapter):
    def __init__(self, *, terminate_at: int | None = None) -> None:
        self.task = ""
        self.steps = 0
        self.actions: list[EnvAction] = []
        self.terminate_at = terminate_at
        self.closed = False

    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        del seed
        self.task = task or ""
        self.steps = 0
        return self.observe()

    def observe(self) -> EnvObservation:
        return _observation(self.task, self.steps)

    def step(self, action: EnvAction) -> StepResult:
        self.actions.append(action)
        self.steps += 1
        terminated = self.terminate_at is not None and self.steps >= self.terminate_at
        return StepResult(
            observation=self.observe(),
            reward=1.0 if terminated else 0.0,
            terminated=terminated,
            info={"success": True} if terminated else {},
        )

    def close(self) -> None:
        self.closed = True


class ScriptedAgent(AgentAdapter):
    def __init__(self, actions: list[EnvAction]) -> None:
        self.actions = actions
        self.index = 0
        self.memory: list[dict] = []
        self.closed = False

    def start_session(self, *, task: str, metadata: dict | None = None) -> None:
        self.index = 0
        self.memory.append({"type": "session", "task": task, "metadata": metadata})

    def act(self, observation: EnvObservation) -> EnvAction:
        del observation
        action = self.actions[self.index]
        self.index += 1
        return action

    def update_memory(self, event: dict) -> None:
        self.memory.append(event)

    def close(self) -> None:
        self.closed = True


def _tool_action(name: str) -> EnvAction:
    return EnvAction(
        action_type="tool_call",
        command={
            "schema_version": "openeta.agent_command.v1",
            "request": {"kind": "tool_call", "name": name, "parameters": {}},
            "status": "executed",
            "safety_checks": [{"name": "ik", "status": "executed"}],
            "metadata": {"checker_results": {"post_failure_checks": []}},
        },
    )


def _response(name: str) -> EnvAction:
    return EnvAction(
        action_type="response",
        command={
            "schema_version": "openeta.agent_command.v1",
            "request": {"kind": "response", "name": name, "parameters": {}},
            "status": "executed",
        },
    )


def test_run_episode_logs_media_replays_and_never_steps_response(tmp_path: Path) -> None:
    simulator = RecordingSimulator()
    agent = ScriptedAgent([_tool_action("move_to"), _tool_action("gripper_control"), _response("task_complete")])
    bridge = AgentSimBridge(simulator=simulator, agent=agent)
    episode_logger = EpisodeLogger(tmp_path / "episodes")

    outcome = bridge.run_episode(
        task="pick and place the cube",
        seed=7,
        max_steps=5,
        logger=episode_logger,
        environment="dummy",
        episode_id="ep-test",
    )

    assert outcome.status == "success"
    assert outcome.success is True
    assert outcome.turns == 3
    assert outcome.simulator_steps == 2
    assert len(simulator.actions) == 2
    assert all(action.action_type == "tool_call" for action in simulator.actions)
    assert len([event for event in agent.memory if event["type"] == "step_result"]) == 2

    episode_dir = tmp_path / "episodes" / "ep-test"
    metadata = json.loads((episode_dir / "episode.json").read_text())
    lines = (episode_dir / "steps.jsonl").read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert metadata["schema_version"] == EPISODE_SCHEMA_VERSION
    assert metadata["status"] == "success"
    assert metadata["step_count"] == 3
    assert all(event["schema_version"] == STEP_SCHEMA_VERSION for event in events)
    assert events[2]["step_result"] is None
    assert "rgb" not in events[0]["observation"]
    rgb_ref = events[0]["observation"]["camera_refs"][0]["rgb"]
    depth_ref = events[0]["observation"]["camera_refs"][0]["depth"]
    assert (episode_dir / rgb_ref).is_file()
    assert (episode_dir / depth_ref).is_file()

    timeline = build_timeline(episode_dir)
    assert [item["action"]["command"]["request"]["name"] for item in timeline] == [
        "move_to",
        "gripper_control",
        "task_complete",
    ]
    video = export_video(episode_dir, tmp_path / "replay.mp4", camera="front", fps=5)
    assert video.is_file() and video.stat().st_size > 0


def test_run_episode_uses_simulator_success_evidence(tmp_path: Path) -> None:
    simulator = RecordingSimulator(terminate_at=1)
    agent = ScriptedAgent([_tool_action("move_to")])
    outcome = AgentSimBridge(simulator=simulator, agent=agent).run_episode(
        task="move",
        max_steps=2,
        logger=EpisodeLogger(tmp_path),
        episode_id="terminated",
    )
    assert outcome.status == "terminated"
    assert outcome.success is True
    assert outcome.simulator_steps == 1
    assert outcome.final_observation.metadata["step_index"] == 1


def test_run_episode_ask_human_and_close_are_non_mutating() -> None:
    simulator = RecordingSimulator()
    agent = ScriptedAgent([_response("ask_human")])
    bridge = AgentSimBridge(simulator=simulator, agent=agent)
    outcome = bridge.run_episode(task="ambiguous task", max_steps=1)
    assert outcome.status == "need_human"
    assert outcome.success is None
    assert simulator.actions == []

    bridge.close()
    assert agent.closed is True
    assert simulator.closed is True


def test_unnamed_response_is_never_forwarded_to_simulator() -> None:
    simulator = RecordingSimulator()
    agent = ScriptedAgent([EnvAction(action_type="response")])
    outcome = AgentSimBridge(simulator=simulator, agent=agent).run_episode(
        task="stop", max_steps=1
    )
    assert outcome.status == "response"
    assert outcome.simulator_steps == 0
    assert simulator.actions == []
