from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import (
    PlannerBackend,
    PlannerBackendRequest,
    PlannerBackendResult,
    StaticPlannerBackend,
)
from agent.runtime.actions import PipelineStatus
from agent.runtime.episode import action_token_usage
from agent.evals.visual_history import run_visual_history_projection_ablation
from agent.cli.visual_history_eval import build_session_visual_history_ablation
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.planner import PlannerContextConfig, ToolCallingPlanner, build_tool_context
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.skills import build_default_skill_registry
from agent.runtime.visual_history import (
    VisualHistoryConfig,
    VisualHistoryManager,
    build_visual_history_projection,
    observation_history,
    visual_delta_history,
)
from agent.tools.registry import build_default_tool_registry
from adapter.protocol import EnvAction


class RecordingVdmBackend(PlannerBackend):
    def __init__(self, *, payload=None, status=PipelineStatus.PLANNED) -> None:
        self.requests: list[PlannerBackendRequest] = []
        self.payload = payload or {
            "visible_changes": ["the cube moved"],
            "task_progress_evidence": ["the cube is closer to the target"],
            "completion_evidence": [],
            "uncertainties": [],
        }
        self.status = status

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        self.requests.append(request)
        return PlannerBackendResult(
            payload=self.payload,
            status=self.status,
            provider="fixture-provider",
            model="fixture-vdm",
            details={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )


def _png(path: Path, value: int) -> str:
    Image.new("RGB", (4, 4), color=(value, value, value)).save(path)
    return str(path)


def _observation(tmp_path: Path, index: int, *, include_main: bool = True) -> EnvObservation:
    main = _png(tmp_path / f"main-{index}.png", 10 + index)
    wrist = _png(tmp_path / f"wrist-{index}.png", 100 + index)
    depth = _png(tmp_path / f"depth-{index}.png", 200 + index)
    artifacts = [
        {
            "kind": "rgb",
            "frame_id": "wrist",
            "role": "wrist_primary",
            "path": wrist,
        },
        {
            "kind": "depth",
            "frame_id": "agentview",
            "role": "scene_primary",
            "path": depth,
        },
    ]
    cameras = [
        CameraFrame(
            frame_id="wrist",
            role="wrist_primary",
            rgb=[[[0, 0, 0]]],
            timestamp_s=float(index) + 0.5,
        )
    ]
    if include_main:
        artifacts.insert(
            0,
            {
                "kind": "rgb",
                "frame_id": "agentview",
                "role": "scene_primary",
                "path": main,
            },
        )
        # Put wrist first to verify selection never relies on cameras[0].
        cameras.append(
            CameraFrame(
                frame_id="agentview",
                role="scene_primary",
                rgb=[[[0, 0, 0]]],
                timestamp_s=float(index),
            )
        )
    return EnvObservation(
        task="move the cube into the bowl",
        cameras=cameras,
        robot=RobotState(),
        metadata={"step_idx": index, "image_artifacts": artifacts},
    )


def _record_trajectory(
    tmp_path: Path,
    *,
    count: int = 6,
) -> tuple[AgentMemory, RecordingVdmBackend, list[EnvObservation]]:
    memory = AgentMemory(store=JsonMemoryStore(tmp_path / "memory"))
    memory.start_session(task="move the cube into the bowl", session_id="episode")
    backend = RecordingVdmBackend()
    manager = VisualHistoryManager(
        config=VisualHistoryConfig(),
        backend=backend,
    )
    observations = []
    for index in range(count):
        observation = _observation(tmp_path, index)
        observations.append(observation)
        memory.add_observation(observation)
        manager.observe(observation, memory=memory)
    return memory, backend, observations


def test_main_view_vdm_and_bounded_camera_window(tmp_path: Path) -> None:
    memory, backend, observations = _record_trajectory(tmp_path)

    assert len(backend.requests) == 5
    first_observation_record = observation_history(memory)[0]
    persisted_artifacts = [
        (item["kind"], item["frame_id"])
        for item in first_observation_record["visual_artifacts"]
    ]
    assert persisted_artifacts == [
        ("rgb", "agentview"),
        ("rgb", "wrist"),
        ("depth", "agentview"),
    ]
    for index, request in enumerate(backend.requests, start=1):
        assert request.metadata["role"] == "visual_differencing"
        assert request.metadata["isolated_context"] is True
        assert request.tool_context["vision_image_paths"] == [
            str(tmp_path / f"main-{index - 1}.png"),
            str(tmp_path / f"main-{index}.png"),
        ]
        assert "wrist" not in " ".join(request.tool_context["vision_image_paths"])

    context = build_tool_context(
        observation=observations[-1],
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(visual_history=VisualHistoryConfig()),
    )
    assert context["vision_image_paths"] == [
        str(tmp_path / "main-0.png"),
        str(tmp_path / "main-3.png"),
        str(tmp_path / "main-4.png"),
        str(tmp_path / "main-5.png"),
        str(tmp_path / "wrist-5.png"),
    ]
    assert not any("depth" in path for path in context["vision_image_paths"])
    history = context["visual_history"]
    compressed_targets = [
        item["to_observation"]["observation_index"]
        for item in history["compressed_deltas"]
    ]
    assert compressed_targets == [
        1,
        2,
        3,
    ]
    assert history["coverage"]["missing_delta_observation_indices"] == []
    assert [item["role"] for item in context["vision_evidence"]] == [
        "historical_anchor",
        "historical_scene",
        "historical_scene",
        "current_scene",
        "current_scene",
    ]
    assert len(context["agent_context"]["current_observation"]["visual_evidence"]) == 2
    assert context["agent_context"]["visual_history"]["policy"]["vdm_camera_roles"] == [
        "agentview"
    ]


def test_identical_main_view_skips_vdm_but_preserves_delta_coverage(
    tmp_path: Path,
) -> None:
    memory = AgentMemory(store=JsonMemoryStore(tmp_path / "memory"))
    memory.start_session(task="inspect without moving", session_id="episode")
    backend = RecordingVdmBackend()
    manager = VisualHistoryManager(config=VisualHistoryConfig(), backend=backend)
    first = _observation(tmp_path, 0)
    second = _observation(tmp_path, 1)
    first_main = Path(first.metadata["image_artifacts"][0]["path"])
    second_main = Path(second.metadata["image_artifacts"][0]["path"])
    second_main.write_bytes(first_main.read_bytes())

    memory.add_observation(first)
    assert manager.observe(first, memory=memory) is None
    memory.add_observation(second)
    record = manager.observe(second, memory=memory)

    assert backend.requests == []
    assert record is not None
    assert record["status"] == "no_visible_change"
    assert record["derived_by"] == "host_identical_image_check"
    assert record["comparison"] == {
        "method": "sha256",
        "identical": True,
        "content_sha256": record["comparison"]["content_sha256"],
    }
    assert record["comparison"]["content_sha256"]
    assert record.get("usage") is None
    assert len(observation_history(memory)) == 2
    assert len(visual_delta_history(memory)) == 1


def test_initial_overlap_and_recent_window_are_deduplicated(tmp_path: Path) -> None:
    memory, _backend, observations = _record_trajectory(tmp_path, count=3)
    projection = build_visual_history_projection(
        observation=observations[-1],
        memory=memory,
        config=VisualHistoryConfig(),
        current_camera_artifacts=[],
    )
    assert projection["vision_image_paths"] == [
        str(tmp_path / "main-0.png"),
        str(tmp_path / "main-1.png"),
        str(tmp_path / "main-2.png"),
        str(tmp_path / "wrist-2.png"),
    ]
    assert projection["visual_history"]["compressed_deltas"] == []


def test_old_visual_deltas_are_rolled_up_into_a_bounded_model_projection(
    tmp_path: Path,
) -> None:
    memory, _backend, observations = _record_trajectory(tmp_path, count=14)
    projection = build_visual_history_projection(
        observation=observations[-1],
        memory=memory,
        config=VisualHistoryConfig(vdm_recent_delta_limit=4),
        current_camera_artifacts=[],
    )
    history = projection["visual_history"]

    assert len(history["compressed_deltas"]) == 4
    assert [
        item["to_observation"]["observation_index"]
        for item in history["compressed_deltas"]
    ] == [8, 9, 10, 11]
    summary = history["compressed_delta_summary"]
    assert summary["schema_version"] == "openeta.visual_delta_summary.v1"
    assert summary["compacted_delta_count"] == 7
    assert summary["through_observation_index"] == 7
    assert len(summary["visible_changes"]) <= 4
    assert "Full visual_delta records remain" in summary["durable_history_query"]
    assert len(visual_delta_history(memory)) == 13


def test_missing_main_view_is_explicit_and_never_substitutes_wrist(tmp_path: Path) -> None:
    memory = AgentMemory()
    memory.start_session(task="move the cube")
    backend = RecordingVdmBackend()
    manager = VisualHistoryManager(config=VisualHistoryConfig(), backend=backend)
    first = _observation(tmp_path, 0)
    second = _observation(tmp_path, 1, include_main=False)
    memory.add_observation(first)
    manager.observe(first, memory=memory)
    memory.add_observation(second)
    record = manager.observe(second, memory=memory)

    assert backend.requests == []
    assert record is not None
    assert record["status"] == "unavailable"
    assert record["failure"]["missing"] == ["current"]
    assert record["source_camera_roles"] == ["agentview"]


def test_invalid_vdm_output_degrades_without_raising(tmp_path: Path) -> None:
    memory = AgentMemory()
    memory.start_session(task="move the cube")
    backend = RecordingVdmBackend(payload={"visible_changes": "not-a-list"})
    manager = VisualHistoryManager(config=VisualHistoryConfig(), backend=backend)
    for index in range(2):
        observation = _observation(tmp_path, index)
        memory.add_observation(observation)
        record = manager.observe(observation, memory=memory)

    assert record is not None
    assert record["status"] == "failed"
    assert record["failure"]["code"] == "vdm_request_failed"
    assert len(observation_history(memory)) == 2
    assert len(visual_delta_history(memory)) == 1


def test_vdm_completion_claim_remains_derived_history_not_world_truth(tmp_path: Path) -> None:
    memory = AgentMemory()
    memory.start_session(task="move the cube")
    backend = RecordingVdmBackend(
        payload={
            "visible_changes": ["the cube moved"],
            "task_progress_evidence": [],
            "completion_evidence": ["the task appears complete"],
            "uncertainties": [],
        }
    )
    manager = VisualHistoryManager(config=VisualHistoryConfig(), backend=backend)
    observations = []
    for index in range(5):
        observation = _observation(tmp_path, index)
        observations.append(observation)
        memory.add_observation(observation)
        manager.observe(observation, memory=memory)

    context = build_tool_context(
        observation=observations[-1],
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(visual_history=VisualHistoryConfig()),
    )["agent_context"]

    assert "the task appears complete" in str(context["visual_history"])
    assert "the task appears complete" not in str(context["world_evidence"])
    assert context["world_evidence"].get("latest_environment_receipt") is None


def test_visual_delta_resume_reuses_durable_record(tmp_path: Path) -> None:
    memory, backend, observations = _record_trajectory(tmp_path, count=4)
    assert len(backend.requests) == 3

    resumed = AgentMemory(store=JsonMemoryStore(tmp_path / "memory"))
    resumed.resume_session("episode", max_events=2)
    resumed_backend = RecordingVdmBackend()
    resumed_manager = VisualHistoryManager(
        config=VisualHistoryConfig(),
        backend=resumed_backend,
    )
    record = resumed_manager.observe(observations[-1], memory=resumed)

    assert record is not None
    assert record["delta_id"] == "visual_delta:2:3:agentview"
    assert resumed_backend.requests == []
    assert len(observation_history(resumed)) == 4
    assert len(visual_delta_history(resumed)) == 3

    next_observation = _observation(tmp_path, 4)
    resumed.add_observation(next_observation)
    next_record = resumed_manager.observe(next_observation, memory=resumed)
    assert next_record is not None
    assert next_record["delta_id"] == "visual_delta:3:4:agentview"
    assert resumed_backend.requests[0].tool_context["from_observation"][
        "observation_index"
    ] == 3


def test_visual_history_config_has_deterministic_environment_defaults() -> None:
    config = VisualHistoryConfig.from_env({})
    assert config.enabled is True
    assert config.main_camera_role == "agentview"
    assert config.recent_main_turns == 3
    assert config.vdm_recent_delta_limit is None
    assert config.planner_raw_image_capacity == 5

    disabled = VisualHistoryConfig.from_env(
        {
            "OPENETA_VISUAL_HISTORY_ENABLED": "false",
            "OPENETA_VDM_CAMERA_ROLE": "scene_primary",
            "OPENETA_VISUAL_RECENT_MAIN_TURNS": "4",
            "OPENETA_VISUAL_INCLUDE_CURRENT_WRIST": "0",
            "OPENETA_VISUAL_VDM_RECENT_DELTA_LIMIT": "4",
        }
    )
    assert disabled.enabled is False
    assert disabled.main_camera_role == "scene_primary"
    assert disabled.recent_main_turns == 4
    assert disabled.include_current_wrist is False
    assert disabled.vdm_recent_delta_limit == 4


def test_projection_ablation_exposes_raw_and_delta_tradeoffs(tmp_path: Path) -> None:
    memory, _backend, observations = _record_trajectory(tmp_path)
    report = run_visual_history_projection_ablation(
        observation=observations[-1],
        memory=memory,
        current_camera_artifacts=[],
    )
    by_name = {item["variant"]: item for item in report["results"]}

    assert by_name["A_current_main_only"]["raw_image_count"] == 1
    assert by_name["A_current_main_only"]["compressed_delta_count"] == 0
    assert by_name["B_bounded_raw_no_vdm"]["raw_image_count"] == 5
    assert by_name["B_bounded_raw_no_vdm"]["compressed_delta_count"] == 0
    assert by_name["C_bounded_raw_main_vdm"]["raw_image_count"] == 5
    assert by_name["C_bounded_raw_main_vdm"]["compressed_delta_count"] == 3
    assert by_name["D_current_main_full_vdm"]["raw_image_count"] == 1
    assert by_name["D_current_main_full_vdm"]["compressed_delta_count"] == 5

    session_report = build_session_visual_history_ablation(
        memory_root=tmp_path / "memory",
        session_id="episode",
    )
    assert session_report["observation_count"] == 6
    assert session_report["results"] == report["results"]


def test_visual_delta_tokens_are_charged_separately_to_episode_budget() -> None:
    action = EnvAction(
        action_type="response",
        command={
            "metadata": {
                "planner_metadata": {
                    "backend_usage": {"total_tokens": 30},
                    "backend_usage_sources": {"provider": 1},
                    "visual_delta_usage": {
                        "usage": {"total_tokens": 15, "usage_source": "provider"}
                    },
                }
            }
        },
    )

    total, sources = action_token_usage(action)

    assert total == 45
    assert sources == {"provider": 1, "visual_delta:provider": 1}


def test_runtime_generates_delta_before_second_planner_call(tmp_path: Path) -> None:
    vdm = RecordingVdmBackend()
    manager = VisualHistoryManager(config=VisualHistoryConfig(), backend=vdm)
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {"message": "first"},
                    "reasoning": "inspect",
                },
                {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {"message": "second"},
                    "reasoning": "inspect",
                },
            ]
        ),
        context_config=PlannerContextConfig(visual_history=VisualHistoryConfig()),
    )
    runtime = OpenEtaAgentRuntime(
        planner=planner,
        memory=AgentMemory(store=JsonMemoryStore(tmp_path / "runtime-memory")),
        visual_history=manager,
        rollout_enabled=False,
    )
    runtime.start_session(task="move the cube", session_id="runtime")

    runtime.act(_observation(tmp_path, 0))
    second = runtime.act(_observation(tmp_path, 1))

    assert len(vdm.requests) == 1
    usage = second.command["metadata"]["planner_metadata"]["visual_delta_usage"]
    assert usage["delta_id"] == "visual_delta:0:1:agentview"
    assert usage["status"] == "available"
    assert usage["usage"]["total_tokens"] == 15
    assert visual_delta_history(runtime.memory)[0]["duration_s"] >= 0.0


def test_vdm_call_is_recorded_in_lossless_rollout_order(tmp_path: Path) -> None:
    memory_root = tmp_path / "rollout-memory"
    manager = VisualHistoryManager(
        config=VisualHistoryConfig(),
        backend=RecordingVdmBackend(),
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {"kind": "response", "name": "talk", "parameters": {"message": "1"}},
                {"kind": "response", "name": "talk", "parameters": {"message": "2"}},
            ]
        ),
        context_config=PlannerContextConfig(visual_history=VisualHistoryConfig()),
    )
    runtime = OpenEtaAgentRuntime(
        planner=planner,
        memory=AgentMemory(store=JsonMemoryStore(memory_root)),
        visual_history=manager,
    )
    runtime.start_session(task="move the cube", session_id="rollout")

    runtime.act(_observation(tmp_path, 0))
    runtime.act(_observation(tmp_path, 1))

    model_calls = memory_root / "sessions" / "rollout" / "rollout" / "model_calls.jsonl"
    rows = [json.loads(line) for line in model_calls.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert rows[1]["semantic_request"]["metadata"]["role"] == "visual_differencing"
    assert rows[1]["semantic_request"]["tool_context"]["vision_image_paths"] == [
        str(tmp_path / "main-0.png"),
        str(tmp_path / "main-1.png"),
    ]
    assert rows[1]["validation"] == {"accepted": True, "errors": []}
