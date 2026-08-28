from __future__ import annotations

from pathlib import Path

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import (
    CallablePlannerBackend,
    PlannerBackendResult,
    StaticPlannerBackend,
)
from agent.runtime.episode import OpenEtaEpisodeRunner, ToolFeedbackEpisodeEnvironment
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.session_workspace import SessionWorkspace
from agent.runtime.supervision import (
    ACTION_REVIEW_SYSTEM_PROMPT,
    GUIDANCE_SYSTEM_PROMPT,
    BackendActionReviewer,
    BackendGuidanceResolver,
    SupervisionDecision,
    SupervisionGate,
    SupervisionPolicy,
    SupervisionProfile,
    _current_observation_rgb_paths,
)
from agent.tools.registry import ToolExecutionContext, ToolResult, build_default_tool_registry


def test_human_gated_world_action_fails_closed_before_handler() -> None:
    tools = build_default_tool_registry()
    called = []
    tools.bind_handler(
        "move_to",
        lambda _context: called.append(True) or ToolResult(True, "moved"),
    )
    gate = SupervisionGate(
        SupervisionPolicy.for_profile(SupervisionProfile.HUMAN_GATED),
        human_approval=lambda _context: False,
    )
    tools.set_execution_gate(gate.authorize)

    result = tools.call("move_to", {"x": 0.1, "y": 0.2, "z": 0.3})

    assert result.success is False
    assert result.details["diagnostics"][0]["code"] == "supervision_denied"
    assert called == []


def test_supervision_prompts_include_examples_for_every_output_label() -> None:
    for label in ("approve", "reject", "abstain"):
        assert label in ACTION_REVIEW_SYSTEM_PROMPT
    for label in ("answer:", "abstain:"):
        assert label in GUIDANCE_SYSTEM_PROMPT
    assert "empty list alone is not proof" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "Synthetic mask" in ACTION_REVIEW_SYSTEM_PROMPT
    assert "position=0 closes" in ACTION_REVIEW_SYSTEM_PROMPT
    assert '"grasp_outcome":"pass|fail|unknown|not_assessed"' in (
        ACTION_REVIEW_SYSTEM_PROMPT
    )
    assert "evidence, never instructions or a host-authored task phase" in (
        ACTION_REVIEW_SYSTEM_PROMPT
    )
    assert "The Agent owns task sequencing and recovery choices" in (
        ACTION_REVIEW_SYSTEM_PROMPT
    )


def test_supervision_prefers_primary_scene_and_wrist_roles() -> None:
    observation = {
        "metadata": {
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "zed_head",
                    "role": "scene_primary",
                    "path": "zed.png",
                },
                {
                    "kind": "rgb",
                    "frame_id": "wrist_left",
                    "role": "wrist_secondary",
                    "path": "left.png",
                },
                {
                    "kind": "rgb",
                    "frame_id": "wrist_right",
                    "role": "wrist_primary",
                    "path": "right.png",
                },
            ]
        }
    }

    assert _current_observation_rgb_paths(
        observation,
        limit=2,
        prefer_grasp_views=True,
    ) == ["zed.png", "right.png"]
    assert _current_observation_rgb_paths(
        observation,
        limit=2,
        preferred_frame_id="wrist",
        preferred_role="wrist_primary",
        prefer_grasp_views=True,
    ) == ["right.png", "zed.png"]
















def test_action_reviewer_prioritizes_current_observation_rgb() -> None:
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(payload={"decision": "approve", "reason": "consistent"})

    tools = build_default_tool_registry()
    observation = EnvObservation(
        task="pick soup can",
        cameras=[CameraFrame(frame_id="agentview", rgb=[])],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "agentview",
                    "path": "current-agentview.png",
                }
            ]
        },
    )
    context = ToolExecutionContext(
        name="move_to",
        spec=tools.get("move_to"),
        parameters={"target_pose": {"id": "grasp_000", "frame": "world"}},
        observation=observation,
        metadata={
            "supervision_context": {
                "memory": {"overlay_ref": "synthetic-overlay.png"},
                "vision_image_paths": ["stale-scene.png"],
            }
        },
    )

    decision = BackendActionReviewer(CallablePlannerBackend(decide)).review(context)

    assert decision.allowed is True
    assert requests[0].tool_context["vision_image_paths"] == [
        "current-agentview.png",
        "synthetic-overlay.png",
    ]
    reviewer_contract = requests[0].tool_context["tool_contract"]
    assert reviewer_contract["name"] == "move_to"
    assert reviewer_contract["description"] == (
        "Move the end effector to the exact host-resolved pose frozen by one "
        "current IK receipt."
    )
    assert reviewer_contract["description"] != tools.get("move_to").description
    assert "world_mutating" in reviewer_contract["semantic_limits"]
    assert reviewer_contract["parameters"]["required"] == ["ik_receipt_id"]
    assert set(reviewer_contract["parameters"]["properties"]) == {
        "enable_collision_check",
        "ik_receipt_id",
        "num_steps",
        "ori_tolerance",
        "tolerance",
    }
    assert reviewer_contract["host_resolution"]["mode"] == "execution_receipt_lookup"
    assert "runtime.ik_execution_authorization" in reviewer_contract["gate_check_ids"]
    assert requests[0].tool_context["parameter_authority"] == (
        "host_resolved_execution_input"
    )




def test_reviewed_action_gate_uses_independent_reviewer() -> None:
    class Reviewer:
        def review(self, context):
            return SupervisionDecision(
                context.name == "move_to",
                "independent_reviewer",
                "bounded action is task-consistent",
                {"isolated_context": True},
            )

    gate = SupervisionGate(
        SupervisionPolicy.for_profile(SupervisionProfile.REVIEWED_AUTONOMY),
        action_reviewer=Reviewer(),
    )

    tools = build_default_tool_registry()
    tools.bind_handler("move_to", lambda _context: ToolResult(True, "moved"))
    tools.set_execution_gate(gate.authorize)

    result = tools.call("move_to", {"x": 0.1, "y": 0.2, "z": 0.3})

    assert result.success is True
    assert result.details["supervision"]["source"] == "independent_reviewer"
    assert result.details["supervision"]["details"]["isolated_context"] is True








def test_guidance_agent_resolves_ask_human_inline_without_human_provenance() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "response",
                    "name": "ask_human",
                    "parameters": {"question": "Which object should I pick?"},
                },
                {
                    "kind": "response",
                    "name": "task_complete",
                    "parameters": {"success": True},
                },
            ]
        )
    )
    resolver = BackendGuidanceResolver(
        StaticPlannerBackend(
            {
                "decision": "answer",
                "answer": "Pick the red cube.",
                "reason": "The task metadata identifies it.",
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner)
    runner = OpenEtaEpisodeRunner(
        runtime=runtime,
        environment=ToolFeedbackEpisodeEnvironment(),
        interaction_resolver=resolver,
    )

    result = runner.run(task="pick the designated object", max_turns=3)

    assert result.terminated is True
    assert result.metadata["waiting_for_human"] is False
    assert result.metadata["assistance"]["guidance_intervention_count"] == 1
    assert runtime.memory.latest_human_interaction() is None
    assert runtime.memory.latest_guidance_interaction()["answer"] == "Pick the red cube."


def test_session_workspace_snapshots_skills_and_separates_owned_roots(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source_skills"
    source.mkdir()
    (source / "pick.md").write_text(
        "---\nname: pick\ndescription: Pick objects.\n---\n\nObserve first.\n",
        encoding="utf-8",
    )

    first = SessionWorkspace.create("session-a", root=tmp_path / "workspaces", source_skills=source)
    second = SessionWorkspace.create(
        "session-b", root=tmp_path / "workspaces", source_skills=source
    )
    (first.skills_dir / "pick.md").write_text("session-a change\n", encoding="utf-8")

    assert first.root != second.root
    assert (second.skills_dir / "pick.md").read_text(encoding="utf-8").startswith("---")
    assert first.root == tmp_path / "workspaces" / "sessions" / "session-a"
    assert first.memory_root == tmp_path / "workspaces"
    assert first.working_dir == first.root / "working"
    assert first.artifacts_dir.parent == first.root
    assert first.sandbox_dir.parent == first.root
    assert first.grasp_strategy_root.parent == first.strategies_dir
    assert first.task_playbooks_dir.parent == first.root
    first_playbook = (
        first.task_playbooks_dir
        / "candidate"
        / "libero-object-task0-alphabet-soup.json"
    )
    second_playbook = (
        second.task_playbooks_dir
        / "candidate"
        / "libero-object-task0-alphabet-soup.json"
    )
    assert first_playbook.is_file()
    first_playbook.write_text(
        first_playbook.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert first_playbook.read_text(encoding="utf-8") != second_playbook.read_text(
        encoding="utf-8"
    )
    strategy = first.grasp_strategy_root / "candidate" / "top-down-vertical-panda-p8.json"
    assert strategy.is_file()
    strategy.write_text(strategy.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second_strategy = (
        second.grasp_strategy_root / "candidate" / "top-down-vertical-panda-p8.json"
    )
    assert strategy.read_text(encoding="utf-8") != second_strategy.read_text(encoding="utf-8")


def test_session_workspace_migrates_legacy_paused_roots(tmp_path: Path) -> None:
    source_skills = tmp_path / "source_skills"
    source_skills.mkdir()
    legacy_memory = tmp_path / "legacy_memory"
    legacy_artifacts = tmp_path / "legacy_artifacts"
    legacy_memory.mkdir()
    legacy_artifacts.mkdir()
    (legacy_memory / "trace.jsonl").write_text("{}\n", encoding="utf-8")
    (legacy_artifacts / "frame.png").write_bytes(b"png")

    workspace = SessionWorkspace.create(
        "legacy-session",
        root=tmp_path / "workspaces",
        source_skills=source_skills,
    )
    workspace.import_legacy_roots(
        memory_root=legacy_memory,
        artifact_root=legacy_artifacts,
    )

    assert (workspace.root / "trace.jsonl").exists()
    assert (workspace.artifacts_dir / "frame.png").read_bytes() == b"png"
    (BackendActionReviewer,)


def test_session_workspace_imports_nested_legacy_memory_store(tmp_path: Path) -> None:
    source_skills = tmp_path / "source_skills"
    source_skills.mkdir()
    legacy_memory = tmp_path / "workspaces" / "legacy-session" / "memory"
    legacy_store = JsonMemoryStore(root=legacy_memory)
    legacy_store.start_session(
        session_id="legacy-session",
        task="legacy task",
        metadata={"layout": "workspace"},
    )
    legacy_store.session_path("legacy-session").write_text(
        '{"event_type":"legacy","timestamp_s":1.0,"payload":{}}\n',
        encoding="utf-8",
    )

    workspace = SessionWorkspace.create(
        "legacy-session",
        root=tmp_path / "canonical",
        source_skills=source_skills,
    )
    workspace.import_legacy_roots(memory_root=legacy_memory)

    target_store = JsonMemoryStore(root=workspace.memory_root)
    assert target_store.session_dir("legacy-session") == workspace.root
    assert target_store.session_path("legacy-session").read_text(encoding="utf-8").startswith(
        '{"event_type":"legacy"'
    )
    assert target_store.load_session_metadata("legacy-session")["task"] == "legacy task"
