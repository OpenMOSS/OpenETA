from __future__ import annotations

from agent.backends.planner import StaticPlannerBackend
from agent.cli.visual_state_eval import _provider_descriptor
from agent.evals.visual_state import (
    VISUAL_STATE_PROBE_SCHEMA_VERSION,
    VisualStateEvalCase,
    build_visual_state_probe_request,
    load_visual_state_eval_cases,
    run_visual_state_evaluation,
)


def _cases() -> tuple[VisualStateEvalCase, ...]:
    shared = {
        "group_id": "grasp-counterfactual",
        "task": "pick up the mug",
        "state_label_options": ("object attached", "object not attached"),
        "image_paths": ("tests/fixtures/current.png",),
        "allowed_next_actions": ("observe", "gripper_control"),
        "observation_metadata": {
            "step_idx": 7,
            "timestamp_s": 12.5,
            "frame_ids": ["agentview"],
            "camera_roles": ["scene_primary"],
        },
        "last_action": {"name": "gripper_control", "parameters": {"position": 0}},
        "last_result": {"success": True},
        "world_facts": (
            {
                "claim": "the gripper was commanded closed",
                "source": "tool_result",
                "freshness": "previous_transition",
            },
        ),
    }
    return (
        VisualStateEvalCase(
            case_id="mug-attached",
            variant="current_attached",
            expected_state_label="object attached",
            **shared,
        ),
        VisualStateEvalCase(
            case_id="mug-left-behind",
            variant="counterfactual_left_behind",
            expected_state_label="object not attached",
            **shared,
        ),
    )


def test_probe_request_is_minimal_and_labels_current_visual_evidence() -> None:
    request = build_visual_state_probe_request(_cases()[0])

    assert request.metadata["isolated_context"] is True
    assert request.tool_context["schema_version"] == VISUAL_STATE_PROBE_SCHEMA_VERSION
    assert request.tool_context["role"] == "visual_state_probe"
    assert request.tool_context["allowed_next_actions"] == ["observe", "gripper_control"]
    assert "next_action.name from\nallowed_next_actions exactly" in request.system_prompt
    assert "selection_obligation" not in request.tool_context
    assert "grasp_execution" not in request.tool_context
    evidence = request.tool_context["current_observation"]["evidence"]
    assert evidence == [
        {
            "evidence_id": "current_observation:agentview:0",
            "role": "current_scene",
            "camera_role": "scene_primary",
            "frame_id": "agentview",
            "path": "tests/fixtures/current.png",
            "freshness": "current",
            "observation_step": 7,
            "timestamp_s": 12.5,
        }
    ]
    assert request.tool_context["vision_evidence"] == [
        {"role": "current_scene", "path": "tests/fixtures/current.png"}
    ]


def test_evaluation_scores_visual_grounding_and_counterfactual_flip() -> None:
    backend = StaticPlannerBackend(
        [
            {
                "state_label": "object attached",
                "observed_facts": [
                    {
                        "claim": "the mug moved with the gripper",
                        "evidence_ids": ["current_observation:agentview:0"],
                    }
                ],
                "uncertainties": [],
                "next_action": {"name": "observe", "reason": "confirm placement target"},
            },
            {
                "state_label": "object not attached",
                "observed_facts": [
                    {
                        "claim": "the mug remains at the source",
                        "evidence_ids": ["current_observation:agentview:0"],
                    }
                ],
                "uncertainties": [],
                "next_action": {"name": "gripper_control", "reason": "recover safely"},
            },
        ]
    )

    report = run_visual_state_evaluation(backend, cases=_cases())

    assert report["metrics"]["passed"] == 2
    assert report["metrics"]["mean_evidence_citation_rate"] == 1.0
    assert report["metrics"]["expected_flip_group_count"] == 1
    assert report["metrics"]["correct_counterfactual_flip_count"] == 1
    assert report["metrics"]["counterfactual_flip_rate"] == 1.0


def test_evaluation_rejects_uncited_visual_claims_and_invariant_predictions() -> None:
    backend = StaticPlannerBackend(
        [
            {
                "state_label": "object attached",
                "observed_facts": [{"claim": "the mug is attached", "evidence_ids": []}],
                "next_action": {"name": "observe"},
            },
            {
                "state_label": "object attached",
                "observed_facts": [{"claim": "the mug is attached", "evidence_ids": []}],
                "next_action": {"name": "observe"},
            },
        ]
    )

    report = run_visual_state_evaluation(backend, cases=_cases())

    assert report["metrics"]["passed"] == 0
    assert report["metrics"]["mean_evidence_citation_rate"] == 0.0
    assert report["metrics"]["correct_counterfactual_flip_count"] == 0
    assert report["metrics"]["counterfactual_flip_rate"] == 0.0


def test_manifest_loader_validates_unique_cases() -> None:
    manifest = {"cases": [_cases()[0].to_dict(), _cases()[1].to_dict()]}

    loaded = load_visual_state_eval_cases(manifest)

    assert loaded == _cases()


def test_provider_descriptor_removes_all_api_key_fragments() -> None:
    descriptor = _provider_descriptor(
        {
            "provider": "gateway",
            "api_key": "sk-...abcd",
            "fallback": {"provider": "fallback", "api_key": "sk-...efgh"},
        }
    )

    assert descriptor["api_key"] == "<redacted>"
    assert descriptor["fallback"]["api_key"] == "<redacted>"
