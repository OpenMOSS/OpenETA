from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agent.backends.planner import CallablePlannerBackend
from agent.tools.grasp_pose_advisor import (
    GRASP_POSE_ADVISOR_SYSTEM_PROMPT,
    GRASP_SELECTION_ADVICE_SCHEMA,
    GRASP_SELECTION_BUNDLE_SCHEMA,
    BackendGraspPoseAdvisor,
    _project_candidate,
    _target_support_3d,
    build_grasp_selection_bundle,
)


def _write_images(root: Path) -> tuple[Path, Path]:
    rgb = root / "rgb.png"
    mask = root / "mask.png"
    Image.new("RGB", (320, 240), (80, 95, 110)).save(rgb)
    mask_image = Image.new("L", (320, 240), 0)
    for x in range(125, 205):
        for y in range(75, 185):
            mask_image.putpixel((x, y), 255)
    mask_image.save(mask)
    return rgb, mask


def _candidate(candidate_id: str, rank: int, *, x: float, score: float) -> dict:
    return {
        "id": candidate_id,
        "rank": rank,
        "backend_candidate_id": f"native-{rank}",
        "score": score,
        "translation_xyz": [x, 0.0, 0.8],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "gripper_tip_position_xyz": [x + 0.04, 0.0, 0.8],
        "depth": 0.04,
        "width": 0.06,
        "height": 0.03,
    }


def _details(root: Path, *, backend: str) -> dict:
    rgb, mask = _write_images(root)
    candidates = [
        _candidate(f"gpe-{backend}-000", 0, x=-0.02, score=0.9),
        _candidate(f"gpe-{backend}-001", 1, x=0.03, score=0.7),
    ]
    reference = "gripper_tip_position_xyz"
    for candidate in candidates:
        candidate["source_backend"] = backend
        candidate["execution_reference_point"] = reference
        candidate["execution_contact_center_xyz"] = list(candidate[reference])
    return {
        "result_id": f"gpe-{backend}",
        "selected_backend": backend,
        "source_rgb": str(rgb),
        "object_mask": str(mask),
        "camera_frame_id": "agentview",
        "scene_epoch": 3,
        "source": {
            "intrinsics": {
                "fx": 240.0,
                "fy": 240.0,
                "cx": 160.0,
                "cy": 120.0,
                "scale": 1000.0,
            }
        },
        "grasp_candidates": candidates,
    }


@pytest.mark.parametrize("backend", ["anygrasp", "graspgenx"])
def test_renderer_builds_backend_neutral_selection_bundle(
    tmp_path: Path,
    backend: str,
) -> None:
    bundle, artifacts = build_grasp_selection_bundle(
        _details(tmp_path, backend=backend),
        output_root=tmp_path / "selection",
    )

    assert bundle["schema_version"] == GRASP_SELECTION_BUNDLE_SCHEMA
    assert bundle["source_backend"] == backend
    assert [item["candidate_id"] for item in bundle["candidates"]] == [
        f"gpe-{backend}-000",
        f"gpe-{backend}-001",
    ]
    assert Path(bundle["overview_ref"]).is_file()
    assert len(bundle["contact_sheet_refs"]) == 1
    assert Path(bundle["contact_sheet_refs"][0]).is_file()
    assert Path(bundle["bundle_ref"]).is_file()
    expected_field = "gripper_tip_position_xyz"
    assert bundle["candidates"][0]["execution_reference_point"] == expected_field
    assert bundle["candidates"][0]["executed_contact_center_xyz"] == bundle[
        "candidates"
    ][0][expected_field]
    assert "host-executed grasp/contact center" in bundle["visual_semantics"][
        "approach"
    ]
    assert [item["type"] for item in artifacts] == [
        "grasp_selection_overview",
        "grasp_selection_contact_sheet",
        "grasp_selection_bundle",
    ]


def test_renderer_centers_anygrasp_jaws_on_advertised_tip() -> None:
    candidate = _candidate("gpe-anygrasp-000", 0, x=0.0, score=0.9)
    candidate["translation_xyz"] = [0.0, 0.0, 1.0]
    candidate["gripper_tip_position_xyz"] = [0.04, 0.0, 1.0]
    candidate["execution_reference_point"] = "gripper_tip_position_xyz"
    candidate["execution_contact_center_xyz"] = [0.04, 0.0, 1.0]

    projected = _project_candidate(
        candidate,
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0},
        image_size=(320, 240),
    )

    approach_start, contact_center, jaw_a, jaw_b, _height_a, _height_b = (
        projected["pixels"]
    )
    assert approach_start == pytest.approx([0.0, 0.0])
    assert contact_center == pytest.approx([4.0, 0.0])
    assert jaw_a == pytest.approx([4.0, -3.0])
    assert jaw_b == pytest.approx([4.0, 3.0])
    assert projected["executed_contact_center_xyz"] == [0.04, 0.0, 1.0]
    assert projected["gripper_tip_position_xyz"] == [0.04, 0.0, 1.0]


def test_renderer_centers_graspgenx_jaws_on_advertised_tip() -> None:
    candidate = _candidate("gpe-graspgenx-000", 0, x=0.0, score=0.9)
    candidate["execution_reference_point"] = "gripper_tip_position_xyz"
    candidate["execution_contact_center_xyz"] = [0.04, 0.0, 1.0]

    projected = _project_candidate(
        candidate,
        intrinsics={"fx": 100.0, "fy": 100.0, "cx": 0.0, "cy": 0.0},
        image_size=(320, 240),
    )

    approach_start, contact_center, jaw_a, jaw_b, _height_a, _height_b = (
        projected["pixels"]
    )
    assert approach_start == pytest.approx([0.0, 0.0])
    assert contact_center == pytest.approx([4.0, 0.0])
    assert jaw_a == pytest.approx([4.0, -3.0])
    assert jaw_b == pytest.approx([4.0, 3.0])
    assert projected["executed_contact_center_xyz"] == [0.04, 0.0, 1.0]


def test_target_support_3d_exposes_overwide_off_core_contact() -> None:
    points = [
        [0.02, y, z]
        for y in (-0.052, -0.045, 0.045, 0.052)
        for z in (-0.018, -0.012, -0.008)
        for _ in range(10)
    ]

    evidence = _target_support_3d(
        points,
        contact_center=[0.0, 0.0, 0.025],
        rotation=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        gripper_width_m=0.08,
    )

    assert evidence is not None
    assert evidence["closing_span_m"] > 0.08
    assert evidence["closing_span_to_gripper_width"] > 1.05
    assert evidence["binormal_center_offset_half_span"] > 1.0
    assert evidence["approach_surface_interval_m"] == pytest.approx([0.02, 0.02])
    assert evidence["risk_indicators"] == [
        "visible_closing_span_exceeds_gripper_width",
        "executed_center_outside_visible_binormal_core",
        "executed_center_before_visible_target_surface",
    ]


def test_target_support_3d_exposes_contact_beyond_visible_target() -> None:
    points = [
        [-0.02, y, z]
        for y in (-0.02, -0.01, 0.01, 0.02)
        for z in (-0.015, -0.005, 0.005, 0.015)
        for _ in range(10)
    ]

    evidence = _target_support_3d(
        points,
        contact_center=[0.0, 0.0, 0.0],
        rotation=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        gripper_width_m=0.08,
    )

    assert evidence is not None
    assert "executed_center_beyond_visible_target_surface" in evidence[
        "risk_indicators"
    ]


def test_backend_advisor_is_isolated_and_returns_audited_recommendation(
    tmp_path: Path,
) -> None:
    bundle, _artifacts = build_grasp_selection_bundle(
        _details(tmp_path, backend="anygrasp"),
        output_root=tmp_path / "selection",
    )
    requests = []

    def decide(request):
        requests.append(request)
        return {
            "decision": "recommend",
            "recommended_candidate_id": "gpe-anygrasp-001",
            "alternatives": ["gpe-anygrasp-000"],
            "confidence": 0.82,
            "reasons": ["The jaw line crosses the object body with better clearance."],
            "rejected": {"gpe-anygrasp-000": "The contact is close to the left edge."},
            "uncertainties": ["The far finger is partially occluded."],
        }

    advice = BackendGraspPoseAdvisor(
        CallablePlannerBackend(decide, provider="test-provider", model="test-vlm")
    ).advise(bundle, task="pick up the milk carton")

    assert advice["schema_version"] == GRASP_SELECTION_ADVICE_SCHEMA
    assert advice["status"] == "completed"
    assert advice["recommended_candidate_id"] == "gpe-anygrasp-001"
    assert advice["provider"] == "test-provider"
    assert advice["model"] == "test-vlm"
    assert requests[0].metadata == {"isolated_context": True}
    assert requests[0].tool_context["role"] == "read_only_grasp_pose_advisor"
    assert requests[0].tool_context["vision_image_paths"] == [
        bundle["overview_ref"],
        *bundle["contact_sheet_refs"],
    ]


def test_advisor_prompt_prioritizes_transport_stability_over_convenient_approach(
) -> None:
    prompt = GRASP_POSE_ADVISOR_SYSTEM_PROMPT.lower()

    assert "primary objective" in prompt
    assert "without the object slipping or falling" in prompt
    assert "do not trade away grasp stability" in prompt
    assert "broad middle body" in prompt
    assert "cap, neck, top rim, or shoulder" in prompt
    assert "do not recommend the least-bad candidate" in prompt
    assert "you must abstain" in prompt
    assert "deeper insertion value alone" in prompt


def test_single_candidate_is_still_visually_assessed(tmp_path: Path) -> None:
    bundle, _artifacts = build_grasp_selection_bundle(
        _details(tmp_path, backend="anygrasp"),
        output_root=tmp_path / "selection",
    )
    bundle["candidates"] = bundle["candidates"][:1]
    requests = []

    def decide(request):
        requests.append(request)
        return {
            "decision": "abstain",
            "recommended_candidate_id": "",
            "alternatives": [],
            "confidence": 0.4,
            "reasons": ["the only contact is on the top rim"],
            "rejected": {bundle["candidates"][0]["candidate_id"]: "unstable rim"},
            "uncertainties": [],
        }

    advice = BackendGraspPoseAdvisor(CallablePlannerBackend(decide)).advise(
        bundle,
        task="pick the carton",
    )

    assert len(requests) == 1
    assert requests[0].tool_context["candidate_count"] == 1
    assert advice["status"] == "completed"
    assert advice["decision"] == "abstain"


def test_backend_advisor_rejects_hidden_action_output(tmp_path: Path) -> None:
    bundle, _artifacts = build_grasp_selection_bundle(
        _details(tmp_path, backend="graspgenx"),
        output_root=tmp_path / "selection",
    )
    advisor = BackendGraspPoseAdvisor(
        CallablePlannerBackend(
            lambda _request: {
                "decision": "recommend",
                "recommended_candidate_id": "gpe-graspgenx-000",
                "alternatives": [],
                "confidence": 0.9,
                "reasons": ["looks valid"],
                "rejected": {},
                "uncertainties": [],
                "tool_call": "compile_grasp_seed",
            }
        )
    )

    with pytest.raises(ValueError, match="forbidden action fields"):
        advisor.advise(bundle, task="pick")


def test_backend_advisor_normalizes_aliases_and_ignores_unknown_optional_refs(
    tmp_path: Path,
) -> None:
    bundle, _artifacts = build_grasp_selection_bundle(
        _details(tmp_path, backend="anygrasp"),
        output_root=tmp_path / "selection",
    )
    advisor = BackendGraspPoseAdvisor(
        CallablePlannerBackend(
            lambda _request: {
                "decision": "recommend",
                "recommended_candidate_id": "#2",
                "alternatives": ["native-0", "unknown-alternative"],
                "confidence": 0.7,
                "reasons": ["second candidate has a clearer approach"],
                "rejected": {
                    "#1": "edge-biased",
                    "all others": "non-candidate summary",
                },
                "uncertainties": [],
            }
        )
    )

    advice = advisor.advise(bundle, task="pick")

    assert advice["recommended_candidate_id"] == "gpe-anygrasp-001"
    assert advice["alternatives"] == ["gpe-anygrasp-000"]
    assert advice["rejected"] == {"gpe-anygrasp-000": "edge-biased"}
    assert len(advice["validation_warnings"]) == 2
