from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from agent.tools.grasp_geometry import (
    DEFAULT_GRASP_PROFILE,
    assess_target_mask_quality,
    build_compile_grasp_seed_handler,
    build_wrist_alignment_handler,
    camera_optical_forward_world,
    GraspGeometryError,
    compile_grasp_seed,
    compute_wrist_alignment,
    grasp_refinement_hover_pose,
    propose_wrist_viewpoints,
    rebase_camera_direction,
    rebase_camera_grasp_candidate,
    world_up_direction_camera,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


def _profile() -> dict:
    return json.loads(DEFAULT_GRASP_PROFILE.read_text(encoding="utf-8"))


def _candidate() -> dict:
    return {
        "id": "grasp_000",
        "frame": "camera",
        "camera_frame": "opencv",
        "width": 0.06,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }


def _compile_parameters() -> dict:
    return {
        "camera_pose": _candidate(),
        "camera_extrinsics": {
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "camera_frame_id": "agentview",
        "target_class": "upright_can",
        "scene_epoch": 0,
    }


def test_rebase_camera_grasp_candidate_preserves_world_geometry() -> None:
    candidate = {
        **_candidate(),
        "gripper_tip_position_xyz": [2.1, 0.0, 0.0],
        "translation_xyz": [2.0, 0.0, 0.0],
        "execution_contact_center_xyz": [2.1, 0.0, 0.0],
        "execution_reference_point": "gripper_tip_position_xyz",
        "transform_matrix": [
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "model_native_grasp_pose": {
            "frame": "camera",
            "transform_matrix": [
                [1.0, 0.0, 0.0, 2.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        },
    }
    source_extrinsics = {
        "camera_frame": "opencv",
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    target_extrinsics = {
        "camera_frame": "opencv",
        "pos": [1.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }

    rebased = rebase_camera_grasp_candidate(
        candidate,
        source_camera_extrinsics=source_extrinsics,
        target_camera_extrinsics=target_extrinsics,
    )

    assert rebased["translation_xyz"] == pytest.approx([1.0, 0.0, 0.0])
    assert rebased["gripper_tip_position_xyz"] == pytest.approx([1.1, 0.0, 0.0])
    assert rebased["execution_contact_center_xyz"] == pytest.approx([1.1, 0.0, 0.0])
    assert rebased["rotation_matrix"] == candidate["rotation_matrix"]
    assert rebased["transform_matrix"] == [
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert "model_native_grasp_pose" not in rebased
    assert candidate["translation_xyz"] == [2.0, 0.0, 0.0]


def test_rebase_camera_direction_preserves_world_direction() -> None:
    source_extrinsics = {
        "camera_frame": "opencv",
        "pos": [0.0, 0.0, 0.0],
        "mat": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    target_extrinsics = {
        "camera_frame": "opencv",
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }

    rebased = rebase_camera_direction(
        [1.0, 0.0, 0.0],
        source_camera_extrinsics=source_extrinsics,
        target_camera_extrinsics=target_extrinsics,
    )

    assert rebased == pytest.approx([0.0, 1.0, 0.0])


def test_world_up_direction_camera_uses_dynamic_extrinsics() -> None:
    extrinsics = {
        "camera_frame": "opengl",
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
    }

    assert world_up_direction_camera(extrinsics) == pytest.approx([0.0, -1.0, 0.0])


def test_compile_grasp_seed_applies_camera_and_eef_transforms() -> None:
    result = compile_grasp_seed(
        _compile_parameters(),
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["candidate_id"] == "grasp_000"
    assert result["calibration_status"] == "candidate"
    assert result["not_validated"] is True
    assert result["contact_pose"]["frame"] == "world"
    assert result["contact_pose"]["xyz"] == pytest.approx([0.1036, -0.2, -0.3])
    assert result["hover_pose"]["xyz"] == pytest.approx([-0.0464, -0.2, -0.3])
    assert result["hover_pose"]["waypoint_role"] == "grasp_clearance"
    assert "grasp_stage" not in result["hover_pose"]
    assert result["approach_world_xyz"] == [1.0, 0.0, 0.0]
    assert result["hover_offset_world_xyz"] == [-0.15, 0.0, 0.0]
    assert result["requested_pregrasp_distance_m"] == 0.15
    assert result["pregrasp_distance_m"] == 0.15
    assert result["contact_pose"]["rotation_matrix"] == [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]


def test_compile_graspgenx_uses_normalized_tip_contact_center() -> None:
    parameters = _compile_parameters()
    parameters["camera_pose"].update(
        {
            "source_backend": "graspgenx",
            "gripper_tip_position_xyz": [0.2, 0.2, 0.3],
            "execution_reference_point": "gripper_tip_position_xyz",
            "execution_contact_center_xyz": [0.2, 0.2, 0.3],
        }
    )

    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["execution_reference_point"] == "gripper_tip_position_xyz"
    assert result["target_anchor_world_xyz"] == pytest.approx([0.2, -0.2, -0.3])
    assert result["contact_pose"]["xyz"] == pytest.approx([0.2036, -0.2, -0.3])
    assert result["orientation_clamped"] is False
    assert result["strategy_id"] is None
    assert result["strategy_selection"] == "generic_fallback"
    assert result["scene_epoch"] == 0
    guidance = result["execution_guidance"]
    assert guidance["contact"]["recommended_position_tolerance_m"] == 0.005
    assert guidance["contact"]["recommended_orientation_tolerance_rad"] == 0.10
    assert guidance["contact"]["verify_fresh_wrist_contact_before_close"] is True
    assert guidance["near_field_refinement"]["recommended"] is False


def test_compile_recommends_optional_wrist_refinement_for_small_scene_target() -> None:
    parameters = {
        **_compile_parameters(),
        "target_mask_quality": {
            "status": "usable",
            "area_fraction": 0.00873184,
        },
        "grasp_candidate_count": 1,
        "grasp_selection_advice": {"status": "skipped_single_candidate"},
    }
    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    refinement = result["execution_guidance"]["near_field_refinement"]
    assert refinement["recommended"] is True
    assert refinement["agent_discretion"] is True
    assert refinement["target_mask_area_fraction"] == pytest.approx(0.00873184)
    assert refinement["grasp_candidate_count"] == 1
    assert len(refinement["reasons"]) == 2
    assert {item["tool"] for item in refinement["suggested_actions"]} == {
        "compute_wrist_alignment",
        "grasp_pose_estimate",
    }


def test_explicit_strategy_preserves_contact_branch_for_small_scene_target() -> None:
    parameters = {
        **_compile_parameters(),
        "target_class": "upright_bottle",
        "strategy_id": "top-down-vertical-panda-p8",
        "target_mask_quality": {
            "status": "usable",
            "area_fraction": 0.00873184,
        },
        "grasp_candidate_count": 1,
        "grasp_selection_advice": {"status": "skipped_single_candidate"},
    }
    parameters["camera_pose"]["rotation_matrix"] = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]

    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    refinement = result["execution_guidance"]["near_field_refinement"]
    assert refinement["recommended"] is False
    assert refinement["reasons"] == []
    assert refinement["active_strategy_id"] == "top-down-vertical-panda-p8"
    assert "separate raw evidence" in refinement["strategy_continuity"]


def test_normalized_opencv_and_legacy_opengl_extrinsics_are_equivalent() -> None:
    legacy_parameters = _compile_parameters()
    normalized_parameters = deepcopy(legacy_parameters)
    normalized_parameters["camera_extrinsics"] = {
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0],
        "camera_to_world": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "matrix_layout": "row_major",
        "frame_transform": "camera_to_world",
        "camera_frame": "opencv",
    }

    legacy = compile_grasp_seed(
        legacy_parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )
    normalized = compile_grasp_seed(
        normalized_parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    for key in (
        "approach_world_xyz",
        "hover_offset_world_xyz",
        "precontact_pose",
        "wrist_alignment_pose",
    ):
        assert normalized.get(key) == legacy.get(key)
    for pose_key in ("contact_pose", "hover_pose"):
        assert normalized[pose_key]["xyz"] == legacy[pose_key]["xyz"]
        assert (
            normalized[pose_key]["rotation_matrix"]
            == legacy[pose_key]["rotation_matrix"]
        )
    assert camera_optical_forward_world(
        legacy_parameters["camera_extrinsics"]
    ) == camera_optical_forward_world(normalized_parameters["camera_extrinsics"])


def test_grasp_geometry_rejects_unknown_camera_frame() -> None:
    parameters = _compile_parameters()
    parameters["camera_extrinsics"]["camera_frame"] = "backend_guess"

    with pytest.raises(GraspGeometryError, match="unsupported value"):
        compile_grasp_seed(
            parameters,
            profile=_profile(),
            profile_sha256="profile-sha",
        )


def test_refinement_hover_accepts_camera_to_world_matrix() -> None:
    pose = grasp_refinement_hover_pose(
        _candidate(),
        {
            "camera_to_world": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        },
        scene_epoch=3,
        recovery_id="recovery-1",
    )

    assert pose["xyz"] == pytest.approx([0.1, -0.2, -0.1])
    assert pose["source_grasp_id"] == "grasp_000"
    assert pose["recovery_id"] == "recovery-1"
    assert pose["scene_epoch"] == 3
    assert pose["waypoint_role"] == "grasp_refinement_clearance"
    assert "grasp_stage" not in pose


def test_target_mask_quality_rejects_clipped_mask_and_reports_depth(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.png"
    mask = Image.new("L", (8, 8), 0)
    for y in range(2, 6):
        for x in range(0, 3):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("I;16", (8, 8), 1000).save(depth_path)

    quality = assess_target_mask_quality(mask_path, depth_path=depth_path)

    assert quality["schema_version"] == "openeta.target_mask_quality.v1"
    assert quality["status"] == "clipped_mask"
    assert quality["usable_for_targeted_geometry"] is False
    assert quality["touches_image_boundary"] is True
    assert quality["bbox_xyxy"] == [0, 2, 3, 6]
    assert quality["depth_coverage"] == 1.0
    assert quality["failed_checks"] == ["mask_not_clipped"]


def test_target_mask_quality_rejects_nearly_clipped_full_frame_mask(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "near-border-mask.png"
    depth_path = tmp_path / "near-border-depth.png"
    mask = Image.new("L", (100, 100), 0)
    for y in range(1, 80):
        for x in range(1, 80):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("I;16", (100, 100), 1000).save(depth_path)

    quality = assess_target_mask_quality(mask_path, depth_path=depth_path)

    assert quality["status"] == "clipped_mask"
    assert quality["usable_for_targeted_geometry"] is False
    assert quality["touches_image_boundary"] is False
    assert quality["border_clearance_px"] == 1
    assert quality["minimum_border_clearance_fraction"] == 0.02
    assert quality["failed_checks"] == ["minimum_border_clearance"]


def test_wrist_viewpoint_proposals_point_camera_at_compiled_target() -> None:
    compiled = compile_grasp_seed(
        _compile_parameters(),
        profile=_profile(),
        profile_sha256="profile-sha",
    )
    result = propose_wrist_viewpoints(
        {
            "compiled_grasp": compiled,
            "source_packet_id": "packet-current",
            "camera_frame_id": "robot0_eye_in_hand",
            "camera_extrinsics": {
                "camera_frame": "opencv",
                "frame_transform": "camera_to_world",
                "camera_to_world": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "current_eef_pose": {
                "xyz": [0.0, 0.0, 0.0],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            "object_scene_epoch": 0,
            "robot_motion_epoch": 4,
            "standoff_m": 0.18,
        }
    )

    assert result["schema_version"] == "openeta.wrist_viewpoint_proposal.v1"
    assert result["proposal_id"].startswith("wrist_viewpoint:")
    assert result["validity"]["reusable_across_observation_packet_refresh"] is True
    assert result["next_action_contract"]["recommended_tool"] == "ik_preview_check"
    post_reach = result["post_reach_evidence_contract"]
    assert post_reach["viewpoint_reached_is_not_contact_refined"] is True
    assert post_reach["fresh_packet_source"] == (
        "current_observation.source_packet_id_after_move"
    )
    assert [item["tool"] for item in post_reach["recommended_sequence"]] == [
        "sam3",
        "select_sam3_detection",
        "compute_wrist_alignment_or_grasp_pose_estimate",
    ]
    assert len(result["candidates"]) == 3
    for candidate in result["candidates"]:
        pose = candidate["target_pose"]
        camera = candidate["camera_goal"]
        delta = [
            result["target_anchor_world_xyz"][index] - camera["xyz"][index]
            for index in range(3)
        ]
        norm = sum(value * value for value in delta) ** 0.5
        assert camera["optical_forward_world_xyz"] == pytest.approx(
            [value / norm for value in delta], abs=1e-6
        )
        assert pose["waypoint_role"] == "wrist_observation_viewpoint"
        assert pose["viewpoint_candidate_id"] == candidate["candidate_id"]
        assert pose["camera_frame_id"] == "robot0_eye_in_hand"
        assert candidate["requires_ik_preview"] is True


def test_compile_grasp_seed_uses_generic_fallback_for_unlisted_object() -> None:
    parameters = _compile_parameters()
    parameters.pop("target_class")
    parameters["target_geometry_family"] = "apple"

    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["strategy_id"] is None
    assert result["strategy_selection"] == "generic_fallback"
    assert result["orientation_clamped"] is False
    assert result["outside_validated_strategy_scope"] is True
    assert result["approach_world_xyz"] == [1.0, 0.0, 0.0]
    assert result["hover_offset_world_xyz"] == [-0.15, 0.0, 0.0]
    assert result["hover_pose"]["xyz"] == pytest.approx([-0.0464, -0.2, -0.3])
    assert result["hover_pose"]["xyz"][2] == result["contact_pose"]["xyz"][2]
    assert result["contact_pose"]["rotation_matrix"] != [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]


@pytest.mark.parametrize(
    ("geometry_family", "strategy_id"),
    [
        ("upright_bottle", "top-down-vertical-panda-p8"),
        ("bowl", "top-down-bowl-panda-p8"),
        ("drawer_handle", "top-down-drawer-handle-panda-p8"),
    ],
)
def test_compile_grasp_seed_accepts_explicit_candidate_task_family_strategy(
    geometry_family: str,
    strategy_id: str,
) -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = geometry_family
    parameters["strategy_id"] = strategy_id
    if geometry_family in {"upright_bottle", "bowl"}:
        parameters["camera_pose"]["rotation_matrix"] = [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]

    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["strategy_id"] == strategy_id
    assert result["strategy_status"] == "candidate"
    assert result["strategy_selection"] == "explicit"
    assert result["orientation_clamped"] is True
    assert result["approach_world_xyz"] == [0.0, 0.0, -1.0]
    assert result["outside_validated_strategy_scope"] is True


def test_articulated_handle_front_mode_preserves_native_pose_and_provenance() -> None:
    parameters = _compile_parameters()
    parameters.update(
        {
            "target_class": "articulated_handle",
            "approach_mode": "front",
            "strategy_id": "native-front-articulated-handle-panda-p8",
        }
    )

    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["approach_mode"] == "front"
    assert result["strategy_id"] == "native-front-articulated-handle-panda-p8"
    assert result["orientation_clamped"] is False
    assert result["approach_world_xyz"] == [1.0, 0.0, 0.0]
    assert result["hover_pose"]["approach_mode"] == "front"
    assert result["contact_pose"]["approach_mode"] == "front"


def test_approach_mode_changes_compiled_id_and_rejects_incompatible_geometry() -> None:
    top_down = _compile_parameters()
    top_down.update(
        {
            "target_class": "articulated_handle",
            "approach_mode": "top_down",
            "strategy_id": "top-down-drawer-handle-panda-p8",
        }
    )
    front = deepcopy(top_down)
    front.update(
        {
            "approach_mode": "front",
            "strategy_id": "native-front-articulated-handle-panda-p8",
        }
    )

    top_result = compile_grasp_seed(
        top_down,
        profile=_profile(),
        profile_sha256="profile-sha",
    )
    front_result = compile_grasp_seed(
        front,
        profile=_profile(),
        profile_sha256="profile-sha",
    )
    assert top_result["compiled_grasp_id"] != front_result["compiled_grasp_id"]

    nested = deepcopy(front)
    nested.pop("approach_mode")
    nested["articulated_handle_options"] = {"approach_mode": "front"}
    nested_result = compile_grasp_seed(
        nested,
        profile=_profile(),
        profile_sha256="profile-sha",
    )
    assert nested_result["approach_mode"] == "front"

    forged = _compile_parameters()
    forged.update(
        {
            "target_class": "upright_can",
            "approach_mode": "side",
            "strategy_id": "native-side-articulated-handle-panda-p8",
        }
    )
    with pytest.raises(GraspGeometryError, match="reserved for articulated_handle"):
        compile_grasp_seed(
            forged,
            profile=_profile(),
            profile_sha256="profile-sha",
        )


def test_bowl_strategy_rejects_candidate_without_downward_native_approach() -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = "bowl"
    parameters["strategy_id"] = "top-down-bowl-panda-p8"

    with pytest.raises(GraspGeometryError, match="native downward alignment"):
        compile_grasp_seed(
            parameters,
            profile=_profile(),
            profile_sha256="profile-sha",
        )


def test_vertical_top_down_strategy_rejects_upward_or_side_native_candidate() -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = "boxed_item"
    parameters["strategy_id"] = "top-down-vertical-panda-p8"

    with pytest.raises(GraspGeometryError, match="native downward alignment"):
        compile_grasp_seed(
            parameters,
            profile=_profile(),
            profile_sha256="profile-sha",
        )


def test_removed_fallback_markers_cannot_bypass_candidate_validation() -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = "bowl"
    parameters["strategy_id"] = "top-down-bowl-panda-p8"
    parameters["candidate_fallback"] = True
    parameters["camera_pose"]["candidate_fallback"] = True
    parameters["camera_pose"]["final_refinable_fallback"] = True

    with pytest.raises(GraspGeometryError, match="native downward alignment"):
        compile_grasp_seed(
            parameters,
            profile=_profile(),
            profile_sha256="profile-sha",
        )


def test_bowl_candidate_filter_returns_structured_candidate_rejection() -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = "bowl"
    parameters["strategy_id"] = "top-down-bowl-panda-p8"
    spec = build_default_tool_registry().get("compile_grasp_seed")

    result = build_compile_grasp_seed_handler()(
        ToolExecutionContext(
            name="compile_grasp_seed",
            spec=spec,
            parameters=parameters,
        )
    )

    assert result.success is False
    assert result.details["outputs"] == {
        "reason": "grasp_seed_candidate_rejected",
        "candidate_rejection": True,
        "candidate_id": "grasp_000",
        "rejection_code": "strategy_alignment_rejected",
        "recovery_class": "perception_refinable",
    }
    assert result.details["diagnostics"][0]["candidate_rejection"] is True


def test_compile_grasp_seed_clamps_requested_hover_to_safe_normal_standoff() -> None:
    parameters = _compile_parameters()
    parameters["pregrasp_distance_m"] = 0.04

    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["requested_pregrasp_distance_m"] == 0.04
    assert result["pregrasp_distance_m"] == 0.15
    assert result["hover_offset_world_xyz"] == [-0.15, 0.0, 0.0]


def test_compile_grasp_seed_enforces_physical_gripper_width() -> None:
    parameters = _compile_parameters()
    parameters["camera_pose"]["width"] = 0.081

    with pytest.raises(GraspGeometryError, match="camera_pose.width"):
        compile_grasp_seed(
            parameters,
            profile=_profile(),
            profile_sha256="profile-sha",
        )


def test_compile_grasp_seed_rejects_strategy_above_physical_width() -> None:
    strategy = {
        "schema_version": "openeta.grasp_strategy.v1",
        "status": "candidate",
        "strategy_id": "oversized",
        "compatibility": {"calibration_ids": ["graspnet-eef-panda-p8"]},
        "automatic_activation": {"target_geometry_families": ["apple"]},
        "constraints": {"grasp_width_bounds_m": [0.02, 0.09]},
        "pose_policy": {
            "orientation": "preserve_candidate",
            "approach_axis": "preserve_candidate",
        },
    }
    parameters = _compile_parameters()
    parameters["target_class"] = "apple"
    parameters["strategy_id"] = "oversized"

    with pytest.raises(GraspGeometryError, match="exceeds calibration"):
        compile_grasp_seed(
            parameters,
            profile=_profile(),
            profile_sha256="profile-sha",
            strategies=[strategy],
        )


def test_compile_grasp_seed_keeps_legacy_v1_object_allowlist() -> None:
    profile = deepcopy(_profile())
    profile["schema_version"] = "libero.grasp_to_eef_calibration.v1"
    profile.pop("compatibility")
    profile["restricted_geometry"] = {
        "target_classes": ["upright_can", "boxed_item"],
        "width_bounds_m": [0.02, 0.075],
        "approach_axis": "world_-Z",
        "eef_orientation": "top_down",
    }
    parameters = _compile_parameters()
    parameters["target_class"] = "apple"

    with pytest.raises(GraspGeometryError, match="legacy target_class"):
        compile_grasp_seed(
            parameters,
            profile=profile,
            profile_sha256="legacy-profile-sha",
        )


def test_wrist_alignment_with_clipped_mask_and_clamped_correction_emits_no_pose(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.png"
    mask = Image.new("L", (8, 8), 0)
    for y in range(4, 7):
        for x in range(5, 8):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("I;16", (8, 8), 1000).save(depth_path)
    compiled = compile_grasp_seed(
        _compile_parameters(),
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    result = compute_wrist_alignment(
        {
            "compiled_grasp": compiled,
            "target_mask": str(mask_path),
            "depth": str(depth_path),
            "intrinsics": {
                "fx": 100.0,
                "fy": 100.0,
                "cx": 4.0,
                "cy": 4.0,
                "width": 8,
                "height": 8,
                "scale": 1000.0,
            },
            "camera_extrinsics": {
                **_compile_parameters()["camera_extrinsics"],
                "camera_frame": "opencv",
            },
            "current_eef_pose": {
                "xyz": [0.0, 0.0, 0.6],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "max_correction_m": 0.02,
            "scene_epoch": 0,
        },
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["target_pixel_xy"] == pytest.approx([6.0, 5.0])
    assert result["correction_clamped"] is True
    assert sum(value * value for value in result["correction_world_xyz"]) ** 0.5 == pytest.approx(
        0.02, abs=1e-6
    )
    assert result["status"] == "requires_better_view"
    assert result["executable_reference"] is False
    assert result["aligned_hover_pose"] is None
    assert result["adjusted_contact_pose"] is None
    failed_codes = {
        check["code"] for check in result["operating_region"]["failed_checks"]
    }
    assert "target_mask_not_clipped" in failed_codes
    assert "eef_near_compiled_clearance" in failed_codes
    assert "correction_within_limit" in failed_codes
    assert result["desired_pixel_xy"] == [4.0, 4.0]
    assert result["gripper_center_projection"]["calibration_id"] == (
        "graspnet-eef-panda-p8"
    )


def test_bowl_wrist_alignment_targets_nearest_shallow_rim_pixel(tmp_path: Path) -> None:
    mask_path = tmp_path / "bowl-mask.png"
    depth_path = tmp_path / "bowl-depth.png"
    mask = Image.new("L", (9, 9), 0)
    depth = Image.new("I;16", (9, 9), 0)
    for y in range(1, 8):
        for x in range(1, 8):
            mask.putpixel((x, y), 255)
            depth.putpixel((x, y), 1100)
    for x, y in ((4, 2), (3, 2), (5, 2), (4, 1), (3, 1), (5, 1)):
        depth.putpixel((x, y), 1000)
    mask.save(mask_path)
    depth.save(depth_path)
    parameters = _compile_parameters()
    parameters["target_class"] = "bowl"
    parameters["strategy_id"] = "top-down-bowl-panda-p8"
    parameters["camera_pose"]["rotation_matrix"] = [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ]
    compiled = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )
    compiled["hover_pose"]["xyz"] = [0.0, 0.0, 0.6]

    result = compute_wrist_alignment(
        {
            "compiled_grasp": compiled,
            "target_mask": str(mask_path),
            "depth": str(depth_path),
            "intrinsics": {
                "fx": 100.0,
                "fy": 100.0,
                "cx": 4.0,
                "cy": 4.0,
                "width": 9,
                "height": 9,
                "scale": 1000.0,
            },
            "camera_extrinsics": {
                **_compile_parameters()["camera_extrinsics"],
                "camera_frame": "opencv",
            },
            "current_eef_pose": {
                "xyz": [0.0, 0.0, 0.6],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "scene_epoch": 0,
        },
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["target_region"] == "nearest_shallow_surface"
    assert result["target_pixel_xy"] == [4, 2]
    assert result["target_depth_m"] == 1.0
    assert compiled["precontact_pose"]["waypoint_role"] == "grasp_precontact"
    assert result["status"] == "aligned_reference_ready"
    assert result["executable_reference"] is True
    assert result["adjusted_precontact_pose"]["waypoint_role"] == "grasp_precontact"
    assert result["adjusted_precontact_pose"]["alignment_id"] == result["alignment_id"]


def test_wrist_alignment_projects_libero_grip_site_instead_of_principal_point(
    tmp_path: Path,
) -> None:
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.png"
    mask = Image.new("L", (512, 512), 0)
    for y in range(330, 351):
        for x in range(245, 266):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("I;16", (512, 512), 1000).save(depth_path)
    compiled = compile_grasp_seed(
        _compile_parameters(),
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    result = compute_wrist_alignment(
        {
            "compiled_grasp": compiled,
            "target_mask": str(mask_path),
            "depth": str(depth_path),
            "intrinsics": {
                "fx": 333.6256954473487,
                "fy": 333.6256954473487,
                "cx": 256.0,
                "cy": 256.0,
                "width": 512,
                "height": 512,
                "scale": 1000.0,
            },
            "camera_extrinsics": {
                "camera_frame": "opengl",
                "frame_transform": "camera_to_world",
                "matrix_layout": "row_major",
                "mat": [
                    0.0004918296876253447,
                    0.9983867450712568,
                    0.056777331476359764,
                    -0.9999998786606127,
                    0.0004926243565984145,
                    3.5101782591695496e-13,
                    -2.7969896037469133e-05,
                    -0.05677732458703341,
                    0.998386866214906,
                ],
                "pos": [
                    -0.0930379221590543,
                    2.4631217821431277e-05,
                    0.3552841355231346,
                ],
            },
            "current_eef_pose": {
                "xyz": [
                    -0.14846466056582405,
                    -4.2543992093357714e-14,
                    0.2612794757296404,
                ],
                "quat_xyzw": [
                    0.9995966048795353,
                    0.00024621283211107015,
                    -0.028400120485872912,
                    -6.995295789799863e-06,
                ],
            },
            "scene_epoch": 0,
        },
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["desired_pixel_xy"] == pytest.approx([256.0, 427.972008])
    assert result["desired_pixel_xy"] != [256.0, 256.0]
    assert result["gripper_center_projection"]["camera_extrinsics_source"] == (
        "live_camera_to_world"
    )


def test_wrist_alignment_composes_loaded_eye_in_hand_calibration(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.png"
    mask = Image.new("L", (8, 8), 0)
    for y in range(3, 6):
        for x in range(3, 6):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("I;16", (8, 8), 1000).save(depth_path)
    compiled = compile_grasp_seed(
        _compile_parameters(),
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    result = compute_wrist_alignment(
        {
            "compiled_grasp": compiled,
            "target_mask": str(mask_path),
            "depth": str(depth_path),
            "intrinsics": {
                "fx": 100.0,
                "fy": 100.0,
                "cx": 4.0,
                "cy": 4.0,
                "width": 8,
                "height": 8,
                "scale": 1000.0,
            },
            "camera_extrinsics": {
                "type": "T_gripper_cam",
                "frame": "gripper",
                "T_gripper_cam": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            },
            "current_eef_pose": {
                "xyz": [0.0, 0.0, 0.6],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "scene_epoch": 0,
        },
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["desired_pixel_xy"] == [4.0, 4.0]
    assert result["gripper_center_projection"]["camera_extrinsics_source"] == (
        "T_world_eef_x_T_gripper_cam"
    )


def test_wrist_alignment_rejects_agent_supplied_gripper_pixel() -> None:
    compiled = compile_grasp_seed(
        _compile_parameters(),
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    with pytest.raises(GraspGeometryError, match="host-derived"):
        compute_wrist_alignment(
            {
                "compiled_grasp": compiled,
                "desired_pixel_xy": [4.0, 4.0],
            },
            profile=_profile(),
            profile_sha256="profile-sha",
        )


def test_wrist_alignment_handler_loads_session_profile_automatically(
    tmp_path: Path,
) -> None:
    profile = _profile()
    profile["wrist_alignment"]["eef_to_gripper_center_xyz"] = [0.01, 0.0, 0.0]
    profile_path = tmp_path / "grasp_profile.json"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    compiled = compile_grasp_seed(
        _compile_parameters(),
        profile=profile,
        profile_sha256=profile_sha256,
    )
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.png"
    mask = Image.new("L", (10, 10), 0)
    for y in range(3, 6):
        for x in range(4, 7):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("I;16", (10, 10), 1000).save(depth_path)
    spec = build_default_tool_registry().get("compute_wrist_alignment")
    assert set(spec.parameters) == {"bundle_id", "max_correction_m"}

    result = build_wrist_alignment_handler(profile_path)(
        ToolExecutionContext(
            name="compute_wrist_alignment",
            spec=spec,
            parameters={
                "compiled_grasp": compiled,
                "target_mask": str(mask_path),
                "depth": str(depth_path),
                "intrinsics": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 4.0,
                    "cy": 4.0,
                    "width": 10,
                    "height": 10,
                    "scale": 1000.0,
                },
                "camera_extrinsics": {
                    "camera_frame": "opencv",
                    "frame_transform": "camera_to_world",
                    "matrix_layout": "row_major",
                    "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    "pos": [0.0, 0.0, 0.0],
                },
                "current_eef_pose": {
                    "xyz": [0.0, 0.0, 1.0],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "scene_epoch": 0,
            },
        )
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["desired_pixel_xy"] == [5.0, 4.0]
    assert outputs["gripper_center_projection"]["profile_sha256"] == profile_sha256


def test_wrist_alignment_handler_reports_requires_better_view_without_poses(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "grasp_profile.json"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")
    profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    compiled = compile_grasp_seed(
        _compile_parameters(),
        profile=_profile(),
        profile_sha256=profile_sha256,
    )
    mask_path = tmp_path / "mask.png"
    depth_path = tmp_path / "depth.png"
    mask = Image.new("L", (8, 8), 0)
    for y in range(3, 6):
        for x in range(3, 6):
            mask.putpixel((x, y), 255)
    mask.save(mask_path)
    Image.new("I;16", (8, 8), 1000).save(depth_path)
    spec = build_default_tool_registry().get("compute_wrist_alignment")

    result = build_wrist_alignment_handler(profile_path)(
        ToolExecutionContext(
            name="compute_wrist_alignment",
            spec=spec,
            parameters={
                "compiled_grasp": compiled,
                "target_mask": str(mask_path),
                "depth": str(depth_path),
                "intrinsics": {
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 4.0,
                    "cy": 4.0,
                    "width": 8,
                    "height": 8,
                    "scale": 1000.0,
                },
                "camera_extrinsics": {
                    **_compile_parameters()["camera_extrinsics"],
                    "camera_frame": "opencv",
                },
                "current_eef_pose": {
                    "xyz": [0.0, 0.0, 0.6],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "scene_epoch": 0,
            },
        )
    )

    assert result.success is True
    assert result.details["semantic_outcome"] == "requires_better_view"
    assert result.details["outputs"]["executable_reference"] is False
    assert result.details["outputs"]["aligned_hover_pose"] is None
    assert result.details["recovery_options"]
    assert result.details["recovery_options"][0]["action"] == (
        "run_full_wrist_grasp_estimate"
    )
    assert "Do not repeat SAM3 on unchanged wrist pixels" in result.content
