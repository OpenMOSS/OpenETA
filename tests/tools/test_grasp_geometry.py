from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from agent.tools.grasp_geometry import (
    DEFAULT_GRASP_PROFILE,
    build_compile_grasp_seed_handler,
    camera_optical_forward_world,
    GraspGeometryError,
    compile_grasp_seed,
    compute_wrist_alignment,
    grasp_refinement_hover_pose,
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
    assert result["hover_pose"]["xyz"] == pytest.approx([0.1036, -0.2, -0.15])
    assert result["approach_world_xyz"] == [0.0, 0.0, -1.0]
    assert result["hover_offset_world_xyz"] == [0.0, 0.0, 0.15]
    assert result["requested_pregrasp_distance_m"] == 0.15
    assert result["pregrasp_distance_m"] == 0.15
    assert result["contact_pose"]["rotation_matrix"] == [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    assert result["orientation_clamped"] is True
    assert result["strategy_id"] == "top-down-vertical-panda-p8"
    assert result["strategy_selection"] == "automatic_geometry_family"
    assert result["scene_epoch"] == 0


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
def test_compile_grasp_seed_selects_candidate_task_family_strategy(
    geometry_family: str,
    strategy_id: str,
) -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = geometry_family
    if geometry_family == "bowl":
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
    assert result["strategy_selection"] == "automatic_geometry_family"
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


def test_approach_mode_changes_compiled_id_and_rejects_agent_forgery() -> None:
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

    with pytest.raises(GraspGeometryError, match="native downward alignment"):
        compile_grasp_seed(
            parameters,
            profile=_profile(),
            profile_sha256="profile-sha",
        )


def test_bowl_candidate_filter_returns_structured_candidate_rejection() -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = "bowl"
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


def test_final_refinable_candidate_bypasses_only_strategy_filter() -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = "bowl"
    parameters["camera_pose"]["final_refinable_fallback"] = True

    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["final_refinable_fallback"] is True
    assert result["candidate_id"] == "grasp_000"
    assert result["gripper_width_m"] == pytest.approx(0.06)


def test_bowl_score_fallback_bypasses_alignment_filter_with_explicit_warning() -> None:
    parameters = _compile_parameters()
    parameters["target_class"] = "bowl"
    parameters["camera_pose"]["candidate_fallback"] = True

    result = compile_grasp_seed(
        parameters,
        profile=_profile(),
        profile_sha256="profile-sha",
    )

    assert result["candidate_fallback"] is True
    assert "score-selected fallback" in result["warning"]


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
    assert result["hover_offset_world_xyz"] == [0.0, 0.0, 0.15]


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


def test_wrist_alignment_uses_full_frame_mask_and_clamps_correction(tmp_path: Path) -> None:
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
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 4.0, "cy": 4.0, "scale": 1000.0},
            "camera_extrinsics": _compile_parameters()["camera_extrinsics"],
            "current_eef_pose": {"xyz": [0.0, 0.0, 0.6]},
            "desired_pixel_xy": [4.0, 4.0],
            "max_correction_m": 0.02,
            "scene_epoch": 0,
        }
    )

    assert result["target_pixel_xy"] == pytest.approx([6.0, 5.0])
    assert result["correction_clamped"] is True
    assert sum(value * value for value in result["correction_world_xyz"]) ** 0.5 == pytest.approx(
        0.02, abs=1e-6
    )
    assert result["aligned_hover_pose"]["frame"] == "world"


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
                "scale": 1000.0,
            },
            "camera_extrinsics": _compile_parameters()["camera_extrinsics"],
            "current_eef_pose": {"xyz": [0.0, 0.0, 0.6]},
            "desired_pixel_xy": [4.0, 4.0],
            "scene_epoch": 0,
        }
    )

    assert result["target_region"] == "nearest_shallow_surface"
    assert result["target_pixel_xy"] == [4, 2]
    assert result["target_depth_m"] == 1.0
    assert compiled["precontact_pose"]["grasp_stage"] == "precontact"
    assert result["adjusted_precontact_pose"]["grasp_stage"] == "precontact"
    assert result["adjusted_precontact_pose"]["alignment_id"] == result["alignment_id"]
