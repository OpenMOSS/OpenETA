from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

from agent.tools.grasp_geometry import (
    DEFAULT_GRASP_PROFILE,
    build_compile_grasp_seed_handler,
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
    assert result["hover_pose"]["xyz"] == pytest.approx([0.1036, -0.2, -0.05])
    assert result["approach_world_xyz"] == [0.0, 0.0, -1.0]
    assert result["hover_offset_world_xyz"] == [0.0, 0.0, 0.25]
    assert result["requested_pregrasp_distance_m"] == 0.25
    assert result["pregrasp_distance_m"] == 0.25
    # Top-down clamp yields [[1,0,0],[0,-1,0],[0,0,-1]]; the forced +pi roll about
    # the eef Z (Rz(pi), right-multiplied) flips the X/Y columns' signs.
    assert result["contact_pose"]["rotation_matrix"] == [
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    assert result["orientation_clamped"] is True
    assert result["strategy_id"] == "top-down-vertical-panda-p8"
    assert result["strategy_selection"] == "automatic_geometry_family"
    assert result["scene_epoch"] == 0


def test_candidate_world_pose_matches_compile_generic_fallback() -> None:
    from agent.tools.grasp_geometry import candidate_world_pose

    # A tilted candidate under an unlisted class -> generic_fallback, so compile
    # preserves the estimator orientation and the preview must match it exactly.
    candidate = _candidate()
    candidate["rotation_matrix"] = [
        [0.7071, 0.0, 0.7071],
        [0.0, 1.0, 0.0],
        [-0.7071, 0.0, 0.7071],
    ]
    extrinsics = {"pos": [0.1, 0.2, 0.3], "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}

    preview = candidate_world_pose(candidate, extrinsics)
    assert preview is not None

    params = {
        "camera_pose": candidate,
        "camera_extrinsics": extrinsics,
        "camera_frame_id": "agentview",
        "target_class": "banana",  # unlisted -> generic_fallback, no orientation clamp
        "scene_epoch": 0,
    }
    compiled = compile_grasp_seed(params, profile=_profile(), profile_sha256="profile-sha")

    assert compiled["strategy_selection"] == "generic_fallback"
    assert compiled["orientation_clamped"] is False
    # Same approach math on both sides.
    assert preview["approach_world_xyz"] == compiled["approach_world_xyz"]
    assert preview["world_downward_alignment"] == compiled["native_downward_alignment"]
    # Preview is the grasp frame; compile's contact pose adds T_grasp_eef on top,
    # so the grasp-frame rotation is a reference, not the final EEF rotation.
    assert preview["frame"] == "world"
    assert preview["grasp_frame"] == "graspnet"


def test_candidate_world_pose_returns_none_on_malformed_input() -> None:
    from agent.tools.grasp_geometry import candidate_world_pose

    assert candidate_world_pose({"rotation_matrix": [[1.0]]}, {}) is None
    assert (
        candidate_world_pose(_candidate(), {"pos": [0.0, 0.0], "mat": [1.0, 2.0]})
        is None
    )


def test_compile_grasp_seed_normalizes_opencv_extrinsics_basis() -> None:
    # Regression: real-bench extrinsics report the rotation in the OpenCV basis
    # (camera_frame="opencv", world<-opencv). The pipeline multiplies by
    # _OPENCV_TO_OPENGL downstream, so an opencv-basis mat used verbatim applies
    # diag(1,-1,-1) twice and mirrors the grasp across the camera axes (metre-scale
    # world error). An opencv-tagged mat must yield the same world pose as its
    # OpenGL-native equivalent (mat_opencv @ diag(1,-1,-1)).
    opencv_mat = [
        0.588760235022, 0.691645417959, -0.418315672038,
        0.793378186547, -0.395473791376, 0.46276509532,
        0.154636472957, -0.604340215566, -0.781575629789,
    ]
    # OpenGL-native basis: right-multiply the 3x3 by diag(1,-1,-1).
    opengl_mat = [
        opencv_mat[0], -opencv_mat[1], -opencv_mat[2],
        opencv_mat[3], -opencv_mat[4], -opencv_mat[5],
        opencv_mat[6], -opencv_mat[7], -opencv_mat[8],
    ]
    pos = [0.451245742852, -1.001174802857, 0.784206747707]

    opencv_params = _compile_parameters()
    opencv_params["camera_extrinsics"] = {
        "pos": pos,
        "mat": opencv_mat,
        "camera_frame": "opencv",
    }
    opengl_params = _compile_parameters()
    opengl_params["camera_extrinsics"] = {
        "pos": pos,
        "mat": opengl_mat,
        "camera_frame": "opengl",
    }

    opencv_result = compile_grasp_seed(
        opencv_params, profile=_profile(), profile_sha256="profile-sha"
    )
    opengl_result = compile_grasp_seed(
        opengl_params, profile=_profile(), profile_sha256="profile-sha"
    )

    assert opencv_result["contact_pose"]["xyz"] == pytest.approx(
        opengl_result["contact_pose"]["xyz"]
    )
    assert opencv_result["hover_pose"]["xyz"] == pytest.approx(
        opengl_result["hover_pose"]["xyz"]
    )


def test_compile_grasp_seed_rejects_unknown_extrinsics_frame() -> None:
    params = _compile_parameters()
    params["camera_extrinsics"] = {
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "camera_frame": "ros",
    }
    with pytest.raises(GraspGeometryError, match="camera_frame"):
        compile_grasp_seed(params, profile=_profile(), profile_sha256="profile-sha")


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

    assert pose["xyz"] == pytest.approx([0.1, -0.2, -0.05])
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
    assert result["hover_offset_world_xyz"] == [-0.25, 0.0, 0.0]
    assert result["hover_pose"]["xyz"] == pytest.approx([-0.1464, -0.2, -0.3])
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
    assert result["pregrasp_distance_m"] == 0.25
    assert result["hover_offset_world_xyz"] == [0.0, 0.0, 0.25]


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
