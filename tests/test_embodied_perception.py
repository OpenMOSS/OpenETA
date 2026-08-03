from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from agent.tools.embodied_perception import ObservationBoundPerception, render_anygrasp_candidates


def _png_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _episode_frame(tmp_path: Path, *, observation_id: str = "obs-000001") -> dict:
    media = tmp_path / "media" / "frames"
    media.mkdir(parents=True, exist_ok=True)
    rgb = Image.new("RGB", (96, 96), (80, 90, 100))
    ImageDraw.Draw(rgb).rectangle((30, 30, 65, 65), fill=(20, 20, 20))
    rgb.save(media / "rgb.png")
    Image.new("I;16", (96, 96), 1000).save(media / "depth.png")
    return {
        "observation_id": observation_id,
        "frames": [
            {
                "frame_id": "frame-000001-agentview",
                "observation_id": observation_id,
                "camera_id": "agentview",
                "rgb_path": "media/frames/rgb.png",
                "depth_path": "media/frames/depth.png",
                "metadata": {
                    "intrinsics": {
                        "fx": 80.0,
                        "fy": 80.0,
                        "cx": 48.0,
                        "cy": 48.0,
                        "width": 96,
                        "height": 96,
                    },
                    "extrinsics": {"frame_transform": "camera_to_world"},
                },
            }
        ],
    }


def _sam3_response(mask_path: Path, *, count: int = 1) -> dict:
    detections = []
    for index in range(count):
        detections.append(
            {
                "label": "black bowl",
                "score": 0.9 - index * 0.1,
                "bbox_xyxy": [30 + index, 30, 65 + index, 65],
                "mask": {"format": "png", "base64": _png_base64(mask_path)},
                "area_px": 1000 - index,
            }
        )
    return {
        "success": True,
        "content": "SAM3 segmentation completed.",
        "details": {
            "detection_count": count,
            "detections": detections,
            "artifacts": [],
        },
    }


def _anygrasp_response() -> dict:
    return {
        "success": True,
        "content": "AnyGrasp grasp detection completed.",
        "details": {
            "tool": "anygrasp",
            "backend": "anygrasp_mcp",
            "model": "anygrasp_sdk",
            "mode": "targeted",
            "candidate_count": 1,
            "grasp_candidates": [
                {
                    "id": "backend-0",
                    "frame": "camera",
                    "camera_frame": "opencv",
                    "score": 0.87,
                    "translation_xyz": [0.0, 0.0, 1.0],
                    "rotation_matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "depth": 0.03,
                    "width": 0.1,
                    "height": 0.03,
                    "gripper_tip_position_xyz": [0.05, 0.0, 1.0],
                }
            ],
            "artifacts": [],
            "metadata": {},
        },
    }


def _make_perception(tmp_path: Path, *, sam3_response: dict | None = None) -> ObservationBoundPerception:
    mask = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(mask).rectangle((30, 30, 65, 65), fill=255)
    mask_path = tmp_path / "mask-source.png"
    mask.save(mask_path)
    response = sam3_response or _sam3_response(mask_path)

    def sam3(_request: dict) -> dict:
        return response

    def anygrasp(request: dict) -> dict:
        assert request["intrinsics"]["scale"] == 1000.0
        assert request["mode"] == "targeted"
        assert request["rgb"]["format"] == "png"
        assert request["depth"]["format"] == "png"
        assert request["target_mask"]["format"] == "png"
        assert request["rgb"]["base64"]
        assert request["depth"]["base64"]
        assert request["target_mask"]["base64"]
        return _anygrasp_response()

    return ObservationBoundPerception(
        artifact_root=tmp_path,
        sam3=sam3,
        anygrasp=anygrasp,
        output_root=tmp_path / "perception",
    )


def test_observation_bound_pipeline_keeps_binding_without_static_pose_render(
    tmp_path: Path,
) -> None:
    record = _episode_frame(tmp_path)
    perception = _make_perception(tmp_path)

    segmented = perception.segment_object(record, "black bowl")
    assert segmented.success is True
    assert segmented.details["tool"] == "segment_object"
    assert segmented.details["backend_tool"] == "sam3"
    assert segmented.details["observation"] == {
        "observation_id": "obs-000001",
        "frame_id": "frame-000001-agentview",
        "camera_id": "agentview",
    }
    assert Path(segmented.details["selected_detection"]["mask_ref"]).is_file()

    grasps = perception.propose_grasps(record)

    assert grasps.success is True
    assert grasps.details["tool"] == "propose_grasps"
    assert grasps.details["backend_tool"] == "anygrasp"
    assert grasps.details["observation"] == segmented.details["observation"]
    assert grasps.details["selected_detection"]["id"] == "detection_000"
    assert grasps.details["inspection_surface"] == "viser"
    assert grasps.details["static_pose_rendering"] is False
    assert "visualization" not in grasps.details
    assert not any(
        item.get("type") == "anygrasp_candidate_overlay"
        for item in grasps.details.get("artifacts", [])
        if isinstance(item, dict)
    )


def test_observation_bound_pipeline_rejects_stale_segmentation(tmp_path: Path) -> None:
    record = _episode_frame(tmp_path)
    perception = _make_perception(tmp_path)
    assert perception.segment_object(record, "black bowl").success is True

    newer_record = _episode_frame(tmp_path, observation_id="obs-000002")
    result = perception.propose_grasps(newer_record)

    assert result.success is False
    assert result.details["reason"] == "stale_segmentation"


def test_multiple_masks_require_explicit_object_selection(tmp_path: Path) -> None:
    record = _episode_frame(tmp_path)
    mask = Image.new("L", (96, 96), 255)
    mask_path = tmp_path / "mask-source.png"
    mask.save(mask_path)
    perception = _make_perception(tmp_path, sam3_response=_sam3_response(mask_path, count=2))

    assert perception.segment_object(record, "bowl").success is True
    result = perception.propose_grasps(record)

    assert result.success is False
    assert result.details["reason"] == "object_selection_required"


def test_target_mask_removes_sparse_depth_outliers_before_anygrasp(
    tmp_path: Path,
) -> None:
    record = _episode_frame(tmp_path)
    depth_path = tmp_path / "media" / "frames" / "depth.png"
    depth = np.full((96, 96), 1000, dtype=np.uint16)
    depth[30:66, 30:66] = 1200
    depth[30:32, 30:32] = 900
    depth[65, 65] = 1400
    Image.fromarray(depth, mode="I;16").save(depth_path)

    mask = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(mask).rectangle((30, 30, 65, 65), fill=255)
    mask_path = tmp_path / "mask-source.png"
    mask.save(mask_path)
    captured: dict[str, np.ndarray] = {}

    def sam3(_request: dict) -> dict:
        return _sam3_response(mask_path)

    def anygrasp(request: dict) -> dict:
        captured["mask"] = np.asarray(
            Image.open(BytesIO(base64.b64decode(request["target_mask"]["base64"])))
        )
        return _anygrasp_response()

    perception = ObservationBoundPerception(
        artifact_root=tmp_path,
        sam3=sam3,
        anygrasp=anygrasp,
        output_root=tmp_path / "perception",
    )
    assert perception.segment_object(record, "black bowl").success is True

    result = perception.propose_grasps(record)

    assert result.success is True
    cleaned = captured["mask"] > 0
    assert cleaned[30:32, 30:32].sum() == 0
    assert cleaned[65, 65] == 0
    assert int(cleaned.sum()) == 1291
    diagnostics = result.details["depth_coherence"]
    assert diagnostics["cluster_count"] == 3
    assert diagnostics["removed_pixel_count"] == 5
    selected = result.details["selected_detection"]
    assert selected["mask_ref"] != selected["original_mask_ref"]
    assert Path(selected["mask_ref"]).is_file()


def test_grasp_decision_board_limits_top_k_and_projects_wrist_view(tmp_path: Path) -> None:
    primary = tmp_path / "primary.png"
    wrist = tmp_path / "wrist.png"
    Image.new("RGB", (96, 96), (40, 50, 60)).save(primary)
    Image.new("RGB", (96, 96), (60, 50, 40)).save(wrist)
    candidates = [
        {
            "id": f"grasp_{index:03d}",
            "rank": index,
            "score": 1.0 - index * 0.1,
            "translation_xyz": [0.0, 0.0, 1.0],
            "gripper_tip_position_xyz": [0.05, 0.0, 1.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "width": 0.08,
        }
        for index in range(6)
    ]
    overlay, visuals = render_anygrasp_candidates(
        primary,
        candidates,
        intrinsics={"fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 48.0},
        output_dir=tmp_path / "rendered",
        camera_extrinsics={
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "camera_frame": "opencv",
        },
        wrist_rgb_path=wrist,
        wrist_intrinsics={"fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 48.0},
        wrist_extrinsics={
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "camera_frame": "opencv",
        },
        top_k=2,
        selected_id="grasp_004",
    )

    assert overlay.is_file()
    assert Image.open(overlay).size[0] > 96
    visible = {item["id"] for item in visuals if item["visible"]}
    assert visible == {"grasp_000", "grasp_001", "grasp_004"}
    assert all(item["wrist_projected"] for item in visuals if item["visible"])
    selected = next(item for item in visuals if item["id"] == "grasp_004")
    assert selected["selected"] is True

    _, default_visuals = render_anygrasp_candidates(
        primary,
        candidates,
        intrinsics={"fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 48.0},
        output_dir=tmp_path / "default-rendered",
    )
    assert [item["id"] for item in default_visuals if item["visible"]] == ["grasp_000"]

    _, focused_visuals = render_anygrasp_candidates(
        primary,
        candidates,
        intrinsics={"fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 48.0},
        output_dir=tmp_path / "focused-rendered",
        focus_id="grasp_004",
    )
    focused = [item for item in focused_visuals if item["visible"]]
    assert [item["id"] for item in focused] == ["grasp_004"]
    assert focused[0]["display_label"] == "G4"
    assert focused[0]["focused"] is True
    assert focused[0]["selected"] is False


def test_grasp_board_ranks_score_when_rank_missing_and_marks_wrist_offscreen(tmp_path: Path) -> None:
    primary = tmp_path / "primary.png"
    wrist = tmp_path / "wrist.png"
    Image.new("RGB", (96, 96), (40, 50, 60)).save(primary)
    Image.new("RGB", (96, 96), (60, 50, 40)).save(wrist)
    candidates = [
        {
            "id": "low",
            "score": 0.1,
            "translation_xyz": [0.0, 0.0, 1.0],
            "gripper_tip_position_xyz": [0.0, 0.0, 1.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "width": 0.04,
        },
        {
            "id": "high",
            "score": 0.9,
            "translation_xyz": [0.05, 0.0, 1.0],
            "gripper_tip_position_xyz": [0.05, 0.0, 1.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "width": 0.04,
        },
        {
            "id": "middle",
            "score": 0.5,
            "translation_xyz": [-0.05, 0.0, 1.0],
            "gripper_tip_position_xyz": [-0.05, 0.0, 1.0],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "width": 0.04,
        },
    ]
    overlay, visuals = render_anygrasp_candidates(
        primary,
        candidates,
        intrinsics={"fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 48.0},
        output_dir=tmp_path / "rendered",
        camera_extrinsics={
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "camera_frame": "opencv",
        },
        wrist_rgb_path=wrist,
        wrist_intrinsics={"fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 48.0},
        wrist_extrinsics={
            "pos": [1.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "camera_frame": "opencv",
        },
        top_k=2,
    )

    assert overlay.is_file()
    assert (tmp_path / "rendered" / "wrist_topk.png").is_file()
    visible = [item for item in visuals if item["visible"]]
    assert [item["id"] for item in visible] == ["high", "middle"]
    assert [item["display_label"] for item in visible] == ["G0", "G1"]
    assert all(item["wrist_projected"] for item in visible)
    assert all(not item["wrist_in_frame"] for item in visible)


def test_grasp_board_projects_registered_world_frame_panda_pose(tmp_path: Path) -> None:
    primary = tmp_path / "primary.png"
    Image.new("RGB", (96, 96), (40, 50, 60)).save(primary)
    candidate = {
        "id": "refined_001",
        "rank": 10,
        "pose_frame": "world",
        "eef_frame": "panda_grip_site",
        "transform_world_from_grip_site": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "width": 0.08,
        "source_backend": "refined",
    }

    overlay, visuals = render_anygrasp_candidates(
        primary,
        [candidate],
        intrinsics={"fx": 80.0, "fy": 80.0, "cx": 48.0, "cy": 48.0},
        output_dir=tmp_path / "world-rendered",
        camera_extrinsics={
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "camera_frame": "opencv",
        },
        focus_id="refined_001",
    )

    assert overlay.is_file()
    assert visuals[0]["projected"] is True
    assert visuals[0]["in_frame"] is True
    assert visuals[0]["center_px"] == [48, 48]
