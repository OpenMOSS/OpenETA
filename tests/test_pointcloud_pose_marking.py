from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tools.pointcloud_pose_marking import (
    AXIS_COLORS,
    VIEW_SPECS,
    _font,
    _layout_metric_tick_labels,
    annotate_active_grip_site_views,
    annotate_pending_constraint_views,
    annotate_views,
    camera_ray_from_image_click,
    mark_world_point,
    pose_from_points,
    projected_camera_ray_segments,
    project_world_to_view,
    render_move_feedback_crop,
    render_move_feedback_local_cloud,
    render_pose_candidate_frame_inspection,
    render_pose_candidate_frame_preview_views,
    render_pointcloud_contact_sheet,
    render_observation_pointcloud_views,
    render_world_pointcloud_views,
    solve_camera_ray_constraint,
    voxel_fuse_pointcloud,
    workspace_pointcloud,
    operator_scene_bounds,
    operator_scene_lookat,
    world_bounds_including_points,
    world_bounds_from_points,
    world_pointcloud_from_record,
)


def _record(root: Path) -> dict:
    rgb = root / "rgb.png"
    depth = root / "depth.png"
    Image.new("RGB", (8, 8), (80, 100, 120)).save(rgb)
    Image.fromarray(np.full((8, 8), 1000, dtype=np.uint16), mode="I;16").save(depth)
    return {
        "observation_id": "obs-test",
        "frames": [{
            "camera_id": "agentview",
            "frame_id": "frame-agentview",
            "rgb_path": str(rgb),
            "depth_path": str(depth),
            "metadata": {
                "intrinsics": {
                    "fx": 8.0, "fy": 8.0, "cx": 4.0, "cy": 4.0,
                    "scale": 1000.0,
                },
                "extrinsics": {
                    "pos": [0.0, 0.0, 0.0],
                    "mat": [
                        1.0, 0.0, 0.0,
                        0.0, 1.0, 0.0,
                        0.0, 0.0, 1.0,
                    ],
                    "frame_transform": "camera_to_world",
                    "camera_frame": "opencv",
                },
            },
        }],
    }


def _views(tmp_path: Path, *, image_size: int = 128):
    return render_observation_pointcloud_views(
        _record(tmp_path),
        artifact_root=tmp_path,
        output_root=tmp_path / "views",
        image_size=image_size,
    )


def _solve(views, target: np.ndarray, first_name: str, second_name: str):
    first, second = views[first_name], views[second_name]
    u1, v1 = project_world_to_view(first, target)
    point, pending = mark_world_point(first, u=u1, v=v1)
    assert point is None
    assert pending["status"] == "pending"
    u2, v2 = project_world_to_view(second, target)
    return mark_world_point(
        second,
        u=u2,
        v=v2,
        pending_constraint=pending["pending_constraint"],
    )


def test_contact_sheet_uses_clean_metric_views(tmp_path: Path) -> None:
    views = _views(tmp_path, image_size=32)
    for index, name in enumerate(
        ("pointcloud_top", "pointcloud_front", "pointcloud_side")
    ):
        view = views[name]
        Image.new("RGB", (32, 32), (250, 0, index)).save(view.image_path)
        assert view.clean_image_path is not None
        Image.new("RGB", (32, 32), (0, 250, index)).save(
            view.clean_image_path
        )

    sheet = np.asarray(
        Image.open(
            render_pointcloud_contact_sheet(
                views,
                output_root=tmp_path / "sheet",
            )
        ).convert("RGB")
    )

    assert sheet[10, 10].tolist() == [0, 250, 0]
    assert sheet[10, 42].tolist() == [0, 250, 1]
    assert sheet[42, 10].tolist() == [0, 250, 2]


def test_move_feedback_crop_magnifies_target_actual_without_history_overlay(
    tmp_path: Path,
) -> None:
    view = _views(tmp_path, image_size=128)["pointcloud_front"]
    assert view.clean_image_path is not None
    # Simulate a heavily annotated persistent source. The feedback artifact
    # must ignore it and use the clean calibrated image.
    Image.new("RGB", (128, 128), (255, 0, 0)).save(view.image_path)
    output = render_move_feedback_crop(
        view,
        output_path=tmp_path / "move_feedback_front.png",
        target_position_xyz_m=[0.0, 0.0, 0.98],
        actual_position_xyz_m=[0.0, 0.0, 1.00],
        actual_pad_contact_centers_world_m=[
            [-0.02, 0.0, 1.00],
            [0.02, 0.0, 1.00],
        ],
    )

    image = np.asarray(Image.open(output).convert("RGB"), dtype=np.uint8)
    assert image.shape == (384, 384, 3)
    green = (
        (image[..., 0] < 100)
        & (image[..., 1] > 220)
        & (image[..., 2] < 160)
    )
    magenta = (
        (image[..., 0] > 220)
        & (image[..., 1] < 80)
        & (image[..., 2] > 220)
    )
    assert np.any(green)
    assert np.any(magenta)
    green_y = float(np.mean(np.nonzero(green)[0]))
    magenta_y = float(np.mean(np.nonzero(magenta)[0]))
    # World +Z is screen up. The actual point at z=1.00 must be clearly above
    # the target at z=0.98 after local magnification.
    assert green_y - magenta_y > 10.0
    # The synthetic solid-red historical overlay was not used as the source.
    solid_red = (
        (image[..., 0] > 250)
        & (image[..., 1] < 5)
        & (image[..., 2] < 5)
    )
    assert np.count_nonzero(solid_red) < image.shape[0] * image.shape[1] // 20


def test_compact_move_feedback_crop_renders_actual_pad_boxes_without_labels(
    tmp_path: Path,
) -> None:
    view = _views(tmp_path, image_size=128)["pointcloud_front"]
    assert view.clean_image_path is not None
    background = (18, 28, 38)
    Image.new("RGB", (128, 128), background).save(view.clean_image_path)

    output = render_move_feedback_crop(
        view,
        output_path=tmp_path / "move_feedback_front_compact.png",
        target_position_xyz_m=[0.0, 0.0, 0.98],
        actual_position_xyz_m=[0.0, 0.0, 1.00],
        actual_pad_contact_centers_world_m=[
            [-0.03, 0.0, 1.00],
            [0.03, 0.0, 1.00],
        ],
        actual_pad_boxes=[
            {
                "center_world_m": [-0.038, 0.0, 1.00],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.012, 0.004, 0.012],
            },
            {
                "center_world_m": [0.038, 0.0, 1.00],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.012, 0.004, 0.012],
            },
        ],
        compact_actual_pad_geometry=True,
    )

    image = np.asarray(Image.open(output).convert("RGB"), dtype=np.uint8)
    assert image.shape == (384, 384, 3)
    orange = (
        (image[..., 0] > 240)
        & (image[..., 1] > 70)
        & (image[..., 1] < 150)
        & (image[..., 2] < 80)
    )
    orange_y, orange_x = np.nonzero(orange)
    assert len(orange_x) > 40
    # Real projected pad boxes occupy two spatially separated regions rather
    # than two fixed-size glyphs at the contact-center pixels.
    assert float(np.max(orange_x) - np.min(orange_x)) > 45.0
    assert float(np.max(orange_y) - np.min(orange_y)) > 15.0
    # Compact mode has no duplicated local axis or text in the upper-left.
    assert np.all(image[:45, :90] == np.asarray(background, dtype=np.uint8))


def test_move_feedback_contact_corridor_connects_actual_inner_pad_faces(
    tmp_path: Path,
) -> None:
    view = _views(tmp_path, image_size=128)["pointcloud_front"]
    assert view.clean_image_path is not None
    background = np.asarray((18, 28, 38), dtype=np.uint8)
    Image.new("RGB", (128, 128), tuple(background)).save(
        view.clean_image_path
    )

    output = render_move_feedback_crop(
        view,
        output_path=tmp_path / "move_feedback_front_corridor.png",
        target_position_xyz_m=[0.0, 0.0, 0.98],
        actual_position_xyz_m=[0.0, 0.0, 1.00],
        actual_pad_contact_centers_world_m=[
            [-0.03, 0.0, 1.00],
            [0.03, 0.0, 1.00],
        ],
        actual_pad_boxes=[
            {
                "center_world_m": [-0.034, 0.0, 1.00],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.004, 0.008, 0.012],
            },
            {
                "center_world_m": [0.034, 0.0, 1.00],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.004, 0.008, 0.012],
            },
        ],
        compact_actual_pad_geometry=True,
        render_actual_pad_contact_corridor=True,
    )

    image = np.asarray(Image.open(output).convert("RGB"), dtype=np.uint8)
    pale = (
        (image[..., 0] > 240)
        & (image[..., 1] > 190)
        & (image[..., 2] > 70)
    )
    pale_y, pale_x = np.nonzero(pale)
    assert len(pale_x) > 40
    # The contact visualization spans the gap between the two live inner
    # faces, rather than drawing only disconnected finger boxes.
    assert float(np.max(pale_x) - np.min(pale_x)) > 25.0
    assert float(np.max(pale_y) - np.min(pale_y)) > 20.0
    corridor_edge = (
        (image[..., 0] > 240)
        & (image[..., 1] > 180)
        & (image[..., 2] < 120)
    )
    edge_y, edge_x = np.nonzero(corridor_edge)
    assert len(edge_x) > 40
    assert float(np.max(edge_x) - np.min(edge_x)) > 25.0
    assert float(np.max(edge_y) - np.min(edge_y)) > 20.0
    # The corridor is an outline only; it must not obscure the point cloud
    # with an additional translucent fill.
    dark_fill = (
        (image[..., 0] > background[0])
        & (image[..., 0] < 100)
        & (image[..., 1] > background[1])
        & (image[..., 1] < 100)
        & (image[..., 2] > background[2])
        & (image[..., 2] < 100)
    )
    assert np.count_nonzero(dark_fill) == 0


def test_move_feedback_closing_corridor_adds_inward_motion_cues(
    tmp_path: Path,
) -> None:
    view = _views(tmp_path, image_size=128)["pointcloud_front"]
    assert view.clean_image_path is not None
    Image.new("RGB", (128, 128), (18, 28, 38)).save(
        view.clean_image_path
    )

    output = render_move_feedback_crop(
        view,
        output_path=tmp_path / "move_feedback_front_closing_corridor.png",
        target_position_xyz_m=[0.0, 0.0, 0.98],
        actual_position_xyz_m=[0.0, 0.0, 1.00],
        actual_pad_contact_centers_world_m=[
            [-0.03, 0.0, 1.00],
            [0.03, 0.0, 1.00],
        ],
        actual_pad_boxes=[
            {
                "center_world_m": [-0.034, 0.0, 1.00],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.004, 0.008, 0.012],
            },
            {
                "center_world_m": [0.034, 0.0, 1.00],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.004, 0.008, 0.012],
            },
        ],
        compact_actual_pad_geometry=True,
        render_actual_pad_contact_corridor=True,
        render_actual_pad_closing_cues=True,
    )

    image = np.asarray(Image.open(output).convert("RGB"), dtype=np.uint8)
    closing_cue = (
        (image[..., 0] == 255)
        & (image[..., 1] == 245)
        & (image[..., 2] == 120)
    )
    cue_y, cue_x = np.nonzero(closing_cue)
    assert len(cue_x) > 30
    assert float(np.max(cue_x) - np.min(cue_x)) > 25.0
    center_x = int(round(float(np.median(cue_x))))
    assert np.count_nonzero(closing_cue[:, center_x]) == 0
    # Each filled arrowhead narrows toward the corridor center. This checks
    # direction rather than depending on a particular gap size or crop scale.
    left_counts = np.count_nonzero(
        closing_cue[:, :center_x],
        axis=0,
    )
    right_counts = np.count_nonzero(
        closing_cue[:, center_x + 1 :],
        axis=0,
    )
    assert left_counts[np.nonzero(left_counts)[0][-1]] < np.max(left_counts)
    assert right_counts[np.nonzero(right_counts)[0][0]] < np.max(right_counts)


def test_move_feedback_local_cloud_excludes_distant_depth_layer(
    tmp_path: Path,
) -> None:
    view = _views(tmp_path, image_size=128)["pointcloud_front"]
    local_object = np.asarray(
        [
            [x, -0.01, z]
            for x in np.linspace(-0.025, 0.025, 25)
            for z in np.linspace(0.97, 1.01, 20)
        ],
        dtype=np.float64,
    )
    distant_occluder = np.asarray(
        [
            [x, -0.35, z]
            for x in np.linspace(-0.06, 0.06, 30)
            for z in np.linspace(0.94, 1.04, 30)
        ],
        dtype=np.float64,
    )
    points = np.concatenate((local_object, distant_occluder), axis=0)
    colors = np.concatenate(
        (
            np.tile(np.asarray([[30, 210, 70]], dtype=np.uint8), (len(local_object), 1)),
            np.tile(np.asarray([[230, 30, 30]], dtype=np.uint8), (len(distant_occluder), 1)),
        ),
        axis=0,
    )

    output = render_move_feedback_local_cloud(
        view,
        output_path=tmp_path / "move_feedback_local.png",
        world_points=points,
        colors_rgb=colors,
        target_position_xyz_m=[0.0, 0.0, 0.98],
        actual_position_xyz_m=[0.0, 0.0, 1.00],
        actual_pad_contact_centers_world_m=[
            [-0.03, 0.0, 1.00],
            [0.03, 0.0, 1.00],
        ],
        actual_pad_boxes=[
            {
                "center_world_m": [-0.038, 0.0, 1.00],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.008, 0.004, 0.008],
            },
            {
                "center_world_m": [0.038, 0.0, 1.00],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.008, 0.004, 0.008],
            },
        ],
        hidden_margin_m=0.08,
    )

    local_source = output.with_name("move_feedback_local.local-cloud.png")
    local_metadata = output.with_name(
        "move_feedback_local.local-cloud.json"
    )
    image = np.asarray(Image.open(local_source).convert("RGB"), dtype=np.uint8)
    green_scene = (
        (image[..., 0] < 120)
        & (image[..., 1] > 150)
        & (image[..., 2] < 150)
    )
    red_scene = (
        (image[..., 0] > 180)
        & (image[..., 1] < 120)
        & (image[..., 2] < 120)
    )
    assert np.count_nonzero(green_scene) > 100
    assert np.count_nonzero(red_scene) == 0
    assert output.is_file()
    metadata = json.loads(local_metadata.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == (
        "openeta.move_feedback_local_cloud.v1"
    )
    assert metadata["source_view"] == "pointcloud_front"
    assert metadata["horizontal_axis"] == "x"
    assert metadata["vertical_axis"] == "z"
    assert metadata["hidden_axis"] == "y"
    assert metadata["actual_pad_contact_centers_world_m"] == [
        [-0.03, 0.0, 1.0],
        [0.03, 0.0, 1.0],
    ]


def test_candidate_frame_inspection_renders_two_cross_sections_and_pad_boxes(
    tmp_path: Path,
) -> None:
    views = _views(tmp_path, image_size=64)
    output = render_pose_candidate_frame_inspection(
        views,
        output_path=tmp_path / "candidate-frame.png",
        target_position_xyz_m=[0.0, 0.0, 1.0],
        target_rotation_matrix=np.eye(3),
        target_pad_boxes=[
            {
                "center_world_m": [0.025, 0.0, 1.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.01, 0.015, 0.02],
            },
            {
                "center_world_m": [-0.025, 0.0, 1.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.01, 0.015, 0.02],
            },
        ],
        size=192,
    )

    assert output == tmp_path / "candidate-frame.png"
    assert output.is_file()
    image = np.asarray(Image.open(output).convert("RGB"))
    assert image.shape == (192, 384, 3)
    # Both candidate-frame panels contain the orange grip-site glyph and
    # projected pad collision boxes.
    for panel in (image[:, :192], image[:, 192:]):
        orange = (
            (panel[:, :, 0] > 220)
            & (panel[:, :, 1] > 90)
            & (panel[:, :, 1] < 190)
            & (panel[:, :, 2] < 80)
        )
        assert int(np.count_nonzero(orange)) >= 30


def test_candidate_frame_preview_returns_two_clean_rgb_views(
    tmp_path: Path,
) -> None:
    local_points = np.asarray(
        [
            [-0.03, 0.00, -0.02],
            [0.00, 0.00, 0.00],
            [0.03, 0.00, 0.02],
            [0.00, 0.03, 0.01],
        ],
        dtype=np.float64,
    )
    colors = np.asarray(
        [
            [210, 30, 20],
            [20, 210, 30],
            [20, 30, 210],
            [210, 180, 20],
        ],
        dtype=np.uint8,
    )
    paths = render_pose_candidate_frame_preview_views(
        local_points + np.asarray([0.1, -0.2, 1.0]),
        colors,
        output_root=tmp_path / "preview",
        target_position_xyz_m=[0.1, -0.2, 1.0],
        target_rotation_matrix=np.eye(3),
        target_pad_contact_centers_world_m=[
            [0.075, -0.2, 1.0],
            [0.125, -0.2, 1.0],
        ],
        target_pad_sweep_start_centers_world_m=[
            [0.055, -0.2, 1.0],
            [0.145, -0.2, 1.0],
        ],
        target_pad_boxes=[
            {
                "center_world_m": [0.075, -0.2, 1.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.008, 0.012, 0.02],
            },
            {
                "center_world_m": [0.125, -0.2, 1.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.008, 0.012, 0.02],
            },
        ],
        size=192,
    )

    assert set(paths) == {"pointcloud_front", "pointcloud_side"}
    for path in paths.values():
        image = np.asarray(Image.open(path).convert("RGB"))
        assert image.shape == (192, 192, 3)
        orange = (
            (image[..., 0] > 220)
            & (image[..., 1] > 90)
            & (image[..., 1] < 225)
            & (image[..., 2] < 110)
        )
        assert int(np.count_nonzero(orange)) >= 30
        # The treatment keeps the observed RGB hue instead of reducing the
        # cloud to grayscale.
        colored_scene = (
            np.max(image, axis=2).astype(int)
            - np.min(image, axis=2).astype(int)
        ) > 40
        assert int(np.count_nonzero(colored_scene)) > 20


def test_candidate_frame_preview_semantic_axes_are_versioned(
    tmp_path: Path,
) -> None:
    points = np.asarray(
        [[-0.02, 0.0, 0.0], [0.02, 0.0, 0.01]],
        dtype=np.float64,
    )
    colors = np.asarray([[220, 40, 30], [30, 220, 40]], dtype=np.uint8)

    legacy = render_pose_candidate_frame_preview_views(
        points,
        colors,
        output_root=tmp_path / "legacy",
        target_position_xyz_m=[0.0, 0.0, 0.0],
        target_rotation_matrix=np.eye(3),
        size=128,
    )
    semantic = render_pose_candidate_frame_preview_views(
        points,
        colors,
        output_root=tmp_path / "semantic",
        target_position_xyz_m=[0.0, 0.0, 0.0],
        target_rotation_matrix=np.eye(3),
        size=128,
        axis_label_semantics="jaw_lat_app_v2",
    )

    assert legacy["pointcloud_front"].name != semantic["pointcloud_front"].name
    legacy_metadata = json.loads(
        next((tmp_path / "legacy").glob("candidate-frame.*.json")).read_text()
    )
    semantic_metadata = json.loads(
        next((tmp_path / "semantic").glob("candidate-frame.*.json")).read_text()
    )
    assert legacy_metadata["schema_version"].endswith(".v1")
    assert legacy_metadata["views"]["pointcloud_front"] == {
        "screen_right": "+X jaw",
        "screen_up": "+Z approach",
        "hidden_axis": "Y",
    }
    assert semantic_metadata["schema_version"].endswith(".v2")
    assert semantic_metadata["views"] == {
        "pointcloud_front": {
            "screen_right": "+JAW",
            "screen_up": "+APP",
            "hidden_axis": "LAT",
        },
        "pointcloud_side": {
            "screen_right": "+JAW",
            "screen_up": "+LAT",
            "hidden_axis": "APP",
        },
    }
    assert semantic_metadata["candidate_axes_world"] == {
        "JAW": [1.0, 0.0, 0.0],
        "LAT": [0.0, 1.0, 0.0],
        "APP": [0.0, 0.0, 1.0],
    }


def test_candidate_frame_preview_can_publish_mark_point_mapping(
    tmp_path: Path,
) -> None:
    render_pose_candidate_frame_preview_views(
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0, 3), dtype=np.uint8),
        output_root=tmp_path,
        target_position_xyz_m=[0.1, -0.2, 0.3],
        target_rotation_matrix=np.eye(3),
        axis_label_semantics="jaw_lat_app_v2",
        size=128,
        markable=True,
    )

    metadata = json.loads(
        next(tmp_path.glob("candidate-frame.*.json")).read_text()
    )
    assert metadata["mark_point_mapping"] == (
        "candidate_frame_two_view_v1"
    )
    assert metadata["candidate_origin_world_m"] == [0.1, -0.2, 0.3]
    assert metadata["candidate_rotation_world"] == np.eye(3).tolist()
    assert metadata["image_size_xy"] == [128, 128]


def test_candidate_frame_preview_v3_places_positive_app_screen_down(
    tmp_path: Path,
) -> None:
    paths = render_pose_candidate_frame_preview_views(
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0, 3), dtype=np.uint8),
        output_root=tmp_path,
        target_position_xyz_m=[0.0, 0.0, 0.0],
        target_rotation_matrix=np.eye(3),
        axis_label_semantics="jaw_lat_app_v3",
        size=128,
    )

    metadata = json.loads(
        next(tmp_path.glob("candidate-frame.*.json")).read_text()
    )
    assert metadata["schema_version"].endswith(".v3")
    assert metadata["views"]["pointcloud_front"] == {
        "screen_right": "+JAW",
        "screen_up": "-APP",
        "screen_down": "+APP",
        "hidden_axis": "LAT",
    }
    image = np.asarray(Image.open(paths["pointcloud_front"]).convert("RGB"))
    # Positive APP is candidate-local +Z. In v3 it maps to increasing image
    # rows, i.e. screen-down, matching the physical top-down approach cue.
    blue = (
        (image[..., 0] < 120)
        & (image[..., 1] > 80)
        & (image[..., 2] > 170)
    )
    ys, _xs = np.nonzero(blue)
    assert len(ys) > 0
    assert int(ys.max()) > int(ys.min())


def test_candidate_frame_visual_only_renderer_changes_pixels_not_slots(
    tmp_path: Path,
) -> None:
    points = np.asarray(
        [
            [-0.035, 0.0, -0.015],
            [-0.015, 0.0, 0.0],
            [0.015, 0.0, 0.0],
            [0.035, 0.0, 0.015],
        ],
        dtype=np.float64,
    )
    colors = np.asarray(
        [
            [190, 50, 30],
            [30, 180, 50],
            [30, 50, 190],
            [190, 160, 30],
        ],
        dtype=np.uint8,
    )
    common = {
        "world_points": points,
        "colors_rgb": colors,
        "target_position_xyz_m": [0.0, 0.0, 0.0],
        "target_rotation_matrix": np.eye(3),
        "target_pad_boxes": [
            {
                "center_world_m": [-0.012, 0.0, 0.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.006, 0.01, 0.012],
            },
            {
                "center_world_m": [0.012, 0.0, 0.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.006, 0.01, 0.012],
            },
        ],
        "target_pad_sweep_start_boxes": [
            {
                "center_world_m": [-0.04, 0.0, 0.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.006, 0.01, 0.012],
            },
            {
                "center_world_m": [0.04, 0.0, 0.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.006, 0.01, 0.012],
            },
        ],
        "size": 192,
        "axis_label_semantics": "jaw_lat_app_v2",
    }
    control = render_pose_candidate_frame_preview_views(
        output_root=tmp_path / "control",
        visual_mode="current_candidate_frame_v1",
        **common,
    )
    treatment = render_pose_candidate_frame_preview_views(
        output_root=tmp_path / "treatment",
        visual_mode="candidate_corridor_local_v2",
        **common,
    )

    assert set(control) == set(treatment) == {
        "pointcloud_front",
        "pointcloud_side",
    }
    for name in control:
        control_image = np.asarray(Image.open(control[name]).convert("RGB"))
        treatment_image = np.asarray(
            Image.open(treatment[name]).convert("RGB")
        )
        assert control_image.shape == treatment_image.shape == (192, 192, 3)
        assert not np.array_equal(control_image, treatment_image)

    control_metadata = json.loads(
        next((tmp_path / "control").glob("candidate-frame.*.json")).read_text()
    )
    treatment_metadata = json.loads(
        next(
            (tmp_path / "treatment").glob("candidate-frame.*.json")
        ).read_text()
    )
    assert control_metadata["visual_mode"] == "current_candidate_frame_v1"
    assert treatment_metadata["visual_mode"] == "candidate_corridor_local_v2"
    assert set(control_metadata["views"]) == set(treatment_metadata["views"])
    assert control_metadata["schema_version"] == treatment_metadata[
        "schema_version"
    ]


def test_candidate_frame_preview_is_rigid_transform_equivariant(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(7)
    local_points = rng.uniform(-0.07, 0.07, size=(500, 3))
    colors = rng.integers(20, 220, size=(500, 3), dtype=np.uint8)
    angle = np.deg2rad(37.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.asarray([0.18, -0.11, 0.92], dtype=np.float64)
    world_points = local_points @ rotation.T + target

    baseline = render_pose_candidate_frame_preview_views(
        local_points,
        colors,
        output_root=tmp_path / "baseline",
        target_position_xyz_m=[0.0, 0.0, 0.0],
        target_rotation_matrix=np.eye(3),
        size=160,
    )
    transformed = render_pose_candidate_frame_preview_views(
        world_points,
        colors,
        output_root=tmp_path / "transformed",
        target_position_xyz_m=target,
        target_rotation_matrix=rotation,
        size=160,
    )

    for name in ("pointcloud_front", "pointcloud_side"):
        first = np.asarray(Image.open(baseline[name]).convert("RGB"))
        second = np.asarray(Image.open(transformed[name]).convert("RGB"))
        assert np.array_equal(first, second)


def test_candidate_frame_preview_excludes_geometry_outside_pad_corridor(
    tmp_path: Path,
) -> None:
    target = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    # Green points sit inside the pad-centered hidden-axis corridor. Red
    # points represent a nearby background plane outside that corridor.
    inside = np.asarray(
        [[x, y, 0.015] for x in np.linspace(-0.04, 0.04, 9)
         for y in np.linspace(-0.04, 0.04, 9)],
        dtype=np.float64,
    )
    outside = np.asarray(
        [[x, y, 0.045] for x in np.linspace(-0.04, 0.04, 9)
         for y in np.linspace(-0.04, 0.04, 9)],
        dtype=np.float64,
    )
    paths = render_pose_candidate_frame_preview_views(
        np.concatenate([inside, outside]),
        np.concatenate(
            [
                np.tile([20, 220, 30], (len(inside), 1)),
                np.tile([230, 20, 20], (len(outside), 1)),
            ]
        ).astype(np.uint8),
        output_root=tmp_path / "corridor",
        target_position_xyz_m=target,
        target_rotation_matrix=np.eye(3),
        target_pad_boxes=[
            {
                "center_world_m": [-0.02, 0.0, 0.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.008, 0.004, 0.008],
            },
            {
                "center_world_m": [0.02, 0.0, 0.0],
                "rotation_world": np.eye(3).tolist(),
                "half_size_m": [0.008, 0.004, 0.008],
            },
        ],
        size=192,
    )

    side = np.asarray(Image.open(paths["pointcloud_side"]).convert("RGB"))
    green = (
        (side[..., 0] < 120)
        & (side[..., 1] > 180)
        & (side[..., 2] < 130)
    )
    red = (
        (side[..., 0] > 180)
        & (side[..., 1] < 130)
        & (side[..., 2] < 130)
    )
    assert int(np.count_nonzero(green)) > 100
    # Exclude the small lower-left X-axis glyph from the scene-color check.
    red[side.shape[0] - 55 :, :70] = False
    assert int(np.count_nonzero(red)) == 0


def test_candidate_frame_preview_can_render_full_pad_swept_footprint(
    tmp_path: Path,
) -> None:
    points = np.zeros((0, 3), dtype=np.float64)
    colors = np.zeros((0, 3), dtype=np.uint8)
    closed_boxes = [
        {
            "center_world_m": [-0.008, 0.0, 0.0],
            "rotation_world": np.eye(3).tolist(),
            "half_size_m": [0.006, 0.004, 0.008],
        },
        {
            "center_world_m": [0.008, 0.0, 0.0],
            "rotation_world": np.eye(3).tolist(),
            "half_size_m": [0.006, 0.004, 0.008],
        },
    ]
    open_boxes = [
        {**closed_boxes[0], "center_world_m": [-0.040, 0.0, 0.0]},
        {**closed_boxes[1], "center_world_m": [0.040, 0.0, 0.0]},
    ]
    common = {
        "world_points": points,
        "colors_rgb": colors,
        "target_position_xyz_m": [0.0, 0.0, 0.0],
        "target_rotation_matrix": np.eye(3),
        "target_pad_boxes": closed_boxes,
        "size": 192,
    }
    control = render_pose_candidate_frame_preview_views(
        output_root=tmp_path / "control",
        **common,
    )
    treatment = render_pose_candidate_frame_preview_views(
        output_root=tmp_path / "treatment",
        target_pad_sweep_start_boxes=open_boxes,
        **common,
    )

    control_front = np.asarray(
        Image.open(control["pointcloud_front"]).convert("RGB")
    )
    treatment_front = np.asarray(
        Image.open(treatment["pointcloud_front"]).convert("RGB")
    )
    # This pixel is inside the left pad's open-to-close swept volume, but
    # outside both endpoint pad boxes and away from a 20 mm grid line.
    sample_u = int(round((-0.027 + 0.06) / 0.12 * 191))
    sample_v = int(round((0.06 - 0.005) / 0.12 * 191))
    assert treatment_front[sample_v, sample_u, 0] > (
        control_front[sample_v, sample_u, 0] + 20
    )
    assert not np.array_equal(control_front, treatment_front)
    metadata = json.loads(
        next(
            (tmp_path / "treatment").glob("candidate-frame.*.json")
        ).read_text()
    )
    assert metadata["closing_sweep_geometry"] == (
        "pad_collision_box_swept_volume_v1"
    )


def test_candidate_frame_preview_can_render_open_pad_capture_corridor(
    tmp_path: Path,
) -> None:
    points = np.zeros((0, 3), dtype=np.float64)
    colors = np.zeros((0, 3), dtype=np.uint8)
    closed_boxes = [
        {
            "center_world_m": [-0.012, 0.0, 0.0],
            "rotation_world": np.eye(3).tolist(),
            "half_size_m": [0.006, 0.010, 0.012],
        },
        {
            "center_world_m": [0.012, 0.0, 0.0],
            "rotation_world": np.eye(3).tolist(),
            "half_size_m": [0.006, 0.010, 0.012],
        },
    ]
    open_boxes = [
        {**closed_boxes[0], "center_world_m": [-0.044, 0.0, 0.0]},
        {**closed_boxes[1], "center_world_m": [0.044, 0.0, 0.0]},
    ]
    open_contacts = [
        [-0.038, 0.0, 0.0],
        [0.038, 0.0, 0.0],
    ]
    common = {
        "world_points": points,
        "colors_rgb": colors,
        "target_position_xyz_m": [0.0, 0.0, 0.0],
        "target_rotation_matrix": np.eye(3),
        "target_pad_boxes": closed_boxes,
        "target_pad_sweep_start_centers_world_m": open_contacts,
        "size": 192,
    }
    control = render_pose_candidate_frame_preview_views(
        output_root=tmp_path / "control-corridor",
        **common,
    )
    treatment = render_pose_candidate_frame_preview_views(
        output_root=tmp_path / "treatment-corridor",
        target_pad_capture_corridor_boxes=open_boxes,
        **common,
    )

    control_front = np.asarray(
        Image.open(control["pointcloud_front"]).convert("RGB")
    )
    treatment_front = np.asarray(
        Image.open(treatment["pointcloud_front"]).convert("RGB")
    )
    pale = (
        (treatment_front[..., 0] > 240)
        & (treatment_front[..., 1] > 180)
        & (treatment_front[..., 2] < 150)
    )
    pale_y, pale_x = np.nonzero(pale)
    assert len(pale_x) > 50
    assert float(np.max(pale_x) - np.min(pale_x)) > 90.0
    assert float(np.max(pale_y) - np.min(pale_y)) > 25.0
    assert not np.array_equal(control_front, treatment_front)
    metadata = json.loads(
        next(
            (tmp_path / "treatment-corridor").glob(
                "candidate-frame.*.json"
            )
        ).read_text()
    )
    assert metadata["closing_corridor_geometry"] == (
        "open_inner_pad_faces_capture_corridor_v1"
    )


def test_every_point_starts_as_two_axis_pending_constraint(tmp_path: Path) -> None:
    views = _views(tmp_path, image_size=32)
    payload = np.load(views["pointcloud_top"].lookup_path)
    assert np.asarray(payload["valid"]).any()
    point, source = mark_world_point(views["pointcloud_top"], u=16, v=16)
    assert point is None
    assert source["status"] == "pending"
    assert source["needs_complementary_view"] == [
        "pointcloud_front",
        "pointcloud_side",
    ]
    assert source["pending_constraint"]["fixed_axes"] == ["x", "y"]


def test_agentview_click_defines_ray_without_reading_depth(tmp_path: Path) -> None:
    record = _record(tmp_path)
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=4,
        v=4,
    )

    assert ray["source_kind"] == "agentview_camera_ray"
    assert ray["pixel_xy"] == [4, 4]
    assert ray["origin_xyz_m"] == pytest.approx([0.0, 0.0, 0.0])
    assert ray["direction_xyz"] == pytest.approx([0.0, 0.0, 1.0])
    assert ray["workspace_t_range_m"] == pytest.approx([0.85, 1.35])


def test_scene_bounds_keep_object_suite_scale_and_dynamic_views(
    tmp_path: Path,
) -> None:
    points = np.asarray(
        [
            [-0.12, -0.24, 0.04],
            [-0.08, -0.20, 0.08],
            [-0.05, 0.28, 0.00],
            [0.02, 0.30, 0.16],
        ],
        dtype=np.float64,
    )
    colors = np.full((len(points), 3), 120, dtype=np.uint8)
    bounds = world_bounds_from_points(points, quantile=0.0, padding_m=0.01)
    cropped, cropped_colors = workspace_pointcloud(
        points, colors, bounds=bounds
    )
    assert cropped.shape == points.shape
    assert cropped_colors.shape == colors.shape

    record = {"observation_id": "obs-object-scale"}
    views = render_world_pointcloud_views(
        points,
        colors,
        observation_record=record,
        artifact_root=tmp_path,
        output_root=tmp_path / "object-views",
        image_size=64,
        world_bounds=bounds,
    )
    assert set(views) == {
        "pointcloud_top",
        "pointcloud_front",
        "pointcloud_side",
    }
    for view in views.values():
        metadata = json.loads(view.metadata_path.read_text())
        np.testing.assert_allclose(
            np.asarray(metadata["world_bounds_m"], dtype=np.float64),
            bounds,
        )
        assert np.asarray(np.load(view.lookup_path)["valid"]).any()


def test_operator_scene_bounds_ignores_distant_background_depth() -> None:
    points = np.asarray(
        [
            [-1.99, -1.10, 0.02],
            [0.15, 0.05, 0.07],
            [0.42, 1.10, 0.38],
            [-0.80, 0.80, 0.01],
        ],
        dtype=np.float64,
    )
    bounds = operator_scene_bounds(points)
    np.testing.assert_allclose(
        bounds,
        np.asarray([[-0.60, 0.60], [-0.60, 0.70], [-0.15, 0.60]]),
    )

    spatial_points = np.asarray(
        [[-0.2, -0.2, 0.90], [0.0, 0.0, 1.0], [0.2, 0.2, 1.1]],
        dtype=np.float64,
    )
    spatial_bounds = operator_scene_bounds(spatial_points)
    np.testing.assert_allclose(
        spatial_bounds,
        np.asarray([[-0.45, 0.35], [-0.35, 0.50], [0.85, 1.35]]),
    )


def test_world_bounds_can_include_current_grip_site_free_space() -> None:
    bounds = np.asarray(
        [[-0.60, 0.60], [-0.60, 0.70], [-0.15, 0.60]],
        dtype=np.float64,
    )
    expanded = world_bounds_including_points(
        bounds,
        [[0.07, 0.20, 0.708]],
        padding_m=0.10,
    )

    np.testing.assert_allclose(expanded[:2], bounds[:2])
    np.testing.assert_allclose(expanded[2], [-0.15, 0.808])


def test_operator_scene_lookat_tracks_scene_scale() -> None:
    object_points = np.vstack(
        [
            np.repeat([[0.0, 0.0, 0.0]], 100, axis=0),
            np.repeat([[0.2, 0.1, 0.12]], 20, axis=0),
        ]
    )
    object_bounds = operator_scene_bounds(object_points)
    object_lookat = operator_scene_lookat(object_points, bounds=object_bounds)
    assert object_lookat[2] < 0.20

    spatial_points = np.vstack(
        [
            np.repeat([[0.0, 0.0, 0.90]], 100, axis=0),
            np.repeat([[0.2, 0.1, 1.12]], 20, axis=0),
        ]
    )
    spatial_bounds = operator_scene_bounds(spatial_points)
    spatial_lookat = operator_scene_lookat(spatial_points, bounds=spatial_bounds)
    assert 0.85 <= spatial_lookat[2] <= 1.35


def test_camera_ray_accepts_scene_bounds_without_changing_default(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    object_bounds = np.asarray(
        [[-0.5, 0.5], [-0.5, 0.5], [-0.1, 0.3]],
        dtype=np.float64,
    )
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=4,
        v=4,
        world_bounds=object_bounds,
    )
    assert ray["workspace_t_range_m"] == pytest.approx([0.0, 0.3])


def test_agentview_visible_feature_rejects_a_different_depth_layer(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    views = _views(tmp_path, image_size=384)
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=4,
        v=4,
        artifact_root=tmp_path,
    )
    assert ray["visible_surface"]["ray_parameter_m"] == pytest.approx(1.0)

    farther = np.asarray(ray["origin_xyz_m"]) + 1.08 * np.asarray(
        ray["direction_xyz"]
    )
    front = views["pointcloud_front"]
    u, v = project_world_to_view(front, farther)
    point, result = mark_world_point(
        front,
        u=u,
        v=v,
        pending_constraint=ray,
        enforce_visible_surface_layer=True,
    )

    assert point is None
    assert result["status"] == "different_visible_depth_layer"
    assert result["visible_surface_delta_m"] == pytest.approx(0.08, abs=0.01)
    assert result["source_kind"] == "agentview_camera_ray"
    assert result["pending_constraint"] == ray
    assert result["visible_surface_layer_tolerance_m"] == pytest.approx(0.02)
    assert result["visible_surface_ray_pixel_xy"] == list(
        project_world_to_view(
            front,
            np.asarray(ray["visible_surface"]["xyz_m"]),
        )
    )


def test_agentview_visible_feature_solves_at_rgbd_surface_depth(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    views = _views(tmp_path, image_size=384)
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=4,
        v=4,
        artifact_root=tmp_path,
    )
    visible_xyz = np.asarray(ray["visible_surface"]["xyz_m"])
    front = views["pointcloud_front"]
    u, v = project_world_to_view(front, visible_xyz)

    point, result = mark_world_point(
        front,
        u=u,
        v=v,
        pending_constraint=ray,
        enforce_visible_surface_layer=True,
    )

    assert point is not None
    assert point == pytest.approx(visible_xyz, abs=0.01)
    assert result["status"] == "solved"
    assert abs(result["visible_surface_delta_m"]) <= 0.02
    assert result["source_kind"] == "agentview_ray_orthographic_intersection"


def test_agentview_ray_plus_orthographic_click_solves_depth(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    views = _views(tmp_path, image_size=128)
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=4,
        v=4,
    )
    target = np.asarray([0.0, 0.0, 1.0])
    u, v = project_world_to_view(views["pointcloud_front"], target)

    point, result = mark_world_point(
        views["pointcloud_front"],
        u=u,
        v=v,
        pending_constraint=ray,
    )

    assert result["status"] == "solved"
    assert result["source_kind"] == "agentview_ray_orthographic_intersection"
    assert point == pytest.approx(target, abs=0.01)


def test_agentview_ray_retry_reports_nearest_rendered_ray_pixel(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    views = _views(tmp_path, image_size=128)
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=4,
        v=4,
    )
    target = np.asarray([0.0, 0.0, 1.0])
    u, v = project_world_to_view(views["pointcloud_front"], target)
    point, result = mark_world_point(
        views["pointcloud_front"],
        u=u + 10,
        v=v,
        pending_constraint=ray,
    )

    assert point is None
    assert result["status"] == "inconsistent_views"
    nearest = result["nearest_ray_pixel_xy"]
    assert len(nearest) == 2
    assert all(isinstance(value, int) for value in nearest)
    assert 0 <= nearest[0] < views["pointcloud_front"].width
    assert 0 <= nearest[1] < views["pointcloud_front"].height


def test_projected_camera_ray_segments_expose_exact_clickable_line(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    views = _views(tmp_path, image_size=128)
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=4,
        v=4,
    )

    segments = projected_camera_ray_segments(views, ray)

    assert set(segments) == {
        "pointcloud_top",
        "pointcloud_front",
        "pointcloud_side",
    }
    assert segments["pointcloud_top"]["displayed_world_axes"] == ["x", "y"]
    assert segments["pointcloud_front"]["displayed_world_axes"] == ["x", "z"]
    assert segments["pointcloud_side"]["displayed_world_axes"] == ["y", "z"]
    for segment in segments.values():
        assert len(segment["pixel_start_xy"]) == 2
        assert len(segment["pixel_end_xy"]) == 2
        assert len(segment["visible_surface_pixel_xy"]) == 2
        assert segment["world_start_xyz_m"] != segment["world_end_xyz_m"]


def test_pending_agentview_ray_draws_visible_surface_marker(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    views = _views(tmp_path, image_size=128)
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=4,
        v=4,
        artifact_root=tmp_path,
    )

    rendered = annotate_pending_constraint_views(
        views,
        ray,
        output_root=tmp_path / "pending-visible-surface",
        point_id="visible_feature",
        show_visible_surface_marker=True,
    )

    front = views["pointcloud_front"]
    marker_u, marker_v = project_world_to_view(
        front,
        np.asarray(ray["visible_surface"]["xyz_m"]),
    )
    overlay = np.asarray(
        Image.open(rendered["pointcloud_front"]).convert("RGB")
    )
    marker_patch = overlay[
        max(0, marker_v - 9) : marker_v + 10,
        max(0, marker_u - 9) : marker_u + 10,
    ]
    assert np.any(
        np.all(marker_patch == np.asarray([80, 255, 255]), axis=-1)
    )


def test_solved_agentview_anchor_ray_is_visual_only_and_opt_in(
    tmp_path: Path,
) -> None:
    views = _views(tmp_path, image_size=128)
    mark = {
        "surface": {
            "xyz_m": [0.0, 0.0, 1.0],
            "source_kind": "agentview_first_visible_surface",
            "agentview_source_ray": {
                "camera_id": "agentview",
                "pixel_xy": [4, 4],
                "origin_xyz_m": [-1.0, 0.0, 1.0],
                "direction_xyz": [1.0, 0.0, 0.0],
            },
        }
    }

    control = annotate_views(
        views,
        mark,
        output_root=tmp_path / "solved-control",
    )
    treatment = annotate_views(
        views,
        mark,
        output_root=tmp_path / "solved-treatment",
        solved_agentview_anchor_ray_visual_mode="surface_inward_segment_v1",
    )
    segment = annotate_views(
        views,
        mark,
        output_root=tmp_path / "solved-segment",
        solved_agentview_anchor_ray_visual_mode="surface_inward_segment_v2",
    )

    control_top = np.asarray(
        Image.open(control["pointcloud_top"]).convert("RGB")
    )
    treatment_top = np.asarray(
        Image.open(treatment["pointcloud_top"]).convert("RGB")
    )
    assert not np.array_equal(control_top, treatment_top)
    cyan = np.all(
        treatment_top == np.asarray([80, 255, 255], dtype=np.uint8),
        axis=-1,
    )
    assert int(np.count_nonzero(cyan)) >= 8
    assert ".marked." in control["pointcloud_top"].name
    assert ".marked." in treatment["pointcloud_top"].name
    segment_top = np.asarray(
        Image.open(segment["pointcloud_top"]).convert("RGB")
    )
    assert not np.array_equal(control_top, segment_top)
    assert not np.array_equal(treatment_top, segment_top)
    # The segment treatment is intentionally non-directional: it retains the
    # calibrated line-of-sight cue without adding an arrow head.
    assert int(np.count_nonzero(
        np.all(segment_top == np.asarray([80, 255, 255], dtype=np.uint8), axis=-1)
    )) >= 8


def test_pending_constraint_can_render_explicit_active_reference(
    tmp_path: Path,
) -> None:
    views = _views(tmp_path)
    target = np.asarray([0.0, 0.0, 1.0])
    top = views["pointcloud_top"]
    u, v = project_world_to_view(top, target)
    _, pending = mark_world_point(top, u=u, v=v)

    rendered = annotate_pending_constraint_views(
        views,
        pending["pending_constraint"],
        output_root=tmp_path / "pending-with-reference",
        point_id="execution_grip_site",
        marks={
            "semantic_target_reference": {
                "xyz_m": [0.1, 0.1, 1.0],
            }
        },
    )

    assert set(rendered) == set(views)
    baseline = np.asarray(Image.open(top.image_path).convert("RGB"))
    overlay = np.asarray(Image.open(rendered["pointcloud_top"]).convert("RGB"))
    semantic_u, semantic_v = project_world_to_view(
        top, np.asarray([0.1, 0.1, 1.0])
    )
    assert not np.array_equal(
        overlay[
            semantic_v - 12 : semantic_v + 13,
            semantic_u - 12 : semantic_u + 13,
        ],
        baseline[
            semantic_v - 12 : semantic_v + 13,
            semantic_u - 12 : semantic_u + 13,
        ],
    )
@pytest.mark.parametrize("endpoint_key", ["pixel_start_xy", "pixel_end_xy"])
def test_published_ray_endpoint_pixel_is_clickable(
    tmp_path: Path,
    endpoint_key: str,
) -> None:
    record = _record(tmp_path)
    views = _views(tmp_path, image_size=384)
    ray = camera_ray_from_image_click(
        record,
        camera_id="agentview",
        u=3,
        v=5,
    )
    front = views["pointcloud_front"]
    endpoint = projected_camera_ray_segments(
        views, ray
    )["pointcloud_front"][endpoint_key]
    _, click = mark_world_point(
        front,
        u=endpoint[0],
        v=endpoint[1],
    )

    result = solve_camera_ray_constraint(
        ray,
        click["pending_constraint"],
    )

    assert result["status"] == "solved"
    assert (
        float(ray["workspace_t_range_m"][0])
        <= result["ray_parameter_m"]
        <= float(ray["workspace_t_range_m"][1])
    )


def test_rendered_views_publish_world_directions_and_actual_grip_site(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    points, colors = world_pointcloud_from_record(record, artifact_root=tmp_path)
    views = render_world_pointcloud_views(
        points,
        colors,
        observation_record=record,
        artifact_root=tmp_path,
        output_root=tmp_path / "views",
        image_size=128,
        grip_site_xyz=[0.0, 0.1, 1.0],
        grip_site_rotation=np.eye(3).tolist(),
        grip_site_aperture_m=0.06,
        finger_pad_contact_centers_world_m=[
            [-0.026, 0.1, 0.996],
            [0.026, 0.1, 0.996],
        ],
        target_grip_site_xyz=[0.02, 0.1, 1.0],
        compact_grip_site_overlay=True,
    )
    expected = {
        "pointcloud_top": ("+X", "+Y"),
        "pointcloud_front": ("+X", "+Z"),
        "pointcloud_side": ("+Y", "+Z"),
    }
    for name, view in views.items():
        metadata = json.loads(view.metadata_path.read_text())
        assert metadata["world_frame_directions"] == {
            "screen_right": expected[name][0],
            "screen_up": expected[name][1],
        }
        assert metadata["actual_grip_site_xyz_m"] == pytest.approx(
            [0.0, 0.1, 1.0]
        )
        assert metadata["actual_grip_site_jaw_axis_world"] == pytest.approx(
            [1.0, 0.0, 0.0]
        )
        assert metadata["actual_grip_site_approach_axis_world"] == pytest.approx(
            [0.0, 0.0, 1.0]
        )
        assert metadata["actual_gripper_aperture_m"] == pytest.approx(0.06)
        assert np.asarray(
            metadata["actual_finger_pad_contact_centers_world_m"]
        ) == pytest.approx(
            np.asarray([[-0.026, 0.1, 0.996], [0.026, 0.1, 0.996]])
        )
        assert metadata["commanded_grip_site_xyz_m"] == pytest.approx(
            [0.02, 0.1, 1.0]
        )
        assert metadata["grid_interval_m"] == pytest.approx(0.1)
        assert metadata["axis_colors"] == {
            axis.upper(): list(color) for axis, color in AXIS_COLORS.items()
        }
        image = np.asarray(Image.open(view.image_path).convert("RGB"))
        magenta_pixels = np.all(image == [255, 0, 255], axis=2)
        orange_pixels = np.all(image == [255, 110, 40], axis=2)
        legacy_legend_pixels = np.all(image == [255, 120, 255], axis=2)
        assert 1 <= int(magenta_pixels.sum()) <= 16
        assert int(orange_pixels.sum()) <= 32
        assert not legacy_legend_pixels.any()
        payload = np.load(view.lookup_path)
        assert payload["xyz_m"].shape == (128, 128, 3)
        assert payload["valid"].shape == (128, 128)


def test_metric_edge_tick_band_is_readable_without_changing_xyz(
    tmp_path: Path,
) -> None:
    xs = np.linspace(-0.35, 0.35, 220)
    ys = np.linspace(-0.30, 0.45, 220)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack(
        (
            grid_x.ravel(),
            grid_y.ravel(),
            np.full(grid_x.size, 1.0),
        )
    )
    colors = np.full((len(points), 3), 255, dtype=np.uint8)
    record = {"observation_id": "obs-tick-band"}
    legacy = render_world_pointcloud_views(
        points,
        colors,
        observation_record=record,
        artifact_root=tmp_path,
        output_root=tmp_path / "legacy",
        image_size=128,
        metric_edge_ticks=True,
        metric_tick_band_style="legacy_v1",
    )["pointcloud_top"]
    band = render_world_pointcloud_views(
        points,
        colors,
        observation_record=record,
        artifact_root=tmp_path,
        output_root=tmp_path / "band",
        image_size=128,
        metric_edge_ticks=True,
        metric_tick_band_style="readable_v1",
    )["pointcloud_top"]
    labels = render_world_pointcloud_views(
        points,
        colors,
        observation_record=record,
        artifact_root=tmp_path,
        output_root=tmp_path / "labels",
        image_size=128,
        metric_edge_ticks=True,
        metric_tick_band_style="readable_v2",
    )["pointcloud_top"]
    decluttered = render_world_pointcloud_views(
        points,
        colors,
        observation_record=record,
        artifact_root=tmp_path,
        output_root=tmp_path / "decluttered",
        image_size=128,
        metric_edge_ticks=True,
        metric_tick_band_style="readable_v3",
    )["pointcloud_top"]

    legacy_image = np.asarray(Image.open(legacy.image_path).convert("RGB"))
    band_image = np.asarray(Image.open(band.image_path).convert("RGB"))
    labels_image = np.asarray(Image.open(labels.image_path).convert("RGB"))
    decluttered_image = np.asarray(
        Image.open(decluttered.image_path).convert("RGB")
    )
    assert band_image[46:59].mean() < legacy_image[46:59].mean() * 0.65
    assert np.array_equal(band_image[70:, :30], legacy_image[70:, :30])
    assert labels_image[46:59].mean() < legacy_image[46:59].mean()
    assert labels_image[70:, :30].mean() < legacy_image[70:, :30].mean()
    assert not np.array_equal(decluttered_image, labels_image)

    legacy_lookup = np.load(legacy.lookup_path)
    for view in (band, labels, decluttered):
        lookup = np.load(view.lookup_path)
        assert np.array_equal(legacy_lookup["valid"], lookup["valid"])
        assert np.allclose(
            legacy_lookup["xyz_m"],
            lookup["xyz_m"],
            equal_nan=True,
        )


def test_metric_tick_label_layout_removes_overlapping_edge_labels() -> None:
    draw = ImageDraw.Draw(Image.new("RGB", (384, 384)))
    layouts = _layout_metric_tick_labels(
        draw,
        [
            ((0, 46), "-0.6"),
            ((20, 46), "-0.5"),
            ((339, 46), "+0.5"),
            ((356, 46), "+0.6"),
            ((2, 46), "+0.7"),
            ((2, 46), "+0.6"),
            ((2, 54), "+0.5"),
            ((2, 83), "+0.4"),
        ],
        font=_font(9),
        declutter=True,
    )

    boxes = [box for _position, _label, box in layouts]
    assert len(boxes) < 8
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            assert (
                box[2] < other[0]
                or other[2] < box[0]
                or box[3] < other[1]
                or other[3] < box[1]
            )


def test_point_view_composition_preserves_actual_active_and_authored_layers(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    points, colors = world_pointcloud_from_record(record, artifact_root=tmp_path)
    actual = np.asarray([-0.22, -0.20, 1.0])
    active = np.asarray([0.20, -0.12, 1.05])
    authored = np.asarray([0.02, 0.28, 1.12])
    views = render_world_pointcloud_views(
        points,
        colors,
        observation_record=record,
        artifact_root=tmp_path,
        output_root=tmp_path / "composed-views",
        image_size=256,
        grip_site_xyz=actual.tolist(),
        grip_site_rotation=np.eye(3).tolist(),
        grip_site_aperture_m=0.06,
        finger_pad_contact_centers_world_m=[
            (actual + np.asarray([-0.026, 0.0, -0.004])).tolist(),
            (actual + np.asarray([0.026, 0.0, -0.004])).tolist(),
        ],
    )
    active_paths = annotate_active_grip_site_views(
        views,
        output_root=tmp_path / "composed-views" / "active",
        point_id="P_active",
        target_position_xyz_m=active.tolist(),
        target_pad_contact_centers_world_m=[
            (active + np.asarray([-0.026, 0.0, -0.004])).tolist(),
            (active + np.asarray([0.026, 0.0, -0.004])).tolist(),
        ],
    )
    final_paths = annotate_views(
        views,
        {"P_new": {"xyz_m": authored.tolist()}},
        output_root=tmp_path / "composed-views" / "final",
        source_paths=active_paths,
    )

    top = views["pointcloud_top"]
    base = np.asarray(Image.open(top.image_path).convert("RGB"))
    clean = np.asarray(Image.open(top.clean_image_path).convert("RGB"))
    active_image = np.asarray(
        Image.open(active_paths["pointcloud_top"]).convert("RGB")
    )
    final = np.asarray(Image.open(final_paths["pointcloud_top"]).convert("RGB"))

    def patch(image: np.ndarray, xyz: np.ndarray, radius: int = 8) -> np.ndarray:
        u, v = project_world_to_view(top, xyz)
        return image[v - radius : v + radius + 1, u - radius : u + radius + 1]

    assert not np.array_equal(patch(base, actual), patch(clean, actual))
    assert np.array_equal(patch(final, actual), patch(base, actual))
    assert not np.array_equal(patch(active_image, active), patch(base, active))
    assert np.array_equal(patch(final, active), patch(active_image, active))
    assert not np.array_equal(
        patch(final, authored), patch(active_image, authored)
    )


def test_persistent_pointcloud_view_omits_actual_pose_axes_but_keeps_pads(
    tmp_path: Path,
) -> None:
    record = _record(tmp_path)
    points, colors = world_pointcloud_from_record(record, artifact_root=tmp_path)
    grip = np.asarray([0.0, 0.1, 1.0])
    views = render_world_pointcloud_views(
        points,
        colors,
        observation_record=record,
        artifact_root=tmp_path,
        output_root=tmp_path / "axis-free-views",
        image_size=256,
        grip_site_xyz=grip.tolist(),
        grip_site_rotation=np.eye(3).tolist(),
        grip_site_aperture_m=0.06,
        finger_pad_contact_centers_world_m=[
            (grip + np.asarray([-0.026, 0.0, -0.004])).tolist(),
            (grip + np.asarray([0.026, 0.0, -0.004])).tolist(),
        ],
    )

    image = np.asarray(
        Image.open(views["pointcloud_front"].image_path).convert("RGB")
    )
    # Persistent views retain orange pad centers.
    orange = (
        (image[..., 0] > 220)
        & (image[..., 1] > 60)
        & (image[..., 1] < 210)
        & (image[..., 2] < 100)
    )
    assert int(np.count_nonzero(orange)) > 10
    # Cyan is reserved for APP in explicit pose previews and should not appear
    # around the actual grip-site in the persistent scene view.
    cyan = (
        (image[..., 0] < 100)
        & (image[..., 1] > 170)
        & (image[..., 2] > 190)
    )
    assert int(np.count_nonzero(cyan)) == 0


def test_operator_pointcloud_refs_can_be_published_as_absolute_paths(
    tmp_path: Path,
) -> None:
    view = _views(tmp_path, image_size=32)["pointcloud_top"]

    dashboard_ref = view.public_dict(tmp_path)
    operator_ref = view.public_dict(tmp_path, absolute_paths=True)

    assert not Path(dashboard_ref["image_path"]).is_absolute()
    assert Path(operator_ref["image_path"]).is_absolute()
    assert operator_ref["image_path"] == str(view.image_path.resolve())
    assert operator_ref["lookup_path"] == str(view.lookup_path.resolve())
    assert operator_ref["metadata_path"] == str(view.metadata_path.resolve())


def test_voxel_fuse_pointcloud_averages_each_voxel() -> None:
    points = np.asarray(
        [
            [0.001, 0.001, 1.001],
            [0.003, 0.003, 1.003],
            [0.021, 0.001, 1.001],
        ],
        dtype=np.float64,
    )
    colors = np.asarray(
        [
            [10, 20, 30],
            [30, 40, 50],
            [100, 110, 120],
        ],
        dtype=np.uint8,
    )

    fused_points, fused_colors = voxel_fuse_pointcloud(
        points, colors, voxel_m=0.01
    )

    assert fused_points.shape == (2, 3)
    assert fused_colors.shape == (2, 3)
    assert any(
        np.allclose(point, [0.002, 0.002, 1.002])
        and np.array_equal(color, [20, 30, 40])
        for point, color in zip(fused_points, fused_colors)
    )
    assert any(
        np.allclose(point, [0.021, 0.001, 1.001])
        and np.array_equal(color, [100, 110, 120])
        for point, color in zip(fused_points, fused_colors)
    )


def test_world_view_axes_keep_right_and_up_as_positive() -> None:
    assert VIEW_SPECS["pointcloud_top"]["horizontal_axis"] == "x"
    assert VIEW_SPECS["pointcloud_top"]["vertical_axis"] == "y"
    assert VIEW_SPECS["pointcloud_front"]["horizontal_axis"] == "x"
    assert VIEW_SPECS["pointcloud_front"]["vertical_axis"] == "z"
    assert VIEW_SPECS["pointcloud_side"]["horizontal_axis"] == "y"
    assert VIEW_SPECS["pointcloud_side"]["vertical_axis"] == "z"


def test_all_complementary_view_pairs_solve_an_ordinary_point(tmp_path: Path) -> None:
    views = _views(tmp_path)
    target = np.array([0.0, 0.1, 1.0])
    for first_name, second_name in (
        ("pointcloud_top", "pointcloud_front"),
        ("pointcloud_top", "pointcloud_side"),
        ("pointcloud_front", "pointcloud_side"),
    ):
        xyz, source = _solve(views, target, first_name, second_name)
        assert source["status"] == "solved"
        assert source["source_kind"] == "two_view_intersection"
        assert np.allclose(xyz, target, atol=0.008)
        assert set(source["view_contributions_m"]) == {first_name, second_name}
        assert "support" not in source
        assert "projections" not in source


def test_empty_space_point_is_not_rejected_by_pointcloud_support(tmp_path: Path) -> None:
    xyz, source = _solve(
        _views(tmp_path),
        np.array([0.0, 0.0, 1.2]),
        "pointcloud_top",
        "pointcloud_front",
    )
    assert source["status"] == "solved"
    assert np.allclose(xyz, [0.0, 0.0, 1.2], atol=0.008)
    assert "support" not in source


def test_same_view_requires_complementary_click_and_keeps_pending(tmp_path: Path) -> None:
    top = _views(tmp_path)["pointcloud_top"]
    u, v = project_world_to_view(top, np.array([0.0, 0.0, 1.0]))
    _, pending = mark_world_point(top, u=u, v=v)
    point, source = mark_world_point(
        top, u=u, v=v, pending_constraint=pending["pending_constraint"]
    )
    assert point is None
    assert source["status"] == "complementary_view_required"
    assert source["pending_constraint"] == pending["pending_constraint"]


def test_inconsistent_shared_axis_keeps_latest_click_for_retry(tmp_path: Path) -> None:
    views = _views(tmp_path)
    target = np.array([0.0, 0.0, 1.0])
    top, front = views["pointcloud_top"], views["pointcloud_front"]
    u1, v1 = project_world_to_view(top, target)
    _, pending = mark_world_point(top, u=u1, v=v1)
    u2, v2 = project_world_to_view(front, target)
    point, source = mark_world_point(
        front,
        u=u2 + 4,
        v=v2,
        pending_constraint=pending["pending_constraint"],
    )
    assert point is None
    assert source["status"] == "inconsistent_views"
    assert source["shared_axis"] == "x"
    assert source["shared_axis_residual_m"] > 0.015
    assert source["pending_constraint"]["view"] == "pointcloud_front"


def test_three_arbitrary_point_ids_produce_right_handed_pose() -> None:
    pose = pose_from_points(
        {
            "P0": {"xyz_m": [0.0, 0.0, 1.0]},
            "P1": {"xyz_m": [0.0, 0.0, 0.9]},
            "P2": {"xyz_m": [0.1, 0.0, 1.0]},
        },
        position_point_id="P0",
        approach_from_point_id="P1",
        jaw_toward_point_id="P2",
    )
    rotation = np.asarray(pose["rotation_matrix"])
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    assert np.isclose(np.linalg.det(rotation), 1.0)
    assert pose["point_ids"] == {
        "position": "P0",
        "approach_from": "P1",
        "jaw_toward": "P2",
    }


def test_rgbd_fusion_accepts_multiple_calibrated_frames(tmp_path: Path) -> None:
    record = _record(tmp_path)
    second_rgb = tmp_path / "rgb2.png"
    second_depth = tmp_path / "depth2.png"
    Image.new("RGB", (8, 8), (200, 20, 20)).save(second_rgb)
    Image.fromarray(np.full((8, 8), 1000, dtype=np.uint16), mode="I;16").save(second_depth)
    record["frames"].append({
        **record["frames"][0],
        "camera_id": "virtual-01",
        "frame_id": "frame-virtual-01",
        "rgb_path": str(second_rgb),
        "depth_path": str(second_depth),
        "metadata": {**record["frames"][0]["metadata"], "extrinsics": {
            **record["frames"][0]["metadata"]["extrinsics"], "pos": [0.1, 0.0, 0.0]
        }},
    })
    single, single_colors = world_pointcloud_from_record(
        record, artifact_root=tmp_path, camera_ids=("agentview",)
    )
    fused, fused_colors = world_pointcloud_from_record(
        record, artifact_root=tmp_path, camera_ids=("agentview", "virtual-01")
    )
    assert len(fused) == 2 * len(single)
    assert len(fused_colors) == 2 * len(single_colors)
    assert np.isclose(fused[:, 0].max() - single[:, 0].max(), 0.1, atol=1e-6)
