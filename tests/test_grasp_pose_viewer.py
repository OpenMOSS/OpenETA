from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.embodied.grasp_pose_viewer import (
    ANYGRASP_TO_PANDA_AXES,
    SUPPORT_CLEARANCE_REQUIRED_MM,
    GraspPose,
    GraspPoseViewer,
    camera_preset,
    estimate_support_plane,
    load_anygrasp_poses,
    load_execution_poses,
    load_graspgenx_poses,
    load_graspgenx_prefilter_poses,
    load_scene_cloud,
    orbit_camera_state,
    pose_with_support_clearance,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_anygrasp_pose_is_converted_to_world_panda_execution_geometry(
    tmp_path: Path,
) -> None:
    response = tmp_path / "anygrasp.json"
    _write_json(
        response,
        {
            "details": {
                "grasp_candidates": [
                    {
                        "id": "grasp_007",
                        "rank": 7,
                        "frame": "camera",
                        "camera_frame": "opencv",
                        "score": 0.8,
                        "width": 0.1,
                        "translation_xyz": [0.1, 0.2, 0.3],
                        "rotation_matrix": np.eye(3).tolist(),
                    }
                ]
            }
        },
    )
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [1.0, 2.0, 3.0]

    poses = load_anygrasp_poses(
        response,
        camera_to_world_opencv=camera_to_world,
        proposal_set_id="ps-test",
    )

    assert len(poses) == 1
    pose = poses[0]
    assert pose.pose_id == "ps-test/anygrasp/007"
    assert pose.label == "AG7"
    assert pose.aperture_m == pytest.approx(0.1)
    assert pose.aperture_semantics == "proposed"
    assert pose.aperture_status == "exceeds_robot_limit"
    assert pose.robot_max_aperture_m == pytest.approx(0.08)
    np.testing.assert_allclose(
        pose.world_from_grip_site[:3, :3], ANYGRASP_TO_PANDA_AXES
    )
    np.testing.assert_allclose(
        pose.world_from_grip_site[:3, 3], [1.1, 2.2, 3.3]
    )
    np.testing.assert_allclose(
        pose.world_from_gripper_base[:3, 3],
        pose.world_from_grip_site[:3, 3]
        - 0.097 * pose.world_from_grip_site[:3, 2],
    )


def test_viewer_state_exposes_exact_proposal_and_result_identity(
    tmp_path: Path,
) -> None:
    transform = np.eye(4)
    pose = GraspPose(
        pose_id="proposal-a/anygrasp/000",
        label="AG0",
        backend="anygrasp",
        backend_rank=0,
        score=0.9,
        world_from_gripper_base=transform,
        world_from_grip_site=transform,
        source_id="grasp_000",
    )
    viewer = GraspPoseViewer.__new__(GraspPoseViewer)
    viewer.lock = threading.RLock()
    viewer.proposal_set_id = "proposal-a"
    viewer.scene_mode = "proposal"
    viewer.episode_root = tmp_path / "episode"
    viewer.observation_id = "obs-000010"
    viewer.proposal_id = "proposal-a"
    viewer.result_id = "result-a"
    viewer.revision = 1
    viewer.show_backend = {
        "anygrasp": True,
        "graspgenx": True,
        "refined": True,
        "target": True,
        "actual": True,
    }
    viewer.scope = "default"
    viewer.focus_pose_id = pose.pose_id
    viewer.poses = [pose]
    viewer.clients = {}
    viewer.scene = SimpleNamespace(diagnostics={})

    state = viewer.state()

    assert state["proposal_id"] == "proposal-a"
    assert state["result_id"] == "result-a"
    assert state["poses"][0]["source_id"] == "grasp_000"
    assert state["poses"][0]["support_clearance_mm"] is None
    assert state["poses"][0]["support_clearance_required_mm"] == 20.0
    assert state["poses"][0]["support_clearance_status"] == "unknown"
    assert state["poses"][0]["selectable"] is False


def test_viewer_bootstraps_client_connected_before_callback_registration() -> None:
    client = SimpleNamespace(client_id=7)

    class FakeServer:
        def __init__(self) -> None:
            self.connect_callback = None
            self.disconnect_callback = None

        def on_client_connect(self, callback):
            self.connect_callback = callback
            return callback

        def on_client_disconnect(self, callback):
            self.disconnect_callback = callback
            return callback

        def get_clients(self):
            return {7: client}

    viewer = GraspPoseViewer.__new__(GraspPoseViewer)
    viewer.lock = threading.RLock()
    viewer.server = FakeServer()
    viewer.clients = {}
    camera_updates: list[tuple[int, str]] = []
    state_writes: list[bool] = []
    viewer._set_client_camera = lambda value, preset, _focus: camera_updates.append(
        (value.client_id, preset)
    )
    viewer._write_state = lambda: state_writes.append(True)

    viewer._install_client_tracking()

    assert viewer.clients == {7: client}
    assert camera_updates == [(7, "scene")]
    assert state_writes == [True]
    assert viewer.server.disconnect_callback is not None
    viewer.server.disconnect_callback(client)
    assert viewer.clients == {}
    assert state_writes == [True, True]


def test_viewer_resyncs_clients_from_viser_authority() -> None:
    client = SimpleNamespace(client_id=9)

    class FakeServer:
        def get_clients(self):
            return {9: client}

    viewer = GraspPoseViewer.__new__(GraspPoseViewer)
    viewer.lock = threading.RLock()
    viewer.server = FakeServer()
    viewer.clients = {}

    viewer._sync_connected_clients()

    assert viewer.clients == {9: client}


def test_support_plane_fit_and_signed_clearance_handle_tilted_plane() -> None:
    normal = np.asarray([0.15, -0.20, 1.0], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    axis_x = np.asarray([1.0, 0.0, -normal[0] / normal[2]])
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    origin = np.asarray([0.2, -0.1, 0.4])
    coordinates = np.linspace(-0.12, 0.12, 9)
    support_points = np.asarray(
        [origin + x * axis_x + y * axis_y for x in coordinates for y in coordinates]
    )

    plane = estimate_support_plane(support_points)

    assert plane is not None
    point, estimated_normal, offset = plane
    np.testing.assert_allclose(estimated_normal, normal, atol=1e-8)
    assert abs(float(np.dot(estimated_normal, point) + offset)) < 1e-10
    collision_vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            0.02 * axis_x,
            0.02 * axis_y,
        ]
    )
    transform = np.eye(4)
    transform[:3, 3] = origin + 0.025 * normal
    pose = GraspPose(
        pose_id="tilted",
        label="T",
        backend="refined",
        backend_rank=0,
        score=None,
        world_from_gripper_base=transform,
        world_from_grip_site=transform,
        source_id="tilted",
    )

    measured = pose_with_support_clearance(
        pose,
        collision_vertices,
        support_plane_normal_world=estimated_normal,
        support_plane_offset_m=offset,
    )

    assert measured.support_clearance_mm == pytest.approx(25.0, abs=1e-8)
    assert measured.support_clearance_status == "eligible"


def test_run18_r0_r1_support_clearance_boundaries_use_collision_mesh() -> None:
    collision_vertices = np.asarray(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.004], [-0.01, 0.0, 0.008]]
    )
    plane_normal = np.asarray([0.0, 0.0, 1.0])
    statuses: list[tuple[float, str, bool]] = []
    for clearance_mm in (28.39, 0.39, SUPPORT_CLEARANCE_REQUIRED_MM):
        transform = np.eye(4)
        transform[2, 3] = clearance_mm / 1000.0
        pose = GraspPose(
            pose_id=f"clearance-{clearance_mm}",
            label="R",
            backend="refined",
            backend_rank=0,
            score=None,
            world_from_gripper_base=transform,
            world_from_grip_site=transform,
            source_id="synthetic",
        )
        measured = pose_with_support_clearance(
            pose,
            collision_vertices,
            support_plane_normal_world=plane_normal,
            support_plane_offset_m=0.0,
        )
        statuses.append(
            (
                measured.support_clearance_mm or 0.0,
                measured.support_clearance_status,
                measured.public_dict()["selectable"],
            )
        )

    assert statuses[0][0] == pytest.approx(28.39)
    assert statuses[0][1:] == ("eligible", True)
    assert statuses[1][0] == pytest.approx(0.39)
    assert statuses[1][1:] == ("unsafe", False)
    assert statuses[2][0] == pytest.approx(20.0)
    assert statuses[2][1:] == ("eligible", True)


def test_graspgenx_ids_keep_backend_rank_when_top_k_filters(tmp_path: Path) -> None:
    results = tmp_path / "gx.json"
    pose = np.eye(4)
    grip = np.eye(4)
    grip[2, 3] = 0.097
    _write_json(
        results,
        {
            "grasps": [
                {
                    "id": "graspgenx_004",
                    "rank": 4,
                    "score": 0.7,
                    "branch": "obb",
                    "scene_clearance_mm": 3.2,
                    "transform_world_from_gripper": pose.tolist(),
                    "transform_world_from_grip_site": grip.tolist(),
                },
                {
                    "id": "graspgenx_009",
                    "rank": 9,
                    "score": 0.6,
                    "transform_world_from_gripper": pose.tolist(),
                    "transform_world_from_grip_site": grip.tolist(),
                },
            ]
        },
    )

    poses = load_graspgenx_poses(
        results, proposal_set_id="ps-test", top_k=1
    )

    assert [value.pose_id for value in poses] == ["ps-test/graspgenx/004"]
    assert [value.label for value in poses] == ["GX4"]


def test_explicit_graspgenx_raw_rank_keeps_provenance(tmp_path: Path) -> None:
    prefilter = tmp_path / "prefilter.json"
    _write_json(
        prefilter,
        {
            "grasps": [
                {
                    "raw_rank": 77,
                    "score": 0.78,
                    "branch": "diff",
                    "scene_clearance_m": 0.026,
                    "collision_free": True,
                    "transform_world_from_gripper": np.eye(4).tolist(),
                }
            ]
        },
    )

    poses = load_graspgenx_prefilter_poses(
        prefilter, proposal_set_id="ps-test", raw_ranks=[77]
    )

    assert poses[0].pose_id == "ps-test/graspgenx/raw-077"
    assert poses[0].label == "GXr77"
    assert poses[0].branch == "diff"
    assert poses[0].scene_clearance_mm == 26.0
    assert poses[0].world_from_grip_site[2, 3] == 0.097


def test_pose_camera_presets_are_deterministic_and_pose_relative() -> None:
    transform = np.eye(4)
    transform[:3, 3] = [0.2, 0.3, 0.4]
    pose = GraspPose(
        pose_id="ps-test/graspgenx/000",
        label="GX0",
        backend="graspgenx",
        backend_rank=0,
        score=0.9,
        world_from_gripper_base=transform,
        world_from_grip_site=transform,
        source_id="graspgenx_000",
    )

    first = camera_preset(
        "pose_approach",
        target_center=np.zeros(3),
        scene_radius_m=0.2,
        focus_pose=pose,
    )
    second = camera_preset(
        "pose_approach",
        target_center=np.ones(3),
        scene_radius_m=0.2,
        focus_pose=pose,
    )

    np.testing.assert_allclose(first["position"], second["position"])
    np.testing.assert_allclose(first["look_at"], transform[:3, 3])
    assert first["position"][2] < transform[2, 3]


def test_pose_camera_preset_requires_focus() -> None:
    try:
        camera_preset(
            "pose_jaws",
            target_center=np.zeros(3),
            scene_radius_m=0.2,
        )
    except ValueError as exc:
        assert "requires focus_pose" in str(exc)
    else:
        raise AssertionError("Expected a missing-focus error")


def test_pose_jaws_camera_stays_world_upright_for_top_down_grasp() -> None:
    transform = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.1],
            [1.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, -1.0, 0.9],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    pose = GraspPose(
        pose_id="refined-1",
        label="R0",
        backend="refined",
        backend_rank=0,
        score=0.0,
        world_from_gripper_base=transform,
        world_from_grip_site=transform,
        source_id="grasp-0",
    )

    camera = camera_preset(
        "pose_jaws",
        target_center=np.zeros(3),
        scene_radius_m=0.2,
        focus_pose=pose,
    )

    np.testing.assert_allclose(camera["up_direction"], [0.0, 0.0, 1.0])
    assert camera["position"][2] == transform[2, 3]


def test_execution_poses_are_exact_target_and_actual_world_grip_sites(
    tmp_path: Path,
) -> None:
    comparison = tmp_path / "comparison.json"
    target = np.eye(4)
    target[:3, 3] = [0.10, 0.20, 0.90]
    actual = np.eye(4)
    actual[:3, 3] = [0.13, 0.18, 0.92]
    _write_json(
        comparison,
        {
            "source_grasp_id": "R3",
            "proposed_jaw_width_m": 0.1,
            "measured_aperture_m": 0.042,
            "target_world_from_grip_site": target.tolist(),
            "actual_world_from_grip_site": actual.tolist(),
        },
    )

    poses = load_execution_poses(comparison, proposal_set_id="exec-test")

    assert [pose.pose_id for pose in poses] == [
        "exec-test/target",
        "exec-test/actual",
    ]
    assert [pose.label for pose in poses] == ["TARGET", "ACTUAL"]
    assert [pose.aperture_m for pose in poses] == pytest.approx([0.1, 0.042])
    assert [pose.aperture_semantics for pose in poses] == ["proposed", "measured"]
    assert [pose.aperture_status for pose in poses] == [
        "exceeds_robot_limit",
        "within_robot_limit",
    ]
    assert [pose.robot_max_aperture_m for pose in poses] == pytest.approx(
        [0.08, 0.08]
    )
    np.testing.assert_allclose(poses[0].world_from_grip_site, target)
    np.testing.assert_allclose(poses[1].world_from_grip_site, actual)
    np.testing.assert_allclose(
        poses[0].world_from_gripper_base[:3, 3],
        target[:3, 3] - 0.097 * target[:3, 2],
    )


def test_orbit_camera_rotates_about_focus_and_preserves_requested_radius() -> None:
    value = orbit_camera_state(
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        azimuth_delta_deg=90.0,
        elevation_delta_deg=30.0,
        zoom_scale=0.5,
    )

    np.testing.assert_allclose(value["look_at"], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(
        value["position"],
        [0.0, 0.5 * np.cos(np.deg2rad(30.0)), 0.25],
        atol=1e-8,
    )
    assert np.isclose(
        np.linalg.norm(value["position"] - value["look_at"]), 0.5
    )
    view = value["look_at"] - value["position"]
    assert abs(float(np.dot(view, value["up_direction"]))) < 1e-8


def test_scene_sample_rejects_observation_binding_mismatch(tmp_path: Path) -> None:
    sample = tmp_path / "00"
    sample.mkdir()
    _write_json(
        sample / "meta_data.json",
        {
            "observation_id": "obs-old",
            "camera_frame": "opencv",
        },
    )

    try:
        load_scene_cloud(sample, expected_observation_id="obs-current")
    except ValueError as exc:
        assert "observation mismatch" in str(exc)
        assert "obs-current" in str(exc)
        assert "obs-old" in str(exc)
    else:
        raise AssertionError("Expected a stale Viser sample to be rejected")
