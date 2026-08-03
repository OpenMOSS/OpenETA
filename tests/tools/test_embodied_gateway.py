from __future__ import annotations

import base64
import json
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import pytest

from tools.embodied_gateway import (
    EmbodiedGateway,
    GatewayResult,
    _build_anygrasp_depth_horizon_fallback,
    _classify_grasp_contact,
    _gripper_action_outcome,
    _gripper_action_payload_ok,
    _gripper_close_evidence,
    _mask_side_engagement_diagnostics,
    _motion_payload_ok,
    _move_to_operator_result,
    _move_to_close_confirmation_key,
    _move_to_close_preview_id,
    _move_to_preview_base_matches,
    _operator_gripper_feedback,
    _pose_preview_artifact_id,
    _task_terminal_success,
)
from tools.pointcloud_pose_marking import project_world_to_view


def test_release_gateway_uses_decluttered_metric_labels(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)

    assert gateway._metric_pointcloud_tick_band_style == "readable_v3"
    gateway.close()


def test_release_rgbd_clicks_solve_visible_surfaces(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    observed = gateway.observe(views=["agentview", "wrist"])
    assert observed.text["returned_views"] == ["agentview", "wrist"]

    agentview = gateway.mark_point_3d(
        view="agentview",
        u=16,
        v=16,
        point_id="agentview-surface",
    )
    wrist = gateway.mark_point_3d(
        view="wrist",
        u=16,
        v=16,
        point_id="wrist-surface",
    )

    for result in (agentview, wrist):
        assert result.success is True
        assert result.text["status"] == "solved"
        assert result.text["xyz_m"] == pytest.approx([0.0, 0.0, 1.0])
    assert gateway._point_marks_3d["agentview-surface"]["source_kind"] == (
        "rgbd_first_visible_surface"
    )
    assert wrist.details["mark"]["source_kind"] == (
        "rgbd_first_visible_surface"
    )
    gateway.close()


def test_release_solved_mark_feedback_is_current_and_pixel_precise(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe(views=["wrist"])
    gateway._point_marks_3d["old-point"] = {
        "point_id": "old-point",
        "observation_id": gateway.current_record["observation_id"],
        "xyz_m": [0.1, 0.1, 1.0],
    }

    solved = gateway.mark_point_3d(
        view="wrist",
        u=16,
        v=16,
        point_id="current-point",
    )

    assert solved.success is True
    assert solved.text["views"] == ["wrist"]
    assert len(solved.images) == 1
    assert ".current-only." in solved.images[0].name
    pixels = np.asarray(Image.open(solved.images[0]).convert("RGB"))
    red = (
        (pixels[..., 0] > 220)
        & (pixels[..., 1] < 80)
        & (pixels[..., 2] < 80)
    )
    ys, xs = np.where(red)
    assert len(xs) > 0
    assert xs.min() >= 14 and xs.max() <= 18
    assert ys.min() >= 14 and ys.max() <= 18
    assert red[16, 16]
    gateway.close()


def test_release_candidate_views_and_close_tokens_use_separate_namespaces(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe(views=["agentview"])
    position = gateway._current_grip_site_xyz_for_render()
    assert position is not None

    candidate = gateway.move_to_target(
        {
            "position_xyz_m": list(position),
            "gripper": "open",
            "preview_only": True,
        }
    )
    assert "preview_id" not in candidate.text
    assert candidate.text["returned_views"]
    assert all(
        ref.startswith("candidate:candidate-")
        for ref in candidate.text["returned_views"]
    )

    close = gateway.move_to_target({"gripper": "close"})
    assert close.text["preview_id"].startswith("P-")
    assert close.text["preview_id"] not in " ".join(
        close.text["returned_views"]
    )
    gateway.close()


def test_release_close_preview_requires_one_shot_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    preview_paths: dict[str, Path] = {}
    for name in ("pointcloud_front", "pointcloud_side"):
        path = gateway.root / "pointcloud_views" / f"{name}.candidate.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (40, 40, 40)).save(path)
        preview_paths[name] = path
    monkeypatch.setattr(gateway, "_ensure_operator_pointcloud_views", lambda: None)
    monkeypatch.setattr(
        "tools.embodied_gateway.render_pose_candidate_frame_preview_views",
        lambda *_args, **_kwargs: preview_paths,
    )
    monkeypatch.setattr(
        gateway,
        "_render_agentview_pose_preview",
        lambda **_kwargs: None,
    )

    target = {
        "position_xyz_m": [0.05, 0.0, 1.0],
        "gripper": "close",
    }
    previewed = gateway.move_to_target(target)
    preview_id = previewed.text["preview_id"]
    assert preview_id.startswith("P-")
    assert not any(
        name in {"move_to", "gripper_close"}
        for name, _arguments in transport.calls
    )

    repeated = gateway.move_to_target(target)
    assert repeated.success is False
    assert repeated.text["reason"] == "preview_decision_required"

    committed = gateway.move_to_target(
        {"execute_preview_id": preview_id}
    )
    assert committed.success is True
    assert any(name == "move_to" for name, _arguments in transport.calls)
    assert any(name == "gripper_close" for name, _arguments in transport.calls)

    replayed = gateway.move_to_target(
        {"execute_preview_id": preview_id}
    )
    assert replayed.success is False
    assert replayed.text["reason"] == "stale_preview"
    gateway.close()


def test_release_failed_motion_skips_combined_gripper_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    monkeypatch.setattr(
        gateway,
        "move_to_pose",
        lambda position, rpy, *, reason="": GatewayResult(
            False,
            {
                "kind": "move_to_pose",
                "success": False,
                "message": "motion did not converge",
            },
        ),
    )

    result = gateway.move_to_target(
        {
            "position_delta_mm": [0.0, 0.0, 10.0],
            "gripper": "open",
        }
    )

    assert result.success is False
    assert result.text["motion_status"] == "not_reached"
    assert "gripper" not in result.text
    assert not any(name == "gripper_open" for name, _ in transport.calls)
    gateway.close()


def test_release_gripper_only_failure_preserves_underlying_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    monkeypatch.setattr(
        gateway,
        "step",
        lambda _stage: GatewayResult(
            False,
            {
                "kind": "step",
                "success": False,
                "issue_code": "remote_action_failed",
                "message": "Step failed: executing action in terminated episode",
                "retryable": False,
            },
        ),
    )

    result = gateway.move_to_target({"gripper": "open"})

    assert result.success is False
    assert result.text == {
        "success": False,
        "reason": "remote_action_failed",
        "message": "Step failed: executing action in terminated episode",
        "retryable": False,
    }
    gateway.close()


def test_release_not_reached_marks_only_current_robot_contacts(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorTransport(
        move_response={
            "success": False,
            "start": {"xyz": [0.0, 0.0, 0.5]},
            "end": {"xyz": [0.0, 0.0, 0.5]},
            "target": {"x": 0.05, "y": 0.0, "z": 1.0},
            "steps_executed": 12,
            "reached_target": False,
            "motion_converged": False,
            "position_error_m": 0.5,
            "mujoco_contacts": {
                "source": "mujoco_data.contact",
                "points": [
                    {
                        "position_xyz_m": [-0.2, 0.0, 1.0],
                        "geom_names": [
                            "table_collision",
                            "cream_cheese_collision",
                        ],
                    },
                    {
                        "position_xyz_m": [0.2, 0.0, 1.0],
                        "geom_names": [
                            "gripper0_finger2_pad_collision",
                            "basket_collision",
                        ],
                    },
                ],
            },
        }
    )
    gateway, _transport = _gateway(tmp_path, transport=transport)
    gateway.observe()

    result = gateway.move_to_target(
        {"position_xyz_m": [0.05, 0.0, 1.0]},
        views=["agentview", "wrist"],
    )

    assert result.success is False
    assert result.text["motion_status"] == "not_reached"
    assert "contact_reason" not in result.text
    assert len(result.images) == 2
    pixels = np.asarray(Image.open(result.images[0]).convert("RGB"))
    assert pixels[16, 22].tolist() == [0, 255, 255]
    assert pixels[16, 10].tolist() != [0, 255, 255]
    cyan = np.all(
        pixels == np.asarray([0, 255, 255], dtype=np.uint8),
        axis=-1,
    )
    assert 1 <= int(np.count_nonzero(cyan)) <= 9
    gateway.close()


def test_grasp_contact_classifier_requires_object_to_follow_lift() -> None:
    status, reason, metrics = _classify_grasp_contact(
        baseline_centroid_world_m=[0.0, 0.0, 0.05],
        current_centroid_world_m=[0.0, 0.0, 0.051],
        lift_start_eef_world_m=[0.0, 0.0, 0.10],
        current_eef_world_m=[0.0, 0.0, 0.16],
    )
    assert (status, reason) == ("not_grasped", "target_remained_at_baseline")
    assert metrics["eef_lift_mm"] == 60.0

    status, reason, metrics = _classify_grasp_contact(
        baseline_centroid_world_m=[0.0, 0.0, 0.05],
        current_centroid_world_m=[0.0, 0.0, 0.105],
        lift_start_eef_world_m=[0.0, 0.0, 0.10],
        current_eef_world_m=[0.0, 0.0, 0.16],
    )
    assert (status, reason) == ("confirmed", "target_followed_eef_lift")
    assert metrics["object_lift_mm"] == 55.0


def test_grasp_contact_classifier_keeps_weak_evidence_unknown() -> None:
    status, reason, _metrics = _classify_grasp_contact(
        baseline_centroid_world_m=[0.0, 0.0, 0.05],
        current_centroid_world_m=[0.02, 0.0, 0.06],
        lift_start_eef_world_m=[0.0, 0.0, 0.10],
        current_eef_world_m=[0.0, 0.0, 0.11],
    )
    assert (status, reason) == ("unknown", "insufficient_eef_lift")


def _png_bytes(*, mode: str = "RGB", size: tuple[int, int] = (32, 32), value: int | tuple[int, ...] = 80) -> bytes:
    image = Image.new(mode, size, value)
    if mode == "L":
        image.putpixel((16, 16), 255)
    path = Path("/tmp") / f"openeta-test-{mode}.png"
    image.save(path)
    return path.read_bytes()


class FakeSimulatorTransport:
    def __init__(
        self,
        *,
        move_response: dict[str, Any] | None = None,
        open_aperture_m: float = 0.079,
        post_move_render_aperture_m: float | None = None,
        post_move_render_missing_aperture: bool = False,
        post_move_render_missing_wrist: bool = False,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.render_count = 0
        self.gripper_open = True
        self.open_aperture_m = open_aperture_m
        self.post_move_render_aperture_m = post_move_render_aperture_m
        self.post_move_render_missing_aperture = post_move_render_missing_aperture
        self.post_move_render_missing_wrist = post_move_render_missing_wrist
        self.move_seen = False
        self.move_response = move_response or {
            "success": True,
            "start": {"xyz": [0.0, 0.0, 0.5]},
            "end": {"xyz": [0.05, 0.0, 1.0]},
            "target": {"x": 0.05, "y": 0.0, "z": 1.0},
            "steps_executed": 3,
            "reached_target": True,
            "motion_converged": True,
            "position_error_m": 0.0,
            "settling_converged": True,
        }

    def call_tool(self, name: str, arguments: dict[str, Any], *, timeout_s: float | None = None) -> dict[str, Any]:
        self.calls.append((name, dict(arguments)))
        if name == "create_env":
            return {"handle": "fake-handle", "session_id": "fake-session", "success": True}
        if name in {"reset_env", "render_env"}:
            self.render_count += 1
            result = self._observation(self.render_count)
            if name == "render_env" and self.move_seen:
                if self.post_move_render_missing_wrist:
                    result["cameras"] = [
                        camera
                        for camera in result["cameras"]
                        if camera.get("camera_id") != "wrist"
                    ]
                state = result["robot"]["gripper_state"]
                if self.post_move_render_missing_aperture:
                    state.pop("aperture_m", None)
                elif self.post_move_render_aperture_m is not None:
                    state["aperture_m"] = self.post_move_render_aperture_m
            return result
        if name == "render_multiview_env":
            base = self._observation(max(1, self.render_count))["cameras"][0]
            cameras = [
                {
                    **base,
                    "camera_id": f"virtual-{index:02d}",
                    "frame_id": f"virtual-{index:02d}",
                    "virtual_camera": {
                        "azimuth_deg": float(index * 60),
                        "elevation_deg": -25.0,
                    },
                }
                for index in range(7)
            ]
            return {
                "kind": "multiview_render",
                "success": True,
                "physics_stepped": False,
                "robot_visuals_hidden": bool(arguments.get("hide_robot")),
                "hidden_robot_visual_geom_count": (
                    53 if arguments.get("hide_robot") else 0
                ),
                "camera_count": 7,
                "cameras": cameras,
            }
        if name == "move_to":
            self.move_seen = True
            return dict(self.move_response)
        if name in {"gripper_open", "gripper_close"}:
            self.gripper_open = name == "gripper_open"
            self.render_count += 1
            result = self._observation(self.render_count)
            result["steps_executed"] = 10
            return result
        if name == "check_task":
            return {"available": True, "success": False}
        if name == "close_env":
            return {"ok": True}
        raise AssertionError(f"unexpected simulator tool: {name}")

    def _observation(self, step: int) -> dict[str, Any]:
        rgb = base64.b64encode(_png_bytes(value=(80 + step, 90, 100))).decode("ascii")
        wrist_rgb = base64.b64encode(
            _png_bytes(value=(40, 60 + step, 80))
        ).decode("ascii")
        depth = base64.b64encode(_png_bytes(mode="I;16", value=1000)).decode("ascii")
        return {
            "success": True,
            "task": "pick the black bowl",
            "cameras": [
                {
                    "frame_id": "agentview",
                    "camera_id": "agentview",
                    "width": 32,
                    "height": 32,
                    "rgb_base64": rgb,
                    "depth_base64": depth,
                    "intrinsics": {"fx": 30.0, "fy": 30.0, "cx": 16.0, "cy": 16.0},
                    "extrinsics": {
                        "pos": [0.0, 0.0, 0.0],
                        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "frame_transform": "camera_to_world",
                        "camera_frame": "opencv",
                    },
                }
                ,
                {
                    "frame_id": "wrist",
                    "camera_id": "wrist",
                    "width": 32,
                    "height": 32,
                    "rgb_base64": wrist_rgb,
                    "depth_base64": depth,
                    "intrinsics": {"fx": 30.0, "fy": 30.0, "cx": 16.0, "cy": 16.0},
                    "extrinsics": {
                        "pos": [0.0, 0.0, 0.0],
                        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        "frame_transform": "camera_to_world",
                        "camera_frame": "opencv",
                    },
                },
            ],
            "robot": {
                "end_effector_pose": {
                    "xyz": [0.0, 0.0, 0.5],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "gripper_state": {
                    "open": self.gripper_open,
                    "aperture_m": (
                        self.open_aperture_m if self.gripper_open else 0.0025
                    ),
                    "finger_qvel": [0.0, 0.0],
                    "geometry": {
                        "source": "live_mujoco_collision_geometry",
                        "eef_frame": "panda_grip_site",
                        "finger_pad_geom_centers_world_m": [
                            [-0.03, 0.0, 0.496],
                            [0.03, 0.0, 0.496],
                        ],
                        "finger_pad_geom_rotations_world": [
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        ],
                        "finger_pad_inner_contact_centers_world_m": [
                            [-0.026, 0.0, 0.496],
                            [0.026, 0.0, 0.496],
                        ],
                        "finger_pad_geom_half_sizes_m": [
                            [0.008, 0.004, 0.008],
                            [0.008, 0.004, 0.008],
                        ],
                    },
                },
                "metadata": {"ee_pose_frame_consistent": True},
            },
        }


def _sam3_response(_request: dict[str, Any]) -> dict[str, Any]:
    mask = base64.b64encode(_png_bytes(mode="L", value=0)).decode("ascii")
    return {
        "success": True,
        "content": "segmented",
        "details": {
            "detection_count": 1,
            "detections": [
                {
                    "label": "black bowl",
                    "score": 0.95,
                    "bbox_xyxy": [8, 8, 24, 24],
                    "mask": {"format": "png", "base64": mask},
                    "area_px": 200,
                }
            ],
            "artifacts": [],
        },
    }


def _anygrasp_response(_request: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": True,
        "content": "grasps",
        "details": {
            "candidate_count": 1,
            "grasp_candidates": [
                {
                    "id": "backend-0",
                    "frame": "camera",
                    "camera_frame": "opencv",
                    "score": 0.88,
                    "translation_xyz": [0.0, 0.0, 1.0],
                    "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "gripper_tip_position_xyz": [0.05, 0.0, 1.0],
                    "depth": 0.03,
                    "width": 0.08,
                    "height": 0.03,
                }
            ],
            "artifacts": [],
        },
    }


def _gateway(
    tmp_path: Path,
    *,
    sam3: Any = _sam3_response,
    anygrasp: Any = _anygrasp_response,
    transport: FakeSimulatorTransport | None = None,
) -> tuple[EmbodiedGateway, FakeSimulatorTransport]:
    transport = transport or FakeSimulatorTransport()
    gateway = EmbodiedGateway(
        root=tmp_path / "episode",
        env_id="openeta/libero_libero_spatial_task0-v0",
        task="pick the black bowl",
        transport=transport,
        sam3=sam3,
        anygrasp=anygrasp,
    )
    gateway._grasp_inspector = BoundEligibleGraspInspector(gateway)
    return gateway, transport


def test_pointcloud_render_uses_live_grip_pose_and_aperture(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorTransport(open_aperture_m=0.042)
    gateway, _transport = _gateway(tmp_path, transport=transport)

    result = gateway.observe(
        views=["pointcloud_top", "pointcloud_front", "pointcloud_side"]
    )

    assert result.success is True
    assert gateway._current_grip_site_xyz_for_render() == pytest.approx(
        [0.0, 0.0, 0.5]
    )
    assert np.asarray(
        gateway._current_grip_site_rotation_for_render()
    ) == pytest.approx(np.eye(3))
    assert gateway._current_gripper_aperture_for_render() == pytest.approx(0.042)
    for view in gateway._pointcloud_views.values():
        metadata = json.loads(view.metadata_path.read_text(encoding="utf-8"))
        assert metadata["actual_gripper_aperture_m"] == pytest.approx(0.042)
        # The fake transport now publishes the same live collision-geometry
        # contract as LIBERO, so orthographic views must retain the measured
        # inner-pad contact centers as well as aperture.
        assert np.asarray(
            metadata["actual_finger_pad_contact_centers_world_m"]
        ) == pytest.approx(
            np.asarray(
                [
                    [-0.026, 0.0, 0.496],
                    [0.026, 0.0, 0.496],
                ]
            )
        )
    gateway.close()


def test_anygrasp_depth_horizon_fallback_rescales_camera_pose(tmp_path: Path) -> None:
    depth_payload = base64.b64encode(_png_bytes(mode="I;16", value=1200)).decode("ascii")
    mask_payload = base64.b64encode(_png_bytes(mode="L", value=0)).decode("ascii")
    calls: list[dict[str, Any]] = []

    def backend(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(request)
        if len(calls) == 1:
            return {
                "success": False,
                "content": "AnyGrasp grasp detection failed: empty_target_mask.",
                "details": {"reason": "empty_target_mask"},
            }
        return {
            "success": True,
            "content": "AnyGrasp grasp detection completed.",
            "details": {
                "candidate_count": 1,
                "grasp_candidates": [
                    {
                        "id": "backend-0",
                        "frame": "camera",
                        "camera_frame": "opencv",
                        "score": 0.8,
                        "translation_xyz": [0.1, 0.2, 0.9],
                        "rotation_matrix": [
                            [1.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0],
                        ],
                        "depth": 0.03,
                        "width": 0.04,
                        "height": 0.03,
                        "gripper_tip_position_xyz": [0.13, 0.2, 0.9],
                    }
                ],
                "metadata": {},
            },
        }

    wrapped = _build_anygrasp_depth_horizon_fallback(
        backend,
        artifact_root=tmp_path / "fallback",
    )
    result = wrapped(
        {
            "mode": "targeted",
            "rgb": {"format": "png", "base64": base64.b64encode(_png_bytes()).decode("ascii")},
            "depth": {"format": "png", "base64": depth_payload},
            "target_mask": {"format": "png", "base64": mask_payload},
            "intrinsics": {"fx": 30.0, "fy": 30.0, "cx": 16.0, "cy": 16.0, "scale": 1000.0},
        }
    )

    assert result["success"] is True
    assert len(calls) == 2
    candidate = result["details"]["grasp_candidates"][0]
    assert candidate["translation_xyz"] == pytest.approx([0.1333333333, 0.2666666667, 1.2])
    assert candidate["width"] == pytest.approx(0.0533333333)
    assert candidate["depth"] == pytest.approx(0.03)
    assert candidate["gripper_tip_position_xyz"] == pytest.approx([0.1633333333, 0.2666666667, 1.2])
    fallback = result["details"]["metadata"]["client_depth_horizon_fallback"]
    assert fallback["depth_scale_factor"] == pytest.approx(0.75)
    assert Path(fallback["adapted_depth_ref"]).is_file()


def test_object_reference_returns_visual_asset_without_scene_location(tmp_path: Path) -> None:
    texture = (
        tmp_path
        / "LIBERO"
        / "libero"
        / "libero"
        / "assets"
        / "stable_hope_objects"
        / "alphabet_soup"
        / "texture_map.png"
    )
    texture.parent.mkdir(parents=True)
    texture.write_bytes(_png_bytes(value=(20, 40, 80)))
    transport = FakeSimulatorTransport()
    gateway = EmbodiedGateway(
        root=tmp_path / "episode",
        env_id="openeta/libero_libero_object_task0-v0",
        task="pick up the alphabet soup",
        transport=transport,
        sam3=_sam3_response,
        anygrasp=_anygrasp_response,
        libero_dir=tmp_path / "LIBERO",
    )

    result = gateway.inspect_object_reference("alphabet soup")

    assert result.success is True
    assert result.text["matched_asset_names"] == ["alphabet_soup"]
    assert len(result.images) == 1
    assert result.images[0].is_file()
    assert "coordinates" not in result.text
    assert "scene location" in result.text["message"]

    segmented = gateway.segment_object("soup can")
    assert segmented.success is True
    assert segmented.text["canonical_reference_attached"] is True
    assert segmented.text["canonical_reference_target"] == "alphabet soup"
    assert segmented.images[0] != result.images[0]
    assert segmented.images[0].is_file()
    assert "reference_comparisons" in str(segmented.images[0])
    assert "on the left" in segmented.text["message"]


def _events(root: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]


def test_operator_gripper_feedback_reports_measurement_without_contact_guess() -> None:
    feedback = _operator_gripper_feedback(
        {
            "observation": {
                "robot": {
                    "gripper_state": {
                        "open": False,
                        "aperture_m": 0.002549,
                        "finger_qvel": [-0.0032, 0.0033],
                    }
                }
            }
        }
    )

    assert feedback == {
        "open": False,
        "aperture_mm": 2.55,
        "finger_speed_max_m_s": 0.0033,
    }
    assert "contact" not in feedback


def test_gripper_validation_rejects_measured_close_noop() -> None:
    success, message = _gripper_action_payload_ok(
        "close_gripper",
        before={"open": True, "aperture_mm": 73.29},
        after={"open": True, "aperture_mm": 73.29},
    )

    assert success is False
    assert "no effective measured closure" in message


def test_gripper_outcome_separates_completion_from_requested_state() -> None:
    blocked_open = _gripper_action_outcome(
        "open_gripper",
        before={"open": False, "aperture_mm": 9.0},
        after={"open": False, "aperture_mm": 21.1},
        action_completed=True,
    )
    assert blocked_open == {
        "requested_state": "open",
        "action_completed": True,
        "requested_state_reached": False,
        "moved_toward_requested_state": True,
        "blocked_or_in_contact": True,
    }

    reached_open = _gripper_action_outcome(
        "open_gripper",
        before={"open": False, "aperture_mm": 9.0},
        after={"open": True, "aperture_mm": 71.7},
        action_completed=True,
    )
    assert reached_open["requested_state_reached"] is True
    assert reached_open["blocked_or_in_contact"] is False


def test_gripper_outcome_uses_aperture_over_coarse_open_flag() -> None:
    intermediate_open = _gripper_action_outcome(
        "open_gripper",
        before={"open": False, "aperture_mm": 8.0},
        after={"open": True, "aperture_mm": 43.69},
        action_completed=True,
    )
    assert intermediate_open["requested_state_reached"] is False
    assert intermediate_open["moved_toward_requested_state"] is True
    assert intermediate_open["blocked_or_in_contact"] is True

    wide_close = _gripper_action_outcome(
        "close_gripper",
        before={"open": True, "aperture_mm": 71.0},
        after={"open": False, "aperture_mm": 39.84},
        action_completed=True,
    )
    assert wide_close["requested_state_reached"] is False
    assert wide_close["moved_toward_requested_state"] is True
    assert wide_close["blocked_or_in_contact"] is True


def test_gripper_outcome_treats_simulator_close_tolerance_as_near_closed() -> None:
    near_closed = _gripper_action_outcome(
        "close_gripper",
        before={"open": True, "aperture_mm": 71.0},
        after={"open": False, "aperture_mm": 3.47},
        action_completed=True,
    )

    assert near_closed["requested_state_reached"] is True
    assert near_closed["blocked_or_in_contact"] is False


def test_gripper_close_evidence_does_not_claim_grasp_from_full_closure() -> None:
    evidence = _gripper_close_evidence(
        {
            "aperture_mm": 8.4,
            "requested_state_reached": False,
            "blocked_or_in_contact": True,
        }
    )

    assert evidence["state"] == "closure_obstructed"
    assert evidence["semantic_grasp_confirmed"] is False
    assert "compatible with contact" in evidence["interpretation"]
    assert "does not identify" in evidence["interpretation"]
    assert "strong evidence of contact" in evidence["interpretation"]


def test_gripper_close_evidence_distinguishes_near_closed_from_obstructed() -> None:
    near_closed = _gripper_close_evidence(
        {
            "aperture_mm": 1.0,
            "requested_state_reached": True,
            "blocked_or_in_contact": False,
        }
    )
    assert near_closed["state"] == "near_full_closure"
    assert "no positive grasp evidence" in near_closed["interpretation"]

    obstructed = _gripper_close_evidence(
        {
            "aperture_mm": 42.0,
            "requested_state_reached": False,
            "blocked_or_in_contact": True,
        }
    )
    assert obstructed["state"] == "closure_obstructed"
    assert "strong evidence of contact" in obstructed["interpretation"]
    assert "verify target retention" in obstructed["interpretation"]


def test_gripper_validation_allows_wide_object_contact_after_finger_motion() -> None:
    success, message = _gripper_action_payload_ok(
        "close_gripper",
        before={"open": True, "aperture_mm": 79.0},
        after={"open": True, "aperture_mm": 69.5},
    )

    assert success is True
    assert message == ""


def test_task_terminal_reward_overrides_unmet_motion_tolerance() -> None:
    payload = {
        "success": False,
        "terminated": True,
        "reward": 1,
        "reached_target": False,
        "motion_converged": False,
        "position_error_m": 0.0073,
    }

    assert _task_terminal_success(payload) is True
    assert _motion_payload_ok(payload) == (True, "task completed during motion")


def test_motion_feedback_never_reports_zero_error_without_final_eef_pose() -> None:
    success, message = _motion_payload_ok(
        {
            "success": False,
            "reached_target": False,
            "motion_converged": False,
            "end": {"xyz": []},
            "position_error_m": None,
            "pose_feedback_available": False,
            "controller_error": "simulator step returned no final EEF pose",
        }
    )

    assert success is False
    assert "no final EEF pose" in message
    assert "0.000" not in message


def test_mask_side_engagement_reports_shallow_pose_without_rewriting_it() -> None:
    diagnostics = _mask_side_engagement_diagnostics(0.025)

    assert diagnostics["visible_top_to_grip_site_mm"] == 25.0
    assert diagnostics["panda_base_to_grip_site_reference_mm"] == 97.0
    assert diagnostics["engagement_ratio"] == 0.2577
    assert [item["code"] for item in diagnostics["warnings"]] == [
        "low_visible_surface_engagement"
    ]
    assert "shallower or deeper" in diagnostics["warnings"][0]["message"]


def test_mask_side_engagement_warns_when_target_is_below_visible_floor() -> None:
    diagnostics = _mask_side_engagement_diagnostics(
        0.060,
        target_z_m=0.8877,
        mask_low_quantile_z_m=0.9109,
    )

    assert diagnostics["grip_site_minus_mask_low_quantile_mm"] == -23.2
    assert "not collision clearance" in diagnostics[
        "mask_low_quantile_clearance_semantics"
    ]
    assert [item["code"] for item in diagnostics["warnings"]] == [
        "target_below_mask_low_quantile"
    ]


class FakeGraspInspector:
    def __init__(self, state: dict[str, Any], image: Path | None = None) -> None:
        self.value = dict(state)
        self.image = image
        self.configure_calls: list[dict[str, Any]] = []
        self.capture_calls: list[dict[str, Any]] = []
        self.add_pose_calls: list[dict[str, Any]] = []

    def state(self) -> dict[str, Any]:
        return dict(self.value)

    def configure(self, **arguments: Any) -> dict[str, Any]:
        self.configure_calls.append(dict(arguments))
        return {**self.value, "displayed_pose_ids": [arguments.get("focus_pose_id")]}

    def capture_image(self, **arguments: Any) -> tuple[dict[str, Any], Path | None]:
        self.capture_calls.append(dict(arguments))
        return {**self.value, "view_id": "view-1"}, self.image

    def add_pose(self, **arguments: Any) -> dict[str, Any]:
        self.add_pose_calls.append(dict(arguments))
        added_pose = {
            "pose_id": arguments.get("pose_id"),
            "source_id": arguments.get("source_id"),
            "backend": "refined",
            "support_clearance_mm": 28.39,
            "support_clearance_required_mm": 20.0,
            "support_clearance_status": "eligible",
        }
        self.value.setdefault("poses", []).append(added_pose)
        return {
            **self.value,
            "success": True,
            "added_pose_id": arguments.get("pose_id"),
            "added_pose": added_pose,
            "focused_pose_id": arguments.get("pose_id"),
        }


class BoundEligibleGraspInspector:
    """Current-proposal viewer double with mutable support clearance."""

    def __init__(self, gateway: EmbodiedGateway) -> None:
        self.gateway = gateway
        self.clearance_by_grasp: dict[str, float | None] = {}
        self.extra_poses: list[dict[str, Any]] = []

    def _pose(self, grasp_id: str, *, backend: str = "anygrasp") -> dict[str, Any]:
        clearance = self.clearance_by_grasp.get(grasp_id, 28.39)
        return {
            "pose_id": grasp_id if backend == "refined" else f"viewer/{grasp_id}",
            "source_id": grasp_id,
            "backend": backend,
            "support_clearance_mm": clearance,
            "support_clearance_required_mm": 20.0,
            "support_clearance_status": (
                "unknown"
                if clearance is None
                else "eligible"
                if clearance >= 20.0
                else "unsafe"
            ),
        }

    def state(self) -> dict[str, Any]:
        record = self.gateway.current_record or {}
        candidates = (
            self.gateway.latest_grasp.details.get("grasp_candidates", [])
            if self.gateway.latest_grasp is not None
            else []
        )
        poses = [
            self._pose(
                str(item.get("id")),
                backend=(
                    "refined"
                    if item.get("source_backend") == "refined"
                    else "anygrasp"
                ),
            )
            for item in candidates
            if isinstance(item, dict) and item.get("id") is not None
        ]
        poses.extend(self.extra_poses)
        return {
            "success": True,
            "episode_root": str(self.gateway.root),
            "observation_id": record.get("observation_id"),
            "scene_mode": "proposal",
            "proposal_id": self.gateway.latest_grasp_proposal_id,
            "result_id": self.gateway.latest_grasp_result_id,
            "viewer_clients": [],
            "poses": poses,
        }

    def configure(self, **arguments: Any) -> dict[str, Any]:
        return {**self.state(), "displayed_pose_ids": [arguments.get("focus_pose_id")]}

    def capture_image(self, **_arguments: Any) -> tuple[dict[str, Any], Path | None]:
        image = self.gateway.root / "test-grasp-inspection.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
        return {**self.state(), "view_id": "test-view"}, image

    def add_pose(self, **arguments: Any) -> dict[str, Any]:
        pose_id = str(arguments.get("pose_id"))
        pose = self._pose(pose_id, backend="refined")
        pose["source_id"] = arguments.get("source_id")
        self.extra_poses.append(pose)
        return {**self.state(), "added_pose_id": pose_id, "added_pose": pose}


def test_gateway_rejects_grasp_inspector_bound_to_another_scene(
    tmp_path: Path,
) -> None:
    inspector = FakeGraspInspector(
        {
            "success": True,
            "episode_root": str(tmp_path / "another-episode"),
            "observation_id": "obs-1",
        }
    )
    gateway, _transport = _gateway(tmp_path)
    gateway._grasp_inspector = inspector
    gateway.current_record = {"observation_id": "obs-1"}

    result = gateway.capture_grasp_view(camera="top")

    assert result.success is False
    assert result.text["code"] == "stale_scene"
    assert inspector.capture_calls == []


def test_get_grasp_inspector_waits_for_current_execution_scene(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    observation = gateway.observe()

    class SwitchingInspector:
        def __init__(self) -> None:
            self.calls = 0

        def state(self) -> dict[str, Any]:
            self.calls += 1
            return {
                "success": True,
                "episode_root": str(gateway.root),
                "observation_id": (
                    "obs-stale"
                    if self.calls == 1
                    else observation.text["observation_id"]
                ),
                "scene_mode": "execution",
                "viewer_clients": [{"viewer_id": "0", "connected": True}],
            }

    inspector = SwitchingInspector()
    gateway._grasp_inspector = inspector

    result = gateway.get_grasp_inspector()

    assert result.success is True
    assert result.text["observation_id"] == observation.text["observation_id"]
    assert inspector.calls == 2
    gateway.close()


def test_refinement_waits_for_proposal_scene_even_when_observation_matches(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    observation = gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    base_grasp_id = proposed.text["candidate_ids"][0]
    before_ids = [
        item["id"] for item in gateway.latest_grasp.details["grasp_candidates"]
    ]
    inspector = FakeGraspInspector(
        {
            "success": True,
            "episode_root": str(gateway.root),
            "observation_id": observation.text["observation_id"],
            "scene_mode": "execution",
            "viewer_clients": [{"viewer_id": "0", "connected": True}],
        }
    )
    gateway._grasp_inspector = inspector

    result = gateway.refine_grasp_pose(
        base_grasp_id=base_grasp_id,
        translation_delta_mm=[0.0, 0.0, 10.0],
        rotation_delta_deg=[0.0, 0.0, 0.0],
        reason="wait for exact proposal scene",
    )

    assert result.success is False
    assert result.text["reason"] == "inspector_not_ready"
    assert result.text["retryable"] is True
    assert inspector.add_pose_calls == []
    assert [
        item["id"] for item in gateway.latest_grasp.details["grasp_candidates"]
    ] == before_ids
    gateway.close()


def test_proposal_atomically_rebinds_detection_and_refinement_rejects_mask_drift(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    observation = gateway.observe()
    gateway.segment_object("black bowl")

    proposed = gateway.propose_grasps()

    assert proposed.success is True
    assert gateway.selected_detection == proposed.details["selected_detection"]
    assert gateway.selected_detection is not proposed.details["selected_detection"]
    proposal_event = next(
        event
        for event in reversed(_events(gateway.root))
        if event["kind"] == "tool_result"
        and event["payload"]["tool"] == "propose_grasps"
    )
    proposal_summary = proposal_event["payload"]["result"]
    assert proposal_summary["proposal_id"] == gateway.latest_grasp_proposal_id
    assert proposal_summary["result_id"] == gateway.latest_grasp_result_id
    assert proposal_summary["selected_detection"] == gateway.selected_detection
    canonical_ref = Path(proposal_summary["canonical_grasp_candidates_ref"])
    assert canonical_ref.is_file()
    assert str(canonical_ref) in proposal_event["artifact_refs"]
    inspector = FakeGraspInspector(
        {
            "success": True,
            "episode_root": str(gateway.root),
            "observation_id": observation.text["observation_id"],
            "scene_mode": "proposal",
            "proposal_id": gateway.latest_grasp_proposal_id,
            "result_id": gateway.latest_grasp_result_id,
            "viewer_clients": [{"viewer_id": "0", "connected": True}],
        }
    )
    gateway._grasp_inspector = inspector
    gateway.selected_detection = dict(gateway.selected_detection or {})
    gateway.selected_detection["mask_ref"] = "/mask/from-another-detection.png"

    refined = gateway.refine_grasp_pose(
        base_grasp_id=proposed.text["candidate_ids"][0],
        translation_delta_mm=[0.0, 0.0, 5.0],
        rotation_delta_deg=[0.0, 0.0, 0.0],
        reason="must not refine a grasp against a different mask",
    )

    assert refined.success is False
    assert refined.text["reason"] == "invalid_refinement"
    assert "selected detection does not match the active grasp proposal" in refined.text[
        "message"
    ]
    assert inspector.add_pose_calls == []
    gateway.close()


def test_same_observation_reproposal_requires_exact_viewer_proposal_identity(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    observation = gateway.observe()
    gateway.segment_object("black bowl")
    first = gateway.propose_grasps()
    first_proposal_id = gateway.latest_grasp_proposal_id
    first_result_id = gateway.latest_grasp_result_id
    inspector = FakeGraspInspector(
        {
            "success": True,
            "episode_root": str(gateway.root),
            "observation_id": observation.text["observation_id"],
            "scene_mode": "proposal",
            "proposal_id": first_proposal_id,
            "result_id": first_result_id,
            "viewer_clients": [{"viewer_id": "0", "connected": True}],
        }
    )
    gateway._grasp_inspector = inspector

    second = gateway.propose_grasps()

    assert second.success is True
    assert second.text["observation_id"] == first.text["observation_id"]
    assert gateway.latest_grasp_proposal_id != first_proposal_id
    assert gateway.latest_grasp_result_id != first_result_id
    base_grasp_id = second.text["candidate_ids"][0]
    stale = gateway.refine_grasp_pose(
        base_grasp_id=base_grasp_id,
        translation_delta_mm=[0.0, 0.0, 5.0],
        rotation_delta_deg=[0.0, 0.0, 0.0],
        reason="viewer must reload the new proposal",
    )
    assert stale.success is False
    assert stale.text["reason"] == "inspector_not_ready"
    assert "different grasp proposal" in stale.text["message"]
    assert inspector.add_pose_calls == []

    inspector.value.update(
        {
            "proposal_id": gateway.latest_grasp_proposal_id,
            "result_id": gateway.latest_grasp_result_id,
        }
    )
    current = gateway.refine_grasp_pose(
        base_grasp_id=base_grasp_id,
        translation_delta_mm=[0.0, 0.0, 5.0],
        rotation_delta_deg=[0.0, 0.0, 0.0],
        reason="viewer now matches the new proposal",
    )
    assert current.success is True
    assert [call["pose_id"] for call in inspector.add_pose_calls] == ["refined_000"]
    gateway.close()


def test_gateway_allows_exactly_bound_grasp_inspector_capture(
    tmp_path: Path,
) -> None:
    image = tmp_path / "capture.png"
    image.write_bytes(b"png")
    gateway, _transport = _gateway(tmp_path)
    inspector = FakeGraspInspector(
        {
            "success": True,
            "episode_root": str(gateway.root),
            "observation_id": "obs-1",
        },
        image=image,
    )
    gateway._grasp_inspector = inspector
    gateway.current_record = {"observation_id": "obs-1"}

    result = gateway.capture_grasp_view(camera="pose_jaws")

    assert result.success is True
    assert result.images == [image]
    assert inspector.capture_calls == [{"camera": "pose_jaws"}]


def test_gateway_passes_explicit_orbit_to_exactly_bound_inspector(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    inspector = FakeGraspInspector(
        {
            "success": True,
            "episode_root": str(gateway.root),
            "observation_id": "obs-1",
        }
    )
    gateway._grasp_inspector = inspector
    gateway.current_record = {"observation_id": "obs-1"}

    result = gateway.configure_grasp_view(
        pose_scope="focus",
        focus_pose_id="exec-test/target",
        orbit_azimuth_deg=45.0,
        orbit_elevation_deg=15.0,
        zoom_scale=0.7,
    )

    assert result.success is True
    assert inspector.configure_calls == [
        {
            "show_anygrasp": True,
            "show_graspgenx": True,
            "show_refined": True,
            "pose_scope": "focus",
            "camera_preset": "keep",
            "focus_pose_id": "exec-test/target",
            "orbit_azimuth_deg": 45.0,
            "orbit_elevation_deg": 15.0,
            "zoom_scale": 0.7,
        }
    ]


def test_gateway_pipeline_is_observation_bound_and_operator_projection_is_safe(tmp_path: Path) -> None:
    gateway, transport = _gateway(tmp_path)

    observation = gateway.observe()
    assert observation.success is True
    segmented = gateway.segment_object("black bowl")
    assert segmented.success is True
    proposed = gateway.propose_grasps()
    assert proposed.success is True
    grasp_id = proposed.text["candidate_ids"][0]
    selected = gateway.select_grasp(grasp_id, reason="best visible approach")
    assert selected.success is True

    stepped = gateway.step("close_gripper")
    assert stepped.success is True
    checked = gateway.check_task()
    assert checked.success is True
    assert checked.text["task_success"] is False
    assert checked.text["episode_status"] == "running"
    assert gateway.episode_status == "running"
    assert gateway.failure_case is None
    gateway.close()

    root = tmp_path / "episode"
    operator = json.loads((root / "operator.json").read_text())
    operator_text = json.dumps(operator)
    assert "intrinsics" not in operator_text
    assert "extrinsics" not in operator_text
    assert "gripper_state" not in operator_text
    assert "end_effector_pose" not in operator_text

    events = _events(root)
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    assert len({event["event_id"] for event in events}) == len(events)
    assert any(event["kind"] == "operator_choice" for event in events)
    assert any(
        event["kind"] == "tool_result"
        and event["payload"]["tool"] == "close_gripper"
        and event["frame_refs"]["post"]
        for event in events
    )
    assert [event["kind"] for event in events].count("episode_end") == 1
    assert [name for name, _args in transport.calls].count("close_env") == 1


def test_model_can_refine_candidate_and_execute_exact_world_panda_pose(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    observation = gateway.observe()
    inspector_image = tmp_path / "refined-jaws-view.png"
    inspector_image.write_bytes(b"png")
    inspector = FakeGraspInspector(
        {
            "success": True,
            "episode_root": str(gateway.root),
            "observation_id": observation.text["observation_id"],
            "scene_mode": "proposal",
            "viewer_clients": [{"viewer_id": "0"}],
        },
        image=inspector_image,
    )
    gateway._grasp_inspector = inspector
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    inspector.value.update(
        {
            "proposal_id": gateway.latest_grasp_proposal_id,
            "result_id": gateway.latest_grasp_result_id,
        }
    )
    base_grasp_id = proposed.text["candidate_ids"][0]

    refined = gateway.refine_grasp_pose(
        base_grasp_id=base_grasp_id,
        translation_delta_mm=[10.0, 0.0, 0.0],
        rotation_delta_deg=[0.0, 0.0, 0.0],
        frame="grasp_local",
        reason="move Panda closing axis 10 mm toward the visible rim",
    )

    assert refined.success is True
    assert refined.text["pose_id"] == "refined_000"
    candidate = refined.details["candidate"]
    assert candidate["pose_frame"] == "world"
    assert candidate["eef_frame"] == "panda_grip_site"
    assert candidate["parent_grasp_id"] == base_grasp_id
    assert candidate["provenance"] == {
        "origin": "model_refinement",
        "translation_delta_mm": [10.0, 0.0, 0.0],
        "rotation_delta_deg": [0.0, 0.0, 0.0],
        "delta_frame": "grasp_local",
    }
    # AnyGrasp identity rotation maps Panda local +X to world +Y, so a
    # +10 mm local-X refinement must move world Y and nothing else.
    assert candidate["translation_xyz"] == [0.0, 0.01, 1.0]
    assert inspector.add_pose_calls[0]["pose_id"] == "refined_000"
    assert inspector.configure_calls[0]["camera_preset"] == "pose_jaws"
    assert inspector.configure_calls[0]["viewer_id"] == "0"
    assert inspector.capture_calls == [{"camera": "current", "viewer_id": "0"}]
    assert refined.text["inspector_capture_success"] is True
    assert refined.images[-1] == inspector_image

    selected = gateway.select_grasp("refined_000", reason="green pose is clear")
    assert selected.success is True
    checkpoint = gateway.step("move_to_selected_pregrasp")
    assert checkpoint.success is True
    moved = gateway.step("approach_selected_grasp")
    assert moved.success is True
    move_args = [args for name, args in transport.calls if name == "move_to"]
    assert move_args[0]["x"] == pytest.approx(-0.08)
    assert move_args[0]["y"] == 0.01
    assert move_args[0]["z"] == 1.0
    assert move_args[1]["x"] == 0.0
    assert move_args[1]["y"] == 0.01
    assert move_args[1]["z"] == 1.0
    control = json.loads(next((gateway.root / "control").glob("*.json")).read_text())
    assert control["target_semantics"] == "explicit_world_panda_grip_site"
    assert control["control_pose"]["rotation_adapter"] == "none_world_panda_grip_site"
    gateway.close()


def test_refinement_rejects_invalid_delta_without_terminal_failure(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()

    result = gateway.refine_grasp_pose(
        base_grasp_id=proposed.text["candidate_ids"][0],
        translation_delta_mm=[0.0, 0.0, 0.0],
        rotation_delta_deg=[0.0, 0.0, 0.0],
        frame="camera",
    )

    assert result.success is False
    assert result.text["reason"] == "invalid_refinement"
    assert gateway.failure_case is None
    assert gateway.episode_status == "running"
    gateway.close()


def test_gateway_python_mark_is_visible_and_passed_to_explicit_sam3_call(tmp_path: Path) -> None:
    sam3_requests: list[dict[str, Any]] = []

    def sam3(request: dict[str, Any]) -> dict[str, Any]:
        sam3_requests.append(dict(request))
        return _sam3_response(request)

    gateway, _transport = _gateway(tmp_path, sam3=sam3)
    gateway.observe()
    marked = gateway.python_mark_object(
        "mark_box(4, 5, 20, 21, label='black bowl')"
    )
    assert marked.success is True
    assert marked.text["mark_count"] == 1
    assert len(marked.images) == 2
    assert all(path.is_file() for path in marked.images)
    assert marked.images[1].name.endswith(".zoom.png")
    zoom = Image.open(marked.images[1])
    assert zoom.width >= 32
    assert zoom.height >= 32
    assert marked.details["crop_box_xyxy"] == (0, 0, 32, 32)
    assert gateway.current_visualization["kind"] == "manual_mark"

    segmented = gateway.segment_object("black bowl")
    assert segmented.success is True
    assert len(sam3_requests) == 1
    assert sam3_requests[0]["points"] == [{"x": 12.0, "y": 13.0, "label": 1}]
    source_image = Image.open(
        BytesIO(base64.b64decode(sam3_requests[0]["image_base64"]))
    ).convert("RGB")
    assert source_image.getpixel((4, 5))[0] <= source_image.getpixel((4, 5))[1]
    assert segmented.details["manual_marks"][0]["label"] == "black bowl"
    gateway.close()


def test_gateway_point_mark_is_forwarded_as_a_foreground_prompt(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def sam3(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(request))
        return _sam3_response(request)

    gateway, _transport = _gateway(tmp_path, sam3=sam3)
    gateway.observe()
    gateway.python_mark_object("mark_point(12, 14, label='bowl')")

    result = gateway.segment_object("bowl")

    assert result.success is True
    assert gateway.failure_case is None
    assert calls[0]["points"] == [{"x": 12.0, "y": 14.0, "label": 1}]
    gateway.close()


def test_gateway_python_mark_accepts_visual_style_kwargs(tmp_path: Path) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()

    result = gateway.python_mark_object(
        "mark_point(12, 14, label='candidate', color='lime', radius=8, opacity=0.5)"
    )

    assert result.success is True
    assert len(result.images) == 2
    assert result.details["marks"][0]["style"] == {
        "color": "lime",
        "radius": 8,
    }
    gateway.close()


def test_gateway_rejects_multiple_manual_marks_without_guessing(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def sam3(request: dict[str, Any]) -> dict[str, Any]:
        calls.append(dict(request))
        return _sam3_response(request)

    gateway, _transport = _gateway(tmp_path, sam3=sam3)
    gateway.observe()
    gateway.python_mark_object("mark_point(12, 14); mark_point(18, 20)")

    result = gateway.segment_object("bowl")

    assert result.success is False
    assert result.text["reason"] == "single_manual_mark_required"
    assert result.text["retryable"] is True
    assert gateway.failure_case is None
    assert calls == []
    gateway.close()


def test_gateway_reports_explicit_sam3_no_detections(tmp_path: Path) -> None:
    def no_detection_sam3(_request: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": True,
            "content": "SAM3 segmentation completed with no detections.",
            "details": {"detection_count": 0, "detections": [], "artifacts": []},
        }

    gateway, _transport = _gateway(tmp_path, sam3=no_detection_sam3)
    gateway.observe()
    result = gateway.segment_object("black bowl")
    assert result.success is False
    assert result.text["issue_code"] == "no_detections"
    assert result.text["retryable"] is True
    assert gateway.failure_case is None
    assert gateway.issues[-1]["code"] == "no_detections"

    retried = gateway.segment_object("bowl")

    assert retried.success is False
    assert gateway.episode_status == "running"
    gateway.close()


def test_gateway_records_motion_not_converged_and_allows_recovery(tmp_path: Path) -> None:
    transport = FakeSimulatorTransport(
        move_response={
            "success": False,
            "start": {"xyz": [0.0, 0.0, 0.5]},
            "end": {"xyz": [0.0, 0.0, 0.5]},
            "target": {"x": 0.05, "y": 0.0, "z": 1.0},
            "steps_executed": 100,
            "reached_target": False,
            "motion_converged": False,
            "position_error_m": 0.5,
        }
    )
    gateway = EmbodiedGateway(
        root=tmp_path / "episode",
        env_id="openeta/libero_libero_spatial_task0-v0",
        task="pick the black bowl",
        transport=transport,
        sam3=_sam3_response,
        anygrasp=_anygrasp_response,
    )
    gateway._grasp_inspector = BoundEligibleGraspInspector(gateway)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])
    pre_move_observation_id = gateway.current_record["observation_id"]

    moved = gateway.step("move_to_selected_pregrasp")
    assert moved.success is False
    assert moved.text["issue_code"] == "motion_not_converged"
    assert moved.text["retryable"] is True
    assert moved.images
    assert moved.text["observation_id"] != pre_move_observation_id
    assert gateway.episode_status == "running"
    assert gateway.failure_case is None
    assert gateway.issues[-1]["code"] == "motion_not_converged"

    recovered = gateway.step("open_gripper")
    assert recovered.success is True
    assert [name for name, _args in transport.calls].count("gripper_open") == 2
    gateway.close()
    events = _events(tmp_path / "episode")
    assert any(
        event["kind"] == "issue"
        and event["payload"]["code"] == "motion_not_converged"
        for event in events
    )
    failed_move = next(
        event
        for event in events
        if event["kind"] == "tool_result"
        and event["payload"]["tool"] == "move_to_selected_pregrasp"
    )
    assert failed_move["frame_refs"]["post"]


def test_gateway_lifts_from_current_observed_eef_pose_and_returns_image(tmp_path: Path) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()

    lifted = gateway.step("lift_grasp")

    assert lifted.success is True
    move_args = next(args for name, args in transport.calls if name == "move_to")
    assert move_args["x"] == 0.0
    assert move_args["y"] == 0.0
    assert move_args["z"] == 0.6
    assert "roll" not in move_args
    assert lifted.images
    assert lifted.text["motion"] == {
        "position_error_mm": 0.0,
        "steps_executed": 3,
        "settled": True,
    }
    assert "position error 0.0 mm" in lifted.text["message"]
    assert lifted.text["observation_id"] == gateway.current_record["observation_id"]
    assert lifted.text["viser_execution_scene"]
    comparison = next(
        (tmp_path / "episode" / "control" / "execution-comparison").glob(
            "*/comparison.json"
        )
    )
    payload = json.loads(comparison.read_text())
    assert payload["source_grasp_id"] == "unbound-lift"
    assert payload["target_world_from_grip_site"][2][3] == 0.6
    gateway.close()


def test_gateway_executes_anygrasp_center_with_panda_grip_site_axes(tmp_path: Path) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])

    checkpoint = gateway.step("move_to_selected_pregrasp")
    assert checkpoint.success is True
    moved = gateway.step("approach_selected_grasp")

    assert moved.success is True
    action_names = [name for name, _args in transport.calls]
    first_open = action_names.index("gripper_open")
    first_move = action_names.index("move_to")
    assert first_open < first_move
    move_args = [args for name, args in transport.calls if name == "move_to"]
    assert len(move_args) == 2
    # Pregrasp is a single direct move to the 80 mm approach-axis standoff.
    assert move_args[0]["x"] == pytest.approx(-0.08)
    assert move_args[0]["y"] == 0.0
    assert move_args[0]["z"] == pytest.approx(1.0)
    assert move_args[0]["roll"] == 90.0
    assert "max_translation_step_m" not in move_args[0]
    assert move_args[0]["tracking_stall_steps"] == 12
    assert move_args[0]["tracking_min_aligned_progress_m"] == 0.001
    assert move_args[0]["tracking_cross_track_tolerance_m"] == 0.02
    assert move_args[0]["tracking_min_error_improvement_m"] == 0.00005
    # Contact approach uses conservative contact parameters.
    assert move_args[1]["x"] == 0.0
    assert move_args[1]["y"] == 0.0
    assert move_args[1]["z"] == 1.0
    assert move_args[1]["roll"] == 90.0
    assert move_args[1]["pitch"] == 0.0
    assert move_args[1]["yaw"] == 90.0
    assert move_args[1]["max_translation_step_m"] == 0.003
    assert move_args[1]["tracking_stall_steps"] == 6
    assert checkpoint.text["motion_plan"] == "open_staged_pregrasp_checkpoint"
    assert checkpoint.text["motion"]["steps_executed"] == 13
    assert checkpoint.text["approach_gripper"]["aperture_mm"] == 79.0
    assert moved.text["motion"]["steps_executed"] == 3

    control_artifacts = sorted((tmp_path / "episode" / "control").glob("*.json"))
    assert len(control_artifacts) == 2
    controls = [json.loads(path.read_text()) for path in control_artifacts]
    control = next(
        payload
        for payload in controls
        if payload["stage"] == "move_to_selected_pregrasp"
    )
    assert control["approach_open_arguments"]["handle"] == "fake-handle"
    assert control["approach_open_result"]["steps_executed"] == 10
    assert control["approach_gripper"]["aperture_mm"] == 79.0
    assert control["target_semantics"] == "anygrasp_jaw_center_to_panda_grip_site"
    assert control["control_pose"]["eef_frame"] == "panda_grip_site"
    assert control["control_pose"]["rotation_adapter"] == "anygrasp_to_panda_grip_site"
    assert control["control_pose"]["translation_xyz"] == [0.0, 0.0, 1.0]
    assert control["control_pose"]["rotation_matrix"] == [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    gateway.close()


def test_public_one_call_selected_grasp_path_requires_wrist_checkpoint(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])
    calls_before = list(transport.calls)

    blocked = gateway.step("move_to_selected_grasp")

    assert blocked.success is False
    assert blocked.text["reason"] == "wrist_checkpoint_required"
    assert blocked.text["required_next_tool"] == "move_to_selected_pregrasp"
    assert transport.calls == calls_before
    gateway.close()


def test_pregrasp_checkpoint_returns_wrist_and_defers_exact_contact(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    grasp_id = proposed.text["candidate_ids"][0]
    gateway.select_grasp(grasp_id)

    checkpoint = gateway.step("move_to_selected_pregrasp")

    assert checkpoint.success is True
    assert checkpoint.text["checkpoint_status"] == "ready"
    assert checkpoint.text["required_next_tool"] == "approach_selected_grasp"
    assert checkpoint.text["motion_plan"] == "open_staged_pregrasp_checkpoint"
    wrist_checkpoint = checkpoint.text["wrist_checkpoint"]
    assert wrist_checkpoint["evidence_observation_id"] == checkpoint.text[
        "observation_id"
    ]
    assert wrist_checkpoint["selected_grasp_id"] == grasp_id
    assert wrist_checkpoint["grasp_attempt_id"] == checkpoint.text[
        "grasp_attempt_id"
    ]
    assert wrist_checkpoint["frames"]["agentview"]["camera_id"] == "agentview"
    assert wrist_checkpoint["frames"]["agentview"]["frame_id"]
    assert wrist_checkpoint["frames"]["wrist"]["camera_id"] == "wrist"
    assert wrist_checkpoint["frames"]["wrist"]["frame_id"]
    assert wrist_checkpoint["gripper"]["aperture_mm"] == 79.0
    assert wrist_checkpoint["gripper"]["measured_observation_id"] == (
        checkpoint.text["observation_id"]
    )
    assert "centered between the fingers" in checkpoint.text["message"]
    assert len(checkpoint.images) == 2
    assert any("agentview" in path.name for path in checkpoint.images)
    assert any("wrist" in path.name for path in checkpoint.images)
    assert [name for name, _args in transport.calls].count("gripper_open") == 1
    pregrasp_moves = [args for name, args in transport.calls if name == "move_to"]
    assert len(pregrasp_moves) == 1
    assert pregrasp_moves[0]["x"] == pytest.approx(-0.08)
    assert pregrasp_moves[0]["y"] == 0.0
    assert pregrasp_moves[0]["z"] == pytest.approx(1.0)
    assert pregrasp_moves[0]["roll"] == 90.0
    assert pregrasp_moves[0]["tracking_stall_steps"] == 12
    comparison = next(
        (gateway.root / "control" / "execution-comparison").glob(
            "*/comparison.json"
        )
    )
    comparison_payload = json.loads(comparison.read_text())
    assert comparison_payload["execution_stage"] == "move_to_selected_pregrasp"
    assert comparison_payload["target_semantics"] == (
        "selected_grasp_pregrasp_standoff"
    )
    assert comparison_payload["target_world_from_grip_site"][0][3] == (
        pytest.approx(-0.08)
    )
    assert "PREGRASP TARGET/ACTUAL" in checkpoint.text["viser_execution_scene"]
    assert gateway._pending_grasp_approach is not None
    assert gateway._pending_grasp_approach["selected_grasp"]["id"] == grasp_id
    assert gateway._pending_grasp_approach["checkpoint_observation_id"] == (
        checkpoint.text["observation_id"]
    )

    observed = gateway.observe(views=["agentview", "wrist"])
    assert len(observed.images) == 2
    # Mutating the live selection cannot alter the immutable pose approved at
    # the wrist checkpoint.
    assert gateway._pending_grasp_approach is not None
    gateway.selected_grasp = dict(
        gateway._pending_grasp_approach["selected_grasp"]
    )
    gateway.selected_grasp["translation_xyz"] = [9.0, 9.0, 9.0]

    approached = gateway.step("approach_selected_grasp")

    assert approached.success is True
    all_moves = [args for name, args in transport.calls if name == "move_to"]
    assert len(all_moves) == 2
    assert all_moves[1]["x"] == 0.0
    assert all_moves[1]["y"] == 0.0
    assert all_moves[1]["z"] == 1.0
    assert all_moves[1]["max_translation_step_m"] == 0.003
    assert all_moves[1]["tracking_stall_steps"] == 6
    assert gateway._pending_grasp_approach is None
    retry = gateway.step("approach_selected_grasp")
    assert retry.success is False
    assert retry.text["retryable"] is True
    assert len([args for name, args in transport.calls if name == "move_to"]) == 2
    gateway.close()


@pytest.mark.parametrize(
    ("transport_kwargs", "expected_failure"),
    [
        (
            {"post_move_render_missing_wrist": True},
            "post-pregrasp wrist frame is missing",
        ),
        (
            {"post_move_render_missing_aperture": True},
            "post-pregrasp measured gripper aperture is missing",
        ),
        (
            {"post_move_render_aperture_m": 0.042},
            "post-pregrasp measured gripper aperture is insufficient (42.0 mm < 65.0 mm)",
        ),
    ],
)
def test_pregrasp_checkpoint_fails_closed_without_atomic_wrist_evidence(
    tmp_path: Path,
    transport_kwargs: dict[str, Any],
    expected_failure: str,
) -> None:
    transport = FakeSimulatorTransport(**transport_kwargs)
    gateway, _transport = _gateway(tmp_path, transport=transport)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])

    checkpoint = gateway.step("move_to_selected_pregrasp")

    assert checkpoint.success is False
    assert checkpoint.text["checkpoint_status"] == "blocked"
    assert checkpoint.text["issue_code"] == (
        "pregrasp_checkpoint_evidence_incomplete"
    )
    assert checkpoint.text["issue_component"] == "pregrasp_checkpoint"
    evidence = checkpoint.text["wrist_checkpoint"]
    assert evidence["decision"] == "blocked_missing_atomic_evidence"
    assert expected_failure in evidence["evidence_failures"]
    assert evidence["evidence_observation_id"] == checkpoint.text[
        "observation_id"
    ]
    assert evidence["gripper"]["measured_observation_id"] == checkpoint.text[
        "observation_id"
    ]
    assert gateway._pending_grasp_approach is None

    move_count = len(
        [args for name, args in transport.calls if name == "move_to"]
    )
    blocked_approach = gateway.step("approach_selected_grasp")
    assert blocked_approach.success is False
    assert len([args for name, args in transport.calls if name == "move_to"]) == (
        move_count
    )
    gateway.close()


def test_pregrasp_checkpoint_is_invalidated_by_manual_motion_and_open(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])
    assert gateway.step("move_to_selected_pregrasp").success is True

    nudged = gateway.nudge_end_effector(
        [0.0, 0.0, 5.0], reason="wrist shows insufficient vertical clearance"
    )
    assert nudged.success is True
    assert gateway._pending_grasp_approach is None
    assert gateway.step("approach_selected_grasp").success is False

    # Explicit gripper changes also consume any checkpoint because the wrist
    # evidence no longer describes the same approach state.
    gateway._pending_grasp_approach = {"sentinel": True}
    opened = gateway.step("open_gripper")
    assert opened.success is True
    assert gateway._pending_grasp_approach is None
    gateway.close()


def test_active_grasp_actions_and_verification_return_wrist_context(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])

    checkpoint = gateway.step("move_to_selected_pregrasp")
    approached = gateway.step("approach_selected_grasp")
    closed = gateway.step("close_gripper")
    lifted = gateway.step("lift_grasp")
    verified = gateway.verify_grasp()

    for result in (checkpoint, approached, closed, lifted, verified):
        assert any("wrist" in path.name for path in result.images)
    gateway.close()


def test_failed_contact_approach_consumes_checkpoint(tmp_path: Path) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])
    assert gateway.step("move_to_selected_pregrasp").success is True
    transport.move_response = {
        "success": False,
        "start": {"xyz": [-0.08, 0.0, 1.0]},
        "end": {"xyz": [-0.04, 0.02, 1.0]},
        "target": {"x": 0.0, "y": 0.0, "z": 1.0},
        "steps_executed": 8,
        "reached_target": False,
        "motion_converged": False,
        "position_error_m": 0.044,
        "controller_error": "cartesian_tracking_stalled",
    }

    approached = gateway.step("approach_selected_grasp")

    assert approached.success is False
    assert approached.text["issue_code"] == "cartesian_tracking_stalled"
    assert approached.text["motion"]["controller_error"] == (
        "cartesian_tracking_stalled"
    )
    assert gateway._pending_grasp_approach is None
    assert gateway.step("approach_selected_grasp").success is False
    gateway.close()


def test_staged_grasp_stops_before_contact_when_pregrasp_does_not_converge(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorTransport(
        move_response={
            "success": False,
            "start": {"xyz": [0.0, 0.0, 0.5]},
            "end": {"xyz": [-0.04, 0.0, 0.8]},
            "target": {"x": -0.08, "y": 0.0, "z": 1.0},
            "steps_executed": 220,
            "reached_target": False,
            "motion_converged": False,
            "position_error_m": 0.2,
            "settling_converged": False,
        }
    )
    gateway, _transport = _gateway(tmp_path, transport=transport)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])

    moved = gateway.step("move_to_selected_pregrasp")

    assert moved.success is False
    assert moved.text["motion_plan"] == "open_staged_pregrasp_checkpoint"
    assert moved.text["message"].startswith(
        "pregrasp phase move_to did not converge"
    )
    comparison = next(
        (gateway.root / "control" / "execution-comparison").glob(
            "*/comparison.json"
        )
    )
    comparison_payload = json.loads(comparison.read_text())
    assert comparison_payload["execution_stage"] == "move_to_selected_pregrasp"
    assert comparison_payload["target_semantics"] == (
        "selected_grasp_pregrasp_standoff"
    )
    assert moved.text["viser_execution_scene"].startswith(
        "A PREGRASP TARGET/ACTUAL scene"
    )
    assert len(moved.images) == 2
    assert any("agentview" in path.name for path in moved.images)
    assert any("wrist" in path.name for path in moved.images)
    issue_count = len(gateway.issues)
    stale = gateway.report_issue(
        "move_to_selected_pregrasp",
        "capture_timeout",
        "A stale model context did not see the completed action result.",
    )
    assert stale.success is False
    assert stale.text["issue_recorded"] is False
    assert stale.text["reason"] == "stale_action_context"
    authority = stale.text["authoritative_action"]
    assert authority["success"] is False
    assert authority["issue_code"] == "motion_not_converged"
    assert authority["grasp_attempt_id"] == "grasp-attempt-001"
    assert len(gateway.issues) == issue_count
    observed = gateway.observe(views=["agentview", "wrist"])
    assert len(observed.images) == 2
    assert any("wrist" in path.name for path in observed.images)
    assert [name for name, _arguments in transport.calls].count("move_to") == 1
    gateway.close()


def test_selected_grasp_stops_before_motion_when_approach_aperture_is_too_small(
    tmp_path: Path,
) -> None:
    transport = FakeSimulatorTransport(open_aperture_m=0.042)
    gateway, _transport = _gateway(tmp_path, transport=transport)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])

    moved = gateway.step("move_to_selected_pregrasp")

    assert moved.success is False
    assert moved.text["motion_plan"] == "open_staged_pregrasp_checkpoint"
    assert moved.text["approach_gripper"]["aperture_mm"] == 42.0
    assert moved.text["message"].startswith(
        "approach_open phase gripper_open did not establish enough approach clearance"
    )
    assert [name for name, _arguments in transport.calls].count("gripper_open") == 1
    assert [name for name, _arguments in transport.calls].count("move_to") == 0
    control = next((tmp_path / "episode" / "control").glob("*.json"))
    payload = json.loads(control.read_text())
    assert payload["approach_gripper"]["aperture_mm"] == 42.0
    assert payload["remote_result"]["motion_phase"] == "approach_open"
    gateway.close()


def test_execution_comparison_separates_proposed_robot_and_measured_aperture(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()

    comparison = gateway._render_execution_comparison(
        executed_grasp={"id": "AG0", "width": 0.1},
        control_pose={
            "translation_xyz": [0.0, 0.0, 0.5],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
        },
        action_id="aperture-semantics",
    )

    assert comparison is not None
    payload = json.loads(comparison.read_text())
    assert payload["schema_version"] == "openeta.execution_pose_comparison.v2"
    assert payload["proposed_jaw_width_m"] == pytest.approx(0.1)
    assert payload["proposed_aperture_status"] == "exceeds_robot_limit"
    assert payload["robot_max_aperture_m"] == pytest.approx(0.08)
    assert payload["jaw_width_m"] == pytest.approx(0.08)
    assert payload["jaw_width_semantics"] == "robot_clamped_compatibility"
    assert payload["viewer_mesh_aperture_m"] == pytest.approx(0.08)
    assert payload["viewer_mesh_aperture_semantics"] == (
        "physical_open_geometry"
    )
    assert payload["measured_aperture_m"] == pytest.approx(0.079)
    gateway.close()


def test_nudge_end_effector_uses_current_actual_pose_and_can_repeat(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()

    first = gateway.nudge_end_effector(
        [10.0, -20.0, 30.0],
        [0.0, 0.0, 0.0],
        frame="world",
        reason="move closer after visual inspection",
    )
    second = gateway.nudge_end_effector(
        [0.0, 0.0, 10.0],
        frame="eef_local",
        reason="small approach correction",
    )

    assert first.success is True
    assert second.success is True
    assert first.text["kind"] == second.text["kind"] == "nudge_end_effector"
    assert first.text["direct_move_id"] == "M0"
    assert second.text["direct_move_id"] == "M1"
    assert first.text["grasp_attempt_id"] is None
    assert second.text["grasp_attempt_id"] is None
    assert len(first.images) == 2
    assert any("wrist" in path.name for path in first.images)
    assert len(second.images) == 2
    assert any("wrist" in path.name for path in second.images)
    move_calls = [args for name, args in transport.calls if name == "move_to"]
    assert [name for name, _args in transport.calls].count("gripper_open") == 0
    assert len(move_calls) == 2
    assert move_calls[0]["x"] == 0.01
    assert move_calls[0]["y"] == -0.02
    assert move_calls[0]["z"] == 0.53
    assert move_calls[0]["tolerance"] == 0.002
    assert move_calls[0]["num_steps"] == 320
    assert move_calls[1]["x"] == 0.0
    assert move_calls[1]["y"] == 0.0
    assert move_calls[1]["z"] == 0.51
    gateway.close()


def test_nudge_end_effector_can_use_latest_wrist_camera_axes(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    assert gateway.current_record is not None
    wrist = next(
        frame
        for frame in gateway.current_record["frames"]
        if frame["camera_id"] == "wrist"
    )
    # Wrist +X (image-right) points along world +Y in this retained frame.
    wrist["metadata"]["extrinsics"]["mat"] = [
        0.0, -1.0, 0.0,
        1.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
    ]

    nudged = gateway.nudge_end_effector(
        [10.0, 0.0, 0.0],
        frame="wrist_camera",
        reason="move right in the current wrist image",
    )

    assert nudged.success is True
    assert nudged.text["control_mode"] == "nudge_wrist_camera"
    move_call = next(args for name, args in transport.calls if name == "move_to")
    assert move_call["x"] == pytest.approx(0.0)
    assert move_call["y"] == pytest.approx(0.01)
    assert move_call["z"] == pytest.approx(0.5)
    action_path = next((gateway.root / "control").glob("action-*.json"))
    action = json.loads(action_path.read_text())
    provenance = action["selected_grasp"]["provenance"]
    assert provenance["delta_frame"] == "wrist_camera"
    assert provenance["wrist_camera_axes"] == {
        "x": "image_right",
        "y": "image_down",
        "z": "camera_forward",
    }
    gateway.close()


def test_direct_nudge_during_grasp_recovery_inherits_active_attempt(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])
    moved = gateway.step("move_to_selected_pregrasp")

    nudged = gateway.nudge_end_effector(
        [0.0, 0.0, -10.0],
        reason="recover the current grasp approach",
    )

    assert moved.text["grasp_attempt_id"] == "grasp-attempt-001"
    assert nudged.text["grasp_attempt_id"] == "grasp-attempt-001"
    gateway.close()


def test_gateway_records_operator_observed_issue_and_allows_retry(tmp_path: Path) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    frame_ids = list(gateway.current_record["frame_ids"])

    reported = gateway.report_issue(
        "grasp_execution",
        "grasp_not_lifted",
        "Post-lift image shows the bowl remaining on the table.",
    )

    assert reported.success is True
    assert reported.text["recorded"] is True
    assert reported.text["next_step"] == "continue_recovery"
    assert "tool failure" in reported.text["message"]
    assert "issue_code" not in reported.text
    assert "grasp_attempt_id" not in reported.text
    assert reported.images
    assert gateway.episode_status == "running"
    assert gateway.failure_case is None
    assert gateway.issues[-1]["category"] == "operator_observed"
    events = _events(tmp_path / "episode")
    issue = next(event for event in events if event["kind"] == "issue")
    assert issue["frame_refs"]["input"] == frame_ids

    recovered = gateway.step("open_gripper")
    assert recovered.success is True
    gateway.close()


def test_viewer_capture_timeout_report_is_not_blocked_by_action_guard(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()

    reported = gateway.report_issue(
        "grasp_inspector",
        "capture_timeout",
        "The actual Viser capture worker exceeded its render deadline.",
    )

    assert reported.success is True
    assert reported.text["recorded"] is True
    assert gateway.issues[-1]["component"] == "grasp_inspector"
    assert gateway.issues[-1]["code"] == "capture_timeout"
    gateway.close()


def test_operator_can_run_a_second_grasp_attempt_in_the_same_episode(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()

    gateway.segment_object("black bowl")
    first_candidates = gateway.propose_grasps()
    gateway.select_grasp(first_candidates.text["candidate_ids"][0])
    first_checkpoint = gateway.step("move_to_selected_pregrasp")
    assert first_checkpoint.success is True
    first_move = gateway.step("approach_selected_grasp")
    assert first_move.success is True
    assert first_move.text["grasp_attempt_id"] == "grasp-attempt-001"
    first_observation_id = first_move.text["observation_id"]

    issue = gateway.report_issue(
        "grasp_execution",
        "missed_grasp",
        "The object did not move with the gripper; retry with a corrected pose.",
    )
    assert issue.text["recorded"] is True
    assert issue.text["next_step"] == "continue_recovery"
    assert gateway.step("open_gripper").success is True

    gateway.segment_object("black bowl")
    second_candidates = gateway.propose_grasps()
    gateway.select_grasp(second_candidates.text["candidate_ids"][0])
    second_checkpoint = gateway.step("move_to_selected_pregrasp")
    assert second_checkpoint.success is True
    second_move = gateway.step("approach_selected_grasp")

    assert second_move.success is True
    assert second_move.text["grasp_attempt_id"] == "grasp-attempt-002"
    assert second_move.text["observation_id"] != first_observation_id
    assert [name for name, _args in transport.calls].count("move_to") == 4
    assert gateway.episode_status == "running"
    assert gateway.failure_case is None
    assert gateway.issues[-1]["code"] == "missed_grasp"
    gateway.close()


def test_finish_success_requires_native_task_confirmation(tmp_path: Path) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    checked = gateway.check_task()
    assert checked.text["task_success"] is False

    rejected = gateway.finish_episode("success", reason="looks done")

    assert rejected.success is False
    assert rejected.text["retryable"] is True
    assert gateway.episode_status == "running"
    assert [name for name, _args in transport.calls].count("close_env") == 0
    gateway.finish_episode("abort", reason="test cleanup")
    assert gateway.episode_status == "aborted"


def test_finish_episode_records_optional_operator_feedback(tmp_path: Path) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()

    feedback = {
        "tool_contract_issues": ["observe returned more spatial fields than needed"],
        "context_issues": ["The distinction between source view and crop view was unclear"],
        "redundant_information": ["pixel residuals in the normal result"],
        "missing_capabilities": ["a compact local gripper preview"],
        "helpful_evidence": ["wrist image after move_to"],
        "blocked_step": "pregrasp alignment",
        "confidence": "medium",
    }
    finished = gateway.finish_episode(
        "abort",
        reason="feedback-only test",
        operator_feedback=feedback,
    )

    assert finished.success is True
    assert finished.text["operator_feedback"] == feedback
    assert finished.details["operator_feedback"] == feedback
    assert gateway.episode_status == "aborted"


def test_finish_failure_requires_structured_postmortem(tmp_path: Path) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()

    rejected = gateway.finish_episode(
        "failure",
        reason="could not retain the object",
    )

    assert rejected.success is False
    assert rejected.text["reason"] == "failure_postmortem_required"
    assert rejected.text["retryable"] is True
    assert rejected.text["episode_status"] == "running"
    assert set(rejected.text["missing_or_invalid_fields"]) == {
        "progress_stopped_at",
        "expected_observation",
        "actual_observation",
        "evidence_refs",
        "recovery_attempts",
    }
    assert [name for name, _args in transport.calls].count("close_env") == 0

    postmortem = {
        "progress_stopped_at": "retention",
        "expected_observation": "The bowl should rise between the fingers.",
        "actual_observation": "The gripper rose while the bowl remained on the table.",
        "evidence_refs": [
            "obs-000008 agentview: bowl remained on table",
            "obs-000008 wrist: fingers rose without the bowl",
        ],
        "recovery_attempts": [
            "Retried a centered close; bowl did not follow lift.",
            "Retried a rim-offset close; bowl did not follow lift.",
        ],
        "diagnostic_hypotheses": [
            {
                "suspected_layer": "gripper_or_contact",
                "explanation": "The authored grip site did not create opposing contacts.",
                "supporting_evidence": "The fingers closed without carrying the bowl.",
                "missing_or_conflicting_evidence": (
                    "Contact forces and exact collision contacts were not visible."
                ),
                "confidence": "medium",
            }
        ],
        "proposed_intervention": {
            "independent_variable": "stalled-endpoint pad footprint visibility",
            "control_condition": "Show the authored target footprint only.",
            "treatment_condition": (
                "Also show the actual pad footprint at the stalled endpoint."
            ),
            "held_constant": [
                "model",
                "seed schedule",
                "task",
                "point-cloud backend",
                "move_to schema",
                "controller",
            ],
            "predicted_effect": (
                "Operator revises non-straddling grip sites before closing."
            ),
            "primary_metric": "first close-lift target retention rate",
            "adoption_criterion": (
                "Treatment improves first close-lift retention on the predeclared "
                "multi-seed set without increasing schema or identity errors."
            ),
            "execution_scope": "system_change",
            "current_tool_plan": [],
            "attempted_in_episode": False,
            "attempt_evidence_refs": [],
            "planned_current_tool_trials": 0,
            "completed_current_tool_trials": 0,
            "remaining_current_tool_actions": [],
            "contradicting_attempts": [],
            "operator_claimed_exhaustion_reason": "",
        },
    }
    finished = gateway.finish_episode(
        "failure",
        reason="measured retention recovery exhausted",
        failure_postmortem=postmortem,
    )

    assert finished.success is True
    assert finished.text["episode_status"] == "failed"
    recorded = finished.text["failure_postmortem"]
    assert recorded["progress_stopped_at"] == "retention"
    assert recorded["diagnostic_hypotheses"] == postmortem[
        "diagnostic_hypotheses"
    ]
    assert recorded["proposed_intervention"] == {
        **postmortem["proposed_intervention"],
        "evidence_status": "hypothesis_only",
        "adoption_status": "pending_controlled_ab",
    }
    assert gateway.failure_case["details"]["operator_postmortem"] == recorded


def _current_tool_failure_postmortem(
    *,
    attempted_in_episode: bool,
    attempt_evidence_refs: list[str] | None = None,
    exhaustion_reason: str = "",
) -> dict[str, Any]:
    return {
        "progress_stopped_at": "retention",
        "expected_observation": "The bowl should rise between the fingers.",
        "actual_observation": "The gripper rose while the bowl remained on the table.",
        "evidence_refs": [
            "obs-000008 agentview: bowl remained on table",
            "obs-000008 wrist: fingers rose without the bowl",
        ],
        "recovery_attempts": [
            "Retried a centered close; bowl did not follow lift.",
        ],
        "diagnostic_hypotheses": [
            {
                "suspected_layer": "gripper_or_contact",
                "explanation": "The authored grip site may be vertically misaligned.",
                "supporting_evidence": "The fingers closed without carrying the bowl.",
                "missing_or_conflicting_evidence": (
                    "Exact collision contacts were not visible."
                ),
                "confidence": "medium",
            }
        ],
        "proposed_intervention": {
            "independent_variable": "grip-site world-frame Z",
            "control_condition": "Current centered grip-site Z.",
            "treatment_condition": "Same pose with grip-site Z raised by 5 mm.",
            "held_constant": [
                "model",
                "task",
                "object identity",
                "jaw direction",
                "approach direction",
            ],
            "predicted_effect": "The fingers create opposing contacts and retain the bowl.",
            "primary_metric": "close-lift target retention",
            "adoption_criterion": (
                "The treatment retains the intended bowl in repeated matched trials."
            ),
            "execution_scope": "current_tools",
            "current_tool_plan": [
                "move_to the same world-frame pose with Z+5 mm and close",
                "move_to a measured lift and inspect agentview plus wrist",
            ],
            "attempted_in_episode": attempted_in_episode,
            "attempt_evidence_refs": attempt_evidence_refs or [],
            "planned_current_tool_trials": 1,
            "completed_current_tool_trials": (
                1 if attempted_in_episode else 0
            ),
            "remaining_current_tool_actions": (
                [] if attempted_in_episode else [
                    "move_to Z+5 mm, lift, and inspect"
                ]
            ),
            "contradicting_attempts": [],
            "operator_claimed_exhaustion_reason": exhaustion_reason,
        },
    }


def test_finish_failure_rejects_unattempted_current_tool_intervention(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    postmortem = _current_tool_failure_postmortem(
        attempted_in_episode=False,
    )

    rejected = gateway.finish_episode(
        "failure",
        reason="retention recovery stopped",
        failure_postmortem=postmortem,
    )

    assert rejected.success is False
    assert rejected.text["reason"] == "executable_recovery_remaining"
    assert rejected.text["retryable"] is True
    assert rejected.text["episode_status"] == "running"
    assert rejected.text["proposed_intervention"]["execution_scope"] == (
        "current_tools"
    )
    assert gateway.episode_status == "running"
    assert gateway.failure_case is None
    assert [name for name, _args in transport.calls].count("close_env") == 0
    gateway.finish_episode("abort", reason="test cleanup")


def test_finish_failure_accepts_minimal_evidence_without_questionnaire(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()

    finished = gateway.finish_episode(
        "failure",
        reason="Measured recovery was exhausted.",
        failure_postmortem={
            "progress_stopped_at": "retention",
            "expected_observation": "The object should rise with the gripper.",
            "actual_observation": "The gripper rose while the object stayed put.",
            "evidence_refs": ["obs-000002", "wrist-frame-000003"],
            "recovery_attempts": ["Changed the measured grip point and retested lift."],
        },
    )

    assert finished.success is True
    assert finished.text["episode_status"] == "failed"
    assert finished.text["failure_postmortem"] == {
        "progress_stopped_at": "retention",
        "expected_observation": "The object should rise with the gripper.",
        "actual_observation": "The gripper rose while the object stayed put.",
        "evidence_refs": ["obs-000002", "wrist-frame-000003"],
        "recovery_attempts": ["Changed the measured grip point and retested lift."],
    }
    assert [name for name, _args in transport.calls].count("close_env") == 1


def test_finish_failure_requires_evidence_for_claimed_current_tool_attempt(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    postmortem = _current_tool_failure_postmortem(
        attempted_in_episode=True,
        exhaustion_reason="The matched Z scan did not retain the bowl.",
    )

    rejected = gateway.finish_episode(
        "failure",
        reason="retention recovery stopped",
        failure_postmortem=postmortem,
    )

    assert rejected.success is False
    assert rejected.text["reason"] == "failure_postmortem_required"
    assert (
        "proposed_intervention.attempt_evidence_refs"
        in rejected.text["missing_or_invalid_fields"]
    )
    assert gateway.episode_status == "running"
    assert [name for name, _args in transport.calls].count("close_env") == 0
    gateway.finish_episode("abort", reason="test cleanup")


def test_finish_failure_accepts_evidenced_exhausted_current_tool_intervention(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    postmortem = _current_tool_failure_postmortem(
        attempted_in_episode=True,
        attempt_evidence_refs=[
            "move-000014 target Z=0.961 m completed",
            "obs-000015 wrist: bowl remained below the fingers after lift",
        ],
        exhaustion_reason=(
            "The exact +5 mm treatment was executed and failed the declared "
            "retention metric."
        ),
    )

    finished = gateway.finish_episode(
        "failure",
        reason="measured current-tool recovery exhausted",
        failure_postmortem=postmortem,
    )

    assert finished.success is True
    assert finished.text["episode_status"] == "failed"
    intervention = finished.text["failure_postmortem"][
        "proposed_intervention"
    ]
    assert intervention["execution_scope"] == "current_tools"
    assert intervention["attempted_in_episode"] is True
    assert intervention["attempt_evidence_refs"] == postmortem[
        "proposed_intervention"
    ]["attempt_evidence_refs"]
    assert intervention["operator_claimed_exhaustion_reason"].startswith(
        "The exact +5 mm treatment"
    )
    assert [name for name, _args in transport.calls].count("close_env") == 1


def test_finish_failure_rejects_incomplete_declared_current_tool_trials(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    postmortem = _current_tool_failure_postmortem(
        attempted_in_episode=True,
        attempt_evidence_refs=[
            "obs-000015: first matched treatment trial completed",
        ],
        exhaustion_reason="One declared trial was attempted.",
    )
    intervention = postmortem["proposed_intervention"]
    intervention["planned_current_tool_trials"] = 2
    intervention["completed_current_tool_trials"] = 1
    intervention["remaining_current_tool_actions"] = [
        "Repeat the same Z+5 mm close-lift-inspect trial once."
    ]

    rejected = gateway.finish_episode(
        "failure",
        reason="one matched trial failed",
        failure_postmortem=postmortem,
    )

    assert rejected.success is False
    assert rejected.text["reason"] == "executable_recovery_remaining"
    assert rejected.text["episode_status"] == "running"
    returned = rejected.text["proposed_intervention"]
    assert returned["planned_current_tool_trials"] == 2
    assert returned["completed_current_tool_trials"] == 1
    assert returned["remaining_current_tool_actions"] == [
        "Repeat the same Z+5 mm close-lift-inspect trial once."
    ]
    assert gateway.failure_case is None
    assert [name for name, _args in transport.calls].count("close_env") == 0
    gateway.finish_episode("abort", reason="test cleanup")


def test_gateway_rejects_stale_grasp_and_records_local_move_failure(tmp_path: Path) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    grasp_id = proposed.text["candidate_ids"][0]

    gateway.observe()
    stale = gateway.select_grasp(grasp_id)
    assert stale.success is False

    # Rebuild the current observation/perception chain, then force the local
    # pose conversion boundary to fail.  The action and tool result must still
    # be durable instead of escaping as an uncaught exception.
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])
    gateway._selected_move_arguments = lambda: (_ for _ in ()).throw(RuntimeError("bad pose"))  # type: ignore[method-assign]
    result = gateway.step("move_to_selected_pregrasp")
    assert result.success is False
    assert "bad pose" in result.text["message"]
    assert gateway.episode_status == "running"
    assert gateway.failure_case is None
    assert gateway.issues[-1]["component"] == "move_to_selected_pregrasp"

    events = _events(tmp_path / "episode")
    assert any(event["kind"] == "action" for event in events)
    assert any(
        event["kind"] == "tool_result"
        and event["payload"]["tool"] == "move_to_selected_pregrasp"
        and event["payload"]["success"] is False
        for event in events
    )
    gateway.close()


def test_gateway_surfaces_exception_type_when_transport_error_has_no_message(
    tmp_path: Path,
) -> None:
    class TimeoutTransport(FakeSimulatorTransport):
        def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            *,
            timeout_s: float | None = None,
        ) -> dict[str, Any]:
            if name == "move_to":
                raise TimeoutError()
            return super().call_tool(name, arguments, timeout_s=timeout_s)

    gateway, _transport = _gateway(tmp_path, transport=TimeoutTransport())
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    gateway.select_grasp(proposed.text["candidate_ids"][0])

    result = gateway.step("move_to_selected_pregrasp")

    assert result.success is False
    assert result.text["issue_code"] == "remote_completion_unknown"
    assert result.text["retryable"] is False
    assert "not canceled" in result.text["message"]
    assert "completion is unknown" in result.text["message"]
    assert result.details["remote"]["error_type"] == "TimeoutError"
    assert result.details["remote"]["completion_state"] == "unknown"
    assert result.details["remote"]["canceled"] is False
    assert [name for name, _arguments in _transport.calls].count("render_env") == 1
    issue_count = len(gateway.issues)
    duplicate = gateway.report_issue(
        "move_to_selected_pregrasp",
        "remote_completion_unknown",
        "The pending call may have timed out.",
    )
    assert duplicate.success is False
    assert duplicate.text["issue_recorded"] is False
    assert duplicate.text["reason"] == "stale_action_context"
    assert duplicate.text["authoritative_action"]["issue_code"] == (
        "remote_completion_unknown"
    )
    assert len(gateway.issues) == issue_count
    gateway.close()


def test_world_action_guard_rejects_concurrent_robot_actions_and_releases(
    tmp_path: Path,
) -> None:
    class BlockingMoveTransport(FakeSimulatorTransport):
        def __init__(self) -> None:
            super().__init__()
            self.move_started = threading.Event()
            self.release_move = threading.Event()
            self.move_timeouts: list[float | None] = []

        def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            *,
            timeout_s: float | None = None,
        ) -> dict[str, Any]:
            if name == "move_to":
                self.calls.append((name, dict(arguments)))
                self.move_timeouts.append(timeout_s)
                self.move_started.set()
                assert self.release_move.wait(timeout=5.0)
                return dict(self.move_response)
            return super().call_tool(name, arguments, timeout_s=timeout_s)

    transport = BlockingMoveTransport()
    gateway, _transport = _gateway(tmp_path, transport=transport)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    grasp_id = proposed.text["candidate_ids"][0]
    gateway.select_grasp(grasp_id)

    first_result: list[Any] = []
    worker = threading.Thread(
        target=lambda: first_result.append(
            gateway.step("move_to_selected_pregrasp")
        )
    )
    worker.start()
    assert transport.move_started.wait(timeout=2.0)

    # Plan-state operations remain available while the remote world action is
    # in flight, but every second robot action fails fast without transport I/O.
    assert gateway.select_grasp(grasp_id, reason="keep current plan").success is True
    blocked_step = gateway.step("lift_grasp")
    blocked_move = gateway.move_to_pose([0.0, 0.0, 0.8], [0.0, 0.0, 0.0])
    blocked_nudge = gateway.nudge_end_effector([0.0, 0.0, 5.0])
    for blocked in (blocked_step, blocked_move, blocked_nudge):
        assert blocked.success is False
        assert blocked.text["retryable"] is True
        assert blocked.text["issue_code"] == "action_in_progress"
        assert blocked.text["active_action"] == "step"
        assert blocked.text["active_stage"] == "move_to_selected_pregrasp"
        assert blocked.text["active_started_at_unix_s"] > 0
        assert blocked.text["active_duration_s"] >= 0
    issues_before = len(gateway.issues)
    stale_pending = gateway.report_issue(
        "move_to_selected_pregrasp",
        "capture_timeout",
        "No result has appeared yet.",
    )
    assert stale_pending.success is False
    assert stale_pending.text["issue_recorded"] is False
    assert stale_pending.text["reason"] == "world_action_in_progress"
    authoritative = stale_pending.text["authoritative_action"]
    assert authoritative["status"] == "in_progress"
    assert authoritative["stage"] == "move_to_selected_pregrasp"
    assert authoritative["grasp_attempt_id"] == "grasp-attempt-001"
    assert len(gateway.issues) == issues_before
    assert [name for name, _arguments in transport.calls].count("move_to") == 1

    transport.release_move.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    assert first_result[0].success is True
    assert transport.move_timeouts == [420.0]

    stale_completed = gateway.report_issue(
        "move_to_selected_pregrasp",
        "capture_timeout",
        "The old context still thinks the call is pending.",
    )
    assert stale_completed.success is False
    assert stale_completed.text["issue_recorded"] is False
    assert stale_completed.text["reason"] == "stale_action_context"
    completed = stale_completed.text["authoritative_action"]
    assert completed["status"] == "completed"
    assert completed["success"] is True
    assert completed["grasp_attempt_id"] == "grasp-attempt-001"
    assert len(gateway.issues) == issues_before

    # The guard is released in finally after a normal return.
    released = gateway.step("open_gripper")
    assert released.text.get("issue_code") != "action_in_progress"
    gateway.close()


def test_terminal_episode_ignores_late_world_action_completion(
    tmp_path: Path,
) -> None:
    class BlockingLiftTransport(FakeSimulatorTransport):
        def __init__(self) -> None:
            super().__init__()
            self.move_started = threading.Event()
            self.release_move = threading.Event()

        def call_tool(
            self,
            name: str,
            arguments: dict[str, Any],
            *,
            timeout_s: float | None = None,
        ) -> dict[str, Any]:
            if name == "move_to":
                self.calls.append((name, dict(arguments)))
                self.move_started.set()
                assert self.release_move.wait(timeout=5.0)
                return dict(self.move_response)
            return super().call_tool(name, arguments, timeout_s=timeout_s)

    transport = BlockingLiftTransport()
    transport.move_response.update(
        {
            "terminated": True,
            "reward": 1,
        }
    )
    gateway, _transport = _gateway(tmp_path, transport=transport)
    observed = gateway.observe()
    observation_id = observed.text["observation_id"]
    render_count = transport.render_count

    late_result: list[Any] = []
    worker = threading.Thread(
        target=lambda: late_result.append(gateway.step("lift_grasp"))
    )
    worker.start()
    assert transport.move_started.wait(timeout=2.0)

    finished = gateway.finish_episode(
        "failure",
        reason="operator terminated while lift completion is unknown",
        failure_postmortem={
            "progress_stopped_at": "recovery",
            "expected_observation": (
                "The in-flight lift should return an authoritative result."
            ),
            "actual_observation": (
                "Episode finalization began before lift completion was known."
            ),
            "evidence_refs": [
                "active_stage=lift_grasp at terminal invalidation"
            ],
            "recovery_attempts": [
                "No safe recovery was possible while action ownership was unknown."
            ],
            "diagnostic_hypotheses": [
                {
                    "suspected_layer": "unknown",
                    "explanation": (
                        "Action lifecycle visibility may be insufficient for safe recovery."
                    ),
                    "supporting_evidence": (
                        "The lift remained active when finalization was requested."
                    ),
                    "missing_or_conflicting_evidence": (
                        "The eventual lift result was not yet available."
                    ),
                    "confidence": "high",
                }
            ],
            "proposed_intervention": {
                "independent_variable": "active-action completion visibility",
                "control_condition": "Current terminal invalidation behavior.",
                "treatment_condition": (
                    "Expose authoritative action completion before allowing failure."
                ),
                "held_constant": [
                    "model",
                    "task",
                    "controller",
                    "move command",
                ],
                "predicted_effect": (
                    "Fewer episodes finalize while a recoverable action is unresolved."
                ),
                "primary_metric": (
                    "terminal episodes with active_action_unknown"
                ),
                "adoption_criterion": (
                    "Treatment eliminates premature terminal invalidation in the "
                    "predeclared lifecycle test set without deadlocking cleanup."
                ),
                "execution_scope": "system_change",
                "current_tool_plan": [],
                "attempted_in_episode": False,
                "attempt_evidence_refs": [],
                "planned_current_tool_trials": 0,
                "completed_current_tool_trials": 0,
                "remaining_current_tool_actions": [],
                "contradicting_attempts": [],
                "operator_claimed_exhaustion_reason": "",
            },
        },
    )
    assert finished.success is True
    assert finished.text["episode_status"] == "failed"
    assert finished.text["issue_code"] == "active_action_unknown"
    assert finished.text["active_action_unknown"]["active_stage"] == "lift_grasp"
    assert finished.text["active_action_unknown"]["canceled"] is False
    assert not any(name == "close_env" for name, _arguments in transport.calls)

    transport.release_move.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()
    ignored = late_result[0]
    assert ignored.success is False
    assert ignored.text["reason"] == "episode_terminated"
    assert ignored.text["issue_code"] == "ignored_late_completion"
    assert ignored.text["episode_status"] == "failed"
    assert gateway.last_task_success is None
    assert gateway.current_record["observation_id"] == observation_id
    assert transport.render_count == render_count

    diagnostic = Path(ignored.details["diagnostic_ref"])
    assert diagnostic.is_file()
    assert json.loads(diagnostic.read_text())["ignored_late_completion"] is True
    episode = json.loads((gateway.root / "episode.json").read_text())
    assert episode["status"] == "failed"
    assert episode["success"] is False
    events = _events(gateway.root)
    assert events[-1]["kind"] == "episode_end"
    assert not any(
        event["kind"] == "observation"
        and event["payload"].get("source") == "post_lift_grasp"
        for event in events
    )


def test_gateway_marks_sam3_service_failure_as_terminal_failure_case(tmp_path: Path) -> None:
    def failed_sam3(_request: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": False,
            "content": "SAM3 segmentation failed: service unavailable.",
            "details": {
                "reason": "service_unavailable",
                "raw_output_ref": "/tmp/sam3-failure.json",
            },
        }

    gateway, _transport = _gateway(tmp_path, sam3=failed_sam3)
    gateway.observe()
    result = gateway.segment_object("black bowl")

    assert result.success is False
    assert result.text["episode_status"] == "failed"
    assert result.text["failure_component"] == "sam3"
    assert gateway.failure_case is not None
    assert gateway.failure_case["code"] == "service_unavailable"

    episode = json.loads((tmp_path / "episode" / "episode.json").read_text())
    assert episode["status"] == "failed"
    assert episode["success"] is False
    events = _events(tmp_path / "episode")
    failure_events = [event for event in events if event["kind"] == "failure_case"]
    assert len(failure_events) == 1
    assert failure_events[0]["payload"]["component"] == "sam3"
    assert failure_events[0]["payload"]["code"] == "service_unavailable"

    blocked = gateway.propose_grasps()
    assert blocked.success is False
    assert blocked.text["failure_case_id"] == gateway.failure_case["failure_case_id"]
    gateway.close()


def test_gateway_marks_anygrasp_service_failure_and_prevents_action(tmp_path: Path) -> None:
    def failed_anygrasp(_request: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": False,
            "content": "AnyGrasp grasp detection failed: service unavailable.",
            "details": {
                "reason": "service_unavailable",
                "raw_output_ref": "/tmp/anygrasp-failure.json",
            },
        }

    gateway, transport = _gateway(tmp_path, anygrasp=failed_anygrasp)
    gateway.observe()
    gateway.segment_object("black bowl")
    result = gateway.propose_grasps()

    assert result.success is False
    assert result.text["episode_status"] == "failed"
    assert result.text["failure_component"] == "anygrasp"
    assert gateway.failure_case is not None
    assert gateway.failure_case["code"] == "service_unavailable"

    blocked = gateway.step("close_gripper")
    assert blocked.success is False
    assert "failure_case_id" in blocked.text
    assert not any(name == "gripper_close" for name, _args in transport.calls)
    gateway.close()


def test_gateway_keeps_no_grasp_candidates_recoverable(tmp_path: Path) -> None:
    def empty_anygrasp(_request: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": False,
            "content": "AnyGrasp returned no grasp candidates.",
            "details": {
                "reason": "no_grasp_candidates",
                "backend": "anygrasp_mcp",
                "metadata": {"region_point_count": 3},
            },
        }

    gateway, transport = _gateway(tmp_path, anygrasp=empty_anygrasp)
    gateway.observe()
    gateway.segment_object("black bowl")
    result = gateway.propose_grasps()

    assert result.success is False
    assert result.text["episode_status"] == "running"
    assert result.text["retryable"] is True
    assert result.text["issue_code"] == "no_grasp_candidates"
    assert gateway.failure_case is None
    assert gateway.episode_status == "running"

    recovered = gateway.move_to_pose(
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        reason="recover after empty proposal",
    )
    assert recovered.success is True
    assert [name for name, _arguments in transport.calls].count("move_to") == 1
    gateway.close()


def test_terminal_direct_move_has_no_choice_or_state_side_effect(tmp_path: Path) -> None:
    def failed_anygrasp(_request: dict[str, Any]) -> dict[str, Any]:
        return {
            "success": False,
            "content": "AnyGrasp service unavailable.",
            "details": {"reason": "service_unavailable"},
        }

    gateway, transport = _gateway(tmp_path, anygrasp=failed_anygrasp)
    gateway.observe()
    gateway.segment_object("black bowl")
    gateway.propose_grasps()
    before_events = _events(gateway.root)
    before_selection = gateway.selected_grasp

    blocked = gateway.move_to_pose(
        [0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0],
        reason="must remain terminal",
    )

    assert blocked.success is False
    assert blocked.text["episode_status"] == "failed"
    assert gateway.selected_grasp is before_selection
    assert gateway._manual_move_seq == 0
    assert _events(gateway.root) == before_events
    assert not any(name == "move_to" for name, _arguments in transport.calls)
    gateway.close()


def test_gateway_selection_does_not_depend_on_static_pose_rendering(tmp_path: Path) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    grasp_id = proposed.text["candidate_ids"][0]

    result = gateway.select_grasp(grasp_id)

    assert proposed.images == []
    assert proposed.text["inspection_surface"] == "viser"
    assert result.success is True
    assert result.images == []
    assert gateway.selected_grasp is not None
    assert any(event["kind"] == "operator_choice" for event in _events(tmp_path / "episode"))
    gateway.close()


@pytest.mark.parametrize(
    ("clearance_mm", "expected_status"),
    [(0.39, "unsafe"), (None, "unknown")],
)
def test_raw_anygrasp_selection_fails_closed_on_support_clearance(
    tmp_path: Path,
    clearance_mm: float | None,
    expected_status: str,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    grasp_id = proposed.text["candidate_ids"][0]
    inspector = gateway._grasp_inspector
    assert isinstance(inspector, BoundEligibleGraspInspector)
    inspector.clearance_by_grasp[grasp_id] = clearance_mm

    selected = gateway.select_grasp(grasp_id)

    assert selected.success is False
    assert selected.text["issue_code"] == "unsafe_grasp_clearance"
    assert selected.text["support_clearance_status"] == expected_status
    assert selected.text["support_clearance_required_mm"] == 20.0
    assert gateway.selected_grasp is None
    assert not any(name == "move_to" for name, _arguments in transport.calls)
    gateway.close()


def test_refined_unsafe_pose_registers_and_inspects_but_cannot_select(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    inspector = gateway._grasp_inspector
    assert isinstance(inspector, BoundEligibleGraspInspector)
    inspector.clearance_by_grasp["refined_000"] = 0.39

    refined = gateway.refine_grasp_pose(
        base_grasp_id=proposed.text["candidate_ids"][0],
        translation_delta_mm=[0.0, 0.0, 0.0],
        rotation_delta_deg=[0.0, 0.0, 0.0],
    )

    assert refined.success is True
    assert refined.text["support_clearance_mm"] == 0.39
    assert refined.text["support_clearance_status"] == "unsafe"
    assert refined.details["candidate"]["support_clearance_status"] == "unsafe"
    inspected = gateway.inspect_grasp("refined_000")
    assert inspected.success is True
    assert inspected.text["support_clearance_status"] == "unsafe"
    blocked = gateway.select_grasp("refined_000")
    assert blocked.success is False
    assert blocked.text["issue_code"] == "unsafe_grasp_clearance"
    assert blocked.text["support_clearance_status"] == "unsafe"
    gateway.close()


def test_move_rechecks_support_clearance_after_eligible_selection(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    grasp_id = proposed.text["candidate_ids"][0]
    inspector = gateway._grasp_inspector
    assert isinstance(inspector, BoundEligibleGraspInspector)

    selected = gateway.select_grasp(grasp_id)
    assert selected.success is True
    assert selected.text["support_clearance_mm"] == 28.39
    inspector.clearance_by_grasp[grasp_id] = 0.39

    moved = gateway.step("move_to_selected_pregrasp")

    assert moved.success is False
    assert moved.text["issue_code"] == "unsafe_grasp_clearance"
    assert moved.text["support_clearance_status"] == "unsafe"
    assert not any(name == "move_to" for name, _arguments in transport.calls)
    assert gateway._grasp_attempt_seq == 0
    gateway.close()


def test_gateway_inspect_grasp_is_read_only_and_focuses_one_pose(tmp_path: Path) -> None:
    gateway, _transport = _gateway(tmp_path)
    observation = gateway.observe()
    gateway.segment_object("black bowl")
    proposed = gateway.propose_grasps()
    grasp_id = proposed.text["candidate_ids"][0]
    inspector_image = tmp_path / "viser-inspection.png"
    inspector_image.write_bytes(b"png")
    inspector_pose_id = "ps-test/anygrasp/000"
    inspector = FakeGraspInspector(
        {
            "success": True,
            "episode_root": str(gateway.root),
            "observation_id": observation.text["observation_id"],
            "scene_mode": "proposal",
            "proposal_id": gateway.latest_grasp_proposal_id,
            "result_id": gateway.latest_grasp_result_id,
            "viewer_clients": [{"viewer_id": "0", "connected": True}],
            "poses": [
                {
                    "pose_id": inspector_pose_id,
                    "source_id": grasp_id,
                    "backend": "anygrasp",
                    "support_clearance_mm": 28.39,
                    "support_clearance_required_mm": 20.0,
                    "support_clearance_status": "eligible",
                }
            ],
        },
        image=inspector_image,
    )
    gateway._grasp_inspector = inspector

    inspected = gateway.inspect_grasp(grasp_id)

    assert inspected.success is True
    assert inspected.text["kind"] == "grasp_inspection"
    assert inspected.text["inspected_grasp_id"] == grasp_id
    assert inspected.text["inspector_pose_id"] == inspector_pose_id
    assert gateway.selected_grasp is None
    assert gateway.selected_grasp_observation_id == ""
    assert len(inspected.images) == 1
    assert inspected.images[0] == inspector_image
    assert inspector.configure_calls == [
        {
            "show_anygrasp": True,
            "show_graspgenx": True,
            "show_refined": True,
            "pose_scope": "focus",
            "focus_pose_id": inspector_pose_id,
            "camera_preset": "pose_jaws",
            "viewer_id": "0",
        }
    ]
    assert inspector.capture_calls == [{"camera": "current", "viewer_id": "0"}]

    selected = gateway.select_grasp(grasp_id)

    assert selected.success is True
    assert selected.images == []

    events = _events(tmp_path / "episode")
    inspect_event = next(
        event
        for event in events
        if event["kind"] == "tool_result"
        and event["payload"]["tool"] == "inspect_grasp"
        and event["payload"]["success"] is True
    )
    assert str(inspector_image) in inspect_event["artifact_refs"]
    assert any(event["kind"] == "operator_choice" for event in events)
    gateway.close()


def test_move_to_accepts_natural_point_pair_translation_fields(
    tmp_path: Path,
) -> None:
    gateway, transport = _gateway(tmp_path)
    gateway.observe()
    gateway._point_marks_3d.update(
        {
            "A": {
                "point_id": "A",
                "observation_id": gateway.current_record["observation_id"],
                "xyz_m": [0.0, 0.0, 1.0],
            },
            "B": {
                "point_id": "B",
                "observation_id": gateway.current_record["observation_id"],
                "xyz_m": [0.02, -0.03, 1.04],
            },
        }
    )

    result = gateway.move_to_target(
        {
            "position_from_point_id": "A",
            "position_to_point_id": "B",
        }
    )

    assert result.success is True
    assert any(tool == "move_to" for tool, _arguments in transport.calls)
    resolution = result.details["move_to_full_result"]["target_resolution"]
    assert resolution["position_delta_mm"] == pytest.approx(
        [20.0, -30.0, 40.0]
    )
    assert resolution["constraints_used"] == [
        "position_from_point_id",
        "position_to_point_id",
    ]
    assert resolution["position_source"] == (
        "measured_point_pair_delta"
    )
    gateway.close()


def test_move_to_resolves_position_point_id_and_preserves_orientation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    observation_id = gateway.current_record["observation_id"]
    gateway._point_marks_3d["P0"] = {
        "point_id": "P0",
        "observation_id": observation_id,
        "xyz_m": [0.05, -0.02, 1.05],
    }
    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy, reason=reason)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    result = gateway.move_to_target(
        {
            "observation_id": observation_id,
            "position_point_id": "P0",
        },
        reason="point-id test",
    )

    assert result.success is True
    assert captured["position"] == pytest.approx([0.05, -0.02, 1.05])
    full = result.details["move_to_full_result"]
    resolution = full["target_resolution"]
    assert resolution["constraints_used"] == ["position_point_id"]
    assert resolution["inherited_constraints"] == ["current_orientation"]
    assert resolution["point_ids"]["position"] == "P0"
    gateway.close()


def test_move_to_resolves_world_delta_from_actual_grip_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    observation_id = gateway.current_record["observation_id"]
    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy, reason=reason)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    result = gateway.move_to_target(
        {
            "observation_id": observation_id,
            "position_delta_mm": [30.0, -10.0, 25.0],
            "delta_frame": "world",
        },
        reason="world delta test",
    )

    assert result.success is True
    assert captured["position"] == pytest.approx([0.03, -0.01, 0.525])
    full = result.details["move_to_full_result"]
    resolution = full["target_resolution"]
    assert resolution["constraints_used"] == ["position_delta_mm"]
    assert resolution["inherited_constraints"] == ["current_orientation"]
    assert resolution["delta_frame"] == "world"
    assert resolution["base_actual_position_xyz_m"] == pytest.approx(
        [0.0, 0.0, 0.5]
    )
    assert resolution["position_delta_mm"] == pytest.approx([30.0, -10.0, 25.0])
    assert resolution["resolved_target_pose"]["position_xyz_m"] == pytest.approx(
        [0.03, -0.01, 0.525]
    )
    assert full["target_minus_actual_mm_world"] == pytest.approx(
        [30.0, -10.0, 25.0]
    )
    assert full["actual_position_delta_mm_world"] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert "does not by itself prove collision" in full[
        "convergence_interpretation"
    ]
    gateway.close()


def test_move_to_resolves_grip_site_delta_from_actual_axes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway._grip_site_position_delta_enabled = True
    current = np.eye(4, dtype=np.float64)
    current[:3, :3] = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    current[:3, 3] = [0.1, 0.2, 0.5]
    monkeypatch.setattr(
        gateway,
        "_current_world_from_grip_site",
        lambda: current.copy(),
    )
    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy, reason=reason)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    result = gateway.move_to_target(
        {
            "position_delta_mm": [10.0, 20.0, 30.0],
            "delta_frame": "grip_site",
        },
        reason="candidate-frame correction",
    )

    assert result.success is True
    # Current grip-site columns are JAW=+Y, LAT=+X, APP=-Z.
    assert captured["position"] == pytest.approx([0.12, 0.21, 0.47])
    resolution = result.details["move_to_full_result"]["target_resolution"]
    assert resolution["position_source"] == "grip_site_delta"
    assert resolution["delta_frame"] == "grip_site"
    assert resolution["position_delta_mm"] == pytest.approx(
        [10.0, 20.0, 30.0]
    )
    assert resolution["position_delta_mm_world"] == pytest.approx(
        [20.0, 10.0, -30.0]
    )
    assert "position_delta_mm_world" not in result.text
    gateway.close()


def test_move_to_resolves_point_defined_delta_from_actual_grip_site(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    observation_id = gateway.current_record["observation_id"]
    gateway._point_marks_3d["object_center_now"] = {
        "point_id": "object_center_now",
        "observation_id": observation_id,
        "xyz_m": [0.10, 0.20, 0.90],
    }
    gateway._point_marks_3d["destination_center"] = {
        "point_id": "destination_center",
        "observation_id": observation_id,
        "xyz_m": [0.04, 0.24, 0.92],
    }
    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy, reason=reason)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    result = gateway.move_to_target(
        {
            "position_delta_from_point_id": "object_center_now",
            "position_delta_to_point_id": "destination_center",
        },
        reason="measured point delta",
    )

    assert result.success is True
    assert captured["position"] == pytest.approx([-0.06, 0.04, 0.52])
    full = result.details["move_to_full_result"]
    resolution = full["target_resolution"]
    assert resolution["constraints_used"] == [
        "position_delta_from_point_id",
        "position_delta_to_point_id",
    ]
    assert resolution["delta_frame"] == "world"
    assert resolution["base_actual_position_xyz_m"] == pytest.approx(
        [0.0, 0.0, 0.5]
    )
    assert resolution["position_delta_mm"] == pytest.approx(
        [-60.0, 40.0, 20.0]
    )
    assert resolution["point_ids"]["position_delta_from"] == "object_center_now"
    assert resolution["point_ids"]["position_delta_to"] == "destination_center"
    assert resolution["point_provenance_observation_ids"] == {
        "object_center_now": observation_id,
        "destination_center": observation_id,
    }
    assert resolution["point_world_xyz_m"] == {
        "object_center_now": pytest.approx([0.10, 0.20, 0.90]),
        "destination_center": pytest.approx([0.04, 0.24, 0.92]),
    }
    assert "point_references" not in result.text
    gateway.close()


@pytest.mark.parametrize(
    "target",
    [
        {"position_delta_from_point_id": "A"},
        {"position_delta_to_point_id": "B"},
    ],
)
def test_move_to_rejects_incomplete_point_defined_delta(
    tmp_path: Path, target: dict[str, Any]
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    result = gateway.move_to_target(target)
    assert result.success is False
    assert "requires both" in result.text["message"]
    gateway.close()


@pytest.mark.parametrize(
    "target",
    [
        {
            "position_xyz_m": [0.0, 0.0, 1.0],
            "position_delta_mm": [0.0, 0.0, 10.0],
        },
        {
            "position_point_id": "P0",
            "position_delta_mm": [0.0, 0.0, 10.0],
        },
        {
            "position_delta_mm": [0.0, 0.0, 10.0],
            "position_delta_from_point_id": "P0",
            "position_delta_to_point_id": "P1",
        },
    ],
)
def test_move_to_rejects_multiple_position_sources(
    tmp_path: Path, target: dict[str, Any]
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    gateway._point_marks_3d["P0"] = {
        "point_id": "P0",
        "observation_id": gateway.current_record["observation_id"],
        "xyz_m": [0.0, 0.0, 1.0],
    }
    result = gateway.move_to_target(
        {
            **target,
            "observation_id": gateway.current_record["observation_id"],
        }
    )
    assert result.success is False
    assert "use exactly one of" in result.text["message"]
    gateway.close()


def test_move_to_rejects_unknown_target_fields_with_minimal_gripper_example(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()

    result = gateway.move_to_target({"gripper_action": "close"})

    assert result.success is False
    assert result.text["reason"] == "invalid_target"
    assert result.text["unknown_target_fields"] == ["gripper_action"]
    assert '{"gripper":"close"}' in result.text["message"]
    gateway.close()


def test_move_to_rejects_non_world_delta_and_stale_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()

    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy, reason=reason)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    implicit_current = gateway.move_to_target(
        {
            "position_delta_mm": [0.0, 0.0, 10.0],
            "delta_frame": "world",
        }
    )
    assert implicit_current.success is True
    assert captured["position"] == pytest.approx([0.0, 0.0, 0.51])
    full = implicit_current.details["move_to_full_result"]
    resolution = full["target_resolution"]
    assert resolution["observation_id"] == gateway.current_record["observation_id"]
    assert resolution["base_actual_position_xyz_m"] == pytest.approx(
        [0.0, 0.0, 0.5]
    )
    assert resolution["resolved_target_pose"]["position_xyz_m"] == pytest.approx(
        [0.0, 0.0, 0.51]
    )

    non_world = gateway.move_to_target(
        {
            "observation_id": gateway.current_record["observation_id"],
            "position_delta_mm": [0.0, 0.0, 10.0],
            "delta_frame": "eef_local",
        }
    )
    assert non_world.success is False
    assert "'world'" in non_world.text["message"]

    stale = gateway.move_to_target(
        {
            "observation_id": "stale-observation",
            "position_delta_mm": [0.0, 0.0, 10.0],
            "delta_frame": "world",
        }
    )
    assert stale.success is False
    assert stale.text["reason"] == "stale_target"
    gateway.close()


def test_move_to_resolves_three_independent_point_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    observation_id = gateway.current_record["observation_id"]
    for point_id, xyz in {
        "P0": [0.0, 0.0, 1.0],
        "P1": [0.0, 0.0, 0.9],
        "P2": [0.1, 0.0, 1.0],
    }.items():
        gateway._point_marks_3d[point_id] = {
            "point_id": point_id,
            "observation_id": observation_id,
            "xyz_m": xyz,
        }
    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    result = gateway.move_to_target({
        "observation_id": observation_id,
        "position_point_id": "P0",
        "approach_from_point_id": "P1",
        "jaw_toward_point_id": "P2",
    })

    assert result.success is True
    assert captured["position"] == pytest.approx([0.0, 0.0, 1.0])
    full = result.details["move_to_full_result"]
    resolution = full["target_resolution"]
    assert resolution["orientation_resolution_policy"] == "three_marked_points"
    assert set(resolution["constraints_used"]) == {
        "position_point_id",
        "approach_from_point_id",
        "jaw_toward_point_id",
    }
    rotation = np.asarray(resolution["resolved_target_pose"]["rotation_matrix"])
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    gateway.close()


def test_move_to_orientation_only_preserves_actual_grip_site_position(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    observation_id = gateway.current_record["observation_id"]
    current = gateway._current_world_from_grip_site()
    center = current[:3, 3]
    gateway._point_marks_3d["approach"] = {
        "point_id": "approach",
        "observation_id": observation_id,
        "xyz_m": (center - np.asarray([0.0, 0.0, 0.1])).tolist(),
    }
    gateway._point_marks_3d["jaw"] = {
        "point_id": "jaw",
        "observation_id": observation_id,
        "xyz_m": (center + np.asarray([0.1, 0.0, 0.0])).tolist(),
    }
    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    result = gateway.move_to_target(
        {
            "approach_from_point_id": "approach",
            "jaw_toward_point_id": "jaw",
        }
    )

    assert result.success is True
    assert captured["position"] == pytest.approx(center.tolist())
    resolution = result.details["move_to_full_result"]["target_resolution"]
    assert resolution["orientation_resolution_policy"] == (
        "two_marked_direction_anchors_current_position"
    )
    assert "current_position" in resolution["inherited_constraints"]
    assert result.text["actual_directions_world"]["approach"] == pytest.approx(
        [0.0, 0.0, 1.0]
    )
    assert result.text["actual_directions_world"]["jaw"] == pytest.approx(
        [1.0, 0.0, 0.0]
    )
    gateway.close()


def test_move_to_jaw_only_preserves_current_approach_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    current = gateway._current_world_from_grip_site()
    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    result = gateway.move_to_target(
        {"jaw_direction_world": [0.0, 1.0, 0.0]}
    )

    assert result.success is True
    assert captured["position"] == pytest.approx(
        current[:3, 3].tolist()
    )
    resolution = result.details["move_to_full_result"]["target_resolution"]
    assert resolution["orientation_resolution_policy"] == (
        "jaw_preserve_current_approach"
    )
    assert resolution["constraints_used"] == ["jaw_direction"]
    assert resolution["inherited_constraints"] == [
        "approach_axis_from_current"
    ]
    rotation = np.asarray(
        resolution["resolved_target_pose"]["rotation_matrix"]
    )
    assert rotation[:, 2] == pytest.approx(current[:3, 2])
    assert rotation[:, 0] == pytest.approx([0.0, 1.0, 0.0])
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    gateway.close()


def test_move_to_rejects_jaw_parallel_to_current_approach(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    current = gateway._current_world_from_grip_site()

    result = gateway.move_to_target(
        {"jaw_direction_world": current[:3, 2].tolist()}
    )

    assert result.success is False
    assert result.text["reason"] == "invalid_target"
    assert "must not be parallel" in result.text["message"]
    gateway.close()


def test_move_to_preview_resolves_pose_without_motion_or_gripper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    observation_id = gateway.current_record["observation_id"]
    current = gateway._current_world_from_grip_site()
    center = current[:3, 3]
    for point_id, xyz in {
        "P0": center.tolist(),
        "P1": (center - np.asarray([0.0, 0.0, 0.1])).tolist(),
        "P2": (center + np.asarray([0.1, 0.0, 0.0])).tolist(),
    }.items():
        gateway._point_marks_3d[point_id] = {
            "point_id": point_id,
            "observation_id": observation_id,
            "xyz_m": xyz,
        }
    monkeypatch.setattr(
        gateway,
        "move_to_pose",
        lambda *_args, **_kwargs: pytest.fail("preview moved the robot"),
    )
    monkeypatch.setattr(
        gateway,
        "step",
        lambda *_args, **_kwargs: pytest.fail("preview changed the gripper"),
    )
    monkeypatch.setattr(
        gateway,
        "_ensure_operator_pointcloud_views",
        lambda: None,
    )
    gateway._pointcloud_views = {"pointcloud_top": object()}  # type: ignore[assignment]
    preview_path = gateway.root / "pointcloud_views" / "candidate.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_bytes(b"preview")
    monkeypatch.setattr(
        "tools.embodied_gateway.annotate_pose_preview_views",
        lambda *_args, **_kwargs: {"pointcloud_top": preview_path},
    )
    monkeypatch.setattr(
        "tools.embodied_gateway.annotate_active_grip_site_views",
        lambda *_args, **_kwargs: {"pointcloud_top": preview_path},
    )

    result = gateway.move_to_target(
        {
            "observation_id": observation_id,
            "position_point_id": "P0",
            "approach_from_point_id": "P1",
            "jaw_toward_point_id": "P2",
            "preview_only": True,
        },
        views=["pointcloud_top"],
    )

    assert result.success is True
    assert result.text["motion_status"] == "previewed"
    assert result.images == [preview_path]
    assert result.details["move_to_full_result"]["preview_only"] is True
    gateway._pointcloud_views = {}
    gateway.close()


def test_move_to_preview_accepts_gripper_but_does_not_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    observation_id = gateway.current_record["observation_id"]
    current = gateway._current_world_from_grip_site()
    gateway._point_marks_3d["P0"] = {
        "point_id": "P0",
        "observation_id": observation_id,
        "xyz_m": current[:3, 3].tolist(),
    }
    monkeypatch.setattr(gateway, "step", lambda *_a, **_k: pytest.fail("preview changed gripper"))
    monkeypatch.setattr(gateway, "move_to_pose", lambda *_a, **_k: pytest.fail("preview moved robot"))
    monkeypatch.setattr(gateway, "_ensure_operator_pointcloud_views", lambda: None)
    preview_path = gateway.root / "pointcloud_views" / "candidate.png"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_bytes(b"preview")
    monkeypatch.setattr(
        "tools.embodied_gateway.annotate_pose_preview_views",
        lambda *_a, **_k: {"pointcloud_top": preview_path},
    )
    monkeypatch.setattr(
        "tools.embodied_gateway.annotate_active_grip_site_views",
        lambda *_a, **_k: {"pointcloud_top": preview_path},
    )
    gateway._pointcloud_views = {"pointcloud_top": object()}  # type: ignore[assignment]

    result = gateway.move_to_target(
        {
            "observation_id": observation_id,
            "position_point_id": "P0",
            "gripper": "close",
            "preview_only": True,
        },
        views=["pointcloud_top"],
    )

    assert result.success is True
    assert result.text["motion_status"] == "previewed"
    assert "gripper" not in result.text
    gateway._pointcloud_views = {}
    gateway.close()


def test_move_to_close_confirmation_key_is_resolved_pose_bound() -> None:
    current = np.eye(4)
    kwargs = {
        "observation_id": "obs-1",
        "target_position_xyz_m": [0.1, 0.2, 0.9],
        "target_rotation_matrix": np.eye(3),
        "current_world_from_grip_site": current,
    }

    baseline = _move_to_close_confirmation_key(**kwargs)
    assert baseline == _move_to_close_confirmation_key(**kwargs)
    assert baseline != _move_to_close_confirmation_key(
        **{**kwargs, "observation_id": "obs-2"}
    )
    assert baseline != _move_to_close_confirmation_key(
        **{**kwargs, "target_position_xyz_m": [0.1, 0.2, 0.91]}
    )
    moved_current = np.eye(4)
    moved_current[0, 3] = 0.01
    assert baseline != _move_to_close_confirmation_key(
        **{**kwargs, "current_world_from_grip_site": moved_current}
    )


def test_move_to_close_preview_id_and_base_match_are_pose_bound() -> None:
    current = np.eye(4)
    kwargs = {
        "target_position_xyz_m": [0.1, 0.2, 0.9],
        "target_rotation_matrix": np.eye(3),
        "current_world_from_grip_site": current,
    }

    baseline = _move_to_close_preview_id(**kwargs)
    assert baseline == _move_to_close_preview_id(**kwargs)
    assert baseline != _move_to_close_preview_id(
        **{**kwargs, "target_position_xyz_m": [0.1, 0.2, 0.91]}
    )
    assert _move_to_preview_base_matches(current, current.copy()) is True
    moved = current.copy()
    moved[0, 3] = 0.001
    assert _move_to_preview_base_matches(current, moved) is False


def test_pose_preview_artifact_id_versions_rendered_geometry_inputs() -> None:
    kwargs = {
        "observation_id": "obs-1",
        "target_position_xyz_m": [0.1, 0.2, 0.9],
        "target_rotation_matrix": np.eye(3),
        "actual_world_from_grip_site": np.eye(4),
        "requested_gripper": "open",
        "target_pad_contact_centers_world_m": [
            [0.1, 0.16, 0.9],
            [0.1, 0.24, 0.9],
        ],
        "target_pad_sweep_start_centers_world_m": None,
    }

    baseline = _pose_preview_artifact_id(**kwargs)
    assert baseline == _pose_preview_artifact_id(**kwargs)
    assert baseline.startswith("preview-")
    candidate = _pose_preview_artifact_id(
        **kwargs,
        identifier_prefix="candidate",
    )
    assert candidate == baseline.replace("preview-", "candidate-", 1)
    assert baseline != _pose_preview_artifact_id(
        **{**kwargs, "requested_gripper": "close"}
    )
    moved_actual = np.eye(4)
    moved_actual[2, 3] = 0.01
    assert baseline != _pose_preview_artifact_id(
        **{**kwargs, "actual_world_from_grip_site": moved_actual}
    )
    assert baseline != _pose_preview_artifact_id(
        **{
            **kwargs,
            "target_pad_contact_centers_world_m": [
                [0.1, 0.18, 0.9],
                [0.1, 0.22, 0.9],
            ],
        }
    )
    assert baseline != _pose_preview_artifact_id(
        **{
            **kwargs,
            "target_pad_capture_corridor_boxes": [
                {
                    "center_world_m": [0.1, 0.15, 0.9],
                    "rotation_world": np.eye(3).tolist(),
                    "half_size_m": [0.008, 0.004, 0.008],
                },
                {
                    "center_world_m": [0.1, 0.25, 0.9],
                    "rotation_world": np.eye(3).tolist(),
                    "half_size_m": [0.008, 0.004, 0.008],
                },
            ],
        }
    )
    assert baseline != _pose_preview_artifact_id(
        **{
            **kwargs,
            "target_pad_sweep_start_boxes": [
                {
                    "center_world_m": [0.1, 0.15, 0.9],
                    "rotation_world": np.eye(3).tolist(),
                    "half_size_m": [0.008, 0.004, 0.008],
                },
                {
                    "center_world_m": [0.1, 0.25, 0.9],
                    "rotation_world": np.eye(3).tolist(),
                    "half_size_m": [0.008, 0.004, 0.008],
                },
            ],
        }
    )


def test_move_to_rejects_missing_but_accepts_provenanced_world_point(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    missing = gateway.move_to_target({"position_point_id": "P0"})
    assert missing.success is False
    assert "missing marked point ids: P0" in missing.text["message"]

    gateway._point_marks_3d["P0"] = {
        "point_id": "P0",
        "observation_id": "old-observation",
        "xyz_m": [0.0, 0.0, 1.0],
    }
    captured: dict[str, Any] = {}

    def fake_move(position, rpy, *, reason=""):
        captured.update(position=position, rpy=rpy)
        return GatewayResult(True, {"kind": "move_to_pose", "success": True})

    monkeypatch.setattr(gateway, "move_to_pose", fake_move)
    retained = gateway.move_to_target({"position_point_id": "P0"})
    assert retained.success is True
    assert captured["position"] == pytest.approx([0.0, 0.0, 1.0])
    assert retained.details["move_to_full_result"]["target_resolution"][
        "point_provenance_observation_ids"
    ] == {"P0": "old-observation"}
    assert "point_references" not in retained.text
    gateway.close()


def test_native_terminal_success_latches_and_blocks_later_world_actions(
    tmp_path: Path,
) -> None:
    terminal_move = {
        "success": True,
        "start": {"xyz": [0.0, 0.0, 0.5]},
        "end": {"xyz": [0.05, 0.0, 1.0]},
        "target": {"x": 0.05, "y": 0.0, "z": 1.0},
        "steps_executed": 3,
        "reached_target": True,
        "motion_converged": True,
        "position_error_m": 0.0,
        "settling_converged": True,
        "terminated": True,
        "reward": 1.0,
    }
    transport = FakeSimulatorTransport(move_response=terminal_move)
    gateway, _transport = _gateway(tmp_path, transport=transport)
    gateway.observe()

    completed = gateway.move_to_target(
        {"position_xyz_m": [0.05, 0.0, 1.0]},
        views=["agentview", "wrist"],
    )
    assert completed.success is True
    assert completed.text["motion_status"] == "task_completed"
    assert gateway.last_task_success is True

    move_calls_before_retry = [
        name for name, _arguments in transport.calls if name == "move_to"
    ]
    rejected = gateway.move_to_target(
        {"position_delta_mm": [100.0, 0.0, 0.0]},
        views=["agentview", "wrist"],
    )
    assert rejected.success is False
    assert rejected.text["reason"] == "task_already_completed"
    assert rejected.text["retryable"] is False
    assert [
        name for name, _arguments in transport.calls if name == "move_to"
    ] == move_calls_before_retry

    checked = gateway.check_task()
    assert checked.success is True
    assert checked.text["task_success"] is True
    assert [
        name for name, _arguments in transport.calls if name == "check_task"
    ] == []

    failed_finalize = gateway.finish_episode("abort", reason="late retry")
    assert failed_finalize.success is False
    assert failed_finalize.text["reason"] == "task_already_completed"
    finalized = gateway.finish_episode("success", reason="native terminal success")
    assert finalized.success is True
    assert gateway.episode_status == "completed"


def test_native_terminal_success_during_motion_does_not_fail_combined_gripper_call(
    tmp_path: Path,
) -> None:
    terminal_move = {
        "success": True,
        "start": {"xyz": [0.0, 0.0, 0.5]},
        "end": {"xyz": [0.05, 0.0, 1.0]},
        "target": {"x": 0.05, "y": 0.0, "z": 1.0},
        "steps_executed": 3,
        "reached_target": True,
        "motion_converged": True,
        "position_error_m": 0.0,
        "settling_converged": True,
        "terminated": True,
        "reward": 1.0,
    }
    transport = FakeSimulatorTransport(move_response=terminal_move)
    gateway, _transport = _gateway(tmp_path, transport=transport)
    gateway.observe()

    completed = gateway.move_to_target(
        {
            "position_xyz_m": [0.05, 0.0, 1.0],
            "gripper": "open",
        }
    )

    assert completed.success is True
    assert completed.text["motion_status"] == "task_completed"
    assert completed.details["move_to_full_result"]["gripper_executed"] is False
    assert (
        completed.details["move_to_full_result"]["gripper_skip_reason"]
        == "task_already_completed"
    )
    assert gateway.last_task_success is True
    assert [
        name for name, _arguments in transport.calls if name == "gripper_open"
    ] == []
    gateway.finish_episode("success", reason="combined terminal motion")


def test_observe_history_allowlist_is_scoped_to_one_tool_result(
    tmp_path: Path,
) -> None:
    gateway, _transport = _gateway(tmp_path)
    gateway.observe()
    target = np.array([0.0, 0.0, 1.0])
    gateway._point_marks_3d["P0"] = {
        "point_id": "P0",
        "observation_id": gateway.current_record["observation_id"],
        "xyz_m": target.tolist(),
    }
    gateway._current_authored_point_id = "P0"

    historical = gateway.observe(
        views=["agentview", "pointcloud_front"],
        history_point_ids=["P0"],
    )

    assert historical.success is True
    assert any(".marked." in path.name for path in historical.images)
    assert gateway._visible_history_point_ids == set()

    # A later world action captures a fresh observation. Its default visual
    # evidence must not inherit the prior one-call history request.
    moved = gateway.move_to_target(
        {
            "position_delta_mm": [5.0, 0.0, 0.0],
        },
        views=["agentview", "pointcloud_front"],
    )
    assert moved.success is True
    assert all(".marked." not in path.name for path in moved.images)
    assert all(".solved." not in path.name for path in moved.images)
    gateway.close()
