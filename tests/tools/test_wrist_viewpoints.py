from copy import deepcopy
import math

import numpy as np
import pytest
from PIL import Image

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.runtime.memory import AgentMemory
from agent.tools.contracts import (
    build_default_tool_contract_catalog,
    check_tool_request_conformance,
)
from agent.tools.grasp_geometry import GraspGeometryError, propose_wrist_viewpoints
from agent.tools.registry import build_default_tool_registry


def _rz(degrees):
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _inputs(*, yaw=20, mount_yaw=35, translation=(0.02, 0.04, 0.01)):
    rotation = _rz(yaw)
    mount_rotation = _rz(mount_yaw)
    eef_xyz = np.array([0.5, -0.1, 0.4])
    camera = np.eye(4)
    camera[:3, :3] = rotation @ mount_rotation
    camera[:3, 3] = eef_xyz + rotation @ translation
    return {
        "compiled_grasp": {
            "schema_version": "openeta.compiled_grasp_seed.v1",
            "compiled_grasp_id": "compiled-view-target",
            "scene_epoch": 0,
            "target_anchor_world_xyz": [0.4, 0.1, 0.1],
        },
        "source_packet_id": "packet-1",
        "camera_frame_id": "robot0_eye_in_hand",
        "camera_extrinsics": {
            "camera_frame": "opencv", "frame_transform": "camera_to_world",
            "camera_to_world": camera.tolist(),
        },
        "current_eef_pose": {"xyz": eef_xyz.tolist(), "rotation_matrix": rotation.tolist()},
        "object_scene_epoch": 0,
        "robot_motion_epoch": 0,
    }


@pytest.mark.parametrize("standoffs", [0.18, [0.18, 0.22], [0.12, 0.22, 0.35]])
def test_sampler_is_bounded_and_covers_azimuth_elevation_and_distance(standoffs):
    result = propose_wrist_viewpoints({**_inputs(), "standoff_m": standoffs})
    candidates = result["candidates"]
    assert len(candidates) == 8
    offsets = np.array([row["camera_goal"]["offset_world_xyz"] for row in candidates])
    assert offsets[:, 0].min() < -0.04 < 0.04 < offsets[:, 0].max()
    assert offsets[:, 1].min() < -0.04 < 0.04 < offsets[:, 1].max()
    assert len({tuple(row) for row in offsets}) == 8
    assert {row["camera_goal"]["elevation_deg"] for row in candidates} == {40.0, 65.0}
    assert len({row["camera_goal"]["azimuth_world_deg"] for row in candidates}) == 4
    assert len({row["camera_goal"]["target_distance_m"] for row in candidates}) > 1
    for row in candidates:
        goal = row["camera_goal"]
        assert goal["standoff_m"] == goal["offset_world_xyz"][2]
        assert goal["target_distance_m"] == pytest.approx(np.linalg.norm(goal["offset_world_xyz"]), abs=2e-6)
        assert row["requires_ik_preview"] is True
        assert row["requires_collision_check"] is True
        assert row["requires_workspace_check"] is True
        quality = row["view_quality_estimates"]
        assert quality["target_visibility"] == "unverified"
        assert quality["occlusion_quality"] == "unknown"
        assert quality["projected_target_size_px"] is None
    assert result["sampling_policy"]["safety_authorization"] is False
    assert result["geometry_intent"]["kind"] == "active_perception"
    assert result["geometry_intent"]["quality_authorizes_motion"] is False


def test_camera_mount_transform_reconstructs_each_target_facing_view():
    inputs = _inputs()
    result = propose_wrist_viewpoints(inputs)
    mount = result["camera_mount"]
    mount_r = np.array(mount["eef_to_camera_rotation_matrix"])
    mount_t = np.array(mount["eef_to_camera_translation_xyz"])
    current = np.array(inputs["camera_extrinsics"]["camera_to_world"])
    target = np.array(result["target_anchor_world_xyz"])
    for row in result["candidates"]:
        eef_r = np.array(row["target_pose"]["rotation_matrix"])
        eef_t = np.array(row["target_pose"]["xyz"])
        camera_r = eef_r @ mount_r
        camera_t = eef_t + eef_r @ mount_t
        np.testing.assert_allclose(eef_r.T @ eef_r, np.eye(3), atol=2e-6)
        assert np.linalg.det(eef_r) == pytest.approx(1, abs=2e-6)
        np.testing.assert_allclose(camera_t, row["camera_goal"]["xyz"], atol=2e-6)
        delta = target - camera_t
        np.testing.assert_allclose(camera_r[:, 2], delta / np.linalg.norm(delta), atol=5e-6)
        np.testing.assert_allclose((camera_r.T @ delta)[:2], [0, 0], atol=2e-6)
        old_bearing = current[:3, 3] - target
        new_bearing = camera_t - target
        cosine = old_bearing @ new_bearing / (np.linalg.norm(old_bearing) * np.linalg.norm(new_bearing))
        expected_angle = math.degrees(math.acos(np.clip(cosine, -1, 1)))
        assert row["view_quality_estimates"]["view_angle_change_deg"] == pytest.approx(expected_angle, abs=0.001)


def test_sampler_rotates_with_world_yaw_instead_of_fixing_lateral_to_world_x():
    original = _inputs()
    rotated = deepcopy(original)
    rotation = _rz(90)
    pose = rotated["current_eef_pose"]
    pose["xyz"] = (rotation @ pose["xyz"]).tolist()
    pose["rotation_matrix"] = (rotation @ pose["rotation_matrix"]).tolist()
    camera = np.array(rotated["camera_extrinsics"]["camera_to_world"])
    camera[:3, :3] = rotation @ camera[:3, :3]
    camera[:3, 3] = rotation @ camera[:3, 3]
    rotated["camera_extrinsics"]["camera_to_world"] = camera.tolist()
    compiled = rotated["compiled_grasp"]
    compiled["target_anchor_world_xyz"] = (rotation @ compiled["target_anchor_world_xyz"]).tolist()
    for left, right in zip(propose_wrist_viewpoints(original)["candidates"],
                           propose_wrist_viewpoints(rotated)["candidates"], strict=True):
        np.testing.assert_allclose(rotation @ left["camera_goal"]["xyz"], right["camera_goal"]["xyz"], atol=2e-6)
        np.testing.assert_allclose(rotation @ left["target_pose"]["rotation_matrix"], right["target_pose"]["rotation_matrix"], atol=2e-6)


@pytest.mark.parametrize("change", ["eef_rotation", "camera_mount_rotation"])
def test_proposal_id_changes_with_rotation_even_when_translations_and_epochs_match(change):
    first = _inputs(yaw=0, mount_yaw=0, translation=(0, 0, 0))
    second = _inputs(yaw=45 if change == "eef_rotation" else 0,
                     mount_yaw=45 if change == "camera_mount_rotation" else 0,
                     translation=(0, 0, 0))
    a, b = propose_wrist_viewpoints(first), propose_wrist_viewpoints(second)
    assert a["candidates"] != b["candidates"]
    assert a["proposal_id"] != b["proposal_id"]


def test_packet_only_refresh_does_not_rename_identical_geometry():
    inputs = _inputs()
    first = propose_wrist_viewpoints(inputs)
    inputs["source_packet_id"] = "packet-2"
    second = propose_wrist_viewpoints(inputs)
    assert first["proposal_id"] == second["proposal_id"]
    assert first["candidates"] == second["candidates"]


def test_coincident_target_and_camera_has_unknown_view_angle_not_nan():
    inputs = _inputs()
    inputs["compiled_grasp"]["target_anchor_world_xyz"] = (
        np.array(inputs["camera_extrinsics"]["camera_to_world"])[:3, 3].tolist()
    )
    result = propose_wrist_viewpoints(inputs)
    assert len(result["candidates"]) == 8
    assert all(row["view_quality_estimates"]["view_angle_change_deg"] is None
               for row in result["candidates"])


@pytest.mark.parametrize("standoff", [[], True, "0.18", [float("nan")], [0.1], [0.4]])
def test_sampler_rejects_invalid_host_standoffs(standoff):
    with pytest.raises(GraspGeometryError):
        propose_wrist_viewpoints({**_inputs(), "standoff_m": standoff})


def test_agent_cannot_set_candidate_count_or_standoffs():
    contract = build_default_tool_contract_catalog(build_default_tool_registry().list()).get("propose_wrist_viewpoints")
    completed = next(row for row in contract.outcomes if row.semantic_outcome == "completed")
    assert completed.executable_reference is False
    assert completed.output_schema["properties"]["candidates"]["maxItems"] == 8
    for field in ("candidate_count", "standoff_m"):
        assert check_tool_request_conformance(contract, {
            "compiled_grasp_id": "compiled-view-target", "source_packet_id": "packet-1",
            "camera_frame_id": "robot0_eye_in_hand", field: 4,
        })


def _memory_with_proposal(tmp_path):
    inputs = _inputs()
    image_path = tmp_path / "wrist.png"
    Image.new("RGB", (8, 8)).save(image_path)
    observation = EnvObservation(
        task="observe target",
        cameras=[CameraFrame(frame_id=inputs["camera_frame_id"], role="wrist",
                             rgb=[], extrinsics=deepcopy(inputs["camera_extrinsics"]))],
        robot=RobotState(end_effector_pose=deepcopy(inputs["current_eef_pose"])),
        metadata={"image_artifacts": [{
            "kind": "rgb", "frame_id": inputs["camera_frame_id"], "role": "wrist",
            "path": str(image_path), "packet_id": inputs["source_packet_id"],
        }]},
    )
    memory = AgentMemory()
    memory.start_session(task="observe target")
    memory.add_observation(observation)
    memory.artifacts["compiled"] = {"value": {"type": "compiled_grasp", **inputs["compiled_grasp"]}}
    result = propose_wrist_viewpoints(inputs)
    memory.add_action(EnvAction(action_type="tool_call", command={
        "request": {"kind": "tool_call", "name": "propose_wrist_viewpoints"},
        "status": "executed",
        "tool_calls": [{"name": "propose_wrist_viewpoints", "status": "executed",
                        "result": {"success": True, "details": {"outputs": result}}}],
    }))
    return memory, result, observation


def test_actual_generated_candidates_resolve_by_exact_id_and_expire_after_motion(tmp_path):
    memory, result, _ = _memory_with_proposal(tmp_path)
    for row in result["candidates"]:
        resolved = memory.resolve_wrist_viewpoint_candidate(
            proposal_id=result["proposal_id"], candidate_id=row["candidate_id"],
        )
        assert resolved["parameters"]["target_pose"] == row["target_pose"]
    memory.add_action(EnvAction(action_type="tool_call", command={
        "request": {"kind": "tool_call", "name": "move_to"},
        "status": "executed",
        "tool_calls": [{"name": "move_to", "status": "executed",
                        "result": {"success": True, "details": {
                            "outputs": {"motion_summary": {"steps_executed": 1}},
                        }}}],
    }))
    assert memory.robot_motion_epoch() == 1
    with pytest.raises(ValueError, match="robot_motion_epoch"):
        memory.resolve_wrist_viewpoint_candidate(
            proposal_id=result["proposal_id"], candidate_id="wrist_view_00",
        )


def _resolve(memory, result):
    return memory.resolve_wrist_viewpoint_candidate(
        proposal_id=result["proposal_id"], candidate_id="wrist_view_00",
    )


def _recorded_proposal(memory):
    event = next(row for row in reversed(memory.events) if row.event_type == "action")
    return event.payload["command"]["tool_calls"][0]


@pytest.mark.parametrize("epoch", [None, False, "0", 0.5, -1, {}])
@pytest.mark.parametrize("field", ["object_scene_epoch", "robot_motion_epoch"])
def test_viewpoint_missing_or_malformed_epoch_does_not_mean_zero(tmp_path, field, epoch):
    memory, result, _ = _memory_with_proposal(tmp_path)
    _recorded_proposal(memory)["result"]["details"]["outputs"][field] = epoch
    with pytest.raises(ValueError, match=field):
        _resolve(memory, result)


@pytest.mark.parametrize("patch", [
    {"success": False}, {"success": "true"}, {"operational_success": False},
    {"call_status": "failed"},
])
def test_unsuccessful_proposal_cannot_resolve_even_with_complete_geometry(tmp_path, patch):
    memory, result, _ = _memory_with_proposal(tmp_path)
    call = _recorded_proposal(memory)
    if "success" in patch:
        call["result"]["success"] = patch["success"]
    if "operational_success" in patch:
        call["result"]["details"].update(patch)
    if "call_status" in patch:
        call["status"] = patch["call_status"]
    with pytest.raises(ValueError, match="successfully produced"):
        _resolve(memory, result)


def test_resolved_pose_is_not_a_mutable_alias_of_retained_proposal(tmp_path):
    memory, result, _ = _memory_with_proposal(tmp_path)
    resolved = _resolve(memory, result)["parameters"]["target_pose"]
    resolved["xyz"][0] += 10
    resolved["rotation_matrix"][0][0] = 99
    assert _resolve(memory, result)["parameters"]["target_pose"] == result["candidates"][0]["target_pose"]


def test_duplicate_candidate_id_is_rejected(tmp_path):
    memory, result, _ = _memory_with_proposal(tmp_path)
    candidates = _recorded_proposal(memory)["result"]["details"]["outputs"]["candidates"]
    candidates.append(deepcopy(candidates[0]))
    with pytest.raises(ValueError, match="exactly once"):
        _resolve(memory, result)


@pytest.mark.parametrize("change", ["translation", "rotation", "missing"])
def test_mount_changes_without_robot_epoch_invalidate_old_viewpoint(tmp_path, change):
    memory, result, observation = _memory_with_proposal(tmp_path)
    refreshed = deepcopy(observation)
    refreshed.metadata["image_artifacts"][0]["packet_id"] = "packet-2"
    if change == "translation":
        refreshed.cameras[0].extrinsics["camera_to_world"][0][3] += 0.01
    elif change == "rotation":
        matrix = np.array(refreshed.cameras[0].extrinsics["camera_to_world"])
        matrix[:3, :3] = _rz(10) @ matrix[:3, :3]
        refreshed.cameras[0].extrinsics["camera_to_world"] = matrix.tolist()
    else:
        refreshed.cameras[0].extrinsics = {}
    memory.add_observation(refreshed)
    assert memory.robot_motion_epoch() == 0
    with pytest.raises(ValueError, match="mount changed|calibrated viewpoint inputs"):
        _resolve(memory, result)


def test_equivalent_fresh_packet_keeps_exact_old_candidate_resolvable(tmp_path):
    memory, result, observation = _memory_with_proposal(tmp_path)
    refreshed = deepcopy(observation)
    refreshed.metadata["image_artifacts"][0]["packet_id"] = "packet-2"
    memory.add_observation(refreshed)
    assert _resolve(memory, result)["parameters"]["target_pose"] == result["candidates"][0]["target_pose"]


@pytest.mark.parametrize("change", ["compiled_id", "anchor", "contact", "target_geometry"])
def test_changed_compiled_target_cannot_reuse_old_viewpoint(tmp_path, change):
    memory, result, _ = _memory_with_proposal(tmp_path)
    provenance = {
        "evidence_id": "grasp-lineage", "compiled_grasp_id": result["compiled_grasp_id"],
        "target_evidence_id": "sam3:sam-1:det-1", "target_identity_anchor_id": "anchor-1",
        "object_scene_epoch": 0,
    }
    if change == "compiled_id":
        provenance["compiled_grasp_id"] = "compiled-new"
    elif change == "anchor":
        memory.save_fact("selected_sam3_detection", {
            "result_id": "sam-2", "id": "det-2", "identity_anchor_id": "anchor-2",
        }, source="test")
    elif change == "contact":
        provenance["contact_geometry_invalidated_at_s"] = 1.0
    else:
        memory.artifacts["compiled"]["value"]["target_anchor_world_xyz"][0] += 0.1
    memory.save_fact("grasp_provenance", provenance, source="test")
    with pytest.raises(ValueError, match="superseded|no longer current|target geometry changed"):
        _resolve(memory, result)


def test_same_instance_new_mask_does_not_expire_unchanged_view_geometry(tmp_path):
    memory, result, _ = _memory_with_proposal(tmp_path)
    memory.save_fact("grasp_provenance", {
        "evidence_id": "grasp-lineage", "compiled_grasp_id": result["compiled_grasp_id"],
        "target_evidence_id": "sam3:sam-1:det-1", "target_identity_anchor_id": "anchor-1",
        "object_scene_epoch": 0,
    }, source="test")
    memory.save_fact("selected_sam3_detection", {
        "result_id": "sam-2", "id": "det-2", "identity_anchor_id": "anchor-1",
    }, source="test")
    assert _resolve(memory, result)["parameters"]["target_pose"] == result["candidates"][0]["target_pose"]


@pytest.mark.parametrize("field", ["schema_version", "source_packet_id", "camera_mount"])
def test_incomplete_legacy_geometry_cannot_be_repaired_by_guessing(tmp_path, field):
    memory, result, _ = _memory_with_proposal(tmp_path)
    _recorded_proposal(memory)["result"]["details"]["outputs"].pop(field)
    with pytest.raises(ValueError, match="schema|binding|mount"):
        _resolve(memory, result)
