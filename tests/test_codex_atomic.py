import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from test_codex_motion_hook import rig
from test_codex_host import _register_test_camera
from tools.codex_atomic import AtomicTools
from tools.codex_atomic_geometry import orientation, project, surface_point, quat_matrix, matrix_quat
from sim.mcp_server.collision import resolve_contact_authorization


def body(result):
    return json.loads(result.content[0].text)


@pytest.fixture
def atomic(rig, tmp_path):
    host, state = rig
    host.runner.current_observation.robot.end_effector_pose = {
        "xyz": [0., 0., 1.], "quat_xyzw": [0.,0.,0.,1.]}
    _register_test_camera(host, tmp_path, "atomic")
    host.runner.current_observation.cameras[0].extrinsics["camera_frame"] = "opencv"
    host.runtime.memory.add_observation(host.runner.current_observation)
    host.atomic = AtomicTools(host)
    host.schemas = host.atomic.schemas
    host.tool_profile = "atomic"
    return host, state


def mark(host):
    view = body(host.call("episode_status", {}))["views"][0]
    return host.call("mark_point", {"source_packet_id": view["source_packet_id"],
        "camera_frame_id": view["camera_frame_id"], "x": 4, "y": 4})


def test_calibrated_surface_roundtrip_and_depth_edges(tmp_path):
    rgb, depth = tmp_path / "rgb.png", tmp_path / "depth.png"
    Image.new("RGB", (8,8)).save(rgb)
    values = np.full((8,8), 1000, dtype=np.uint16)
    values[3,3] = 1100
    Image.fromarray(values).save(depth)
    source = {"rgb": str(rgb), "depth": str(depth),
              "intrinsics": {"fx": 8, "fy": 8, "cx": 4, "cy": 4},
              "extrinsics": {"camera_frame": "opencv", "camera_to_world":
                             [[0,-1,0,.2],[1,0,0,.3],[0,0,1,.4],[0,0,0,1]]}}
    xyz, info = surface_point(source, 4,4)
    assert np.allclose(xyz, [.2,.3,1.4])
    assert info["depth_edge"]
    assert np.allclose(project(source, xyz), [4,4])
    with pytest.raises(ValueError, match="outside"):
        surface_point(source, 8,4)


def test_arbitrary_approach_and_jaw_form_right_handed_frame():
    matrix = orientation([0,0,0,1], [1,2,-3], [3,-1,1])
    assert np.allclose(matrix.T @ matrix, np.eye(3))
    assert np.isclose(np.linalg.det(matrix), 1)
    assert np.allclose(matrix[:,2], np.array([1,2,-3])/np.sqrt(14))
    with pytest.raises(ValueError):
        orientation([0,0,0,1], [0,0,-1], [0,0,1])


@pytest.mark.parametrize("quat", [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], [.1,-.2,.3,.4]])
def test_quaternion_conversion_roundtrip_including_half_turns(quat):
    matrix = quat_matrix(quat)
    assert np.allclose(quat_matrix(matrix_quat(matrix)), matrix)


def test_bad_crop_remains_recoverable(atomic):
    host, _ = atomic
    # Exercise rendering without a dummy environment replacing the test camera.
    host.atomic.crop = [-1,0,4,4]
    with pytest.raises(ValueError, match="outside"):
        host.atomic.result()
    host.atomic.crop = None
    assert not host.call("episode_status", {}).isError


def test_atomic_surface_only_six_tools_and_no_privileged_state(atomic):
    host, _ = atomic
    assert set(host.schemas) == {"observe", "mark_point", "move_to", "gripper_control", "episode_status", "finish_episode"}
    host.runner.current_observation.objects = [{"name": "SECRET_OBJECT", "position": [123,456,789]}]
    result = host.call("episode_status", {})
    assert "SECRET_OBJECT" not in result.content[0].text
    assert len([x for x in result.content if x.type == "image"]) == 1
    assert host.call("sam3", {}).isError


def test_mark_preview_and_motion_use_existing_budgeted_ik(atomic):
    host, state = atomic
    point = body(mark(host))["feedback"]["point"]
    assert np.allclose(point["xyz_m"], [0,0,1])
    args = {"point_id": point["point_id"], "offset_m": [.02,0,.03],
            "approach_world": [1,2,-3], "jaw_world": [3,-1,1]}
    result = host.call("move_to", {**args, "preview": True})
    assert not result.isError
    assert not state["calls"]
    assert host.runner.tool_call_count == 0
    result = host.call("move_to", args)
    assert not result.isError, body(result)
    assert [name for name,_ in state["calls"]] == ["ik_preview_check", "move_to"]
    assert host.runner.tool_call_count == 3
    assert np.allclose(state["calls"][1][1]["target_pose"]["xyz"], [.02,0,1.03])


def test_failed_ik_and_invalid_directions_do_not_move(atomic):
    host, state = atomic
    assert host.call("move_to", {"approach_world": [0,0,0]}).isError
    assert not state["calls"]
    state.update(feasible=False, reason="ik_search_no_solution")
    assert host.call("move_to", {"delta_m": [.01,0,0]}).isError
    assert [name for name,_ in state["calls"]] == ["ik_preview_check"]


@pytest.mark.parametrize('load_uncertain,rotate,budget', [
    (False, False, 150), (False, True, 150), (True, False, 150), (True, True, 300)])
def test_only_potentially_loaded_reorientation_gets_longer_checked_horizon(
    atomic, load_uncertain, rotate, budget,
):
    host, state = atomic
    host.atomic.symmetry_load_uncertain = load_uncertain
    args = {'delta_m': [.01, 0, 0]}
    if rotate:
        args.update(approach_world=[1, 0, 0], jaw_world=[0, 1, 0])
    result = host.call('move_to', args)
    assert not result.isError, body(result)
    assert [name for name, _ in state['calls']] == ['ik_preview_check', 'move_to']
    assert state['calls'][-1][1].get('num_steps', 150) == budget
    assert state['calls'][-1][1]['enable_collision_check'] is True
    assert host.runner.tool_call_count == 3


def test_contact_provenance_is_scoped_and_rejects_stale_points(atomic):
    host, _ = atomic
    point = body(mark(host))["feedback"]["point"]
    grant = host.atomic.authorization(point["point_id"], [0,0,1])
    assert grant["source_kind"] == "model_rgbd_point" and "compiled_grasp_id" not in grant
    with pytest.raises(ValueError, match="0.10"):
        host.atomic.authorization(point["point_id"], [1,1,1])
    pose = {"xyz": [0,0,1], "quat_xyzw": [0,0,0,1]}
    host.atomic.pending = (pose, grant)
    assert host.atomic.motion_authorization(pose) == grant
    with pytest.raises(ValueError, match="does not match"):
        host.atomic.motion_authorization({**pose, "xyz": [0,0,1.01]})
    host.atomic.pending = None
    assert host.atomic.motion_authorization(pose) is None
    host.atomic.generation += 1
    with pytest.raises(ValueError, match="predates"):
        host.atomic.authorization(point["point_id"], [0,0,1])


def test_old_image_click_is_rejected(atomic, tmp_path):
    host, _ = atomic
    view = body(host.call("episode_status", {}))["views"][0]
    _register_test_camera(host, tmp_path, "new")
    result = host.call("mark_point", {"source_packet_id": view["source_packet_id"],
            "camera_frame_id": view["camera_frame_id"], "x":4, "y":4})
    assert result.isError and "current image" in body(result)["error"]["message"]


def test_point_contact_resolver_requires_evidence_and_rejects_ambiguity():
    grant = {"schema_version": "openeta.model_point_contact.v1", "source_kind": "model_rgbd_point",
             "session_id": "s", "point_id": "p", "source_packet_id": "obs-1", "camera_frame_id": "wrist",
             "waypoint_role": "grasp_contact", "target_anchor_world_xyz": [0,0,1]}
    obj = {"name": "one", "category": "object", "position": [0,0,1], "dims": [.04,.04,.04]}
    selected, receipt = resolve_contact_authorization(grant, [obj])
    assert selected == obj and receipt["source_kind"] == "model_rgbd_point"
    assert resolve_contact_authorization({**grant, "point_id": ""}, [obj])[0] is None
    assert resolve_contact_authorization(grant, [obj, {**obj, "name": "two"}])[0] is None
    assert resolve_contact_authorization({**grant,"target_anchor_world_xyz": [1,0,1]}, [obj])[0] is None
