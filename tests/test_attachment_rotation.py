"""Carried-object endpoint checks use the measured grasp and current body frames."""
import itertools
import math

import numpy as np
import pytest

from sim.mcp_server.collision import (
    _attachment_geometry_at_orientation,
    check_attached_object_collision,
)


IDENTITY = [0., 0., 0., 1.]
TURN_Y = [0., math.sqrt(.5), 0., math.sqrt(.5)]


def proxy(relative=(0, 0, 0), dims=(.15, .04, .04)):
    return {"status": "tentative", "object_name": "private_bottle",
            "relative_xyz": list(relative), "dims": list(dims),
            "anchor_eef_quat_xyzw": IDENTITY}


def test_rotated_bottle_fits_corridor_but_horizontal_bottle_does_not():
    attachment = proxy()
    basket = {"name": "private_basket", "category": "basket",
              "aabb_min": [-.05, -.05, 0], "aabb_max": [.05, .05, .15]}
    old, info = check_attached_object_collision(attachment, [basket], [0, 0, .15])
    assert old and info["placement_constraint"] == "receptacle_corridor_too_narrow"
    assert not check_attached_object_collision(
        attachment, [basket], [0, 0, .15], predicted_eef_quat_xyzw=TURN_Y)[0]


def test_rotating_grasp_offset_into_obstacle_is_not_allowed_as_egress():
    attachment = proxy(relative=(.1, 0, 0), dims=(.02, .02, .02))
    box = {"name": "private_obstacle", "category": "box",
           "position": [0, 0, .1], "dims": [.04, .04, .04]}
    # Current bottle is at x=.1,z=.2; rotation alone brings it to x=0,z=.1.
    assert not check_attached_object_collision(attachment, [box], [0, 0, .2])[0]
    hit, info = check_attached_object_collision(
        attachment, [box], [0, 0, .2], baseline_eef_xyz=[0, 0, .2],
        predicted_eef_quat_xyzw=TURN_Y, baseline_eef_quat_xyzw=IDENTITY)
    assert hit and info["new_or_worsened"]
    assert info["baseline_overlap_volume_m3"] == 0
    assert info["predicted_attached_center_xyz"] == pytest.approx([0, 0, .1])


def test_baseline_rotation_is_used_for_monotonic_escape():
    attachment = proxy(relative=(.1, 0, 0), dims=(.02, .02, .02))
    box = {"name": "private_obstacle", "category": "box",
           "position": [0, 0, .1], "dims": [.04, .04, .04]}
    hit, info = check_attached_object_collision(
        attachment, [box], [0, 0, .22], baseline_eef_xyz=[0, 0, .2],
        predicted_eef_quat_xyzw=TURN_Y, baseline_eef_quat_xyzw=TURN_Y)
    assert not hit and info["egress_from_initial_overlap"]


def test_rotated_aabb_contains_all_original_corners_for_arbitrary_anchor():
    # Independent Rodrigues construction provides the reference, including
    # non-identity anchors and quaternion sign / scale invariance.
    rng = np.random.default_rng(913)
    for _ in range(40):
        matrices, quats = [], []
        for _ in range(2):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(-math.pi, math.pi)
            x, y, z = axis
            cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            matrices.append(np.eye(3) + math.sin(angle)*cross + (1-math.cos(angle))*(cross@cross))
            quats.append([*(axis*math.sin(angle/2)), math.cos(angle/2)])
        attachment = proxy(relative=rng.uniform(-.1, .1, 3), dims=rng.uniform(.02, .2, 3))
        attachment["anchor_eef_quat_xyzw"] = [-3*float(v) for v in quats[0]]
        relative, dims = _attachment_geometry_at_orientation(attachment, [2*float(v) for v in quats[1]])
        delta = matrices[1] @ matrices[0].T
        corners = np.array([delta @ (np.array(attachment["relative_xyz"]) +
                   np.array(sign)*np.array(attachment["dims"])/2)
                   for sign in itertools.product((-1, 1), repeat=3)])
        assert np.allclose(relative, (corners.min(axis=0)+corners.max(axis=0))/2)
        assert np.allclose(dims, corners.max(axis=0)-corners.min(axis=0))


@pytest.mark.parametrize("anchor", [None, [], [0, 0, 0, 0], [0, 0, 0, float('nan')]])
def test_legacy_or_missing_anchor_preserves_translation_only_contract(anchor):
    attachment = proxy(relative=(.1, .2, .3))
    attachment["anchor_eef_quat_xyzw"] = anchor
    assert _attachment_geometry_at_orientation(attachment, TURN_Y) == (
        attachment["relative_xyz"], attachment["dims"])


def test_grasp_captures_measured_body_orientation_without_claiming_attachment():
    from sim.mcp_server.server import _arm_attachment_proxy
    obj = {"name": "private_bottle", "position": [0, 0, .1], "dims": [.05]*3}
    meta = {"_collision_objects": [obj]}
    obs = {"observation": {"robot": {
        "end_effector_pose": {"xyz": [0, 0, .15], "quat_xyzw": TURN_Y},
        "gripper_state": {"openness": .4}}}}
    receipt = _arm_attachment_proxy(meta, obs, authorized_object=obj)
    assert meta["_attachment_proxy"]["anchor_eef_quat_xyzw"] == TURN_Y
    assert receipt["attachment_proven"] is False
    assert "anchor_eef_quat_xyzw" not in receipt


@pytest.mark.parametrize("explicit", [False, True])
def test_server_endpoint_uses_current_or_explicit_orientation(monkeypatch, explicit):
    from sim.mcp_server import server as s
    attachment = proxy()
    meta = {"backend": "libero", "_attachment_proxy": attachment, "_collision_objects": []}
    monkeypatch.setattr(s, "_session_envs", {"test": {"h": meta}})
    monkeypatch.setattr(s, "_touch_session", lambda *a: None)
    monkeypatch.setattr(s, "require_controller_capability", lambda *a, **k: {
        "goal_executor": "openeta.worker_mink_goal.v1"})
    monkeypatch.setattr(s, "cartesian_scales", lambda *a, **k: (.05, .5))
    monkeypatch.setattr(s, "cartesian_command_frame", lambda *a, **k: "world")
    monkeypatch.setattr(s, "_session_last_obs", {"test": {s._obs_key(meta): {
        "observation": {"robot": {"end_effector_pose": {
            "xyz": [0, 0, .3], "quat_xyzw": TURN_Y}}}}}})
    captured = {}
    def check(a, objects, xyz, **kwargs):
        captured.update(kwargs)
        return True, {"message": "test endpoint rejection"}
    monkeypatch.setattr(s, "check_attached_object_collision", check)
    def forbidden(*a, **k):
        raise AssertionError("rejected endpoint dispatched")
    monkeypatch.setattr(s, "_proxy_controller_goal", forbidden)
    angles = {"roll": 0, "pitch": 0, "yaw": 0} if explicit else {}
    result = s.move_to.__wrapped__(handle="h", session_id="test", x=0, y=0, z=.2, **angles)
    assert result["code"] == "attached_object_endpoint_collision"
    assert captured["baseline_eef_quat_xyzw"] == TURN_Y
    assert captured["predicted_eef_quat_xyzw"] == pytest.approx(IDENTITY if explicit else TURN_Y)
