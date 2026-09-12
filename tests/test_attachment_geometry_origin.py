"""Asset origins must not translate the carried collision box away from its shape."""
import copy

import pytest

from sim.mcp_server.collision import check_attached_object_collision
from sim.mcp_server.server import _arm_attachment_proxy


def wine():
    # Goal 2's saved neck grasp: free-joint origin is near the bottle bottom.
    return {
        "name": "wine_bottle_1", "category": "wine_bottle",
        "position": [-.21356609, -.04654448, .89931025],
        "aabb_min": [-.23554661, -.06913044, .90031465],
        "aabb_max": [-.19292885, -.02450836, 1.05739563],
        "dims": [.04261776, .04462208, .15708098],
    }


def close(xyz):
    return {"observation": {"robot": {
        "end_effector_pose": {"xyz": xyz, "quat_xyzw": [0, 0, 0, 1]},
        "gripper_state": {"openness": .1869},
    }}}


def test_recorded_neck_grasp_uses_shape_center_and_arms_collision_protection():
    obj = wine()
    eef = [-.21323866, -.04709407, 1.02829464]
    meta = {}
    receipt = _arm_attachment_proxy(meta, close(eef), authorized_object=obj)
    assert receipt["status"] == "tentative"
    assert receipt["attachment_proven"] is False
    assert receipt["eef_to_target_distance_m"] == pytest.approx(.04945036, abs=2e-5)
    proxy = meta["_attachment_proxy"]
    for i in range(3):
        center = eef[i] + proxy["relative_xyz"][i]
        assert center - proxy["dims"][i] / 2 == pytest.approx(obj["aabb_min"][i])
        assert center + proxy["dims"][i] / 2 == pytest.approx(obj["aabb_max"][i])
    shelf = {"name": "shelf", "category": "shelf",
             "aabb_min": [-.24, -.08, 1.20], "aabb_max": [-.19, -.02, 1.22]}
    hit, _ = check_attached_object_collision(proxy, [shelf], [eef[0], eef[1], 1.20])
    assert hit  # The previous origin-centred box missed the bottle's upper part.


def test_proxy_is_invariant_to_asset_origin_and_stale_dims_when_bounds_exist():
    first, second = wine(), wine()
    second["position"] = [5, 6, 7]
    second["dims"] = [3, 3, 3]
    metas = [{}, {}]
    for meta, obj in zip(metas, (first, second)):
        _arm_attachment_proxy(meta, close([-.213, -.047, 1.028]), authorized_object=obj)
    assert metas[0]["_attachment_proxy"] == metas[1]["_attachment_proxy"]


def test_near_origin_does_not_authorize_distant_geometry():
    obj = wine()
    obj["position"] = [0, 0, 0]
    meta = {}
    receipt = _arm_attachment_proxy(meta, close([0, 0, 0]), authorized_object=obj)
    assert receipt["status"] == "not_armed"
    assert receipt["reason"] == "authorized_target_outside_contact_envelope"
    assert "_attachment_proxy" not in meta


def test_legacy_position_and_dimensions_keep_existing_geometry():
    obj = {"name": "legacy", "category": "box", "position": [0, 0, .1],
           "dims": [.04, .06, .08]}
    original = copy.deepcopy(obj)
    meta = {}
    _arm_attachment_proxy(meta, close([0, 0, .15]), authorized_object=obj)
    assert meta["_attachment_proxy"]["relative_xyz"] == pytest.approx([0, 0, -.05])
    assert meta["_attachment_proxy"]["dims"] == pytest.approx(obj["dims"])
    assert obj == original
