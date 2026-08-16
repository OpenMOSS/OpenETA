from __future__ import annotations

from sim.mcp_server.collision import check_attached_object_collision
from sim.mcp_server.server import _arm_attachment_proxy, _refresh_attachment_proxy


def _observation(eef_xyz, openness: float):
    return {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": list(eef_xyz)},
                "gripper_state": {"openness": openness, "open": openness > 0.5},
            }
        }
    }


def test_attachment_proxy_requires_post_close_co_motion() -> None:
    meta = {
        "_collision_objects": [
            {
                "name": "milk_1",
                "category": "milk",
                "position": [0.0, 0.0, 0.10],
                "dims": [0.06, 0.06, 0.12],
            }
        ]
    }
    _arm_attachment_proxy(meta, _observation([0.0, 0.0, 0.17], 0.63))
    assert meta["_attachment_proxy"]["status"] == "tentative"

    meta["_collision_objects"][0]["position"] = [0.0, 0.0, 0.12]
    _refresh_attachment_proxy(meta, _observation([0.0, 0.0, 0.19], 0.63))
    assert meta["_attachment_proxy"]["status"] == "confirmed"

    _refresh_attachment_proxy(meta, _observation([0.0, 0.08, 0.19], 0.01))
    assert "_attachment_proxy" not in meta


def test_attached_object_collision_blocks_rim_but_allows_centered_entry() -> None:
    attachment = {
        "status": "confirmed",
        "object_name": "milk_1",
        "relative_xyz": [0.0, 0.0, -0.07],
        "dims": [0.06, 0.06, 0.14],
    }
    basket = {
        "name": "basket_1",
        "category": "basket",
        "aabb_min": [-0.12, 0.15, -0.01],
        "aabb_max": [0.12, 0.36, 0.16],
    }

    high_collision, _ = check_attached_object_collision(
        attachment,
        [basket],
        [0.0, 0.10, 0.34],
    )
    assert high_collision is False

    rim_collision, info = check_attached_object_collision(
        attachment,
        [basket],
        [0.0, 0.15, 0.24],
    )
    assert rim_collision is True
    assert info["collision_type"] == "attached_object_world"
    assert info["obstacle"] == "basket_1"
    assert "Raise or reroute" in info["message"]

    centered_collision, _ = check_attached_object_collision(
        attachment,
        [basket],
        [0.0, 0.255, 0.24],
    )
    assert centered_collision is False
