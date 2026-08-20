from __future__ import annotations

from sim.mcp_server.collision import (
    check_attached_object_collision,
    resolve_contact_authorization,
)
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


def test_attachment_proxy_stays_tentative_until_independent_visual_verdict() -> None:
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

    # The reset-time object catalogue is deliberately stale.  Even a large EEF
    # lift must not let the host invent either co-motion or a drop verdict.
    receipt = _refresh_attachment_proxy(meta, _observation([0.0, 0.0, 0.25], 0.63))
    assert meta["_attachment_proxy"]["status"] == "tentative"
    assert receipt["status"] == "tentative"
    assert receipt["reason"] == "awaiting_independent_co_motion_evidence"
    assert abs(receipt["eef_displacement_since_close_m"] - 0.08) < 1e-9
    assert receipt["attachment_proven"] is False

    retired = _refresh_attachment_proxy(meta, _observation([0.0, 0.08, 0.25], 0.01))
    assert "_attachment_proxy" not in meta
    assert retired["status"] == "retired"
    assert retired["reason"] == "aperture_collapsed_to_empty_close"


def test_attachment_proxy_uses_host_bound_target_instead_of_nearest_object() -> None:
    target = {
        "name": "salad_dressing_1",
        "category": "salad_dressing",
        "position": [0.0, 0.0, 0.10],
        "dims": [0.05, 0.05, 0.14],
    }
    distractor = {
        "name": "ketchup_1",
        "category": "ketchup",
        "position": [0.0, 0.0, 0.14],
        "dims": [0.05, 0.05, 0.14],
    }
    meta = {"_collision_objects": [target, distractor]}

    receipt = _arm_attachment_proxy(
        meta,
        _observation([0.0, 0.0, 0.17], 0.42),
        authorized_object=target,
    )

    assert meta["_attachment_proxy"]["object_name"] == "salad_dressing_1"
    assert meta["_attachment_proxy"]["binding_source"] == (
        "host_compiled_target_provenance"
    )
    assert receipt["status"] == "tentative"
    assert receipt["target_object_name"] == "salad_dressing_1"
    assert receipt["attachment_proven"] is False


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


def test_attached_object_collision_allows_only_monotonic_escape_from_initial_overlap() -> None:
    attachment = {
        "status": "confirmed",
        "object_name": "bottle_1",
        "relative_xyz": [0.0, 0.0, -0.05],
        "dims": [0.06, 0.06, 0.14],
    }
    neighbour = {
        "name": "box_1",
        "category": "box",
        "position": [0.0, 0.0, 0.07],
        "dims": [0.08, 0.08, 0.10],
    }

    escape_collision, escape_info = check_attached_object_collision(
        attachment,
        [neighbour],
        [0.0, 0.0, 0.16],
        baseline_eef_xyz=[0.0, 0.0, 0.15],
    )
    assert escape_collision is False
    assert escape_info["egress_from_initial_overlap"] is True
    assert escape_info["egress_obstacles"] == ["box_1"]

    worsening_collision, worsening_info = check_attached_object_collision(
        attachment,
        [neighbour],
        [0.0, 0.0, 0.14],
        baseline_eef_xyz=[0.0, 0.0, 0.15],
    )
    assert worsening_collision is True
    assert worsening_info["new_or_worsened"] is True
    assert worsening_info["overlap_volume_m3"] > worsening_info[
        "baseline_overlap_volume_m3"
    ]


def test_contact_authorization_resolves_one_non_receptacle_target() -> None:
    authorization = {
        "schema_version": "openeta.contact_authorization.v1",
        "compiled_grasp_id": "compiled-1",
        "waypoint_role": "grasp_contact",
        "target_anchor_world_xyz": [0.01, 0.0, 0.12],
        "target_evidence_id": "sam3:result:target",
        "object_scene_epoch": 3,
    }
    objects = [
        {
            "name": "salad_dressing_1",
            "category": "salad_dressing",
            "position": [0.0, 0.0, 0.10],
            "aabb_min": [-0.03, -0.03, 0.03],
            "aabb_max": [0.03, 0.03, 0.17],
        },
        {
            "name": "basket_1",
            "category": "basket",
            "position": [0.02, 0.0, 0.10],
            "aabb_min": [-0.10, -0.10, 0.0],
            "aabb_max": [0.14, 0.10, 0.20],
        },
        {
            "name": "ketchup_1",
            "category": "ketchup",
            "position": [0.20, 0.0, 0.10],
            "aabb_min": [0.17, -0.03, 0.03],
            "aabb_max": [0.23, 0.03, 0.17],
        },
    ]

    target, receipt = resolve_contact_authorization(authorization, objects)

    assert target is not None
    assert target["name"] == "salad_dressing_1"
    assert receipt["ok"] is True
    assert receipt["target_object_name"] == "salad_dressing_1"


def test_contact_authorization_rejects_ambiguous_or_agent_like_payload() -> None:
    objects = [
        {"name": "left", "category": "box", "position": [-0.005, 0.0, 0.1], "dims": [0.04] * 3},
        {"name": "right", "category": "box", "position": [0.005, 0.0, 0.1], "dims": [0.04] * 3},
    ]
    invalid_target, invalid = resolve_contact_authorization(
        {"waypoint_role": "grasp_contact", "target_anchor_world_xyz": [0, 0, 0.1]},
        objects,
    )
    assert invalid_target is None
    assert invalid["code"] == "contact_authorization_schema_mismatch"

    ambiguous_target, ambiguous = resolve_contact_authorization(
        {
            "schema_version": "openeta.contact_authorization.v1",
            "waypoint_role": "grasp_contact",
            "target_anchor_world_xyz": [0, 0, 0.1],
        },
        objects,
    )
    assert ambiguous_target is None
    assert ambiguous["code"] == "contact_target_geometry_ambiguous"
