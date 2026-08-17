"""Coverage for the safety-geometry gaps left unpinned by the attachment tests.

Four behaviours had no test and were therefore free to regress silently:
swept carry paths, the tentative-proxy window, the populated-world path, and
stale-world clearing.
"""

from __future__ import annotations

from sim.mcp_server.collision import check_attached_object_collision
from sim.mcp_server.server import (
    _arm_attachment_proxy,
    _check_attached_object_sweep,
    _safety_obstacles,
)


def _observation(eef_xyz, openness: float):
    return {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": list(eef_xyz)},
                "gripper_state": {"openness": openness, "open": openness > 0.5},
            }
        }
    }


# ── swept carry path ─────────────────────────────────────────────────

def test_sweep_catches_obstacle_the_endpoint_check_tunnels_through() -> None:
    """A thin wall mid-segment is missed by an endpoint-only test."""
    attachment = {
        "status": "confirmed",
        "object_name": "milk_1",
        "relative_xyz": [0.0, 0.0, 0.0],
        "dims": [0.04, 0.04, 0.04],
    }
    # 2 cm thick wall at y = 0.0, spanning x/z generously.
    wall = {
        "name": "wall_1",
        "category": "wall",
        "aabb_min": [-0.5, -0.01, -0.5],
        "aabb_max": [0.5, 0.01, 0.5],
    }
    start = [0.0, -0.30, 0.0]
    end = [0.0, 0.30, 0.0]

    # Endpoint-only: both ends are clear of the wall, so nothing is detected.
    endpoint_hit, _ = check_attached_object_collision(attachment, [wall], end)
    assert endpoint_hit is False

    # Swept: the segment crosses the wall, so it must be caught.
    swept_hit, info = _check_attached_object_sweep(attachment, [wall], start, end)
    assert swept_hit is True
    assert info["obstacle"] == "wall_1"
    assert info["swept_samples"] > 1
    assert 0.0 < info["swept_hit_fraction"] <= 1.0


def test_sweep_sample_density_scales_with_smallest_held_dimension() -> None:
    """A thinner object must be sampled more densely over the same span."""
    thin = {
        "status": "confirmed",
        "object_name": "card_1",
        "relative_xyz": [0.0, 0.0, 0.0],
        "dims": [0.012, 0.08, 0.08],
    }
    thick = dict(thin, dims=[0.12, 0.12, 0.12])
    start, end = [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]

    _, thin_info = _check_attached_object_sweep(thin, [], start, end)
    _, thick_info = _check_attached_object_sweep(thick, [], start, end)
    assert thin_info["swept_samples"] > thick_info["swept_samples"]


def test_sweep_reports_clear_when_nothing_intersects() -> None:
    attachment = {
        "status": "confirmed",
        "object_name": "milk_1",
        "relative_xyz": [0.0, 0.0, 0.0],
        "dims": [0.05, 0.05, 0.05],
    }
    far = {
        "name": "box_1",
        "category": "box",
        "position": [2.0, 2.0, 2.0],
        "dims": [0.05, 0.05, 0.05],
    }
    hit, info = _check_attached_object_sweep(
        attachment, [far], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]
    )
    assert hit is False
    assert info["swept_samples"] >= 1


# ── tentative-proxy window ───────────────────────────────────────────

def test_tentative_proxy_is_armed_and_carries_usable_geometry() -> None:
    """The lift immediately after a close must have checkable geometry.

    Confirmation needs >= 1.5 cm of motion, so gating the carry check on
    ``confirmed`` left the first lift unguarded.
    """
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
    proxy = meta["_attachment_proxy"]
    assert proxy["status"] == "tentative"

    # A tentative proxy must already produce a real verdict against an
    # obstacle sitting directly in the carry path.
    blocker = {
        "name": "shelf_1",
        "category": "shelf",
        "aabb_min": [-0.2, -0.2, 0.28],
        "aabb_max": [0.2, 0.2, 0.32],
    }
    hit, info = _check_attached_object_sweep(
        proxy, [blocker], [0.0, 0.0, 0.17], [0.0, 0.0, 0.40]
    )
    assert hit is True
    assert info["obstacle"] == "shelf_1"


def test_thin_object_still_arms_a_proxy() -> None:
    """Aperture below the old 0.08 cutoff must not silently skip protection."""
    meta = {
        "_collision_objects": [
            {
                "name": "card_1",
                "category": "card",
                "position": [0.0, 0.0, 0.10],
                "dims": [0.01, 0.06, 0.09],
            }
        ]
    }
    _arm_attachment_proxy(meta, _observation([0.0, 0.0, 0.13], 0.02))
    assert meta["_attachment_proxy"]["status"] == "tentative"
    assert meta["_attachment_proxy"]["measured_aperture"] == 0.02
    assert meta["_attachment_probe"]["armed"] is True


def test_failed_arming_leaves_a_diagnostic() -> None:
    meta = {"_collision_objects": [
        {"name": "far_1", "category": "box", "position": [1.0, 1.0, 1.0], "dims": [0.05] * 3}
    ]}
    _arm_attachment_proxy(meta, _observation([0.0, 0.0, 0.17], 0.63))
    assert "_attachment_proxy" not in meta
    probe = meta["_attachment_probe"]
    assert probe["armed"] is False
    assert probe["reason"] == "no_object_within_grasp_radius"


# ── safety obstacles: populated world, target exclusion ──────────────

def test_safety_obstacles_ignore_public_exposure_flag() -> None:
    """Privileged geometry must reach the checker regardless of _expose_objects.

    Gating on the public flag left cuRobo's world empty by default, so
    max_world_penetration was structurally 0.0.
    """
    meta = {
        "_expose_objects": False,
        "_collision_objects": [
            {"name": "a", "position": [0.5, 0.0, 0.1], "dims": [0.05] * 3},
            {"name": "b", "position": [0.6, 0.0, 0.1], "dims": [0.05] * 3},
        ],
    }
    assert len(_safety_obstacles(meta)) == 2


def test_safety_obstacles_drop_only_the_approach_target() -> None:
    meta = {
        "_collision_objects": [
            {"name": "target", "position": [0.50, 0.00, 0.10], "dims": [0.05] * 3},
            {"name": "bystander", "position": [0.50, 0.30, 0.10], "dims": [0.05] * 3},
        ]
    }
    names = {
        o["name"]
        for o in _safety_obstacles(meta, approach_target_xyz=(0.50, 0.01, 0.10))
    }
    assert names == {"bystander"}


def test_safety_obstacles_keep_everything_when_target_is_far() -> None:
    meta = {
        "_collision_objects": [
            {"name": "a", "position": [0.5, 0.0, 0.1], "dims": [0.05] * 3},
        ]
    }
    obstacles = _safety_obstacles(meta, approach_target_xyz=(0.0, 0.0, 0.9))
    assert [o["name"] for o in obstacles] == ["a"]


def test_safety_obstacles_exclude_the_held_object() -> None:
    meta = {
        "_attachment_proxy": {"status": "confirmed", "object_name": "held"},
        "_collision_objects": [
            {"name": "held", "position": [0.5, 0.0, 0.1], "dims": [0.05] * 3},
            {"name": "other", "position": [0.5, 0.4, 0.1], "dims": [0.05] * 3},
        ],
    }
    assert [o["name"] for o in _safety_obstacles(meta)] == ["other"]
