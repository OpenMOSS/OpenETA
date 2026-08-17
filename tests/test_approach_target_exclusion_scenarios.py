"""The approach-target exclusion must not resurrect the false-positive it replaces.

Restoring the arm-vs-world check is only safe if reaching toward a grasp target
still succeeds.  These cases walk a realistic LIBERO-style scene through reach,
carry, and place.
"""

from __future__ import annotations

from sim.mcp_server.server import _safety_obstacles

_SCENE = [
    {"name": "milk_1", "category": "milk", "position": [0.52, -0.05, 0.92], "dims": [0.06, 0.06, 0.12]},
    {"name": "bowl_1", "category": "bowl", "position": [0.52, 0.22, 0.90], "dims": [0.14, 0.14, 0.06]},
    {"name": "cookies_1", "category": "cookies", "position": [0.30, 0.18, 0.92], "dims": [0.08, 0.05, 0.10]},
    {"name": "cabinet_1", "category": "cabinet", "position": [0.70, 0.00, 1.20], "dims": [0.40, 0.60, 0.30]},
]


def _meta(**extra):
    meta = {"_collision_objects": [dict(o) for o in _SCENE]}
    meta.update(extra)
    return meta


def test_reaching_the_grasp_target_does_not_treat_it_as_an_obstacle() -> None:
    """The original false positive: reaching milk_1 must not abort on milk_1."""
    names = {
        o["name"]
        for o in _safety_obstacles(_meta(), approach_target_xyz=(0.52, -0.05, 0.93))
    }
    assert "milk_1" not in names
    # Everything else in the scene is still guarded.
    assert names == {"bowl_1", "cookies_1", "cabinet_1"}


def test_the_cabinet_remains_an_obstacle_while_reaching() -> None:
    """The regression this fixes: the arm could previously hit the cabinet."""
    obstacles = _safety_obstacles(_meta(), approach_target_xyz=(0.52, -0.05, 0.93))
    assert any(o["name"] == "cabinet_1" for o in obstacles)


def test_carrying_toward_a_receptacle_keeps_the_receptacle_guarded() -> None:
    """The attached-object path passes no approach target on purpose.

    The receptacle corridor is the intended placement mechanism, so dropping
    bowl_1 from the obstacle set would silently disable rim rejection.
    """
    meta = _meta(_attachment_proxy={"status": "confirmed", "object_name": "milk_1"})
    names = {o["name"] for o in _safety_obstacles(meta)}
    assert "bowl_1" in names
    assert "milk_1" not in names  # held, not an obstacle


def test_placing_into_a_receptacle_excludes_only_the_receptacle() -> None:
    meta = _meta(_attachment_proxy={"status": "confirmed", "object_name": "milk_1"})
    names = {
        o["name"]
        for o in _safety_obstacles(meta, approach_target_xyz=(0.52, 0.22, 0.98))
    }
    assert names == {"cookies_1", "cabinet_1"}


def test_a_cluttered_neighbour_is_not_silently_dropped() -> None:
    """Only the single nearest object goes, never a whole neighbourhood."""
    crowded = _meta()
    crowded["_collision_objects"].append(
        {"name": "milk_2", "category": "milk", "position": [0.52, 0.01, 0.92], "dims": [0.06] * 3}
    )
    names = {
        o["name"]
        for o in _safety_obstacles(crowded, approach_target_xyz=(0.52, -0.05, 0.93))
    }
    assert "milk_1" not in names       # nearest → the intended target
    assert "milk_2" in names           # 6 cm away → still an obstacle


def test_no_approach_target_guards_the_entire_scene() -> None:
    assert len(_safety_obstacles(_meta())) == len(_SCENE)


def test_malformed_target_degrades_to_full_guarding() -> None:
    assert len(_safety_obstacles(_meta(), approach_target_xyz=(0.1, 0.2))) == len(_SCENE)
    assert len(_safety_obstacles(_meta(), approach_target_xyz=("a", "b", "c"))) == len(_SCENE)


def test_objects_with_malformed_positions_are_skipped_not_fatal() -> None:
    meta = _meta()
    meta["_collision_objects"].append({"name": "broken", "position": None})
    obstacles = _safety_obstacles(meta, approach_target_xyz=(0.52, -0.05, 0.93))
    assert any(o["name"] == "broken" for o in obstacles)
