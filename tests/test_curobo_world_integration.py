"""Real-cuRobo integration for the world-population and staleness fixes.

`test_collision_world_staleness.py` covers the same bookkeeping against a stub.
This module runs it on an actual GPU, because the one thing a stub cannot answer
is whether cuRobo itself accepts `update_world(WorldConfig())` as a way to clear
obstacles -- the call the staleness fix depends on.  Skips cleanly without a GPU
or a cuRobo build.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("curobo")

if not torch.cuda.is_available():
    pytest.skip("cuRobo requires CUDA", allow_module_level=True)

from curobo.geom.types import Cuboid, WorldConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

from sim.mcp_server.collision import CollisionChecker

# Franka home configuration, well clear of the obstacles used below.  cuRobo's
# franka.yml locks the fingers, so this is the 7 arm joints only.
_FRANKA_RETRACT = [0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]

_SMALL_BOX = {"name": "box_1", "position": [0.5, 0.0, 0.5], "dims": [0.2, 0.2, 0.2]}
_FAR_BOX = {"name": "far_1", "position": [8.0, 8.0, 8.0], "dims": [0.1, 0.1, 0.1]}
# Large enough to swallow the arm, so a populated world must report penetration.
_ENGULFING = {"name": "wall_1", "position": [0.0, 0.0, 0.4], "dims": [3.0, 3.0, 3.0]}


@pytest.fixture(scope="module")
def robot_world():
    config = RobotWorldConfig.load_from_config(
        "franka.yml", WorldConfig(), collision_activation_distance=0.0
    )
    return RobotWorld(config)


@pytest.fixture(scope="module")
def checker():
    instance = CollisionChecker("libero")
    if not instance._available:
        pytest.skip("CollisionChecker reports cuRobo unavailable")
    return instance


def _primitive_registered(rw) -> bool:
    return bool(rw.world_model.collision_types.get("primitive", False))


# ── cuRobo API: the assumption the stub could not verify ─────────────

def test_curobo_accepts_an_empty_world_config_as_a_clear(robot_world) -> None:
    """The staleness fix calls this; a raise here would break it at runtime."""
    robot_world.update_world(
        WorldConfig(cuboid=[Cuboid(name="b", pose=[0.5, 0.0, 0.5, 1, 0, 0, 0], dims=[0.2] * 3)])
    )
    robot_world.update_world(WorldConfig())  # must not raise


def test_primitive_flag_stays_true_after_a_clear(robot_world) -> None:
    """Why CollisionChecker tracks its own count instead of asking cuRobo.

    ``collision_types["primitive"]`` means "primitive checking is registered",
    not "obstacles exist" -- it never returns to False.  Reading it as emptiness
    reported ``world_checked: true`` for an empty world.
    """
    robot_world.update_world(
        WorldConfig(cuboid=[Cuboid(name="b", pose=[0.5, 0.0, 0.5, 1, 0, 0, 0], dims=[0.2] * 3)])
    )
    assert _primitive_registered(robot_world) is True

    robot_world.update_world(WorldConfig())
    assert _primitive_registered(robot_world) is True  # NOT False


def test_clear_then_repopulate_is_reusable(robot_world) -> None:
    """move_to cycles this every batch, so it must survive repetition."""
    for index in range(3):
        robot_world.update_world(
            WorldConfig(
                cuboid=[
                    Cuboid(
                        name=f"b{index}",
                        pose=[0.5, 0.0, 0.5, 1, 0, 0, 0],
                        dims=[0.2] * 3,
                    )
                ]
            )
        )
        robot_world.update_world(WorldConfig())


# ── CollisionChecker: the populated-world path ──────────────────────

def test_empty_object_list_reports_world_not_checked(checker) -> None:
    """The old default path: no geometry, so no world verdict may be claimed."""
    detected, info = checker.check(_FRANKA_RETRACT, [])
    assert info["available"] is True
    assert info["world_checked"] is False
    assert info["obstacle_count"] == 0
    assert detected is False


def test_populated_world_is_actually_checked(checker) -> None:
    """Before the fix this path was unreachable by default."""
    detected, info = checker.check(_FRANKA_RETRACT, [_SMALL_BOX])
    assert info["world_checked"] is True
    assert info["obstacle_count"] == 1
    assert "max_world_penetration" in info


def test_distant_obstacle_yields_no_collision(checker) -> None:
    detected, info = checker.check(_FRANKA_RETRACT, [_FAR_BOX])
    assert detected is False
    assert info["world_checked"] is True
    # Signed distance: negative means outside the obstacle.
    assert info["max_world_penetration"] <= 0.0


def test_engulfing_obstacle_is_detected(checker) -> None:
    """A real positive penetration -- the signal that was structurally 0.0."""
    detected, info = checker.check(_FRANKA_RETRACT, [_ENGULFING])
    assert detected is True
    assert info["world_checked"] is True
    assert info["max_world_penetration"] > 0.0


def test_clearing_the_world_does_not_leave_a_stale_verdict(checker) -> None:
    """The staleness bug: obstacles used to persist at their old poses."""
    detected_before, info_before = checker.check(_FRANKA_RETRACT, [_ENGULFING])
    assert detected_before is True
    assert info_before["max_world_penetration"] > 0.0

    detected_after, info_after = checker.check(_FRANKA_RETRACT, [])
    assert detected_after is False
    assert info_after["world_checked"] is False
    assert info_after["obstacle_count"] == 0


def test_resizing_in_place_changes_reported_penetration(checker) -> None:
    """Regression on real cuRobo: the change hash omitted ``dims``.

    Same name, same pose, larger box.  Penetration used to stay pinned to the
    first size seen because the world was never re-uploaded.
    """
    # Centred on the arm's largest collision sphere at _FRANKA_RETRACT.
    at_sphere = [0.0, -0.03, 0.333]
    readings = []
    for extent in (0.05, 0.20, 1.00):
        _, info = checker.check(
            _FRANKA_RETRACT,
            [{"name": "same", "position": at_sphere, "dims": [extent] * 3}],
        )
        readings.append(info["max_world_penetration"])

    assert len(set(readings)) == len(readings), f"penetration frozen: {readings}"
    assert readings == sorted(readings), f"not monotonic in size: {readings}"


def test_shrinking_the_obstacle_set_updates_the_verdict(checker) -> None:
    detected_both, _ = checker.check(_FRANKA_RETRACT, [_ENGULFING, _FAR_BOX])
    assert detected_both is True

    detected_far_only, info = checker.check(_FRANKA_RETRACT, [_FAR_BOX])
    assert detected_far_only is False
    assert info["obstacle_count"] == 1


def test_self_collision_is_evaluated_independently(checker) -> None:
    _, info = checker.check(_FRANKA_RETRACT, [])
    assert info["self_checked"] is True
    assert "max_self_penetration" in info
