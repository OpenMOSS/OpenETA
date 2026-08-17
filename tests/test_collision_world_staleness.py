"""The cuRobo world must not retain obstacles from a previous update.

Returning early on an empty object list left the prior call's geometry resident
at stale poses, so a later query tested the arm against last-frame obstacles
and reported that verdict as authoritative.  cuRobo needs CUDA, so the
RobotWorld is stubbed and only the update bookkeeping is exercised.
"""

from __future__ import annotations

import sys
import types

import pytest

from sim.mcp_server.collision import CollisionChecker


class _FakeWorld:
    def __init__(self) -> None:
        self.collision_types: dict[str, bool] = {}


class _FakeRobotWorld:
    def __init__(self) -> None:
        self.world_model = _FakeWorld()
        self.updates: list[object] = []

    def update_world(self, world_config: object) -> None:
        self.updates.append(world_config)
        # Mirror real cuRobo (0.7.7, verified 2026-08-17): once the primitive
        # collision type is registered it stays True for the checker's lifetime,
        # including after an empty update.  An earlier version of this stub set
        # the flag from the cuboid list, which is behaviour cuRobo does not have
        # and which hid a bug in the code under test.
        self.world_model.collision_types["primitive"] = True


@pytest.fixture()
def stub_curobo(monkeypatch: pytest.MonkeyPatch):
    """Install a minimal curobo.geom.types providing Cuboid/WorldConfig."""

    class Cuboid:
        def __init__(self, name: str, pose: list[float], dims: list[float]) -> None:
            self.name, self.pose, self.dims = name, pose, dims

    class WorldConfig:
        def __init__(self, cuboid: list[Cuboid] | None = None) -> None:
            self.cuboid = cuboid or []

    curobo = types.ModuleType("curobo")
    geom = types.ModuleType("curobo.geom")
    geom_types = types.ModuleType("curobo.geom.types")
    geom_types.Cuboid = Cuboid
    geom_types.WorldConfig = WorldConfig
    curobo.geom = geom
    geom.types = geom_types
    for name, module in {
        "curobo": curobo,
        "curobo.geom": geom,
        "curobo.geom.types": geom_types,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return geom_types


def _checker() -> CollisionChecker:
    checker = CollisionChecker.__new__(CollisionChecker)
    checker._robot_world = _FakeRobotWorld()
    checker._last_objects_hash = None
    checker._obstacle_count = 0
    return checker


def test_empty_object_list_clears_the_previous_world(stub_curobo) -> None:
    checker = _checker()
    populated = [{"name": "a", "position": [0.4, 0.0, 0.1], "dims": [0.05, 0.05, 0.05]}]

    assert checker._update_world_if_changed(populated) is True
    assert checker._obstacle_count == 1

    # The stale-world bug: this used to return False without clearing, leaving
    # obstacle "a" resident at its old pose.
    assert checker._update_world_if_changed([]) is True
    assert checker._robot_world.updates[-1].cuboid == []
    # The count, not cuRobo's flag, is what makes the next query safe.
    assert checker._obstacle_count == 0


def test_obstacle_count_not_curobo_flag_tracks_emptiness(stub_curobo) -> None:
    """Regression: cuRobo keeps collision_types["primitive"] True after a clear.

    Reading that flag reported ``world_checked: true`` for an empty world, i.e.
    exactly the false-confidence the verdict fields exist to prevent.
    """
    checker = _checker()
    checker._update_world_if_changed(
        [{"name": "a", "position": [0.4, 0.0, 0.1], "dims": [0.05, 0.05, 0.05]}]
    )
    checker._update_world_if_changed([])

    # cuRobo still claims primitive checking is on ...
    assert checker._robot_world.world_model.collision_types["primitive"] is True
    # ... but we know the world is empty.
    assert checker._obstacle_count == 0
    assert checker._max_world_penetration_esdf(checker._robot_world, None) == 0.0


def test_resizing_an_object_in_place_triggers_an_update(stub_curobo) -> None:
    """Regression: the change hash ignored ``dims``.

    An object at a fixed pose whose size changed -- a perception update refining
    a bounding box -- was checked against the previous geometry.  On real cuRobo
    this pinned reported penetration to the first size ever seen.
    """
    checker = _checker()
    small = [{"name": "a", "position": [0.4, 0.0, 0.1], "dims": [0.05, 0.05, 0.05]}]
    grown = [{"name": "a", "position": [0.4, 0.0, 0.1], "dims": [0.50, 0.50, 0.50]}]

    assert checker._update_world_if_changed(small) is True
    assert checker._update_world_if_changed(grown) is True
    assert checker._robot_world.updates[-1].cuboid[0].dims == [0.50, 0.50, 0.50]


def test_unchanged_objects_skip_a_redundant_update(stub_curobo) -> None:
    checker = _checker()
    objects = [{"name": "a", "position": [0.4, 0.0, 0.1], "dims": [0.05, 0.05, 0.05]}]

    assert checker._update_world_if_changed(objects) is True
    assert checker._update_world_if_changed(objects) is False
    assert len(checker._robot_world.updates) == 1


def test_moved_object_triggers_a_fresh_update(stub_curobo) -> None:
    checker = _checker()
    assert checker._update_world_if_changed(
        [{"name": "a", "position": [0.4, 0.0, 0.1], "dims": [0.05] * 3}]
    ) is True
    assert checker._update_world_if_changed(
        [{"name": "a", "position": [0.4, 0.2, 0.1], "dims": [0.05] * 3}]
    ) is True
    assert len(checker._robot_world.updates) == 2


def test_world_has_obstacles_reflects_primitive_registration(stub_curobo) -> None:
    checker = _checker()
    assert CollisionChecker._world_has_obstacles(checker._robot_world) is False
    checker._update_world_if_changed(
        [{"name": "a", "position": [0.4, 0.0, 0.1], "dims": [0.05] * 3}]
    )
    assert CollisionChecker._world_has_obstacles(checker._robot_world) is True
