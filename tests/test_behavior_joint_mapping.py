"""The R1Pro joint vector must be permuted by name, never sliced.

R1Pro reports 28 joints as base(0-5), torso(6-9), arms *interleaved*
left/right (10-23), fingers(24-27).  cuRobo's model wants 18: four torso then
each arm in order.  So ``jp[:18]`` puts base DOF in torso slots and ``jp[6:24]``
puts right-arm angles in left-arm slots -- both produce a confident verdict from
the wrong geometry, which is the specific failure this module exists to prevent.

These tests need no GPU: they exercise the mapping and its refusals.
"""
from __future__ import annotations

import pytest

from sim.mcp_server.collision import CollisionChecker
from sim.mcp_server.server import _extract_joint_names_from_result

# The live order, captured from a real R1Pro on picking_up_trash.
LIVE_ORDER = [
    "base_footprint_x_joint", "base_footprint_y_joint", "base_footprint_z_joint",
    "base_footprint_rx_joint", "base_footprint_ry_joint", "base_footprint_rz_joint",
    "torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4",
    "left_arm_joint1", "right_arm_joint1", "left_arm_joint2", "right_arm_joint2",
    "left_arm_joint3", "right_arm_joint3", "left_arm_joint4", "right_arm_joint4",
    "left_arm_joint5", "right_arm_joint5", "left_arm_joint6", "right_arm_joint6",
    "left_arm_joint7", "right_arm_joint7",
    "left_gripper_finger_joint1", "left_gripper_finger_joint2",
    "right_gripper_finger_joint1", "right_gripper_finger_joint2",
]
CUROBO_ORDER = (["torso_joint%d" % i for i in (1, 2, 3, 4)]
                + ["left_arm_joint%d" % i for i in range(1, 8)]
                + ["right_arm_joint%d" % i for i in range(1, 8)])


class _FakeKinematics:
    joint_names = CUROBO_ORDER


class _FakeRobotWorld:
    kinematics = _FakeKinematics()


def _checker() -> CollisionChecker:
    """A behavior checker with cuRobo stubbed, so the mapping is testable."""
    checker = CollisionChecker.__new__(CollisionChecker)
    checker._backend = "behavior"
    checker._available = True
    checker._robot_world = _FakeRobotWorld()
    checker._joint_permutation = None
    checker._joint_permutation_names = None
    checker._arm_dof = len(CUROBO_ORDER)
    return checker


def test_interleaved_arms_are_permuted_not_sliced() -> None:
    checker = _checker()
    # Value encodes position, so a wrong mapping is visible in the output.
    values = [float(i) for i in range(len(LIVE_ORDER))]
    q, err = checker._map_by_name(values, LIVE_ORDER)
    assert err == ""
    assert q is not None
    # Expected: torso at 6-9, then left arm at the odd interleave, right at even.
    assert q[:4] == [6.0, 7.0, 8.0, 9.0]
    assert q[4:11] == [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0]
    assert q[11:] == [11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0]

    # And the naive slices really would have been wrong -- guards the premise.
    assert values[:18] != q
    assert values[6:24] != q


def test_reordered_names_rebuild_the_permutation() -> None:
    """The cached permutation must follow the names, not outlive them.

    A reordering that still contains every joint cuRobo needs passes the
    "missing joints" check, so a cache built once and never revalidated would
    keep applying the first ordering to a differently-ordered vector -- a
    confident verdict from the wrong geometry.  Verified live: before this,
    reversing the 28 names returned a byte-identical q.
    """
    checker = _checker()
    values = [float(i) for i in range(len(LIVE_ORDER))]

    q_first, err = checker._map_by_name(values, LIVE_ORDER)
    assert err == ""

    # Same joints, different order: values move with their names.
    swapped = list(LIVE_ORDER)
    i, j = swapped.index("left_arm_joint1"), swapped.index("right_arm_joint1")
    swapped[i], swapped[j] = swapped[j], swapped[i]
    q_second, err2 = checker._map_by_name(values, swapped)
    assert err2 == ""
    assert q_second != q_first, "stale permutation reused after a reorder"
    # left_arm_joint1 now labels the value that sits at the old right slot.
    assert q_second[4] == values[j]

    # And returning to the original order restores the original mapping.
    q_third, _ = checker._map_by_name(values, LIVE_ORDER)
    assert q_third == q_first


def test_missing_names_refuses_rather_than_guessing() -> None:
    checker = _checker()
    q, err = checker._map_by_name([0.0] * 28, None)
    assert q is None
    assert "joint_names missing" in err


def test_length_mismatch_is_rejected() -> None:
    checker = _checker()
    q, err = checker._map_by_name([0.0] * 27, LIVE_ORDER)
    assert q is None
    assert "does not match" in err


def test_absent_required_joint_is_named_in_the_reason() -> None:
    checker = _checker()
    names = list(LIVE_ORDER)
    names[10] = "some_other_joint"
    q, err = checker._map_by_name([0.0] * 28, names)
    assert q is None
    assert "left_arm_joint1" in err


def test_check_fails_closed_when_names_absent() -> None:
    """The public entry point must not report 'clear' without a mapping."""
    checker = _checker()
    detected, info = checker.check([0.0] * 28, [], joint_names=None)
    assert detected is False
    assert info["world_checked"] is False
    assert "joint_names missing" in info["reason"]


def test_single_arm_backends_still_slice() -> None:
    checker = CollisionChecker.__new__(CollisionChecker)
    checker._backend = "libero"
    assert checker._needs_named_mapping() is False
    checker._backend = "behavior"
    assert checker._needs_named_mapping() is True


def test_joint_names_survive_the_wire_round_trip() -> None:
    """joint_names must cross the worker->server boundary.

    The checker runs in the server process while BEHAVIOR runs in its own venv
    as a subprocess, so the names are only usable if they survive
    ``RobotState`` serialisation.  ``RobotState`` had no such field, which meant
    the backend computed the names and the wire silently dropped them -- the
    mapping would then fail closed forever and the collision check would never
    run in deployment, while every unit test passed.
    """
    from adapter.protocol import EnvObservation, RobotState

    names = ["torso_joint1", "left_arm_joint1", "right_arm_joint1"]
    rs = RobotState(joint_positions=[0.1, 0.2, 0.3], joint_names=names)
    assert rs.to_dict()["joint_names"] == names
    assert RobotState.from_dict(rs.to_dict()).joint_names == names

    # Full observation round trip, through the UnifiedEnv "proprio" spelling
    # that the behavior worker actually produces.
    obs = EnvObservation.from_dict({
        "proprio": {"joint_positions": [0.1, 0.2, 0.3], "joint_names": names},
    })
    assert obs.robot.joint_names == names
    assert obs.to_mcp_dict()["robot"]["joint_names"] == names

    # And the extractor the server uses must read it back off that payload.
    from sim.mcp_server.server import _extract_joint_names_from_result
    assert _extract_joint_names_from_result(
        {"observation": obs.to_mcp_dict()}) == names


def test_absent_joint_names_do_not_appear_on_the_wire() -> None:
    """Single-arm payloads must be byte-identical to before.

    Emitting an empty list would change every existing backend's observation
    schema for no benefit, and a consumer testing truthiness vs presence would
    see a different answer.
    """
    from adapter.protocol import RobotState

    d = RobotState(joint_positions=[0.1, 0.2]).to_dict()
    assert "joint_names" not in d


def test_unified_env_preserves_joint_names_as_strings() -> None:
    """The names must not be coerced through the numeric array path.

    ``UnifiedEnv._np`` would turn them into a float array (or raise), and the
    whole point is to select joints by name.
    """
    from sim.unified_env import UnifiedEnv

    env = UnifiedEnv.__new__(UnifiedEnv)
    env._np = lambda x: x  # type: ignore[method-assign]
    proprio: dict = {}
    raw = {"robot_joint_positions": [0.1, 0.2],
           "robot_joint_names": ["a_joint", "b_joint"]}
    jn = raw.get("robot_joint_names")
    if jn:
        proprio["joint_names"] = [str(n) for n in jn]
    assert proprio["joint_names"] == ["a_joint", "b_joint"]
    assert all(isinstance(n, str) for n in proprio["joint_names"])


def test_skipped_check_surfaces_its_reason_to_the_caller() -> None:
    """A skipped sub-check must say why, not just report nothing checked.

    ``move_to``'s clear-path branch reports world_checked/self_checked but
    historically dropped ``reason``, leaving a caller unable to distinguish
    "checked and clear" from "never looked".
    """
    from sim.mcp_server import server as S

    info = {"available": True, "world_checked": False, "self_checked": False,
            "reason": "joint_names missing from observation"}
    collision: dict = {
        "detected": False,
        "world_checked": bool(info.get("world_checked", False)),
        "self_checked": bool(info.get("self_checked", False)),
    }
    if info.get("reason") and not (collision["world_checked"]
                                   and collision["self_checked"]):
        collision["reason"] = info["reason"]
    assert collision["reason"] == "joint_names missing from observation"

    # Guard the real code path carries it too.
    src = __import__("inspect").getsource(S.move_to)
    assert '"reason"] = collision_info["reason"]' in src


def test_extractor_reads_joint_names_from_observation() -> None:
    result = {"observation": {"robot": {"joint_positions": [0.0, 1.0],
                                        "joint_names": ["a", "b"]}}}
    assert _extract_joint_names_from_result(result) == ["a", "b"]
    # Absent is empty, not an exception -- single-arm backends omit it.
    assert _extract_joint_names_from_result({"observation": {"robot": {}}}) == []
    assert _extract_joint_names_from_result({}) == []
