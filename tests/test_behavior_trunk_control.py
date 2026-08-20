"""A trunk slot left at 0.0 commands mid-range, so "hold" must be computed.

R1Pro's trunk runs on ``JointController`` in position mode with
``use_delta_commands=False``.  ``Controller._preprocess_command`` scales the
[-1,1] input onto the joint limits, so 0.0 is *not* a no-op -- it resolves to
``(lower+upper)/2``.  For ``torso_joint1`` (limits -1.1345..1.8326) that is
0.349 rad.  Every action built as ``[0.0] * dim`` therefore commanded the torso
to a mid-range pose, and ``move_to`` dragged it there on every call.

These tests need no GPU or simulator: they pin the normalisation arithmetic and
the fail-closed paths.
"""
from __future__ import annotations

from sim.mcp_server.action_codecs import trunk_hold_values, trunk_layout

# R1Pro's real torso limits, read from r1pro_curobo.urdf.
LOWER = [-1.1345, -2.7925, -1.8326, -3.0543]
UPPER = [1.8326, 2.5307, 1.5708, 3.0543]
TRUNK_NAMES = ["torso_joint%d" % i for i in (1, 2, 3, 4)]
TRUNK_SLOTS = [3, 4, 5, 6]

# The live 28-joint order, arms interleaved -- the reason lookup is by name.
LIVE_NAMES = (
    ["base_footprint_%s_joint" % a for a in ("x", "y", "z", "rx", "ry", "rz")]
    + TRUNK_NAMES
    + [n for i in range(1, 8) for n in ("left_arm_joint%d" % i, "right_arm_joint%d" % i)]
    + ["left_gripper_finger_joint1", "left_gripper_finger_joint2",
       "right_gripper_finger_joint1", "right_gripper_finger_joint2"]
)


def _meta(**over) -> dict:
    trunk = {"supported": True, "indices": list(TRUNK_SLOTS),
             "command": "position", "joint_names": list(TRUNK_NAMES),
             "limits_lower": list(LOWER), "limits_upper": list(UPPER)}
    trunk.update(over)
    return {"backend": "behavior", "action_dim": 21, "control_spec": {"trunk": trunk}}


def _positions(**angles) -> list[float]:
    """A full 28-vector with the named torso joints set."""
    values = [0.0] * len(LIVE_NAMES)
    for name, angle in angles.items():
        values[LIVE_NAMES.index(name)] = angle
    return values


def test_zero_command_is_mid_range_not_hold() -> None:
    """Guards the premise: this is why holding must be computed at all."""
    # A trunk sitting at 0.0 rad does NOT normalise to a 0.0 command.
    _, values = trunk_hold_values(_meta(), _positions(), LIVE_NAMES)
    assert values[0] != 0.0
    # Sending 0.0 would instead have meant this angle:
    mid = (LOWER[0] + UPPER[0]) / 2.0
    assert abs(mid - 0.34905) < 1e-4
    # ...so the hold command is whatever maps 0.0 rad back through the scaling.
    span = (UPPER[0] - LOWER[0]) / 2.0
    assert abs(values[0] - (0.0 - mid) / span) < 1e-9


def test_hold_round_trips_through_the_controller_scaling() -> None:
    """Normalise then re-apply the controller's transform: same angle back."""
    angles = {"torso_joint1": 0.5, "torso_joint2": -1.2,
              "torso_joint3": 0.0, "torso_joint4": 2.0}
    slots, values = trunk_hold_values(_meta(), _positions(**angles), LIVE_NAMES)
    assert slots == TRUNK_SLOTS
    for k, name in enumerate(TRUNK_NAMES):
        span = (UPPER[k] - LOWER[k]) / 2.0
        mid = (UPPER[k] + LOWER[k]) / 2.0
        assert abs((values[k] * span + mid) - angles[name]) < 1e-6


def test_joints_are_found_by_name_not_by_slice() -> None:
    """The interleave makes positional guessing pick up the wrong joints."""
    angles = {"torso_joint1": 0.5, "torso_joint2": -1.2,
              "torso_joint3": 0.3, "torso_joint4": 2.0}
    positions = _positions(**angles)
    # Poison every non-torso slot; a slice-based reader would pick these up.
    for i, n in enumerate(LIVE_NAMES):
        if n not in TRUNK_NAMES:
            positions[i] = 99.0
    _, values = trunk_hold_values(_meta(), positions, LIVE_NAMES)
    for k, name in enumerate(TRUNK_NAMES):
        span = (UPPER[k] - LOWER[k]) / 2.0
        mid = (UPPER[k] + LOWER[k]) / 2.0
        assert abs((values[k] * span + mid) - angles[name]) < 1e-6
    # And the naive slice really would have been wrong -- guards the premise.
    assert positions[3:7] != [angles[n] for n in TRUNK_NAMES]


def test_out_of_range_angle_is_clipped_to_the_command_range() -> None:
    beyond = {"torso_joint1": 99.0}
    _, values = trunk_hold_values(_meta(), _positions(**beyond), LIVE_NAMES)
    assert values[0] == 1.0


def test_missing_names_refuses_rather_than_guessing() -> None:
    _, values = trunk_hold_values(_meta(), _positions(), [])
    assert values == []


def test_undeclared_limits_refuse() -> None:
    """Without limits the angle cannot be re-normalised, so report nothing."""
    meta = _meta()
    del meta["control_spec"]["trunk"]["limits_lower"]
    slots, values = trunk_hold_values(meta, _positions(), LIVE_NAMES)
    assert (slots, values) == ([], [])


def test_absent_trunk_joint_refuses() -> None:
    names = list(LIVE_NAMES)
    names[names.index("torso_joint3")] = "something_else"
    slots, values = trunk_hold_values(_meta(), _positions(), names)
    assert (slots, values) == ([], [])


def test_zero_width_limit_is_not_inverted() -> None:
    """A degenerate limit would divide by zero; refuse instead."""
    meta = _meta(limits_lower=[0.0, 0.0, 0.0, 0.0], limits_upper=[0.0, 0.0, 0.0, 0.0])
    slots, values = trunk_hold_values(meta, _positions(), LIVE_NAMES)
    assert (slots, values) == ([], [])


def test_length_disagreement_between_names_and_slots_refuses() -> None:
    meta = _meta(joint_names=TRUNK_NAMES[:3])
    slots, values = trunk_hold_values(meta, _positions(), LIVE_NAMES)
    assert (slots, values) == ([], [])


def test_trunkless_robot_returns_none_not_an_error() -> None:
    """Fixed-base and trunk-less robots are legitimate, not failures."""
    assert trunk_layout({"control_spec": {}}) is None
    assert trunk_layout({"control_spec": {"trunk": {"supported": False}}}) is None
    assert trunk_hold_values({"control_spec": {}}, [0.0], ["a"]) == ([], [])


def test_cartesian_action_leaves_trunk_at_zero_without_the_overlay() -> None:
    """Pins the bug's mechanism in move_to's own encoder.

    ``make_cartesian_action`` writes only the arm slots, so trunk slots arrive
    at the controller as 0.0 -- mid-range.  The server overlays the hold on top;
    this test documents why that overlay is required rather than optional.
    """
    from sim.mcp_server.action_codecs import make_cartesian_action

    meta = _meta()
    meta["control_spec"]["cartesian_delta"] = {
        "supported": True, "position_indices": [7, 8, 9],
        "rotation_indices": [10, 11, 12],
    }
    act = make_cartesian_action(meta, (0.1, 0.0, 0.0), "behavior")
    assert [act[s] for s in TRUNK_SLOTS] == [0.0, 0.0, 0.0, 0.0]


def test_overlay_writes_the_held_values_into_the_action() -> None:
    from sim.mcp_server.server import _overlay_trunk_hold

    meta = _meta()
    meta["_trunk_hold"] = (TRUNK_SLOTS, [0.1, 0.2, 0.3, 0.4])
    act = [0.0] * 21
    _overlay_trunk_hold(meta, act)
    assert [act[s] for s in TRUNK_SLOTS] == [0.1, 0.2, 0.3, 0.4]
    # Non-trunk slots untouched.
    assert act[:3] == [0.0, 0.0, 0.0] and act[7:13] == [0.0] * 6


def test_overlay_is_a_noop_when_nothing_was_captured() -> None:
    """Backends without a trunk must produce byte-identical actions."""
    from sim.mcp_server.server import _overlay_trunk_hold

    act = [0.0] * 7
    _overlay_trunk_hold({"backend": "libero"}, act)
    assert act == [0.0] * 7


def test_capture_reads_names_and_positions_off_an_observation() -> None:
    from sim.mcp_server.server import _capture_trunk_hold

    meta = _meta()
    result = {"observation": {"robot": {
        "joint_positions": _positions(torso_joint1=0.5),
        "joint_names": LIVE_NAMES,
    }}}
    _capture_trunk_hold(meta, result)
    slots, values = meta["_trunk_hold"]
    assert slots == TRUNK_SLOTS
    span = (UPPER[0] - LOWER[0]) / 2.0
    mid = (UPPER[0] + LOWER[0]) / 2.0
    assert abs((values[0] * span + mid) - 0.5) < 1e-6


def test_capture_without_names_latches_nothing() -> None:
    """No latch is better than a wrong latch: the slots stay untouched."""
    from sim.mcp_server.server import _capture_trunk_hold

    meta = _meta()
    _capture_trunk_hold(meta, {"observation": {"robot": {
        "joint_positions": _positions(), "joint_names": []}}})
    assert "_trunk_hold" not in meta
    _capture_trunk_hold(meta, {})
    assert "_trunk_hold" not in meta
