"""Unit tests for the latched two-state gripper.

The gripper is a two-state latched actuator: ``gripper_close`` /
``gripper_open`` set a persistent command that is then held on the gripper
action dimension of every subsequent motion step (e.g. the ``move_to`` control
loop), so a grasped object stays clamped while the arm moves rather than the
fingers relaxing to zero force.

These tests exercise the pure action-builders directly.
"""
from __future__ import annotations

import sim.mcp_server.server as s


def test_move_to_uses_latched_gripper_on_each_step(monkeypatch):
    """The real move_to loop must use the same latch overlay as the helper."""
    meta = {
        "backend": "libero",
        "_gripper_cmd": 1.0,
        "control_spec": {
            "controller": {
                "controller_id": "robosuite.osc_pose",
                "configured_name": "OSC_POSE",
                "command_interface": "normalized_cartesian_delta_pose",
                "goal_executor": "openeta.outer_closed_loop_cartesian.v1",
                "execution_location": "mcp_server",
                "supports_position": True,
                "supports_orientation": True,
            },
            "cartesian_delta": {"supported": True},
        },
    }
    sent_actions = []
    state = {"steps": 0}

    monkeypatch.setattr(s, "_session_envs", {"test": {"h": meta}})
    monkeypatch.setattr(s, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(s, "cartesian_scales", lambda *_args, **_kwargs: (1.0, 1.0))
    monkeypatch.setattr(s, "cartesian_command_frame", lambda *_args, **_kwargs: "world")
    monkeypatch.setattr(s, "_proxy_observe", lambda *_args, **_kwargs: {"observation": {}})
    monkeypatch.setattr(s, "_proxy_render", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(s, "_extract_ee_xyz_from_result", lambda result: [
        0.0 if state["steps"] < 2 else 0.1,
        0.0,
        0.0,
    ])

    def fake_step(*_args, **kwargs):
        sent_actions.append(list(_args[1]))
        state["steps"] += 1
        return {"observation": {}}

    monkeypatch.setattr(s, "_proxy_step", fake_step)

    # Exercise the synchronous implementation underneath FastMCP's async
    # worker-thread adapter.  The adapter itself belongs to transport tests;
    # nesting anyio.to_thread.run_sync inside asyncio.run can wait forever
    # without an AnyIO task context.
    s.move_to.__wrapped__(
        handle="h",
        x=0.1,
        y=0.0,
        z=0.0,
        num_steps=6,
        tolerance=0.001,
        session_id="test",
    )

    assert sent_actions
    assert all(action[-1] == 1.0 for action in sent_actions)


def test_default_gripper_is_open():
    """Before any gripper call the latched command defaults to open (-1)."""
    meta = {"backend": "libero"}
    assert s._gripper_cmd(meta) == -1.0


def test_uncommanded_move_leaves_gripper_dim_untouched():
    """Until gripper_open/close is called, a motion step must not force the
    gripper dim — it stays at the codec's neutral 0.0."""
    meta = {"backend": "libero"}  # no _gripper_cmd key
    act = s._make_action_for_step(meta, (0.1, 0.0, 0.0), "libero")
    assert act[-1] == 0.0
    assert act[0] == 0.1


def test_move_step_holds_closed_command():
    """A closed latch is applied on the gripper dim of a motion step."""
    meta = {"backend": "libero", "_gripper_cmd": 1.0}
    act = s._make_action_for_step(meta, (0.1, 0.0, 0.0), "libero")
    assert len(act) == 7
    assert act[-1] == 1.0  # keeps clamping while moving
    assert act[0] == 0.1


def test_move_step_holds_open_command():
    meta = {"backend": "libero", "_gripper_cmd": -1.0}
    act = s._make_action_for_step(meta, (0.0, 0.05, 0.0), "libero")
    assert act[-1] == -1.0


def test_gripper_command_survives_rotation_slots():
    """Rotation deltas fill slots 3-5; gripper stays on the last slot."""
    meta = {"backend": "libero", "_gripper_cmd": 1.0}
    act = s._make_action_for_step(meta, (0.1, 0.0, 0.0), "libero",
                                  delta_rot=[0.01, 0.02, 0.03])
    assert act[3:6] == [0.01, 0.02, 0.03]
    assert act[-1] == 1.0


def test_metaworld_4d_gripper_on_last_dim():
    meta = {"backend": "metaworld", "_gripper_cmd": 1.0}
    act = s._make_action_for_step(meta, (0.1, 0.0, 0.0), "metaworld")
    assert len(act) == 4
    assert act[-1] == 1.0


def test_gripper_action_polarity():
    """The dedicated gripper action uses +1 close / -1 open on the last dim."""
    meta = {"backend": "libero"}
    close = s._make_gripper_action(meta, open=False, backend="libero")
    open_ = s._make_gripper_action(meta, open=True, backend="libero")
    assert close[-1] == 1.0
    assert open_[-1] == -1.0
    # only the gripper dim is actuated
    assert close[:-1] == [0.0] * 6


def test_gripper_tools_use_distinct_actuation_and_contact_settle_horizons(monkeypatch):
    meta = {"backend": "libero"}
    calls = []

    monkeypatch.setattr(s, "_session_envs", {"test": {"h": meta}})
    monkeypatch.setattr(s, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        s,
        "make_gripper_action",
        lambda *_args, open_gripper, **_kwargs: [-1.0 if open_gripper else 1.0],
    )

    def fake_step(_meta, action, *, num_steps):
        calls.append((list(action), num_steps))
        openness = 1.0 if action[-1] < 0 else 0.35
        return {
            "observation": {
                "robot": {"gripper_state": {"openness": openness}}
            }
        }

    monkeypatch.setattr(s, "_proxy_step", fake_step)

    opened = s.gripper_open.__wrapped__(handle="h", session_id="test")
    closed = s.gripper_close.__wrapped__(handle="h", session_id="test")

    assert calls == [
        ([-1.0], s._GRIPPER_OPEN_STEPS),
        ([1.0], s._GRIPPER_CLOSE_STEPS),
    ]
    assert opened["gripper_actuation_receipt"] == {
        "schema_version": "openeta.gripper_actuation_receipt.v1",
        "command": "open",
        "command_latched": True,
        "steps_executed": s._GRIPPER_OPEN_STEPS,
        "settling_policy": "stationary_position_actuation",
        "measured_open_fraction": 1.0,
        "interpretation": (
            "The binary command remained applied for the reported stationary "
            "physics horizon. Aperture is contact evidence, not attachment proof."
        ),
    }
    assert closed["gripper_actuation_receipt"]["command"] == "close"
    assert closed["gripper_actuation_receipt"]["steps_executed"] == 60
    assert closed["gripper_actuation_receipt"]["settling_policy"] == (
        "stationary_continuous_position_hold"
    )
    assert closed["gripper_actuation_receipt"]["measured_open_fraction"] == 0.35
    assert meta["_gripper_cmd"] == 1.0
