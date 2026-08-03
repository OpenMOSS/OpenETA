from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("anyio")

from sim.envs.behavior.direct_env import (
    BehaviorDirectEnv,
    _configure_agent_cartesian_control,
)
from sim.mcp_server import collision, server, session
from sim.mcp_server.action_codecs import (
    ControlCodecError,
    LIBERO_PANDA_CLOSED_ENDPOINT_M,
    LIBERO_PANDA_ENDPOINT_TOLERANCE_M,
    cartesian_scales,
    make_cartesian_action,
    make_gripper_action,
)


BEHAVIOR_META = {
    "action_dim": 18,
    "control_spec": {
        "schema_version": "openeta.sim_control.v1",
        "cartesian_delta": {
            "supported": True,
            "position_indices": [7, 8, 9],
            "rotation_indices": [10, 11, 12],
            "command_frame": "robot_base",
            "position_scale_m": 0.05,
            "rotation_scale_rad": 0.25,
        },
        "gripper": {
            "supported": True,
            "indices": [13],
            "open_value": 1.0,
            "close_value": -1.0,
        },
    },
}


def test_behavior_ik_config_and_runtime_layout_are_explicit() -> None:
    config = {"controller_config": {"arm_left": {}, "arm_right": {}}}
    _configure_agent_cartesian_control(config)
    assert config["controller_config"]["arm_left"]["name"] == "InverseKinematicsController"
    assert config["controller_config"]["arm_right"]["mode"] == "pose_delta_ori"
    assert config["controller_config"]["arm_right"]["command_output_limits"][1] == [
        0.05,
        0.05,
        0.05,
        0.25,
        0.25,
        0.25,
    ]

    robot = SimpleNamespace(
        arm_names=("left", "right"),
        default_arm="left",
        arm_action_idx={"left": np.arange(1, 7), "right": np.arange(7, 13)},
        gripper_action_idx={"left": np.array([6]), "right": np.array([13])},
    )
    direct = object.__new__(BehaviorDirectEnv)
    direct._env = SimpleNamespace(robots=[robot])
    spec = direct.openeta_control_spec
    assert spec["cartesian_delta"]["arm"] == "right"
    assert spec["cartesian_delta"]["position_indices"] == [7, 8, 9]
    assert spec["gripper"]["indices"] == [13]


def test_behavior_codec_writes_only_declared_arm_and_gripper_slots() -> None:
    action = make_cartesian_action(
        BEHAVIOR_META,
        [0.1, -0.2, 0.3],
        "behavior",
        delta_rot=[0.4, 0.5, -0.6],
    )
    assert action[7:13] == [0.1, -0.2, 0.3, 0.4, 0.5, -0.6]
    assert sum(abs(value) for value in action[:7] + action[13:]) == 0.0

    opened = make_gripper_action(BEHAVIOR_META, open_gripper=True, backend="behavior")
    closed = make_gripper_action(BEHAVIOR_META, open_gripper=False, backend="behavior")
    assert opened[13] == 1.0
    assert closed[13] == -1.0
    assert sum(abs(value) for value in opened) == 1.0


def test_libero_cartesian_scales_match_robosuite_osc_pose_contract() -> None:
    assert cartesian_scales({}, "libero") == (0.05, 0.5)


def test_libero_cartesian_move_explicitly_holds_gripper() -> None:
    action = make_cartesian_action(
        {"action_dim": 7},
        [0.1, -0.2, 0.3],
        "libero",
    )
    assert action[:3] == [0.1, -0.2, 0.3]
    assert action[-1] == 0.0


def test_move_to_reports_zero_step_oriented_pose_as_converged(monkeypatch) -> None:
    target = [0.1, 0.2, 0.3]
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {
                    "xyz": target,
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "joint_velocities": [0.0] * 7,
            }
        }
    }

    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        *target,
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["success"] is True
    assert result["steps_executed"] == 0
    assert result["end"]["quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert result["orientation_error_rad"] == pytest.approx(0.0)


def test_move_to_holds_an_in_tolerance_pose_until_joint_velocity_settles(monkeypatch) -> None:
    target = [0.1, 0.2, 0.3]
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": target},
                "joint_velocities": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        }
    }
    settled = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": target},
                "joint_velocities": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    actions: list[list[float]] = []

    monkeypatch.setattr(
        server,
        "_proxy_step",
        lambda _meta, action, **_kwargs: actions.append(list(action)) or settled,
    )
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: settled)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        *target,
        num_steps=6,
        velocity_tolerance=0.05,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["success"] is True
    assert result["pose_converged"] is True
    assert result["velocity_converged"] is True
    assert result["settling_converged"] is True
    assert result["settle_steps_completed"] == 5
    assert result["steps_executed"] == 6
    assert actions == [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * 6


def test_move_to_control_steps_never_return_camera_payloads(monkeypatch) -> None:
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    final = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.01, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    calls: list[dict] = []

    def step(_meta, _action, **kwargs):
        calls.append(kwargs)
        return final

    monkeypatch.setattr(server, "_proxy_step", step)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: final)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {
            "sid": {
                "handle": {
                    "backend": "libero",
                    "action_dim": 7,
                    "remote_handle": "remote",
                }
            }
        },
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    server.move_to.__wrapped__(
        "handle",
        0.1,
        0.0,
        0.0,
        num_steps=1,
        settle_steps=0,
        enable_collision_check=False,
        session_id="sid",
    )

    assert len(calls) == 1
    assert calls[0]["render"] is False
    assert calls[0]["include_cameras"] is False


@pytest.mark.parametrize(
    ("tool", "open_gripper"),
    [(server.gripper_open, True), (server.gripper_close, False)],
)
def test_gripper_commands_skip_inline_render_and_camera_payloads(
    monkeypatch, tool, open_gripper
) -> None:
    meta = {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}
    calls: list[dict] = []

    def make_action(received_meta, *, open_gripper, backend):
        assert received_meta is meta
        assert backend == "libero"
        return [1.0 if open_gripper else -1.0]

    def step(received_meta, action, **kwargs):
        calls.append({"meta": received_meta, "action": action, **kwargs})
        return {"observation": {"robot": {}}}

    monkeypatch.setattr(server, "make_gripper_action", make_action)
    monkeypatch.setattr(server, "_proxy_step", step)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})

    result = tool.__wrapped__("handle", session_id="sid")

    assert result == {
        "observation": {"robot": {}},
        "steps_executed": server._LIBERO_GRIPPER_MAX_STEPS,
    }
    assert len(calls) == server._LIBERO_GRIPPER_MAX_STEPS
    assert all(
        call == {
            "meta": meta,
            "action": [1.0 if open_gripper else -1.0],
            "num_steps": 1,
            "render": False,
            "include_cameras": False,
        }
        for call in calls
    )


def test_libero_adaptive_gripper_stops_at_endpoint(monkeypatch) -> None:
    meta = {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}
    # Reaching the nominal closed reference is not sufficient: the primitive
    # must collect stable samples before returning the command.
    apertures = iter(
        [0.010, 0.004, 0.0026, 0.0025, 0.0025, 0.0025, 0.0025]
    )
    velocities = iter([0.01] * 7)
    calls: list[dict] = []

    def step(received_meta, action, **kwargs):
        calls.append({"meta": received_meta, "action": action, **kwargs})
        aperture = next(apertures)
        velocity = next(velocities)
        return {
            "observation": {
                "robot": {
                    "gripper_state": {
                        "aperture_m": aperture,
                        "finger_qvel": [velocity, velocity],
                    }
                }
            },
            "terminated": False,
            "truncated": False,
        }

    monkeypatch.setattr(server, "_proxy_step", step)
    result = server._adaptive_libero_gripper_step(
        meta,
        [-1.0],
    )

    assert result["steps_executed"] == 7
    assert len(calls) == 7


def test_libero_near_closed_threshold_matches_nominal_reference_tolerance() -> None:
    assert (
        LIBERO_PANDA_CLOSED_ENDPOINT_M
        + LIBERO_PANDA_ENDPOINT_TOLERANCE_M
        == pytest.approx(0.0035)
    )


def test_libero_adaptive_gripper_stops_when_contact_state_is_stable(monkeypatch) -> None:
    meta = {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}
    calls: list[dict] = []

    def step(received_meta, action, **kwargs):
        calls.append({"meta": received_meta, "action": action, **kwargs})
        return {
            "observation": {
                "robot": {
                    "gripper_state": {
                        "aperture_m": 0.0084,
                        "finger_qvel": [0.0005, 0.0005],
                    }
                }
            },
            "terminated": False,
            "truncated": False,
        }

    monkeypatch.setattr(server, "_proxy_step", step)
    result = server._adaptive_libero_gripper_step(
        meta,
        [-1.0],
    )

    assert result["steps_executed"] == server._LIBERO_GRIPPER_STABLE_STEPS + 1
    assert len(calls) == result["steps_executed"]


def test_move_to_does_not_claim_success_while_joint_velocity_is_high(monkeypatch) -> None:
    target = [0.1, 0.2, 0.3]
    moving = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": target},
                "joint_velocities": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            }
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }

    monkeypatch.setattr(server, "_proxy_step", lambda _meta, _action, **_kwargs: moving)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: moving)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": moving}})

    result = server.move_to.__wrapped__(
        "handle",
        *target,
        num_steps=2,
        velocity_tolerance=0.05,
        settle_steps=1,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["success"] is False
    assert result["pose_converged"] is True
    assert result["velocity_converged"] is False
    assert result["motion_converged"] is False
    assert result["steps_executed"] == 2


def test_move_to_preserves_truncation_without_claiming_termination(monkeypatch) -> None:
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    truncated = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.01, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": True,
    }
    monkeypatch.setattr(server, "_proxy_step", lambda *_args, **_kwargs: truncated)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: truncated)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        0.2,
        0.0,
        0.0,
        num_steps=10,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["steps_executed"] == 1
    assert result["terminated"] is False
    assert result["truncated"] is True
    assert result["success"] is False


def test_move_to_reports_missing_final_eef_pose_as_controller_error(monkeypatch) -> None:
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    missing_pose = {
        "observation": {"robot": {}},
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    monkeypatch.setattr(server, "_proxy_step", lambda *_args, **_kwargs: missing_pose)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: missing_pose)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        0.2,
        0.0,
        0.0,
        num_steps=10,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["steps_executed"] == 1
    assert result["end"]["xyz"] == []
    assert result["pose_feedback_available"] is False
    assert result["position_error_xyz"] == []
    assert result["position_error_m"] is None
    assert result["success"] is False
    assert result["controller_error"] == "simulator step returned no final EEF pose"


def test_move_to_uses_euclidean_position_tolerance(monkeypatch) -> None:
    current = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            },
            "metadata": {
                "mujoco_contacts": {
                    "source": "mujoco_data.contact",
                    "points": [
                        {
                            "position_xyz_m": [0.0, 0.0, 0.1],
                            "geom_names": [
                                "gripper0_hand_collision",
                                "table_collision",
                            ],
                        }
                    ],
                }
            },
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }
    actions: list[list[float]] = []

    monkeypatch.setattr(
        server,
        "_proxy_step",
        lambda _meta, action, **_kwargs: actions.append(list(action)) or current,
    )
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: current)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": current}})

    result = server.move_to.__wrapped__(
        "handle",
        0.009,
        0.009,
        0.009,
        num_steps=1,
        tolerance=0.01,
        settle_steps=0,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["success"] is False
    assert result["pose_converged"] is False
    assert result["position_error_m"] > 0.01
    assert result["position_tolerance_metric"] == "euclidean_norm"
    assert actions and any(abs(value) > 0.0 for value in actions[0][:3])
    assert result["mujoco_contacts"] == current["observation"]["metadata"][
        "mujoco_contacts"
    ]


def test_move_to_uses_actual_final_batch_size_for_delta(monkeypatch) -> None:
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    actions: list[list[float]] = []

    def step(_meta, action, **_kwargs):
        actions.append(list(action))
        return {
            "observation": initial["observation"],
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }

    monkeypatch.setattr(server, "_proxy_step", step)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: initial)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        0.03,
        0.0,
        0.0,
        num_steps=1,
        enable_collision_check=False,
        session_id="sid",
    )

    assert len(actions) == 1
    assert actions[0][0] == pytest.approx(0.6)
    assert "tracking" not in result


def test_move_to_limits_each_cartesian_translation_setpoint_norm(monkeypatch) -> None:
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    actions: list[list[float]] = []

    monkeypatch.setattr(
        server,
        "_proxy_step",
        lambda _meta, action, **_kwargs: actions.append(list(action))
        or {
            **initial,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        },
    )
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: initial)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        0.03,
        0.04,
        0.0,
        num_steps=1,
        max_translation_step_m=0.01,
        enable_collision_check=False,
        session_id="sid",
    )

    assert len(actions) == 1
    commanded_translation_m = np.linalg.norm(actions[0][:3]) * 0.05
    assert commanded_translation_m == pytest.approx(0.01)
    assert actions[0][:3] == pytest.approx([0.12, 0.16, 0.0])
    assert result["max_translation_step_m"] == pytest.approx(0.01)


def test_move_to_tracking_guard_stops_on_lateral_drift_without_claiming_contact(
    monkeypatch,
) -> None:
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    positions = iter(([0.004, 0.0, 0.0], [0.008, 0.0, 0.0], [0.012, 0.0, 0.0]))
    actions: list[list[float]] = []

    def step(_meta, action, **_kwargs):
        actions.append(list(action))
        return {
            "observation": {
                "robot": {
                    "end_effector_pose": {"xyz": list(next(positions))},
                    "joint_velocities": [0.0] * 7,
                }
            },
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }

    monkeypatch.setattr(server, "_proxy_step", step)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: initial)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        0.0,
        0.0,
        -0.04,
        num_steps=20,
        max_translation_step_m=0.005,
        tracking_stall_steps=3,
        tracking_min_aligned_progress_m=0.001,
        tracking_cross_track_tolerance_m=0.01,
        tracking_min_error_improvement_m=0.001,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["success"] is False
    assert result["controller_error"] == "cartesian_tracking_stalled"
    assert result["steps_executed"] == 3
    assert result["suspected_contact_constraint"] is True
    assert "contact" not in result
    assert result["tracking"]["status"] == "stalled"
    assert result["tracking"]["contact_confirmed"] is False
    window = result["tracking"]["latest_window"]
    assert window["command_aligned_progress_m"] == pytest.approx(0.0)
    assert window["cross_track_drift_m"] == pytest.approx(0.012)
    assert window["insufficient_aligned_progress"] is True
    assert window["significant_cross_track_drift"] is True
    assert all(np.linalg.norm(action[:3]) * 0.05 <= 0.005 + 1e-12 for action in actions)


def test_move_to_tracking_guard_stops_when_error_does_not_improve(monkeypatch) -> None:
    stationary = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        },
        "reward": 0.0,
        "terminated": False,
        "truncated": False,
    }

    monkeypatch.setattr(server, "_proxy_step", lambda *_args, **_kwargs: stationary)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: stationary)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": stationary}})

    result = server.move_to.__wrapped__(
        "handle",
        0.0,
        0.0,
        -0.04,
        num_steps=10,
        tracking_stall_steps=2,
        tracking_min_aligned_progress_m=0.001,
        tracking_cross_track_tolerance_m=0.01,
        tracking_min_error_improvement_m=0.001,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["controller_error"] == "cartesian_tracking_stalled"
    assert result["steps_executed"] == 2
    window = result["tracking"]["latest_window"]
    assert window["cross_track_drift_m"] == pytest.approx(0.0)
    assert window["significant_cross_track_drift"] is False
    assert window["position_error_improvement_m"] == pytest.approx(0.0)
    assert window["error_not_improving"] is True


def test_move_to_tracking_guard_allows_command_aligned_progress(monkeypatch) -> None:
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {"xyz": [0.0, 0.0, 0.0]},
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    positions = iter(([0.0, 0.0, -0.003], [0.0, 0.0, -0.006], [0.0, 0.0, -0.009]))

    def step(_meta, _action, **_kwargs):
        return {
            "observation": {
                "robot": {
                    "end_effector_pose": {"xyz": list(next(positions))},
                    "joint_velocities": [0.0] * 7,
                }
            },
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }

    monkeypatch.setattr(server, "_proxy_step", step)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: initial)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        0.0,
        0.0,
        -0.02,
        num_steps=3,
        tracking_stall_steps=3,
        tracking_min_aligned_progress_m=0.001,
        tracking_cross_track_tolerance_m=0.001,
        tracking_min_error_improvement_m=0.001,
        enable_collision_check=False,
        session_id="sid",
    )

    assert "controller_error" not in result
    assert result["tracking"]["status"] == "not_stalled"
    window = result["tracking"]["latest_window"]
    assert window["command_aligned_progress_m"] == pytest.approx(0.009)
    assert window["insufficient_aligned_progress"] is False


def test_move_to_tracking_guard_ignores_translation_drift_during_rotation(
    monkeypatch,
) -> None:
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {
                    "xyz": [0.0, 0.0, 0.0],
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    positions = iter(
        ([0.0, 0.004, 0.0], [0.0, 0.008, 0.0], [0.0, 0.012, 0.0])
    )

    def step(_meta, _action, **_kwargs):
        return {
            "observation": {
                "robot": {
                    "end_effector_pose": {
                        "xyz": list(next(positions)),
                        # Keep the orientation far from the requested 90-degree
                        # yaw so this batch is unambiguously rotational.
                        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_velocities": [0.0] * 7,
                }
            },
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }

    monkeypatch.setattr(server, "_proxy_step", step)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: initial)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {
            "sid": {
                "handle": {
                    "backend": "libero",
                    "action_dim": 7,
                    "remote_handle": "remote",
                }
            }
        },
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"remote": initial}})

    result = server.move_to.__wrapped__(
        "handle",
        0.0,
        0.0,
        -0.04,
        roll=0.0,
        pitch=0.0,
        yaw=90.0,
        num_steps=3,
        tracking_stall_steps=3,
        tracking_min_aligned_progress_m=0.001,
        tracking_cross_track_tolerance_m=0.01,
        tracking_min_error_improvement_m=0.001,
        enable_collision_check=False,
        session_id="sid",
    )

    assert result["success"] is False
    assert "controller_error" not in result
    assert result["tracking"]["status"] == "not_stalled"
    assert result["tracking"]["samples_observed"] == 0
    assert result["tracking"]["latest_window"] is None


def test_move_to_orientation_hold_falls_back_to_authored_target(
    monkeypatch,
) -> None:
    current_xyz = [0.0, 0.0, 0.0]
    initial = {
        "observation": {
            "robot": {
                "end_effector_pose": {
                    "xyz": list(current_xyz),
                    "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "joint_velocities": [0.0] * 7,
            }
        }
    }
    actions: list[list[float]] = []

    def step(_meta, action, **_kwargs):
        actions.append(list(action))
        current_xyz[0] += float(action[0]) * 0.01
        return {
            "observation": {
                "robot": {
                    "end_effector_pose": {
                        "xyz": list(current_xyz),
                        # Keep the in-place orientation hold infeasible.
                        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    "joint_velocities": [0.0] * 7,
                }
            },
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
        }

    monkeypatch.setattr(server, "_proxy_step", step)
    monkeypatch.setattr(server, "_proxy_render", lambda _meta: initial)
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {
            "sid": {
                "handle": {
                    "backend": "libero",
                    "action_dim": 7,
                    "remote_handle": "remote",
                }
            }
        },
    )
    monkeypatch.setattr(
        server,
        "_session_last_obs",
        {"sid": {"remote": initial}},
    )

    result = server.move_to.__wrapped__(
        "handle",
        0.03,
        0.0,
        0.0,
        roll=0.0,
        pitch=0.0,
        yaw=90.0,
        num_steps=45,
        tolerance=0.001,
        enable_collision_check=False,
        session_id="sid",
    )

    assert all(action[0] == pytest.approx(0.0) for action in actions[:33])
    assert any(abs(action[0]) > 0.0 for action in actions[33:])
    assert result["end"]["xyz"][0] > 0.0
    assert result["orientation_hold"] == {
        "used": True,
        "completed": False,
        "fallback_to_target_pose": True,
        "step_limit": 33,
        "position_xyz": [0.0, 0.0, 0.0],
    }


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"max_translation_step_m": 0.0}, "max_translation_step_m"),
        ({"tracking_stall_steps": 0}, "tracking_stall_steps"),
        ({"tracking_stall_steps": 2.5}, "tracking_stall_steps"),
        (
            {"tracking_stall_steps": 2, "tracking_cross_track_tolerance_m": -0.001},
            "tracking thresholds",
        ),
    ],
)
def test_move_to_rejects_invalid_tracking_safety_parameters(
    monkeypatch, arguments, message
) -> None:
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"handle": {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}}},
    )

    result = server.move_to.__wrapped__(
        "handle",
        0.0,
        0.0,
        0.0,
        enable_collision_check=False,
        session_id="sid",
        **arguments,
    )

    assert result["ok"] is False
    assert result["code"] == "invalid_control_parameter"
    assert message in result["error"]


def test_unknown_and_undeclared_backends_fail_closed() -> None:
    with pytest.raises(ControlCodecError) as behavior_error:
        make_cartesian_action({}, [1, 2, 3], "behavior")
    assert behavior_error.value.code == "unsupported_cartesian_control"
    with pytest.raises(ControlCodecError) as unknown_error:
        make_cartesian_action({}, [1, 2, 3], "mystery_sim")
    assert unknown_error.value.code == "unsupported_cartesian_control"

    monkey_meta = {"backend": "behavior", "action_dim": 18, "remote_handle": "remote"}
    server._session_envs["sid"] = {"handle": monkey_meta}
    try:
        result = server.move_to.__wrapped__(
            "handle", 0.1, 0.2, 0.3, session_id="sid"
        )
    finally:
        server._session_envs.pop("sid", None)
    assert result["ok"] is False
    assert result["code"] == "unsupported_cartesian_control"


def test_ttl_cleanup_closes_releases_and_removes_every_handle(monkeypatch) -> None:
    calls: list[tuple] = []

    class Manager:
        def proxy_handle_op(self, meta, path, method="GET"):
            calls.append(("close", meta["remote_handle"], path, method))
            return {"ok": True}

        def release_worker(self, worker_url):
            calls.append(("release", worker_url))

    monkeypatch.setattr(session, "_get_mgr", lambda: Manager())
    monkeypatch.setattr(collision, "remove_checker", lambda handle: calls.append(("checker", handle)))
    monkeypatch.setattr(
        session,
        "_session_envs",
        {"sid": {"local": {"remote_handle": "remote", "worker_url": "worker"}}},
    )
    monkeypatch.setattr(session, "_session_last_obs", {"sid": {"remote": {}}})
    monkeypatch.setattr(session, "_session_last_activity", {"sid": 1.0})
    monkeypatch.setattr(session, "_session_stream_interval", {"sid": 0.1})
    monkeypatch.setattr(session, "_sse_sessions", {"sid"})

    session._cleanup_session("sid")

    assert ("release", "worker") in calls
    assert ("checker", "local") in calls
    assert "sid" not in session._session_envs
    assert "sid" not in session._session_last_obs


def test_close_env_keeps_draining_handle_retryable_after_remote_error(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class Manager:
        def proxy_handle_op(self, meta, path, method="GET"):
            calls.append(("proxy", path))
            raise RuntimeError("transport down")

        def release_worker(self, worker_url):
            calls.append(("release", worker_url))

    monkeypatch.setattr(server, "_get_mgr", lambda: Manager())
    monkeypatch.setattr(server, "_touch_session", lambda sid: None)
    monkeypatch.setattr(server, "remove_checker", lambda handle: calls.append(("checker", handle)))
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"sid": {"local": {"remote_handle": "remote", "worker_url": "worker"}}},
    )
    monkeypatch.setattr(server, "_session_last_obs", {"sid": {"local": {}}})

    first = server.close_env.__wrapped__("local", session_id="sid")
    second = server.close_env.__wrapped__("local", session_id="sid")

    assert first["ok"] is False
    assert first["cleanup_state"] == "draining"
    assert first["retryable"] is True
    assert first["cleanup_errors"][0].startswith("remote_close:")
    assert second["ok"] is False
    assert second["cleanup_state"] == "draining"
    assert second["retryable"] is True
    assert calls.count(("proxy", "/env/remote")) == 2
    assert ("release", "worker") not in calls
    assert ("checker", "local") not in calls
    assert "local" in server._session_envs["sid"]
