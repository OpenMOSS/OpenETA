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


def test_libero_cartesian_scales_match_robosuite_osc_pose_contract() -> None:
    assert cartesian_scales({}, "libero") == (0.05, 0.5)


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


def test_move_to_stops_on_worker_error_and_preserves_last_pose(monkeypatch) -> None:
    meta = {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}
    start = [0.1, 0.2, 0.3]
    calls = 0

    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_observe",
        lambda *_args, **_kwargs: {
            "observation": {"robot": {"end_effector_pose": {"xyz": start}}}
        },
    )
    monkeypatch.setattr(server, "_proxy_render", lambda *_args, **_kwargs: {})

    def fail_step(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {"error": "Step failed: executing action in terminated episode"}

    monkeypatch.setattr(server, "_proxy_step", fail_step)

    result = server.move_to.__wrapped__(
        "handle", 0.2, 0.2, 0.3, num_steps=100, session_id="sid"
    )

    assert calls == 1
    assert result["ok"] is False
    assert result["code"] == "control_step_failed"
    assert result["steps_executed"] == 1
    assert result["end"]["xyz"] == start
    assert result["reached_target"] is False
    assert result["stop_reason"] == "control_step_failed"
    assert "terminated episode" in result["error"]


def test_move_to_receipt_reports_controller_residual_and_iteration_limit(
    monkeypatch,
) -> None:
    meta = {"backend": "libero", "action_dim": 7, "remote_handle": "remote"}
    start = [0.0, 0.0, 0.0]

    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_observe",
        lambda *_args, **_kwargs: {
            "observation": {"robot": {"end_effector_pose": {"xyz": start}}}
        },
    )
    monkeypatch.setattr(server, "_proxy_render", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        server,
        "_proxy_step",
        lambda *_args, **_kwargs: {
            "observation": {"robot": {"end_effector_pose": {"xyz": start}}}
        },
    )

    result = server.move_to.__wrapped__(
        "handle", 0.1, 0.0, 0.0, num_steps=2, session_id="sid"
    )

    assert result["steps_executed"] == 2
    assert result["reached_target"] is False
    assert result["position_error_m"] == pytest.approx(0.1)
    assert result["max_axis_position_error_m"] == pytest.approx(0.1)
    assert result["stop_reason"] == "iteration_limit"


def test_ik_preview_check_returns_structured_unreachable_without_moving(monkeypatch) -> None:
    meta = {"backend": "libero", "remote_handle": "remote"}
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    calls: list[dict] = []

    def fake_preview(_meta, body):
        calls.append(body)
        return {
            "status": "unreachable",
            "kinematic_status": "unreachable",
            "feasible": False,
            "reason_code": "full_pose_infeasible",
            "message": "Position and orientation cannot be satisfied together.",
            "position_only_reachable": True,
            "orientation_only_reachable": True,
            "best_candidate": {
                "joint_positions": [0.0] * 7,
                "max_axis_position_error_m": 0.011,
                "orientation_error_rad": 0.05,
            },
        }

    monkeypatch.setattr(server, "_proxy_reachability", fake_preview)
    result = server.ik_preview_check.__wrapped__(
        "handle",
        0.1,
        0.2,
        0.3,
        roll=180.0,
        pitch=0.0,
        yaw=0.0,
        session_id="sid",
    )

    assert result["success"] is False
    assert result["status"] == "unreachable"
    assert result["reason_code"] == "full_pose_infeasible"
    assert result["collision"] == {"checked": False}
    assert result["path"]["checked"] is False
    assert calls[0]["target_xyz"] == [0.1, 0.2, 0.3]
    assert calls[0]["target_euler_xyz_deg"] == [180.0, 0.0, 0.0]


def test_ik_preview_unknown_does_not_become_a_false_rejection(monkeypatch) -> None:
    meta = {"backend": "libero", "remote_handle": "remote"}
    monkeypatch.setattr(server, "_session_envs", {"sid": {"handle": meta}})
    monkeypatch.setattr(server, "_touch_session", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        server,
        "_proxy_reachability",
        lambda *_args, **_kwargs: {
            "status": "unknown",
            "kinematic_status": "unknown",
            "feasible": None,
            "reason_code": "ik_search_timeout",
            "message": "Search budget expired.",
        },
    )

    result = server.ik_preview_check.__wrapped__(
        "handle", 0.1, 0.2, 0.3, session_id="sid"
    )

    assert result["ok"] is True
    assert result["success"] is True
    assert result["status"] == "unknown"
    assert result["feasible"] is None


def test_trajectory_pose_arguments_accept_quaternion_and_validate_endpoint() -> None:
    arguments = server._trajectory_pose_arguments(
        {"frame": "world", "xyz": [0.1, 0.2, 0.3], "quat_xyzw": [0, 0, 0, 1]},
        index=0,
    )
    assert arguments == {
        "x": 0.1,
        "y": 0.2,
        "z": 0.3,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    }
    assert server._trajectory_waypoint_reached(
        {"end": {"xyz": [0.101, 0.2, 0.3]}},
        arguments,
        tolerance=0.002,
    ) is True
    assert server._trajectory_waypoint_reached(
        {"end": {"xyz": [0.104, 0.2, 0.3]}},
        arguments,
        tolerance=0.002,
    ) is False


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


def test_close_env_is_idempotent_and_releases_after_remote_error(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class Manager:
        def proxy_handle_op(self, meta, path, method="GET"):
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
    assert first["cleanup_errors"][0].startswith("remote_close:")
    assert ("release", "worker") in calls
    assert second == {"ok": True, "already_closed": True, "cleanup_errors": []}
