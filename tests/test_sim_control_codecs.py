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
