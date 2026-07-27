from __future__ import annotations

import math

import pytest

pytest.importorskip("anyio")
from sim.mcp_server import server
from sim.bench_worker import _sanitize_json_payload


def test_world_vector_is_rotated_into_pandaomron_base_frame() -> None:
    half_sqrt = math.sqrt(0.5)
    # Base yaw is +90 degrees: world +X is base -Y.
    result = server._world_vector_to_base(
        [1.0, 0.0, 0.0], [0.0, 0.0, half_sqrt, half_sqrt]
    )
    assert result == pytest.approx([0.0, -1.0, 0.0], abs=1e-7)


def test_robocasa_action_codecs_use_official_12d_layout() -> None:
    arm = server._make_action_for_step({}, (0.1, 0.2, 0.3), "robocasa", [0.4, 0.5, 0.6])
    assert arm == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    opened = server._make_gripper_action({}, open=True, backend="robocasa")
    assert opened[6] == -1.0
    assert opened[11] == -1.0
    assert sum(abs(value) for value in opened) == 2.0


def test_fixed_panda_action_codecs_use_7d_layout() -> None:
    meta = {"action_dim": 7, "robot": "Panda"}
    arm = server._make_action_for_step(
        meta, (0.1, 0.2, 0.3), "robocasa", [0.4, 0.5, 0.6]
    )
    assert arm == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]

    opened = server._make_gripper_action(meta, open=True, backend="robocasa")
    assert opened == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def test_base_control_encodes_torso_base_and_mode(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(server, "_touch_session", lambda _sid: None)
    monkeypatch.setattr(
        server,
        "_session_envs",
        {"session": {"handle": {"backend": "robocasa", "action_dim": 12}}},
    )

    def fake_step(meta, action, *, num_steps):
        calls.append((meta, action, num_steps))
        return {"reward": 0.0}

    monkeypatch.setattr(server, "_proxy_step", fake_step)
    result = server.base_control.__wrapped__(
        "handle",
        forward=2.0,
        lateral=-2.0,
        yaw=0.25,
        torso=0.5,
        num_steps=3,
        session_id="session",
    )

    _meta, action, steps = calls[0]
    assert action == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.25, 0.5, 1.0]
    assert steps == 3
    assert result["control"]["forward"] == 1.0


def test_worker_replaces_nonfinite_observation_values_with_diagnostics_paths() -> None:
    payload = {
        "observation": {
            "robot": {
                "joint_positions": [0.1, float("nan")],
                "joint_velocities": [float("inf"), -float("inf")],
            }
        },
        "reward": 0.0,
    }

    sanitized, paths = _sanitize_json_payload(payload)

    assert sanitized["observation"]["robot"]["joint_positions"] == [0.1, 0.0]
    assert sanitized["observation"]["robot"]["joint_velocities"] == [0.0, 0.0]
    assert paths == [
        "$.observation.robot.joint_positions[1]",
        "$.observation.robot.joint_velocities[0]",
        "$.observation.robot.joint_velocities[1]",
    ]
