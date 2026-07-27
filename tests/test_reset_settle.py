"""Unit tests for post-reset physics settling.

Right after a reset, objects can spawn slightly above their resting pose (or
with residual velocity), so the first observation shows them hovering /
jittering.  ``_settle_env`` steps the env a few times with a hold action
(zero arm motion, latched gripper) to let physics come to rest before the
first observation is returned.
"""
from __future__ import annotations

import sim.mcp_server.server as s


def _make_recorder(monkeypatch, *, terminate_at=None):
    """Patch _proxy_step to record calls; return the list of recorded calls."""
    calls: list[dict] = []

    def fake_step(meta, action, num_steps=1, render=True):
        calls.append({"action": list(action), "num_steps": num_steps, "render": render})
        term = terminate_at is not None and len(calls) >= terminate_at
        return {
            "observation": {"robot": {"end_effector_pose": {"xyz": [0.1, 0.2, 0.3]}}},
            "reward": 0.0,
            "terminated": term,
            "truncated": False,
        }

    monkeypatch.setattr(s, "_proxy_step", fake_step)
    return calls


def test_settle_runs_configured_steps(monkeypatch):
    calls = _make_recorder(monkeypatch)
    meta = {"backend": "libero"}
    out = s._settle_env(meta, "libero")
    assert len(calls) == s._SETTLE_STEPS
    # returns the last step result
    assert out["observation"]["robot"]["end_effector_pose"]["xyz"] == [0.1, 0.2, 0.3]


def test_settle_hold_action_has_zero_motion(monkeypatch):
    calls = _make_recorder(monkeypatch)
    meta = {"backend": "libero"}
    s._settle_env(meta, "libero")
    for c in calls:
        act = c["action"]
        assert act[:6] == [0.0] * 6, f"hold step should not move the arm: {act}"


def test_settle_holds_latched_gripper(monkeypatch):
    calls = _make_recorder(monkeypatch)
    meta = {"backend": "libero", "_gripper_cmd": 1.0}  # latched closed
    s._settle_env(meta, "libero")
    for c in calls:
        assert c["action"][-1] == 1.0, "settle must keep clamping a closed gripper"


def test_settle_renders_only_final_step(monkeypatch):
    calls = _make_recorder(monkeypatch)
    meta = {"backend": "libero"}
    s._settle_env(meta, "libero")
    renders = [c["render"] for c in calls]
    assert renders[-1] is True, "final settle step must render a fresh frame"
    assert all(r is False for r in renders[:-1]), "intermediate steps must skip render"


def test_settle_stops_early_on_termination(monkeypatch):
    calls = _make_recorder(monkeypatch, terminate_at=2)
    meta = {"backend": "libero"}
    s._settle_env(meta, "libero")
    assert len(calls) == 2, "settle should stop once the episode terminates"


def test_settle_noop_when_disabled(monkeypatch):
    calls = _make_recorder(monkeypatch)
    monkeypatch.setattr(s, "_SETTLE_STEPS", 0)
    out = s._settle_env({"backend": "libero"}, "libero")
    assert calls == []
    assert out == {}
