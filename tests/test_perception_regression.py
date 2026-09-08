"""Diagnostic entry guard fixtures; these do not call live services."""
import pytest

from scripts import perception_regression as diagnostic


@pytest.mark.parametrize("name", ["move_to", "gripper_control", "follow_eef_trajectory", "step_env"])
def test_transport_refuses_motion_before_dispatch(monkeypatch, name):
    class Remote:
        def call_tool(self, *args, **kwargs):
            pytest.fail("forbidden operation reached remote transport")

    monkeypatch.setattr(diagnostic, "SseSimulatorMcpTransport", lambda url: Remote())
    transport = diagnostic.DiagnosticTransport()
    with pytest.raises(RuntimeError, match="refuses"):
        transport.call_tool(name, {})
    assert transport.calls == []


def test_create_timeout_does_not_allow_second_create(monkeypatch):
    calls = []

    class Remote:
        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append(name)
            raise TimeoutError("unknown create outcome")

    monkeypatch.setattr(diagnostic, "SseSimulatorMcpTransport", lambda url: Remote())
    transport = diagnostic.DiagnosticTransport()
    with pytest.raises(TimeoutError):
        transport.call_tool("create_env", {"seed": 0})
    with pytest.raises(RuntimeError, match="one create attempt"):
        transport.call_tool("create_env", {"seed": 0})
    assert calls == ["create_env"]


def test_explicit_opt_in_precedes_loading_provider(monkeypatch):
    monkeypatch.setattr("sys.argv", ["perception_regression", "spatial0"])
    monkeypatch.setattr(diagnostic, "load_planner_provider_config", lambda: pytest.fail("loaded without opt-in"))
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("seconds", [0, -1, 601])
def test_invalid_episode_budget_is_rejected_before_loading_provider(monkeypatch, seconds):
    monkeypatch.setattr("sys.argv", ["perception_regression", "spatial0", "--allow-authorized-test-data",
                                     "--episode-seconds", str(seconds)])
    monkeypatch.setattr(diagnostic, "load_planner_provider_config", lambda: pytest.fail("loaded with invalid budget"))
    with pytest.raises(SystemExit) as exc:
        diagnostic.main()
    assert exc.value.code == 2


def test_perception_allowlist_keeps_startup_and_identity_without_external_escape():
    assert {"create_simulator_env", "sam3", "select_sam3_detection", "reject_sam3_detections"} <= diagnostic.PERCEPTION_TOOLS
    assert not ({"move_to", "gripper_control", "python_exec", "web_fetch", "web_search", "retrieve_asset_reference",
                 "molmopoint", "estimate_depth_prior", "register_skill", "promote_grasp_strategy"} & diagnostic.PERCEPTION_TOOLS)


@pytest.mark.parametrize("name,arguments", [
    ("move_to", {"num_steps": True}), ("move_to", {"num_steps": 201}),
    ("move_to", {"enable_collision_check": False}),
    ("follow_eef_trajectory", {"trajectory": [{}]*6}),
    ("follow_eef_trajectory", {"trajectory": [{}], "num_steps_per_waypoint": 101}),
])
def test_task_motion_budget_rejects_before_remote_dispatch(monkeypatch, name, arguments):
    class Remote:
        def call_tool(self, *args, **kwargs):
            pytest.fail("invalid motion dispatched")
    monkeypatch.setattr(diagnostic, "SseSimulatorMcpTransport", lambda url: Remote())
    transport = diagnostic.DiagnosticTransport(allow_motion=True)
    with pytest.raises(RuntimeError):
        transport.call_tool(name, arguments)
    assert transport.calls == []


def test_task_motion_reserves_full_requested_budget_even_when_remote_outcome_unknown(monkeypatch):
    calls = []
    class Remote:
        def call_tool(self, name, arguments, *, timeout_s=None):
            calls.append(name)
            raise TimeoutError("unknown outcome")
    monkeypatch.setattr(diagnostic, "SseSimulatorMcpTransport", lambda url: Remote())
    transport = diagnostic.DiagnosticTransport(allow_motion=True)
    transport.reserved_controller_steps = 5800
    with pytest.raises(TimeoutError):
        transport.call_tool("move_to", {"num_steps": 200})
    assert transport.reserved_controller_steps == 6000
    with pytest.raises(RuntimeError, match="total controller"):
        transport.call_tool("gripper_close", {})
    assert calls == ["move_to"]


def test_task_mode_cannot_be_enabled_for_other_diagnostics(monkeypatch):
    monkeypatch.setattr("sys.argv", ["perception_regression", "long9", "--allow-authorized-test-data", "--allow-task-motion"])
    monkeypatch.setattr(diagnostic, "load_planner_provider_config", lambda: pytest.fail("loaded without valid mode"))
    with pytest.raises(SystemExit):
        diagnostic.main()
