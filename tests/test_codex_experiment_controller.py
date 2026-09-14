import pytest

from scripts import codex_sim_server, codex_plugin_smoke
from tools.codex_controller import require_controller


def test_default_service_overrides_ambient_osc_and_launcher_requires_mink(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENETA_LIBERO_CONTROLLER_PROFILE", "osc_pose")
    args = codex_sim_server.parser().parse_args(["--mink-dependency-path", str(tmp_path)])
    command, env = codex_sim_server.launch_spec(args)
    launch = codex_plugin_smoke.parser().parse_args([
        "--sim-url", "http://localhost:18778/sse", "--task", "fixture", "--output", str(tmp_path)])
    assert launch.controller == env["OPENETA_LIBERO_CONTROLLER_PROFILE"] == "mink_joint_velocity"
    assert env["OPENETA_LIBERO_MINK_DEPENDENCY_PATH"] == str(tmp_path)
    assert env['OPENETA_LIBERO_GRIP_STABILIZATION'] == '1'
    assert env['OPENETA_LIBERO_CARTESIAN_SEGMENT'] == '1'
    assert command[1:] == ["-m", "sim.mcp_server", "--host", "127.0.0.1", "--port", "18778"]
    require_controller({"controller_id": "mink.robosuite_joint_velocity"}, launch.controller)


def test_explicit_osc_does_not_require_or_inherit_mink_dependencies(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENETA_LIBERO_MINK_DEPENDENCY_PATH", "/stale/path")
    args = codex_sim_server.parser().parse_args([
        "--controller", "osc_pose", "--mink-dependency-path", str(tmp_path / "missing")])
    _, env = codex_sim_server.launch_spec(args)
    assert env["OPENETA_LIBERO_CONTROLLER_PROFILE"] == "osc_pose"
    assert "OPENETA_LIBERO_MINK_DEPENDENCY_PATH" not in env
    assert env['OPENETA_LIBERO_GRIP_STABILIZATION'] == '0'
    assert env['OPENETA_LIBERO_CARTESIAN_SEGMENT'] == '0'
    require_controller({"controller_id": "robosuite.osc_pose"}, "osc_pose")


def test_fixture_stabilization_can_be_disabled_for_paired_baseline(tmp_path):
    args = codex_sim_server.parser().parse_args([
        '--mink-dependency-path', str(tmp_path), '--no-fixture-grip-stabilization'])
    _, env = codex_sim_server.launch_spec(args)
    assert env['OPENETA_LIBERO_GRIP_STABILIZATION'] == '0'


def test_cartesian_tracking_can_be_disabled_without_changing_grip_policy(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENETA_LIBERO_CARTESIAN_SEGMENT', '1')
    args = codex_sim_server.parser().parse_args([
        '--mink-dependency-path', str(tmp_path), '--no-cartesian-segment'])
    _, env = codex_sim_server.launch_spec(args)
    assert env['OPENETA_LIBERO_CARTESIAN_SEGMENT'] == '0'
    assert env['OPENETA_LIBERO_GRIP_STABILIZATION'] == '1'


@pytest.mark.parametrize("capabilities", [None, {}, {"controller_id": "robosuite.osc_pose"}])
def test_missing_or_wrong_runtime_identity_cannot_be_reported_as_mink(capabilities):
    with pytest.raises(RuntimeError, match="Expected controller mink_joint_velocity"):
        require_controller(capabilities, "mink_joint_velocity")


def test_missing_or_binary_shadowing_overlay_never_falls_back_to_osc(tmp_path):
    args = codex_sim_server.parser().parse_args(["--mink-dependency-path", str(tmp_path / "missing")])
    with pytest.raises(RuntimeError, match="not a directory"):
        codex_sim_server.launch_spec(args)
    args.mink_dependency_path = tmp_path
    (tmp_path / "numpy").mkdir()
    with pytest.raises(RuntimeError, match="must not shadow"):
        codex_sim_server.launch_spec(args)


def test_local_fixture_patch_requires_explicit_mink_experiment_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('OPENETA_LIBERO_FIXTURE_CONTACT_PATCH', '1')
    args = codex_sim_server.parser().parse_args(['--mink-dependency-path', str(tmp_path)])
    assert codex_sim_server.launch_spec(args)[1]['OPENETA_LIBERO_FIXTURE_CONTACT_PATCH'] == '0'
    args.fixture_contact_patch = True
    assert codex_sim_server.launch_spec(args)[1]['OPENETA_LIBERO_FIXTURE_CONTACT_PATCH'] == '1'
    args.controller = 'osc_pose'
    assert codex_sim_server.launch_spec(args)[1]['OPENETA_LIBERO_FIXTURE_CONTACT_PATCH'] == '0'
