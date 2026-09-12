"""Atomic semantics use physical grip_site axes, not robosuite hand-body axes."""
import numpy as np
from test_codex_atomic import atomic, body
from test_codex_motion_hook import rig
from tools.codex_atomic_geometry import quat_matrix


def test_observed_jaw_axis_and_local_delta_use_physical_site(atomic):
    host, state = atomic
    # The fixture's underlying hand-body orientation is identity. The Panda
    # finger slide axes are along body Y, and grip-site X points along -Y.
    measured = body(host.call('episode_status', {}))['robot']
    np.testing.assert_allclose(measured['jaw_world'], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(measured['approach_world'], [0, 0, 1], atol=1e-12)
    result = host.call('move_to', {'delta_m': [.01, 0, 0], 'delta_frame': 'grip_site'})
    assert not result.isError
    target = state['calls'][-1][1]['target_pose']
    np.testing.assert_allclose(target['xyz'], [0, -.01, 1], atol=1e-12)
    np.testing.assert_allclose(quat_matrix(target['quat_xyzw']), np.eye(3), atol=1e-12)


def test_authored_horizontal_vertical_jaws_reach_both_adapter_paths(atomic):
    from agent.tools.sim_mcp import SimulatorMcpToolProxy
    from sim.mcp_server.server import _euler_to_quat
    host, state = atomic
    result = host.call('move_to', {'xyz_m': [.05, -.13, 1.09],
        'approach_world': [0, -1, 0], 'jaw_world': [0, 0, -1]})
    assert not result.isError
    proxy = SimulatorMcpToolProxy(transport=None)
    for name, parameters in state['calls']:
        parameters = {**parameters, 'handle': 'fixture'}
        fn = proxy._ik_preview_arguments if name == 'ik_preview_check' else proxy._move_to_arguments
        euler = fn(parameters)
        body_rotation = quat_matrix(_euler_to_quat(*np.radians([euler[k] for k in ('roll', 'pitch', 'yaw')])))
        # Independent fixed-site transform from the Panda XML: site axes are
        # body -Y, +X, +Z, so site X=-body Y and site Z=body Z.
        np.testing.assert_allclose(-body_rotation[:, 1], [0, 0, -1], atol=1e-12)
        np.testing.assert_allclose(body_rotation[:, 2], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(body(result)['feedback']['target']['jaw_world'], [0, 0, -1], atol=1e-12)
