"""Preserve authored rotations across both MCP adapters and worker encoding."""
import math

import numpy as np
import pytest

from agent.tools.sim_mcp import SimulatorMcpToolProxy
from sim.mcp_server.server import _euler_to_quat
from tools.codex_atomic_geometry import quat_matrix


@pytest.mark.parametrize('pitch', [-90, -89.9999999, -89.999, 0, 89.999, 89.9999999, 90])
@pytest.mark.parametrize('roll,yaw', [(90, 0), (-90, 0), (33, 72), (180, -123)])
@pytest.mark.parametrize('scale', [1, -2])
def test_agent_to_preview_and_worker_preserves_rotation(pitch, roll, yaw, scale):
    q = np.array(_euler_to_quat(*np.radians([roll, pitch, yaw]))) * scale
    pose = {'xyz': [.1, .2, 1.], 'quat_xyzw': q.tolist()}
    proxy = SimulatorMcpToolProxy(transport=None)
    parameters = {'handle': 'test-environment', 'target_pose': pose}
    # These are the actual two argument adapters, followed by the actual
    # server encoder used to construct the worker quaternion.
    for method in (proxy._ik_preview_arguments, proxy._move_to_arguments):
        args = method(parameters)
        got = _euler_to_quat(*(math.radians(args[k]) for k in ('roll', 'pitch', 'yaw')))
        np.testing.assert_allclose(quat_matrix(got), quat_matrix(q), atol=1e-8, rtol=0)
        assert [args[k] for k in ('x', 'y', 'z')] == pose['xyz']


def test_recorded_goal3_horizontal_pose_keeps_minus_y_approach():
    proxy = SimulatorMcpToolProxy(transport=None)
    args = proxy._move_to_arguments({'handle': 'test-environment', 'target_pose': {
        'xyz': [0, 0, 1], 'quat_xyzw': [.5, .5, -.5, .5]}})
    got = _euler_to_quat(*(math.radians(args[k]) for k in ('roll', 'pitch', 'yaw')))
    np.testing.assert_allclose(quat_matrix(got)[:, 2], [0, -1, 0], atol=1e-12)
    np.testing.assert_allclose(quat_matrix(got)[:, 0], [0, 0, -1], atol=1e-12)
