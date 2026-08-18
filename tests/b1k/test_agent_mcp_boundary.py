"""B1K branch gates for the stable Agent-to-Simulator-MCP boundary.

These tests intentionally avoid importing OmniGibson / Isaac Sim so they can
run in ordinary Agent CI. Real simulator readiness is covered by the GPU smoke
commands documented in ``docs/b1k-adaptation.md``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("anyio")

from agent.tools.sim_mcp import DEFAULT_SIMULATOR_MCP_TOOL_NAMES
from sim.envs.behavior.direct_env import BehaviorDirectEnv


EXPECTED_AGENT_SIMULATOR_TOOLS = (
    "create_simulator_env",
    "close_simulator_env",
    "observe",
    "ik_preview_check",
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)


def test_b1k_uses_the_shared_agent_simulator_tool_boundary() -> None:
    """B1K must adapt behind shared tools instead of exposing raw actions."""

    assert DEFAULT_SIMULATOR_MCP_TOOL_NAMES == EXPECTED_AGENT_SIMULATOR_TOOLS
    assert "step_env" not in DEFAULT_SIMULATOR_MCP_TOOL_NAMES
    # Base motion is a known B1K gap until its Agent parameter schema and MCP
    # handler receive shared contract review.
    assert "lower_body_control_policy" not in DEFAULT_SIMULATOR_MCP_TOOL_NAMES


class _FakeR1Pro:
    arm_names = ("left", "right")
    default_arm = "left"
    gripper_control_idx = {
        "left": np.array([2, 3]),
        "right": np.array([4, 5]),
    }
    joint_lower_limits = np.zeros(6, dtype=np.float32)
    joint_upper_limits = np.ones(6, dtype=np.float32)

    def get_joint_positions(self):
        return np.array([0.0, 0.0, 0.1, 0.2, 0.7, 0.9], dtype=np.float32)

    def get_joint_velocities(self):
        return np.zeros(6, dtype=np.float32)

    def get_position_orientation(self):
        return np.array([1.0, 2.0, 0.0]), np.array([0.0, 0.0, 0.0, 1.0])

    def get_eef_pose(self, arm):
        x = 0.4 if arm == "right" else 0.2
        return np.array([x, 0.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0])


def test_b1k_gripper_state_publishes_canonical_openness_and_legacy_alias() -> None:
    """Agent consumers receive continuous openness without breaking old logs."""

    direct = object.__new__(BehaviorDirectEnv)
    direct._env = SimpleNamespace(robots=[_FakeR1Pro()])

    proprio = direct._structured_proprio()

    assert proprio["metadata"]["primary_arm"] == "right"
    assert proprio["gripper_open"] == pytest.approx(0.8)
    assert proprio["gripper_state"] == {
        "open": True,
        "openness": pytest.approx(0.8),
        "open_fraction": pytest.approx(0.8),
    }
    assert proprio["ee_pose"] == pytest.approx(
        [0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    )
