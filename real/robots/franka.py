"""Franka Emika Panda / FR3 arm driver (planned).

7-DoF arm intended to be driven via ``franky`` (libfranka bindings) or a
ROS 2 / franka_ros2 bridge. This is a stub that satisfies the ``RobotArm``
contract so it can be registered and referenced; the motion methods raise until
the driver is implemented.
"""

from __future__ import annotations

from adapter.protocol import JsonDict, RobotState
from real.robots.base import RobotArm, RobotConfig

_NOT_IMPLEMENTED = (
    "FrankaArm driver is not implemented yet. Planned backend: franky/libfranka "
    "(see real/README.md)."
)


class FrankaArm(RobotArm):
    """Franka 7-DoF arm. Placeholder pending a concrete backend."""

    def __init__(self, config: RobotConfig) -> None:
        if config.dof != 7:
            config.dof = 7
        super().__init__(config)

    def connect(self) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def disconnect(self) -> None:
        self._connected = False

    def get_state(self) -> RobotState:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def move_to_joint(self, joint_positions: list[float], *, blocking: bool = True) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)

    def move_to_pose(self, pose: JsonDict, *, blocking: bool = True) -> None:
        raise NotImplementedError(_NOT_IMPLEMENTED)
