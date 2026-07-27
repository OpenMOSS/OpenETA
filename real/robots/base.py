"""Robot-arm hardware abstraction for real-robot deployment.

A ``RobotArm`` exposes a minimal, driver-agnostic control surface plus a
:meth:`RobotArm.get_state` method that returns :class:`adapter.protocol.RobotState`
so proprioception flows through the same contract the simulator uses.

Concrete drivers (UR5e via ur_rtde, Franka via franky/libfranka, ...) live in
sibling modules and import their SDK lazily.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from adapter.protocol import JsonDict, RobotState


@dataclass(slots=True)
class RobotConfig:
    """Static configuration for a single arm.

    Attributes
    ----------
    name:
        Stable identifier, e.g. ``"ur5e_left"``.
    ip:
        Controller IP / hostname for network-controlled arms.
    dof:
        Number of actuated joints (6 for UR5e, 7 for Franka).
    home_joints:
        Optional joint configuration used by :meth:`RobotArm.home`.
    max_velocity, max_acceleration:
        Motion caps applied by drivers that accept them.
    has_gripper:
        Whether a gripper is attached and controllable.
    extra:
        Driver-specific options (payload, TCP offset, gripper model, ...).
    """

    name: str
    ip: str | None = None
    dof: int = 6
    home_joints: list[float] = field(default_factory=list)
    max_velocity: float = 0.25
    max_acceleration: float = 0.5
    has_gripper: bool = True
    extra: JsonDict = field(default_factory=dict)


class RobotArm(ABC):
    """Base class for a real robot arm.

    Poses use ``xyz`` in metres and ``quat_xyzw`` orientation, matching the
    ``end_effector_pose`` convention already used in the adapter layer
    (see ``adapter/dummy_sim.py``).
    """

    def __init__(self, config: RobotConfig) -> None:
        self.config = config
        self._connected = False

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def connect(self) -> None:
        """Establish the control connection. Idempotent."""

    @abstractmethod
    def disconnect(self) -> None:
        """Release the control connection. Idempotent."""

    @abstractmethod
    def get_state(self) -> RobotState:
        """Return current proprioception as a ``RobotState``."""

    @abstractmethod
    def move_to_joint(self, joint_positions: list[float], *, blocking: bool = True) -> None:
        """Move to an absolute joint configuration."""

    @abstractmethod
    def move_to_pose(self, pose: JsonDict, *, blocking: bool = True) -> None:
        """Move the tool to an absolute Cartesian pose ``{"xyz", "quat_xyzw"}``."""

    # -- optional gripper API; no-ops when has_gripper is False -----------
    def open_gripper(self) -> None:
        return None

    def close_gripper(self) -> None:
        return None

    def home(self, *, blocking: bool = True) -> None:
        """Move to ``config.home_joints`` if configured."""
        if self.config.home_joints:
            self.move_to_joint(self.config.home_joints, blocking=blocking)

    # -- context manager sugar --------------------------------------------
    def __enter__(self) -> "RobotArm":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.disconnect()
