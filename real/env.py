"""Real-robot environment implementing the OpenETA SimulatorAdapter contract.

``RealRobotEnv`` composes one or more :class:`~real.cameras.base.Camera` sensors
and a :class:`~real.robots.base.RobotArm` and presents them through
:class:`adapter.sim.SimulatorAdapter` (``reset`` / ``observe`` / ``step``). The
agent can therefore drive real hardware with the same code path it uses for the
simulator.

Safety note: ``step`` executes physical motion. Action dispatch is intentionally
minimal and explicit; extend :meth:`RealRobotEnv._apply_action` as the real
action vocabulary is defined for the deployment profile.
"""

from __future__ import annotations

from collections.abc import Sequence

from adapter.protocol import EnvAction, EnvObservation, RobotState, StepResult
from adapter.sim import SimulatorAdapter
from real.cameras.base import Camera
from real.robots.base import RobotArm


class RealRobotEnv(SimulatorAdapter):
    """Drive real cameras + arm through the simulator adapter interface."""

    def __init__(
        self,
        cameras: Sequence[Camera],
        robot: RobotArm | None = None,
        *,
        task: str = "",
        home_on_reset: bool = False,
    ) -> None:
        self._cameras = list(cameras)
        self._robot = robot
        self._task = task
        self._home_on_reset = home_on_reset
        self._started = False

    # -- lifecycle --------------------------------------------------------
    def _ensure_started(self) -> None:
        if self._started:
            return
        for cam in self._cameras:
            cam.start()
        if self._robot is not None:
            self._robot.connect()
        self._started = True

    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        self._ensure_started()
        if task is not None:
            self._task = task
        # Homing physically moves the arm; opt-in only. Observation-first
        # deployments keep this off so reset() is a pure read.
        if self._home_on_reset and self._robot is not None:
            self._robot.home()
        return self.observe()

    def observe(self) -> EnvObservation:
        self._ensure_started()
        frames = [cam.read() for cam in self._cameras]
        robot_state = self._robot.get_state() if self._robot is not None else RobotState()
        return EnvObservation(
            task=self._task,
            cameras=frames,
            robot=robot_state,
            metadata={"deployment": "real", "num_cameras": len(frames)},
        )

    def step(self, action: EnvAction) -> StepResult:
        self._ensure_started()
        self._apply_action(action)
        return StepResult(observation=self.observe(), info={"action_type": action.action_type})

    def close(self) -> None:
        for cam in self._cameras:
            try:
                cam.stop()
            except Exception:  # pragma: no cover - best effort teardown
                pass
        if self._robot is not None:
            try:
                self._robot.disconnect()
            except Exception:  # pragma: no cover
                pass
        self._started = False

    # -- action dispatch --------------------------------------------------
    def _apply_action(self, action: EnvAction) -> None:
        """Translate an EnvAction into physical motion.

        Recognised ``command`` shapes:
          * ``{"joint_positions": [...]}``     -> move_to_joint
          * ``{"pose": {"xyz", "quat_xyzw"}}`` -> move_to_pose
          * ``{"gripper": "open"|"close"}``    -> gripper control
        """
        if self._robot is None:
            raise RuntimeError("RealRobotEnv.step() requires a robot arm.")
        cmd = action.command or {}
        if "joint_positions" in cmd:
            self._robot.move_to_joint(list(cmd["joint_positions"]))
        if "pose" in cmd:
            self._robot.move_to_pose(dict(cmd["pose"]))
        gripper = cmd.get("gripper")
        if gripper == "open":
            self._robot.open_gripper()
        elif gripper == "close":
            self._robot.close_gripper()
