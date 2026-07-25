"""Universal Robots UR5e arm driver (via ur_rtde).

Depends on ``ur_rtde`` from the optional ``real`` extra. Imports are deferred to
:meth:`UR5eArm.connect` so the module stays importable without the SDK.
"""

from __future__ import annotations

from adapter.protocol import JsonDict, RobotState
from real.robots.base import RobotArm, RobotConfig


def _rotvec_to_quat_xyzw(rotvec: list[float]) -> list[float]:
    """Convert a UR axis-angle rotation vector to a ``[x, y, z, w]`` quaternion."""
    import math

    rx, ry, rz = rotvec
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    if angle < 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    axis = (rx / angle, ry / angle, rz / angle)
    s = math.sin(angle / 2.0)
    return [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)]


class UR5eArm(RobotArm):
    """UR5e 6-DoF arm controlled over the RTDE interface.

    Proprioception uses ``rtde_receive.RTDEReceiveInterface``; motion uses
    ``rtde_control.RTDEControlInterface``. The control interface is connected
    **lazily** on the first motion command, so read-only observation works even
    when the robot rejects external control (e.g. no control-script running).
    An optional Robotiq gripper can be wired in via ``config.extra``.
    """

    def __init__(self, config: RobotConfig) -> None:
        if config.dof != 6:
            config.dof = 6
        super().__init__(config)
        self._ctrl = None
        self._recv = None
        # Optional Robotiq gripper, read over the URCap socket on the UR
        # controller. Built lazily on first state read so a missing/absent
        # gripper never blocks arm proprioception.
        self._gripper = None

    def connect(self) -> None:
        """Open the read-only receive interface. Motion control connects lazily."""
        if self._connected:
            return
        if not self.config.ip:
            raise ValueError("UR5eArm requires config.ip (controller address).")
        try:
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:  # pragma: no cover - hardware path
            raise RuntimeError(
                "ur_rtde is required for UR5eArm. Install the 'real' extra: "
                "uv sync --extra real"
            ) from exc

        self._recv = RTDEReceiveInterface(self.config.ip)
        self._connected = True

    def _ensure_control(self):
        """Connect the RTDE control interface on demand (first motion command)."""
        if self._ctrl is not None:
            return self._ctrl
        try:
            from rtde_control import RTDEControlInterface
        except ImportError as exc:  # pragma: no cover - hardware path
            raise RuntimeError(
                "ur_rtde is required for UR5eArm motion. Install the 'real' extra."
            ) from exc
        self._ctrl = RTDEControlInterface(self.config.ip)
        return self._ctrl

    def disconnect(self) -> None:
        for handle in (self._ctrl, self._recv):
            if handle is not None:
                try:
                    handle.disconnect()
                except Exception:  # pragma: no cover - best effort teardown
                    pass
        if self._gripper is not None:
            try:
                self._gripper.close()
            except Exception:  # pragma: no cover - best effort teardown
                pass
            self._gripper = None
        self._ctrl = None
        self._recv = None
        self._connected = False

    def _ensure_gripper(self):
        """Return the Robotiq client (lazily built), or None if no gripper.

        A gripper is used when ``config.has_gripper`` is set or ``extra.gripper``
        names a Robotiq model. Talks to the URCap socket (default port 63352) on
        the arm's controller IP; ``extra.gripper_port`` overrides it.
        """
        model = str(self.config.extra.get("gripper", "")).lower()
        wants_robotiq = self.config.has_gripper or model in ("robotiq", "robotiq_2f", "hand-e")
        if not wants_robotiq or not self.config.ip:
            return None
        if self._gripper is None:
            from real.robots.robotiq import RobotiqGripperReader

            port = int(self.config.extra.get("gripper_port", 63352))
            self._gripper = RobotiqGripperReader(self.config.ip, port=port)
        return self._gripper

    def _read_gripper(self) -> JsonDict:
        """Return the Robotiq gripper snapshot, or a lightweight stub. Read-only."""
        gripper = self._ensure_gripper()
        if gripper is None:
            return {"model": str(self.config.extra.get("gripper", "")).lower() or "none"}
        try:
            return gripper.read_state()
        except Exception as exc:  # pragma: no cover - hardware path
            return {"model": "robotiq", "connected": False, "error": str(exc)}

    def activate_gripper(self, *, wait: bool = True) -> JsonDict:
        """Activate the Robotiq gripper (physically re-homes fingers). MOTION."""
        gripper = self._ensure_gripper()
        if gripper is None:
            return {"ok": False, "error": "no gripper configured"}
        return gripper.activate(wait=wait)

    def open_gripper(self) -> None:
        """Fully open the gripper (position 0). MOTION. Auto-activates if needed."""
        gripper = self._ensure_gripper()
        if gripper is None:
            return None
        if gripper.read_state().get("status") != "ready":
            gripper.activate(wait=True)
        gripper.go_to(0)

    def close_gripper(self) -> None:
        """Fully close the gripper (position 255). MOTION. Auto-activates if needed."""
        gripper = self._ensure_gripper()
        if gripper is None:
            return None
        if gripper.read_state().get("status") != "ready":
            gripper.activate(wait=True)
        gripper.go_to(255)

    def get_state(self) -> RobotState:
        if not self._connected or self._recv is None:
            raise RuntimeError("UR5eArm.get_state() called before connect().")
        q = list(self._recv.getActualQ())
        qd = list(self._recv.getActualQd())
        tcp = list(self._recv.getActualTCPPose())  # [x,y,z, rx,ry,rz] axis-angle
        rotvec = tcp[3:]
        return RobotState(
            joint_positions=q,
            joint_velocities=qd,
            end_effector_pose={
                "xyz": tcp[:3],
                "rotvec": rotvec,  # UR native axis-angle
                "quat_xyzw": _rotvec_to_quat_xyzw(rotvec),
            },
            gripper_state=self._read_gripper(),
            metadata={"arm": self.name, "controller_ip": self.config.ip},
        )

    def move_to_joint(self, joint_positions: list[float], *, blocking: bool = True) -> None:
        ctrl = self._ensure_control()
        ctrl.moveJ(
            list(joint_positions),
            self.config.max_velocity,
            self.config.max_acceleration,
        )

    def move_to_pose(self, pose: JsonDict, *, blocking: bool = True) -> None:
        ctrl = self._ensure_control()
        xyz = list(pose["xyz"])
        rotvec = list(pose.get("rotvec") or pose.get("rpy") or [0.0, 0.0, 0.0])
        ctrl.moveL(
            xyz + rotvec,
            self.config.max_velocity,
            self.config.max_acceleration,
        )
