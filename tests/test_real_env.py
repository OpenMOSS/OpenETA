"""Hardware-free tests for the real/ deployment package.

These use fake Camera/RobotArm implementations so the SimulatorAdapter wiring,
registry, and action dispatch are covered without any vendor SDK or device.
"""

from __future__ import annotations

import pytest

from adapter.protocol import CameraFrame, EnvAction, RobotState
from adapter.sim import SimulatorAdapter
from real import RealRobotEnv, available_cameras, available_robots, make_camera, make_robot
from real.cameras.base import Camera, CameraConfig
from real.robots.base import RobotArm, RobotConfig


class FakeCamera(Camera):
    def start(self) -> None:
        self._started = True

    def read(self) -> CameraFrame:
        return CameraFrame(
            frame_id=self.name,
            rgb=[[[0, 0, 0]]],
            depth=[[0.5]],
            intrinsics={"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0, "scale": 1.0},
        )

    def stop(self) -> None:
        self._started = False


class FakeArm(RobotArm):
    # Reports a fixed starting EEF pose; move_to_pose "teleports" there so tests
    # can read back the reached pose from get_state().
    START_XYZ = [0.10, -0.49, 0.28]
    START_ROTVEC = [0.24, 3.09, 0.19]

    def __init__(self, config: RobotConfig) -> None:
        super().__init__(config)
        self.moved_joint: list[float] | None = None
        self.moved_pose: dict | None = None
        self.gripper = "unknown"
        self._eef_xyz = list(self.START_XYZ)
        self._eef_rotvec = list(self.START_ROTVEC)

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_state(self) -> RobotState:
        return RobotState(
            joint_positions=[0.0] * self.config.dof,
            end_effector_pose={"xyz": list(self._eef_xyz), "rotvec": list(self._eef_rotvec)},
        )

    def move_to_joint(self, joint_positions, *, blocking=True) -> None:
        self.moved_joint = list(joint_positions)

    def move_to_pose(self, pose, *, blocking=True) -> None:
        self.moved_pose = dict(pose)
        # Simulate a perfect move so get_state() reflects the commanded pose.
        self._eef_xyz = list(pose["xyz"])
        if "rotvec" in pose:
            self._eef_rotvec = list(pose["rotvec"])

    def open_gripper(self) -> None:
        self.gripper = "open"

    def close_gripper(self) -> None:
        self.gripper = "close"


def _env():
    cam = FakeCamera(CameraConfig(name="wrist"))
    arm = FakeArm(RobotConfig(name="ur5e", dof=6))
    return RealRobotEnv(cameras=[cam], robot=arm, task="t"), arm


def test_registry_lists_known_drivers():
    assert "realsense" in available_cameras()
    assert {"webcam", "opencv", "rtsp"} <= set(available_cameras())
    assert {"ur5e", "franka"} <= set(available_robots())


def test_make_unknown_driver_raises():
    with pytest.raises(KeyError):
        make_camera("nope", CameraConfig(name="x"))
    with pytest.raises(KeyError):
        make_robot("nope", RobotConfig(name="x"))


def test_env_is_simulator_adapter():
    env, _ = _env()
    assert isinstance(env, SimulatorAdapter)


def test_reset_and_observe_shape():
    env, _ = _env()
    obs = env.reset(task="pick")
    assert obs.task == "pick"
    assert [c.frame_id for c in obs.cameras] == ["wrist"]
    assert obs.cameras[0].intrinsics["fx"] == 600.0
    assert len(obs.robot.joint_positions) == 6
    env.close()


def test_step_dispatches_motion():
    env, arm = _env()
    env.reset()
    env.step(EnvAction(action_type="tool_call", command={"joint_positions": [1, 2, 3, 4, 5, 6]}))
    assert arm.moved_joint == [1, 2, 3, 4, 5, 6]
    env.step(EnvAction(action_type="tool_call", command={"pose": {"xyz": [0.1, 0.2, 0.3]}, "gripper": "close"}))
    assert arm.moved_pose == {"xyz": [0.1, 0.2, 0.3]}
    assert arm.gripper == "close"
    env.close()


def test_webcam_config_forces_rgb_only():
    # WebcamCamera turns depth off regardless of the requested config.
    from real.cameras.webcam import WebcamCamera

    cam = WebcamCamera(CameraConfig(name="cam0", depth_enabled=True, device=0))
    assert cam.config.depth_enabled is False


def test_rgb_only_camera_flows_through_env():
    # An RGB-only camera (depth=None) is a valid observation source.
    class RGBOnlyCamera(FakeCamera):
        def read(self) -> CameraFrame:
            return CameraFrame(frame_id=self.name, rgb=[[[10, 20, 30]]], depth=None)

    cam = RGBOnlyCamera(CameraConfig(name="webcam", depth_enabled=False))
    arm = FakeArm(RobotConfig(name="ur5e", dof=6))
    env = RealRobotEnv(cameras=[cam], robot=arm, task="rgb")
    obs = env.reset()
    assert obs.cameras[0].depth is None
    assert obs.cameras[0].frame_id == "webcam"
    env.close()


def test_reset_does_not_home_by_default():
    # Observation-first: reset() must not command motion unless opted in.
    class HomeTracker(FakeArm):
        homed = False

        def home(self) -> None:
            HomeTracker.homed = True

    cam = FakeCamera(CameraConfig(name="wrist"))
    arm = HomeTracker(RobotConfig(name="ur5e", dof=6))
    env = RealRobotEnv(cameras=[cam], robot=arm)
    env.reset()
    assert HomeTracker.homed is False
    env.close()


def _mgr(lock_file: str, *, task: str = "pick"):
    from real.mcp.observation_core import RealEnvManager

    def factory():
        cam = FakeCamera(CameraConfig(name="wrist"))
        arm = FakeArm(RobotConfig(name="ur5e", dof=6))
        return RealRobotEnv(cameras=[cam], robot=arm, task=task)

    return RealEnvManager(factory, task=task, lock_file=lock_file)


def test_create_reset_observe_lifecycle(tmp_path):
    mgr = _mgr(str(tmp_path / "lock"))
    created = mgr.create_env(env_id="real")
    assert created["backend"] == "real"
    handle = created["handle"]
    assert handle and created["session_id"]

    # reset_env / observe_env return the observation at TOP level (sim shape).
    obs = mgr.reset_env(handle)
    assert obs["task"] == "pick"
    cam = obs["cameras"][0]
    assert cam["frame_id"] == "wrist"
    assert cam["rgb_base64"]  # base64 PNG present, no dashboard on real
    assert len(obs["robot"]["joint_positions"]) == 6

    rendered = mgr.render_env(handle)
    assert rendered["cameras"][0]["frame_id"] == "wrist"

    active = mgr.list_active_envs()
    assert active["count"] == 1 and active["envs"][0]["handle"] == handle
    mgr.close()


def test_second_create_in_same_manager_errors(tmp_path):
    mgr = _mgr(str(tmp_path / "lock"))
    mgr.create_env()
    second = mgr.create_env()
    assert "error" in second and "already open" in second["error"]
    mgr.close()


def test_cross_process_lock_blocks_second_manager(tmp_path):
    # Two managers sharing one lock file mimic two processes: the second
    # create_env must fail fast (non-blocking) while the first holds the lock.
    lock = str(tmp_path / "lock")
    mgr_a = _mgr(lock)
    mgr_b = _mgr(lock)
    created_a = mgr_a.create_env()
    assert "handle" in created_a
    busy = mgr_b.create_env()
    assert "error" in busy and "busy" in busy["error"].lower()
    # After A closes, B can acquire the lock and create.
    mgr_a.close_env(
        created_a["handle"],
        session_id=created_a["session_id"],
    )
    assert "handle" in mgr_b.create_env()
    mgr_b.close()


def test_close_is_idempotent_and_releases_lock(tmp_path):
    lock = str(tmp_path / "lock")
    mgr = _mgr(lock)
    created = mgr.create_env()
    first = mgr.close_env(
        created["handle"],
        session_id=created["session_id"],
    )
    assert first["ok"] is True and first["already_closed"] is False
    second = mgr.close_env()
    assert second["already_closed"] is True
    # Lock released: a fresh manager can create again.
    mgr2 = _mgr(lock)
    assert "handle" in mgr2.create_env()
    mgr2.close()


def test_close_rejects_missing_or_stale_environment_identity(tmp_path):
    lock = str(tmp_path / "lock")
    mgr = _mgr(lock)
    created = mgr.create_env(session_id="current-session")

    missing = mgr.close_env()
    wrong_handle = mgr.close_env("stale-handle", session_id="current-session")
    wrong_session = mgr.close_env(created["handle"], session_id="stale-session")

    assert missing["ok"] is False and "requires" in missing["error"]
    assert wrong_handle["ok"] is False and "handle mismatch" in wrong_handle["error"]
    assert wrong_session["ok"] is False and "session_id mismatch" in wrong_session["error"]
    assert mgr.list_active_envs()["count"] == 1

    closed = mgr.close_env(created["handle"], session_id="current-session")
    assert closed["ok"] is True
    assert mgr.list_active_envs()["count"] == 0

    replacement = _mgr(lock)
    assert "handle" in replacement.create_env()
    replacement.close()


def test_unknown_handle_is_rejected(tmp_path):
    mgr = _mgr(str(tmp_path / "lock"))
    mgr.create_env()
    result = mgr.observe_env("bogus-handle")
    assert "error" in result and "bogus-handle" in result["error"]
    mgr.close()


def test_ops_before_create_error(tmp_path):
    mgr = _mgr(str(tmp_path / "lock"))
    assert "error" in mgr.observe_env("anything")
    assert "error" in mgr.reset_env("anything")


def test_base_control_unsupported(tmp_path):
    # base_control is unsupported on a fixed-base UR5e.
    mgr = _mgr(str(tmp_path / "lock"))
    mgr.create_env()
    assert "error" in mgr.base_control(command="forward")
    mgr.close()


def test_move_to_absolute_within_cap_commands_arm(tmp_path):
    # move_to drives the arm to an absolute pose within the safety cap and
    # returns the reached pose. Orientation is locked when no rpy is given.
    mgr = _mgr(str(tmp_path / "lock"))
    handle = mgr.create_env()["handle"]
    mgr.reset_env(handle)
    env = mgr._env
    arm = env._robot
    start_x = arm.START_XYZ[0]

    moved = mgr.move_to(handle, start_x + 0.03, arm.START_XYZ[1], arm.START_XYZ[2])
    assert "error" not in moved
    assert moved["info"]["not_implemented"] is False
    assert moved["info"]["relative"] is False
    assert moved["steps_executed"] == 1
    # Arm was commanded and reached the target; orientation preserved.
    assert arm.moved_pose is not None
    assert abs(arm.moved_pose["xyz"][0] - (start_x + 0.03)) < 1e-9
    assert arm.moved_pose["rotvec"] == arm.START_ROTVEC
    assert abs(moved["end"]["xyz"][0] - (start_x + 0.03)) < 1e-9
    mgr.close()


def test_move_to_over_translation_cap_is_rejected(tmp_path):
    # A move beyond MAX_TRANSLATION_M is rejected outright — arm never commanded.
    mgr = _mgr(str(tmp_path / "lock"))
    handle = mgr.create_env()["handle"]
    mgr.reset_env(handle)
    arm = mgr._env._robot

    over = mgr.move_to(handle, arm.START_XYZ[0] + 0.80, arm.START_XYZ[1], arm.START_XYZ[2])
    assert "error" in over
    assert "exceeds safety cap" in over["error"]
    assert over["requested_distance_m"] > mgr.MAX_TRANSLATION_M
    assert arm.moved_pose is None  # no motion was commanded
    mgr.close()


def test_move_to_over_rotation_cap_is_rejected(tmp_path):
    # A rotation beyond MAX_ROTATION_RAD is rejected outright.
    mgr = _mgr(str(tmp_path / "lock"))
    handle = mgr.create_env()["handle"]
    mgr.reset_env(handle)
    arm = mgr._env._robot
    mgr.MAX_ROTATION_RAD = 0.5

    over = mgr.move_to(handle, arm.START_XYZ[0], arm.START_XYZ[1], arm.START_XYZ[2],
                       roll=90.0, pitch=0.0, yaw=0.0)
    assert "error" in over
    assert "exceeds safety cap" in over["error"]
    assert over["requested_rotation_rad"] > mgr.MAX_ROTATION_RAD
    assert arm.moved_pose is None
    mgr.close()


def test_step_env_relative_delta_commands_arm(tmp_path):
    # step_env applies a relative Cartesian delta (scaled by num_steps) from the
    # current EEF pose and returns the sim step_result shape.
    mgr = _mgr(str(tmp_path / "lock"))
    handle = mgr.create_env()["handle"]
    mgr.reset_env(handle)
    arm = mgr._env._robot
    start_x = arm.START_XYZ[0]

    stepped = mgr.step_env(handle, action=[0.01, 0.0, 0.0], num_steps=2)
    assert stepped["info"]["not_implemented"] is False
    assert stepped["observation"]["cameras"][0]["frame_id"] == "wrist"
    assert stepped["reward"] == 0.0 and stepped["terminated"] is False
    # 0.01 * 2 steps = 0.02 m along +X from the start.
    assert abs(arm.moved_pose["xyz"][0] - (start_x + 0.02)) < 1e-9
    mgr.close()


def test_step_env_over_cap_is_rejected(tmp_path):
    # A relative delta scaled past the cap is rejected; arm never commanded.
    mgr = _mgr(str(tmp_path / "lock"))
    handle = mgr.create_env()["handle"]
    arm = mgr._env._robot

    over = mgr.step_env(handle, action=[0.10, 0.0, 0.0], num_steps=10)  # 1.0 m > cap
    assert "error" in over
    assert arm.moved_pose is None
    mgr.close()


def test_gripper_tools_actuate_via_env(tmp_path):
    # gripper_open/close route through env.step -> robot.{open,close}_gripper.
    # FakeArm records the calls; the manager returns a real observation with
    # not_implemented=False (gripper motion IS wired, unlike step_env/move_to).
    lock = str(tmp_path / "lock")

    class RecordingArm(FakeArm):
        def __init__(self, config):
            super().__init__(config)
            self.gripper_calls = []

        def open_gripper(self):
            self.gripper_calls.append("open")

        def close_gripper(self):
            self.gripper_calls.append("close")

    arm = RecordingArm(RobotConfig(name="ur5e", dof=6))

    def factory():
        return RealRobotEnv(cameras=[FakeCamera(CameraConfig(name="wrist"))],
                            robot=arm, task="pick")

    from real.mcp.observation_core import RealEnvManager
    mgr = RealEnvManager(factory, task="pick", lock_file=lock)
    handle = mgr.create_env()["handle"]
    mgr.reset_env(handle)

    opened = mgr.gripper_open(handle)
    assert opened["info"]["not_implemented"] is False
    assert opened["info"]["gripper"] == "open"
    assert opened["observation"]["cameras"][0]["frame_id"] == "wrist"

    closed = mgr.gripper_close(handle)
    assert closed["info"]["gripper"] == "close"

    assert arm.gripper_calls == ["open", "close"]
    mgr.close()


def test_create_connect_failure_is_reported_and_lock_released(tmp_path):
    from real.mcp.observation_core import RealEnvManager

    def bad_factory():
        raise RuntimeError("no hardware")

    lock = str(tmp_path / "lock")
    mgr = RealEnvManager(bad_factory, lock_file=lock)
    result = mgr.create_env()
    assert "error" in result and "no hardware" in result["error"]
    # The lock must have been released on failure, so a good manager can create.
    good = _mgr(lock)
    assert "handle" in good.create_env()
    good.close()
