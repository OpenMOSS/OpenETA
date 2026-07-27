"""Backend for the real-robot MCP server.

Owns at most **one** :class:`~real.env.RealRobotEnv` at a time behind a
cross-process exclusive lock (``fcntl.flock``): the bench has a single physical
UR5e + cameras, so only one env may drive it. ``create_env`` acquires the lock
non-blocking and errors out if another process already holds it; ``close_env``
releases it. A crashed holder's lock is auto-released by the OS.

The wire contract mirrors the simulator MCP server
(``sim/mcp_server/server.py``) so the agent's ``sim_mcp`` client works
unchanged: ``create_env`` → ``reset_env`` → ``observe_env``/``render_env`` →
``close_env`` handle+session lifecycle, plus control tools
(``step_env``/``move_to``/``gripper_*``/``base_control``).

Return shapes match sim exactly:
  * ``create_env`` -> ``{session_id, handle, env_id, action_dim, backend, ...}``
  * ``reset_env``/``observe_env``/``render_env`` -> observation dict at TOP level
    (``task, cameras, robot, objects, metadata``); cameras carry base64 images.
  * ``step_env``/``gripper_*`` -> ``{observation, reward, terminated, truncated, info}``
  * ``close_env`` -> ``{ok, already_closed, cleanup_errors}``
  * any failure -> ``{"error": "..."}``

All control tools issue REAL motion:
  * ``gripper_open``/``gripper_close`` drive the Robotiq gripper (auto-activating
    it if needed).
  * ``move_to`` drives the UR5e end-effector to an absolute Cartesian pose
    (world metres, rpy in degrees) via RTDE ``moveL``.
  * ``step_env`` applies a *relative* Cartesian delta (``[dx,dy,dz]`` or
    ``[dx,dy,dz,droll,dpitch,dyaw]``) from the current EEF pose.

Every arm move is guarded by a SAFETY ENVELOPE computed against the current
end-effector pose: the net translation and net rotation each have a hard cap
(``MAX_TRANSLATION_M`` / ``MAX_ROTATION_RAD``). Exceeding either cap REJECTS the
move with an ``error`` — the request is never silently clamped. When no
orientation is requested the current orientation is locked exactly (the current
rotvec is reused verbatim). Motion runs at the arm's configured velocity/accel.
"""

from __future__ import annotations

import errno
import fcntl
import os
import time
import uuid
from typing import Any

from adapter.protocol import EnvObservation
from real.env import RealRobotEnv

DEFAULT_LOCK_FILE = "/tmp/openeta_real_env.lock"


class RealEnvBusyError(RuntimeError):
    """Raised when another process already holds the real-robot lock."""

    def __init__(self, holder: str) -> None:
        self.holder = holder
        super().__init__(f"real robot busy: another env holds the lock ({holder})")


class RealEnvManager:
    """Manage a single locked RealRobotEnv and serialize it to the MCP wire."""

    def __init__(self, env_factory, *, task: str = "", lock_file: str = DEFAULT_LOCK_FILE) -> None:
        # env_factory: zero-arg callable returning a RealRobotEnv. Deferred so
        # constructing the manager never touches hardware (imports stay cheap
        # and unit tests can inject a fake factory).
        self._env_factory = env_factory
        self._task = task
        self._lock_file = lock_file
        self._env: RealRobotEnv | None = None
        self._lock_fd: int | None = None
        self._session_id: str = ""
        self._handle: str = ""
        self._env_id: str = ""
        # Strict lifecycle: after create_env the arm has NOT been homed, so
        # control tools are gated until reset_env runs the home primitive.
        self._needs_reset: bool = False

    # -- locking ----------------------------------------------------------
    def _acquire_lock(self, holder_info: str) -> None:
        """Grab the cross-process exclusive lock, non-blocking.

        Raises RealEnvBusyError if another process holds it. The lock is an
        advisory flock on a file whose contents record the current holder for
        diagnostics; flock is released automatically if this process dies.
        """
        fd = os.open(self._lock_file, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                try:
                    prev = os.read(fd, 4096).decode("utf-8", "replace").strip()
                except OSError:
                    prev = ""
                os.close(fd)
                raise RealEnvBusyError(prev or "held by another process") from None
            os.close(fd)
            raise
        # We hold the lock: stamp it with our identity for other processes.
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} {holder_info} t={time.time():.0f}".encode())
        os.fsync(fd)
        self._lock_fd = fd

    def _release_lock(self) -> None:
        if self._lock_fd is not None:
            try:
                os.ftruncate(self._lock_fd, 0)
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            finally:
                try:
                    os.close(self._lock_fd)
                except OSError:
                    pass
                self._lock_fd = None

    # -- lifecycle --------------------------------------------------------
    def create_env(self, *, env_id: str = "", task: str = "", session_id: str = "",
                   **_ignored: Any) -> dict[str, Any]:
        """Acquire the lock, build the env, reset it, and return a handle.

        Extra kwargs (render_mode/seed/image_*/include_objects/robot) are
        accepted for wire-compatibility with the simulator ``create_env`` but
        have no effect on real hardware, which is described by the config file.
        """
        if self._env is not None:
            return {"error": "an env is already open in this server; close it first"}
        sid = session_id or str(uuid.uuid4())
        handle = str(uuid.uuid4())[:12]
        try:
            self._acquire_lock(f"session={sid} handle={handle} env_id={env_id or 'real'}")
        except RealEnvBusyError as exc:
            return {"error": str(exc)}
        try:
            env = self._env_factory()
            if task:
                self._task = task
            env.reset(task=self._task or None)
        except Exception as exc:  # pragma: no cover - hardware path
            self._release_lock()
            return {"error": f"create_env failed to connect: {type(exc).__name__}: {exc}"}
        self._env = env
        self._session_id = sid
        self._handle = handle
        self._env_id = env_id or "real"
        # Arm connected but NOT homed yet. Control is blocked until
        # reset_env homes the arm and clears this flag.
        self._needs_reset = True
        return {
            "session_id": sid,
            "handle": handle,
            "env_id": self._env_id,
            "action_dim": None,
            "backend": "real",
            "robot": "",
            "control_spec": {},
            "action_hint": (
                "Real UR5e bench. Call reset_env first: it homes the arm and "
                "unlocks motion. Control tools (move_to/step_env/gripper_*) issue "
                "REAL motion, capped per move at %.2f m / %.3f rad; over-cap is "
                "rejected, not clamped." % (self.MAX_TRANSLATION_M, self.MAX_ROTATION_RAD)
            ),
        }

    def close_env(self, handle: str = "", *, session_id: str = "") -> dict[str, Any]:
        """Close the matching env and release the lock. Idempotent."""
        if self._env is None:
            return {"ok": True, "already_closed": True, "cleanup_errors": []}
        if not handle:
            return {
                "ok": False,
                "error": "close_env requires the active environment handle",
            }
        if handle != self._handle:
            return {
                "ok": False,
                "error": f"close_env handle mismatch: {handle}",
            }
        if session_id and self._session_id and session_id != self._session_id:
            return {
                "ok": False,
                "error": f"close_env session_id mismatch: {session_id}",
            }
        return self._close_active_env()

    def _close_active_env(self) -> dict[str, Any]:
        """Close the active env without client ownership checks."""

        if self._env is None:
            return {"ok": True, "already_closed": True, "cleanup_errors": []}
        cleanup_errors: list[str] = []
        try:
            self._env.close()
        except Exception as exc:  # pragma: no cover - hardware path
            cleanup_errors.append(f"env_close: {type(exc).__name__}: {exc}")
        finally:
            self._env = None
            self._session_id = ""
            self._handle = ""
            self._env_id = ""
            self._needs_reset = False
            self._release_lock()
        return {"ok": not cleanup_errors, "already_closed": False, "cleanup_errors": cleanup_errors}

    def close(self) -> None:
        """Teardown for process exit / tests."""
        self._close_active_env()

    # -- handle validation ------------------------------------------------
    def _resolve(self, handle: str) -> tuple[RealRobotEnv | None, dict[str, Any] | None]:
        if self._env is None:
            return None, {"error": "no env open; call create_env first"}
        if handle and handle != self._handle:
            return None, {"error": f"Unknown: {handle}"}
        return self._env, None

    def _require_reset(self) -> dict[str, Any] | None:
        """Block control tools until reset_env has homed the arm.

        Returns an error dict when the env was created but not yet reset
        (arm not homed / at an unknown pose), else None.
        """
        if self._needs_reset:
            return {"error": "env not reset; call reset_env first "
                             "(it homes the arm and unlocks motion)"}
        return None

    # -- observation tools ------------------------------------------------
    def reset_env(self, handle: str = "", *, seed: int | None = None,
                  session_id: str = "") -> dict[str, Any]:
        """Home the arm to a known pose, then return the initial observation.

        This is the ONLY place the arm is homed: it physically moves the UR5e
        to ``config.home_joints`` via moveJ. It also clears the post-create
        ``_needs_reset`` gate, unlocking the control tools. Homing needs
        ``home_joints`` configured; otherwise ``robot.home()`` is a no-op and
        the arm stays put (still unlocks control).
        """
        env, err = self._resolve(handle)
        if err:
            return err
        try:
            robot = getattr(env, "_robot", None)
            if robot is not None:
                robot.home()  # moveJ to config.home_joints (no-op if unset)
            obs = env.reset(task=self._task or None)
        except Exception as exc:  # pragma: no cover - hardware path
            return {"error": f"reset_env failed: {type(exc).__name__}: {exc}"}
        # Arm is homed and settled: unlock the control tools.
        self._needs_reset = False
        return self._observation_to_mcp(obs)

    def observe_env(self, handle: str = "", *, session_id: str = "") -> dict[str, Any]:
        env, err = self._resolve(handle)
        if err:
            return err
        try:
            obs = env.observe()
        except Exception as exc:  # pragma: no cover - hardware path
            return {"error": f"observe_env failed: {type(exc).__name__}: {exc}"}
        return self._observation_to_mcp(obs)

    def render_env(self, handle: str = "", *, session_id: str = "") -> dict[str, Any]:
        return self.observe_env(handle, session_id=session_id)

    def list_active_envs(self, *, session_id: str = "") -> dict[str, Any]:
        envs = []
        if self._env is not None:
            envs.append({"index": 1, "handle": self._handle, "env_id": self._env_id,
                         "backend": "real"})
        return {"session_id": self._session_id, "count": len(envs), "envs": envs}

    # -- control tools (REAL MOTION) --------------------------------------
    # Every arm move is validated against the SAFETY ENVELOPE below, computed
    # relative to the CURRENT end-effector pose. Exceeding a cap REJECTS the
    # move (never clamps). See the module docstring.

    # Hard caps on a single commanded move's net displacement from the current
    # EEF pose. Tune here; both are enforced in _move_pose().
    MAX_TRANSLATION_M = 0.60       # 60 cm max net translation per move
    MAX_ROTATION_RAD = 3.141592653589793  # 180 deg max net rotation per move

    def step_env(self, handle: str = "", action: list | None = None, *,
                 num_steps: int = 1, session_id: str = "") -> dict[str, Any]:
        """Apply a *relative* Cartesian delta from the current EEF pose.

        ``action`` is ``[dx, dy, dz]`` (metres) or
        ``[dx, dy, dz, droll, dpitch, dyaw]`` (rotation in **degrees**), scaled
        by ``num_steps``. With no action, no motion is issued.
        """
        env, err = self._resolve(handle)
        if err:
            return err
        gate = self._require_reset()
        if gate:
            return gate
        if not action:
            return self._step_result(env.observe(), info={
                "not_implemented": False, "no_action": True, "num_steps": num_steps,
            })

        vals = [float(v) for v in action]
        scale = max(1, int(num_steps))
        dx, dy, dz = (vals[0:3] + [0.0, 0.0, 0.0])[:3]
        droll = vals[3] * scale if len(vals) > 3 else None
        dpitch = vals[4] * scale if len(vals) > 4 else None
        dyaw = vals[5] * scale if len(vals) > 5 else None
        result = self._move_pose(
            env, dx * scale, dy * scale, dz * scale,
            roll=droll, pitch=dpitch, yaw=dyaw, relative=True,
        )
        if "error" in result:
            return result
        # Reshape move result into the sim step_result wire shape.
        obs = env.observe()
        info = dict(result.get("info", {}))
        info.update({"requested_action": action, "num_steps": num_steps,
                     "target": result.get("target"), "start": result.get("start"),
                     "end": result.get("end")})
        return self._step_result(obs, info=info)

    def move_to(self, handle: str = "", x: float = 0.0, y: float = 0.0, z: float = 0.0, *,
                roll: float | None = None, pitch: float | None = None, yaw: float | None = None,
                num_steps: int = 100, tolerance: float = 0.002, ori_tolerance: float = 0.05,
                session_id: str = "", enable_collision_check: bool = True) -> dict[str, Any]:
        """Move to an absolute pose after enforcing the per-command safety caps.

        The simulator-compatible planning, tolerance, and collision-check
        arguments are currently accepted but not implemented by the real
        backend.
        """
        env, err = self._resolve(handle)
        if err:
            return err
        gate = self._require_reset()
        if gate:
            return gate
        return self._move_pose(env, x, y, z, roll=roll, pitch=pitch, yaw=yaw,
                               relative=False)

    def _move_pose(self, env, x: float, y: float, z: float, *,
                   roll: float | None, pitch: float | None, yaw: float | None,
                   relative: bool) -> dict[str, Any]:
        """Move the EEF, enforcing the safety envelope against the current pose.

        ``relative`` adds ``x/y/z`` (and rpy) to the current pose; otherwise the
        args are absolute world targets (rpy in degrees). Orientation is locked
        to the current pose when no rpy is supplied. Returns the sim ``move_to``
        wire shape ``{target, start, end, steps_executed, terminated, reward,
        info}``, or ``{error}`` if a cap is exceeded or the arm rejects motion.
        """
        import math

        from adapter.protocol import EnvAction
        from real.robots import pose_math as pm

        robot = getattr(env, "_robot", None)
        if robot is None:
            return {"error": "move requires a robot arm"}

        # Reference: the CURRENT end-effector pose.
        eef = (robot.get_state().end_effector_pose) or {}
        cur_xyz = list(eef.get("xyz") or [0.0, 0.0, 0.0])
        cur_rotvec = list(eef.get("rotvec") or [0.0, 0.0, 0.0])
        cur_quat = list(eef.get("quat_xyzw") or pm.rotvec_to_quat(cur_rotvec))

        # -- target translation + net-distance cap -----------------------
        tgt_xyz = [cur_xyz[i] + (x, y, z)[i] for i in range(3)] if relative else [x, y, z]
        dist = math.dist(cur_xyz, tgt_xyz)
        if dist > self.MAX_TRANSLATION_M:
            return {"error": (f"translation {dist:.4f} m exceeds safety cap "
                              f"{self.MAX_TRANSLATION_M} m; request rejected"),
                    "requested_distance_m": round(dist, 4),
                    "max_translation_m": self.MAX_TRANSLATION_M}

        # -- target orientation + net-rotation cap ------------------------
        has_rot = any(v is not None for v in (roll, pitch, yaw))
        if has_rot:
            drpy = [math.radians(v or 0.0) for v in (roll, pitch, yaw)]
            rot_q = pm.rpy_to_quat(*drpy)
            # relative: pre-compose the delta onto the current orientation.
            tgt_quat = pm.quat_mul(rot_q, cur_quat) if relative else rot_q
            tgt_rotvec = pm.quat_to_rotvec(tgt_quat)
            ang = pm.quat_angle_between(cur_quat, tgt_quat)
            if ang > self.MAX_ROTATION_RAD:
                return {"error": (f"rotation {ang:.4f} rad exceeds safety cap "
                                  f"{self.MAX_ROTATION_RAD} rad; request rejected"),
                        "requested_rotation_rad": round(ang, 4),
                        "max_rotation_rad": self.MAX_ROTATION_RAD}
        else:
            # Lock orientation: reuse the current rotvec verbatim (no re-encode).
            tgt_quat, tgt_rotvec, ang = cur_quat, cur_rotvec, 0.0

        pose = {"xyz": tgt_xyz, "rotvec": tgt_rotvec, "quat_xyzw": tgt_quat}
        start = {"xyz": cur_xyz, "quat_xyzw": cur_quat}
        target: dict[str, Any] = {"x": tgt_xyz[0], "y": tgt_xyz[1], "z": tgt_xyz[2]}
        if has_rot:
            target["roll"], target["pitch"], target["yaw"] = roll, pitch, yaw

        try:
            env.step(EnvAction(action_type="tool_call", command={"pose": pose}))
        except Exception as exc:  # pragma: no cover - hardware path
            return {"error": f"move failed: {type(exc).__name__}: {exc}",
                    "target": target, "start": start}

        end_eef = (robot.get_state().end_effector_pose) or {}
        end = {"xyz": end_eef.get("xyz"), "quat_xyzw": end_eef.get("quat_xyzw")}
        return {
            "target": target, "start": start, "end": end,
            "requested_distance_m": round(dist, 4),
            "requested_rotation_rad": round(ang, 4),
            "steps_executed": 1, "terminated": False, "reward": 0.0,
            "info": {"relative": relative, "not_implemented": False},
        }

    def gripper_open(self, handle: str = "", *, session_id: str = "") -> dict[str, Any]:
        return self._gripper_move(handle, "open")

    def gripper_close(self, handle: str = "", *, session_id: str = "") -> dict[str, Any]:
        return self._gripper_move(handle, "close")

    def _gripper_move(self, handle: str, direction: str) -> dict[str, Any]:
        """Actuate the real gripper via env.step, then return a fresh observation.

        This issues PHYSICAL motion on the Robotiq gripper (auto-activating it if
        needed). The arm itself is not moved.
        """
        env, err = self._resolve(handle)
        if err:
            return err
        gate = self._require_reset()
        if gate:
            return gate
        from adapter.protocol import EnvAction

        try:
            result = env.step(EnvAction(action_type="tool_call", command={"gripper": direction}))
        except Exception as exc:  # pragma: no cover - hardware path
            return {"error": f"gripper_{direction} failed: {type(exc).__name__}: {exc}"}
        obs = result.observation if result.observation is not None else env.observe()
        return self._step_result(obs, info={"gripper": direction, "not_implemented": False})

    def base_control(self, forward: float = 0.0, lateral: float = 0.0, yaw: float = 0.0,
                     torso: float = 0.0, command: str = "", num_steps: int = 10,
                     session_id: str = "") -> dict[str, Any]:
        # UR5e is a fixed-base arm: base control is not applicable. Mirror the
        # simulator's rejection of base_control on non-mobile backends.
        return {"error": "base_control not supported on fixed-base UR5e"}

    # -- serialization ----------------------------------------------------
    @staticmethod
    def _observation_to_mcp(obs: EnvObservation) -> dict[str, Any]:
        # Top-level observation dict, matching sim. observe/render carry base64
        # image payloads (real bench has no dashboard; perception tools consume
        # the pixels directly).
        return obs.to_mcp_dict()

    def _step_result(self, obs: EnvObservation, *, info: dict[str, Any]) -> dict[str, Any]:
        return {
            "observation": self._observation_to_mcp(obs),
            "reward": 0.0, "terminated": False, "truncated": False,
            "info": info,
        }
