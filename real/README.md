# OpenETA Real-Robot Deployment (`real/`)

Hardware abstraction for driving real cameras and robot arms through the same
contracts OpenETA already uses for the simulator. A `RealRobotEnv` implements
`adapter.sim.SimulatorAdapter` and emits `adapter.protocol` types
(`CameraFrame`, `RobotState`, `EnvObservation`), so the agent and the existing
point-cloud / grasp / placement tools consume real sensor data unchanged.

## Layout

```text
real/
  cameras/    RGB-D camera drivers
    base.py       Camera ABC + CameraConfig
    realsense.py  Intel RealSense (D400 series + L515)
    webcam.py     RGB-only USB/UVC webcam + RTSP/HTTP stream (OpenCV)
  mcp/        MCP server exposing real hardware to the agent
    observation_server.py  sim-compatible server (create_env/reset_env/observe_env/…)
    observation_core.py    RealEnvManager: one lock-guarded RealRobotEnv -> MCP wire
  robots/     robot-arm drivers
    base.py       RobotArm ABC + RobotConfig
    ur5e.py       Universal Robots UR5e (ur_rtde)
    franka.py     Franka 7-DoF (stub, planned: franky/libfranka)
  registry.py  string-keyed driver registries (make_camera / make_robot)
  env.py       RealRobotEnv (SimulatorAdapter implementation)
  config/      example hardware configs
  examples/    runnable smoke scripts
```

## Supported hardware

| Kind   | Driver key(s)                                      | Backend        | Status |
|--------|----------------------------------------------------|----------------|--------|
| Camera | `realsense`, `realsense_l515`, `realsense_d435`    | `pyrealsense2` | ready  |
| Camera | `webcam`, `opencv`, `rtsp`                         | `opencv-python` | ready (RGB-only) |
| Arm    | `ur5e`                                             | `ur_rtde`      | ready  |
| Arm    | `franka`                                           | franky/libfranka | stub |

RGB-only cameras (webcam/RTSP) produce `CameraFrame.depth = None`. They cannot
self-report intrinsics, so supply calibrated `fx, fy, cx, cy` via
`CameraConfig.intrinsics` when downstream geometry (point clouds, grasp) is
needed. Select the source with `CameraConfig.device` (OpenCV index like `0`, or
an RTSP/HTTP URL).

## Install

Hardware SDKs are an optional extra and imported lazily, so `real/` stays
importable (and unit-testable) without them:

```bash
uv sync --extra real   # installs pyrealsense2, ur_rtde, opencv-python
```

## Quickstart

```python
from real.config.example_ur5e_realsense import build_env

env = build_env(task="pick up the red block")
obs = env.reset()          # observation-first: does NOT home the arm by default
obs = env.observe()        # RGB-D frames + robot proprioception
# env.step(EnvAction(action_type="tool_call", command={"pose": {...}}))
env.close()
```

`reset()` is a pure read by default. Pass `home_on_reset=True` to `RealRobotEnv`
to physically home the arm on reset (moves the robot — opt in deliberately).

## Real-robot MCP server

Exposes the **same tool contract as the simulator** MCP server
(`sim/mcp_server/server.py`) so the agent's `sim_mcp` client drives real
hardware unchanged: a `create_env` → `reset_env` → `observe_env`/`render_env` →
`close_env` handle+session lifecycle.

Because the bench is a single physical UR5e + cameras, `create_env` acquires a
**cross-process exclusive lock** (`fcntl.flock` on `--lock-file`, default
`/tmp/openeta_real_env.lock`). Only one env may exist at a time across all
processes; a second `create_env` returns `{"error": "real robot busy: …"}`
immediately (non-blocking). `close_env` releases the lock, and a crashed
holder's lock is auto-released by the OS.

The OpenETA TUI sends one best-effort `close_env` call when it actually exits
through `/quit`, Ctrl-C at the input prompt, EOF, `--once`, or an unhandled
error. Interrupting only the current agent run does not close the environment.
While an environment is active, `close_env` requires its matching `handle` and
also validates `session_id` when both sides provide one, so a stale client
cannot close a newer environment. Stopping the real MCP server itself performs
an internal forced cleanup before process exit.

```bash
uv sync --extra real
cp real/config/ur5e_bench.example.json real/config/ur5e_bench.json
uv run python -m real.mcp.observation_server --transport stdio \
    --config real/config/ur5e_bench.json
```

Lifecycle tools (return shapes match sim exactly):
- `create_env(env_id="", …)` — connect + lock; returns `{session_id, handle, backend, …}`.
  Sim-only args (render_mode/seed/image_*/robot) are accepted but ignored; the
  cell is defined by `--config`.
- `reset_env(handle)` — explicitly calls `robot.home()` and therefore moves the
  UR5e when `home_joints` is configured, then unlocks control tools and returns
  a top-level observation. Keep the configured environment's `home_on_reset`
  false to avoid duplicate homing.
- `observe_env(handle)` / `render_env(handle)` — read-only top-level observations
  (`task, cameras, robot, objects, metadata`); cameras carry `rgb_base64`
  (+ `depth_base64` for RGB-D).
- `close_env(handle, session_id="")` — validate ownership, close, and release
  the lock; idempotent after the environment is closed.
- `list_active_envs()` — the active env (0 or 1).

Control tools issue **physical motion** after a successful `reset_env`:
- `step_env` — relative Cartesian delta from the current end-effector pose.
- `move_to` — absolute Cartesian target via RTDE `moveL`.
- `gripper_open`, `gripper_close` — actuate the Robotiq gripper.
- `base_control` — unsupported and always errors because the UR5e is fixed-base.

Arm commands reject net translations over 0.60 m or rotations over 180 degrees
instead of clamping them. The simulator-compatible `move_to` arguments
`num_steps`, `tolerance`, `ori_tolerance`, and `enable_collision_check` are
currently ignored; the real MCP backend does not perform collision checking.

Legacy flat flags (`--robot-ip`, `--camera-device`, …) still build a single
webcam + optional UR5e for quick checks when `--config` is omitted. Use
`--transport sse --port N` for a long-lived HTTP/SSE server with a `/` health
endpoint.

Or build directly from the registry:

```python
import os

from real import make_camera, make_robot, RealRobotEnv
from real.cameras.base import CameraConfig
from real.robots.base import RobotConfig

cam = make_camera("realsense", CameraConfig(name="wrist"))
arm = make_robot("ur5e", RobotConfig(name="ur5e", ip=os.environ["OPENETA_UR5E_IP"]))
env = RealRobotEnv(cameras=[cam], robot=arm)
```

## Adding new hardware

1. Subclass `Camera`/`CameraConfig` (in `real/cameras/`) or
   `RobotArm`/`RobotConfig` (in `real/robots/`). Import the vendor SDK **inside**
   `start()`/`connect()`, never at module top level.
2. Return `CameraFrame` / `RobotState` from `read()` / `get_state()` so data
   flows through the shared contract. Depth is metric metres; intrinsics carry
   `fx, fy, cx, cy, scale`.
3. Register the driver in `real/registry.py` (or call `register_camera` /
   `register_robot` at import time).
4. Add the SDK to the `real` extra in `pyproject.toml`.

## Conventions

- **Poses:** `xyz` in metres. UR reports orientation as an axis-angle `rotvec`;
  convert to `quat_xyzw` downstream where a quaternion is required.
- **Depth:** linear metric depth in metres in `CameraFrame.depth`.
- **Safety:** `RealRobotEnv.step()` executes physical motion. Keep the action
  vocabulary explicit in `_apply_action` and validate commands before running on
  hardware.
