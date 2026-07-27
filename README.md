# OpenETA Real-Robot Release

OpenETA is a closed-loop embodied agent runtime with a real-robot MCP backend
for UR5e arms, Robotiq grippers, and Intel RealSense cameras. The agent uses the
same observation and tool-call contracts for simulation and physical hardware,
while the real backend owns hardware connection, lifecycle, and command
serialization.

## Safety

This software can command physical hardware.

- `reset_env` may move the arm to configured home joints.
- `move_to`, `step_env`, `gripper_open`, and `gripper_close` issue physical
  motion after reset.
- The real MCP backend rejects individual arm commands over 0.60 m or 180
  degrees, but it does not currently implement collision checking.
- Validate workspace limits, payload, tool geometry, emergency stops, and
  controller safety settings before enabling motion.

## Install

Python 3.10 or newer and `uv` are recommended:

```bash
uv sync --extra real
```

The `real` extra installs `pyrealsense2`, `ur_rtde`, and OpenCV. Vendor SDKs are
imported lazily so non-hardware tests can run without connected devices.

## Configure

Tracked files contain templates only. Create local configuration files before
starting services:

```bash
cp .env.example .env
cp .mcp.example.json .mcp.json
cp real/config/ur5e_bench.example.json real/config/ur5e_bench.json
```

Set the controller address, camera serials, calibrated extrinsics, model
endpoints, and credentials locally. `.env`, `.mcp.json`, and
`real/config/ur5e_bench.json` are ignored by Git.

## Start The Real MCP Server

```bash
uv run python -m real.mcp.observation_server \
  --transport sse \
  --host 0.0.0.0 \
  --port 8780 \
  --config real/config/ur5e_bench.json
```

The service exposes `create_env`, `reset_env`, `observe_env`, `render_env`,
`move_to`, `step_env`, gripper controls, and lifecycle cleanup. A cross-process
file lock prevents multiple environments from controlling the same cell.

## Run The Agent

Configure `openeta-ur5e` in `.mcp.json`, then select it as the environment MCP:

```bash
OPENETA_SIM_MCP_SERVER=openeta-ur5e uv run openeta
```

Perception MCP services are optional and configured independently in
`.mcp.json`. See `docs/mcp-services.md` for supported service processes and
`real/README.md` for the hardware adapter contract.

## Calibration

Do not publish deployment calibration. Keep camera serials, transforms, and
cell geometry in the ignored local bench configuration. The generic hand-eye
workflow is documented in `real/calibration/README.md`.

## Test

```bash
uv run pytest -q
```

Unit tests use fake hardware and do not command a physical robot.

## Layout

```text
adapter/    Shared action and observation contracts
agent/      Closed-loop planner, memory, tools, and CLI
real/       Camera, robot, calibration, and real MCP implementations
tools/      Perception and manipulation MCP services
scripts/    Service launch and evaluation utilities
tests/      Offline unit and integration tests
```
