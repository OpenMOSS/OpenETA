#!/usr/bin/env python
"""Real-robot MCP server for OpenETA.

Exposes the same tool contract as the simulator MCP server
(``sim/mcp_server/server.py``) so the agent drives real hardware through the
identical ``create_env`` → ``reset_env`` → ``observe_env``/``render_env`` →
``close_env`` lifecycle. A single physical UR5e + cameras is guarded by a
cross-process exclusive lock: only one env may exist at a time (``create_env``
errors if another process holds the lock; ``close_env`` releases it).

Control tools (``step_env``/``move_to``/``gripper_open``/``gripper_close``/
``base_control``) are registered but **stubbed** — they validate the handle and
return a real observation without issuing motion. Wire real motion later behind
collision / joint-limit / velocity guards.

    uv sync --extra real
    uv run python -m real.mcp.observation_server --transport stdio \
        --config real/config/ur5e_bench.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from real.mcp.observation_core import RealEnvManager  # noqa: E402


mcp = FastMCP("real-observation", log_level="WARNING")
_MANAGER: RealEnvManager | None = None


def _mgr() -> RealEnvManager:
    if _MANAGER is None:
        raise RuntimeError("real env manager not configured")
    return _MANAGER


@mcp.tool()
def create_env(env_id: str = "", *, render_mode: str = "rgb_array", seed: int = 0,
               task: str = "", session_id: str = "",
               image_width: int | None = None, image_height: int | None = None,
               include_objects: bool = False, robot: str = "") -> dict[str, Any]:
    """Create the real-robot environment (single, lock-guarded).

    Connects to the physical UR5e + cameras and acquires an exclusive
    cross-process lock. Only ONE env may exist across all processes; if another
    process already holds the lock this returns ``{"error": ...}`` immediately
    (non-blocking). Call ``close_env`` to release it.

    ``env_id`` and the sim-only args (render_mode/seed/image_*/include_objects/
    robot) are accepted for wire-compatibility but ignored — the real cell is
    defined by the server's ``--config`` file.

    Returns ``{session_id, handle, env_id, action_dim, backend, action_hint}``.
    Keep ``handle`` (and ``session_id``) for every later call.
    """
    return _mgr().create_env(
        env_id=env_id, task=task, session_id=session_id,
        render_mode=render_mode, seed=seed, image_width=image_width,
        image_height=image_height, include_objects=include_objects, robot=robot,
    )


@mcp.tool()
def reset_env(handle: str = "", *, seed: int | None = None,
              session_id: str = "") -> dict[str, Any]:
    """Reset the env and return the initial observation (top-level dict).

    Observation-first: reset does NOT home/move the arm unless the underlying
    env was built with ``home_on_reset=True``. Returns ``task, cameras, robot,
    objects, metadata``; cameras carry base64 PNG frames.
    """
    return _mgr().reset_env(handle, seed=seed, session_id=session_id)


@mcp.tool()
def observe_env(handle: str = "", *, session_id: str = "") -> dict[str, Any]:
    """Return the current observation without moving the robot.

    Returns the same structure as ``reset_env``: ``task, cameras, robot,
    objects, metadata``. Cameras include ``rgb_base64`` and (for RGB-D)
    ``depth_base64``; recover metric depth with ``depth_m = px / 1000.0``.
    """
    return _mgr().observe_env(handle, session_id=session_id)


@mcp.tool()
def render_env(handle: str = "", *, session_id: str = "") -> dict[str, Any]:
    """Return a fresh render (fresh camera frames + proprio). Same shape as observe_env."""
    return _mgr().render_env(handle, session_id=session_id)


@mcp.tool()
def close_env(handle: str = "", *, session_id: str = "") -> dict[str, Any]:
    """Close the matching env and release the exclusive lock. Idempotent.

    ``handle`` is required while an env is active. ``session_id`` is also
    checked when both client and server have one, preventing a stale client
    from closing a newer environment. Returns ``{ok, already_closed,
    cleanup_errors}``. After this, another process may ``create_env``.
    """
    return _mgr().close_env(handle, session_id=session_id)


@mcp.tool()
def list_active_envs(*, session_id: str = "") -> dict[str, Any]:
    """Return the active env (0 or 1) as ``{session_id, count, envs}``."""
    return _mgr().list_active_envs(session_id=session_id)


# ── control tools (REAL MOTION, safety-capped) ─────────────────────────
@mcp.tool()
def step_env(handle: str = "", action: list | None = None, *,
             num_steps: int = 1, session_id: str = "") -> dict[str, Any]:
    """Step the env with a RELATIVE Cartesian delta from the current EEF pose.

    ``action`` is ``[dx, dy, dz]`` (metres) or ``[dx, dy, dz, droll, dpitch,
    dyaw]`` (rotation in degrees), scaled by ``num_steps``. Net translation and
    rotation are capped for safety — an over-cap request is REJECTED (returns
    ``error``), never clamped. Shape matches sim:
    ``{observation, reward, terminated, truncated, info}``."""
    return _mgr().step_env(handle, action, num_steps=num_steps, session_id=session_id)


@mcp.tool()
def move_to(handle: str = "", x: float = 0.0, y: float = 0.0, z: float = 0.0, *,
            roll: float | None = None, pitch: float | None = None, yaw: float | None = None,
            num_steps: int = 100, tolerance: float = 0.002, ori_tolerance: float = 0.05,
            session_id: str = "", enable_collision_check: bool = True) -> dict[str, Any]:
    """Move the end-effector to an ABSOLUTE Cartesian pose via RTDE moveL.

    ``x,y,z`` are world metres; ``roll,pitch,yaw`` are degrees (all three
    required together to control orientation, else orientation is locked to the
    current pose). The net move from the current EEF pose is capped for safety —
    an over-cap request is REJECTED (returns ``error``), never clamped. Returns
    ``{target, start, end, steps_executed, terminated, reward, info}``."""
    return _mgr().move_to(
        handle, x, y, z, roll=roll, pitch=pitch, yaw=yaw, num_steps=num_steps,
        tolerance=tolerance, ori_tolerance=ori_tolerance, session_id=session_id,
        enable_collision_check=enable_collision_check,
    )


@mcp.tool()
def gripper_open(handle: str = "", *, session_id: str = "") -> dict[str, Any]:
    """Open the Robotiq gripper (real motion, auto-activates). Shape matches step_env."""
    return _mgr().gripper_open(handle, session_id=session_id)


@mcp.tool()
def gripper_close(handle: str = "", *, session_id: str = "") -> dict[str, Any]:
    """Close the Robotiq gripper (real motion, auto-activates). Shape matches step_env."""
    return _mgr().gripper_close(handle, session_id=session_id)


@mcp.tool()
def base_control(forward: float = 0.0, lateral: float = 0.0, yaw: float = 0.0,
                 torso: float = 0.0, command: str = "", num_steps: int = 10,
                 session_id: str = "") -> dict[str, Any]:
    """[STUB] Base control. UR5e is fixed-base, so this always returns an error."""
    return _mgr().base_control(
        forward=forward, lateral=lateral, yaw=yaw, torso=torso,
        command=command, num_steps=num_steps, session_id=session_id,
    )


def _build_env_factory(args: argparse.Namespace):
    """Return a zero-arg callable that builds a RealRobotEnv.

    Two ways to describe the cell:
      * ``--config <path>`` (preferred): build the multi-camera cell from a
        JSON config via ``real.config.loader``.
      * legacy flat flags (``--camera-device``, ``--robot-ip``, ...): a single
        webcam + optional UR5e, for quick single-camera checks.
    """
    if args.config:
        def factory():
            from real.config.loader import build_env_from_file
            task = args.task if args.task else None
            return build_env_from_file(args.config, task=task)
        return factory

    def factory():
        from real.cameras.base import CameraConfig
        from real.cameras.webcam import WebcamCamera
        from real.env import RealRobotEnv
        from real.robots.base import RobotConfig
        from real.robots.ur5e import UR5eArm

        intrinsics: dict[str, float] = {}
        if args.fx and args.fy:
            intrinsics = {
                "fx": args.fx, "fy": args.fy,
                "cx": args.cx if args.cx is not None else args.width / 2.0,
                "cy": args.cy if args.cy is not None else args.height / 2.0,
                "scale": 1000.0,  # depth PNG is uint16 mm; depth_m = raw / scale
            }
        camera = WebcamCamera(
            CameraConfig(
                name=args.camera_frame_id, device=args.camera_device,
                width=args.width, height=args.height, fps=args.fps,
                intrinsics=intrinsics,
            )
        )
        robot = None
        if args.robot_ip:
            robot = UR5eArm(RobotConfig(name=args.arm_name, ip=args.robot_ip, dof=6))
        # home_on_reset stays False: observation-first, never moves the arm.
        return RealRobotEnv(cameras=[camera], robot=robot, task=args.task)

    return factory


def main() -> int:
    global _MANAGER
    parser = argparse.ArgumentParser(description="OpenETA real-robot MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--lock-file", default="/tmp/openeta_real_env.lock",
                        help="Path to the cross-process exclusive lock file.")
    parser.add_argument("--config", default="",
                        help="Path to a JSON hardware config (see real/config/ur5e_bench.json). "
                        "When set, the flat --camera-*/--robot-* flags are ignored.")
    parser.add_argument("--robot-ip", default="", help="UR5e controller IP (RTDE, read-only).")
    parser.add_argument("--arm-name", default="arm_1")
    parser.add_argument("--camera-device", default="/dev/cam_left",
                        help="OpenCV device index or /dev path or RTSP/HTTP URL.")
    parser.add_argument("--camera-frame-id", default="wrist")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--task", default="")
    args = parser.parse_args()

    _MANAGER = RealEnvManager(
        _build_env_factory(args), task=args.task, lock_file=args.lock_file,
    )

    try:
        if args.transport == "stdio":
            mcp.run(transport="stdio")
            return 0

        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route

        async def health(_request):
            return JSONResponse({"ok": True, "server": "real-observation"})

        health_app = Starlette(routes=[Route("/", health, methods=["GET"])])
        sse_transport = SseServerTransport("/sse/messages/")

        async def combined(scope, receive, send):
            if scope["type"] == "http":
                path = scope["path"]
                if path == "/sse" and scope["method"] == "GET":
                    async with sse_transport.connect_sse(scope, receive, send) as streams:
                        await mcp._mcp_server.run(
                            streams[0], streams[1],
                            mcp._mcp_server.create_initialization_options(),
                        )
                    return
                if path.startswith("/sse/messages/") and scope["method"] == "POST":
                    await sse_transport.handle_post_message(scope, receive, send)
                    return
            await health_app(scope, receive, send)

        print(f"\n  Real MCP SSE: http://{args.host}:{args.port}/sse")
        print(f"  Health:       http://{args.host}:{args.port}/")
        uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
        return 0
    finally:
        if _MANAGER is not None:
            _MANAGER.close()


if __name__ == "__main__":
    raise SystemExit(main())
