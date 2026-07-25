#!/usr/bin/env python
"""GraspGenX MCP server for OpenETA."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from tools.graspgenx_core import (
    CAMERA_FRAME,
    FRAME,
    GRASP_FRAME,
    LIST_TOOL_NAME,
    SERVER_NAME,
    TOOL_NAME,
    GraspGenXBackend,
    failure_result,
)


_BACKEND: GraspGenXBackend | None = None
_PREDICT_LOCK = threading.Lock()


def predict_grasps(
    depth: dict[str, Any],
    object_mask: dict[str, Any],
    intrinsics: dict[str, Any],
    gripper_name: str,
    up_direction_camera: list[float],
) -> dict[str, Any]:
    """Predict targeted camera-frame grasps for a named gripper.

    This geometry-only MCP tool accepts Base64-encoded image bytes, not local
    paths, artifact references, NPY arrays, or public point clouds. RGB is not
    required or accepted.

    Args:
        depth: Raw depth image payload as
            ``{"format": "png", "base64": "<base64-encoded depth>"}``.
            The format field is advisory; the bytes must be Pillow-decodable.
            Depth in meters is raw depth divided by ``intrinsics.scale``. For
            uint16 millimeter depth, use ``scale=1000``.
        object_mask: Aligned object-mask image payload. Nonzero pixels select
            the target object and zero pixels represent visible scene geometry.
        intrinsics: Finite pinhole-camera values ``fx``, ``fy``, ``cx``, ``cy``
            and depth ``scale``. ``fx``, ``fy`` and ``scale`` must be positive.
        gripper_name: Required name from the schema enum. Call
            ``list_grippers`` for the same validated list and its geometry.
        up_direction_camera: Nonzero gravity-opposing world-up vector expressed
            in the OpenCV camera frame. It is normalized internally. A camera
            looking straight down commonly uses approximately ``[0, 0, -1]``.

    Example:
        {
            "depth": {
                "format": "png",
                "base64": "<base64-encoded uint16 depth>"
            },
            "object_mask": {
                "format": "png",
                "base64": "<base64-encoded binary mask>"
            },
            "intrinsics": {
                "fx": 618.0,
                "fy": 618.0,
                "cx": 256.0,
                "cy": 256.0,
                "scale": 1000.0
            },
            "gripper_name": "franka_panda",
            "up_direction_camera": [0.0, 0.0, -1.0]
        }

    Depth and mask dimensions must match and inputs are never resized. Valid
    depth uses ``0 < camera_z < 1 meter`` by default. At least 100 valid target
    points are required; larger point clouds, including those above 3500 points,
    are passed to GraspGenX without wrapper downsampling or padding.

    The fixed GraspMoE planner combines diffusion and OBB candidates. Scores are
    sorted descending without a 0.7 cutoff and at most 20 collision-free grasps
    are returned. Collision filtering only covers visible non-target geometry in
    the supplied depth image; it is not robot motion planning. Inference is
    stochastic, so repeated calls can differ.

    Returned poses are in ``camera/opencv``. Each candidate contains both the
    GraspNet/AnyGrasp-compatible pose used by OpenETA and the original GraspGenX
    grasp-frame pose. The tool does not return Base64 payloads, point clouds,
    meshes, visualizations, world/robot poses, or execute motion.
    """

    with _PREDICT_LOCK:
        try:
            if _BACKEND is None:
                return failure_result(
                    reason="model_load_failed",
                    metadata={
                        "frame": FRAME,
                        "camera_frame": CAMERA_FRAME,
                        "grasp_frame": GRASP_FRAME,
                    },
                )
            return _BACKEND.predict_grasps(
                depth=depth,
                object_mask=object_mask,
                intrinsics=intrinsics,
                gripper_name=gripper_name,
                up_direction_camera=up_direction_camera,
            )
        finally:
            _release_cuda_cache()


def list_grippers() -> dict[str, Any]:
    """List validated GraspGenX grippers and public compatibility geometry.

    The result is sorted by gripper name and is generated from the same asset
    scan used to build ``predict_grasps.gripper_name``. It includes gripper type,
    fingertip depth, and open/mid sweep-volume extents and offsets. Calling this
    tool never loads model weights, creates a gripper sampler, or occupies CUDA.
    """

    if _BACKEND is None:
        return {
            "success": False,
            "content": "GraspGenX gripper listing failed: model_load_failed.",
            "details": {
                "tool": LIST_TOOL_NAME,
                "reason": "model_load_failed",
                "gripper_count": 0,
                "grippers": [],
                "model_loaded": False,
            },
        }
    return _BACKEND.list_grippers()


def build_mcp(gripper_names: list[str] | tuple[str, ...]) -> FastMCP:
    """Build a FastMCP instance whose gripper parameter has a dynamic enum."""

    names = tuple(sorted(str(name) for name in gripper_names))
    if not names:
        raise ValueError("at least one gripper name is required")
    mcp = FastMCP(SERVER_NAME, log_level="WARNING")

    def dynamic_predict_grasps(
        depth: dict[str, Any],
        object_mask: dict[str, Any],
        intrinsics: dict[str, Any],
        gripper_name: str,
        up_direction_camera: list[float],
    ) -> dict[str, Any]:
        return predict_grasps(
            depth=depth,
            object_mask=object_mask,
            intrinsics=intrinsics,
            gripper_name=gripper_name,
            up_direction_camera=up_direction_camera,
        )

    dynamic_predict_grasps.__annotations__ = {
        "depth": dict[str, Any],
        "object_mask": dict[str, Any],
        "intrinsics": dict[str, Any],
        "gripper_name": Literal.__getitem__(names),
        "up_direction_camera": list[float],
        "return": dict[str, Any],
    }
    dynamic_predict_grasps.__name__ = TOOL_NAME
    dynamic_predict_grasps.__doc__ = predict_grasps.__doc__
    mcp.tool()(dynamic_predict_grasps)
    mcp.tool()(list_grippers)
    return mcp


def health_payload() -> dict[str, Any]:
    backend = _BACKEND
    return {
        "ok": backend is not None,
        "server": SERVER_NAME,
        "tools": [LIST_TOOL_NAME, TOOL_NAME],
        "model_loaded": bool(backend is not None and backend.model_loaded),
        "gripper_count": 0 if backend is None else len(backend.grippers),
    }


def _release_cuda_cache() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    global _BACKEND
    parser = argparse.ArgumentParser(description="OpenETA GraspGenX MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--graspgenx-root", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--gripper-descriptions-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    source_root = Path(args.graspgenx_root).expanduser().resolve()
    if not (source_root / "graspgenx" / "__init__.py").is_file():
        parser.error(
            "--graspgenx-root must contain the graspgenx Python package"
        )
    try:
        backend = GraspGenXBackend(
            graspgenx_root=source_root,
            checkpoint_root=args.checkpoint_root,
            gripper_descriptions_root=args.gripper_descriptions_root,
            device=args.device,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    _BACKEND = backend

    for name, reason in sorted(backend.invalid_grippers.items()):
        print(f"Skipping invalid GraspGenX gripper asset {name!r}: {reason}", file=sys.stderr)

    mcp = build_mcp(list(backend.grippers))
    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return 0

    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(_request):
        return JSONResponse(health_payload())

    health_app = Starlette(routes=[Route("/", health, methods=["GET"])])
    sse_transport = SseServerTransport("/sse/messages/")

    async def combined(scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            if path == "/sse" and scope["method"] == "GET":
                async with sse_transport.connect_sse(scope, receive, send) as streams:
                    await mcp._mcp_server.run(
                        streams[0],
                        streams[1],
                        mcp._mcp_server.create_initialization_options(),
                    )
                return
            if path.startswith("/sse/messages/") and scope["method"] == "POST":
                await sse_transport.handle_post_message(scope, receive, send)
                return
        await health_app(scope, receive, send)

    print(f"\n  GraspGenX MCP SSE: http://{args.host}:{args.port}/sse")
    print(f"  Health:              http://{args.host}:{args.port}/")
    uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
