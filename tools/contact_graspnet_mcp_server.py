#!/usr/bin/env python
"""Contact-GraspNet MCP server for OpenETA."""

from __future__ import annotations

import argparse
import math
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from tools.contact_graspnet_core import (
    CAMERA_FRAME,
    DEFAULT_DEPTH_MAX,
    DEFAULT_DEPTH_MIN,
    DEFAULT_MAX_CANDIDATES,
    FRAME,
    GRASP_FRAME,
    MODE,
    TOOL_NAME,
    ContactGraspNetBackend,
    failure_result,
)


mcp = FastMCP("contact_graspnet", log_level="WARNING")
_BACKEND: ContactGraspNetBackend | None = None
_PREDICT_LOCK = threading.Lock()


@mcp.tool()
def predict_grasps(
    depth: dict[str, Any],
    object_mask: dict[str, Any],
    intrinsics: dict[str, Any],
) -> dict[str, Any]:
    """Predict targeted camera-frame grasps from aligned depth and object mask.

    This MCP wire contract accepts base64-encoded PNG bytes, not local paths,
    artifact references, NPY arrays, or public point clouds. RGB is deliberately
    absent because Contact-GraspNet inference consumes geometry only.

    Args:
        depth: Aligned uint16 raw-depth PNG payload encoded as
            ``{"format": "png", "base64": "<base64-encoded depth png>"}``.
            Depth in meters is ``raw_depth / intrinsics["scale"]``. For uint16
            millimeter depth, use ``scale=1000``.
        object_mask: Aligned binary object-mask PNG payload encoded as
            ``{"format": "png", "base64": "<base64-encoded mask png>"}``.
            Nonzero pixels select the single targeted object.
        intrinsics: Finite pinhole-camera values ``fx``, ``fy``, ``cx``, ``cy``,
            and depth ``scale``. ``fx``, ``fy``, and ``scale`` must be positive.

    Example:
        {
            "depth": {
                "format": "png",
                "base64": "<base64-encoded uint16 depth png>"
            },
            "object_mask": {
                "format": "png",
                "base64": "<base64-encoded binary object mask png>"
            },
            "intrinsics": {
                "fx": 618.0,
                "fy": 618.0,
                "cx": 256.0,
                "cy": 256.0,
                "scale": 1000.0
            }
        }

    Depth and mask dimensions must match; inputs are never resized. Valid depth
    defaults to 0.2-1.8 meters. The service always uses targeted local-region
    inference and contact filtering. Returned poses are expressed in the OpenCV
    camera frame and normalized to the GraspNet grasp-frame convention used by
    OpenETA AnyGrasp and AnyPlace. Predictions assume a Panda-compatible two-
    finger gripper. The tool does not return RGB, base64 payloads, point clouds,
    visualizations, robot/world poses, or execute motion.
    """

    if _BACKEND is None:
        return failure_result(
            reason="model_load_failed",
            content="Contact-GraspNet grasp prediction failed: model_load_failed.",
            metadata={
                "frame": FRAME,
                "camera_frame": CAMERA_FRAME,
                "grasp_frame": GRASP_FRAME,
                "mode": MODE,
            },
        )
    with _PREDICT_LOCK:
        try:
            return _BACKEND.predict_grasps(
                depth=depth,
                object_mask=object_mask,
                intrinsics=intrinsics,
            )
        finally:
            _release_cuda_cache()


def _release_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    global _BACKEND
    parser = argparse.ArgumentParser(description="OpenETA Contact-GraspNet MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8776)
    parser.add_argument("--contact-graspnet-root", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_MIN)
    parser.add_argument("--depth-max", type=float, default=DEFAULT_DEPTH_MAX)
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    args = parser.parse_args()

    backend_root = Path(args.contact_graspnet_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not backend_root.is_dir():
        parser.error(f"--contact-graspnet-root is not a directory: {backend_root}")
    if not checkpoint_dir.is_dir():
        parser.error(f"--checkpoint-dir is not a directory: {checkpoint_dir}")
    if not (checkpoint_dir / "config.yaml").is_file():
        parser.error("--checkpoint-dir must contain config.yaml")
    if not (checkpoint_dir / "checkpoints" / "model.pt").is_file():
        parser.error("--checkpoint-dir must contain checkpoints/model.pt")
    if not math.isfinite(args.depth_min) or args.depth_min < 0:
        parser.error("--depth-min must be finite and non-negative")
    if not math.isfinite(args.depth_max) or args.depth_max <= args.depth_min:
        parser.error("--depth-max must be finite and greater than --depth-min")
    if args.max_candidates <= 0:
        parser.error("--max-candidates must be positive")

    _BACKEND = ContactGraspNetBackend(
        contact_graspnet_root=backend_root,
        checkpoint_dir=checkpoint_dir,
        seed=args.seed,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        max_candidates=args.max_candidates,
    )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return 0

    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(_request):
        return JSONResponse({"ok": True, "server": TOOL_NAME})

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

    print(f"\n  Contact-GraspNet MCP SSE: http://{args.host}:{args.port}/sse")
    print(f"  Health:                  http://{args.host}:{args.port}/")
    uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
