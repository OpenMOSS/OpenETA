#!/usr/bin/env python
"""UniDepth V2 MCP server for OpenETA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from tools.unidepth_v2_core import (  # noqa: E402
    DEFAULT_MODEL_ID,
    UniDepthV2Backend,
)


mcp = FastMCP("unidepth-v2", log_level="WARNING")
_BACKEND: UniDepthV2Backend | None = None


@mcp.tool()
def estimate_depth(
    rgb: dict[str, Any],
    intrinsics: dict[str, Any],
    *,
    camera_id: str = "camera",
    camera_model: str = "pinhole",
    calibration_profile_id: str = "",
    bundle_id: str = "",
    resolution_level: int | None = None,
) -> dict[str, Any]:
    """Estimate dense metric depth from one calibrated RGB image.

    Use this tool as a geometric prior for aligned RGB-D enhancement. It is not
    a sensor-depth replacement and its predictions must not be used as final
    collision-clearance evidence.

    Args:
        rgb: RGB payload as {"format": "png", "base64": "..."}; the MCP wire
            contract uses base64 image bytes, not local file paths.
        intrinsics: Calibrated pinhole fx, fy, cx, and cy for this exact image.
        camera_id: Source camera/frame identifier used for provenance.
        camera_model: Camera model. This service currently accepts "pinhole".
        calibration_profile_id: Active real-robot calibration profile id.
        bundle_id: Caller correlation id with no model semantics.
        resolution_level: Optional UniDepth V2 level in [0, 10); higher levels
            use a larger inference resolution and more compute.

    Example:
        {
            "rgb": {"format": "png", "base64": "<base64-encoded rgb png>"},
            "intrinsics": {
                "fx": 600.0,
                "fy": 600.0,
                "cx": 320.0,
                "cy": 240.0
            },
            "camera_id": "wrist",
            "camera_model": "pinhole",
            "calibration_profile_id": "wrist-rgbd-v3",
            "resolution_level": 4
        }

    The response encodes metric depth and confidence as float32 NPY base64.
    confidence_semantics="higher_is_better"; confidence is only a relative
    within-image ranking signal, not calibrated variance. Clients should
    materialize both arrays to local temporary files and keep calibrated sensor
    measurements as the safety channel.
    """

    if _BACKEND is None:
        return {
            "success": False,
            "content": "UniDepth V2 depth estimation failed: backend not configured.",
            "details": {
                "tool": "estimate_depth",
                "backend": "unidepth_v2_mcp",
                "model": DEFAULT_MODEL_ID,
                "reason": "model_load_failed",
                "confidence_semantics": "higher_is_better",
                "artifacts": [],
                "metadata": {},
            },
        }
    return _BACKEND.estimate_depth(
        rgb=rgb,
        intrinsics=intrinsics,
        camera_id=camera_id,
        camera_model=camera_model,
        calibration_profile_id=calibration_profile_id,
        bundle_id=bundle_id,
        resolution_level=resolution_level,
    )


def main() -> int:
    global _BACKEND
    parser = argparse.ArgumentParser(description="OpenETA UniDepth V2 MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8779)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resolution-level", type=int, default=4)
    args = parser.parse_args()

    _BACKEND = UniDepthV2Backend(
        model_id=args.model_id,
        device=args.device,
        resolution_level=args.resolution_level,
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
        return JSONResponse({"ok": True, "server": "unidepth-v2"})

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

    print(f"\n  UniDepth V2 MCP SSE: http://{args.host}:{args.port}/sse")
    print(f"  Health:              http://{args.host}:{args.port}/")
    uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
