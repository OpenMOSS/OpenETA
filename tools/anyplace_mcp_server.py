#!/usr/bin/env python
"""AnyPlace MCP server for OpenETA."""

from __future__ import annotations

import argparse
import threading
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from tools.anyplace_core import AnyPlaceBackend


mcp = FastMCP("anyplace", log_level="WARNING")
_BACKEND: AnyPlaceBackend | None = None
_PREDICT_LOCK = threading.Lock()


@mcp.tool()
def predict_placement(
    rgb: dict[str, Any] | None = None,
    depth: dict[str, Any] | None = None,
    object_mask: dict[str, Any] | None = None,
    placement_region_mask: dict[str, Any] | None = None,
    intrinsics: dict[str, Any] | None = None,
    selected_grasp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict AnyPlace transforms and corresponding camera-frame grasp poses.

    All image payloads must come from one aligned RGBD observation. The MCP
    wire contract uses base64-encoded image bytes, not local file paths and not
    OpenETA artifact refs. Depth is converted to metres as
    ``raw_depth / intrinsics["scale"]`` and projected in the OpenCV camera
    frame. Valid depth is truncated at 1 metre by default. The object and
    placement-region masks select the two AnyPlace point clouds internally.

    Args:
        rgb: RGB image payload such as
            ``{"format": "png", "base64": "<base64-encoded rgb png>"}``.
        depth: Aligned raw depth image payload such as
            ``{"format": "png", "base64": "<base64-encoded depth png>"}``.
        object_mask: Binary mask used by AnyGrasp for the selected object,
            encoded as ``<base64-encoded object mask png>``.
        placement_region_mask: Binary mask for valid local placement geometry,
            encoded as ``<base64-encoded placement-region mask png>``.
        intrinsics: Pinhole camera values ``fx``, ``fy``, ``cx``, ``cy``, and
            depth ``scale``. For uint16 millimetre depth, use ``scale=1000``.
        selected_grasp: One model-native AnyGrasp candidate in
            ``frame=camera`` and ``camera_frame=opencv``. The agent handler is
            responsible for unwrapping local Selected Grasp provenance before
            this MCP call.

    Example:
        {
            "rgb": {"format": "png", "base64": "<base64-encoded rgb png>"},
            "depth": {"format": "png", "base64": "<base64-encoded depth png>"},
            "object_mask": {"format": "png", "base64": "<base64-encoded object mask png>"},
            "placement_region_mask": {"format": "png", "base64": "<base64-encoded placement-region mask png>"},
            "intrinsics": {"fx": 618.0, "fy": 618.0, "cx": 256.0, "cy": 256.0, "scale": 1000.0},
            "selected_grasp": {"id": "grasp_003", "frame": "camera", "camera_frame": "opencv", "...": "..."}
        }

    The tool returns exactly five placement candidates in backend order. Each
    bundles ``object_placement_transform.transform_matrix`` with the
    corresponding ``place_grasp_pose``. It does not return point clouds, choose
    a best candidate, transform to robot/world frames, or execute motion. Do
    not materialize input base64 payloads in planner, memory, or action logs.
    """

    if _BACKEND is None:
        return {
            "success": False,
            "content": "AnyPlace placement prediction failed: backend not configured.",
            "details": {
                "tool": "anyplace",
                "backend": "anyplace_mcp",
                "model": "anyplace_multitask",
                "frame": "camera",
                "camera_frame": "opencv",
                "candidate_count": 0,
                "placement_candidates": [],
                "reason": "model_load_failed",
                "metadata": {},
            },
        }
    with _PREDICT_LOCK:
        try:
            return _BACKEND.predict_placement(
                rgb=rgb,
                depth=depth,
                object_mask=object_mask,
                placement_region_mask=placement_region_mask,
                intrinsics=intrinsics,
                selected_grasp=selected_grasp,
            )
        finally:
            _release_cuda_cache()


def _release_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    global _BACKEND
    parser = argparse.ArgumentParser(description="OpenETA AnyPlace MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--anyplace-root", required=True)
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    anyplace_root = Path(args.anyplace_root)
    config_path = Path(args.config_path)
    if not anyplace_root.exists():
        parser.error(f"--anyplace-root does not exist: {anyplace_root}")
    if not config_path.exists():
        parser.error(f"--config-path does not exist: {config_path}")

    _BACKEND = AnyPlaceBackend(
        anyplace_root=anyplace_root,
        config_path=config_path,
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
        return JSONResponse({"ok": True, "server": "anyplace"})

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

    print(f"\n  AnyPlace MCP SSE: http://{args.host}:{args.port}/sse")
    print(f"  Health:           http://{args.host}:{args.port}/")
    uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
