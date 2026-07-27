#!/usr/bin/env python
"""AnyGrasp MCP server for OpenETA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from tools.anygrasp_core import AnyGraspBackend


mcp = FastMCP("anygrasp", log_level="WARNING")
_BACKEND: AnyGraspBackend | None = None


@mcp.tool()
def detect_grasps(
    rgb: dict[str, Any],
    depth: dict[str, Any],
    intrinsics: dict[str, Any],
    *,
    mode: str = "targeted",
    target_mask: dict[str, Any] | None = None,
    approach_steering: list[float] | None = None,
    approach_thresh: float | None = None,
    collision_detection: bool = True,
    dense_grasp: bool = False,
    depth_truncation: float | None = None,
    depth_min: float | None = None,
) -> dict[str, Any]:
    """Detect camera-frame grasp candidates from RGB-D input.

    Use this MCP tool after perception has selected either a target region or
    the whole scene. The MCP wire contract uses base64 payloads, not local file
    paths and not OpenETA artifact refs.

    Args:
        rgb: RGB image payload as {"format": "...", "base64": "..."}.
        depth: Raw depth image payload as {"format": "...", "base64": "..."}.
            Depth in meters is computed as raw_depth / intrinsics["scale"].
            For uint16 millimeter depth, use scale=1000.
        intrinsics: Camera intrinsics with finite fx, fy, cx, cy, and scale.
            fx, fy, and scale must be positive.
        mode: "targeted" requires target_mask; "scene" uses the full scene and
            must omit target_mask.
        target_mask: Optional mask payload as {"format": "...", "base64": "..."}.
            Nonzero pixels are the target grasp region and zero pixels are
            background.
        approach_steering: Optional camera-frame [x, y, z] approach direction.
        approach_thresh: Optional approach direction threshold in radians.
        collision_detection: Whether to run AnyGrasp collision filtering.
        dense_grasp: Whether to request dense grasp generation.
        depth_truncation: Optional per-call far cut in metres. Points beyond this
            range are dropped before inference, overriding the server default for
            this call only. Set this per sensor/scene (e.g. ~1.5 for D435, ~2.0
            for L515) so distant stray returns never enter the cloud.
        depth_min: Optional near cut in metres. Points closer than this are
            dropped, removing sub-working-distance noise such as the D435 dead
            zone (~0.2 m). Must be below depth_truncation. Defaults to 0.0.

    Example:
        {
            "rgb": {"format": "png", "base64": "<base64-encoded rgb png>"},
            "depth": {"format": "png", "base64": "<base64-encoded depth png>"},
            "intrinsics": {
                "fx": 600.0,
                "fy": 600.0,
                "cx": 320.0,
                "cy": 240.0,
                "scale": 1000.0
            },
            "mode": "targeted",
            "target_mask": {
                "format": "png",
                "base64": "<base64-encoded binary mask png>"
            },
            "collision_detection": true,
            "dense_grasp": false
        }

    rgb, depth, and target_mask must share image dimensions; this tool does not
    resize inputs. If the response includes image payloads as base64, clients
    should materialize those payloads into local temporary files, then pass file
    refs to downstream tools instead of reading or logging the base64 directly.
    Successful candidates use ranking="score_descending" and include zero-based
    rank plus backend_index for traceability. The first candidate is the greedy
    default, not a guarantee that downstream safety or motion checks will pass.
    Result metadata describes the decoded depth dtype, raw and metric ranges,
    normalized intrinsics, depth truncation, and valid point count. Invalid depth
    scales, apparent uint16 scale mismatches, and empty post-filter point clouds
    are returned as structured input failures before model inference.
    """

    if _BACKEND is None:
        return {
            "success": False,
            "content": "AnyGrasp grasp detection failed: backend not configured.",
            "details": {
                "tool": "anygrasp",
                "backend": "anygrasp_mcp",
                "model": "anygrasp_sdk",
                "mode": mode,
                "candidate_count": 0,
                "grasp_candidates": [],
                "artifacts": [],
                "reason": "model_load_failed",
                "metadata": {},
            },
        }
    try:
        return _BACKEND.detect_grasps(
            rgb=rgb,
            depth=depth,
            intrinsics=intrinsics,
            mode=mode,
            target_mask=target_mask,
            approach_steering=approach_steering,
            approach_thresh=approach_thresh,
            collision_detection=collision_detection,
            dense_grasp=dense_grasp,
            depth_truncation=depth_truncation,
            depth_min=depth_min,
        )
    finally:
        _release_cuda_cache()


def _release_cuda_cache() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    global _BACKEND
    parser = argparse.ArgumentParser(description="OpenETA AnyGrasp MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--max-gripper-width", type=float, default=0.1)
    parser.add_argument("--gripper-height", type=float, default=0.03)
    parser.add_argument("--depth-truncation", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()

    _BACKEND = AnyGraspBackend(
        sdk_root=args.sdk_root,
        checkpoint_path=args.checkpoint_path,
        max_gripper_width=args.max_gripper_width,
        gripper_height=args.gripper_height,
        depth_truncation=args.depth_truncation,
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
        return JSONResponse({"ok": True, "server": "anygrasp"})

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

    print(f"\n  AnyGrasp MCP SSE: http://{args.host}:{args.port}/sse")
    print(f"  Health:          http://{args.host}:{args.port}/")
    uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
