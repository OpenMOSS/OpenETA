#!/usr/bin/env python
"""SAM3 MCP server for OpenETA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from tools.sam3_core import segment_image_prompt


mcp = FastMCP("sam3", log_level="WARNING")


@mcp.tool()
def segment(
    image_base64: str,
    prompt: str = "",
    *,
    box_xyxy: list[float] | None = None,
    image_format: str = "png",
    confidence_threshold: float = 0.5,
) -> dict:
    """Segment one image with either a text prompt or one positive box.

    Use this MCP tool when a caller has RGB image bytes and needs object
    masks/boxes for an open-vocabulary phrase. The MCP wire contract uses
    base64 payloads, not local file paths and not OpenETA artifact refs.

    Args:
        image_base64: Base64-encoded source image bytes.
        prompt: Short text phrase for the target object or concept.
        box_xyxy: Optional positive box in source-image pixel coordinates.
            When present, the backend uses SAM3's geometric-prompt path on the
            original image; `prompt` is retained only as a readable label.
        image_format: Source image encoding such as png, jpg, or jpeg.
        confidence_threshold: Detection confidence threshold.

    Example:
        {
            "image_base64": "<base64-encoded png bytes>",
            "image_format": "png",
            "prompt": "black shoe",
            "confidence_threshold": 0.5
        }

    Detections are returned with ranking=score_descending, a zero-based rank,
    and backend_index preserving the model's original output position. The
    response may include mask and overlay image payloads as base64. Clients
    should materialize those payloads into local temporary files, then pass file
    refs to downstream tools instead of reading or logging the base64 directly.
    """

    return segment_image_prompt(
        image_base64=image_base64,
        prompt=prompt,
        box_xyxy=box_xyxy,
        image_format=image_format,
        confidence_threshold=confidence_threshold,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenETA SAM3 MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8773)
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return 0

    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def health(_request):
        return JSONResponse(
            {
                "ok": True,
                "server": "sam3",
            }
        )

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

    print(f"\n  SAM3 MCP SSE: http://{args.host}:{args.port}/sse")
    print(f"  Health:       http://{args.host}:{args.port}/")
    uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
