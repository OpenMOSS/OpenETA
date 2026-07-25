#!/usr/bin/env python
"""MolmoPoint MCP server for OpenETA."""

from __future__ import annotations

import argparse
import re
import sys
import threading
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcp.server.fastmcp import FastMCP

from tools.molmopoint_core import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    TOOL_NAME,
    MolmoPointBackend,
    failure_result,
)


mcp = FastMCP(TOOL_NAME, log_level="WARNING")
_BACKEND: MolmoPointBackend | None = None
_POINT_LOCK = threading.Lock()


@mcp.tool()
def point_image(
    images: list[dict[str, Any]],
    prompt: str,
) -> dict[str, Any]:
    """Point to one or more locations in an ordered set of images.

    The MCP wire contract accepts one to four base64-encoded PNG or JPEG image
    envelopes plus one complete English pointing prompt. It does not accept
    local paths, URLs, artifact references, video, masks, depth, or intrinsics.
    The prompt is passed to the official MolmoPoint chat template after trimming
    outer whitespace; the service does not add or rewrite ``Point to``.

    Args:
        images: Ordered image envelopes in the form
            ``{"format": "png", "base64": "<base64-encoded image>"}``.
        prompt: Complete natural-language pointing instruction. Use ``any`` or
            ``a`` for one instance, and ``all``, ``every``, or ``each`` for all
            instances. Relational descriptions are supported.

    Single-instance example::

        {
          "images": [{"format": "png", "base64": "<image>"}],
          "prompt": "Point to any green can in the image."
        }

    All-instances example::

        {
          "images": [{"format": "jpeg", "base64": "<image>"}],
          "prompt": "Point to every red block in the image."
        }

    Relational example::

        {
          "images": [{"format": "png", "base64": "<image>"}],
          "prompt": "Locate the cup to the left of the bowl."
        }

    Reference-image example::

        {
          "images": [
            {"format": "png", "base64": "<reference image>"},
            {"format": "png", "base64": "<scene image>"}
          ],
          "prompt": "Look at the object in Image 1. In Image 2, point to the same object."
        }

    Prompt names such as Image 1 are one-based, while returned ``image_index``
    values are zero-based, so Image 2 maps to ``image_index=1``. The model may
    return points for both reference and scene images; callers must select by
    ``image_index``. Zero points is a successful result. Points retain model
    order and use finite, top-left-origin pixel coordinates in each original
    image. The tool does not return confidence, cross-image object identities,
    raw model text, base64, masks, visualizations, or 3D coordinates.

    Each image is limited to 8192 pixels per side and 32 million pixels; the
    ordered set is limited to 64 million pixels in total. Images are not
    resized. Multi-frame images and nonstandard EXIF orientations are rejected.
    If any input or returned point is invalid, the whole call fails atomically
    and returns no points.
    """

    if _BACKEND is None:
        return failure_result(
            reason="model_load_failed",
            metadata={"model_revision": DEFAULT_MODEL_REVISION},
        )
    with _POINT_LOCK:
        return _BACKEND.point_images(images=images, prompt=prompt)


def main() -> int:
    global _BACKEND
    parser = argparse.ArgumentParser(description="OpenETA MolmoPoint MCP server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--hf-home", type=Path)
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.model_revision):
        parser.error("--model-revision must be a full 40-character commit SHA")
    torch = _validate_cuda_runtime(parser)
    model_path = _resolve_local_snapshot(
        parser=parser,
        model_id=args.model_id,
        model_revision=args.model_revision,
        hf_home=args.hf_home,
    )
    _BACKEND = MolmoPointBackend(
        model_path=model_path,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    del torch

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

    print(f"\n  MolmoPoint MCP SSE: http://{args.host}:{args.port}/sse")
    print(f"  Health:             http://{args.host}:{args.port}/")
    uvicorn.run(combined, host=args.host, port=args.port, log_level="warning")
    return 0


def _validate_cuda_runtime(parser: argparse.ArgumentParser) -> Any:
    try:
        import torch
    except ImportError:
        parser.error("MolmoPoint requires PyTorch in its dedicated service environment")
    if not torch.cuda.is_available():
        parser.error("MolmoPoint requires a CUDA device; CPU fallback is not supported")
    if not torch.cuda.is_bf16_supported():
        parser.error("MolmoPoint requires CUDA bfloat16 support")
    return torch


def _resolve_local_snapshot(
    *,
    parser: argparse.ArgumentParser,
    model_id: str,
    model_revision: str,
    hf_home: Path | None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(
            repo_id=model_id,
            revision=model_revision,
            cache_dir=(hf_home / "hub") if hf_home is not None else None,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001 - startup diagnostic boundary.
        parser.error(
            "MolmoPoint model snapshot is unavailable locally for "
            f"{model_id}@{model_revision}: {type(exc).__name__}"
        )
    return Path(model_path)


if __name__ == "__main__":
    raise SystemExit(main())
