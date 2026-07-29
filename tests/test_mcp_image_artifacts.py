"""Tests for MCP image artifact materialization."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from adapter.protocol import CameraFrame
from agent.runtime.image_artifacts import DEFAULT_MCP_IMAGE_OUTPUT_ROOT, materialize_mcp_images
from agent.runtime.runtime import OpenEtaAgentRuntime


PNG_1X1 = base64.b64encode(
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\nIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00"
    b"\x18\xdd\x8d\xb0"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
).decode("ascii")


def test_materialize_mcp_images_writes_files_and_scrubs_payload(tmp_path: Path) -> None:
    payload = {
        "observation": {
            "task": "pick cube",
            "cameras": [
                {
                    "frame_id": "front",
                    "role": "scene_primary",
                    "rgb_base64": PNG_1X1,
                    "depth_base64": PNG_1X1,
                    "width": 1,
                    "height": 1,
                }
            ],
        },
        "reward": 0.0,
    }

    bundle = materialize_mcp_images(
        payload,
        output_root=tmp_path,
        bundle_id="bundle-a",
    )

    camera = bundle.payload["observation"]["cameras"][0]
    assert "rgb_base64" not in camera
    assert "depth_base64" not in camera
    assert camera["rgb_base64_omitted"] is True
    assert camera["depth_base64_omitted"] is True
    assert camera["rgb_ref"] == "observation.cameras.0.front.rgb"
    assert camera["depth_ref"] == "observation.cameras.0.front.depth"
    assert {image.role for image in bundle.images} == {"scene_primary"}
    assert {image.to_dict()["role"] for image in bundle.images} == {"scene_primary"}
    assert len(bundle.images) == 2
    for image in bundle.images:
        path = Path(image.path)
        assert path.exists()
        assert path.read_bytes().startswith(b"\x89PNG")
        assert path.parent.parent.name in {"rgb", "depth"}
    assert PNG_1X1 not in json.dumps(bundle.to_dict())


def test_camera_role_round_trips_through_mcp_camera_packet() -> None:
    camera = CameraFrame(
        frame_id="zed_head",
        role="scene_primary",
        rgb=[[[1, 2, 3]]],
        depth=[[1.25]],
        extrinsics={
            "camera_frame": "opencv",
            "normalized_from": "omnigibson_usd",
        },
    )

    serialized = camera.to_dict()
    mcp_payload = camera.to_mcp_dict()
    restored = CameraFrame.from_dict(serialized)

    assert serialized["role"] == "scene_primary"
    assert mcp_payload["role"] == "scene_primary"
    assert mcp_payload["depth_encoding"] == "uint16_png"
    assert mcp_payload["depth_scale"] == 1000.0
    assert restored.role == "scene_primary"


def test_legacy_opengl_camera_keeps_existing_mcp_depth_shape() -> None:
    payload = CameraFrame(
        frame_id="agentview",
        rgb=[[[1, 2, 3]]],
        depth=[[1.25]],
        extrinsics={"camera_frame": "opengl"},
    ).to_mcp_dict()

    assert "depth_encoding" not in payload
    assert "depth_scale" not in payload


def test_default_mcp_image_output_root_uses_repo_tmp_image_tree() -> None:
    assert DEFAULT_MCP_IMAGE_OUTPUT_ROOT == Path("tmp") / "image"


def test_materialized_images_are_isolated_by_session_and_safe_from_traversal(
    tmp_path: Path,
) -> None:
    payload = {"frame_id": "front", "rgb_base64": PNG_1X1}

    first = materialize_mcp_images(
        payload,
        output_root=tmp_path,
        session_id="session-a",
        bundle_id="same-bundle",
    )
    second = materialize_mcp_images(
        payload,
        output_root=tmp_path,
        session_id="session-b",
        bundle_id="same-bundle",
    )
    escaped = materialize_mcp_images(
        payload,
        output_root=tmp_path,
        session_id="../outside",
        bundle_id="../../same-bundle",
    )

    first_path = Path(first.images[0].path)
    second_path = Path(second.images[0].path)
    escaped_path = Path(escaped.images[0].path)
    assert first_path != second_path
    assert first_path.relative_to(tmp_path).parts[0] == "session-a"
    assert second_path.relative_to(tmp_path).parts[0] == "session-b"
    assert escaped_path.is_relative_to(tmp_path.resolve())
    assert ".." not in escaped_path.relative_to(tmp_path.resolve()).parts


def test_materialize_mcp_images_tool_returns_lightweight_refs(tmp_path: Path) -> None:
    runtime = OpenEtaAgentRuntime()
    result = runtime.tools.call(
        "materialize_mcp_images",
        {
            "payload": {
                "cameras": [
                    {
                        "frame_id": "agentview",
                        "rgb_base64": PNG_1X1,
                        "width": 1,
                        "height": 1,
                    }
                ]
            },
            "output_root": str(tmp_path),
            "bundle_id": "runtime-bundle",
        },
    )

    assert result.success is True
    assert result.details["result_type"] == "bookkeeping"
    assert result.details["outputs"]["bundle_id"] == "runtime-bundle"
    assert result.details["outputs"]["payload"]["cameras"][0]["rgb_ref"] == (
        "cameras.0.agentview.rgb"
    )
    assert len(result.details["artifacts"]) == 1
    assert Path(result.details["artifacts"][0]["path"]).exists()
    assert PNG_1X1 not in json.dumps(result.details)
