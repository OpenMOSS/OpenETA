from __future__ import annotations

import json
from pathlib import Path

from agent.runtime.response_artifacts import (
    DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT,
    build_observation_snapshot,
    build_response_reference,
    materialize_json_response,
)


def test_materialize_json_response_strips_preview_values(tmp_path: Path) -> None:
    payload = {
        "ok": True,
        "handle": "env-1",
        "preview": "drop me",
        "cameras": [
            {
                "frame_id": "front",
                "rgb_path": "/tmp/front.png",
                "preview": "drop nested",
            }
        ],
        "content": "full response text",
        "results": [
            {"id": "openeta/demo-0", "description": "first"},
            {"id": "openeta/demo-1", "description": "second"},
        ],
    }

    artifact = materialize_json_response(
        payload,
        output_root=tmp_path,
        bundle_id="bundle",
    )
    saved = json.loads(Path(artifact.path).read_text(encoding="utf-8"))
    reference = build_response_reference(payload, artifact)

    assert saved["content"] == "full response text"
    assert "preview" not in saved
    assert "preview" not in saved["cameras"][0]
    assert reference["handle"] == "env-1"
    assert reference["response_path"] == artifact.path
    assert reference["cameras"][0]["rgb_path"] == "/tmp/front.png"
    assert reference["results_count"] == 2
    assert "results" not in reference


def test_default_response_output_root_uses_repo_tmp_tool_result_tree() -> None:
    assert DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT == Path("tmp") / "tool_result"


def test_camera_roles_survive_compact_refs_and_observation_snapshots(
    tmp_path: Path,
) -> None:
    payload = {
        "observation": {
            "task": "pick the cup",
            "cameras": [
                {
                    "frame_id": "zed_head",
                    "role": "scene_primary",
                    "rgb_path": str(tmp_path / "zed.png"),
                    "intrinsics": {
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 50.0,
                        "cy": 50.0,
                    },
                }
            ],
        }
    }
    image_artifacts = [
        {
            "kind": "rgb",
            "frame_id": "zed_head",
            "role": "scene_primary",
            "path": str(tmp_path / "zed.png"),
        }
    ]
    artifact = materialize_json_response(payload, output_root=tmp_path)

    reference = build_response_reference(
        payload,
        artifact,
        image_artifacts=image_artifacts,
    )
    snapshot = build_observation_snapshot(
        payload,
        image_artifacts=image_artifacts,
    )

    assert reference["cameras"][0]["role"] == "scene_primary"
    assert snapshot["observation"]["cameras"][0]["role"] == "scene_primary"
    assert snapshot["observation"]["metadata"]["image_artifacts"][0]["role"] == (
        "scene_primary"
    )


def test_json_responses_with_same_bundle_are_isolated_by_session(tmp_path: Path) -> None:
    first = materialize_json_response(
        {"session": "a"},
        output_root=tmp_path,
        session_id="session-a",
        bundle_id="same",
    )
    second = materialize_json_response(
        {"session": "b"},
        output_root=tmp_path,
        session_id="session-b",
        bundle_id="same",
    )

    assert first.path != second.path
    assert Path(first.path).relative_to(tmp_path).parts[0] == "session-a"
    assert Path(second.path).relative_to(tmp_path).parts[0] == "session-b"
    assert json.loads(Path(first.path).read_text())["session"] == "a"
    assert json.loads(Path(second.path).read_text())["session"] == "b"
