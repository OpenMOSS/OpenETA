from __future__ import annotations

from pathlib import Path

from agent.runtime.text_artifacts import (
    DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT,
    grep_text_artifact,
    materialize_long_texts,
)


def test_default_text_output_root_uses_repo_tmp_tool_result_tree() -> None:
    assert DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT == Path("tmp") / "tool_result" / "text"


def test_materialize_long_texts_writes_files_and_keeps_grep_refs(tmp_path: Path) -> None:
    payload = {
        "content": "header\n" + ("needle line\n" * 400),
        "nested": {"short": "ok"},
    }

    bundle = materialize_long_texts(
        payload,
        output_root=tmp_path,
        bundle_id="bundle",
        max_inline_chars=100,
        preview_chars=32,
    )

    assert len(bundle.artifacts) == 1
    artifact = bundle.artifacts[0]
    assert Path(artifact.path).exists()
    assert bundle.payload["content_text_omitted"] is True
    assert bundle.payload["content_text_path"] == artifact.path
    assert "needle line" in Path(artifact.path).read_text(encoding="utf-8")
    assert "grep -n" in bundle.payload["content_grep_hint"]

    matches = grep_text_artifact(artifact.path, "needle", max_matches=2)
    assert matches["match_count"] == 2
    assert matches["truncated"] is True


def test_materialize_long_texts_does_not_replace_base64_image_fields(tmp_path: Path) -> None:
    image_payload = "a" * 500
    payload = {
        "cameras": [
            {
                "frame_id": "front",
                "rgb_base64": image_payload,
                "content": "log\n" + ("needle line\n" * 100),
            }
        ]
    }

    bundle = materialize_long_texts(
        payload,
        output_root=tmp_path,
        bundle_id="bundle",
        max_inline_chars=100,
        preview_chars=32,
    )

    camera = bundle.payload["cameras"][0]
    assert camera["rgb_base64"] == image_payload
    assert "rgb_base64_text_path" not in camera
    assert camera["content_text_omitted"] is True


def test_long_texts_with_same_bundle_are_isolated_by_session(tmp_path: Path) -> None:
    first = materialize_long_texts(
        {"content": "a" * 200},
        output_root=tmp_path,
        session_id="session-a",
        bundle_id="same",
        max_inline_chars=10,
    )
    second = materialize_long_texts(
        {"content": "b" * 200},
        output_root=tmp_path,
        session_id="session-b",
        bundle_id="same",
        max_inline_chars=10,
    )

    first_path = Path(first.artifacts[0].path)
    second_path = Path(second.artifacts[0].path)
    assert first_path != second_path
    assert first_path.relative_to(tmp_path).parts[0] == "session-a"
    assert second_path.relative_to(tmp_path).parts[0] == "session-b"
    assert first_path.read_text() == "a" * 200
    assert second_path.read_text() == "b" * 200
