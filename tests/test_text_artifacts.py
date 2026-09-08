from __future__ import annotations

from pathlib import Path
import os

import pytest

from agent.runtime.text_artifacts import (
    DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT,
    grep_text_artifact,
    materialize_long_texts,
    read_text_artifact,
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


@pytest.mark.parametrize("count,truncated", [(0, False), (2, False), (3, True)])
def test_grep_requires_an_actual_extra_matching_line(tmp_path, count, truncated):
    path = tmp_path / "text.txt"
    path.write_text("hit\n" * count + "unrelated\n")
    result = grep_text_artifact(path, "hit", max_matches=2)
    assert result["match_count"] == min(count, 2)
    assert result["truncated"] is truncated
    assert result["total_match_count"] == (None if truncated else count)


def test_grep_snippet_and_read_cursor_reach_match_in_long_unicode_line(tmp_path):
    path = tmp_path / "data.jsonl"
    text = "中🙂" * 5000 + "TARGET" + "界" * 2000
    path.write_text(text, encoding="utf-8")
    result = grep_text_artifact(path, "target")
    hit = result["matches"][0]
    assert "TARGET" in hit["text"]
    assert len(hit["text"]) <= 500
    assert hit["match_start_column"] == 10000
    assert hit["snippet_clipped"] is True
    assert hit["match_clipped"] is False
    assert result["truncated"] is False
    page = read_text_artifact(path, cursor=hit["read_cursor"], max_chars=6)
    assert page["text"] == "TARGET"
    assert page["line"] == 1 and page["column"] == 10000


def test_grep_continuation_does_not_repeat_or_skip_matching_lines(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("hit one\nskip\nhit two\nhit three\n")
    first = grep_text_artifact(path, "hit", max_matches=2)
    second = grep_text_artifact(path, "hit", max_matches=2, cursor=first["next_cursor"])
    assert [hit["line"] for hit in first["matches"] + second["matches"]] == [1, 3, 4]
    assert second["truncated"] is False
    with pytest.raises(ValueError, match="different query"):
        grep_text_artifact(path, "other", cursor=first["next_cursor"])


@pytest.mark.parametrize("text", ["", "abc", "中🙂\r\nlong\rline\n" + "界" * 10000])
def test_text_pages_reconstruct_normalized_text_without_gaps(tmp_path, text):
    path = tmp_path / "text.txt"
    path.write_bytes(text.encode())
    pages, cursor = [], None
    while True:
        page = read_text_artifact(path, max_chars=127, cursor=cursor)
        prefix = "".join(pages)
        assert page["char_offset"] == len(prefix)
        assert page["line"] == prefix.count("\n") + 1
        assert page["column"] == len(prefix.rsplit("\n", 1)[-1])
        pages.append(page["text"])
        cursor = page["next_cursor"]
        if cursor is None:
            assert page["truncated"] is False
            break
    assert "".join(pages) == text.replace("\r\n", "\n").replace("\r", "\n")


def test_text_cursor_rejects_changed_or_different_file(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text("abcdef")
    second.write_text("abcdef")
    page = read_text_artifact(first, max_chars=2)
    with pytest.raises(ValueError, match="different file"):
        read_text_artifact(second, cursor=page["next_cursor"])
    first.write_text("ABCDEF")
    with pytest.raises(ValueError, match="changed"):
        read_text_artifact(first, cursor=page["next_cursor"])


def test_text_reader_rejects_nonregular_files_without_blocking(tmp_path):
    path = tmp_path / "fifo"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        read_text_artifact(path)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 201])
def test_grep_rejects_invalid_limits_before_reading(tmp_path, limit):
    with pytest.raises(ValueError, match="max_matches"):
        grep_text_artifact(tmp_path / "missing", "x", max_matches=limit)


def test_grep_reports_clipping_of_a_match_larger_than_snippet(tmp_path):
    path = tmp_path / "text"
    path.write_text("a" * 10000)
    hit = grep_text_artifact(path, "a+")["matches"][0]
    assert len(hit["text"]) == 500
    assert hit["match_end_column"] == 10000
    assert hit["match_clipped"] is True


def test_text_reader_detects_rewrite_during_page_read(tmp_path, monkeypatch):
    import agent.runtime.text_artifacts as artifacts
    path = tmp_path / "text"
    path.write_text("original text")
    skip = artifacts._skip_text

    def rewrite(handle, offset):
        position = skip(handle, offset)
        path.write_text("modified text")
        return position

    monkeypatch.setattr(artifacts, "_skip_text", rewrite)
    with pytest.raises(ValueError, match="changed during reading"):
        read_text_artifact(path, max_chars=4)


def test_zero_width_pattern_is_a_line_match_not_an_infinite_cursor(tmp_path):
    path = tmp_path / "text"
    path.write_text("alpha\nbeta\n")
    first = grep_text_artifact(path, "^", max_matches=1)
    second = grep_text_artifact(path, "^", max_matches=1, cursor=first["next_cursor"])
    assert first["matches"][0]["line"] == 1
    assert second["matches"][0]["line"] == 2
    assert second["truncated"] is False
