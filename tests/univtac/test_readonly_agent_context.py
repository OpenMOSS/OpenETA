from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from sim.envs.univtac.agent_context import (
    load_validated_pre_action_snapshot,
    materialize_operator_visible_input,
    project_operator_visible_context,
    validate_agent_input_directory,
)
from sim.envs.univtac.contract import UniVTACContractError, UniVTACTaskSnapshot


def _descriptor(path: str, shape: tuple[int, int, int]) -> dict:
    return {
        "path": path,
        "shape": list(shape),
        "dtype": "uint8",
        "value_range": [0.0, 255.0],
        "encoding": "png",
        "stored_dtype": "uint8",
    }


def _operator(root: Path) -> dict:
    images = {
        "head.png": (480, 270),
        "wrist.png": (480, 270),
        "left.png": (320, 240),
        "right.png": (320, 240),
    }
    for name, size in images.items():
        Image.new("RGB", size, (10, 20, 30)).save(root / name)
    return {
        "task_instruction": "Pull the key out of the slot.",
        "step_identifiers": {
            "snapshot_id": "pull-out-key-seed-1000000-ready:pre_action",
            "action_id": "pull-out-key-seed-1000000-ready",
            "phase": "pre_action",
            "simulator_step": 238,
            "take_action_count": 0,
        },
        "proprio": {"joint": [0.0] * 9, "ee": [0.0] * 7},
        "cameras": {
            "head": {"rgb": _descriptor("head.png", (270, 480, 3))},
            "wrist": {"rgb": _descriptor("wrist.png", (270, 480, 3))},
        },
        "tactile": {
            "left_tactile": {"rgb_marker": _descriptor("left.png", (240, 320, 3))},
            "right_tactile": {"rgb_marker": _descriptor("right.png", (240, 320, 3))},
        },
    }


def test_projection_uses_fixed_four_visible_images_and_no_paths_in_text(tmp_path: Path) -> None:
    context = project_operator_visible_context(_operator(tmp_path), tmp_path)
    assert [image.label for image in context.images] == [
        "camera/head/rgb",
        "camera/wrist/rgb",
        "tactile/left_tactile/rgb_marker",
        "tactile/right_tactile/rgb_marker",
    ]
    assert [(image.width, image.height) for image in context.images] == [
        (480, 270),
        (480, 270),
        (320, 240),
        (320, 240),
    ]
    assert all(image.dtype == "uint8" for image in context.images)
    assert all("path" not in descriptor for descriptor in context.visual_descriptors)


def test_materialized_input_contains_exactly_context_and_four_images(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    context = project_operator_visible_context(_operator(source), source)
    target = tmp_path / "agent_input"
    materialize_operator_visible_input(context, target)
    validate_agent_input_directory(target)
    files = sorted(
        path.relative_to(target).as_posix() for path in target.rglob("*") if path.is_file()
    )
    assert files == [
        "context.json",
        "images/head_rgb.png",
        "images/left_tactile_rgb_marker.png",
        "images/right_tactile_rgb_marker.png",
        "images/wrist_rgb.png",
    ]
    text = (target / "context.json").read_text(encoding="utf-8")
    assert str(source) not in text
    assert "host_only" not in text
    (target / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(UniVTACContractError, match="exactly five"):
        validate_agent_input_directory(target)


def test_projection_rejects_traversal_symlink_wrong_shape_and_raw_tactile(tmp_path: Path) -> None:
    operator = _operator(tmp_path)
    operator["cameras"]["head"]["rgb"]["path"] = "../head.png"
    with pytest.raises(UniVTACContractError, match="relative"):
        project_operator_visible_context(operator, tmp_path)

    root = tmp_path / "root"
    root.mkdir()
    operator = _operator(root)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (480, 270)).save(outside)
    (root / "head.png").unlink()
    (root / "head.png").symlink_to(outside)
    with pytest.raises(UniVTACContractError, match="symlink"):
        project_operator_visible_context(operator, root)

    (root / "head.png").unlink()
    operator = _operator(root)
    Image.new("RGB", (10, 10)).save(root / "head.png")
    with pytest.raises(UniVTACContractError, match="decoded shape"):
        project_operator_visible_context(operator, root)

    operator = _operator(root)
    operator["tactile"]["left_tactile"] = {
        "rgb": operator["tactile"]["left_tactile"].pop("rgb_marker")
    }
    with pytest.raises(Exception, match="rgb_marker"):
        project_operator_visible_context(operator, root)


def test_loading_snapshot_drops_host_only_sentinel(tmp_path: Path) -> None:
    operator = _operator(tmp_path)
    snapshot = UniVTACTaskSnapshot(
        snapshot_id="pull-out-key-seed-1000000-ready:pre_action",
        task_name="pull_out_key",
        seed=1_000_000,
        phase="pre_action",
        action_id="pull-out-key-seed-1000000-ready",
        simulator_step=238,
        take_action_count=0,
        task_instruction="Pull the key out of the slot.",
        external_camera={},
        tactile_sensors={},
        proprio={},
        operator_visible=operator,
        host_only={"sentinel": "DO_NOT_LEAK_HOST_ONLY_R0912"},
        artifacts={},
    )
    path = tmp_path / "snapshot_pre.json"
    path.write_text(snapshot.to_json(indent=2), encoding="utf-8")
    visible = load_validated_pre_action_snapshot(path, tmp_path)
    assert "DO_NOT_LEAK_HOST_ONLY_R0912" not in json.dumps(visible)
    assert "task_name" not in visible and "seed" not in visible
