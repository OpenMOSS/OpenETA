from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import TextContent
from PIL import Image as PilImage

from sim.envs.univtac.contract import UniVTACTaskSnapshot
from sim.envs.univtac.tactile_causal_pilot import stage_condition
from tools.univtac_operator_mcp_server import (
    build_observe_blocks,
    build_server,
    record_operator_context,
)

EXPECTED_LABELS = [
    "camera/head/rgb",
    "camera/wrist/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
]


def _descriptor(path: str, shape: tuple[int, int, int]) -> dict:
    return {
        "path": path,
        "shape": list(shape),
        "dtype": "uint8",
        "value_range": [0.0, 255.0],
        "encoding": "png",
        "stored_dtype": "uint8",
    }


def make_episode(tmp_path: Path) -> tuple[Path, Path]:
    episode = tmp_path / "episode"
    simulator = episode / "simulator"
    seed_dir = simulator / "pull_out_key_seed1000000"
    specs = {
        "pre/camera/head_rgb.png": (480, 270),
        "pre/camera/wrist_rgb.png": (480, 270),
        "pre/tactile/left_tactile_rgb_marker.png": (320, 240),
        "pre/tactile/right_tactile_rgb_marker.png": (320, 240),
    }
    for index, (relative, size) in enumerate(specs.items()):
        path = seed_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        PilImage.new("RGB", size, (index * 20, 30, 40)).save(path)
    operator_visible = {
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
            "head": {
                "rgb": _descriptor(
                    "pull_out_key_seed1000000/pre/camera/head_rgb.png", (270, 480, 3)
                )
            },
            "wrist": {
                "rgb": _descriptor(
                    "pull_out_key_seed1000000/pre/camera/wrist_rgb.png", (270, 480, 3)
                )
            },
        },
        "tactile": {
            "left_tactile": {
                "rgb_marker": _descriptor(
                    "pull_out_key_seed1000000/pre/tactile/left_tactile_rgb_marker.png",
                    (240, 320, 3),
                )
            },
            "right_tactile": {
                "rgb_marker": _descriptor(
                    "pull_out_key_seed1000000/pre/tactile/right_tactile_rgb_marker.png",
                    (240, 320, 3),
                )
            },
        },
    }
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
        operator_visible=operator_visible,
        host_only={"sentinel": "DO_NOT_LEAK_HOST_ONLY_R0913"},
        artifacts={},
    )
    snapshot_path = seed_dir / "snapshot_pre.json"
    snapshot_path.write_text(snapshot.to_json(indent=2), encoding="utf-8")
    return episode, snapshot_path


def test_observe_returns_one_text_and_four_native_images_without_host_only(
    tmp_path: Path,
) -> None:
    episode, snapshot = make_episode(tmp_path)
    blocks = build_observe_blocks(episode_root=episode, snapshot_path=snapshot)
    assert len(blocks) == 5
    assert isinstance(blocks[0], TextContent)
    assert all(isinstance(block, Image) for block in blocks[1:])
    payload = json.loads(blocks[0].text)
    assert [item["label"] for item in payload["images"]] == EXPECTED_LABELS
    assert "DO_NOT_LEAK_HOST_ONLY_R0913" not in blocks[0].text
    assert not ({"host_only", "press_depth", "actor_state", "plan_success"} & payload.keys())
    for block in blocks[1:]:
        content = block.to_image_content()
        assert content.type == "image"
        assert content.mimeType == "image/png"


def test_operator_context_records_relative_paths_and_refuses_second_observe(
    tmp_path: Path,
) -> None:
    episode, snapshot = make_episode(tmp_path)
    blocks = build_observe_blocks(episode_root=episode, snapshot_path=snapshot)
    row = record_operator_context(episode_root=episode, blocks=blocks, timestamp_s=1.5)
    assert row["tool"] == "observe"
    assert row["arguments"] == {}
    assert len(row["response_image_paths"]) == 4
    assert all(not Path(path).is_absolute() for path in row["response_image_paths"])
    with __import__("pytest").raises(RuntimeError, match="exactly once"):
        record_operator_context(episode_root=episode, blocks=blocks)


def test_server_exposes_only_observe(tmp_path: Path) -> None:
    episode, snapshot = make_episode(tmp_path)
    server = build_server(episode_root=episode, snapshot_path=snapshot)
    assert [tool.name for tool in server._tool_manager.list_tools()] == ["observe"]


def test_condition_manifest_returns_two_or_swapped_four_images_without_disclosure(
    tmp_path: Path,
) -> None:
    source_episode, snapshot = make_episode(tmp_path)
    simulator = source_episode / "simulator"
    source_dir = simulator / "pull_out_key_seed1000000/pre"
    image_map = {
        "camera/head/rgb": source_dir / "camera/head_rgb.png",
        "camera/wrist/rgb": source_dir / "camera/wrist_rgb.png",
        "tactile/left_tactile/rgb_marker": source_dir / "tactile/left_tactile_rgb_marker.png",
        "tactile/right_tactile/rgb_marker": source_dir / "tactile/right_tactile_rgb_marker.png",
    }
    visual = tmp_path / "visual"
    visual.mkdir()
    stage_condition(
        condition="visual_only",
        context_images=image_map,
        episode_root=visual,
        simulator_root=simulator,
    )
    visual_blocks = build_observe_blocks(
        episode_root=visual,
        snapshot_path=snapshot,
        condition_manifest=visual / "condition.json",
    )
    assert len(visual_blocks) == 3
    assert len(json.loads(visual_blocks[0].text)["images"]) == 2

    swapped = tmp_path / "swapped"
    swapped.mkdir()
    stage_condition(
        condition="swapped_tactile",
        context_images=image_map,
        episode_root=swapped,
        simulator_root=simulator,
    )
    swapped_blocks = build_observe_blocks(
        episode_root=swapped,
        snapshot_path=snapshot,
        condition_manifest=swapped / "condition.json",
    )
    text = swapped_blocks[0].text
    assert "swapped_tactile" not in text and "source_label" not in text
    left_target = np.asarray(PilImage.open(swapped_blocks[3].path))
    original_right = np.asarray(PilImage.open(image_map[EXPECTED_LABELS[3]]))
    right_target = np.asarray(PilImage.open(swapped_blocks[4].path))
    original_left = np.asarray(PilImage.open(image_map[EXPECTED_LABELS[2]]))
    assert np.array_equal(left_target, original_right)
    assert np.array_equal(right_target, original_left)
