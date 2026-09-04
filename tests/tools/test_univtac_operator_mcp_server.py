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
    StagedGroundingSession,
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


def test_six_image_difference_manifest_returns_native_images_without_host_metadata(
    tmp_path: Path,
) -> None:
    source_episode, snapshot = make_episode(tmp_path)
    episode = tmp_path / "difference"
    images = episode / "images"
    images.mkdir(parents=True)
    records = []
    labels = [
        "camera/head/rgb",
        "camera/wrist/rgb",
        "tactile/left_tactile/current_rgb_marker",
        "tactile/left_tactile/difference",
        "tactile/right_tactile/current_rgb_marker",
        "tactile/right_tactile/difference",
    ]
    for index, label in enumerate(labels):
        path = images / f"image_{index}.png"
        PilImage.new("RGB", (8, 6), (index, 20, 30)).save(path)
        records.append(
            {
                "label": label,
                "path": f"images/{path.name}",
                "source_path": f"host-only-{index}",
                "shape": [6, 8, 3],
                "dtype": "uint8",
                "media_type": "image/png",
            }
        )
    manifest = {
        "condition": "explicit_difference",
        "simulator_root": str(source_episode / "simulator"),
        "images": records,
    }
    (episode / "condition.json").write_text(json.dumps(manifest), encoding="utf-8")
    blocks = build_observe_blocks(
        episode_root=episode,
        snapshot_path=snapshot,
        condition_manifest=episode / "condition.json",
    )
    assert len(blocks) == 7
    text = blocks[0].text
    assert "explicit_difference" not in text and "source_path" not in text
    assert "host-only" not in text
    assert [item["label"] for item in json.loads(text)["images"]] == labels


def test_structured_tactile_summary_is_visible_without_host_reference(
    tmp_path: Path,
) -> None:
    source_episode, snapshot = make_episode(tmp_path)
    episode = tmp_path / "structured"
    images = episode / "images"
    images.mkdir(parents=True)
    labels = [
        "camera/head/rgb",
        "camera/wrist/rgb",
        "tactile/left_tactile/current_rgb_marker",
        "tactile/left_tactile/difference",
        "tactile/right_tactile/current_rgb_marker",
        "tactile/right_tactile/difference",
    ]
    records = []
    for index, label in enumerate(labels):
        path = images / f"image_{index}.png"
        PilImage.new("RGB", (8, 6), (index, 20, 30)).save(path)
        records.append(
            {
                "label": label,
                "path": f"images/{path.name}",
                "shape": [6, 8, 3],
                "dtype": "uint8",
                "media_type": "image/png",
            }
        )
    per_sensor = {
        "normalized_horizontal_saliency": [0.2, 0.6, 0.2],
        "active_pixel_ratio": 0.25,
        "mean_absolute_change": 19.0,
        "p95_absolute_change": 103.0,
        "salient_mass": 1.8,
    }
    manifest = {
        "condition": "host-only-condition-name",
        "simulator_root": str(source_episode / "simulator"),
        "images": records,
        "model_visible_structured_tactile_summary": {
            "vector_order": ["left", "center", "right"],
            "left_tactile": per_sensor,
            "right_tactile": per_sensor,
        },
        "structured_source_mapping": {"host-only": "DO_NOT_SHOW"},
        "reference_region": "DO_NOT_SHOW",
        "press_depth": "DO_NOT_SHOW",
    }
    (episode / "condition.json").write_text(json.dumps(manifest), encoding="utf-8")
    blocks = build_observe_blocks(
        episode_root=episode,
        snapshot_path=snapshot,
        condition_manifest=episode / "condition.json",
    )
    payload = json.loads(blocks[0].text)
    assert len(blocks) == 7
    assert payload["tactile_change_summary"]["vector_order"] == [
        "left",
        "center",
        "right",
    ]
    text = blocks[0].text
    assert "host-only-condition-name" not in text
    assert "DO_NOT_SHOW" not in text
    assert "press_depth" not in text and "reference_region" not in text


def test_scalar_unilateral_summary_returns_exactly_four_images_without_mapping_leak(
    tmp_path: Path,
) -> None:
    source_episode, snapshot = make_episode(tmp_path)
    episode = tmp_path / "unilateral"
    images = episode / "images"
    images.mkdir(parents=True)
    records = []
    for index, label in enumerate(EXPECTED_LABELS):
        path = images / f"image_{index}.png"
        PilImage.new("RGB", (8, 6), (index, 20, 30)).save(path)
        records.append(
            {
                "label": label,
                "path": f"images/{path.name}",
                "shape": [6, 8, 3],
                "dtype": "uint8",
                "media_type": "image/png",
            }
        )
    scalar = {
        "active_pixel_ratio": 0.2,
        "mean_absolute_change": 18.0,
        "p95_absolute_change": 103.0,
        "salient_mass": 1.8,
    }
    manifest = {
        "condition": "swapped_structured_guidance",
        "simulator_root": str(source_episode / "simulator"),
        "images": records,
        "model_visible_structured_tactile_summary": {
            "left_tactile": scalar,
            "right_tactile": scalar,
        },
        "image_source_mapping": {"left_tactile": "current", "right_tactile": "baseline"},
        "expected_image_change_side": "left",
        "counterfactual_multimodal_input": True,
    }
    (episode / "condition.json").write_text(json.dumps(manifest), encoding="utf-8")
    blocks = build_observe_blocks(
        episode_root=episode,
        snapshot_path=snapshot,
        condition_manifest=episode / "condition.json",
    )
    assert len(blocks) == 5
    payload = json.loads(blocks[0].text)
    assert payload["tactile_change_summary"] == {
        "left_tactile": scalar,
        "right_tactile": scalar,
    }
    text = blocks[0].text
    assert not any(
        term in text
        for term in (
            "swapped_structured_guidance",
            "baseline",
            "current",
            "expected_image_change_side",
            "counterfactual_multimodal_input",
        )
    )


def test_staged_grounding_state_machine_commits_before_guidance(tmp_path: Path) -> None:
    source_episode, snapshot = make_episode(tmp_path)
    episode = tmp_path / "staged"
    images = episode / "images"
    images.mkdir(parents=True)
    records = []
    for index, label in enumerate(EXPECTED_LABELS):
        path = images / f"image_{index}.png"
        PilImage.new("RGB", (8, 6), (index, 20, 30)).save(path)
        records.append(
            {
                "label": label,
                "path": f"images/{path.name}",
                "shape": [6, 8, 3],
                "dtype": "uint8",
                "media_type": "image/png",
            }
        )
    scalar = {
        "active_pixel_ratio": 0.2,
        "mean_absolute_change": 18.0,
        "p95_absolute_change": 103.0,
        "salient_mass": 1.8,
    }
    manifest = {
        "simulator_root": str(source_episode / "simulator"),
        "images": records,
        "model_visible_structured_tactile_summary": {
            "left_tactile": scalar,
            "right_tactile": {key: 0.0 for key in scalar},
        },
    }
    manifest_path = episode / "condition.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    session = StagedGroundingSession(
        episode_root=episode,
        snapshot_path=snapshot,
        condition_manifest=manifest_path,
    )
    with __import__("pytest").raises(RuntimeError, match="committed"):
        session.observe_structured_guidance()
    with __import__("pytest").raises(RuntimeError, match="observe_images"):
        session.record_image_judgment(
            tactile_image_access="available",
            left_tactile_state="clear_change",
            right_tactile_state="little_or_no_change",
            tactile_changed_side="left",
            left_visual_cue="localized_colored_disturbance",
            right_visual_cue="regular_grid",
            evidence_summary="The left image has a localized disturbance.",
        )
    image_blocks = session.observe_images()
    assert len(image_blocks) == 5
    assert "tactile_change_summary" not in image_blocks[0].text
    with __import__("pytest").raises(RuntimeError, match="exactly once"):
        session.observe_images()
    judgment = {
        "tactile_image_access": "available",
        "left_tactile_state": "clear_change",
        "right_tactile_state": "little_or_no_change",
        "tactile_changed_side": "left",
        "left_visual_cue": "localized_colored_disturbance",
        "right_visual_cue": "regular_grid",
        "evidence_summary": "The left image has a localized disturbance.",
    }
    session.record_image_judgment(**judgment)
    assert json.loads((episode / "image_judgment.json").read_text()) == judgment
    with __import__("pytest").raises(RuntimeError, match="observe_images"):
        session.record_image_judgment(**judgment)
    guidance = session.observe_structured_guidance()
    assert json.loads(guidance[0].text)["tactile_change_summary"]["left_tactile"] == scalar
    with __import__("pytest").raises(RuntimeError, match="committed"):
        session.observe_structured_guidance()
    rows = [json.loads(line) for line in (episode / "operator_context.jsonl").read_text().splitlines()]
    assert [row["seq"] for row in rows] == [1, 2, 3]
    assert [row["tool"] for row in rows] == [
        "observe_images",
        "record_image_judgment",
        "observe_structured_guidance",
    ]


def test_staged_server_exposes_only_three_grounding_tools(tmp_path: Path) -> None:
    source_episode, snapshot = make_episode(tmp_path)
    episode = tmp_path / "staged-server"
    images = episode / "images"
    images.mkdir(parents=True)
    records = []
    for index, label in enumerate(EXPECTED_LABELS):
        path = images / f"image_{index}.png"
        PilImage.new("RGB", (8, 6), (index, 20, 30)).save(path)
        records.append(
            {
                "label": label,
                "path": f"images/{path.name}",
                "shape": [6, 8, 3],
                "dtype": "uint8",
                "media_type": "image/png",
            }
        )
    scalar = {
        "active_pixel_ratio": 0.2,
        "mean_absolute_change": 18.0,
        "p95_absolute_change": 103.0,
        "salient_mass": 1.8,
    }
    manifest = episode / "condition.json"
    manifest.write_text(
        json.dumps(
            {
                "simulator_root": str(source_episode / "simulator"),
                "images": records,
                "model_visible_structured_tactile_summary": {
                    "left_tactile": scalar,
                    "right_tactile": scalar,
                },
            }
        ),
        encoding="utf-8",
    )
    server = build_server(
        episode_root=episode,
        snapshot_path=snapshot,
        condition_manifest=manifest,
        mode="staged",
    )
    assert [tool.name for tool in server._tool_manager.list_tools()] == [
        "observe_images",
        "record_image_judgment",
        "observe_structured_guidance",
    ]
