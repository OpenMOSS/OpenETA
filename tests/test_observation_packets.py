from __future__ import annotations

from pathlib import Path

import pytest

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.observation_packets import ObservationPacketResolutionError


def _observation(*, packet_id: str, rgb: Path, depth: Path) -> EnvObservation:
    return EnvObservation(
        task="pick cube",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                role="scene_primary",
                rgb=[],
                intrinsics={"fx": 100.0, "fy": 101.0, "cx": 32.0, "cy": 24.0},
                extrinsics={"camera_to_world": [[1.0, 0.0, 0.0, 0.0]]},
                timestamp_s=12.5,
            )
        ],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": str(rgb),
                    "packet_id": packet_id,
                },
                {
                    "kind": "depth",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": str(depth),
                    "packet_id": packet_id,
                },
            ]
        },
    )


def test_historical_packet_resolution_survives_resume(tmp_path: Path) -> None:
    rgb_old = tmp_path / "old-rgb.png"
    depth_old = tmp_path / "old-depth.png"
    rgb_new = tmp_path / "new-rgb.png"
    depth_new = tmp_path / "new-depth.png"
    for path in (rgb_old, depth_old, rgb_new, depth_new):
        path.write_bytes(b"artifact")

    store = JsonMemoryStore(tmp_path / "memory")
    memory = AgentMemory(store=store)
    memory.start_session(task="pick cube", session_id="packet-session")
    memory.add_observation(
        _observation(packet_id="packet-old", rgb=rgb_old, depth=depth_old)
    )
    memory.add_observation(
        _observation(packet_id="packet-new", rgb=rgb_new, depth=depth_new)
    )

    old = memory.resolve_observation_packet("packet-old")
    assert old["rgb"] == str(rgb_old)
    assert old["depth"] == str(depth_old)
    assert old["frame_id"] == "agentview"
    assert old["intrinsics"]["fx"] == 100.0

    resumed = AgentMemory(store=JsonMemoryStore(tmp_path / "memory"))
    resumed.resume_session("packet-session")
    assert resumed.resolve_observation_packet("packet-old")["rgb"] == str(rgb_old)


def test_unknown_packet_error_exposes_bounded_repair_refs(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    rgb.write_bytes(b"rgb")
    depth.write_bytes(b"depth")
    memory = AgentMemory()
    memory.start_session(task="pick cube", session_id="packet-session")
    memory.add_observation(_observation(packet_id="packet-current", rgb=rgb, depth=depth))

    with pytest.raises(ObservationPacketResolutionError) as raised:
        memory.resolve_observation_packet("packet-missing")

    assert raised.value.code == "unknown_source_packet_id"
    assert raised.value.details["active_agent_session_id"] == "packet-session"
    assert raised.value.details["recent_source_packets"] == [
        {
            "source_packet_id": "packet-current",
            "observation_index": 0,
            "camera_frame_ids": ["agentview"],
        }
    ]


def test_duplicate_packet_id_with_different_contents_fails_closed(tmp_path: Path) -> None:
    first_rgb = tmp_path / "first.png"
    second_rgb = tmp_path / "second.png"
    depth = tmp_path / "depth.png"
    for path in (first_rgb, second_rgb, depth):
        path.write_bytes(b"artifact")
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.add_observation(
        _observation(packet_id="packet-reused", rgb=first_rgb, depth=depth)
    )

    with pytest.raises(ObservationPacketResolutionError) as raised:
        memory.add_observation(
            _observation(packet_id="packet-reused", rgb=second_rgb, depth=depth)
        )

    assert raised.value.code == "duplicate_source_packet_id"


def test_packet_resolution_reports_missing_local_artifact(tmp_path: Path) -> None:
    missing_rgb = tmp_path / "missing.png"
    depth = tmp_path / "depth.png"
    depth.write_bytes(b"depth")
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.add_observation(
        _observation(packet_id="packet-missing-file", rgb=missing_rgb, depth=depth)
    )

    with pytest.raises(ObservationPacketResolutionError) as raised:
        memory.resolve_observation_packet("packet-missing-file")

    assert raised.value.code == "source_artifact_missing"
    assert raised.value.details["resolved_path"] == str(missing_rgb)
