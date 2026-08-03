from __future__ import annotations

import json
from pathlib import Path

from logger.observability import EpisodeObservability


def _events(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]


def test_observations_get_unique_frames_and_collision_free_media(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    rgb.write_bytes(b"rgb")
    depth.write_bytes(b"depth")
    episode_root = tmp_path / "episode"

    writer = EpisodeObservability(
        episode_root,
        episode_id="ep-1",
        session_id="session-1",
        task="pick the cube",
        env_id="libero-task0",
        clock=lambda: 1.0,
    )
    first = writer.record_observation(
        [{"camera_id": "agentview", "rgb_path": rgb, "depth_path": depth}],
        source="reset_env",
    )
    second = writer.record_observation(
        [{"camera_id": "agentview", "rgb_path": rgb, "depth_path": depth}],
        source="render_env",
    )

    first_frame = first["frames"][0]
    second_frame = second["frames"][0]
    assert first["observation_id"] != second["observation_id"]
    assert first_frame["frame_id"] != second_frame["frame_id"]
    assert first_frame["camera_id"] == second_frame["camera_id"] == "agentview"
    assert first_frame["rgb_path"] != second_frame["rgb_path"]
    assert (episode_root / first_frame["rgb_path"]).read_bytes() == b"rgb"
    assert (episode_root / second_frame["depth_path"]).read_bytes() == b"depth"

    observation_events = [event for event in _events(episode_root) if event["kind"] == "observation"]
    assert observation_events[0]["frame_refs"]["output"] == [first_frame["frame_id"]]
    assert observation_events[1]["frame_refs"]["output"] == [second_frame["frame_id"]]


def test_action_tool_and_post_observation_refs_are_explicit(tmp_path: Path) -> None:
    writer = EpisodeObservability(tmp_path / "episode", episode_id="ep-2")
    observation = writer.record_observation([{"camera_id": "agentview"}])
    frame_id = observation["frame_ids"][0]
    action = writer.record_action(
        request={"name": "move_to", "target": "grasp"},
        input_frames=[frame_id],
        turn_id=3,
    )
    tool = writer.record_tool_start(
        tool="move_to",
        action_id=action["action_id"],
        input_frames=[frame_id],
        turn_id=3,
        parameters={"target": "grasp"},
    )
    post = writer.record_observation([{"camera_id": "agentview"}], source="post_move")
    result = writer.record_tool_result(
        tool="move_to",
        success=True,
        action_id=action["action_id"],
        tool_call_id=tool["tool_call_id"],
        input_frames=[frame_id],
        post_frames=post["frame_ids"],
        artifact_refs=["json-response-1"],
        turn_id=3,
        result={"stage": "reached"},
    )

    events = _events(tmp_path / "episode")
    assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
    action_event = next(event for event in events if event["kind"] == "action")
    result_event = next(event for event in events if event["kind"] == "tool_result")
    assert action_event["action_id"] == action["action_id"]
    assert action_event["frame_refs"]["input"] == [frame_id]
    assert result_event["tool_call_id"] == tool["tool_call_id"]
    assert result_event["frame_refs"]["post"] == post["frame_ids"]
    assert result_event["artifact_refs"] == ["json-response-1"]
    assert result["event_id"] == result_event["event_id"]


def test_finish_updates_episode_metadata_and_writes_terminal_event(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    writer = EpisodeObservability(root, episode_id="ep-3", task="pick")
    observation = writer.record_observation([{"camera_id": "agentview"}])
    writer.finish(status="success", success=True, final_frames=observation["frame_ids"])

    episode = json.loads((root / "episode.json").read_text())
    assert episode["episode_id"] == "ep-3"
    assert episode["status"] == "success"
    assert episode["success"] is True
    assert episode["final_frame_ids"] == observation["frame_ids"]
    assert _events(root)[-1]["kind"] == "episode_end"


def test_issue_is_append_only_and_does_not_change_episode_status(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    writer = EpisodeObservability(root, episode_id="ep-issue")
    observation = writer.record_observation([{"camera_id": "agentview"}])

    issue = writer.record_issue(
        category="attempt",
        component="grasp_execution",
        code="missed_grasp",
        message="Object remained on the table.",
        input_frames=observation["frame_ids"],
    )

    episode = json.loads((root / "episode.json").read_text())
    assert issue["terminal"] is False
    assert episode["status"] == "running"
    event = _events(root)[-1]
    assert event["kind"] == "issue"
    assert event["payload"]["code"] == "missed_grasp"
    assert event["frame_refs"]["input"] == observation["frame_ids"]
