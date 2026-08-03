from __future__ import annotations

import json
from pathlib import Path

from scripts.embodied.grasp_inspector_supervisor import (
    _browser_command,
    _latest_execution,
    _latest_proposal,
    _latest_scene,
    _stop_child,
)


def test_browser_command_uses_isolated_playwright_chromium() -> None:
    assert _browser_command("playwright", "http://127.0.0.1:8081") == [
        "playwright",
        "open",
        "-b",
        "chromium",
        "http://127.0.0.1:8081",
    ]
    assert _browser_command("google-chrome", "http://127.0.0.1:8081") == [
        "google-chrome",
        "--new-window",
        "http://127.0.0.1:8081",
    ]


def _proposal_event(
    observation_id: str,
    mask_ref: str,
    canonical_artifact: Path,
    *,
    seq: int = 0,
    proposal_id: str = "proposal-result-000001",
    result_id: str = "result-000001",
) -> dict:
    return {
        "seq": seq,
        "kind": "tool_result",
        "artifact_refs": [str(canonical_artifact)],
        "payload": {
            "tool": "propose_grasps",
            "success": True,
            "result": {
                "observation": {"observation_id": observation_id},
                "selected_detection": {"mask_ref": mask_ref},
                "proposal_id": proposal_id,
                "result_id": result_id,
                "canonical_grasp_candidates_ref": str(canonical_artifact),
            },
        },
    }


def test_latest_proposal_tracks_new_observation_for_multi_attempt_reload(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "grasp_candidates.canonical.json"
    second = tmp_path / "second" / "grasp_candidates.canonical.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}")
    second.write_text("{}")
    events = [
        _proposal_event(
            "obs-000002",
            "/mask/first.png",
            first,
            seq=2,
            proposal_id="proposal-result-first",
            result_id="result-first",
        ),
        _proposal_event(
            "obs-000014",
            "/mask/second.png",
            second,
            seq=14,
            proposal_id="proposal-result-second",
            result_id="result-second",
        ),
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    proposal = _latest_proposal(tmp_path)

    assert proposal == {
        "scene_kind": "proposal",
        "scene_key": "proposal:obs-000014:proposal-result-second:result-second",
        "scene_seq": 14,
        "observation_id": "obs-000014",
        "proposal_id": "proposal-result-second",
        "result_id": "result-second",
        "mask_ref": "/mask/second.png",
        "canonical_grasp_candidates_ref": str(second),
    }


def test_latest_proposal_reloads_same_observation_for_new_identity(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "grasp_candidates.canonical.json"
    second = tmp_path / "second" / "grasp_candidates.canonical.json"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("{}")
    second.write_text("{}")
    events_path = tmp_path / "events.jsonl"
    first_event = _proposal_event(
        "obs-000010",
        "/mask/same.png",
        first,
        seq=10,
        proposal_id="proposal-result-a",
        result_id="result-a",
    )
    events_path.write_text(json.dumps(first_event) + "\n")

    initial = _latest_proposal(tmp_path)

    second_event = _proposal_event(
        "obs-000010",
        "/mask/same.png",
        second,
        seq=11,
        proposal_id="proposal-result-b",
        result_id="result-b",
    )
    events_path.write_text(
        json.dumps(first_event) + "\n" + json.dumps(second_event) + "\n"
    )
    reloaded = _latest_proposal(tmp_path)

    assert initial is not None and reloaded is not None
    assert initial["observation_id"] == reloaded["observation_id"]
    assert initial["scene_key"] == (
        "proposal:obs-000010:proposal-result-a:result-a"
    )
    assert reloaded["scene_key"] == (
        "proposal:obs-000010:proposal-result-b:result-b"
    )
    assert reloaded["canonical_grasp_candidates_ref"] == str(second)


def test_latest_execution_binds_comparison_to_post_action_observation(
    tmp_path: Path,
) -> None:
    comparison = (
        tmp_path
        / "control"
        / "execution-comparison"
        / "action-000007"
        / "comparison.json"
    )
    comparison.parent.mkdir(parents=True)
    comparison.write_text(
        json.dumps(
            {
                "action_id": "action-000007",
                "observation_id": "obs-000009",
            }
        )
    )
    event = {
        "seq": 22,
        "kind": "tool_result",
        "action_id": "action-000007",
        "frame_refs": {"post": ["frame-post-agentview"]},
        "payload": {"tool": "move_to_selected_pregrasp", "success": False},
    }
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n")

    execution = _latest_execution(tmp_path)

    assert execution == {
        "scene_kind": "execution",
        "scene_key": "execution:action-000007",
        "scene_seq": 22,
        "observation_id": "obs-000009",
        "action_id": "action-000007",
        "comparison": str(comparison),
    }


def test_latest_execution_accepts_lift_motion_scene(tmp_path: Path) -> None:
    comparison = (
        tmp_path
        / "control"
        / "execution-comparison"
        / "action-000009"
        / "comparison.json"
    )
    comparison.parent.mkdir(parents=True)
    comparison.write_text(
        json.dumps({"action_id": "action-000009", "observation_id": "obs-000013"})
    )
    event = {
        "seq": 67,
        "kind": "tool_result",
        "action_id": "action-000009",
        "frame_refs": {"post": ["frame-post-agentview"]},
        "payload": {"tool": "lift_grasp", "success": True},
    }
    (tmp_path / "events.jsonl").write_text(json.dumps(event) + "\n")

    execution = _latest_execution(tmp_path)

    assert execution is not None
    assert execution["scene_key"] == "execution:action-000009"
    assert execution["observation_id"] == "obs-000013"


def test_latest_scene_prefers_newest_event_sequence(tmp_path: Path) -> None:
    response = tmp_path / "proposal" / "grasp_candidates.canonical.json"
    response.parent.mkdir(parents=True)
    response.write_text("{}")
    comparison = (
        tmp_path
        / "control"
        / "execution-comparison"
        / "action-000001"
        / "comparison.json"
    )
    comparison.parent.mkdir(parents=True)
    comparison.write_text(
        json.dumps({"observation_id": "obs-000003"})
    )
    events = [
        {
            "seq": 8,
            "kind": "tool_result",
            "action_id": "action-000001",
            "frame_refs": {"post": ["frame-post-agentview"]},
            "payload": {"tool": "move_to_selected_grasp", "success": True},
        },
        _proposal_event(
            "obs-000004", "/mask/new.png", response, seq=12
        ),
    ]
    (tmp_path / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )

    scene = _latest_scene(tmp_path)

    assert scene is not None
    assert scene["scene_kind"] == "proposal"
    assert scene["scene_key"] == (
        "proposal:obs-000004:proposal-result-000001:result-000001"
    )


def test_stop_child_terminates_and_reaps_running_viewer() -> None:
    class FakeChild:
        def __init__(self) -> None:
            self.running = True
            self.terminated = False
            self.wait_calls: list[float] = []

        def poll(self):
            return None if self.running else 0

        def terminate(self) -> None:
            self.terminated = True
            self.running = False

        def wait(self, timeout: float):
            self.wait_calls.append(timeout)
            return 0

        def kill(self) -> None:
            self.running = False

    child = FakeChild()

    _stop_child(child)  # type: ignore[arg-type]

    assert child.terminated is True
    assert child.wait_calls == [5]
