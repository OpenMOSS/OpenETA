#!/usr/bin/env python3
"""Auto-bind the persistent Viser inspector to the current Operator scene.

The supervisor watches one append-only episode log.  It loads either the newest
successful grasp proposal or the newest post-move TARGET/ACTUAL comparison,
always exporting the exact observation referenced by that event.  This removes
startup timing races and stale-scene bugs during multi-attempt runs.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _latest_proposal(root: Path) -> dict[str, Any] | None:
    events = root / "events.jsonl"
    if not events.is_file():
        return None
    for line in reversed(events.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        payload = event.get("payload", {})
        result = payload.get("result", {})
        if (
            event.get("kind") != "tool_result"
            or payload.get("tool") != "propose_grasps"
            or payload.get("success") is not True
            or not isinstance(result, dict)
        ):
            continue
        observation = result.get("observation")
        detection = result.get("selected_detection")
        if not isinstance(observation, dict) or not isinstance(detection, dict):
            continue
        observation_id = observation.get("observation_id")
        mask_ref = detection.get("mask_ref")
        if not isinstance(observation_id, str) or not isinstance(mask_ref, str):
            continue
        proposal_id = result.get("proposal_id")
        result_id = result.get("result_id")
        canonical_ref = result.get("canonical_grasp_candidates_ref")
        artifact_refs = event.get("artifact_refs", [])
        canonical_artifacts = [
            value
            for value in artifact_refs
            if isinstance(value, str)
            and value.endswith("/grasp_candidates.canonical.json")
        ]
        if not isinstance(canonical_ref, str) and canonical_artifacts:
            canonical_ref = canonical_artifacts[0]
        if (
            not isinstance(proposal_id, str)
            or not proposal_id
            or not isinstance(result_id, str)
            or not result_id
            or not isinstance(canonical_ref, str)
            or not Path(canonical_ref).is_file()
        ):
            continue
        return {
            "scene_kind": "proposal",
            "scene_key": f"proposal:{observation_id}:{proposal_id}:{result_id}",
            "scene_seq": int(event.get("seq", 0)),
            "observation_id": observation_id,
            "proposal_id": proposal_id,
            "result_id": result_id,
            "mask_ref": mask_ref,
            "canonical_grasp_candidates_ref": canonical_ref,
        }
    return None


def _latest_execution(root: Path) -> dict[str, Any] | None:
    events = root / "events.jsonl"
    if not events.is_file():
        return None
    for line in reversed(events.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        event = json.loads(line)
        payload = event.get("payload", {})
        # The execution-comparison artifact is the authoritative signal that
        # this action has a renderable TARGET/ACTUAL scene.  Do not couple the
        # supervisor to gateway stage names: staged grasp execution currently
        # records names such as ``move_to_selected_pregrasp`` and
        # ``approach_selected_grasp``, while direct recovery moves use other
        # names.  Actions without a comparison artifact (for example gripper
        # open/close) are filtered below.
        if event.get("kind") != "tool_result":
            continue
        action_id = event.get("action_id")
        post_frames = event.get("frame_refs", {}).get("post", [])
        if not isinstance(action_id, str) or not post_frames:
            continue
        comparison = (
            root
            / "control"
            / "execution-comparison"
            / action_id
            / "comparison.json"
        )
        if not comparison.is_file():
            continue
        value = json.loads(comparison.read_text(encoding="utf-8"))
        observation_id = value.get("observation_id")
        if not isinstance(observation_id, str):
            continue
        return {
            "scene_kind": "execution",
            "scene_key": f"execution:{action_id}",
            "scene_seq": int(event.get("seq", 0)),
            "observation_id": observation_id,
            "action_id": action_id,
            "comparison": str(comparison),
        }
    return None


def _latest_scene(root: Path) -> dict[str, Any] | None:
    values = [value for value in (_latest_proposal(root), _latest_execution(root)) if value]
    return max(values, key=lambda value: int(value.get("scene_seq", 0))) if values else None


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


def _wait_control(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        try:
            with opener.open(
                f"http://{host}:{port}/state", timeout=0.5
            ) as response:
                value = json.load(response)
            if value.get("success"):
                return True
        except Exception:  # noqa: BLE001 - readiness polling.
            pass
        time.sleep(0.1)
    return False


def _browser_command(browser: str, url: str) -> list[str]:
    """Build an isolated visible browser command for one persistent Viser URL."""

    if Path(browser).name == "playwright":
        return [browser, "open", "-b", "chromium", url]
    return [browser, "--new-window", url]


def _launch_browser(args: argparse.Namespace) -> subprocess.Popen[str]:
    return subprocess.Popen(
        _browser_command(args.browser, f"http://{args.host}:{args.port}"),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _launch_for_proposal(
    args: argparse.Namespace,
    proposal: dict[str, str],
) -> subprocess.Popen[str]:
    observation_id = proposal["observation_id"]
    proposal_id = proposal["proposal_id"]
    result_id = proposal["result_id"]
    work_root = args.artifact_root / observation_id / proposal_id
    sample_root = work_root / "sample"
    if sample_root.exists():
        import shutil

        shutil.rmtree(sample_root)
    export_command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "embodied" / "export_libero_graspgenx_sample.py"),
        "--episode-root",
        str(args.episode_root),
        "--observation-id",
        observation_id,
        "--mask",
        proposal["mask_ref"],
        "--output-root",
        str(sample_root),
    ]
    export_result = subprocess.run(
        export_command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if export_result.returncode != 0:
        raise RuntimeError(
            f"sample export failed: {export_result.stderr or export_result.stdout}"
        )
    viewer_artifacts = work_root / "viewer"
    command = [
        str(args.viewer_python),
        str(REPO_ROOT / "scripts" / "embodied" / "grasp_pose_viewer.py"),
        "--sample-dir",
        str(sample_root / "00"),
        "--episode-root",
        str(args.episode_root),
        "--observation-id",
        observation_id,
        "--anygrasp-response",
        proposal["canonical_grasp_candidates_ref"],
        "--proposal-id",
        proposal_id,
        "--result-id",
        result_id,
        "--proposal-set-id",
        proposal_id,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--control-host",
        args.control_host,
        "--control-port",
        str(args.control_port),
        "--artifact-root",
        str(viewer_artifacts),
    ]
    log_path = work_root / "viewer.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    if args.viewer_pythonpath:
        env["PYTHONPATH"] = args.viewer_pythonpath
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    if not _wait_control(args.control_host, args.control_port, args.start_timeout):
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError(f"Viser control did not become ready; see {log_path}")
    _write_json(
        args.artifact_root / "status.json",
        {
            "status": "ready",
            "scene_kind": "proposal",
            "scene_key": proposal["scene_key"],
            "observation_id": observation_id,
            "proposal_id": proposal_id,
            "result_id": result_id,
            "mask_ref": proposal["mask_ref"],
            "canonical_grasp_candidates_ref": proposal[
                "canonical_grasp_candidates_ref"
            ],
            "viewer_pid": process.pid,
            "viewer_url": f"http://{args.host}:{args.port}",
            "control_url": f"http://{args.control_host}:{args.control_port}",
            "sample_root": str(sample_root),
            "viewer_artifacts": str(viewer_artifacts),
        },
    )
    return process


def _launch_for_execution(
    args: argparse.Namespace,
    scene: dict[str, Any],
) -> subprocess.Popen[str]:
    observation_id = str(scene["observation_id"])
    action_id = str(scene["action_id"])
    comparison = Path(str(scene["comparison"]))
    work_root = args.artifact_root / f"execution-{action_id}"
    sample_root = work_root / "sample"
    if sample_root.exists():
        import shutil

        shutil.rmtree(sample_root)
    export_result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "embodied" / "export_execution_viser_sample.py"),
            "--episode-root",
            str(args.episode_root),
            "--comparison",
            str(comparison),
            "--output-root",
            str(sample_root),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if export_result.returncode != 0:
        raise RuntimeError(
            f"execution sample export failed: {export_result.stderr or export_result.stdout}"
        )
    viewer_artifacts = work_root / "viewer"
    command = [
        str(args.viewer_python),
        str(REPO_ROOT / "scripts" / "embodied" / "grasp_pose_viewer.py"),
        "--sample-dir",
        str(sample_root / "00"),
        "--episode-root",
        str(args.episode_root),
        "--observation-id",
        observation_id,
        "--execution-comparison",
        str(comparison),
        "--proposal-set-id",
        f"exec-{args.episode_root.name}-{action_id}",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--control-host",
        args.control_host,
        "--control-port",
        str(args.control_port),
        "--artifact-root",
        str(viewer_artifacts),
    ]
    log_path = work_root / "viewer.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    if args.viewer_pythonpath:
        env["PYTHONPATH"] = args.viewer_pythonpath
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    if not _wait_control(args.control_host, args.control_port, args.start_timeout):
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError(f"Execution Viser did not become ready; see {log_path}")
    _write_json(
        args.artifact_root / "status.json",
        {
            "status": "ready",
            "scene_kind": "execution",
            "scene_key": scene["scene_key"],
            "observation_id": observation_id,
            "action_id": action_id,
            "comparison": str(comparison),
            "viewer_pid": process.pid,
            "viewer_url": f"http://{args.host}:{args.port}",
            "control_url": f"http://{args.control_host}:{args.control_port}",
            "sample_root": str(sample_root),
            "viewer_artifacts": str(viewer_artifacts),
        },
    )
    return process


def _stop_child(child: subprocess.Popen[str] | None) -> None:
    if child is None or child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--viewer-python",
        type=Path,
        default=Path("python3"),
    )
    parser.add_argument(
        "--viewer-pythonpath",
        default="",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8082)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--start-timeout", type=float, default=15.0)
    parser.add_argument("--browser", default="playwright")
    parser.add_argument("--open-browser", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.episode_root = args.episode_root.expanduser().resolve()
    args.artifact_root = args.artifact_root.expanduser().resolve()
    args.artifact_root.mkdir(parents=True, exist_ok=True)

    stop = False
    child: subprocess.Popen[str] | None = None
    browser_child: subprocess.Popen[str] | None = None
    active_scene_key: str | None = None

    def handle_signal(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    _write_json(args.artifact_root / "status.json", {"status": "waiting_for_proposals"})
    try:
        while not stop:
            if child is not None and child.poll() is not None:
                _write_json(
                    args.artifact_root / "status.json",
                    {"status": "viewer_exited", "returncode": child.returncode},
                )
                return 1
            scene = _latest_scene(args.episode_root)
            if scene is None:
                time.sleep(args.poll_interval)
                continue
            observation_id = str(scene["observation_id"])
            scene_key = str(scene["scene_key"])
            if scene_key == active_scene_key:
                time.sleep(args.poll_interval)
                continue
            if child is not None:
                _write_json(
                    args.artifact_root / "status.json",
                    {
                        "status": "reloading",
                        "previous_scene_key": active_scene_key,
                        "scene_key": scene_key,
                        "observation_id": observation_id,
                    },
                )
                _stop_child(child)
                child = None
                deadline = time.monotonic() + 5.0
                while (
                    _port_open(args.control_host, args.control_port)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.05)
            if _port_open(args.control_host, args.control_port):
                _write_json(
                    args.artifact_root / "status.json",
                    {
                        "status": "blocked_port_in_use",
                        "control_port": args.control_port,
                        "observation_id": observation_id,
                    },
                )
                time.sleep(args.poll_interval)
                continue
            try:
                if scene["scene_kind"] == "execution":
                    child = _launch_for_execution(
                        args,
                        scene,
                    )
                else:
                    child = _launch_for_proposal(
                        args,
                        scene,
                    )
                if args.open_browser and (
                    browser_child is None or browser_child.poll() is not None
                ):
                    browser_child = _launch_browser(args)
                active_scene_key = scene_key
            except Exception as exc:  # noqa: BLE001 - preserve launch failure.
                _write_json(
                    args.artifact_root / "status.json",
                    {
                        "status": "launch_failed",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "scene": scene,
                    },
                )
                return 1
            time.sleep(args.poll_interval)
    finally:
        _stop_child(child)
        _stop_child(browser_child)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
