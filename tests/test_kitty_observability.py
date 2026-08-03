from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from scripts.embodied.episode_web_dashboard import Handler, ThreadingHTTPServer
from scripts.embodied.kitty_logger_view import format_event, print_snapshot, read_status
from scripts.embodied.kitty_simulator_view import _image_key


def test_logger_view_formats_observation_media_and_terminal_state(tmp_path: Path, capsys) -> None:
    root = tmp_path / "episode"
    root.mkdir()
    events = [
        {
            "kind": "observation",
            "seq": 2,
            "payload": {
                "observation_id": "obs-1",
                "source": "reset",
                "sim_step": 0,
                "frames": [
                    {
                        "camera_id": "agentview",
                        "rgb_path": "media/frames/frame.rgb.png",
                        "depth_path": "media/frames/frame.depth.png",
                    }
                ],
            },
        },
        {"kind": "episode_end", "seq": 3, "payload": {"status": "stopped", "success": False}},
    ]
    (root / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n")
    (root / "current.json").write_text(json.dumps({"status": "stopped", "sim_step": 1}))

    assert "agentview: rgb=media/frames/frame.rgb.png" in format_event(root, events[0])
    assert print_snapshot(root) == 2
    output = capsys.readouterr().out
    assert "episode_end status=stopped success=False" in output
    assert "kitten icat" in output


def test_logger_view_uses_episode_manifest_for_completed_status(tmp_path: Path, capsys) -> None:
    root = tmp_path / "episode"
    root.mkdir()
    (root / "events.jsonl").write_text(
        json.dumps({"kind": "episode_end", "seq": 1, "payload": {"status": "completed", "success": True}}) + "\n"
    )
    (root / "episode.json").write_text(json.dumps({"status": "completed", "success": True}))
    (root / "current.json").write_text(json.dumps({"sim_step": 4}))

    assert read_status(root) == "completed"
    assert print_snapshot(root) == 1
    assert "status=completed" in capsys.readouterr().out


def test_logger_view_surfaces_failure_case_and_stops_on_failed_status(tmp_path: Path, capsys) -> None:
    root = tmp_path / "episode"
    root.mkdir()
    failure = {
        "kind": "failure_case",
        "seq": 2,
        "payload": {
            "failure_case_id": "failure-000001",
            "component": "sam3",
            "code": "service_unavailable",
            "message": "SAM3 segmentation failed.",
        },
    }
    (root / "events.jsonl").write_text(
        json.dumps(failure) + "\n"
    )
    (root / "episode.json").write_text(json.dumps({"status": "failed", "success": False}))
    (root / "current.json").write_text(json.dumps({"status": "failed", "sim_step": 1}))

    assert "FAILURE case=failure-000001 component=sam3 code=service_unavailable" in format_event(root, failure)
    assert print_snapshot(root) == 1
    assert "FAILURE" in capsys.readouterr().out


def test_kitty_session_has_simulator_logger_and_operator_surfaces() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    session = (repo_root / "scripts/embodied/kitty-session.conf").read_text()
    launches = [line for line in session.splitlines() if line.startswith("launch ")]
    assert len(launches) == 4
    assert "LIBERO simulator view" in launches[0]
    assert "episode logger" in launches[1]
    assert "Operator MCP context mirror" in launches[2]
    assert "Operator Codex" in launches[3]
    for name in (
        "simulator_pane.sh",
        "logger_pane.sh",
        "operator_context_pane.sh",
        "operator_pane.sh",
    ):
        assert (repo_root / "scripts/embodied" / name).stat().st_mode & 0o111


def test_simulator_pane_creates_its_artifact_root_before_log_redirect() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts/embodied/simulator_pane.sh").read_text()
    assert 'mkdir -p "$ROOT"' in script
    assert script.index('mkdir -p "$ROOT"') < script.index('SERVER_LOG="$ROOT/simulator.log"')


def test_operator_refuses_to_start_before_simulator_mcp_handshake() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts/embodied/operator_codex.sh").read_text()

    assert "probe_simulator_mcp()" in script
    assert "await session.initialize()" in script
    assert 'required = {"create_env", "observe_env", "move_to"}' in script
    assert "OPENETA_SIM_READY_TIMEOUT_S:-90" in script
    assert "Refusing to start Operator: simulator MCP is not ready" in script
    assert script.index("probe_simulator_mcp()") < script.index(
        "# Codex requires a trusted working root."
    )


def test_operator_runtime_paths_are_namespaced_by_full_episode_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    operator = (repo_root / "scripts/embodied/operator_codex.sh").read_text()
    chrome = (
        repo_root / "scripts/embodied/launch_operator_chrome.sh"
    ).read_text()
    kitty = (repo_root / "scripts/launch_embodied_kitty.sh").read_text()

    assert 'printf \'%s\' "$EPISODE_ROOT" | sha256sum | cut -c1-12' in operator
    assert (
        "openeta-codex-home-$EPISODE_NAME-$EPISODE_ROOT_ID"
        in operator
    )
    for launcher in (chrome, kitty):
        assert 'printf \'%s\' "$ROOT" | sha256sum | cut -c1-12' in launcher
        assert (
            "openeta-operator-workspace-$(basename \"$ROOT\")-$EPISODE_ROOT_ID"
            in launcher
        )


def test_operator_owned_runtime_paths_are_removed_without_touching_explicit_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    operator = (repo_root / "scripts/embodied/operator_codex.sh").read_text()
    chrome = (
        repo_root / "scripts/embodied/launch_operator_chrome.sh"
    ).read_text()
    kitty = (repo_root / "scripts/launch_embodied_kitty.sh").read_text()

    assert 'OPERATOR_CODEX_HOME_OWNED="${OPENETA_OPERATOR_CODEX_HOME_OWNED:-0}"' in operator
    assert "OPERATOR_CODEX_HOME_OWNED=1" in operator
    assert "cleanup_operator_codex_home()" in operator
    assert '"${XDG_RUNTIME_DIR:-/tmp}"/openeta-codex-home-*' in operator
    assert "cleanup_operator_codex_home" in operator
    assert "exec \"$CODEX_BIN\"" not in operator

    for launcher in (chrome, kitty):
        assert 'OPENETA_OPERATOR_ROOT_OWNED="${OPENETA_OPERATOR_ROOT_OWNED:-0}"' in launcher
        assert "OPENETA_OPERATOR_ROOT_OWNED=1" in launcher
        assert '"${XDG_RUNTIME_DIR:-/tmp}"/openeta-operator-workspace-*' in launcher
        assert 'rm -rf -- "$OPENETA_OPERATOR_ROOT"' in launcher


def test_launcher_preserves_explicit_libero_environment_for_general_tasks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts/launch_embodied_kitty.sh").read_text()

    assert (
        'OPENETA_LIBERO_ENV_ID="${OPENETA_LIBERO_ENV_ID:-openeta/libero_libero_spatial_task0-v0}"'
        in script
    )
    assert 'OPENETA_OPERATOR_IMAGE_WIDTH="${OPENETA_OPERATOR_IMAGE_WIDTH:-512}"' in script
    assert 'OPENETA_OPERATOR_IMAGE_HEIGHT="${OPENETA_OPERATOR_IMAGE_HEIGHT:-512}"' in script
    assert (
        'OPENETA_LIBERO_MAX_EPISODE_STEPS="${OPENETA_LIBERO_MAX_EPISODE_STEPS:-5000}"'
        in script
    )
    assert "unset TMUX TMUX_PANE" in script
    assert 'openeta_stop_episode_execution_services "$ROOT"' in script
    assert "replay_dashboard.sh\" start" in script
    assert "OPENETA_KITTY_DETACH=1 is deprecated" in script
    assert "launch_operator_chrome.sh" in script

    manual = (
        repo_root / "scripts/embodied/launch_manual_chrome.sh"
    ).read_text()
    assert 'openeta_stop_episode_execution_services "$ROOT"' in manual
    assert "replay_dashboard.sh\" start" in manual

    chrome = (
        repo_root / "scripts/embodied/launch_operator_chrome.sh"
    ).read_text()
    assert "Chrome-first embodied Operator launcher" in chrome
    assert 'OPENETA_OPERATOR_POINTCLOUD_MODE:-live-multiview-consensus' in chrome
    assert (
        'OPENETA_OPERATOR_TASK="${OPENETA_OPERATOR_TASK:-pick up the black bowl '
        'between the plate and the ramekin and place it on the plate}"'
        in chrome
    )
    assert 'printf \'  task=%s\\n\' "$OPENETA_OPERATOR_TASK"' in chrome
    assert 'openeta_stop_episode_execution_services "$ROOT"' in chrome
    assert "replay_dashboard.sh\" start" in chrome
    assert "operator_pane.sh" in chrome
    assert "kitty " not in chrome


def test_openeta_light_tui_runs_the_codex_cli_in_the_foreground() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    launcher = (
        repo_root / "scripts/embodied/launch_openeta_light_tui.sh"
    )
    script = launcher.read_text()
    chrome = (
        repo_root / "scripts/embodied/launch_operator_chrome.sh"
    ).read_text()

    assert launcher.stat().st_mode & 0o111
    assert "OPENETA_OPERATOR_RUNTIME=cli" in script
    assert "OPENETA_CODEX_MODE=interactive" in script
    assert "OPENETA_OPERATOR_FOREGROUND=1" in script
    assert 'exec "$REPO_ROOT/scripts/embodied/launch_operator_chrome.sh"' in script
    assert 'OPENETA_OPERATOR_FOREGROUND:-0' in chrome
    assert 'bash "$REPO_ROOT/scripts/embodied/operator_pane.sh"' in chrome
    foreground = chrome.index('OPENETA_OPERATOR_FOREGROUND:-0')
    detached = chrome.index('setsid bash "$REPO_ROOT/scripts/embodied/operator_pane.sh"')
    assert foreground < detached


def test_kitty_image_viewers_suppress_terminal_transmission_errors() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    simulator = (repo_root / "scripts/embodied/kitty_simulator_view.py").read_text()
    context = (repo_root / "scripts/embodied/kitty_operator_context_view.py").read_text()
    assert "stderr=subprocess.DEVNULL" in simulator
    assert "stderr=subprocess.DEVNULL" in context
    assert "Clear the old text/image region" in simulator
    assert "image transmission failed; artifact=" in simulator


def test_operator_gateway_receives_high_resolution_rgbd_defaults() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts/embodied/operator_codex.sh").read_text()

    assert 'IMAGE_WIDTH="${OPENETA_OPERATOR_IMAGE_WIDTH:-512}"' in script
    assert 'IMAGE_HEIGHT="${OPENETA_OPERATOR_IMAGE_HEIGHT:-512}"' in script
    assert '--image-width "$IMAGE_WIDTH"' in script
    assert '--image-height "$IMAGE_HEIGHT"' in script
    assert 'OPERATOR_RUNTIME="${OPENETA_OPERATOR_RUNTIME:-app-server}"' in script
    assert 'operator_app_server.py' in script
    assert 'GATEWAY_PORT="${OPENETA_OPERATOR_MCP_PORT:-8780}"' in script
    assert 'persistent_gateway_args[7]="streamable-http"' in script
    assert 'gateway_url="http://${GATEWAY_HOST}:${GATEWAY_PORT}/mcp"' in script
    assert 'url = %s\\nrequired = true\\ntool_timeout_sec = 600' in script
    assert 'Persistent Operator Gateway exited during startup' in script
    assert '^(200|400|405|406)$' in script
    assert 'generic 404' in script
    assert 'command = %s\\nargs = %s\\nrequired = true' not in script
    assert "enable_request_compression = false" in script
    assert "-c 'features.enable_request_compression=false'" in script


def test_operator_defaults_to_release_profile_and_medium_terra() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts/embodied/operator_codex.sh").read_text()

    assert 'OPERATOR_MODEL="${OPENETA_OPERATOR_MODEL:-gpt-5.6-terra}"' in script
    assert (
        'OPERATOR_CONTEXT_PROFILE="${OPENETA_OPERATOR_CONTEXT_PROFILE:-openeta-light}"'
        in script
    )
    assert (
        'TASK="${OPENETA_OPERATOR_TASK:-pick up the black bowl between the plate '
        'and the ramekin and place it on the plate}"'
        in script
    )


def test_release_operator_prompt_is_compact_and_evidence_driven() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = (repo_root / "scripts/embodied/operator_codex.sh").read_text()
    prompt = (
        repo_root
        / "configs/embodied/operator-context/components/prompts/"
        "operator-kiss-v12-compact-cross-tool.md"
    ).read_text()
    prompt = " ".join(prompt.split())

    assert "Every image named in a result's views list is markable" in prompt
    assert "Marks are immutable world coordinates" in prompt
    assert "A solved mark is visible geometry" in prompt
    assert "move_to controls the Panda grip-site" in prompt
    assert "motion=not_reached means the requested endpoint was not achieved" in prompt
    assert "Endpoint arrival alone does not establish contact" in prompt
    assert "container" not in prompt.lower()
    assert "must use" not in prompt.lower()
    assert script.count('if [[ -z "$OPERATOR_PROMPT" ]]') == 1


def test_simulator_view_redraw_key_changes_when_pose_artifact_is_replaced(tmp_path: Path) -> None:
    image = tmp_path / "pose.png"
    image.write_bytes(b"first")
    first = _image_key("grasp_candidates", image)
    image.write_bytes(b"second-render")
    second = _image_key("grasp_candidates", image)
    assert first != second


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_port(port: int) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise AssertionError(f"port {port} did not become ready")


def test_lifecycle_refuses_to_kill_unrelated_port_owner(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    port = _unused_port()
    process = subprocess.Popen(
        ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_for_port(port)
        command = (
            f"source {repo_root}/scripts/embodied/service_lifecycle.sh; "
            f"openeta_assert_or_reclaim_port {tmp_path}/episode {port} test"
        )
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
        assert result.returncode == 2
        assert "occupied by unrelated PID" in result.stderr
        assert process.poll() is None
    finally:
        os.killpg(process.pid, 15)
        process.wait(timeout=5)


def test_lifecycle_stops_registered_dashboard_process_group(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    port = _unused_port()
    process = subprocess.Popen(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/embodied/episode_web_dashboard.py"),
            "--root",
            str(episode_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_for_port(port)
        command = (
            f"source {repo_root}/scripts/embodied/service_lifecycle.sh; "
            f"openeta_register_service {episode_root} dashboard {process.pid} {port}; "
            f"openeta_stop_episode_services {episode_root}"
        )
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        process.wait(timeout=5)
        assert not (episode_root / ".service-launch-claim").exists()
        assert (episode_root / "services.tsv").read_text() == ""
    finally:
        if process.poll() is None:
            os.killpg(process.pid, 15)
            process.wait(timeout=5)


def test_lifecycle_can_stop_execution_and_retain_dashboard_replay(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    port = _unused_port()
    dashboard = subprocess.Popen(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/embodied/episode_web_dashboard.py"),
            "--root",
            str(episode_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    execution = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                "exec -a 'scripts/embodied/operator_app_server.py "
                f"--episode-root {episode_root}' sleep 60"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_for_port(port)
        command = (
            f"source {repo_root}/scripts/embodied/service_lifecycle.sh; "
            f"openeta_register_service {episode_root} dashboard {dashboard.pid} {port}; "
            f"openeta_register_service {episode_root} operator-app-server {execution.pid} -; "
            f"openeta_stop_episode_execution_services {episode_root}"
        )
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        execution.wait(timeout=5)
        assert dashboard.poll() is None
        registry = (episode_root / "services.tsv").read_text()
        assert "\tdashboard\t" not in registry
        assert registry.startswith("dashboard\t")
        assert "operator-app-server" not in registry
    finally:
        command = (
            f"source {repo_root}/scripts/embodied/service_lifecycle.sh; "
            f"openeta_stop_episode_services {episode_root}"
        )
        subprocess.run(["bash", "-c", command], check=False)
        if dashboard.poll() is None:
            os.killpg(dashboard.pid, 15)
            dashboard.wait(timeout=5)
        if execution.poll() is None:
            os.killpg(execution.pid, 15)
            execution.wait(timeout=5)


def test_lifecycle_can_stop_observability_and_retain_execution(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    episode_root = tmp_path / "episode"
    episode_root.mkdir()
    port = _unused_port()
    dashboard = subprocess.Popen(
        [
            str(repo_root / ".venv/bin/python"),
            str(repo_root / "scripts/embodied/episode_web_dashboard.py"),
            "--root",
            str(episode_root),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=repo_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    execution = subprocess.Popen(
        [
            "bash",
            "-c",
            (
                "exec -a 'scripts/embodied/operator_app_server.py "
                f"--episode-root {episode_root}' sleep 60"
            ),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        _wait_for_port(port)
        command = (
            f"source {repo_root}/scripts/embodied/service_lifecycle.sh; "
            f"openeta_register_service {episode_root} dashboard {dashboard.pid} {port}; "
            f"openeta_register_service {episode_root} operator-app-server {execution.pid} -; "
            f"openeta_stop_episode_observability_services {episode_root}"
        )
        result = subprocess.run(["bash", "-c", command], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        dashboard.wait(timeout=5)
        assert execution.poll() is None
        registry = (episode_root / "services.tsv").read_text()
        assert "dashboard" not in registry
        assert "operator-app-server" in registry
    finally:
        command = (
            f"source {repo_root}/scripts/embodied/service_lifecycle.sh; "
            f"openeta_stop_episode_services {episode_root}"
        )
        subprocess.run(["bash", "-c", command], check=False)
        for process in (dashboard, execution):
            if process.poll() is None:
                os.killpg(process.pid, 15)
                process.wait(timeout=5)


def test_dashboard_operator_trace_is_exact_model_visible_context(tmp_path: Path) -> None:
    import urllib.request

    root = tmp_path / "episode"
    root.mkdir()
    ray_images = [
        root / "pointcloud_views/obs-1/pointcloud_top.pending.png",
        root / "pointcloud_views/obs-1/pointcloud_front.pending.png",
        root / "pointcloud_views/obs-1/pointcloud_side.pending.png",
    ]
    for image in ray_images:
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"png")
    row = {
        "seq": 2,
        "tool": "mark_point",
        "arguments": {"point_id": "P0", "view": "pointcloud_top", "x": 120, "y": 140},
        "response_text_blocks": [
            json.dumps({"status": "pending", "observation_id": "obs-1"})
        ],
        "response_image_paths": [str(path) for path in ray_images],
    }
    (root / "operator_context.jsonl").write_text(json.dumps(row) + "\n")
    contract = {"prompt": "exact operator prompt"}
    (root / "operator_context_contract.json").write_text(json.dumps(contract))

    Handler.root = root
    Handler.viser_url = "#"
    Handler.control_url = ""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/api/operator_trace") as response:
            payload = json.load(response)
        assert payload["row_count"] == 1
        assert payload["rows"] == [row]
        with urllib.request.urlopen(base + "/api/operator_contract") as response:
            assert json.load(response) == contract
        with urllib.request.urlopen(
            base + "/artifact/" + str(ray_images[1])
        ) as response:
            assert response.read() == b"png"
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_displays_versioned_context_identity() -> None:
    from scripts.embodied.episode_web_dashboard import HTML

    assert "contextBadge" in HTML
    assert "resolved_context_sha256" in HTML
    assert "profile.label" in HTML


def test_replay_hub_routes_nested_episode_without_per_episode_server(
    tmp_path: Path,
) -> None:
    import urllib.request

    from scripts.embodied.replay_hub import ReplayHubHandler

    episodes = tmp_path / "episodes"
    root = episodes / "ab" / "seed-017"
    root.mkdir(parents=True)
    (root / "episode.json").write_text(
        json.dumps({"status": "completed", "success": True, "seed": 17})
    )
    (root / "current.json").write_text(
        json.dumps({"status": "completed", "sim_step": 42})
    )
    artifact = root / "media" / "proof.txt"
    artifact.parent.mkdir()
    artifact.write_text("durable proof")
    ReplayHubHandler.episodes_root = episodes
    server = ThreadingHTTPServer(("127.0.0.1", 0), ReplayHubHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/") as response:
            index = response.read().decode()
        assert "ab/seed-017" in index
        with urllib.request.urlopen(
            base + "/episodes/ab/seed-017/"
        ) as response:
            page = response.read().decode()
        assert 'const replayBase="/episodes/ab/seed-017"' in page
        with urllib.request.urlopen(
            base + "/episodes/ab/seed-017/api/current"
        ) as response:
            assert json.load(response)["sim_step"] == 42
        with urllib.request.urlopen(
            base + "/episodes/ab/seed-017/artifact/media/proof.txt"
        ) as response:
            assert response.read() == b"durable proof"
    finally:
        server.shutdown()
        server.server_close()


def test_replay_hub_terminal_sweep_respects_ttl_and_pin(tmp_path: Path) -> None:
    from scripts.embodied.replay_hub import sweep_terminal_replays

    episodes = tmp_path / "episodes"
    expired = episodes / "expired"
    pinned = episodes / "pinned"
    fresh = episodes / "fresh"
    for root, finished in ((expired, 100.0), (pinned, 100.0), (fresh, 950.0)):
        root.mkdir(parents=True)
        (root / "episode.json").write_text(
            json.dumps({"status": "completed", "finished_at_s": finished})
        )
        (root / "services.tsv").write_text("dashboard\t999999\t-\t-\t9000\tcmd\n")
    (pinned / ".replay-pin").touch()

    selected = sweep_terminal_replays(
        episodes,
        ttl_seconds=100.0,
        now=1000.0,
        dry_run=True,
    )

    assert selected == [expired.resolve()]


def test_replay_hub_sweeps_stale_running_orphan_but_not_live_run(
    tmp_path: Path,
) -> None:
    from scripts.embodied import replay_hub

    episodes = tmp_path / "episodes"
    orphan = episodes / "orphan"
    live = episodes / "live"
    incomplete = episodes / "incomplete"
    for root in (orphan, live):
        root.mkdir(parents=True)
        (root / "episode.json").write_text(json.dumps({"status": "running"}))
        (root / "current.json").write_text(json.dumps({"status": "running"}))
        (root / "services.tsv").write_text(
            "dashboard\t999999\t-\t-\t9000\tcmd\n"
        )
        os.utime(root / "episode.json", (100.0, 100.0))
        os.utime(root / "current.json", (100.0, 100.0))
    incomplete.mkdir(parents=True)
    (incomplete / "services.tsv").write_text(
        "dashboard\t999999\t-\t-\t9002\tcmd\n"
    )
    os.utime(incomplete, (100.0, 100.0))
    with (live / "services.tsv").open("a") as stream:
        stream.write(f"gateway\t{os.getpid()}\t-\t-\t9001\tcmd\n")

    selected = replay_hub.sweep_terminal_replays(
        episodes,
        ttl_seconds=100.0,
        orphan_ttl_seconds=500.0,
        now=1000.0,
        dry_run=True,
    )

    assert set(selected) == {incomplete.resolve(), orphan.resolve()}


def test_terminal_dashboard_is_read_only_replay(tmp_path: Path) -> None:
    import urllib.error
    import urllib.request

    root = tmp_path / "episode"
    root.mkdir()
    (root / "current.json").write_text(
        json.dumps({"status": "completed", "sim_step": 42})
    )
    Handler.root = root
    Handler.viser_url = "#"
    Handler.control_url = "http://127.0.0.1:1"
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/api/control_status") as response:
            status = json.load(response)
        assert status == {
            "status": "completed",
            "terminal": True,
            "control_available": False,
            "mode": "replay",
        }
        request = urllib.request.Request(
            base + "/api/manual/observe",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
        assert payload["success"] is False
        assert payload["reason"] == "episode_read_only"
    finally:
        server.shutdown()
        server.server_close()
