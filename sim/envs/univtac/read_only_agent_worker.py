"""One-shot Pull Out Key simulator-to-read-only-Agent integration worker."""

from __future__ import annotations

import base64
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from PIL import Image

from agent.backends.readonly_observation import LOCAL_DUMMY_API_KEY, SYSTEM_PROMPT
from agent.runtime.observation_interfaces import (
    EXPECTED_IMAGE_LABELS,
    READONLY_REPORT_SCHEMA,
    find_absolute_posix_paths,
)
from sim.envs.univtac.agent_context import (
    load_validated_pre_action_snapshot,
    materialize_operator_visible_input,
    project_operator_visible_context,
    validate_agent_input_directory,
)
from sim.envs.univtac.trace import write_json

SUCCESS_CLASSIFICATION = "openeta_readonly_pull_out_key_multimodal_handoff_passed"
SIMULATOR_SUCCESS = "scoped_launcher_and_pull_out_key_seed1000000_gate_passed"
READONLY_RESPONSE = {
    "schema_version": READONLY_REPORT_SCHEMA,
    "mode": "read_only",
    "context_received": True,
    "received_image_labels": list(EXPECTED_IMAGE_LABELS),
    "observation_notes": ["multimodal context transport validated"],
    "uncertainties": ["semantic tactile interpretation was not evaluated"],
    "execution_requested": False,
}
_FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "parallel_tool_calls",
        "host_only",
        "actor_state",
        "task_metadata",
        "press_depth",
        "depth",
        "native_marker_motion",
        "tactile_pose",
        "actor_pose",
        "target_pose",
        "slot_init_pose",
        "cid_present",
        "plan_success",
        "planner",
        "check_success",
        "reward",
        "eval_success",
    }
)


class ReadOnlyWorkerError(RuntimeError):
    """Raised when a read-only integration gate contract fails."""


@dataclass
class CaptureState:
    output_dir: Path
    request_count: int = 0
    request_started_at: str | None = None
    validation_error: dict[str, str] | None = None
    decoded_images: list[dict[str, Any]] = field(default_factory=list)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_readonly_gate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "task": "pull_out_key",
        "seed": 1_000_000,
        "agent_mode": "read_only",
        "backend_mode": "localhost_capture",
        "expected_image_count": 4,
        "simulator_invocation_limit": 1,
        "agent_request_limit": 1,
        "allow_agent_action": False,
        "allow_tools": False,
        "allow_external_network": False,
    }
    missing = sorted(({"simulator_gate_config"} | set(expected)) - config.keys())
    if missing:
        raise ReadOnlyWorkerError(f"read-only gate config is missing: {missing}")
    for key, value in expected.items():
        if config[key] != value:
            raise ReadOnlyWorkerError(f"read-only gate requires {key}={value!r}")
    return dict(config)


def _walk_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_walk_keys(child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            keys.update(_walk_keys(child))
    return keys


def _validate_capture_payload(
    payload: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(payload) != {"model", "temperature", "messages"}:
        raise ReadOnlyWorkerError("request top-level fields are not read-only")
    if payload["temperature"] != 0:
        raise ReadOnlyWorkerError("read-only capture requires temperature=0")
    messages = payload["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ReadOnlyWorkerError("request must contain exactly system and user messages")
    if messages[0] != {"role": "system", "content": SYSTEM_PROMPT}:
        raise ReadOnlyWorkerError("system message does not match read-only policy")
    if messages[1].get("role") != "user":
        raise ReadOnlyWorkerError("second message must be the user observation")
    content = messages[1].get("content")
    if not isinstance(content, list) or len(content) != 5:
        raise ReadOnlyWorkerError("user content must contain one text and four images")
    if set(content[0]) != {"type", "text"} or content[0]["type"] != "text":
        raise ReadOnlyWorkerError("first user content item must be text")
    text_payload = json.loads(content[0]["text"])
    if set(text_payload) != {
        "task_instruction",
        "step_identifiers",
        "proprio",
        "visual_descriptors",
    }:
        raise ReadOnlyWorkerError("Agent text context has fields outside the allowlist")
    forbidden = _walk_keys(text_payload) & _FORBIDDEN_REQUEST_KEYS
    if forbidden:
        raise ReadOnlyWorkerError(f"privileged Agent text keys detected: {sorted(forbidden)}")
    text = content[0]["text"]
    if find_absolute_posix_paths(text_payload) or any(
        token in text for token in ("outputs/", "snapshot_pre.json")
    ):
        raise ReadOnlyWorkerError("local path leaked into Agent text")
    descriptors = text_payload["visual_descriptors"]
    labels = tuple(item["label"] for item in descriptors)
    if labels != EXPECTED_IMAGE_LABELS:
        raise ReadOnlyWorkerError("visual descriptor labels are incomplete or reordered")
    decoded: list[dict[str, Any]] = []
    expected_sizes = ((480, 270), (480, 270), (320, 240), (320, 240))
    for index, (part, label, expected_size) in enumerate(
        zip(content[1:], EXPECTED_IMAGE_LABELS, expected_sizes, strict=True), start=1
    ):
        if set(part) != {"type", "image_url"} or part["type"] != "image_url":
            raise ReadOnlyWorkerError(f"user content item {index} must be image_url")
        image_url = part["image_url"]
        if set(image_url) != {"url"}:
            raise ReadOnlyWorkerError("image_url content contains unexpected fields")
        prefix = "data:image/png;base64,"
        if not image_url["url"].startswith(prefix):
            raise ReadOnlyWorkerError("image content must use image/png data URI")
        encoded = image_url["url"][len(prefix) :]
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(image_bytes)) as image:
                image.load()
                mode = image.mode
                size = image.size
        except Exception as exc:
            raise ReadOnlyWorkerError(f"cannot decode image part {label}: {exc}") from exc
        if mode != "RGB" or size != expected_size:
            raise ReadOnlyWorkerError(
                f"decoded {label} expected RGB {expected_size}, got {mode} {size}"
            )
        decoded.append(
            {
                "label": label,
                "mode": mode,
                "width": size[0],
                "height": size[1],
                "channels": 3,
                "dtype": "uint8",
                "decoded_byte_count": len(image_bytes),
                "base64_character_count": len(encoded),
            }
        )
    return decoded, text_payload


def _capture_handler(state: CaptureState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: Any) -> None:
            return

        def do_POST(self) -> None:
            state.request_count += 1
            state.request_started_at = _utc_now()
            try:
                if state.request_count != 1:
                    raise ReadOnlyWorkerError("capture endpoint accepts exactly one request")
                if self.path != "/v1/chat/completions":
                    raise ReadOnlyWorkerError(f"unexpected request path: {self.path}")
                if self.headers.get("Authorization") != f"Bearer {LOCAL_DUMMY_API_KEY}":
                    raise ReadOnlyWorkerError("capture request did not use the dummy key")
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ReadOnlyWorkerError("capture request must be a JSON object")
                decoded, text_payload = _validate_capture_payload(payload)
                state.decoded_images = decoded
                redacted = json.loads(json.dumps(payload))
                for part, image in zip(
                    redacted["messages"][1]["content"][1:], decoded, strict=True
                ):
                    part["image_url"]["url"] = (
                        f"<redacted: {image['base64_character_count']} encoded bytes>"
                    )
                write_json(state.output_dir / "request_redacted.json", redacted)
                write_json(
                    state.output_dir / "request_summary.json",
                    {
                        "request_count": state.request_count,
                        "path": self.path,
                        "message_count": len(payload["messages"]),
                        "user_content_count": len(payload["messages"][1]["content"]),
                        "text_item_count": 1,
                        "image_item_count": 4,
                        "tools_present": False,
                        "authorization_persisted": False,
                        "text_key_tree": sorted(_walk_keys(text_payload)),
                        "privileged_keys": sorted(
                            _walk_keys(text_payload) & _FORBIDDEN_REQUEST_KEYS
                        ),
                        "local_path_leak": False,
                    },
                )
                write_json(state.output_dir / "decoded_image_summary.json", {"images": decoded})
                response_payload = {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(READONLY_RESPONSE, sort_keys=True),
                            }
                        }
                    ]
                }
                write_json(state.output_dir / "response.json", READONLY_RESPONSE)
                body = json.dumps(response_payload).encode("utf-8")
                self.send_response(200)
            except Exception as exc:  # noqa: BLE001 - persist capture validation failure
                state.validation_error = {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                write_json(state.output_dir / "capture_error.json", state.validation_error)
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _run_logged_process(
    command: list[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    started_at = _utc_now()
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=dict(environment),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    sigterm_sent = False
    sigkill_sent = False
    try:
        stdout, _ = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        sigterm_sent = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            sigkill_sent = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, _ = process.communicate(timeout=15)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(stdout, encoding="utf-8")
    lifecycle = {
        "command": command,
        "cwd": str(cwd),
        "root_pid": process.pid,
        "process_group_id": process.pid,
        "returncode": process.returncode,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "timed_out": timed_out,
        "sigterm_sent": sigterm_sent,
        "sigkill_sent": sigkill_sent,
        "cleanup_complete": process.poll() is not None,
        "stdout_path": str(stdout_path),
    }
    return lifecycle, stdout


def run_read_only_agent_worker(
    *,
    config: Mapping[str, Any],
    runtime_python: Path,
    source_root: Path,
    output_root: Path,
    repo_root: Path,
    headless: bool,
) -> dict[str, Any]:
    config = validate_readonly_gate_config(config)
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise ReadOnlyWorkerError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    lifecycle_dir = output_root / "lifecycle"
    lifecycle_dir.mkdir()
    simulator_root = output_root / "simulator"
    agent_backend_dir = output_root / "agent_backend"
    agent_backend_dir.mkdir()
    run_manifest = {
        "schema_version": "openeta.univtac.readonly_handoff_run.v1",
        "round": "R0.9.12",
        "created_at": _utc_now(),
        "simulator_invocation_limit": 1,
        "simulator_invocation_count": 0,
        "agent_request_limit": 1,
        "project_hash_gate_evaluated": False,
        "parallel_runtime_gate_evaluated": False,
        "unrelated_processes_terminated": False,
        "external_agent_network_allowed": False,
        "status": "starting",
    }
    write_json(output_root / "run_manifest.json", run_manifest)

    simulator_command = [
        sys.executable,
        str(repo_root / "scripts/univtac/run_pull_out_key_gate.py"),
        "--config",
        str((repo_root / config["simulator_gate_config"]).resolve()),
        "--runtime-python",
        str(runtime_python.resolve()),
        "--source-root",
        str(source_root.resolve()),
        "--output-root",
        str(simulator_root),
    ]
    if headless:
        simulator_command.append("--headless")
    run_manifest["simulator_invocation_count"] = 1
    write_json(output_root / "run_manifest.json", run_manifest)
    simulator_lifecycle, _ = _run_logged_process(
        simulator_command,
        cwd=repo_root,
        environment=os.environ,
        stdout_path=lifecycle_dir / "simulator_stdout.log",
        timeout_seconds=1260,
    )
    write_json(lifecycle_dir / "simulator.json", simulator_lifecycle)
    simulator_summary_path = simulator_root / "summary.json"
    if not simulator_summary_path.is_file():
        raise ReadOnlyWorkerError("readonly_worker_simulator_gate_failed: summary missing")
    simulator_summary = json.loads(simulator_summary_path.read_text(encoding="utf-8"))
    if (
        simulator_lifecycle["returncode"] != 0
        or not simulator_lifecycle["cleanup_complete"]
        or simulator_summary.get("classification") != SIMULATOR_SUCCESS
    ):
        raise ReadOnlyWorkerError("readonly_worker_simulator_gate_failed")
    simulator_exited_at = time.monotonic()
    seed_dir = simulator_root / "pull_out_key_seed1000000"
    if (seed_dir / "snapshot_post.json").exists() or (seed_dir / "transition.json").exists():
        raise ReadOnlyWorkerError("readonly_snapshot_contract_failed")
    operator_visible = load_validated_pre_action_snapshot(
        seed_dir / "snapshot_pre.json", simulator_root
    )
    projected = project_operator_visible_context(operator_visible, simulator_root)
    staged = materialize_operator_visible_input(projected, output_root / "agent_input")
    validate_agent_input_directory(output_root / "agent_input")

    state = CaptureState(agent_backend_dir)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _capture_handler(state))
    server.daemon_threads = True
    host, port = server.server_address
    if host != "127.0.0.1" or port <= 0:
        raise ReadOnlyWorkerError("capture server did not bind loopback ephemeral port")
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_started_at = _utc_now()
    server_thread.start()
    write_json(
        agent_backend_dir / "endpoint.json",
        {"host": host, "port": port, "base_url": f"http://{host}:{port}/v1"},
    )
    agent_environment = dict(os.environ)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        agent_environment.pop(name, None)
    agent_environment["NO_PROXY"] = "127.0.0.1,localhost"
    agent_environment["no_proxy"] = "127.0.0.1,localhost"
    agent_environment["OPENETA_READONLY_BASE_URL"] = f"http://{host}:{port}/v1"
    agent_environment["OPENETA_READONLY_MODEL"] = "local-readonly-capture"
    agent_environment["OPENETA_READONLY_DUMMY_KEY"] = LOCAL_DUMMY_API_KEY
    agent_command = [
        sys.executable,
        str(repo_root / "scripts/univtac/run_readonly_observation_agent.py"),
        "--context-root",
        str(output_root / "agent_input"),
    ]
    try:
        agent_request_started_at = time.monotonic()
        agent_lifecycle, agent_stdout = _run_logged_process(
            agent_command,
            cwd=repo_root,
            environment=agent_environment,
            stdout_path=lifecycle_dir / "agent_stdout.log",
            timeout_seconds=60,
        )
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=10)
    write_json(lifecycle_dir / "agent_process.json", agent_lifecycle)
    capture_lifecycle = {
        "host": host,
        "port": port,
        "started_at": server_started_at,
        "stopped_at": _utc_now(),
        "thread_alive": server_thread.is_alive(),
        "request_count": state.request_count,
        "validation_error": state.validation_error,
    }
    write_json(lifecycle_dir / "capture_server.json", capture_lifecycle)
    if agent_lifecycle["returncode"] != 0 or state.validation_error is not None:
        raise ReadOnlyWorkerError("readonly_backend_transport_failed")
    try:
        runtime_result = json.loads(agent_stdout)
    except json.JSONDecodeError as exc:
        raise ReadOnlyWorkerError("readonly_backend_response_invalid") from exc
    write_json(agent_backend_dir / "runtime_result.json", runtime_result)
    report = runtime_result["report"]
    counters = {
        "simulator_invocation_count": 1,
        "simulator_reset_count": int(
            simulator_summary["child_result"]["counters"]["reset_call_count"]
        ),
        "native_observation_count": int(
            simulator_summary["child_result"]["counters"]["observation_call_count"]
        ),
        "snapshot_pre_count": 1,
        "snapshot_post_count": 0,
        "transition_count": 0,
        "agent_process_count": 1,
        "agent_observe_call_count": runtime_result["counters"]["agent_observe_call_count"],
        "backend_http_request_count": state.request_count,
        "external_agent_network_request_count": 0,
        "agent_act_call_count": runtime_result["counters"]["agent_act_call_count"],
        "planner_decide_call_count": runtime_result["counters"]["planner_decide_call_count"],
        "tool_call_count": runtime_result["counters"]["tool_call_count"],
        "action_compile_count": runtime_result["counters"]["action_compile_count"],
        "env_action_count": runtime_result["counters"]["env_action_count"],
        "simulator_step_after_snapshot_count": 0,
    }
    write_json(agent_backend_dir / "counters.json", counters)
    expected_counters = {
        "simulator_invocation_count": 1,
        "simulator_reset_count": 1,
        "native_observation_count": 1,
        "snapshot_pre_count": 1,
        "snapshot_post_count": 0,
        "transition_count": 0,
        "agent_process_count": 1,
        "agent_observe_call_count": 1,
        "backend_http_request_count": 1,
        "external_agent_network_request_count": 0,
        "agent_act_call_count": 0,
        "planner_decide_call_count": 0,
        "tool_call_count": 0,
        "action_compile_count": 0,
        "env_action_count": 0,
        "simulator_step_after_snapshot_count": 0,
    }
    success = all(
        (
            agent_request_started_at >= simulator_exited_at,
            state.request_count == 1,
            report == READONLY_RESPONSE,
            report["execution_requested"] is False,
            counters == expected_counters,
            not any(runtime_result["module_import_state"].values()),
            not server_thread.is_alive(),
            len(staged.images) == 4,
        )
    )
    summary = {
        "schema_version": "openeta.univtac.readonly_handoff_summary.v1",
        "classification": SUCCESS_CLASSIFICATION if success else "readonly_action_path_invoked",
        "simulator_exited_before_agent_request": agent_request_started_at >= simulator_exited_at,
        "simulator_classification": simulator_summary["classification"],
        "agent_input_files": sorted(
            path.relative_to(output_root / "agent_input").as_posix()
            for path in (output_root / "agent_input").rglob("*")
            if path.is_file()
        ),
        "image_labels": list(EXPECTED_IMAGE_LABELS),
        "report": report,
        "counters": counters,
        "module_import_state": runtime_result["module_import_state"],
        "capture_server": capture_lifecycle,
        "simulator_lifecycle": simulator_lifecycle,
        "agent_lifecycle": agent_lifecycle,
        "unrelated_processes_terminated": False,
    }
    write_json(output_root / "summary.json", summary)
    run_manifest.update(
        {
            "status": "complete",
            "classification": summary["classification"],
            "ended_at": _utc_now(),
            "agent_process_count": 1,
            "agent_request_count": state.request_count,
        }
    )
    write_json(output_root / "run_manifest.json", run_manifest)
    if not success:
        raise ReadOnlyWorkerError(summary["classification"])
    return summary
