from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from agent.backends.readonly_observation import (
    OpenAICompatibleReadOnlyObservationBackend,
)
from agent.runtime.observation_interfaces import (
    EXPECTED_IMAGE_LABELS,
    READONLY_REPORT_SCHEMA,
    ReadOnlyImageInput,
    ReadOnlyObservationContext,
    ReadOnlyObservationError,
    ReadOnlyObservationReport,
)
from agent.runtime.observation_runtime import OpenEtaReadOnlyObservationRuntime

REPO_ROOT = Path(__file__).resolve().parents[2]


def _context(tmp_path: Path) -> ReadOnlyObservationContext:
    images = []
    descriptors = []
    sizes = ((480, 270), (480, 270), (320, 240), (320, 240))
    for index, (label, size) in enumerate(zip(EXPECTED_IMAGE_LABELS, sizes, strict=True)):
        path = tmp_path / f"image_{index}.png"
        Image.new("RGB", size, (index, 20, 30)).save(path)
        images.append(ReadOnlyImageInput(label, "image/png", path, *size, 3, "uint8"))
        descriptors.append(
            {
                "label": label,
                "shape": [size[1], size[0], 3],
                "dtype": "uint8",
                "value_range": [0.0, 255.0],
                "media_type": "image/png",
            }
        )
    return ReadOnlyObservationContext(
        instruction="Pull the key out of the slot.",
        step_identifiers={
            "snapshot_id": "pull-out-key-seed-1000000-ready:pre_action",
            "action_id": "pull-out-key-seed-1000000-ready",
            "phase": "pre_action",
            "simulator_step": 238,
            "take_action_count": 0,
        },
        proprio={"joint": [0.0] * 9, "ee": [0.0] * 7},
        visual_descriptors=tuple(descriptors),
        images=tuple(images),
    )


def _report_payload(**updates: object) -> dict:
    payload = {
        "schema_version": READONLY_REPORT_SCHEMA,
        "mode": "read_only",
        "context_received": True,
        "received_image_labels": list(EXPECTED_IMAGE_LABELS),
        "observation_notes": ["multimodal context transport validated"],
        "uncertainties": ["semantic tactile interpretation was not evaluated"],
        "execution_requested": False,
    }
    payload.update(updates)
    return payload


def test_readonly_runtime_has_no_action_surface_and_calls_backend_once(tmp_path: Path) -> None:
    class Backend:
        def observe(self, context: ReadOnlyObservationContext) -> ReadOnlyObservationReport:
            return ReadOnlyObservationReport.from_mapping(_report_payload())

    runtime = OpenEtaReadOnlyObservationRuntime(Backend())
    assert not hasattr(runtime, "act")
    report = runtime.observe(_context(tmp_path))
    assert report.execution_requested is False
    assert runtime.observe_call_count == 1
    with pytest.raises(RuntimeError, match="exactly one"):
        runtime.observe(_context(tmp_path))


def test_readonly_runtime_rejects_wrong_backend_type(tmp_path: Path) -> None:
    class Backend:
        def observe(self, context: ReadOnlyObservationContext) -> dict:
            return _report_payload()

    with pytest.raises(TypeError, match="ReadOnlyObservationReport"):
        OpenEtaReadOnlyObservationRuntime(Backend()).observe(_context(tmp_path))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["action", "tool_calls", "trajectory", "control"])
def test_readonly_report_rejects_action_or_tool_fields(field: str) -> None:
    with pytest.raises(ReadOnlyObservationError, match="action fields"):
        ReadOnlyObservationReport.from_mapping(_report_payload(**{field: []}))


def test_readonly_backend_rejects_external_url_and_real_key() -> None:
    with pytest.raises(ReadOnlyObservationError, match="loopback"):
        OpenAICompatibleReadOnlyObservationBackend(base_url="https://api.openai.com/v1", model="x")
    with pytest.raises(ReadOnlyObservationError, match="dummy key"):
        OpenAICompatibleReadOnlyObservationBackend(
            base_url="http://127.0.0.1:1/v1", model="x", api_key="real-key"
        )


def test_backend_sends_one_text_four_images_and_no_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self) -> bytes:
            content = json.dumps(_report_payload())
            return json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": content}}]}
            ).encode()

    class Opener:
        def open(self, request: urllib.request.Request, timeout: float):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return Response()

    def build_opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build_opener)
    backend = OpenAICompatibleReadOnlyObservationBackend(
        base_url="http://127.0.0.1:43210/v1", model="capture"
    )
    report = backend.observe(_context(tmp_path))
    assert report.execution_requested is False
    payload = captured["payload"]
    assert set(payload) == {"model", "temperature", "messages"}
    assert [message["role"] for message in payload["messages"]] == ["system", "user"]
    content = payload["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text"] + ["image_url"] * 4
    assert all(
        item["image_url"]["url"].startswith("data:image/png;base64,") for item in content[1:]
    )
    text = content[0]["text"]
    assert "/home/" not in text and "outputs/" not in text and "host_only" not in text
    assert not ({"tools", "tool_choice", "functions"} & payload.keys())
    assert captured["url"] == "http://127.0.0.1:43210/v1/chat/completions"
    assert any(isinstance(handler, urllib.request.ProxyHandler) for handler in captured["handlers"])
    assert any(
        isinstance(handler, urllib.request.HTTPRedirectHandler) for handler in captured["handlers"]
    )


def test_backend_rejects_any_absolute_posix_path_in_text_context(tmp_path: Path) -> None:
    context = replace(_context(tmp_path), instruction="/root/private/context.json")
    backend = OpenAICompatibleReadOnlyObservationBackend(
        base_url="http://127.0.0.1:43210/v1", model="capture"
    )
    with pytest.raises(ReadOnlyObservationError, match="local path"):
        backend.observe(context)


def test_importing_readonly_modules_does_not_load_action_runtime() -> None:
    program = """
import json, sys
import agent.runtime.observation_interfaces
import agent.runtime.observation_input
import agent.runtime.observation_runtime
import agent.backends.readonly_observation
print(json.dumps({name: name in sys.modules for name in (
  'adapter.bridge', 'agent.runtime.runtime', 'agent.runtime.planner',
  'agent.backends.planner', 'sim.env_registry')}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert all(value is False for value in json.loads(completed.stdout).values())


def test_readonly_source_does_not_import_env_action_or_define_act() -> None:
    paths = (
        REPO_ROOT / "agent/runtime/observation_interfaces.py",
        REPO_ROOT / "agent/runtime/observation_input.py",
        REPO_ROOT / "agent/runtime/observation_runtime.py",
        REPO_ROOT / "agent/backends/readonly_observation.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "EnvAction" not in text
    assert "def act(" not in text
    assert "ActionPipeline" not in text
    assert "ToolCallingPlanner" not in text
