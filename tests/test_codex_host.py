from pathlib import Path
import json

import pytest
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.planner import ToolCallingPlanner
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry
from tools.codex_host import CodexHost, SubmittedBackend, DisabledModelBackend


@pytest.fixture
def host(tmp_path):
    backend = SubmittedBackend()
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(backend, max_validation_retries=0),
        tools=bind_dummy_tool_handlers(build_default_tool_registry()))
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    runner.start(task="inspect the scene", max_turns=10, timeout_s=60)
    runtime.memory.add_observation(runner.current_observation)
    value = CodexHost(runner, backend, output=tmp_path, max_requests=10)
    yield value
    value.close()


def body(result):
    return json.loads(result.content[0].text)


def _register_test_camera(host, tmp_path, name):
    from adapter.protocol import CameraFrame
    from PIL import Image
    rgb, depth = tmp_path / f"{name}.png", tmp_path / f"{name}-depth.png"
    Image.new("RGB", (8, 8), (90, 30, 10)).save(rgb)
    Image.new("I;16", (8, 8), 1000).save(depth)
    host.runtime.memory.artifact_root = tmp_path / "artifacts"
    obs = host.runner.current_observation
    obs.cameras = [CameraFrame(frame_id="wrist", rgb=[[[90, 30, 10]] * 8] * 8,
        intrinsics={"fx": 8, "fy": 8, "cx": 4, "cy": 4},
        extrinsics={"camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})]
    obs.metadata["image_artifacts"] = [
        {"kind": kind, "frame_id": "wrist", "path": str(path), "packet_id": f"transport-{'x' * 40}-{name}"}
        for kind, path in [("rgb", rgb), ("depth", depth)]]
    host.runtime.memory.add_observation(obs)
    return rgb


def test_native_packet_repair_is_actionable_without_an_extra_inspection(host, tmp_path):
    from agent.tools.registry import ToolResult
    _register_test_camera(host, tmp_path, "fresh")
    calls = []
    host.runtime.tools.bind_handler("sam3", lambda ctx: calls.append(ctx.parameters) or ToolResult(True), replace=True)
    failed = host.call("sam3", {"mode": "points", "points": [{"x": 3, "y": 3, "label": 1}],
                                "source_packet_id": "transport-" + "x" * 40 + "-fresh", "camera_frame_id": "wrist"})
    assert failed.isError
    wire = body(failed)
    assert wire["repair"]["code"] == "invalid_source_packet"
    current = wire["observation_references"]["current"]
    assert current["source_packet_id"].startswith("obs-")
    assert current["camera_frame_ids"] == ["wrist"]
    assert wire["repair"]["recent_source_packets"]
    repaired = host.call("sam3", {"mode": "points", "points": [{"x": 3, "y": 3, "label": 1}],
        "source_packet_id": current["source_packet_id"], "camera_frame_id": "wrist"})
    assert not repaired.isError, body(repaired).get("error")
    assert len(calls) == 1
    assert "repair" not in body(repaired)


def test_native_image_labels_bind_exact_images_and_do_not_promote_derived_images(host, tmp_path):
    from tools.codex_evidence import image_label
    old = _register_test_camera(host, tmp_path, "old")
    fresh = _register_test_camera(host, tmp_path, "fresh")
    old_label = json.loads(image_label(host.runtime.memory, {"path": str(old), "role": "historical"}, 1))
    fresh_label = json.loads(image_label(host.runtime.memory, {"path": str(fresh), "role": "current_scene"}, 2))
    derived = json.loads(image_label(host.runtime.memory, {"path": str(tmp_path / "overlay.png"), "role": "mask_overlay"}, 3))
    assert not old_label["is_current_observation"] and fresh_label["is_current_observation"]
    assert old_label["source_packet_id"] != fresh_label["source_packet_id"]
    assert not derived["is_current_observation"] and "source_packet_id" not in derived
    result = host.call("episode_status", {})
    for i, content in enumerate(result.content):
        if content.type == "image":
            label = json.loads(result.content[i-1].text)
            assert "image_index" in label and "instruction" in label
            assert str(tmp_path) not in result.content[i-1].text


def test_native_repair_excludes_private_payload_and_noncallable_hints(host):
    from tools.codex_evidence import repair_feedback
    command = {"metadata": {"repair_bundle": {"code": "invalid_source_packet",
        "private_geometry": "must-not-leak", "allowed_next_calls": [
            {"tool": "python_exec", "parameters": {"code": "bad"}},
            {"tool": "observe", "parameters": {"unsupported": True}},
            {"tool": "observe", "parameters": {}}]}}}
    repair = repair_feedback(command, host.schemas)
    assert repair["allowed_next_calls"] == [{"tool": "observe", "parameters": {}}]
    assert "must-not-leak" not in json.dumps(repair)


def test_native_observe_runs_existing_host_and_records_action(host):
    result = host.call("observe", {})
    assert not result.isError
    assert host.runner.turn_index == 1
    assert host.runner.tool_call_count >= 1
    assert body(result)["executed"]["name"] == "observe"
    assert (host.output / "host-commands.jsonl").exists()
    assert host.backend.pending is None


def test_invalid_arguments_remain_recoverable_and_do_not_execute(host):
    result = host.call("gripper_control", {"position": "wrong-type"})
    assert result.isError
    assert host.runner.turn_index == 0
    assert host.runner.tool_call_count == 0
    assert not host.runner.truncated
    assert not host.call("observe", {}).isError


def test_unexposed_tool_cannot_bypass_host(host):
    assert host.runtime.tools.can_execute("python_exec")
    assert host.call("python_exec", {"code": "raise Exception('must not run')"}).isError
    assert host.runner.turn_index == 0


def test_fake_ik_reference_reaches_existing_gate_without_motion(host):
    result = host.call("move_to", {"ik_receipt_id": "invented"})
    # May be rejected at schema or existing pipeline admission, never successful motion.
    if not result.isError:
        rows = [json.loads(s) for s in (host.output / "host-commands.jsonl").read_text().splitlines()]
        assert rows[-1]["command"]["status"] in {"blocked", "failed"}
    assert not host.runner.terminated


def test_request_budget_includes_invalid_submissions(host):
    host.max_requests = 1
    assert host.call("missing", {}).isError
    assert host.call("observe", {}).isError
    assert host.closed
    assert host.runner.failure_reason["code"] == "codex_request_limit"
    assert host.runner.turn_index == 0


def test_close_refuses_future_motion_and_is_idempotent(host):
    host.close()
    first = host.cleanup
    host.close()
    assert host.cleanup == first
    assert host.call("observe", {}).isError
    assert body(host.call("episode_status", {}))["episode"]["closed"]


def test_auxiliary_backend_never_calls_api():
    with pytest.raises(RuntimeError, match="disabled"):
        DisabledModelBackend().decide(None)


def test_success_claim_requires_official_environment_evidence(host):
    result = host.call("finish_episode", {"success": True, "reason": "I think it worked"})
    assert result.isError
    assert body(result)["error"]["code"] == "official_success_not_established"
    assert not host.closed
    assert host.runner.turn_index == 0


def test_failure_finish_closes_environment_without_success_claim(host):
    result = host.call("finish_episode", {"success": False, "reason": "bounded smoke complete"})
    assert not result.isError
    assert host.closed
    assert host.runner.terminated
    assert host.cleanup["ok"]


def test_images_use_native_mcp_content_and_hide_transport_paths(host, tmp_path):
    from adapter.protocol import CameraFrame
    from PIL import Image
    import base64
    import io
    host.runtime.memory.artifact_root = tmp_path / "images"
    host.runner.current_observation.cameras = [CameraFrame(
        frame_id="agentview", role="scene_primary", rgb=[[[240, 10, 20]] * 8] * 8)]
    path = tmp_path / "camera.png"
    Image.new("RGB", (8, 8), (240, 10, 20)).save(path)
    host.runner.current_observation.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "agentview", "role": "scene_primary", "path": str(path)}]
    host.runtime.memory.add_observation(host.runner.current_observation)
    result = host.call("episode_status", {})
    images = [c for c in result.content if c.type == "image"]
    assert images
    with Image.open(io.BytesIO(base64.b64decode(images[0].data))) as im:
        assert im.size == (8, 8)
        assert im.convert("RGB").getpixel((0, 0)) == (240, 10, 20)
    assert "vision_image_paths" not in body(result)["context"]


def test_host_invariant_does_not_leave_a_stale_submitted_command(host, monkeypatch):
    from agent.runtime.planner import PlannerDecision
    monkeypatch.setattr("agent.runtime.planner._invariant_obligation_decision",
        lambda *a, **kw: PlannerDecision(action_type="tool_call", action="observe", parameters={}))
    result = host.call("gripper_control", {"position": 1})
    assert body(result)["executed"]["name"] == "observe"
    assert host.backend.pending is None


def test_real_mcp_stdio_call_and_disconnect_cleanup(tmp_path):
    import asyncio
    import os
    import sys
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    fixture = tmp_path / "fixture.py"
    fixture.write_text('''
import asyncio, sys, signal
from pathlib import Path
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.planner import ToolCallingPlanner
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry
from tools.codex_host import CodexHost, SubmittedBackend
from tools.codex_mcp_server import serve
b = SubmittedBackend()
r = OpenEtaEpisodeRunner(runtime=OpenEtaAgentRuntime(
    planner=ToolCallingPlanner(b, max_validation_retries=0),
    tools=bind_dummy_tool_handlers(build_default_tool_registry())), environment=DummyEpisodeEnvironment())
r.start(task="observe fixture", max_turns=5, timeout_s=30)
r.runtime.memory.add_observation(r.current_observation)
h = CodexHost(r,b,output=Path(sys.argv[1]))
try: asyncio.run(serve(h))
except (KeyboardInterrupt, asyncio.CancelledError): pass
finally: h.close()
''')
    root = tmp_path / "host"
    async def exercise():
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        async with stdio_client(StdioServerParameters(command=sys.executable,
            args=[str(fixture), str(root)], env=env)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                catalog = await session.list_tools()
                assert "observe" in {t.name for t in catalog.tools}
                rejected = await session.call_tool("gripper_control", {"position": "wrong"})
                assert rejected.isError
                result = await session.call_tool("observe", {})
                assert not result.isError
                assert body(result)["executed"]["name"] == "observe"
    asyncio.run(asyncio.wait_for(exercise(), timeout=15))
    status = json.loads((root / "host-status.json").read_text())
    assert status["closed"]
    assert status["cleanup"]["ok"]


def test_no_submitted_request_cannot_be_replayed():
    backend = SubmittedBackend()
    backend.pending = {"kind": "tool_call", "name": "observe", "parameters": {}}
    assert backend.decide(None).payload["name"] == "observe"
    with pytest.raises(RuntimeError, match="No external"):
        backend.decide(None)


def test_watchdog_counts_time_between_tools(host):
    host._expire()
    assert host.closed
    assert host.runner.failure_reason["code"] == "episode_timeout"


def test_mcp_advertises_real_contract_and_rejects_unknown(host):
    import asyncio
    from tools.codex_mcp_server import server_for
    from mcp import types
    server = server_for(host)
    result = asyncio.run(server.request_handlers[types.ListToolsRequest](types.ListToolsRequest(method="tools/list")))
    tools = result.root.tools
    assert {t.name for t in tools} == set(host.schemas)
    assert next(t for t in tools if t.name == "observe").inputSchema == host.schemas["observe"].inputSchema


def test_result_indexes_new_environment_image_before_next_decision(host, tmp_path):
    from copy import deepcopy
    from PIL import Image
    from adapter.protocol import CameraFrame
    # A real runner result has new pixels that runtime.act has not indexed yet.
    obs = deepcopy(host.runner.current_observation)
    rgb = tmp_path / 'after-motion.png'
    Image.new('RGB', (8, 8), (1, 2, 3)).save(rgb)
    obs.cameras = [CameraFrame(frame_id='wrist', rgb=[[[1, 2, 3]] * 8] * 8)]
    obs.metadata['image_artifacts'] = [{'kind': 'rgb', 'frame_id': 'wrist',
        'packet_id': 'transport-' + 'x' * 40, 'path': str(rgb)}]
    host.runner.current_observation = obs
    assert not host.runtime.memory.observation_packet_reference_for_path(str(rgb))
    result = host.call('episode_status', {})
    refs = body(result)['observation_references']
    labels = [json.loads(result.content[i-1].text) for i, item in enumerate(result.content) if item.type == 'image']
    assert labels and labels[0]['is_current_observation']
    assert labels[0]['source_packet_id'] == refs['current']['source_packet_id']
    assert labels[0]['camera_frame_id'] == 'wrist'
    event_count = len(host.runtime.memory.events)
    assert body(host.call('episode_status', {}))['observation_references'] == refs
    assert len(host.runtime.memory.events) == event_count
    # The core's next act registers the same immutable image again. The ID
    # already given to Codex must remain valid for that exact image.
    host.runtime.memory.add_observation(obs)
    source = host.runtime.memory.resolve_observation_packet(labels[0]['source_packet_id'], 'wrist')
    assert source['rgb'] == str(rgb)
