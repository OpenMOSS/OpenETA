"""Scene proposal presentation is an Agent aid, never identity/motion authority."""
import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.runtime.memory import AgentMemory, TARGET_IDENTITY_ANCHOR_KEY
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.skills import build_default_skill_registry
from agent.tools.evidence_inspection import inspect_evidence
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


def _scene(tmp_path):
    memory = AgentMemory(artifact_root=tmp_path)
    memory.start_session(task="pick alphabet soup", session_id="scene-test")
    rgb = tmp_path / "scene.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (80, 60), (160, 35, 65)).save(rgb)
    Image.new("L", (80, 60), 255).save(mask)
    outputs = {"result_id": "scene-1", "prompt": "object", "source_image": str(rgb),
               "detection_count": 10, "source_packet_id": "packet-1", "frame_id": "agentview",
               "detections": [{"id": f"detection_{n:03d}", "rank": n,
                               "mask_ref": str(mask), "bbox_xyxy": [10, 10, 40, 40]}
                              for n in range(10)]}
    memory.add_action(EnvAction(action_type="tool_call", command={"status": "executed",
        "tool_calls": [{"name": "sam3", "status": "executed",
                        "result": {"success": True, "details": {"outputs": outputs}}}]}))
    return memory, memory.tool_handoffs()[0]["bundle_id"]


def _inspect(memory, parameters):
    tools = build_default_tool_registry()
    return inspect_evidence(ToolExecutionContext(name="inspect_evidence",
        spec=tools.get("inspect_evidence"), parameters=parameters), memory)


def test_scene_pages_keep_ids_and_original_colors_without_selecting(tmp_path):
    memory, bundle = _scene(tmp_path)
    epochs = (memory.object_scene_epoch(), memory.robot_motion_epoch())
    for offset, expected_next in ((0, 4), (4, 8), (8, None)):
        result = _inspect(memory, {"bundle_id": bundle, "offset": offset})
        assert result.success, result.content
        page = result.details["outputs"]
        assert page["next_offset"] == expected_next
        assert page["candidate_count"] == 10
        assert page["candidates"][0]["id"] == f"detection_{offset:03d}"
        with Image.open(page["candidates"][0]["crop_ref"]) as crop:
            assert crop.getpixel((5, 5)) == (160, 35, 65)
        assert page["authorizes_selection_or_motion"] is False
    assert memory.target_identity_anchor() is None
    assert (memory.object_scene_epoch(), memory.robot_motion_epoch()) == epochs
    assert memory.pending_sam3_selection()["result_id"] == "scene-1"


@pytest.mark.parametrize("params", [{"bundle_id": "bnd-" + "0" * 32},
    {"offset": -1}, {"offset": True}, {"offset": 10}, {"offset": None},
    {"image_ref": "/etc/passwd"}, {"image_ref": None}, {"unknown": "x"}])
def test_bad_inspection_clears_previous_view_without_inference(tmp_path, params):
    memory, bundle = _scene(tmp_path)
    assert _inspect(memory, {"bundle_id": bundle}).success
    args = {"bundle_id": bundle, **params}
    if "image_ref" in params:
        args = params
    assert not _inspect(memory, args).success
    assert "evidence_inspection" not in memory.facts
    assert memory.target_identity_anchor() is None


def test_inspection_blocks_input_and_output_symlink_escapes(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    memory, bundle = _scene(root)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (10, 10)).save(outside)
    (root / "escape.png").symlink_to(outside)
    assert not _inspect(memory, {"image_ref": str(root / "escape.png")}).success
    (root / "evidence_views").symlink_to(tmp_path, target_is_directory=True)
    assert not _inspect(memory, {"bundle_id": bundle}).success


def _reference_lookup(memory, tmp_path, *, failed=False):
    reference = tmp_path / "reference.png"
    Image.new("RGB", (50, 50), (15, 60, 210)).save(reference)
    memory.add_action(EnvAction(action_type="tool_call", command={"status": "executed",
        "tool_calls": [{"name": "retrieve_asset_reference", "status": "executed",
            "result": {"success": not failed, "details": {"outputs": {
                "target_object": "alphabet_soup", "environment": "libero",
                "localization_status": "not_requested",
                "reference_images": [] if failed else [str(reference)],
            }}}}]}))


def test_reference_comparison_covers_beyond_detail_page_without_selecting(tmp_path):
    memory, bundle = _scene(tmp_path)
    _reference_lookup(memory, tmp_path)
    result = _inspect(memory, {"bundle_id": bundle})
    assert result.success
    output = result.details["outputs"]
    comparison = output["reference_comparison"]
    assert comparison["reference_object"] == "alphabet_soup"
    assert comparison["candidate_ids"] == [f"detection_{n:03d}" for n in range(8)]
    assert comparison["next_offset"] == 8
    assert output["next_offset"] == 4
    assert comparison["identity_confirmed"] is False
    assert memory.target_identity_anchor() is None
    with Image.open(comparison["image_ref"]) as picture:
        assert picture.getpixel((160, 150)) == (15, 60, 210)
        assert picture.getpixel((160, 390)) == (160, 35, 65)
    _reference_lookup(memory, tmp_path, failed=True)
    after = _inspect(memory, {"bundle_id": bundle}).details["outputs"]
    assert after["reference_comparison"]["status"] != "available"
    assert len(after["vision_evidence"]) == 1


def test_unregistered_observation_id_remains_rejected(tmp_path):
    memory, _ = _scene(tmp_path)
    result = _inspect(memory, {"image_ref": "observation:0:agentview"})
    assert not result.success
    assert "unknown or ambiguous image evidence ID" in result.content


def _register_picture(memory, path, step):
    memory.add_observation(EnvObservation(task="inspect", cameras=[], robot=RobotState(),
        metadata={"step_idx": step, "image_artifacts": [{"kind": "rgb", "frame_id": "agentview", "path": str(path)}]}))


def test_published_image_ids_resolve_exact_historical_pixels_without_refresh(tmp_path):
    from agent.runtime.visual_history import observation_history, _planner_image_evidence
    memory, _ = _scene(tmp_path)
    original = tmp_path / "scene.png"
    _register_picture(memory, original, 3)
    record = observation_history(memory)[-1]
    reference = _planner_image_evidence(record, record["visual_artifacts"][0], role="current_scene", freshness="current")["evidence_id"]
    first = _inspect(memory, {"image_ref": reference})
    assert first.success, first.content
    assert first.details["outputs"]["image_ref"] == str(original)
    assert _inspect(memory, {"image_ref": "current_observation:3:agentview"}).success
    newer = tmp_path / "new.png"
    Image.new("RGB", (80, 60), (20, 60, 90)).save(newer)
    _register_picture(memory, newer, 8)
    before = (memory.object_scene_epoch(), memory.robot_motion_epoch())
    old = _inspect(memory, {"image_ref": reference}).details["outputs"]
    assert old["image_ref"] == str(original)
    assert old["requested_image_ref"] == reference
    assert old["authorizes_selection_or_motion"] is False
    assert not _inspect(memory, {"image_ref": "current_observation:3:agentview"}).success
    assert _inspect(memory, {"image_ref": "current_observation:8:agentview"}).details["outputs"]["image_ref"] == str(newer)
    assert (memory.object_scene_epoch(), memory.robot_motion_epoch()) == before
    assert memory.target_identity_anchor() is None


def test_observation_id_cannot_bypass_session_image_boundary(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    memory, _ = _scene(root)
    outside = tmp_path / "outside.png"
    Image.new("RGB", (16, 16)).save(outside)
    _register_picture(memory, outside, 0)
    result = _inspect(memory, {"image_ref": "observation:0:agentview"})
    assert not result.success
    assert "inside this session" in result.content


@pytest.mark.parametrize("inspection_kind", ["page", "reference_page", "published_image_id"])
def test_requested_page_reaches_actual_provider_input_under_image_cap(tmp_path, inspection_kind):
    from agent.backends.planner import OpenAICompatiblePlannerBackend, OpenAICompatiblePlannerBackendConfig
    memory, bundle = _scene(tmp_path)
    with_reference = inspection_kind == "reference_page"
    if with_reference:
        _reference_lookup(memory, tmp_path)
    parameters = {"bundle_id": bundle, "offset": 8}
    if inspection_kind == "published_image_id":
        _register_picture(memory, tmp_path / "scene.png", 0)
        parameters = {"image_ref": "observation:0:agentview"}
    result = _inspect(memory, parameters)
    assert result.success, result.content
    page = result.details["outputs"]
    bodies = []

    def transport(url, body, headers, timeout_s):
        bodies.append(body)
        return {"choices": [{"message": {"content": '<decision><kind>response</kind><name>talk</name><parameters><message>seen</message></parameters><reasoning>viewed page</reasoning></decision>'}}]}

    backend = OpenAICompatiblePlannerBackend(OpenAICompatiblePlannerBackendConfig(
        model="fixture", api_base="https://example.invalid", api_key="fixture",
        enable_vision=True, max_attempts=1, max_vision_images=1,
    ), transport=transport)
    ToolCallingPlanner(backend=backend).plan(
        EnvObservation(task="pick alphabet soup", cameras=[], robot=RobotState()),
        memory=memory, tools=build_default_tool_registry(), skills=build_default_skill_registry())
    images = [item for message in bodies[0]["messages"] if isinstance(message["content"], list)
              for item in message["content"] if item.get("type") == "image_url"]
    assert len(images) == 1
    raw = base64.b64decode(images[0]["image_url"]["url"].split(",", 1)[1])
    expected_path = (page["image_ref"] if inspection_kind == "published_image_id" else
                     page["reference_comparison"]["image_ref"] if with_reference else page["contact_sheet_ref"])
    with Image.open(io.BytesIO(raw)) as transmitted, Image.open(expected_path) as expected:
        assert transmitted.size == expected.size
        assert transmitted.convert("RGB").tobytes() == expected.convert("RGB").tobytes()
    text = json.dumps(bodies[0])
    assert "inspect_evidence" in text
    assert ("observation:0:agentview" if inspection_kind == "published_image_id" else "detection_008") in text


def _pending(memory, result):
    memory.save_fact("pending_sam3_selection", {"result_id": result,
        "evidence_role": "target_object", "source_packet_id": result,
        "candidates": [{"id": "one", "mask_ref": "mask.png"}]}, source="test")


def test_initial_identity_misuse_is_explicit_and_old_verifier_cannot_veto_repair():
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    _pending(memory, "first")
    first = memory.resolve_sam3_selection(result_id="first", detection_id="one",
        selection_source="main_agent_vlm", reason="first visual choice",
        identity_anchor_id="alphabet_soup", identity_relation="same_instance")
    assert first["identity_anchor_id"] != "alphabet_soup"
    assert first["identity_continuity"] == "anchor_created"
    assert first["identity_parameter_feedback"]["blocking"] is False
    anchor = dict(memory.target_identity_anchor())
    anchor["exact_instance_verification"] = {"decision": "match", "confidence": 0.99}
    memory.save_fact(TARGET_IDENTITY_ANCHOR_KEY, anchor, source="test")
    _pending(memory, "second")
    with pytest.raises(ValueError, match="target_identity_confirmation_required"):
        memory.resolve_sam3_selection(result_id="second", detection_id="one",
            selection_source="main_agent_vlm", reason="clear close view disagrees",
            identity_anchor_id="wrong", identity_relation="replace_misidentified_anchor")
    corrected = memory.resolve_sam3_selection(result_id="second", detection_id="one",
        selection_source="main_agent_vlm", reason="clear close view disagrees with old verifier",
        identity_anchor_id=first["identity_anchor_id"], identity_relation="replace_misidentified_anchor")
    assert corrected["replaced_identity_anchor_id"] == first["identity_anchor_id"]
    assert corrected["identity_anchor_id"] != first["identity_anchor_id"]


@pytest.mark.parametrize("payload", ["broken JSON", {"decision": None},
    {"decision": "recommend", "recommended_candidate_id": "unknown"}, {"action": "move"}])
def test_invalid_advisor_retains_known_usage_not_rejected_payload(payload):
    from agent.backends.planner import PlannerBackendResult
    from agent.tools.grasp_pose_advisor import BackendGraspPoseAdvisor, GraspAdvisorResponseError
    backend = SimpleNamespace(decide=lambda request: PlannerBackendResult(payload=payload,
        provider="fixture", model="fixture", details={"usage": {"total_tokens": 1234, "private": "secret"}}))
    with pytest.raises(GraspAdvisorResponseError) as caught:
        BackendGraspPoseAdvisor(backend).advise({"candidates": [{"candidate_id": "g1"}],
            "overview_ref": "overview.png", "contact_sheet_refs": []}, task="pick")
    diagnostics = caught.value.advisor_diagnostics
    assert diagnostics["usage"] == {"total_tokens": 1234}
    assert diagnostics["failure_phase"] == "response_validation"
    assert "secret" not in json.dumps(diagnostics)
