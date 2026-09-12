"""Local regressions distilled from recorded failures; no provider or simulator."""
import json
from io import BytesIO
from dataclasses import replace

import pytest
from PIL import Image

from adapter.protocol import EnvAction
from agent.backends.planner import PlannerBackendRequest, OpenAICompatiblePlannerBackendConfig, _planner_user_prompt, _planner_user_content
from agent.runtime.interface_profiles import profile_parameter_errors, _stage_contract
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import _latest_tool_review_vision_evidence, _catalog_reference_vision_evidence
from agent.tools.registry import build_default_tool_registry, ToolExecutionContext
from agent.tools.object_memory import ObjectMemoryBundle, ObjectMemoryReference
from agent.tools.asset_references import build_object_memory_reference_handler


@pytest.mark.parametrize("bad", [
    {"bundle_id": " bnd-" + "a" * 32, "candidate_id": "detection_002"},
    {"bundle_id": "bnd-" + "a" * 32, "detection_id": "detection_002", "target_geometry_family": "can"},
    {"bundle_id": "bnd-" + "a" * 32, "detection_id": "detection_002", "identity_anchor_id": None, "identity_relation": None},
])
def test_recorded_selection_failures_have_exact_tool_repair_focus(bad):
    errors = profile_parameter_errors("bundle_stage3", "select_sam3_detection", bad)
    assert errors
    candidate = {"kind": "tool_call", "name": "select_sam3_detection", "parameters": bad}
    schema = _stage_contract("select_sam3_detection").request_schema
    request = PlannerBackendRequest(system_prompt="fixture", tool_context={"available_tools": [
        {"name": "select_sam3_detection", "parameters": schema}]},
        validation_errors=errors, metadata={"previous_candidate": candidate}, attempt=2)
    feedback = json.loads(_planner_user_prompt(request))["validation_feedback"]
    assert feedback["repair_focus"]["parameters_schema"] == schema
    assert feedback["previous_candidate"] == candidate
    assert "omit optional" in feedback["repair_focus"]["instruction"]
    # This is an explicitly authored test correction, not Host normalization.
    assert not profile_parameter_errors("bundle_stage3", "select_sam3_detection", {
        "bundle_id": "bnd-" + "a" * 32, "detection_id": "detection_002"})


@pytest.mark.parametrize("reference_only", [False, True])
def test_reference_images_do_not_require_localizer_success_or_grant_authority(tmp_path, reference_only):
    scene = tmp_path / "scene.png"
    Image.new("RGB", (64, 64), "gray").save(scene)
    buf = BytesIO()
    Image.new("RGB", (32, 32), "blue").save(buf, format="PNG")
    class Client:
        def resolve(self, **kwargs):
            return ObjectMemoryBundle(query_key="libero/alphabet_soup", namespace="libero",
                asset_id="alphabet_soup", label="alphabet soup",
                references=(ObjectMemoryReference("front", "front.png", buf.getvalue()),), manifest={})
    localizer_calls = []
    class Localizer:
        def localize(self, **kwargs):
            localizer_calls.append(kwargs)
            raise ValueError("ambiguous scene")
    params = {"environment": "libero", "target_object": "alphabet soup", "scene_image": str(scene)}
    if reference_only:
        params["localize"] = False
    handler = build_object_memory_reference_handler(Client(), Localizer(), output_root=tmp_path / "references")
    result = handler(ToolExecutionContext(name="retrieve_asset_reference", parameters=params,
        spec=build_default_tool_registry().get("retrieve_asset_reference")))
    assert result.success is reference_only
    assert len(localizer_calls) == (0 if reference_only else 1)
    outputs = result.details["outputs"]
    assert outputs["reference_status"] == "retrieved"
    assert outputs["identity_confirmed"] is False
    assert "positive_points" not in outputs
    if reference_only:
        from agent.tools.contracts import build_default_tool_contract_catalog, check_tool_result_conformance
        contract = build_default_tool_contract_catalog(build_default_tool_registry().list()).get("retrieve_asset_reference")
        assert not check_tool_result_conformance(contract, result.details)
    memory = AgentMemory(artifact_root=tmp_path / "artifacts")
    memory.start_session(task="pick")
    memory.add_action(EnvAction(action_type="tool_call", command={"status": "executed", "tool_calls": [{
        "name": "retrieve_asset_reference", "parameters": params,
        "result": {"success": result.success, "details": result.details}}]}))
    assert memory.target_asset_reference() is None
    assert memory.pending_reference_localization() is None
    evidence = _latest_tool_review_vision_evidence(memory)
    assert len(evidence) == 1 and evidence[0]["not_world_observation"]
    assert evidence[0]["reference_target"] == "alphabet_soup"
    config = OpenAICompatiblePlannerBackendConfig(api_base="https://example.invalid", api_key="fixture", model="fixture", max_vision_images=1)
    request = PlannerBackendRequest(system_prompt="fixture", tool_context={
        "review_vision_evidence": evidence, "vision_image_paths": [str(scene)],
        "vision_evidence": [{"path": str(scene), "role": "current_scene"}]})
    _, attachments = _planner_user_content(request, config)
    assert [a["path"] for a in attachments] == [str(scene)]
    config = replace(config, max_vision_images=3)
    _, attachments = _planner_user_content(request, config)
    assert any(a.get("role") == "object_appearance_reference" and a["attached"] for a in attachments)
    memory.add_action(EnvAction(action_type="tool_call", command={"tool_calls": [{"name": "get_memory"}]}))
    assert _catalog_reference_vision_evidence(memory) == evidence


@pytest.mark.parametrize("point_key", ["points", "positive_points"])
@pytest.mark.parametrize("packet,frame", [("obs-0001", "wrist"), ("obs-0001", "agentview"), ("obs-0000", "wrist")])
def test_recorded_cross_view_reference_pixels_are_rejected(point_key, packet, frame, monkeypatch):
    from agent.tools.runtime_contract_bindings import _resolve_sam3_input, HostResolutionFailure
    memory = AgentMemory()
    memory.start_session(task="fixture")
    points = [{"x": 218, "y": 240, "label": 1}]
    memory.save_fact("pending_reference_localization", {
        "source_packet_id": "obs-0000", "camera_frame_id": "agentview",
        "positive_points": points, "required_parameter": "positive_points",
    }, source="fixture")
    monkeypatch.setattr(memory, "resolve_observation_packet", lambda p, f: {"packet_id": p, "frame_id": f})
    params = {"source_packet_id": packet, "camera_frame_id": frame, "mode": "points", point_key: points}
    with pytest.raises(HostResolutionFailure, match="Reused reference-localization pixels") as error:
        _resolve_sam3_input(params, memory)
    assert error.value.repair_code == "same_view_packet_mismatch"
    exact = {**params, "source_packet_id": "obs-0000", "camera_frame_id": "agentview"}
    assert _resolve_sam3_input(exact, memory) == exact
    # Independently chosen new-view points remain allowed; no mandatory tool order.
    independent = {**params, point_key: [{"x": 100, "y": 100, "label": 1}]}
    assert _resolve_sam3_input(independent, memory) == independent


def test_projected_open_questions_attach_selection_sheet_and_catalog_before_crops(tmp_path):
    def picture(name):
        path = tmp_path / f"{name}.png"
        Image.new("RGB", (16, 16), "blue").save(path)
        return str(path)
    scene, wrist, source, sheet = [picture(name) for name in ("scene", "wrist", "source", "sheet")]
    references = [picture(f"reference-{i}") for i in range(3)]
    crops = [picture(f"crop-{i}") for i in range(8)]
    request = PlannerBackendRequest(system_prompt="fixture", tool_context={
        "schema_version": "openeta.agent_context.v1",
        "open_questions": {"target_selection": {"selection_bundle": {
            "original_image_ref": source, "contact_sheet_ref": sheet,
            "candidates": [{"crop_ref": path} for path in crops],
        }}},
        "vision_image_paths": [scene, wrist],
        "vision_evidence": [{"path": p, "role": "current_scene"} for p in (scene, wrist)],
        "review_vision_evidence": [{"path": p, "role": "object_appearance_reference", "reference_target": "fixture"} for p in references],
    })
    config = OpenAICompatiblePlannerBackendConfig(api_base="https://example.invalid", api_key="fixture", model="fixture", max_vision_images=8)
    _, attachments = _planner_user_content(request, config)
    attached = [a["path"] for a in attachments if a["attached"]]
    assert attached[:7] == [scene, wrist, sheet, source, *references]
    assert len(attached) == 8
    # A still-pending localization must not suppress the actual SAM3 review.
    request.tool_context["open_questions"]["reference_localization"] = {"required_parameter": "positive_points"}
    _, attachments = _planner_user_content(request, config)
    assert any(a["path"] == sheet and a["attached"] for a in attachments)
