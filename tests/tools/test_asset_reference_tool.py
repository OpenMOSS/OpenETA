from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from adapter.protocol import EnvAction
from agent.runtime.memory import AgentMemory
from agent.runtime.reference_localization import ReferencePointLocalization
from agent.tools.asset_references import (
    _RejectRedirects,
    AssetReferenceCatalog,
    build_asset_reference_handler,
    build_object_memory_reference_handler,
)
from agent.tools.object_memory import (
    ObjectMemoryBundle,
    ObjectMemoryReference,
    ObjectMemoryResolutionError,
    ObjectMemorySearchCandidate,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


def _write_catalog(
    tmp_path: Path,
    *,
    references: list[dict],
    allowed_hosts: list[str] | None = None,
) -> Path:
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "openeta.asset_reference_catalog.v1",
                "allowed_hosts": allowed_hosts or [],
                "environments": [
                    {
                        "id": "libero",
                        "aliases": ["openeta/libero_*"],
                        "objects": [
                            {
                                "id": "alphabet_soup",
                                "aliases": ["alphabet soup", "alphabet soup can"],
                                "references": references,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _context(parameters: dict, *, session_id: str = "asset-session"):
    spec = build_default_tool_registry().get("retrieve_asset_reference")
    return ToolExecutionContext(
        name=spec.name,
        spec=spec,
        parameters=parameters,
        metadata={"session_id": session_id},
    )


def test_asset_reference_handler_materializes_catalog_owned_images(tmp_path: Path) -> None:
    source = tmp_path / "reference-source.png"
    scene = tmp_path / "scene.png"
    Image.new("RGB", (12, 10), "red").save(source)
    Image.new("RGB", (32, 24), "blue").save(scene)
    catalog = AssetReferenceCatalog.load(
        _write_catalog(tmp_path, references=[{"path": source.name}])
    )
    handler = build_asset_reference_handler(catalog, output_root=tmp_path / "outputs")

    result = handler(
        _context(
            {
                "environment": "openeta/libero_pick_alpha",
                "target_object": "alphabet soup can",
                "scene_image": str(scene),
            }
        )
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["environment"] == "libero"
    assert outputs["target_object"] == "alphabet_soup"
    assert outputs["scene_image"] == str(scene)
    assert len(outputs["reference_images"]) == 1
    reference = Path(outputs["reference_images"][0])
    assert reference.is_file()
    assert "asset-session" in reference.parts
    with Image.open(reference) as image:
        assert image.size == (12, 10)
        assert image.mode == "RGB"
    bundle = outputs["localization_bundle"]
    assert bundle["scene_image_ref"] == str(scene)
    assert bundle["reference_image_refs"] == [str(reference)]


def test_asset_reference_catalog_rejects_non_allowlisted_url(tmp_path: Path) -> None:
    catalog = AssetReferenceCatalog.load(
        _write_catalog(
            tmp_path,
            references=[{"url": "https://untrusted.example/object.png"}],
            allowed_hosts=["assets.example.test"],
        )
    )
    resolved = catalog.resolve(environment="libero", target_object="alphabet soup")
    calls: list[str] = []

    try:
        catalog.materialize_reference(
            resolved.references[0],
            output_path=tmp_path / "output.png",
            downloader=lambda url, timeout, limit: calls.append(url) or b"",
            timeout_s=1,
            max_bytes=1024,
        )
    except ValueError as exc:
        assert "not allowlisted" in str(exc)
    else:
        raise AssertionError("expected non-allowlisted URL to be rejected")
    assert calls == []


def test_asset_reference_downloader_rejects_http_redirects() -> None:
    handler = _RejectRedirects()

    redirected = handler.redirect_request(
        object(),
        object(),
        302,
        "Found",
        {},
        "https://untrusted.example/object.png",
    )

    assert redirected is None


def test_asset_reference_result_creates_and_resolves_localization_obligation(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.png"
    scene = tmp_path / "scene.png"
    Image.new("RGB", (8, 8), "red").save(reference)
    Image.new("RGB", (32, 24), "blue").save(scene)
    catalog = AssetReferenceCatalog.load(
        _write_catalog(tmp_path, references=[{"path": reference.name}])
    )
    result = build_asset_reference_handler(catalog, output_root=tmp_path / "outputs")(
        _context(
            {
                "environment": "libero",
                "target_object": "alphabet soup",
                "scene_image": str(scene),
            }
        )
    )
    memory = AgentMemory()
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "retrieve_asset_reference",
                        "result": {
                            "success": result.success,
                            "details": result.details,
                        },
                    }
                ]
            },
        )
    )

    pending = memory.pending_reference_localization()
    assert pending is not None
    assert pending["scene_image"] == str(scene)
    assert pending["required_parameter"] == "roi_bbox_xyxy"
    assert memory.detection_selection_gate_error(
        tool_name="anygrasp",
        parameters={},
    )
    assert memory.detection_selection_gate_error(
        tool_name="sam3",
        parameters={"image": str(scene)},
    )

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "sam3",
                        "parameters": {
                            "image": str(scene),
                            "prompt": "alphabet soup can",
                            "roi_bbox_xyxy": [2, 3, 20, 18],
                        },
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {
                                    "image": str(scene),
                                    "prompt": "alphabet soup can",
                                    "roi_bbox_xyxy": [2, 3, 20, 18],
                                },
                                "outputs": {
                                    "result_id": "sam3-roi",
                                    "detections": [],
                                },
                            },
                        },
                    }
                ]
            },
        )
    )

    assert memory.pending_reference_localization() is None


def test_object_memory_handler_returns_point_and_creates_point_obligation(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "scene.png"
    Image.new("RGB", (64, 48), "gray").save(scene)

    def reference_bytes(color: str) -> bytes:
        buffer = BytesIO()
        Image.new("RGB", (24, 24), color).save(buffer, format="PNG")
        return buffer.getvalue()

    class Client:
        def retrieve(self, *, environment: str, target_object: str):
            assert environment == "openeta/libero_libero_object_task0-v0"
            assert target_object == "alphabet soup"
            return ObjectMemoryBundle(
                query_key="libero/alphabet_soup",
                namespace="libero",
                asset_id="alphabet_soup",
                label="alphabet soup",
                references=tuple(
                    ObjectMemoryReference(view, f"view_{view}.png", reference_bytes(color))
                    for view, color in (("front", "red"), ("side", "green"), ("top", "blue"))
                ),
                manifest={"key": "libero/alphabet_soup"},
            )

    class Localizer:
        def localize(self, **kwargs):
            assert kwargs["image_size"] == (64, 48)
            assert len(kwargs["reference_images"]) == 3
            return ReferencePointLocalization(
                x=22.0,
                y=31.0,
                bbox_xyxy=(16.0, 22.0, 28.0, 40.0),
                confidence=0.9,
                reason="matching label",
                provider="fixture",
                model="fixture-vlm",
                details={"isolated_context": True},
            )

    result = build_object_memory_reference_handler(
        Client(),
        Localizer(),
        output_root=tmp_path / "outputs",
    )(
        _context(
            {
                "environment": "openeta/libero_libero_object_task0-v0",
                "target_object": "alphabet soup",
                "scene_image": str(scene),
            },
            session_id="point-session",
        )
    )

    assert result.success is True
    outputs = result.details["outputs"]
    assert outputs["positive_points"] == [{"x": 22.0, "y": 31.0, "label": 1}]
    assert outputs["bbox_xyxy"] == [16.0, 22.0, 28.0, 40.0]
    assert outputs["resolved_asset_key"] == "libero/alphabet_soup"
    assert outputs["localization_bundle"]["bbox_xyxy"] == outputs["bbox_xyxy"]
    assert Path(outputs["marked_scene_image"]).is_file()
    assert "point-session" in Path(outputs["marked_scene_image"]).parts
    assert len(outputs["reference_images"]) == 3

    memory = AgentMemory()
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "retrieve_asset_reference",
                        "result": {"success": True, "details": result.details},
                    }
                ]
            },
        )
    )
    pending = memory.pending_reference_localization()
    assert pending is not None
    assert pending["required_parameter"] == "positive_points"
    assert pending["positive_points"] == outputs["positive_points"]
    assert pending["bbox_xyxy"] == outputs["bbox_xyxy"]
    assert memory.target_asset_reference()["bbox_xyxy"] == outputs["bbox_xyxy"]
    assert (
        memory.target_asset_reference()["resolved_asset_key"]
        == "libero/alphabet_soup"
    )
    assert memory.detection_selection_gate_error(
        tool_name="sam3",
        parameters={"image": str(scene), "positive_points": [{"x": 23, "y": 31, "label": 1}]},
    )
    assert (
        memory.detection_selection_gate_error(
            tool_name="sam3",
            parameters={"image": str(scene), "positive_points": outputs["positive_points"]},
        )
        is None
    )


def test_molmopoint_result_creates_exact_sam3_point_obligation() -> None:
    scene = "/tmp/current-scene.png"
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "sam3-empty-1",
            "target_prompt": "alphabet soup",
            "source_image": "/tmp/previous-scene.png",
            "candidate_count": 0,
        },
        source="sam3",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "name": "retrieve_asset_reference",
                    "parameters": {
                        "target_object": "alphabet soup",
                        "scene_image": "/tmp/previous-scene.png",
                    },
                },
                "tool_calls": [
                    {
                        "name": "retrieve_asset_reference",
                        "result": {"success": False},
                    }
                ],
            },
        )
    )
    assert memory.detection_selection_gate_error(
        tool_name="retrieve_asset_reference",
        parameters={
            "target_object": "alphabet soup",
            "scene_image": "/tmp/previous-scene.png",
        },
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "molmopoint",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "image_sources": [scene],
                                    "points": [
                                        {
                                            "image_index": 0,
                                            "pixel_x": 276.25,
                                            "pixel_y": 343.5,
                                        }
                                    ],
                                }
                            },
                        },
                    }
                ]
            },
        )
    )

    points = [{"x": 276.25, "y": 343.5, "label": 1}]
    pending = memory.pending_reference_localization()
    assert pending is not None
    assert pending["scene_image"] == scene
    assert pending["positive_points"] == points
    assert pending["localization_bundle"]["source"] == "molmopoint"
    assert memory.detection_selection_gate_error(
        tool_name="retrieve_asset_reference",
        parameters={},
    )
    assert (
        memory.detection_selection_gate_error(
            tool_name="sam3",
            parameters={"image": scene, "positive_points": points},
        )
        is None
    )


def test_object_memory_handler_returns_structured_search_ambiguity(tmp_path: Path) -> None:
    scene = tmp_path / "scene.png"
    Image.new("RGB", (64, 48), "gray").save(scene)
    candidates = (
        ObjectMemorySearchCandidate(
            key="libero/akita_black_bowl",
            namespace="libero",
            asset_id="akita_black_bowl",
            label="bowl",
            aliases=("black bowl",),
            score=0.90,
            match_type="fuzzy",
        ),
        ObjectMemorySearchCandidate(
            key="libero/stone_black_bowl",
            namespace="libero",
            asset_id="stone_black_bowl",
            label="bowl",
            aliases=("black bowl",),
            score=0.85,
            match_type="fuzzy",
        ),
    )

    class Client:
        def resolve(self, **_kwargs):
            raise ObjectMemoryResolutionError(
                "top candidates are ambiguous",
                code="ambiguous_candidates",
                candidates=candidates,
            )

    class Localizer:
        def localize(self, **_kwargs):
            raise AssertionError("ambiguous search must not invoke localization")

    result = build_object_memory_reference_handler(
        Client(),
        Localizer(),
        output_root=tmp_path / "outputs",
    )(
        _context(
            {
                "environment": "openeta/libero_spatial_task0-v0",
                "target_object": "black bowl",
                "scene_image": str(scene),
            },
        )
    )

    assert result.success is False
    outputs = result.details["outputs"]
    assert outputs["reason"] == "object_memory_resolution_failed"
    assert outputs["resolution_code"] == "ambiguous_candidates"
    assert [candidate["key"] for candidate in outputs["search_candidates"]] == [
        "libero/akita_black_bowl",
        "libero/stone_black_bowl",
    ]
