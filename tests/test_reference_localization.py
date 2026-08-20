from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agent.backends.planner import CallablePlannerBackend, PlannerBackendResult
from agent.runtime.reference_localization import (
    REFERENCE_POINT_LOCALIZATION_SYSTEM_PROMPT,
    REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT,
    BackendReferencePointLocalizer,
)


def _images(tmp_path: Path) -> tuple[Path, list[Path]]:
    scene = tmp_path / "scene.png"
    Image.new("RGB", (64, 48), "gray").save(scene)
    references = []
    for view, color in (("front", "red"), ("side", "green"), ("top", "blue")):
        path = tmp_path / f"{view}.png"
        Image.new("RGB", (32, 32), color).save(path)
        references.append(path)
    return scene, references


def test_reference_point_localizer_uses_clean_four_image_context(tmp_path: Path) -> None:
    scene, references = _images(tmp_path)
    requests = []

    def decide(request):
        requests.append(request)
        if request.tool_context["role"] == "reference_point_verifier":
            return PlannerBackendResult(
                payload={
                    "decision": "match",
                    "confidence": 0.92,
                    "reference_geometry": "cylindrical pull-tab can",
                    "candidate_geometry": "cylindrical pull-tab can",
                    "grasp_geometry_family": "upright_can",
                    "geometry_match": True,
                    "matching_attributes": [
                        "blue and orange package colors",
                        "matching label layout",
                    ],
                    "conflicting_attributes": [],
                    "reason": "matching package colors and label layout",
                },
                provider="test-provider",
                model="test-vlm",
            )
        return PlannerBackendResult(
            payload={
                "decision": "locate",
                "point": {"x": 21.5, "y": 30},
                "bbox_xyxy": [16, 20, 28, 40],
                "confidence": 0.84,
                "reason": "matching package label and shape",
            },
            provider="test-provider",
            model="test-vlm",
        )

    result = BackendReferencePointLocalizer(CallablePlannerBackend(decide)).localize(
        environment="libero",
        target_object="alphabet soup",
        scene_image=scene,
        reference_images=references,
        image_size=(64, 48),
    )

    assert result.as_prompt_point() == {"x": 21.5, "y": 30.0, "label": 1}
    assert result.provider == "test-provider"
    assert len(requests) == 2
    request = requests[0]
    assert request.metadata["isolated_context"] is True
    assert request.tool_context["vision_image_paths"] == [
        str(scene),
        *(str(path) for path in references),
    ]
    assert [entry["role"] for entry in request.tool_context["image_order"]] == [
        "scene",
        "reference_view_1",
        "reference_view_2",
        "reference_view_3",
    ]
    verification = requests[1]
    assert verification.tool_context["role"] == "reference_point_verifier"
    assert verification.tool_context["candidate_point"] == {"x": 21.5, "y": 30.0}
    assert verification.tool_context["candidate_crop_box_xyxy"] == [12, 16, 32, 44]
    assert verification.tool_context["candidate_point_in_crop"] == {
        "x": 9.5,
        "y": 14.0,
    }
    assert Path(verification.tool_context["vision_image_paths"][0]).is_file()
    with Image.open(verification.tool_context["vision_image_paths"][0]) as crop:
        assert crop.size == (512, 512)
    assert result.details["attempt_count"] == 1
    assert result.details["verification"]["decision"] == "match"
    assert result.details["verification"]["grasp_geometry_family"] == "upright_can"


def test_reference_point_localizer_requires_exact_instance_attributes() -> None:
    assert "exact asset instance" in REFERENCE_POINT_LOCALIZATION_SYSTEM_PROMPT
    assert "compare at least two" in REFERENCE_POINT_LOCALIZATION_SYSTEM_PROMPT
    assert "tight box around exactly one" in REFERENCE_POINT_LOCALIZATION_SYSTEM_PROMPT
    assert "blue upper label and orange lower label" in (
        REFERENCE_POINT_LOCALIZATION_SYSTEM_PROMPT
    )
    assert "exact same asset" in REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    assert "blue-and-orange" in REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    assert "combining the color" in REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    assert "Physical package geometry is a hard gate" in (
        REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    )
    assert "rectangular carton, packet, or box" in (
        REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    )
    assert "mere presence" in REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    assert "partially occluding object is not a rejection" in (
        REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    )
    assert "grasp_geometry_family" in REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    assert "never relabel an object" in REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT
    assert "ordinal positions do not imply front, side, or top" in (
        REFERENCE_POINT_LOCALIZATION_SYSTEM_PROMPT
    )
    assert "ordinal positions do not imply" in REFERENCE_POINT_VERIFICATION_SYSTEM_PROMPT


def test_reference_point_localizer_excludes_rejected_candidate_and_retries(
    tmp_path: Path,
) -> None:
    scene, references = _images(tmp_path)
    requests = []
    proposal_count = 0

    def decide(request):
        nonlocal proposal_count
        requests.append(request)
        if request.tool_context["role"] == "reference_point_verifier":
            if request.tool_context["candidate_point"]["x"] == 20:
                return {
                    "decision": "reject",
                    "confidence": 0.99,
                    "reason": "candidate is red and green; references are blue and orange",
                }
            return {
                "decision": "match",
                "confidence": 0.9,
                "reference_geometry": "cylindrical can",
                "candidate_geometry": "cylindrical can",
                "geometry_match": True,
                "matching_attributes": ["label colors", "artwork"],
                "conflicting_attributes": [],
                "reason": "label colors and artwork match",
            }
        proposal_count += 1
        point = {"x": 20, "y": 30} if proposal_count == 1 else {"x": 50, "y": 12}
        bbox = [15, 24, 25, 38] if proposal_count == 1 else [45, 6, 56, 19]
        return {
            "decision": "locate",
            "point": point,
            "bbox_xyxy": bbox,
            "confidence": 0.95,
            "reason": "candidate instance",
        }

    result = BackendReferencePointLocalizer(CallablePlannerBackend(decide)).localize(
        environment="libero",
        target_object="alphabet soup",
        scene_image=scene,
        reference_images=references,
        image_size=(64, 48),
    )

    assert result.as_prompt_point() == {"x": 50.0, "y": 12.0, "label": 1}
    assert result.details["attempt_count"] == 2
    assert result.details["rejected_candidate_count"] == 1
    proposals = [
        request for request in requests if request.tool_context["role"] == "reference_point_localizer"
    ]
    rejected = proposals[1].tool_context["examined_candidates"][0]
    assert rejected["x"] == 20.0
    assert rejected["y"] == 30.0
    assert rejected["bbox_xyxy"] == [15.0, 24.0, 25.0, 38.0]
    assert "red and green" in rejected["reason"]
    assert ".excluded-" in rejected["audit_image"]
    assert proposals[1].tool_context["vision_image_paths"][0] == str(scene)


def test_reference_point_localizer_excludes_abstention_and_explores_elsewhere(
    tmp_path: Path,
) -> None:
    scene, references = _images(tmp_path)
    proposals = 0
    requests = []

    def decide(request):
        nonlocal proposals
        requests.append(request)
        if request.tool_context["role"] == "reference_point_verifier":
            if request.tool_context["candidate_point"]["x"] == 20:
                return {
                    "decision": "abstain",
                    "confidence": 0.1,
                    "reference_geometry": "cylindrical can",
                    "candidate_geometry": "blurred",
                    "grasp_geometry_family": "unknown",
                    "geometry_match": False,
                    "matching_attributes": ["blue upper label", "orange lower label"],
                    "conflicting_attributes": [],
                    "reason": "too blurred to confirm",
                }
            return {
                "decision": "match",
                "confidence": 0.9,
                "reference_geometry": "cylindrical can",
                "candidate_geometry": "cylindrical can",
                "grasp_geometry_family": "upright_can",
                "geometry_match": True,
                "matching_attributes": ["blue upper label", "orange lower label"],
                "conflicting_attributes": [],
                "reason": "geometry and label layout match",
            }
        proposals += 1
        if proposals == 1:
            return {
                "decision": "locate",
                "point": {"x": 20, "y": 20},
                "bbox_xyxy": [15, 14, 26, 28],
                "confidence": 0.88,
                "reason": "first appearance-supported candidate",
            }
        return {
            "decision": "locate",
            "point": {"x": 48, "y": 18},
            "bbox_xyxy": [43, 12, 54, 26],
            "confidence": 0.82,
            "reason": "different appearance-supported candidate",
        }

    result = BackendReferencePointLocalizer(CallablePlannerBackend(decide)).localize(
        environment="libero",
        target_object="alphabet soup",
        scene_image=scene,
        reference_images=references,
        image_size=(64, 48),
    )

    assert result.as_prompt_point() == {"x": 48.0, "y": 18.0, "label": 1}
    proposal_requests = [
        request
        for request in requests
        if request.tool_context["role"] == "reference_point_localizer"
    ]
    examined = proposal_requests[1].tool_context["examined_candidates"]
    assert examined[0]["disposition"] == "provisional_abstain"
    assert examined[0]["x"] == 20.0
    assert result.details["candidate_policy"] == "verified_match"


def test_reference_point_localizer_rejects_out_of_bounds_point(tmp_path: Path) -> None:
    scene, references = _images(tmp_path)
    backend = CallablePlannerBackend(
        lambda _request: {
            "decision": "locate",
            "point": {"x": 64, "y": 10},
            "bbox_xyxy": [58, 4, 64, 16],
            "confidence": 0.5,
            "reason": "bad coordinate",
        }
    )

    with pytest.raises(ValueError, match="outside"):
        BackendReferencePointLocalizer(backend).localize(
            environment="libero",
            target_object="alphabet soup",
            scene_image=scene,
            reference_images=references,
            image_size=(64, 48),
        )


def test_reference_point_localizer_requires_bbox_containing_point(
    tmp_path: Path,
) -> None:
    scene, references = _images(tmp_path)
    backend = CallablePlannerBackend(
        lambda _request: {
            "decision": "locate",
            "point": {"x": 20, "y": 10},
            "bbox_xyxy": [22, 4, 30, 16],
            "confidence": 0.5,
            "reason": "point and box disagree",
        }
    )

    with pytest.raises(ValueError, match="outside bbox_xyxy"):
        BackendReferencePointLocalizer(backend).localize(
            environment="libero",
            target_object="alphabet soup",
            scene_image=scene,
            reference_images=references,
            image_size=(64, 48),
        )


def test_reference_point_localizer_rejects_unstructured_match(
    tmp_path: Path,
) -> None:
    scene, references = _images(tmp_path)

    def decide(request):
        if request.tool_context["role"] == "reference_point_verifier":
            return {
                "decision": "match",
                "confidence": 0.99,
                "reason": "similar colors",
            }
        return {
            "decision": "locate",
            "point": {"x": 20, "y": 20},
            "bbox_xyxy": [15, 14, 26, 28],
            "confidence": 0.9,
            "reason": "candidate",
        }

    with pytest.raises(ValueError, match="structured exact-instance gate"):
        BackendReferencePointLocalizer(CallablePlannerBackend(decide)).localize(
            environment="libero",
            target_object="alphabet soup",
            scene_image=scene,
            reference_images=references,
            image_size=(64, 48),
        )


def test_reference_point_localizer_returns_provisional_seed_after_reviewer_abstains(
    tmp_path: Path,
) -> None:
    scene, references = _images(tmp_path)

    def decide(request):
        if request.tool_context["role"] == "reference_point_verifier":
            return {
                "decision": "abstain",
                "confidence": 0.0,
                "reference_geometry": "cylindrical can",
                "candidate_geometry": "too blurred to verify",
                "grasp_geometry_family": "unknown",
                "geometry_match": False,
                "matching_attributes": ["blue upper label", "orange lower label"],
                "conflicting_attributes": [],
                "reason": "the crop is too blurred for exact geometry verification",
            }
        return {
            "decision": "locate",
            "point": {"x": 20, "y": 20},
            "bbox_xyxy": [15, 14, 26, 28],
            "confidence": 0.88,
            "reason": "blue and orange candidate matching the reference layout",
        }

    result = BackendReferencePointLocalizer(CallablePlannerBackend(decide)).localize(
        environment="libero",
        target_object="alphabet soup",
        scene_image=scene,
        reference_images=references,
        image_size=(64, 48),
    )

    assert result.as_prompt_point() == {"x": 20.0, "y": 20.0, "label": 1}
    assert result.confidence == 0.5
    assert result.details["provisional"] is True
    assert result.details["requires_downstream_confirmation"] is True
    assert result.details["verification"]["decision"] == "abstain"
    assert "only a SAM3 seed" in result.reason


def test_reference_point_localizer_ranks_distinct_safe_provisional_candidates(
    tmp_path: Path,
) -> None:
    scene, references = _images(tmp_path)
    proposal_index = 0

    def decide(request):
        nonlocal proposal_index
        if request.tool_context["role"] == "reference_point_verifier":
            x = request.tool_context["candidate_point"]["x"]
            if x == 50:
                return {
                    "decision": "abstain",
                    "confidence": 0.0,
                    "grasp_geometry_family": "boxed_item",
                    "geometry_match": False,
                    "matching_attributes": ["blue", "orange"],
                    "conflicting_attributes": ["rectangular geometry"],
                    "reason": "geometry conflict",
                }
            return {
                "decision": "abstain",
                "confidence": 0.2,
                "grasp_geometry_family": "upright_can" if x == 30 else "unknown",
                "geometry_match": x == 30,
                "matching_attributes": ["blue label", "orange label"],
                "conflicting_attributes": [],
                "reason": "appearance supported but exact identity is blurred",
            }
        proposal_index += 1
        x = (10, 30, 50)[proposal_index - 1]
        return {
            "decision": "locate",
            "point": {"x": x, "y": 20},
            "bbox_xyxy": [x - 4, 14, x + 4, 28],
            "confidence": 0.95 if x == 10 else 0.7,
            "reason": f"candidate at {x}",
        }

    result = BackendReferencePointLocalizer(CallablePlannerBackend(decide)).localize(
        environment="libero",
        target_object="alphabet soup",
        scene_image=scene,
        reference_images=references,
        image_size=(64, 48),
    )

    assert result.as_prompt_point() == {"x": 30.0, "y": 20.0, "label": 1}
    assert result.details["candidate_policy"] == "ranked_provisional"
    assert [
        candidate["positive_points"][0]["x"]
        for candidate in result.details["ranked_candidates"]
    ] == [30.0, 10.0]
    assert result.details["examined_candidate_count"] == 3


def test_reference_point_localizer_does_not_provision_conflicting_abstention(
    tmp_path: Path,
) -> None:
    scene, references = _images(tmp_path)

    def decide(request):
        if request.tool_context["role"] == "reference_point_verifier":
            return {
                "decision": "abstain",
                "confidence": 0.0,
                "reference_geometry": "cylindrical can",
                "candidate_geometry": "rectangular carton",
                "grasp_geometry_family": "boxed_item",
                "geometry_match": False,
                "matching_attributes": ["blue color", "orange color"],
                "conflicting_attributes": ["rectangular package geometry"],
                "reason": "the package geometry conflicts with the reference can",
            }
        return {
            "decision": "locate",
            "point": {"x": 20, "y": 20},
            "bbox_xyxy": [15, 14, 26, 28],
            "confidence": 0.9,
            "reason": "color-similar candidate",
        }

    with pytest.raises(ValueError, match="exhausted exact-instance candidates"):
        BackendReferencePointLocalizer(CallablePlannerBackend(decide)).localize(
            environment="libero",
            target_object="alphabet soup",
            scene_image=scene,
            reference_images=references,
            image_size=(64, 48),
        )
