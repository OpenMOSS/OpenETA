"""Backend-neutral grasp previews and an isolated visual selection advisor."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageFont, ImageOps

from adapter.protocol import JsonDict
from agent.backends.planner import PlannerBackend, PlannerBackendRequest


GRASP_SELECTION_BUNDLE_SCHEMA = "openeta.grasp_selection_bundle.v1"
GRASP_SELECTION_ADVICE_SCHEMA = "openeta.grasp_selection_advice.v1"
GRASP_SELECTION_RENDERER_VERSION = "backend-normalized-contact-support-overlay-v5"
GRASP_SELECTION_MAX_CANDIDATES = 20
GRASP_SELECTION_CANDIDATES_PER_SHEET = 4
# Hidden reasoning and the visible JSON share one completion limit on some
# OpenAI-compatible vision models.  The advisor normally stops far below this
# ceiling, while the extra room prevents a reasoning-only truncation from being
# misread as an abstention.
GRASP_POSE_ADVISOR_MAX_OUTPUT_TOKENS = 8192
GRASP_POSE_ADVISOR_MAX_VISION_IMAGES = 6

_PALETTE = (
    (0, 255, 220, 255),
    (255, 203, 67, 255),
    (255, 99, 132, 255),
    (92, 179, 255, 255),
    (199, 125, 255, 255),
    (92, 230, 130, 255),
    (255, 145, 77, 255),
    (225, 225, 225, 255),
)

GRASP_POSE_ADVISOR_SYSTEM_PROMPT = """You are an isolated OpenETA grasp-pose advisor.
You receive one target-object RGB image with labelled grasp overlays and one or
more contact sheets. Every candidate has already passed the host's physical
width and schema checks. Recommend a visually robust grasp; do not execute it.

Judge geometry rather than backend rank alone. Your primary objective is to
select the grasp most likely to remain secure through closing, lift, transport,
and ordinary direction changes without the object slipping or falling. Prefer
deep, centred, opposing contacts over a broad load-bearing part of the object,
with useful jaw overlap and the apparent centre of mass supported between or
below the fingers. Treat collision-free approach and table/neighbour clearance
as necessary feasibility constraints, but do not trade away grasp stability for
a merely convenient approach among otherwise executable candidates.

The filled dot and jaw line are centred on `executed_contact_center_xyz`, the
backend-normalized point that OpenETA will actually compile onto the calibrated
EEF/grip site. The arrow ends at that same executed contact centre. Native
`translation_xyz` and `gripper_tip_position_xyz` use backend-specific origins;
rank the normalized rendered centre rather than assuming either raw field has
one universal meaning.

Each candidate may also include `target_support_3d`, computed read-only from the
same selected target mask and aligned sensor depth. Use it to correct misleading
2-D overlap. `closing_span_m` estimates the visible target span along the jaw
closing axis; a span larger than `gripper_width_m` is strong evidence that the
jaws cannot surround that cross-section. `closing_center_offset_half_span` and
`binormal_center_offset_half_span` measure how far the executed centre is from
the visible target core in units of half-span; values above 1 place it outside
that core. `approach_surface_interval_m` reports the visible target's p05-p95
range along the approach axis relative to the executed centre. A positive p05
means even the nearest robust visible surface remains in front of the executed
centre; heed the corresponding before-surface risk instead of calling such a
contact deep. A negative p95 means the centre lies beyond the visible target.
These are partial-view measurements, not host rejections, but do not
call a candidate centred or broad-body when its 3-D evidence contradicts that.
Prefer candidates with opposing visible support, feasible closing span, and low
offsets. If every candidate has serious 3-D support risks, abstain.

Penalize shallow, tangential, edge, corner, rim, cap, neck, handle-tip, or tapered
shoulder contacts that can squeeze the object out or lose purchase during lift.
For upright bottles, cans, and cartons, normally prefer opposing side contacts
on the broad middle body rather than the cap, neck, top rim, or shoulder. Make an
exception only when the rendered geometry provides concrete evidence that the
alternative is more stable. Do not recommend the least-bad candidate merely
because every candidate is executable: if every rendered contact is confined to
an unstable rim, cap, neck, tapered shoulder, extreme edge, or similarly shallow
region, you MUST abstain so the main Agent can obtain a different view or a new
candidate set. A deeper insertion value alone does not rescue a contact whose
executed jaw centre is visibly on an unstable object region. Also abstain when
the 2-D projection or occlusion does not support a reliable transport-stability
comparison. Treat the coloured mask and rendered gripper as geometric evidence,
not object appearance. Never invent a candidate id and never output a tool call
or task-stage instruction.

Return exactly one JSON object:
{
  "decision":"recommend|abstain",
  "recommended_candidate_id":"exact id or empty",
  "alternatives":["exact id"],
  "confidence":0.0,
  "reasons":["concise visual reason"],
  "rejected":{"exact id":"concise visual reason"},
  "uncertainties":["concise uncertainty"]
}
For abstain, use an empty recommended_candidate_id and confidence at most 0.5.
"""


class GraspPoseAdvisor(Protocol):
    """Read-only boundary used after host filtering and final candidate ids."""

    def advise(self, selection_bundle: Mapping[str, Any], *, task: str) -> JsonDict:
        """Return one structured recommendation without activating a candidate."""


class BackendGraspPoseAdvisor:
    """Clean-context visual advisor backed by a dedicated planner client."""

    def __init__(self, backend: PlannerBackend) -> None:
        self.backend = backend

    def advise(self, selection_bundle: Mapping[str, Any], *, task: str) -> JsonDict:
        candidate_values = selection_bundle.get("candidates")
        candidates = (
            [dict(value) for value in candidate_values if isinstance(value, Mapping)]
            if isinstance(candidate_values, list)
            else []
        )
        candidate_ids = [str(value.get("candidate_id") or "") for value in candidates]
        if not candidate_ids or any(not value for value in candidate_ids):
            raise ValueError("selection bundle has no valid candidate ids")

        overview_ref = str(selection_bundle.get("overview_ref") or "")
        sheet_values = selection_bundle.get("contact_sheet_refs")
        sheet_refs = (
            [str(value) for value in sheet_values if isinstance(value, str) and value]
            if isinstance(sheet_values, list)
            else []
        )
        vision_paths = [value for value in [overview_ref, *sheet_refs] if value]
        started = time.monotonic()
        result = self.backend.decide(
            PlannerBackendRequest(
                system_prompt=GRASP_POSE_ADVISOR_SYSTEM_PROMPT,
                tool_context={
                    "schema_version": GRASP_SELECTION_ADVICE_SCHEMA,
                    "role": "read_only_grasp_pose_advisor",
                    "task": task,
                    "selection_bundle_id": selection_bundle.get("bundle_id"),
                    "source_backend": selection_bundle.get("source_backend"),
                    "scene_epoch": selection_bundle.get("scene_epoch"),
                    "candidate_count": len(candidates),
                    "candidates": candidates,
                    "image_order": [
                        {
                            "image_number": index,
                            "role": "overview" if index == 1 else "contact_sheet",
                            "path": path,
                        }
                        for index, path in enumerate(vision_paths, start=1)
                    ],
                    "vision_image_paths": vision_paths,
                    "vision_evidence": [
                        {
                            "path": path,
                            "role": (
                                "grasp_candidate_overview"
                                if index == 0
                                else "grasp_candidate_contact_sheet"
                            ),
                            "derived": True,
                        }
                        for index, path in enumerate(vision_paths)
                    ],
                },
                metadata={"isolated_context": True},
            )
        )
        payload = result.payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ValueError("grasp pose advisor returned invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("grasp pose advisor must return one JSON object")
        forbidden = {"tool", "tool_call", "action", "parameters", "compile_grasp_seed"}
        if forbidden.intersection(payload):
            raise ValueError("grasp pose advisor returned forbidden action fields")

        decision = str(payload.get("decision") or "").strip().lower()
        if decision not in {"recommend", "abstain"}:
            raise ValueError("grasp pose advisor returned an invalid decision")
        reference_map = _candidate_reference_map(candidates)
        raw_recommended = str(payload.get("recommended_candidate_id") or "").strip()
        recommended = reference_map.get(raw_recommended, "")
        if decision == "recommend" and not recommended:
            raise ValueError("grasp pose advisor recommended an unknown candidate")
        if decision == "abstain" and raw_recommended:
            raise ValueError("abstaining grasp pose advisor cannot recommend a candidate")
        confidence = _finite_float(payload.get("confidence"))
        if confidence is None or not 0.0 <= confidence <= 1.0:
            raise ValueError("grasp pose advisor returned invalid confidence")
        if decision == "abstain" and confidence > 0.5:
            raise ValueError("abstaining grasp pose advisor confidence exceeds 0.5")

        validation_warnings: list[JsonDict] = []
        raw_alternatives = _validated_string_list(
            payload.get("alternatives"), "alternatives"
        )
        alternatives: list[str] = []
        for reference in raw_alternatives:
            resolved = reference_map.get(reference, "")
            if not resolved:
                validation_warnings.append(
                    {"field": "alternatives", "reference": reference, "reason": "unknown"}
                )
                continue
            if resolved not in alternatives:
                alternatives.append(resolved)
        if recommended and recommended in alternatives:
            raise ValueError("recommended candidate cannot also be an alternative")
        reasons = _validated_string_list(payload.get("reasons"), "reasons")
        if decision == "recommend" and not reasons:
            raise ValueError("grasp pose advisor recommendation requires reasons")
        uncertainties = _validated_string_list(
            payload.get("uncertainties"), "uncertainties"
        )
        rejected_value = payload.get("rejected")
        if rejected_value is None:
            rejected_value = {}
        if not isinstance(rejected_value, Mapping):
            raise ValueError("grasp pose advisor rejected must be an object")
        rejected: JsonDict = {}
        for raw_id, raw_reason in rejected_value.items():
            reference = str(raw_id).strip()
            candidate_id = reference_map.get(reference, "")
            reason = str(raw_reason).strip()
            if not candidate_id:
                validation_warnings.append(
                    {"field": "rejected", "reference": reference, "reason": "unknown"}
                )
                continue
            if not reason:
                validation_warnings.append(
                    {
                        "field": "rejected",
                        "reference": reference,
                        "reason": "empty_reason",
                    }
                )
                continue
            rejected[candidate_id] = reason[:500]

        details = result.details if isinstance(result.details, dict) else {}
        advice: JsonDict = {
            "schema_version": GRASP_SELECTION_ADVICE_SCHEMA,
            "status": "completed",
            "decision": decision,
            "recommended_candidate_id": recommended,
            "alternatives": alternatives[:5],
            "confidence": confidence,
            "reasons": reasons[:6],
            "rejected": rejected,
            "uncertainties": uncertainties[:6],
            "bundle_id": selection_bundle.get("bundle_id"),
            "advisor_role": "read_only_grasp_pose_advisor",
            "provider": result.provider,
            "model": result.model,
            "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
            "isolated_context": True,
        }
        if isinstance(details.get("usage"), Mapping):
            advice["usage"] = dict(details["usage"])
        for key in ("provider_role", "provider_failover", "provider_switch_count"):
            if key in details:
                advice[key] = details[key]
        if validation_warnings:
            advice["validation_warnings"] = validation_warnings
        return advice


def build_grasp_selection_bundle(
    details: Mapping[str, Any],
    *,
    output_root: str | Path,
) -> tuple[JsonDict, list[JsonDict]]:
    """Render final host-executable candidates into stable static artifacts."""

    candidates_value = details.get("grasp_candidates")
    candidates = (
        [dict(value) for value in candidates_value if isinstance(value, Mapping)]
        if isinstance(candidates_value, list)
        else []
    )
    if not candidates:
        raise ValueError("missing executable grasp candidates")
    if len(candidates) > GRASP_SELECTION_MAX_CANDIDATES:
        raise ValueError("too many executable grasp candidates")
    result_id = str(details.get("result_id") or "").strip()
    source_rgb = str(details.get("source_rgb") or "").strip()
    object_mask = str(details.get("object_mask") or "").strip()
    source = details.get("source")
    source = dict(source) if isinstance(source, Mapping) else {}
    intrinsics = source.get("intrinsics")
    if not result_id or not source_rgb or not isinstance(intrinsics, Mapping):
        raise ValueError("incomplete grasp selection source")
    source_path = Path(source_rgb)
    if not source_path.is_file():
        raise ValueError("grasp selection source RGB is unavailable")

    output_dir = Path(output_root) / result_id
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        rgb = image.convert("RGBA")
    mask = _load_mask(object_mask, size=rgb.size)
    target_points = _load_target_points_camera(
        mask_path=object_mask,
        depth_path=str(source.get("depth") or details.get("source_depth") or ""),
        intrinsics=intrinsics,
        image_size=rgb.size,
    )
    projected = [
        _project_candidate(
            candidate,
            intrinsics=intrinsics,
            image_size=rgb.size,
            target_points_camera=target_points,
        )
        for candidate in candidates
    ]
    roi = _selection_roi(mask, projected=projected, image_size=rgb.size)

    overview_ref = output_dir / "overview.png"
    _render_overview(rgb, mask=mask, projected=projected, output_path=overview_ref)
    sheet_refs: list[Path] = []
    for start in range(0, len(projected), GRASP_SELECTION_CANDIDATES_PER_SHEET):
        page = projected[start : start + GRASP_SELECTION_CANDIDATES_PER_SHEET]
        sheet_ref = output_dir / f"contact_sheet_{start // GRASP_SELECTION_CANDIDATES_PER_SHEET + 1:02d}.png"
        _render_contact_sheet(rgb, mask=mask, projected=page, roi=roi, output_path=sheet_ref)
        sheet_refs.append(sheet_ref)

    candidate_summaries = [
        {
            "candidate_id": value["candidate_id"],
            "display_label": value["display_label"],
            "rank": value["rank"],
            "backend_candidate_id": value.get("backend_candidate_id"),
            "score": value["score"],
            "width_m": value["width_m"],
            "depth_m": value["depth_m"],
            "height_m": value["height_m"],
            "translation_xyz": value["translation_xyz"],
            "gripper_tip_position_xyz": value["gripper_tip_position_xyz"],
            "execution_reference_point": value["execution_reference_point"],
            "executed_contact_center_xyz": value["executed_contact_center_xyz"],
            "target_support_3d": value.get("target_support_3d"),
        }
        for value in projected
    ]
    digest_payload = {
        "renderer_version": GRASP_SELECTION_RENDERER_VERSION,
        "result_id": result_id,
        "source_rgb": str(source_path.resolve()),
        "scene_epoch": details.get("scene_epoch"),
        "candidate_ids": [value["candidate_id"] for value in projected],
    }
    digest = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:20]
    bundle: JsonDict = {
        "schema_version": GRASP_SELECTION_BUNDLE_SCHEMA,
        "bundle_id": f"grasp-selection:{digest}",
        "result_id": result_id,
        "source_backend": details.get("selected_backend"),
        "source_rgb": str(source_path.resolve()),
        "object_mask": str(Path(object_mask).resolve()) if object_mask else None,
        "camera_frame_id": details.get("camera_frame_id"),
        "scene_epoch": details.get("scene_epoch"),
        "renderer_version": GRASP_SELECTION_RENDERER_VERSION,
        "candidate_count": len(candidate_summaries),
        "candidates": candidate_summaries,
        "overview_ref": str(overview_ref.resolve()),
        "contact_sheet_refs": [str(path.resolve()) for path in sheet_refs],
        "visual_semantics": {
            "mask": "translucent red target region",
            "approach": "arrow ending at the host-executed grasp/contact center",
            "closing_axis": "jaw line centered on the host-executed grasp origin",
            "contact_center": (
                "filled dot at backend-normalized executed_contact_center_xyz, the "
                "pose compiled onto the calibrated EEF/grip site"
            ),
            "target_support_3d": (
                "read-only p05-p95 visible target span and executed-centre offsets "
                "in each candidate grasp frame, reconstructed from selected-mask depth"
            ),
            "projection_limit": "2-D camera projection; occluded depth remains uncertain",
        },
    }
    bundle_ref = output_dir / "selection_bundle.json"
    bundle["bundle_ref"] = str(bundle_ref.resolve())
    bundle_ref.write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    artifacts: list[JsonDict] = [
        {
            "type": "grasp_selection_overview",
            "kind": "image",
            "tool": "grasp_pose_estimate",
            "label": "executable grasp candidate overview",
            "path": str(overview_ref.resolve()),
            "bundle_id": bundle["bundle_id"],
        },
        *[
            {
                "type": "grasp_selection_contact_sheet",
                "kind": "image",
                "tool": "grasp_pose_estimate",
                "label": f"grasp candidate contact sheet {index}",
                "path": str(path.resolve()),
                "bundle_id": bundle["bundle_id"],
            }
            for index, path in enumerate(sheet_refs, start=1)
        ],
        {
            "type": "grasp_selection_bundle",
            "kind": "json",
            "tool": "grasp_pose_estimate",
            "label": "complete grasp selection evidence",
            "path": str(bundle_ref.resolve()),
            "bundle_id": bundle["bundle_id"],
        },
    ]
    return bundle, artifacts


def _project_candidate(
    candidate: Mapping[str, Any],
    *,
    intrinsics: Mapping[str, Any],
    image_size: tuple[int, int],
    target_points_camera: Any = None,
) -> JsonDict:
    candidate_id = str(candidate.get("id") or "").strip()
    rank = candidate.get("rank")
    score = _finite_float(candidate.get("score"))
    translation = _finite_vector(candidate.get("translation_xyz"), length=3)
    rotation = _finite_matrix(candidate.get("rotation_matrix"), rows=3, columns=3)
    tip = _finite_vector(candidate.get("gripper_tip_position_xyz"), length=3)
    execution_reference_point = str(
        candidate.get("execution_reference_point") or "translation_xyz"
    )
    contact_center = _finite_vector(
        candidate.get("execution_contact_center_xyz"),
        length=3,
    )
    if contact_center is None:
        contact_center = tip if execution_reference_point == "gripper_tip_position_xyz" else translation
    depth = _finite_float(candidate.get("depth"))
    if depth is None:
        depth = _finite_float(candidate.get("gripper_depth"))
    width = _finite_float(candidate.get("width"))
    height = _finite_float(candidate.get("height")) or 0.03
    if (
        not candidate_id
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or score is None
        or translation is None
        or rotation is None
        or tip is None
        or depth is None
        or width is None
        or width <= 0
        or depth <= 0
    ):
        raise ValueError("invalid executable grasp candidate geometry")
    approach_axis = [rotation[row][0] for row in range(3)]
    closing_axis = [rotation[row][1] for row in range(3)]
    binormal_axis = [rotation[row][2] for row in range(3)]
    # Both supported backends publish a model-native base/centre plus an
    # explicit physical tip. Draw the exact normalized point the compiler will
    # execute while retaining each backend's incoming sweep semantics.
    approach_start = (
        list(translation)
        if execution_reference_point == "gripper_tip_position_xyz"
        else [translation[row] - depth * approach_axis[row] for row in range(3)]
    )
    jaw_a = [
        contact_center[row] - width * 0.5 * closing_axis[row] for row in range(3)
    ]
    jaw_b = [
        contact_center[row] + width * 0.5 * closing_axis[row] for row in range(3)
    ]
    height_a = [
        contact_center[row] - height * 0.5 * binormal_axis[row] for row in range(3)
    ]
    height_b = [
        contact_center[row] + height * 0.5 * binormal_axis[row] for row in range(3)
    ]
    points = [approach_start, contact_center, jaw_a, jaw_b, height_a, height_b]
    pixels = _project_points(points, intrinsics=intrinsics, image_size=image_size)
    projected = {
        "candidate_id": candidate_id,
        "display_label": f"#{rank + 1}",
        "rank": rank,
        "backend_candidate_id": candidate.get("backend_candidate_id"),
        "score": score,
        "width_m": width,
        "depth_m": depth,
        "height_m": height,
        "translation_xyz": translation,
        "gripper_tip_position_xyz": tip,
        "execution_reference_point": execution_reference_point,
        "executed_contact_center_xyz": contact_center,
        "pixels": pixels,
    }
    support = _target_support_3d(
        target_points_camera,
        contact_center=contact_center,
        rotation=rotation,
        gripper_width_m=width,
    )
    if support is not None:
        projected["target_support_3d"] = support
    return projected


def _load_target_points_camera(
    *,
    mask_path: str,
    depth_path: str,
    intrinsics: Mapping[str, Any],
    image_size: tuple[int, int],
) -> Any:
    """Reconstruct selected visible target points without changing candidates."""

    if not mask_path or not depth_path:
        return None
    fx = _finite_float(intrinsics.get("fx"))
    fy = _finite_float(intrinsics.get("fy"))
    cx = _finite_float(intrinsics.get("cx"))
    cy = _finite_float(intrinsics.get("cy"))
    scale = _finite_float(intrinsics.get("scale"))
    if (
        fx is None
        or fy is None
        or cx is None
        or cy is None
        or scale is None
        or fx <= 0
        or fy <= 0
        or scale <= 0
    ):
        return None
    try:
        import numpy as np

        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 0
        with Image.open(depth_path) as depth_image:
            depth = np.asarray(depth_image, dtype=np.float64) / scale
    except (OSError, ValueError):
        return None
    width, height = image_size
    if mask.shape != (height, width) or depth.shape != (height, width):
        return None
    valid = mask & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(valid)
    if len(rows) < 100:
        return None
    z = depth[rows, columns]
    points = np.column_stack(
        (
            (columns.astype(np.float64) - cx) * z / fx,
            (rows.astype(np.float64) - cy) * z / fy,
            z,
        )
    )
    return points if np.isfinite(points).all() else None


def _target_support_3d(
    target_points_camera: Any,
    *,
    contact_center: Sequence[float],
    rotation: Sequence[Sequence[float]],
    gripper_width_m: float,
) -> JsonDict | None:
    """Describe visible target support in the candidate-local grasp frame.

    This intentionally emits evidence rather than a score or selection.  The
    p05-p95 interval reduces single-pixel depth noise while keeping enough of
    the visible object cross-section to expose shallow and over-wide contacts.
    """

    if target_points_camera is None:
        return None
    try:
        import numpy as np

        points = np.asarray(target_points_camera, dtype=np.float64)
        center = np.asarray(contact_center, dtype=np.float64)
        basis = np.asarray(rotation, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 100
            or center.shape != (3,)
            or basis.shape != (3, 3)
            or not np.isfinite(points).all()
            or not np.isfinite(center).all()
            or not np.isfinite(basis).all()
        ):
            return None
        local = (points - center) @ basis
        q05, q50, q95 = np.quantile(local, [0.05, 0.50, 0.95], axis=0)
    except (TypeError, ValueError):
        return None

    def axis_evidence(index: int) -> tuple[float, float]:
        low = float(q05[index])
        high = float(q95[index])
        span = max(0.0, high - low)
        midpoint = 0.5 * (low + high)
        offset = abs(midpoint) / max(0.5 * span, 1e-6)
        return span, offset

    approach_span, approach_offset = axis_evidence(0)
    closing_span, closing_offset = axis_evidence(1)
    binormal_span, binormal_offset = axis_evidence(2)
    width_ratio = closing_span / max(float(gripper_width_m), 1e-6)
    risks: list[str] = []
    if width_ratio > 1.05:
        risks.append("visible_closing_span_exceeds_gripper_width")
    if closing_offset > 1.0:
        risks.append("executed_center_outside_visible_closing_core")
    if binormal_offset > 1.0:
        risks.append("executed_center_outside_visible_binormal_core")
    # Five millimetres exceeds ordinary aligned-depth noise while retaining
    # the metric as advisor evidence rather than a host execution gate.
    if float(q05[0]) > 0.005:
        risks.append("executed_center_before_visible_target_surface")
    if float(q95[0]) < -0.005:
        risks.append("executed_center_beyond_visible_target_surface")
    return {
        "schema_version": "openeta.grasp_target_support_3d.v1",
        "source": "selected_mask_aligned_sensor_depth",
        "visible_point_count": int(len(points)),
        "quantile_interval": "p05_p95",
        "closing_span_m": round(closing_span, 6),
        "gripper_width_m": round(float(gripper_width_m), 6),
        "closing_span_to_gripper_width": round(width_ratio, 4),
        "closing_center_offset_half_span": round(closing_offset, 4),
        "binormal_span_m": round(binormal_span, 6),
        "binormal_center_offset_half_span": round(binormal_offset, 4),
        "approach_span_m": round(approach_span, 6),
        "approach_center_offset_half_span": round(approach_offset, 4),
        "approach_surface_interval_m": [
            round(float(q05[0]), 6),
            round(float(q95[0]), 6),
        ],
        "approach_surface_median_offset_m": round(float(q50[0]), 6),
        "risk_indicators": risks,
        "interpretation": (
            "Partial-view geometric evidence for advisor/Agent comparison only; "
            "it does not auto-select or reject a grasp."
        ),
    }


def _project_points(
    points: Sequence[Sequence[float]],
    *,
    intrinsics: Mapping[str, Any],
    image_size: tuple[int, int],
) -> list[list[float] | None]:
    fx = _positive_float(intrinsics.get("fx"), "fx")
    fy = _positive_float(intrinsics.get("fy"), "fy")
    cx = _finite_float(intrinsics.get("cx"))
    cy = _finite_float(intrinsics.get("cy"))
    if cx is None or cy is None:
        raise ValueError("invalid camera intrinsics")
    width, height = image_size
    projected: list[list[float] | None] = []
    for point in points:
        x, y, z = point
        if z <= 1e-6:
            projected.append(None)
            continue
        u = fx * x / z + cx
        v = fy * y / z + cy
        if not math.isfinite(u) or not math.isfinite(v):
            projected.append(None)
            continue
        # Preserve modest off-image geometry for diagnostics but bound drawing.
        u = min(max(u, -width), 2.0 * width)
        v = min(max(v, -height), 2.0 * height)
        projected.append([u, v])
    return projected


def _load_mask(path: str, *, size: tuple[int, int]) -> Image.Image | None:
    if not path:
        return None
    mask_path = Path(path)
    if not mask_path.is_file():
        raise ValueError("grasp selection object mask is unavailable")
    with Image.open(mask_path) as image:
        mask = image.convert("L")
    if mask.size != size:
        raise ValueError("grasp selection RGB and mask dimensions differ")
    return mask.point(lambda value: 255 if value > 0 else 0)


def _selection_roi(
    mask: Image.Image | None,
    *,
    projected: Sequence[Mapping[str, Any]],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    width, height = image_size
    bbox = mask.getbbox() if mask is not None else None
    if bbox is None:
        valid = [
            pixel
            for candidate in projected
            for pixel in candidate.get("pixels", [])
            if isinstance(pixel, list)
        ]
        if valid:
            xs = [float(value[0]) for value in valid]
            ys = [float(value[1]) for value in valid]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1)
        else:
            bbox = (0, 0, width, height)
    left, top, right, bottom = bbox
    padding = max(24, int(0.22 * max(right - left, bottom - top, 1)))
    return (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding),
        min(height, bottom + padding),
    )


def _mask_overlay(rgb: Image.Image, mask: Image.Image | None) -> Image.Image:
    canvas = rgb.copy()
    if mask is None:
        return canvas
    tint = Image.new("RGBA", rgb.size, (255, 40, 28, 0))
    tint.putalpha(mask.point(lambda value: 48 if value else 0))
    return Image.alpha_composite(canvas, tint)


def _render_overview(
    rgb: Image.Image,
    *,
    mask: Image.Image | None,
    projected: Sequence[Mapping[str, Any]],
    output_path: Path,
) -> None:
    canvas = _mask_overlay(rgb, mask)
    draw = ImageDraw.Draw(canvas)
    for candidate in reversed(projected):
        _draw_candidate(draw, candidate, label=True)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, canvas.width, 44), fill=(0, 0, 0, 195))
    draw.text(
        (12, 7),
        f"Executable grasp overview: {len(projected)} candidates",
        fill=(255, 255, 255, 255),
        font=font,
    )
    draw.text(
        (12, 25),
        "arrow=approach  line=jaws  dot=executed contact center  red=target mask",
        fill=(220, 220, 220, 255),
        font=font,
    )
    canvas.convert("RGB").save(output_path, format="PNG")


def _render_contact_sheet(
    rgb: Image.Image,
    *,
    mask: Image.Image | None,
    projected: Sequence[Mapping[str, Any]],
    roi: tuple[int, int, int, int],
    output_path: Path,
) -> None:
    panel_width, panel_height = 480, 360
    columns = 2
    rows = max(1, math.ceil(len(projected) / columns))
    sheet = Image.new("RGB", (panel_width * columns, panel_height * rows), (24, 24, 24))
    font = ImageFont.load_default()
    for index, candidate in enumerate(projected):
        canvas = _mask_overlay(rgb, mask)
        draw = ImageDraw.Draw(canvas)
        _draw_candidate(draw, candidate, label=True)
        crop = canvas.crop(roi).convert("RGB")
        fitted = ImageOps.contain(crop, (panel_width - 16, panel_height - 82))
        panel = Image.new("RGB", (panel_width, panel_height), (18, 18, 18))
        panel.paste(fitted, ((panel_width - fitted.width) // 2, 72))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((0, 0, panel_width, 70), fill=(0, 0, 0))
        panel_draw.text(
            (10, 6),
            f"{candidate['display_label']}  id={candidate['candidate_id']}",
            fill=(255, 255, 255),
            font=font,
        )
        support = candidate.get("target_support_3d")
        if isinstance(support, Mapping):
            span = _finite_float(support.get("closing_span_m"))
            ratio = _finite_float(support.get("closing_span_to_gripper_width"))
            closing_offset = _finite_float(
                support.get("closing_center_offset_half_span")
            )
            binormal_offset = _finite_float(
                support.get("binormal_center_offset_half_span")
            )
            if None not in {span, ratio, closing_offset, binormal_offset}:
                risk = bool(support.get("risk_indicators"))
                panel_draw.text(
                    (10, 46),
                    (
                        f"3D span={span:.3f}m ({ratio:.2f}x jaw)  "
                        f"core offsets=({closing_offset:.2f},{binormal_offset:.2f})"
                    ),
                    fill=(255, 125, 105) if risk else (150, 235, 170),
                    font=font,
                )
        panel_draw.text(
            (10, 26),
            (
                f"score={candidate['score']:.4f}  width={candidate['width_m']:.3f}m  "
                f"depth={candidate['depth_m']:.3f}m"
            ),
            fill=(215, 215, 215),
            font=font,
        )
        x = (index % columns) * panel_width
        y = (index // columns) * panel_height
        sheet.paste(panel, (x, y))
    sheet.save(output_path, format="PNG")


def _draw_candidate(
    draw: ImageDraw.ImageDraw,
    candidate: Mapping[str, Any],
    *,
    label: bool,
) -> None:
    pixels = candidate.get("pixels")
    if not isinstance(pixels, list) or len(pixels) != 6:
        return
    origin, tip, jaw_a, jaw_b, height_a, height_b = pixels
    color = _PALETTE[int(candidate.get("rank") or 0) % len(_PALETTE)]
    width = 5 if int(candidate.get("rank") or 0) == 0 else 3
    if isinstance(origin, list) and isinstance(tip, list):
        draw.line([tuple(origin), tuple(tip)], fill=color, width=width)
        _draw_arrow_head(draw, origin=origin, tip=tip, color=color, width=width)
    if isinstance(jaw_a, list) and isinstance(jaw_b, list):
        draw.line([tuple(jaw_a), tuple(jaw_b)], fill=color, width=width + 1)
        for jaw in (jaw_a, jaw_b):
            x, y = jaw
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), outline=(0, 0, 0, 255), fill=color)
    if isinstance(height_a, list) and isinstance(height_b, list):
        draw.line([tuple(height_a), tuple(height_b)], fill=color, width=max(1, width - 2))
    if isinstance(tip, list):
        x, y = tip
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), outline=(0, 0, 0, 255), fill=color)
        if label:
            draw.text(
                (x + 8, y - 18),
                str(candidate.get("display_label") or ""),
                fill=color,
                font=ImageFont.load_default(),
                stroke_width=2,
                stroke_fill=(0, 0, 0, 230),
            )


def _draw_arrow_head(
    draw: ImageDraw.ImageDraw,
    *,
    origin: Sequence[float],
    tip: Sequence[float],
    color: tuple[int, int, int, int],
    width: int,
) -> None:
    dx = float(tip[0]) - float(origin[0])
    dy = float(tip[1]) - float(origin[1])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return
    ux, uy = dx / norm, dy / norm
    px, py = -uy, ux
    length = 10.0
    spread = 5.0
    left = (tip[0] - length * ux + spread * px, tip[1] - length * uy + spread * py)
    right = (tip[0] - length * ux - spread * px, tip[1] - length * uy - spread * py)
    draw.line([tuple(tip), left], fill=color, width=width)
    draw.line([tuple(tip), right], fill=color, width=width)


def _validated_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"grasp pose advisor {label} must be a list")
    result = [str(item).strip() for item in value]
    if any(not item for item in result):
        raise ValueError(f"grasp pose advisor {label} contains an empty value")
    return [item[:500] for item in result]


def _candidate_reference_map(candidates: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    references: dict[str, str] = {}
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            continue
        references[candidate_id] = candidate_id
        for field in ("display_label", "backend_candidate_id"):
            alias = str(candidate.get(field) or "").strip()
            if alias:
                references.setdefault(alias, candidate_id)
    return references


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value)
    if result is None or result <= 0:
        raise ValueError(f"invalid camera intrinsics {label}")
    return result


def _finite_vector(value: Any, *, length: int) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    values = [_finite_float(item) for item in value]
    if len(values) != length or any(item is None for item in values):
        return None
    return [float(item) for item in values if item is not None]


def _finite_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
) -> list[list[float]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    matrix = [_finite_vector(row, length=columns) for row in value]
    if len(matrix) != rows or any(row is None for row in matrix):
        return None
    return [row for row in matrix if row is not None]
