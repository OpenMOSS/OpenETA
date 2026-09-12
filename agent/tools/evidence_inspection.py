"""Bounded local evidence viewing; never perception inference or selection."""
from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from agent.runtime.tool_bundles import BUNDLE_INDEX_KEY, load_bundle
from agent.tools.registry import make_tool_result


def inspect_evidence(context, memory):
    from PIL import Image
    from agent.runtime.memory import _memory_fact_value, _tool_call_outputs
    from agent.tools.handlers import _build_sam3_selection_artifacts

    memory.facts.pop("evidence_inspection", None)
    try:
        params = context.parameters
        root = memory.artifact_root
        if root is None or not memory.session_id:
            raise ValueError("inspection requires an active session artifact root")
        root = Path(root).resolve()

        def owned_image(value):
            if not isinstance(value, str) or not value:
                raise ValueError("copy an exact saved image file path")
            path = Path(str(value)).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("image must be a regular file inside this session's artifact root")
            if path.stat().st_size > 16 * 1024 * 1024:
                raise ValueError("image exceeds the 16 MiB inspection limit")
            with Image.open(path) as picture:
                if picture.width * picture.height > 16_000_000:
                    raise ValueError("image exceeds the inspection pixel limit")
                picture.verify()
            return path

        if "image_ref" in params:
            if set(params) != {"image_ref"}:
                raise ValueError("image_ref cannot be mixed with bundle_id/offset")
            requested = params["image_ref"]
            evidence = {}
            if isinstance(requested, str) and requested.startswith(("observation:", "current_observation:")):
                from agent.runtime.visual_history import resolve_image_evidence_reference
                evidence = resolve_image_evidence_reference(memory, requested)
            path = owned_image(evidence.get("path", requested))
            outputs = {**evidence, "requested_image_ref": requested,
                       "image_ref": str(path), "inspection_kind": "saved_image"}
            paths = [str(path)]
        else:
            if set(params) - {"bundle_id", "offset"}:
                raise ValueError("unsupported inspection parameters")
            offset = params.get("offset", 0)
            if type(offset) is not int or offset < 0:
                raise ValueError("offset must be a nonnegative integer")
            bundle_id = params.get("bundle_id")
            if not isinstance(bundle_id, str):
                raise ValueError("copy an exact registered SAM3 bundle_id")
            index = _memory_fact_value(memory.facts.get(BUNDLE_INDEX_KEY)) or {}
            reference = index.get(bundle_id)
            if not isinstance(reference, dict):
                raise ValueError("unknown session bundle_id")
            bundle = load_bundle(reference, root=root, session_id=memory.session_id,
                                 expected_kind="sam3_detections")
            result_id = bundle["reference_parameters"]["sam3_result_id"]
            outputs = None
            for event in reversed(memory.events):
                if event.event_type != "action":
                    continue
                command = event.payload.get("command", {})
                for call in reversed(command.get("tool_calls", [])):
                    if call.get("name") != "sam3":
                        continue
                    candidate = _tool_call_outputs(call)
                    if candidate.get("result_id") == result_id:
                        outputs = candidate
                        break
                if outputs is not None:
                    break
            if outputs is None:
                raise ValueError("stored SAM3 result unavailable; inspection never reruns inference")
            detections = [dict(d) for d in outputs.get("detections", []) if isinstance(d, dict)]
            if offset >= len(detections):
                raise ValueError(f"offset outside candidate set; candidate_count={len(detections)}")
            source = owned_image(outputs.get("source_image"))
            # Validate every image this page will read, including symlink escapes.
            for detection in detections[offset:offset + 4]:
                owned_image(detection.get("mask_ref"))
            parent = root / "evidence_views"
            if not parent.resolve().is_relative_to(root):
                raise ValueError("inspection output directory escapes this session")
            folder = parent / uuid4().hex
            folder.mkdir(parents=True, exist_ok=False)
            page, _ = _build_sam3_selection_artifacts(source_image=source,
                detections=detections, output_dir=folder,
                prompt=str(outputs.get("prompt") or "object"), visual_limit=4, offset=offset)
            comparison = {"status": "no_retrieved_reference", "identity_confirmed": False}
            # Use the latest explicit lookup, never silently fall back to another
            # object's older reference after an unsuccessful query.
            for event in reversed(memory.events):
                if event.event_type != "action":
                    continue
                lookups = [c for c in event.payload.get("command", {}).get("tool_calls", [])
                           if c.get("name") == "retrieve_asset_reference"]
                if not lookups:
                    continue
                lookup = _tool_call_outputs(lookups[-1])
                comparison.update({"reference_object": lookup.get("target_object"),
                                   "reference_environment": lookup.get("environment")})
                try:
                    refs = [owned_image(p) for p in lookup.get("reference_images", [])[:3]]
                    if refs:
                        start = (offset // 8) * 8
                        window = detections[start:start + 8]
                        for detection in window:
                            owned_image(detection.get("mask_ref"))
                        comparison.update(_render_reference_comparison(source, refs, window, folder,
                            label=str(lookup.get("target_object") or "retrieved object")))
                        comparison.update({"status": "available", "offset": start,
                            "next_offset": start + len(window) if start + len(window) < len(detections) else None,
                            "candidate_count": len(detections), "visualized_candidate_count": len(window),
                            "reference_image_count": len(refs),
                            "scope": "latest_explicit_lookup_appearance_only_not_scene_identity"})
                except (OSError, ValueError, TypeError, Image.DecompressionBombError):
                    comparison["status"] = "reference_comparison_unavailable"
                break
            outputs = {**page, "bundle_id": bundle_id, "result_id": result_id,
                       "inspection_kind": "detection_page",
                       "source_packet_id": outputs.get("source_packet_id"),
                       "source_camera_id": outputs.get("frame_id"),
                       "source_object_scene_epoch": bundle.get("object_scene_epoch"),
                       "current_object_scene_epoch": memory.object_scene_epoch(),
                       "matches_object_scene_epoch": bundle.get("object_scene_epoch") == memory.object_scene_epoch()}
            outputs["reference_comparison"] = comparison
            paths = ([comparison["image_ref"]] if comparison.get("status") == "available" else []) + [page["contact_sheet_ref"]]
        outputs["authorizes_selection_or_motion"] = False
        outputs["vision_evidence"] = [{"path": p, "role": "requested_evidence_view",
            "derived": True, "freshness": "stored_evidence_not_new_observation"} for p in paths]
        # Host-owned presentation only. It neither changes pending selection nor
        # makes the viewed historical evidence current.
        memory.save_fact("evidence_inspection", outputs, source="inspect_evidence")
        return make_tool_result(context, success=True,
            content="Opened stored evidence for the next planner input. No inference, identity selection, or motion occurred.",
            outputs=outputs, artifacts=[{"type": "evidence_inspection_image", "kind": "image", "path": p} for p in paths])
    except (OSError, ValueError, TypeError, KeyError, Image.DecompressionBombError) as exc:
        return make_tool_result(context, success=False, content=f"Evidence inspection failed: {exc}",
                                diagnostics=[{"code": "invalid_evidence_inspection"}])


def _render_reference_comparison(source, references, candidates, folder, *, label):
    """Bounded untinted comparison: up to three catalog views and eight crops."""
    from PIL import Image, ImageDraw, ImageOps
    from agent.tools.handlers import _sam3_visual_bbox, _sam3_padded_crop_box

    width, height = 320, 240
    rows = 1 + (len(candidates) + 2) // 3
    sheet = Image.new("RGB", (3 * width, rows * height), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)

    def tile(picture, index, title, *, reference=False):
        left, top = index % 3 * width, index // 3 * height
        draw.rectangle((left, top, left + width - 1, top + height - 1),
                       fill=(221, 235, 255) if reference else (255, 255, 255))
        draw.text((left + 8, top + 7), title[:48], fill=(0, 0, 0))
        thumb = ImageOps.contain(picture.convert("RGB"), (width - 16, height - 44))
        sheet.paste(thumb, (left + (width - thumb.width) // 2, top + 36 + (height - 44 - thumb.height) // 2))

    for index, path in enumerate(references):
        with Image.open(path) as reference:
            tile(reference, index, f"REF ONLY: {label} / {index + 1}", reference=True)
    with Image.open(source) as picture:
        rgb = picture.convert("RGB")
        for index, candidate in enumerate(candidates):
            with Image.open(candidate["mask_ref"]) as mask:
                bbox = _sam3_visual_bbox(candidate.get("bbox_xyxy"), mask=mask)
            box = _sam3_padded_crop_box(bbox, image_size=rgb.size)
            tile(rgb.crop(box) if box else rgb, index + 3,
                 f"SCENE: {candidate['id']} (original RGB)")
    path = folder / "reference_comparison.png"
    sheet.save(path)
    return {"image_ref": str(path), "candidate_ids": [c["id"] for c in candidates]}
