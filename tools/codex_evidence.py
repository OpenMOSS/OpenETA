"""Bounded, actionable observation references for the native Codex ingress."""
import json
from pathlib import Path

import jsonschema


def observation_references(memory):
    refs = memory.recent_observation_packet_refs(limit=6)
    return {
        "current": refs[-1] if refs else None,
        "current_packets": [r for r in refs if r["observation_index"] == refs[-1]["observation_index"]] if refs else [],
        "recent": refs,
        "instruction": "Use these registered source_packet_id and camera_frame_id values for perception. "
        "Simulator image artifact IDs and file paths are not source_packet_id values. "
        "For point prompts use the reference on that exact image label; old images are not current evidence.",
    }


def repair_feedback(command, schemas):
    repair = (command or {}).get("metadata", {}).get("repair_bundle") or {}
    if not repair:
        return None
    # Keep actionable public fields, never the full private diagnostic payload.
    result = {k: repair[k] for k in ("code", "violated_invariant", "recent_source_packets") if k in repair}
    calls = []
    for call in repair.get("allowed_next_calls", [])[:12]:
        name = call.get("tool")
        if name not in schemas:
            continue
        parameters = call.get("parameters")
        try:
            jsonschema.validate(parameters, schemas[name].inputSchema)
        except jsonschema.ValidationError:
            continue
        calls.append({"tool": name, "parameters": parameters})
    result["allowed_next_calls"] = calls
    return result


def image_label(memory, attachment, index):
    reference = memory.observation_packet_reference_for_path(attachment.get("path"))
    if reference:
        # A depth/mask artifact may share a packet, but its pixels are not the
        # RGB source accepted by point segmentation.
        try:
            source = memory.resolve_observation_packet(reference["source_packet_id"],
                                                       reference["camera_frame_id"], require_files=False)
            if Path(source["rgb"]).resolve() != Path(attachment["path"]).resolve():
                reference = {}
        except (ValueError, OSError, KeyError):
            reference = {}
    label = {"image_index": index, "role": attachment.get("role", "unclassified_evidence")}
    if reference:
        label.update(reference)
        latest = observation_references(memory)["current_packets"]
        label["is_current_observation"] = any(reference["source_packet_id"] == r["source_packet_id"] for r in latest)
        label["instruction"] = "Use this exact source_packet_id/camera_frame_id when grounding points in this image."
    else:
        label["is_current_observation"] = False
        label["instruction"] = "This image has no registered raw observation reference; do not use it as a point-prompt source."
    return json.dumps(label, ensure_ascii=False)
