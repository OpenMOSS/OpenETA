"""Typed, immutable handoffs. Files are inspectable data, not bearer authority.

Only a reference registered in host memory may be resolved for execution.
Resolution produces an existing public reference request; all downstream gates
still run. Agent workspace JSON is never imported into this registry.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from uuid import uuid4

from adapter.protocol import JsonDict

BUNDLE_SCHEMA = "openeta.tool_bundle.v1"
BUNDLE_INDEX_KEY = "tool_handoff_bundles"
BUNDLE_ID_PATTERN = r"bnd-[0-9a-f]{32}"
MAX_BUNDLE_BYTES = 2_000_000

# The common field is deliberately not an untyped alias for arbitrary IDs.
BUNDLE_CONSUMERS = {
    "compile_grasp_seed": "grasp_candidates",
    "ik_preview_check": "target_pose",
    "move_to": "ik_result",
    "follow_eef_trajectory": "ik_trajectory",
    "select_sam3_detection": "sam3_detections",
    "reject_sam3_detections": "sam3_detections",
    "grasp_pose_estimate": "grasp_input",
    "anyplace": "placement_input",
}

BUNDLE_REFERENCE_FIELDS = frozenset({
    "grasp_result_id", "compiled_grasp_id", "target_pose", "waypoint_role",
    "path_fraction", "viewpoint_proposal_id", "probe_id", "waypoint_index",
    "ik_receipt_id", "ik_receipt_ids", "sam3_result_id", "native_bundle_id",
})


def bundle_reference_conflicts(tool_name: str, parameters: JsonDict) -> set[str]:
    forbidden = BUNDLE_REFERENCE_FIELDS
    if tool_name != "compile_grasp_seed":
        forbidden = forbidden | {"candidate_id"}
    return set(forbidden.intersection(parameters)) if "bundle_id" in parameters else set()


def write_bundle(*, root: Path, session_id: str, kind: str,
                 reference_parameters: JsonDict, summary: JsonDict,
                 object_scene_epoch: int, robot_motion_epoch: int | None,
                 producer: str, parents: list[str] | None = None) -> JsonDict:
    bundle_id = "bnd-" + uuid4().hex
    payload = {
        "schema_version": BUNDLE_SCHEMA, "bundle_id": bundle_id,
        "kind": kind, "session_id": session_id, "producer": producer,
        "parents": list(parents or []),
        "object_scene_epoch": object_scene_epoch,
        "robot_motion_epoch": robot_motion_epoch,
        "reference_parameters": reference_parameters, "summary": summary,
        "authority_note": "Inspecting this file does not authorize execution; current host gates apply.",
    }
    raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    if len(raw) > MAX_BUNDLE_BYTES:
        raise ValueError("tool bundle exceeds manifest size limit")
    directory = root.resolve() / "bundles"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise ValueError("bundle directory must not be a symlink")
    path = directory / f"{bundle_id}.json"
    with path.open("xb") as stream:
        stream.write(raw)
    path.chmod(0o400)
    return {
        "bundle_id": bundle_id, "kind": kind, "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(), "session_id": session_id,
        "object_scene_epoch": object_scene_epoch,
        "robot_motion_epoch": robot_motion_epoch,
        "summary": summary,
    }


def load_bundle(reference: JsonDict, *, root: Path, session_id: str,
                expected_kind: str) -> JsonDict:
    bundle_id = reference.get("bundle_id")
    if not isinstance(bundle_id, str) or re.fullmatch(BUNDLE_ID_PATTERN, bundle_id) is None:
        raise ValueError("invalid registered bundle ID")
    path = root.resolve() / "bundles" / f"{bundle_id}.json"
    if path.parent.is_symlink() or path.is_symlink() or str(path) != reference.get("path"):
        raise ValueError("bundle path does not match its registered immutable file")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BUNDLE_BYTES + 1)
    if len(raw) > MAX_BUNDLE_BYTES or hashlib.sha256(raw).hexdigest() != reference.get("sha256"):
        raise ValueError("bundle integrity check failed")
    payload = json.loads(raw)
    if not isinstance(payload, dict) or any((
        payload.get("schema_version") != BUNDLE_SCHEMA,
        payload.get("bundle_id") != bundle_id,
        payload.get("session_id") != session_id,
        reference.get("session_id") != session_id,
        payload.get("kind") != expected_kind,
        reference.get("kind") != expected_kind,
    )):
        raise ValueError(f"bundle session/type mismatch; expected {expected_kind}")
    return payload
