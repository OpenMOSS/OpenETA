#!/usr/bin/env python3
"""Run observation-bound SAM3 and AnyGrasp against one retained episode frame.

This is a host-side smoke command.  It does not create a simulator or
choose a grasp.  The Operator-facing result is the retained image artifacts;
the JSON summary is for diagnostics and replay.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.tools.embodied_perception import ObservationBoundPerception
from agent.tools.handlers import (
    build_sse_anygrasp_mcp_grasper,
    build_sse_sam3_mcp_segmenter,
)


def _read_current(root: Path) -> dict[str, Any]:
    current_path = root / "current.json"
    if not current_path.is_file():
        raise RuntimeError(f"episode current.json not found: {current_path}")
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    record = payload.get("record")
    if not isinstance(record, dict):
        raise RuntimeError("current.json has no retained observation record")
    return record


def _write_summary(root: Path, *, segmentation: Any, grasps: Any) -> Path:
    summary = {
        "schema_version": "openeta.embodied_perception_smoke.v1",
        "segmentation": {
            "success": segmentation.success,
            "content": segmentation.content,
            "details": segmentation.details,
        },
        "grasps": {
            "success": grasps.success,
            "content": grasps.content,
            "details": grasps.details,
        },
    }
    path = root / "perception" / "smoke_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run(
    root: Path,
    *,
    target: str,
    sam3_url: str,
    anygrasp_url: str,
    camera_id: str,
    detection_id: str | None,
) -> int:
    record = _read_current(root)
    perception = ObservationBoundPerception(
        artifact_root=root,
        sam3=build_sse_sam3_mcp_segmenter(url=sam3_url),
        anygrasp=build_sse_anygrasp_mcp_grasper(url=anygrasp_url),
        output_root=root / "perception",
    )
    segmentation = perception.segment_object(record, target, camera_id=camera_id)
    if segmentation.success:
        grasps = perception.propose_grasps(
            record,
            detection_id=detection_id,
            camera_id=camera_id,
        )
    else:
        grasps = type(segmentation)(
            success=False,
            content="Skipped AnyGrasp because SAM3 segmentation failed.",
            details={"tool": "propose_grasps", "reason": "segmentation_failed"},
        )
    summary_path = _write_summary(root, segmentation=segmentation, grasps=grasps)
    print(json.dumps({
        "observation": segmentation.details.get("observation"),
        "segmentation_success": segmentation.success,
        "segmentation_contact_sheet": segmentation.details.get("selection_bundle", {}).get("contact_sheet_ref"),
        "grasps_success": grasps.success,
        "grasp_overlay": grasps.details.get("visualization", {}).get("overlay_ref"),
        "summary": str(summary_path),
    }, ensure_ascii=False, indent=2))
    return 0 if segmentation.success and grasps.success else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="completed Kitty episode artifact root")
    parser.add_argument("--target", required=True, help="semantic object phrase for SAM3")
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--anygrasp-url", default="http://127.0.0.1:8774/sse")
    parser.add_argument("--camera-id", default="agentview")
    parser.add_argument("--detection-id", default=None)
    args = parser.parse_args(argv)
    return run(
        args.root,
        target=args.target,
        sam3_url=args.sam3_url,
        anygrasp_url=args.anygrasp_url,
        camera_id=args.camera_id,
        detection_id=args.detection_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
