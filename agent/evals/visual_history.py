"""Offline projection ablations for bounded visual history."""

from __future__ import annotations

from pathlib import Path

from adapter.protocol import EnvObservation, JsonDict
from agent.runtime.memory import AgentMemory
from agent.runtime.visual_history import (
    VISUAL_HISTORY_SCHEMA_VERSION,
    VisualHistoryConfig,
    build_visual_history_projection,
)


VISUAL_HISTORY_ABLATION_SCHEMA_VERSION = "openeta.visual_history_ablation.v1"


def visual_history_ablation_variants() -> dict[str, VisualHistoryConfig]:
    """Return the deterministic A-D variants from the design document."""

    return {
        "A_current_main_only": VisualHistoryConfig(
            include_initial_main=False,
            recent_main_turns=1,
            include_current_wrist=False,
            include_vdm=False,
        ),
        "B_bounded_raw_no_vdm": VisualHistoryConfig(include_vdm=False),
        "C_bounded_raw_main_vdm": VisualHistoryConfig(),
        "D_current_main_full_vdm": VisualHistoryConfig(
            include_initial_main=False,
            recent_main_turns=1,
            include_current_wrist=False,
            include_vdm=True,
        ),
    }


def run_visual_history_projection_ablation(
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    current_camera_artifacts: list[JsonDict],
    variants: dict[str, VisualHistoryConfig] | None = None,
) -> JsonDict:
    """Compare visual payload size and historical coverage without provider calls."""

    results: list[JsonDict] = []
    for name, config in (variants or visual_history_ablation_variants()).items():
        projection = build_visual_history_projection(
            observation=observation,
            memory=memory,
            config=config,
            current_camera_artifacts=current_camera_artifacts,
        )
        paths = projection["vision_image_paths"]
        history = projection["visual_history"]
        evidence = projection["vision_evidence"]
        results.append(
            {
                "variant": name,
                "policy": config.to_dict(),
                "raw_image_count": len(paths),
                "raw_image_bytes": sum(_file_size(path) for path in paths),
                "raw_evidence_roles": [
                    str(item.get("role") or "")
                    for item in evidence
                    if isinstance(item, dict)
                ],
                "compressed_delta_count": len(history["compressed_deltas"]),
                "coverage": dict(history["coverage"]),
            }
        )
    return {
        "schema_version": VISUAL_HISTORY_ABLATION_SCHEMA_VERSION,
        "visual_history_schema_version": VISUAL_HISTORY_SCHEMA_VERSION,
        "results": results,
    }


def _file_size(path: object) -> int:
    if not isinstance(path, str) or not path:
        return 0
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0
