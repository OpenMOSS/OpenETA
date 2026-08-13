"""Inspect bounded-visual-history ablations from a durable local session."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter.protocol import EnvObservation, JsonDict, RobotState
from agent.evals.visual_history import run_visual_history_projection_ablation
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.visual_history import observation_history


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare deterministic raw-image and VDM-history projections for one "
            "persisted OpenETA session without calling a model provider."
        )
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--memory-root", default=".openeta_memory")
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = build_session_visual_history_ablation(
        memory_root=_repository_path(args.memory_root),
        session_id=args.session_id,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        _write_output(args.output, report)
    if args.strict:
        production = next(
            item
            for item in report["results"]
            if item["variant"] == "C_bounded_raw_main_vdm"
        )
        if production["coverage"]["missing_delta_observation_indices"]:
            raise SystemExit(1)


def build_session_visual_history_ablation(
    *,
    memory_root: Path,
    session_id: str,
) -> JsonDict:
    memory = AgentMemory(store=JsonMemoryStore(memory_root))
    memory.resume_session(session_id, max_events=1)
    observations = observation_history(memory)
    if not observations:
        raise ValueError(f"session {session_id!r} has no durable observation events")
    latest = observations[-1]
    metadata = latest.get("metadata")
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    environment_step = latest.get("environment_step")
    if isinstance(environment_step, int) and not isinstance(environment_step, bool):
        metadata["step_idx"] = environment_step
    observation = EnvObservation(
        task=str(latest.get("task") or memory.task or ""),
        cameras=[],
        robot=RobotState(),
        metadata=metadata,
    )
    raw_artifacts = latest.get("visual_artifacts")
    current_artifacts = [
        dict(item) for item in raw_artifacts or [] if isinstance(item, dict)
    ]
    report = run_visual_history_projection_ablation(
        observation=observation,
        memory=memory,
        current_camera_artifacts=current_artifacts,
    )
    report.update(
        {
            "session_id": session_id,
            "memory_root": str(memory_root),
            "observation_count": len(observations),
        }
    )
    return report


def _repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("paths must be relative to the repository")
    return path


def _write_output(path: str, report: JsonDict) -> None:
    output = _repository_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
