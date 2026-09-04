"""Pure contracts for the R1.3 causal Insert Hole tactile-action ICL pilot."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sim.envs.univtac.contract import UniVTACContractError

SUPPORT_SEEDS = (1_000_000, 1_000_001)
QUERY_SEEDS = (1_000_003, 1_000_004, 1_000_005)
CONDITIONS = (
    "no_demo_multimodal",
    "correct_icl_multimodal",
    "action_swapped_icl_multimodal",
    "correct_icl_vision_only",
)
OPAQUE_SKILLS = ("skill_slate", "skill_ember")
OPAQUE_TO_CLASS = {
    "skill_ember": "positive_x_correction",
    "skill_slate": "negative_x_correction",
}
CLASS_TO_OPAQUE = {value: key for key, value in OPAQUE_TO_CLASS.items()}
SWAPPED_SKILL = {"skill_ember": "skill_slate", "skill_slate": "skill_ember"}
MAX_SIMULATOR_EPISODES = 15
R13_MCP_TOOLS = ("review_demonstrations", "observe_query", "choose_skill")
IMAGE_NAMES = ("head", "wrist", "left_tactile", "right_tactile")

CODEX_PROMPT = """You are choosing one manipulation procedure for an Insert Hole continuation.

When demonstrations are available, infer the relation between the contact
transition, the opaque procedure, and the successful outcome.

Call `review_demonstrations`, then call `observe_query`, then call
`choose_skill` exactly once.

Choose exactly one opaque procedure. You will not be allowed to try both
procedures or recover after a wrong choice.

Do not invent poses, displacements, or joint commands.
"""


def validate_r13_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "round",
        "task",
        "support_seeds",
        "query_seeds",
        "conditions",
        "opaque_skills",
        "task_config",
        "mode",
        "device",
        "task_instructions",
        "model",
        "reasoning_effort",
        "max_world_changing_choices",
        "max_simulator_episodes",
        "save_host_only",
        "strict_two_tactile_sensors",
        "simulator_timeout_seconds",
        "codex_timeout_seconds",
        "source_commit",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise UniVTACContractError(f"R1.3 config is missing: {missing}")
    expected = {
        "round": "R1.3",
        "task": "insert_hole",
        "support_seeds": list(SUPPORT_SEEDS),
        "query_seeds": list(QUERY_SEEDS),
        "conditions": list(CONDITIONS),
        "opaque_skills": list(OPAQUE_SKILLS),
        "task_config": "demo",
        "mode": "collect",
        "device": "cuda:0",
        "task_instructions": {"insert_hole": "Insert the peg into the hole."},
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "max_world_changing_choices": 1,
        "max_simulator_episodes": MAX_SIMULATOR_EPISODES,
        "save_host_only": True,
        "strict_two_tactile_sensors": True,
        "source_commit": "371fac67917307026be8f00869fcc1b61c623a9f",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise UniVTACContractError(
                f"R1.3 requires {key}={value!r}, got {config[key]!r}"
            )
    validated = copy.deepcopy(dict(config))
    for key in ("simulator_timeout_seconds", "codex_timeout_seconds"):
        validated[key] = float(config[key])
        if validated[key] <= 0:
            raise UniVTACContractError(f"{key} must be positive")
    return validated


def expected_skill(decision_class: str) -> str:
    try:
        return CLASS_TO_OPAQUE[decision_class]
    except KeyError as exc:
        raise UniVTACContractError(
            f"R1.3 requires a non-zero correction class, got {decision_class!r}"
        ) from exc


class OneChoiceBudget:
    def __init__(self) -> None:
        self.choice: str | None = None

    def reserve(self, skill: str) -> None:
        if skill not in OPAQUE_SKILLS:
            raise UniVTACContractError(f"unknown opaque skill: {skill!r}")
        if self.choice is not None:
            raise UniVTACContractError("R1.3 skill choice was already submitted")
        self.choice = skill


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _phase_from_snapshot(
    *, source_episode: Path, snapshot_path: Path, target_root: Path, phase_name: str
) -> dict[str, Any]:
    snapshot = _read_json(snapshot_path)
    visible = snapshot["operator_visible"]
    images: list[dict[str, Any]] = []
    sources = {
        "head": visible["cameras"]["head"]["rgb"],
        "wrist": visible["cameras"]["wrist"]["rgb"],
        "left_tactile": visible["tactile"]["left_tactile"]["rgb_marker"],
        "right_tactile": visible["tactile"]["right_tactile"]["rgb_marker"],
    }
    for name in IMAGE_NAMES:
        descriptor = sources[name]
        source = source_episode / descriptor["path"]
        target = target_root / phase_name / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        images.append(
            {
                "label": f"{phase_name}/{name}",
                "path": target.relative_to(target_root.parent).as_posix(),
                "shape": descriptor["shape"],
                "dtype": descriptor["dtype"],
            }
        )
    return {
        "phase": phase_name,
        "proprio": copy.deepcopy(visible["proprio"]),
        "images": images,
    }


def _support_source(r12_root: Path, seed: int) -> Path:
    return r12_root / "insert_hole" / f"seed_{seed}" / "expert"


def build_balanced_support_bank(*, r12_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    visible_demos = []
    host_sources = []
    expected_classes = ("positive_x_correction", "negative_x_correction")
    for index, (seed, expected_class) in enumerate(
        zip(SUPPORT_SEEDS, expected_classes, strict=True), start=1
    ):
        source = _support_source(r12_root, seed)
        final = _read_json(source / "final_result.json")
        if not final.get("expert_episode_success"):
            raise UniVTACContractError(f"support seed {seed} is not a successful expert")
        if final.get("decision_class") != expected_class:
            raise UniVTACContractError(
                f"support seed {seed} has unexpected class {final.get('decision_class')}"
            )
        support_id = f"support_{chr(96 + index)}"
        target = output_root / support_id
        t0 = _phase_from_snapshot(
            source_episode=source,
            snapshot_path=source
            / "transitions/01_first_downward/snapshot_before.json",
            target_root=target,
            phase_name="before_contact",
        )
        t1 = _phase_from_snapshot(
            source_episode=source,
            snapshot_path=source
            / "transitions/01_first_downward/snapshot_after.json",
            target_root=target,
            phase_name="after_contact",
        )
        outcome = _phase_from_snapshot(
            source_episode=source,
            snapshot_path=source / "snapshot_final.json",
            target_root=target,
            phase_name="successful_outcome",
        )
        visible_demos.append(
            {
                "support_id": support_id,
                "before_contact": t0,
                "after_contact": t1,
                "action": expected_skill(expected_class),
                "outcome_observation": outcome,
                "episode_outcome": "native success",
            }
        )
        host_sources.append(
            {
                "support_id": support_id,
                "source_seed": seed,
                "decision_class": expected_class,
                "native_x_move": final["native_x_move"],
                "native_z_move": final["native_z_move"],
            }
        )
    agent = {
        "schema_version": "openeta.univtac.r13.support.visible.v1",
        "task_instruction": "Insert the peg into the hole.",
        "demonstrations": visible_demos,
        "available_skills": list(OPAQUE_SKILLS),
    }
    host = {
        "schema_version": "openeta.univtac.r13.support.host.v1",
        "opaque_to_class": dict(OPAQUE_TO_CLASS),
        "sources": host_sources,
        "class_balanced": True,
    }
    (output_root / "agent_visible.json").write_text(
        json.dumps(agent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "host_only.json").write_text(
        json.dumps(host, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_visible_payload(agent)
    return {"agent_visible": agent, "host_only": host}


def build_canonical_queries(
    *, oracle_root: Path, output_root: Path
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    visible_queries = []
    host_queries = []
    for index, seed in enumerate(QUERY_SEEDS, start=1):
        source = oracle_root / f"seed_{seed}"
        final = _read_json(source / "final_result.json")
        query_id = f"query_{chr(96 + index)}"
        target = output_root / query_id
        t0 = _phase_from_snapshot(
            source_episode=source,
            snapshot_path=source
            / "transitions/01_first_downward/snapshot_before.json",
            target_root=target,
            phase_name="before_contact",
        )
        t1 = _phase_from_snapshot(
            source_episode=source,
            snapshot_path=source
            / "transitions/01_first_downward/snapshot_after.json",
            target_root=target,
            phase_name="after_contact",
        )
        visible_queries.append(
            {
                "query_id": query_id,
                "task_instruction": "Insert the peg into the hole.",
                "before_contact": t0,
                "after_contact": t1,
                "available_skills": list(OPAQUE_SKILLS),
            }
        )
        host_queries.append(
            {
                "query_id": query_id,
                "seed": seed,
                "decision_class": final["decision_class"],
                "expected_skill": expected_skill(str(final["decision_class"])),
                "native_x_move": final["native_x_move"],
                "native_z_move": final["native_z_move"],
                "native_oracle_success": bool(final["expert_episode_success"]),
            }
        )
    agent = {
        "schema_version": "openeta.univtac.r13.query.visible.v1",
        "queries": visible_queries,
    }
    host = {
        "schema_version": "openeta.univtac.r13.query.host.v1",
        "queries": host_queries,
    }
    (output_root / "agent_visible.json").write_text(
        json.dumps(agent, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "host_only.json").write_text(
        json.dumps(host, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_visible_payload(agent)
    return {"agent_visible": agent, "host_only": host}


def _copy_phase_for_condition(
    phase: Mapping[str, Any],
    *,
    source_root: Path,
    target_root: Path,
    target_prefix: Path,
    visual_only: bool,
) -> dict[str, Any]:
    copied = {"phase": phase["phase"], "proprio": copy.deepcopy(phase["proprio"]), "images": []}
    for descriptor in phase["images"]:
        name = str(descriptor["label"]).split("/", 1)[1]
        if visual_only and "tactile" in name:
            continue
        source = source_root / descriptor["path"]
        target = target_root / target_prefix / str(descriptor["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        child = copy.deepcopy(dict(descriptor))
        child["path"] = target.relative_to(target_root).as_posix()
        copied["images"].append(child)
    return copied


def stage_condition(
    *,
    support_bank: Mapping[str, Any],
    support_root: Path,
    query_bank: Mapping[str, Any],
    query_root: Path,
    seed: int,
    condition: str,
    episode_root: Path,
) -> dict[str, Any]:
    if seed not in QUERY_SEEDS or condition not in CONDITIONS:
        raise UniVTACContractError(f"unsupported R1.3 cell: {seed}/{condition}")
    host_query = next(row for row in query_bank["host_only"]["queries"] if row["seed"] == seed)
    visible_query = next(
        row
        for row in query_bank["agent_visible"]["queries"]
        if row["query_id"] == host_query["query_id"]
    )
    visual_only = condition == "correct_icl_vision_only"
    demonstrations = []
    if condition != "no_demo_multimodal":
        for source in support_bank["agent_visible"]["demonstrations"]:
            action = str(source["action"])
            if condition == "action_swapped_icl_multimodal":
                action = SWAPPED_SKILL[action]
            demonstrations.append(
                {
                    "support_id": source["support_id"],
                    "before_contact": _copy_phase_for_condition(
                        source["before_contact"],
                        source_root=support_root,
                        target_root=episode_root,
                        target_prefix=Path("support"),
                        visual_only=visual_only,
                    ),
                    "after_contact": _copy_phase_for_condition(
                        source["after_contact"],
                        source_root=support_root,
                        target_root=episode_root,
                        target_prefix=Path("support"),
                        visual_only=visual_only,
                    ),
                    "action": action,
                    "outcome_observation": _copy_phase_for_condition(
                        source["outcome_observation"],
                        source_root=support_root,
                        target_root=episode_root,
                        target_prefix=Path("support"),
                        visual_only=visual_only,
                    ),
                    "episode_outcome": "native success",
                }
            )
    query_visible = {
        "query_id": "current_query",
        "task_instruction": visible_query["task_instruction"],
        "before_contact": _copy_phase_for_condition(
            visible_query["before_contact"],
            source_root=query_root,
            target_root=episode_root,
            target_prefix=Path("query"),
            visual_only=visual_only,
        ),
        "after_contact": _copy_phase_for_condition(
            visible_query["after_contact"],
            source_root=query_root,
            target_root=episode_root,
            target_prefix=Path("query"),
            visual_only=visual_only,
        ),
        "available_skills": list(OPAQUE_SKILLS),
    }
    manifest = {
        "schema_version": "openeta.univtac.r13.condition.v1",
        "agent_visible": {
            "demonstrations": demonstrations,
            "query": query_visible,
            "available_skills": list(OPAQUE_SKILLS),
        },
        "host_only": {
            "seed": seed,
            "condition": condition,
            "canonical_query_id": host_query["query_id"],
            "canonical_decision_class": host_query["decision_class"],
            "expected_skill": host_query["expected_skill"],
            "opaque_to_class": dict(OPAQUE_TO_CLASS),
            "image_mode": "visual_only" if visual_only else "multimodal",
        },
    }
    episode_root.mkdir(parents=True, exist_ok=True)
    (episode_root / "condition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_visible_payload(manifest["agent_visible"])
    return manifest


def validate_visible_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    forbidden = {
        "seed",
        "decision_class",
        "native_x_move",
        "native_z_move",
        "relative_pose",
        "actor_state",
        "press_depth",
        "depth",
        "opaque_to_class",
        "check_success",
        "plan_success",
        "positive_x_correction",
        "negative_x_correction",
    }

    def walk(value: Any):
        if isinstance(value, Mapping):
            for key, child in value.items():
                yield str(key)
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)
        elif isinstance(value, str):
            yield value

    leaked = sorted(forbidden & set(walk(payload)))
    if leaked:
        raise UniVTACContractError(f"privileged R1.3 content leaked: {leaked}")
    return copy.deepcopy(dict(payload))


def assert_same_images(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    def paths(payload: Mapping[str, Any]) -> list[str]:
        found = []

        def visit(value: Any) -> None:
            if isinstance(value, Mapping):
                if set(value) >= {"label", "path", "shape", "dtype"}:
                    found.append(str(value["path"]))
                else:
                    for child in value.values():
                        visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)

        visit(payload)
        return found

    left_paths, right_paths = paths(left), paths(right)
    if len(left_paths) != len(right_paths):
        raise UniVTACContractError("condition image counts differ")
    for left_path, right_path in zip(left_paths, right_paths, strict=True):
        if not np.array_equal(np.asarray(Image.open(left_path)), np.asarray(Image.open(right_path))):
            raise UniVTACContractError("condition images differ")


def build_r13_codex_command(
    *,
    codex_bin: str,
    workspace: Path,
    repo_root: Path,
    episode_root: Path,
    final_response_path: Path,
    model: str,
    reasoning_effort: str,
) -> list[str]:
    mcp_args = [
        f"PYTHONPATH={repo_root}",
        "uv",
        "run",
        "--no-project",
        "--python",
        str(repo_root / ".venv/bin/python"),
        "-m",
        "tools.univtac_insert_hole_icl_mcp_server",
        "--episode-root",
        str(episode_root),
    ]
    config = [
        "features.memories=false",
        "features.enable_request_compression=false",
        "memories.use_memories=false",
        "memories.generate_memories=false",
        'history.persistence="none"',
        f"model_reasoning_effort={json.dumps(reasoning_effort)}",
        'mcp_servers.univtac.command="env"',
        f"mcp_servers.univtac.args={json.dumps(mcp_args)}",
        "mcp_servers.univtac.required=true",
        f"mcp_servers.univtac.enabled_tools={json.dumps(list(R13_MCP_TOOLS))}",
        'mcp_servers.univtac.default_tools_approval_mode="approve"',
        "mcp_servers.univtac.tool_timeout_sec=600",
    ]
    command = [
        codex_bin,
        "-m",
        model,
        "exec",
        "-C",
        str(workspace),
        "-s",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--skip-git-repo-check",
        "--json",
        "--output-last-message",
        str(final_response_path),
    ]
    for value in config:
        command.extend(("-c", value))
    command.append(CODEX_PROMPT)
    return command


def summarize_r13_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 12:
        raise UniVTACContractError("R1.3 requires exactly twelve decision/execution rows")
    metrics: dict[str, Any] = {}
    confidence_score = {"low": 1, "medium": 2, "high": 3}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if [int(row["seed"]) for row in selected] != list(QUERY_SEEDS):
            raise UniVTACContractError(f"R1.3 condition is incomplete: {condition}")
        correct = sum(bool(row["selection_correct"]) for row in selected)
        successes = sum(bool(row["native_continuation_success"]) for row in selected)
        class_metrics = {}
        for decision_class in sorted({str(row["canonical_decision_class"]) for row in selected}):
            class_rows = [row for row in selected if row["canonical_decision_class"] == decision_class]
            class_metrics[decision_class] = {
                "correct_count": sum(bool(row["selection_correct"]) for row in class_rows),
                "total": len(class_rows),
            }
        metrics[condition] = {
            "correction_selection_correct_count": correct,
            "correction_selection_accuracy": correct / 3,
            "native_continuation_success_count": successes,
            "native_continuation_success_rate": successes / 3,
            "class_accuracy": class_metrics,
            "average_confidence": sum(
                confidence_score[str(row["confidence"])] for row in selected
            ) / 3,
            "average_duration_seconds": sum(float(row["duration_seconds"]) for row in selected) / 3,
            "token_usage": [copy.deepcopy(row.get("token_usage")) for row in selected],
        }
    c0, c1, c2, c3 = (metrics[name] for name in CONDITIONS)
    gains = {
        "correct_icl_gain_over_no_demo": c1["correction_selection_correct_count"] - c0["correction_selection_correct_count"],
        "correct_icl_gain_over_swapped": c1["correction_selection_correct_count"] - c2["correction_selection_correct_count"],
        "tactile_gain_over_vision_only": c1["correction_selection_correct_count"] - c3["correction_selection_correct_count"],
        "native_success_gain_over_no_demo": c1["native_continuation_success_count"] - c0["native_continuation_success_count"],
        "native_success_gain_over_swapped": c1["native_continuation_success_count"] - c2["native_continuation_success_count"],
        "native_success_gain_over_vision_only": c1["native_continuation_success_count"] - c3["native_continuation_success_count"],
    }
    action_signal = (
        c1["correction_selection_correct_count"] >= 2
        and gains["correct_icl_gain_over_no_demo"] >= 1
        and gains["correct_icl_gain_over_swapped"] >= 1
        and c1["native_continuation_success_count"] >= 2
    )
    tactile_signal = (
        "preliminary_tactile_specific_icl_signal"
        if action_signal
        and (
            gains["tactile_gain_over_vision_only"] >= 1
            or gains["native_success_gain_over_vision_only"] >= 1
        )
        else "generic_visual_or_multimodal_icl_without_tactile_gain"
    )
    c2_rows = [row for row in rows if row["condition"] == CONDITIONS[2]]
    corrupted_follow = sum(bool(row["corrupted_demo_followed"]) for row in c2_rows)
    return {
        "classification": "insert_hole_tactile_action_icl_pilot_completed",
        "development_scope": "3-seed development pilot",
        "condition_metrics": metrics,
        "gains": gains,
        "corrupted_demo_follow_count": corrupted_follow,
        "corrupted_demo_follow_rate": corrupted_follow / 3,
        "action_icl_signal": (
            "preliminary_action_icl_signal"
            if action_signal
            else "no_detectable_action_icl_signal"
        ),
        "tactile_icl_signal": tactile_signal,
        "results": [copy.deepcopy(dict(row)) for row in rows],
    }
