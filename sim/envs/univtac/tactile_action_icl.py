"""Pure protocol and aggregation helpers for the R1.1 live tactile-action ICL pilot."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from sim.envs.univtac.contract import UniVTACContractError
from sim.envs.univtac.native_operation import EXPERT_SEEDS, EXPERT_SEGMENTS

OPAQUE_SKILLS = ("skill_mica", "skill_onyx", "skill_quartz")
OPAQUE_TO_NATIVE = {
    "skill_quartz": "align_key",
    "skill_mica": "settle_alignment",
    "skill_onyx": "pull_key_out",
}
NATIVE_TO_OPAQUE = {value: key for key, value in OPAQUE_TO_NATIVE.items()}
CORRUPTED_LABEL = {
    "skill_quartz": "skill_mica",
    "skill_mica": "skill_onyx",
    "skill_onyx": "skill_quartz",
}
CONDITIONS = (
    "no_demo_multimodal",
    "correct_icl_multimodal",
    "action_swapped_icl_multimodal",
    "correct_icl_visual_only",
)
QUERY_SPECS = {
    1_000_000: {
        "support_seed": 1_000_001,
        "prefix": (),
        "query_start_state": "pre_align",
        "expected_remaining_sequence": (
            "skill_quartz",
            "skill_mica",
            "skill_onyx",
        ),
        "condition_order": CONDITIONS,
    },
    1_000_001: {
        "support_seed": 1_000_002,
        "prefix": ("align_key",),
        "query_start_state": "post_align",
        "expected_remaining_sequence": ("skill_mica", "skill_onyx"),
        "condition_order": (
            "correct_icl_multimodal",
            "action_swapped_icl_multimodal",
            "correct_icl_visual_only",
            "no_demo_multimodal",
        ),
    },
    1_000_002: {
        "support_seed": 1_000_000,
        "prefix": ("align_key", "settle_alignment"),
        "query_start_state": "post_settle",
        "expected_remaining_sequence": ("skill_onyx",),
        "condition_order": (
            "action_swapped_icl_multimodal",
            "correct_icl_visual_only",
            "no_demo_multimodal",
            "correct_icl_multimodal",
        ),
    },
}

CODEX_ICL_PROMPT = """You control one live Pull Out Key continuation episode.

The available manipulation procedures have opaque names.
When demonstrations are provided, infer what each opaque skill
does from its before-state, action, after-state, and outcome.

First call `review_demonstrations`.
Then call `observe`.

Choose the skill that best matches the current state.
After each executed skill, inspect the newly returned observation
before deciding whether another skill is needed.

Use at most three world-changing skill calls.
Call `finish_episode` when you believe the task is complete or
when no useful skill remains.

Do not invent poses, joints, displacements, or low-level actions.
Native task success is evaluated privately.
"""

R11_MCP_TOOLS = (
    "review_demonstrations",
    "observe",
    "execute_skill",
    "finish_episode",
)


@dataclass
class OpaqueSkillBudget:
    """Reserve each Agent-visible opaque skill at most once."""

    executed: list[str] = field(default_factory=list)
    max_calls: int = 3

    @property
    def remaining(self) -> list[str]:
        return [skill for skill in OPAQUE_SKILLS if skill not in self.executed]

    def reserve(self, skill: str) -> int:
        if skill not in OPAQUE_SKILLS:
            raise UniVTACContractError(f"unknown opaque skill: {skill!r}")
        if skill in self.executed:
            raise UniVTACContractError(f"opaque skill already executed: {skill}")
        if len(self.executed) >= self.max_calls:
            raise UniVTACContractError("opaque skill budget is exhausted")
        self.executed.append(skill)
        return len(self.executed)


def validate_r11_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "task",
        "query_seeds",
        "conditions",
        "opaque_skills",
        "model",
        "reasoning_effort",
        "task_config",
        "mode",
        "device",
        "task_instruction",
        "max_world_changing_skills",
        "worker_timeout_seconds",
        "codex_timeout_seconds",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise UniVTACContractError(f"R1.1 config is missing: {missing}")
    expected = {
        "task": "pull_out_key",
        "query_seeds": list(EXPERT_SEEDS),
        "conditions": list(CONDITIONS),
        "opaque_skills": list(OPAQUE_SKILLS),
        "model": "gpt-5.6-terra",
        "reasoning_effort": "medium",
        "task_config": "demo",
        "mode": "collect",
        "device": "cuda:0",
        "task_instruction": "Pull the key out of the slot.",
        "max_world_changing_skills": 3,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise UniVTACContractError(
                f"R1.1 requires {key}={value!r}, got {config[key]!r}"
            )
    validated = dict(config)
    for key in ("worker_timeout_seconds", "codex_timeout_seconds"):
        validated[key] = float(config[key])
        if validated[key] <= 0:
            raise UniVTACContractError(f"{key} must be positive")
    return validated


def query_spec(seed: int) -> dict[str, Any]:
    if seed not in QUERY_SPECS:
        raise UniVTACContractError(f"unsupported R1.1 query seed: {seed}")
    return copy.deepcopy(QUERY_SPECS[seed])


def demonstration_action(native_segment: str, condition: str) -> str:
    if native_segment not in NATIVE_TO_OPAQUE:
        raise UniVTACContractError(f"unknown native segment: {native_segment}")
    opaque = NATIVE_TO_OPAQUE[native_segment]
    return CORRUPTED_LABEL[opaque] if condition == "action_swapped_icl_multimodal" else opaque


def _contact_sheet(paths: Sequence[Path], labels: Sequence[str], output: Path) -> None:
    if len(paths) != len(labels) or len(paths) not in {4, 8}:
        raise UniVTACContractError("contact sheet requires four or eight labelled images")
    images = [Image.open(path).convert("RGB") for path in paths]
    columns = len(paths) // 2
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    label_height = 28
    sheet = Image.new("RGB", (columns * width, 2 * (height + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        row, column = divmod(index, columns)
        x, y = column * width, row * (height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + height + 5), label, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def build_demonstration_bank(
    *, r10_root: Path, output_root: Path
) -> dict[str, Any]:
    bank: dict[str, Any] = {"schema_version": "openeta.univtac.r11.demo_bank.v1", "seeds": {}}
    modalities = {
        "multimodal": ("head", "wrist", "left tactile", "right tactile"),
        "visual_only": ("head", "wrist"),
    }
    for seed in EXPERT_SEEDS:
        demos = []
        for index, segment in enumerate(EXPERT_SEGMENTS, start=1):
            transition_root = r10_root / "expert" / f"seed_{seed}" / "transitions" / segment
            if not (transition_root / "transition.json").is_file():
                raise FileNotFoundError(f"missing R1.0 expert transition: {transition_root}")
            sheets = {}
            for mode, names in modalities.items():
                paths = []
                labels = []
                for phase in ("pre", "post"):
                    for name in names:
                        if name == "head":
                            relative = f"{phase}/camera/head_rgb.png"
                        elif name == "wrist":
                            relative = f"{phase}/camera/wrist_rgb.png"
                        else:
                            sensor = name.split()[0]
                            relative = f"{phase}/tactile/{sensor}_tactile_rgb_marker.png"
                        paths.append(transition_root / relative)
                        labels.append(f"{'before' if phase == 'pre' else 'after'} · {name}")
                sheet = output_root / f"seed_{seed}" / f"example_{index}_{mode}.png"
                _contact_sheet(paths, labels, sheet)
                sheets[mode] = sheet.relative_to(output_root).as_posix()
            demos.append(
                {
                    "example": index,
                    "native_segment": segment,
                    "correct_action": NATIVE_TO_OPAQUE[segment],
                    "sheets": sheets,
                    "execution": "completed",
                    "episode_outcome": "native success",
                }
            )
        bank["seeds"][str(seed)] = demos
    return bank


def stage_r11_condition(
    *, bank: Mapping[str, Any], seed: int, condition: str, bank_root: Path, episode_root: Path
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise UniVTACContractError(f"unknown R1.1 condition: {condition}")
    spec = query_spec(seed)
    support_seed = int(spec["support_seed"])
    source_demos = bank["seeds"][str(support_seed)]
    demonstrations = []
    if condition != "no_demo_multimodal":
        mode = "visual_only" if condition == "correct_icl_visual_only" else "multimodal"
        for source in source_demos:
            index = int(source["example"])
            target = episode_root / "demonstrations" / f"example_{index}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bank_root / source["sheets"][mode], target)
            demonstrations.append(
                {
                    "example": index,
                    "action": demonstration_action(source["native_segment"], condition),
                    "execution": "completed",
                    "episode_outcome": "native success",
                    "image_path": target.relative_to(episode_root).as_posix(),
                }
            )
    manifest = {
        "schema_version": "openeta.univtac.r11.condition.v1",
        "agent_visible": {
            "condition": condition,
            "demonstrations": demonstrations,
            "image_mode": (
                "visual_only" if condition == "correct_icl_visual_only" else "multimodal"
            ),
            "available_skills": list(OPAQUE_SKILLS),
        },
        "host_only": {
            "seed": seed,
            "support_seed": support_seed,
            "query_start_state": spec["query_start_state"],
            "native_prefix": list(spec["prefix"]),
            "opaque_to_native": dict(OPAQUE_TO_NATIVE),
            "expected_remaining_sequence": list(spec["expected_remaining_sequence"]),
        },
    }
    episode_root.mkdir(parents=True, exist_ok=True)
    (episode_root / "condition.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_r11_projection(payload: Mapping[str, Any], *, image_mode: str) -> dict[str, Any]:
    allowed = {
        "task_instruction",
        "step_identifiers",
        "proprio",
        "images",
        "skill_execution",
        "available_skills",
    }
    if set(payload) != allowed:
        raise UniVTACContractError(
            f"R1.1 projection fields differ from allowlist: {sorted(payload)}"
        )
    expected = ["camera/head/rgb", "camera/wrist/rgb"]
    if image_mode == "multimodal":
        expected += [
            "tactile/left_tactile/rgb_marker",
            "tactile/right_tactile/rgb_marker",
        ]
    elif image_mode != "visual_only":
        raise UniVTACContractError(f"invalid R1.1 image mode: {image_mode}")
    images = payload["images"]
    labels = [item.get("label") for item in images] if isinstance(images, list) else []
    if labels != expected:
        raise UniVTACContractError(f"R1.1 image labels are invalid: {labels}")
    available = payload["available_skills"]
    if (
        not isinstance(available, list)
        or len(available) != len(set(available))
        or any(skill not in OPAQUE_SKILLS for skill in available)
        or available != [skill for skill in OPAQUE_SKILLS if skill in available]
    ):
        raise UniVTACContractError("R1.1 available opaque skills are invalid")

    forbidden_keys = {
        "actor_state",
        "press_depth",
        "depth",
        "plan_success",
        "check_success",
        "reward",
        "target_pose",
        "over_rotate",
        "key_rotation",
        "prefix",
        "native_prefix",
        "opaque_to_native",
        "visible_phase",
        "remaining_skills",
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

    flattened = set(walk(payload))
    leaked = sorted(forbidden_keys & flattened)
    semantic = sorted(set(EXPERT_SEGMENTS) & flattened)
    if leaked or semantic:
        raise UniVTACContractError(
            f"privileged R1.1 projection content leaked: {leaked + semantic}"
        )
    return copy.deepcopy(dict(payload))


def build_r11_codex_command(
    *,
    codex_bin: str,
    workspace: Path,
    repo_root: Path,
    episode_root: Path,
    worker_url: str,
    final_response_path: Path,
    model: str = "gpt-5.6-terra",
    reasoning_effort: str = "medium",
) -> list[str]:
    mcp_args = [
        f"PYTHONPATH={repo_root}",
        "uv",
        "run",
        "--no-project",
        "--python",
        str(repo_root / ".venv/bin/python"),
        "-m",
        "tools.univtac_tactile_action_icl_mcp_server",
        "--episode-root",
        str(episode_root),
        "--worker-url",
        worker_url,
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
        f"mcp_servers.univtac.enabled_tools={json.dumps(list(R11_MCP_TOOLS))}",
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
    command.append(CODEX_ICL_PROMPT)
    return command


def summarize_r11_results(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != 12:
        raise UniVTACContractError("R1.1 requires exactly twelve semantic episodes")
    metrics: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        if len(selected) != 3:
            raise UniVTACContractError(f"R1.1 condition is incomplete: {condition}")
        metrics[condition] = {
            "first_skill_correct_count": sum(bool(row["first_skill_correct"]) for row in selected),
            "exact_sequence_count": sum(bool(row["exact_sequence"]) for row in selected),
            "native_continuation_success_count": sum(
                bool(row["native_continuation_success"]) for row in selected
            ),
            "average_agent_skill_count": sum(int(row["agent_skill_count"]) for row in selected) / 3,
            "finish_without_action_count": sum(int(row["agent_skill_count"]) == 0 for row in selected),
            "invalid_or_repeated_skill_count": sum(
                int(row["invalid_or_repeated_skill_count"]) for row in selected
            ),
        }
    c0, c1, c2, c3 = (metrics[name] for name in CONDITIONS)
    gains = {
        "correct_icl_gain_over_no_demo": c1["first_skill_correct_count"] - c0["first_skill_correct_count"],
        "correct_icl_gain_over_swapped": c1["first_skill_correct_count"] - c2["first_skill_correct_count"],
        "tactile_gain_over_visual_only": c1["first_skill_correct_count"] - c3["first_skill_correct_count"],
        "native_success_gain_over_no_demo": c1["native_continuation_success_count"] - c0["native_continuation_success_count"],
        "native_success_gain_over_swapped": c1["native_continuation_success_count"] - c2["native_continuation_success_count"],
        "native_success_gain_over_visual_only": c1["native_continuation_success_count"] - c3["native_continuation_success_count"],
    }
    preliminary = (
        c1["first_skill_correct_count"] >= 2
        and gains["correct_icl_gain_over_no_demo"] >= 1
        and gains["correct_icl_gain_over_swapped"] >= 1
        and c1["native_continuation_success_count"] >= 2
    )
    interpretation = (
        "preliminary_tactile_specific_icl_signal"
        if preliminary
        and (
            gains["tactile_gain_over_visual_only"] >= 1
            or gains["native_success_gain_over_visual_only"] >= 1
        )
        else "generic_multimodal_icl_signal_without_tactile_gain"
        if preliminary
        else "no_detectable_icl_signal"
    )
    corrupted_follow = sum(
        bool(row["corrupted_label_followed"])
        for row in rows
        if row["condition"] == "action_swapped_icl_multimodal"
    )
    return {
        "classification": "tactile_action_icl_live_pilot_completed",
        "development_scope": "3-seed continuation pilot",
        "condition_metrics": metrics,
        "gains": gains,
        "corrupted_demo_first_skill_follow_count": corrupted_follow,
        "corrupted_demonstration_susceptibility": corrupted_follow >= 2,
        "preliminary_action_icl_signal": preliminary,
        "icl_interpretation": interpretation,
    }
