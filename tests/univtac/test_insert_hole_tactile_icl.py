from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.univtac.run_insert_hole_tactile_icl_pilot import (
    _assert_condition_interventions,
    _assert_query_control,
    _assert_same_condition_images,
)
from sim.envs.univtac.insert_hole_tactile_icl import (
    CONDITIONS,
    MAX_SIMULATOR_EPISODES,
    OPAQUE_SKILLS,
    QUERY_SEEDS,
    SUPPORT_SEEDS,
    OneChoiceBudget,
    build_balanced_support_bank,
    build_r13_codex_command,
    stage_condition,
    summarize_r13_results,
    validate_r13_config,
    validate_visible_payload,
)
from tools.univtac_insert_hole_icl_mcp_server import build_server

REPO_ROOT = Path(__file__).resolve().parents[2]


def _snapshot(episode: Path, path: Path, value: int) -> None:
    cameras = {}
    tactile = {}
    for name, shape in (("head", (3, 4, 3)), ("wrist", (3, 4, 3))):
        target = path.parent / f"{path.stem}_{name}.png"
        Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(target)
        cameras[name] = {
            "rgb": {
                "path": target.relative_to(episode).as_posix(),
                "shape": list(shape),
                "dtype": "uint8",
            }
        }
    for name in ("left_tactile", "right_tactile"):
        target = path.parent / f"{path.stem}_{name}.png"
        Image.fromarray(np.full((2, 3, 3), value + 1, dtype=np.uint8)).save(target)
        tactile[name] = {
            "rgb_marker": {
                "path": target.relative_to(episode).as_posix(),
                "shape": [2, 3, 3],
                "dtype": "uint8",
            }
        }
    path.write_text(
        json.dumps(
            {
                "operator_visible": {
                    "cameras": cameras,
                    "tactile": tactile,
                    "proprio": {"joint": [value], "ee": [value + 0.5]},
                }
            }
        ),
        encoding="utf-8",
    )


def _support_source(root: Path, seed: int, decision_class: str) -> None:
    episode = root / "insert_hole" / f"seed_{seed}" / "expert"
    transition = episode / "transitions/01_first_downward"
    transition.mkdir(parents=True)
    _snapshot(episode, transition / "snapshot_before.json", seed % 10)
    _snapshot(episode, transition / "snapshot_after.json", seed % 10 + 2)
    _snapshot(episode, episode / "snapshot_final.json", seed % 10 + 4)
    (episode / "final_result.json").write_text(
        json.dumps(
            {
                "expert_episode_success": True,
                "decision_class": decision_class,
                "native_x_move": 0.04 if "positive" in decision_class else -0.04,
                "native_z_move": -0.01,
            }
        ),
        encoding="utf-8",
    )


def _query_bank(root: Path) -> dict:
    root.mkdir()
    queries = []
    host = []
    for index, seed in enumerate(QUERY_SEEDS, start=1):
        query_id = f"query_{chr(96 + index)}"
        phases = []
        for phase_name, value in (("before_contact", index), ("after_contact", index + 1)):
            images = []
            for name in ("head", "wrist", "left_tactile", "right_tactile"):
                path = root / query_id / phase_name / f"{name}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.full((2, 3, 3), value, dtype=np.uint8)).save(path)
                images.append(
                    {
                        "label": f"{phase_name}/{name}",
                        "path": path.relative_to(root).as_posix(),
                        "shape": [2, 3, 3],
                        "dtype": "uint8",
                    }
                )
            phases.append(
                {
                    "phase": phase_name,
                    "proprio": {"joint": [index], "ee": [index]},
                    "images": images,
                }
            )
        phase, after = phases
        queries.append(
            {
                "query_id": query_id,
                "task_instruction": "Insert the peg into the hole.",
                "before_contact": phase,
                "after_contact": after,
                "available_skills": list(OPAQUE_SKILLS),
            }
        )
        host.append(
            {
                "query_id": query_id,
                "seed": seed,
                "decision_class": "positive_x_correction",
                "expected_skill": "skill_ember",
            }
        )
    return {"agent_visible": {"queries": queries}, "host_only": {"queries": host}}


def test_config_fixes_support_queries_conditions_and_budget() -> None:
    payload = yaml.safe_load(
        (REPO_ROOT / "configs/univtac/insert_hole_tactile_action_icl.yaml").read_text()
    )
    validated = validate_r13_config(payload)
    assert validated["support_seeds"] == list(SUPPORT_SEEDS)
    assert validated["query_seeds"] == list(QUERY_SEEDS)
    assert validated["conditions"] == list(CONDITIONS)
    assert validated["opaque_skills"] == ["skill_slate", "skill_ember"]
    assert validated["max_simulator_episodes"] == MAX_SIMULATOR_EPISODES == 15
    with pytest.raises(ValueError, match="query_seeds"):
        validate_r13_config({**payload, "query_seeds": [*QUERY_SEEDS, 1_000_006]})


def test_one_choice_budget_rejects_second_or_unknown_choice() -> None:
    budget = OneChoiceBudget()
    budget.reserve("skill_slate")
    assert budget.choice == "skill_slate"
    with pytest.raises(ValueError, match="already submitted"):
        budget.reserve("skill_ember")
    with pytest.raises(ValueError, match="unknown opaque skill"):
        OneChoiceBudget().reserve("positive_x_correction")


def test_support_is_balanced_anonymous_and_host_mapping_isolated(tmp_path: Path) -> None:
    r12 = tmp_path / "r12"
    _support_source(r12, SUPPORT_SEEDS[0], "positive_x_correction")
    _support_source(r12, SUPPORT_SEEDS[1], "negative_x_correction")
    bank = build_balanced_support_bank(r12_root=r12, output_root=tmp_path / "bank")
    visible = bank["agent_visible"]
    assert [row["support_id"] for row in visible["demonstrations"]] == [
        "support_a",
        "support_b",
    ]
    assert [row["action"] for row in visible["demonstrations"]] == [
        "skill_ember",
        "skill_slate",
    ]
    text = json.dumps(visible)
    assert all(str(seed) not in text for seed in SUPPORT_SEEDS)
    assert "positive_x_correction" not in text and "native_x_move" not in text
    assert bank["host_only"]["class_balanced"] is True


def test_correct_swapped_keep_images_and_vision_only_only_removes_tactile(
    tmp_path: Path,
) -> None:
    r12 = tmp_path / "r12"
    _support_source(r12, SUPPORT_SEEDS[0], "positive_x_correction")
    _support_source(r12, SUPPORT_SEEDS[1], "negative_x_correction")
    bank_root = tmp_path / "bank"
    bank = build_balanced_support_bank(r12_root=r12, output_root=bank_root)
    query_root = tmp_path / "queries"
    query = _query_bank(query_root)
    correct = stage_condition(
        support_bank=bank,
        support_root=bank_root,
        query_bank=query,
        query_root=query_root,
        seed=QUERY_SEEDS[0],
        condition="correct_icl_multimodal",
        episode_root=tmp_path / "correct",
    )
    swapped = stage_condition(
        support_bank=bank,
        support_root=bank_root,
        query_bank=query,
        query_root=query_root,
        seed=QUERY_SEEDS[0],
        condition="action_swapped_icl_multimodal",
        episode_root=tmp_path / "swapped",
    )
    vision = stage_condition(
        support_bank=bank,
        support_root=bank_root,
        query_bank=query,
        query_root=query_root,
        seed=QUERY_SEEDS[0],
        condition="correct_icl_vision_only",
        episode_root=tmp_path / "vision",
    )
    c_demos = correct["agent_visible"]["demonstrations"]
    s_demos = swapped["agent_visible"]["demonstrations"]
    assert [row["action"] for row in c_demos] == ["skill_ember", "skill_slate"]
    assert [row["action"] for row in s_demos] == ["skill_slate", "skill_ember"]
    for c_demo, s_demo in zip(c_demos, s_demos, strict=True):
        c_copy, s_copy = copy.deepcopy(c_demo), copy.deepcopy(s_demo)
        c_copy.pop("action")
        s_copy.pop("action")
        for c_phase, s_phase in zip(
            (c_copy["before_contact"], c_copy["after_contact"], c_copy["outcome_observation"]),
            (s_copy["before_contact"], s_copy["after_contact"], s_copy["outcome_observation"]),
            strict=True,
        ):
            assert [row["label"] for row in c_phase["images"]] == [
                row["label"] for row in s_phase["images"]
            ]
    assert all(
        len(phase["images"]) == 2
        for demo in vision["agent_visible"]["demonstrations"]
        for phase in (demo["before_contact"], demo["after_contact"], demo["outcome_observation"])
    )
    assert correct["host_only"]["seed"] == swapped["host_only"]["seed"]
    assert "opaque_to_class" not in json.dumps(correct["agent_visible"])
    roots = {
        "correct_icl_multimodal": tmp_path / "correct",
        "action_swapped_icl_multimodal": tmp_path / "swapped",
        "correct_icl_vision_only": tmp_path / "vision",
    }
    no_demo = stage_condition(
        support_bank=bank,
        support_root=bank_root,
        query_bank=query,
        query_root=query_root,
        seed=QUERY_SEEDS[0],
        condition="no_demo_multimodal",
        episode_root=tmp_path / "no_demo",
    )
    assert no_demo["agent_visible"]["demonstrations"] == []
    roots["no_demo_multimodal"] = tmp_path / "no_demo"
    _assert_same_condition_images(tmp_path / "correct", tmp_path / "swapped")
    _assert_query_control(roots)
    _assert_condition_interventions(roots)


def test_visible_validation_rejects_privileged_mapping() -> None:
    with pytest.raises(ValueError, match="privileged R1.3"):
        validate_visible_payload({"opaque_to_class": {"skill_slate": "negative_x_correction"}})


def test_codex_command_is_cli_mcp_not_http(tmp_path: Path) -> None:
    command = build_r13_codex_command(
        codex_bin="codex",
        workspace=tmp_path / "workspace",
        repo_root=REPO_ROOT,
        episode_root=tmp_path / "episode",
        final_response_path=tmp_path / "final.md",
        model="gpt-5.6-terra",
        reasoning_effort="medium",
    )
    joined = " ".join(command)
    assert command[:3] == ["codex", "-m", "gpt-5.6-terra"]
    assert "exec" in command and "tools.univtac_insert_hole_icl_mcp_server" in joined
    assert "choose_skill" in joined and "chat/completions" not in joined
    assert "features.memories=false" in joined and 'history.persistence="none"' in joined
    assert "Call `review_demonstrations`" in command[-1]
    assert "`choose_skill` exactly once" in command[-1]


def test_summary_uses_fixed_denominators_and_signal_rules() -> None:
    rows = []
    choices = {
        "no_demo_multimodal": (False, False, False),
        "correct_icl_multimodal": (True, True, True),
        "action_swapped_icl_multimodal": (False, False, True),
        "correct_icl_vision_only": (True, False, False),
    }
    for condition in CONDITIONS:
        for seed, correct in zip(QUERY_SEEDS, choices[condition], strict=True):
            rows.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "selection_correct": correct,
                    "native_continuation_success": correct,
                    "canonical_decision_class": "positive_x_correction",
                    "confidence": "medium",
                    "duration_seconds": 1.0,
                    "token_usage": {},
                    "corrupted_demo_followed": condition == CONDITIONS[2] and not correct,
                }
            )
    summary = summarize_r13_results(rows)
    assert summary["condition_metrics"][CONDITIONS[1]][
        "correction_selection_correct_count"
    ] == 3
    assert summary["gains"]["correct_icl_gain_over_no_demo"] == 3
    assert summary["action_icl_signal"] == "preliminary_action_icl_signal"
    assert summary["tactile_icl_signal"] == "preliminary_tactile_specific_icl_signal"


def test_mcp_exposes_only_review_observe_and_single_choice(tmp_path: Path) -> None:
    episode = tmp_path / "episode"
    episode.mkdir()
    (episode / "condition.json").write_text(
        json.dumps(
            {
                "agent_visible": {
                    "demonstrations": [],
                    "query": {
                        "query_id": "current_query",
                        "task_instruction": "Insert the peg into the hole.",
                        "before_contact": {"phase": "before_contact", "proprio": {}, "images": []},
                        "after_contact": {"phase": "after_contact", "proprio": {}, "images": []},
                        "available_skills": list(OPAQUE_SKILLS),
                    },
                    "available_skills": list(OPAQUE_SKILLS),
                },
                "host_only": {"seed": QUERY_SEEDS[0]},
            }
        ),
        encoding="utf-8",
    )
    server = build_server(episode_root=episode)
    assert tuple(server._tool_manager._tools) == (
        "review_demonstrations",
        "observe_query",
        "choose_skill",
    )
    choose = server._tool_manager._tools["choose_skill"].fn
    blocks = choose("skill_slate", "medium", "contact transition resembles example")
    assert json.loads((episode / "decision.json").read_text())["choice_count"] == 1
    assert len(blocks) == 1
    with pytest.raises(ValueError, match="already submitted"):
        choose("skill_ember", "low", "second attempt")
