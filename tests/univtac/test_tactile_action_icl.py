from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from scripts.univtac.run_tactile_action_icl_pilot import (
    _action_started,
    _result_from_evidence,
    _validate_operator_rows,
    _write_results,
)
from sim.envs.univtac.tactile_action_icl import (
    CODEX_ICL_PROMPT,
    CONDITIONS,
    CORRUPTED_LABEL,
    OPAQUE_SKILLS,
    OPAQUE_TO_NATIVE,
    QUERY_SPECS,
    R11_MCP_TOOLS,
    OpaqueSkillBudget,
    build_demonstration_bank,
    build_r11_codex_command,
    demonstration_action,
    query_spec,
    stage_r11_condition,
    summarize_r11_results,
    validate_r11_config,
    validate_r11_projection,
)
from tools.univtac_tactile_action_icl_mcp_server import build_server

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_r11_protocol_has_disjoint_support_and_fixed_order() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / "configs/univtac/pull_out_key_tactile_action_icl.yaml").read_text()
    )
    validate_r11_config(config)
    assert tuple(config["opaque_skills"]) == OPAQUE_SKILLS
    assert set(OPAQUE_TO_NATIVE.values()) == {
        "align_key",
        "settle_alignment",
        "pull_key_out",
    }
    for seed, spec in QUERY_SPECS.items():
        assert seed != spec["support_seed"]
        assert len(spec["condition_order"]) == 4
        assert set(spec["condition_order"]) == set(CONDITIONS)
        assert query_spec(seed) == spec


def test_three_query_states_have_three_different_first_skills() -> None:
    assert [QUERY_SPECS[seed]["expected_remaining_sequence"][0] for seed in QUERY_SPECS] == [
        "skill_quartz",
        "skill_mica",
        "skill_onyx",
    ]


def test_opaque_budget_is_agent_side_and_once_only() -> None:
    budget = OpaqueSkillBudget()
    assert budget.reserve("skill_quartz") == 1
    assert budget.remaining == ["skill_mica", "skill_onyx"]
    with pytest.raises(ValueError, match="already executed"):
        budget.reserve("skill_quartz")
    with pytest.raises(ValueError, match="unknown opaque"):
        budget.reserve("align_key")


def test_swapped_demonstration_changes_only_action_label() -> None:
    source = {
        "before": ["same"],
        "after": ["same"],
        "outcome": "native success",
        "action": demonstration_action("align_key", "correct_icl_multimodal"),
    }
    swapped = copy.deepcopy(source)
    swapped["action"] = demonstration_action("align_key", "action_swapped_icl_multimodal")
    assert source["action"] == "skill_quartz"
    assert swapped["action"] == CORRUPTED_LABEL[source["action"]]
    assert {k: v for k, v in source.items() if k != "action"} == {
        k: v for k, v in swapped.items() if k != "action"
    }


def test_prompt_and_opaque_tools_do_not_reveal_native_names() -> None:
    for native in OPAQUE_TO_NATIVE.values():
        assert native not in CODEX_ICL_PROMPT
    assert "review_demonstrations" in CODEX_ICL_PROMPT


def test_demo_bank_uses_three_real_transitions_and_neutral_sheets(tmp_path: Path) -> None:
    r10 = tmp_path / "r10"
    for seed in QUERY_SPECS:
        for segment in ("align_key", "settle_alignment", "pull_key_out"):
            root = r10 / "expert" / f"seed_{seed}" / "transitions" / segment
            for phase in ("pre", "post"):
                for relative in (
                    "camera/head_rgb.png",
                    "camera/wrist_rgb.png",
                    "tactile/left_tactile_rgb_marker.png",
                    "tactile/right_tactile_rgb_marker.png",
                ):
                    path = root / phase / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (8, 6), "white").save(path)
            (root / "transition.json").write_text("{}", encoding="utf-8")
    bank = build_demonstration_bank(r10_root=r10, output_root=tmp_path / "bank")
    assert all(len(demos) == 3 for demos in bank["seeds"].values())
    assert (tmp_path / "bank/seed_1000000/example_1_multimodal.png").is_file()
    assert (tmp_path / "bank/seed_1000000/example_1_visual_only.png").is_file()

    episode = tmp_path / "episode"
    condition = stage_r11_condition(
        bank=bank,
        seed=1_000_000,
        condition="action_swapped_icl_multimodal",
        bank_root=tmp_path / "bank",
        episode_root=episode,
    )
    assert len(condition["agent_visible"]["demonstrations"]) == 3
    assert condition["agent_visible"]["demonstrations"][0]["action"] == "skill_mica"
    assert "opaque_to_native" not in condition["agent_visible"]
    server = build_server(episode_root=episode, worker_url="http://127.0.0.1:1")
    assert tuple(sorted(server._tool_manager._tools)) == tuple(sorted(R11_MCP_TOOLS))
    blocks = server._tool_manager._tools["review_demonstrations"].fn()
    assert len(blocks) == 6
    visible_text = "\n".join(block.text for block in blocks if hasattr(block, "text"))
    assert "skill_mica" in visible_text
    assert "align_key" not in visible_text
    assert "settle_alignment" not in visible_text
    assert "pull_key_out" not in visible_text


def test_r11_projection_and_codex_command_hide_native_semantics(tmp_path: Path) -> None:
    projection = {
        "task_instruction": "Pull the key out of the slot.",
        "step_identifiers": {"snapshot_id": "current", "simulator_step": 1},
        "proprio": {"joint": [], "ee": []},
        "images": [
            {"label": "camera/head/rgb"},
            {"label": "camera/wrist/rgb"},
            {"label": "tactile/left_tactile/rgb_marker"},
            {"label": "tactile/right_tactile/rgb_marker"},
        ],
        "skill_execution": {"skill": None, "status": "not_started"},
        "available_skills": list(OPAQUE_SKILLS),
    }
    validate_r11_projection(projection, image_mode="multimodal")
    leaked = copy.deepcopy(projection)
    leaked["skill_execution"]["skill"] = "align_key"
    with pytest.raises(ValueError, match="leaked"):
        validate_r11_projection(leaked, image_mode="multimodal")

    command = build_r11_codex_command(
        codex_bin="codex",
        workspace=tmp_path / "workspace",
        repo_root=REPO_ROOT,
        episode_root=tmp_path / "episode",
        worker_url="http://127.0.0.1:1",
        final_response_path=tmp_path / "final.md",
    )
    joined = " ".join(command)
    assert "tools.univtac_tactile_action_icl_mcp_server" in joined
    assert 'default_tools_approval_mode="approve"' in joined
    assert "/v1/chat/completions" not in joined
    for native in OPAQUE_TO_NATIVE.values():
        assert native not in joined


def test_r11_summary_uses_fixed_three_seed_counts() -> None:
    rows = []
    for condition in CONDITIONS:
        for seed in QUERY_SPECS:
            rows.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "first_skill_correct": condition == "correct_icl_multimodal",
                    "exact_sequence": False,
                    "native_continuation_success": condition == "correct_icl_multimodal",
                    "agent_skill_count": 1,
                    "invalid_or_repeated_skill_count": 0,
                    "corrupted_label_followed": condition
                    == "action_swapped_icl_multimodal",
                }
            )
    summary = summarize_r11_results(rows)
    assert summary["preliminary_action_icl_signal"] is True
    assert summary["icl_interpretation"] == "preliminary_tactile_specific_icl_signal"
    assert summary["corrupted_demonstration_susceptibility"] is True
    with pytest.raises(ValueError, match="twelve"):
        summarize_r11_results(rows[:-1])
    json.dumps(summary, allow_nan=False)


def test_r11_operator_trace_requires_review_observe_and_finish() -> None:
    rows = [
        {
            "tool": "review_demonstrations",
            "arguments": {},
            "response_image_paths": ["d1", "d2", "d3"],
        },
        {
            "tool": "observe",
            "arguments": {},
            "response_image_paths": ["h", "w", "l", "r"],
        },
        {
            "tool": "execute_skill",
            "arguments": {"skill": "skill_quartz"},
            "response_image_paths": ["h2", "w2", "l2", "r2"],
        },
        {"tool": "finish_episode", "arguments": {}, "response_image_paths": []},
    ]
    trace = _validate_operator_rows(rows, image_mode="multimodal", has_demos=True)
    assert trace["agent_skill_sequence"] == ["skill_quartz"]
    repeated = copy.deepcopy(rows)
    repeated.insert(3, copy.deepcopy(repeated[2]))
    with pytest.raises(RuntimeError, match="once-only"):
        _validate_operator_rows(repeated, image_mode="multimodal", has_demos=True)


def test_r11_retry_boundary_uses_persisted_action_trace(tmp_path: Path) -> None:
    assert _action_started(tmp_path) is False
    (tmp_path / "action_trace.jsonl").write_text("\n", encoding="utf-8")
    assert _action_started(tmp_path) is False
    (tmp_path / "action_trace.jsonl").write_text('{"action":1}\n', encoding="utf-8")
    assert _action_started(tmp_path) is True


def test_r11_recovers_nonretryable_action_timeout_without_native_claim(
    tmp_path: Path,
) -> None:
    manifest = {
        "agent_visible": {
            "condition": "action_swapped_icl_multimodal",
        },
        "host_only": {
            "seed": 1_000_001,
            "support_seed": 1_000_002,
            "query_start_state": "post_align",
            "expected_remaining_sequence": ["skill_mica", "skill_onyx"],
        },
    }
    (tmp_path / "condition.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "episode.json").write_text('{"status":"failed"}', encoding="utf-8")
    (tmp_path / "operator_context.jsonl").write_text(
        json.dumps(
            {"tool": "execute_skill", "arguments": {"skill": "skill_quartz"}}
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "action_trace.jsonl").write_text(
        '{"plan_success_after":true}\n', encoding="utf-8"
    )
    row = _result_from_evidence(tmp_path, native_evaluation_required=False)
    assert row["agent_skill_sequence"] == ["skill_quartz"]
    assert row["native_evaluation_available"] is False
    assert row["native_continuation_success"] is False
    assert row["codex_completed"] is False
    _write_results(tmp_path, [row])
    assert (tmp_path / "results.csv").is_file()
    assert not (tmp_path / "summary.json").exists()
