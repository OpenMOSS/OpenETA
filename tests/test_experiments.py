from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import agent.cli.experiment as experiment_cli
from agent.backends.provider_config import PlannerProviderConfig
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.experiments import (
    ExperimentWorkspace,
    build_proposed_task_playbook_tree,
    build_proposed_grasp_strategy_tree,
    collect_grasp_strategy_candidates,
    collect_skill_candidates,
    collect_task_playbook_candidates,
    compact_strategy_rollout_summary,
    objective_batch_metrics,
    select_supported_grasp_strategy_candidate,
    select_supported_candidates,
    select_supported_task_playbook_candidates,
    strategy_validation_has_no_regression,
    validation_has_no_regression,
    write_grasp_strategy_evidence,
)
from agent.runtime.calibration import calibration_profile_sha256
from agent.runtime.parallel import ParallelEpisodeSpec
from agent.runtime.planner_prompts import compose_main_planner_prompt
from agent.runtime.task_playbooks import task_text_sha256
from agent.tools.grasp_geometry import DEFAULT_GRASP_PROFILE
from agent.tools.grasp_strategies import (
    DEFAULT_GRASP_STRATEGY_ROOT,
    grasp_strategy_sha256,
    grasp_strategy_tree_sha256,
    load_grasp_strategies,
)


def _write_skill(root: Path, *, body: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pick.md").write_text(
        "---\nname: pick\ndescription: Pick an object.\nversion: v1\n"
        "editable: true\n---\n\n" + body + "\n",
        encoding="utf-8",
    )


def _outcome(episode_id: str, session_id: str, *, reward: float) -> dict:
    return {
        "episode_id": episode_id,
        "session_id": session_id,
        "status": "success",
        "episode": {
            "steps": [
                {
                    "step_result": {
                        "reward": reward,
                        "info": {},
                    }
                }
            ]
        },
    }


def test_main_planner_prompt_composition_is_stable_and_hashed() -> None:
    prompt, metadata = compose_main_planner_prompt("base planner contract")

    assert "Embodied Closed-Loop Operating Contract" in prompt
    assert "task_complete" in prompt
    assert metadata["schema_version"] == "openeta.planner_prompt.v1"
    assert metadata["sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def test_strategy_rollout_summary_reads_compact_batch_tool_calls() -> None:
    payload = {
        "outcomes": [
            {
                "episode_id": "episode-1",
                "env_id": "env-1",
                "seed": 0,
                "status": "fail",
                "episode": {
                    "metadata": {"failure_reason": {"code": "episode_timeout"}},
                    "steps": [
                        {
                            "action": {
                                "request_name": "move_to",
                                "tool_calls": [
                                    {
                                        "name": "move_to",
                                        "status": "executed",
                                        "result": {"success": True},
                                    }
                                ],
                            },
                            "step_result": {"reward": 0.0},
                        }
                    ],
                },
            }
        ]
    }

    summary = compact_strategy_rollout_summary(payload)

    assert summary["episodes"][0]["grasp_events"] == [
        {
            "tool": "move_to",
            "status": "executed",
            "success": True,
            "reason": None,
            "candidate_id": None,
            "strategy_id": None,
            "strategy_selection": None,
            "gripper_width_m": None,
            "orientation_clamped": None,
            "reached_target": None,
        }
    ]


def test_preflight_validates_without_creating_environment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"episodes":[{"episode_id":"ep-1","task":"inspect",'
        '"env_id":"env"}]}',
        encoding="utf-8",
    )
    skills = tmp_path / "skills"
    _write_skill(skills, body="Inspect the scene.")
    monkeypatch.setattr(
        experiment_cli,
        "load_planner_provider_config",
        lambda: PlannerProviderConfig(
            model="model",
            api_base="https://provider.invalid/v1",
            api_key="secret",
        ),
    )
    args = SimpleNamespace(
        manifest=str(manifest),
        concurrency=1,
        model="",
        sim_url="https://sim.invalid/sse",
        sam3_url="",
        anygrasp_url="",
        skip_mcp_check=True,
        mcp_timeout_s=1.0,
        require_perception=False,
        baseline_skills=str(skills),
    )

    result = experiment_cli.preflight(args)

    assert result["ok"] is True
    assert result["episode_count"] == 1
    assert result["mcp"]["simulator"]["checked"] is False
    assert result["planner_prompt"]["sha256"]


def test_episode_filter_preserves_requested_order_and_rejects_unknown_ids() -> None:
    specs = [
        ParallelEpisodeSpec("ep-0", "task 0", "env"),
        ParallelEpisodeSpec("ep-1", "task 1", "env"),
        ParallelEpisodeSpec("ep-2", "task 2", "env"),
    ]

    filtered = experiment_cli._filter_episode_specs(specs, ["ep-2", "ep-0", "ep-2"])

    assert [spec.episode_id for spec in filtered] == ["ep-2", "ep-0"]
    try:
        experiment_cli._filter_episode_specs(specs, ["missing"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("unknown episode id should fail")


def test_run_command_output_is_compact_by_default() -> None:
    payload = {
        "schema_version": "openeta.command_run.v1",
        "ok": False,
        "experiment_id": "exp",
        "generation": 0,
        "result_path": "result.json",
        "metrics": {"objective_success_count": 0},
        "candidate_count": 0,
        "batch": {
            "provider_concurrency": {"limit": 2, "max_active": 2},
            "outcomes": [
                {
                    "episode_id": "ep-0",
                    "status": "fail",
                    "episode": {"large": "payload"},
                    "error": {
                        "code": "provider_queue_timeout",
                        "type": "ProviderQueueTimeoutError",
                        "message": "queue timed out",
                    },
                }
            ],
        },
    }

    compact = experiment_cli._compact_command_output("run", payload)

    assert "batch" not in compact
    assert compact["result_path"] == "result.json"
    assert compact["provider_concurrency"]["limit"] == 2
    assert compact["failures"] == [
        {
            "episode_id": "ep-0",
            "status": "fail",
            "code": "provider_queue_timeout",
            "type": "ProviderQueueTimeoutError",
            "message": "queue timed out",
        }
    ]


def test_experiment_collects_only_objective_success_skill_candidates(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_skill(source, body="Observe, then pick.")
    experiment = ExperimentWorkspace.create("test-exp", root=tmp_path / "experiments")
    baseline = experiment.initialize_generation(0, source_skills=source)
    specs = experiment.prepare_specs(
        [ParallelEpisodeSpec("ep-1", "pick cube", "env", metadata={})],
        generation=0,
        phase="train",
        skills_root=baseline,
    )
    assert specs[0].metadata["source_skills_root"] == str(baseline)
    assert specs[0].metadata["source_grasp_strategies_root"] == str(
        experiment.grasp_strategy_baseline(0)
    )
    assert specs[0].metadata["on_need_human"] == "fail"

    successful = experiment.generation_dir(0) / "train" / "sessions" / "s1" / "skills"
    unsupported = experiment.generation_dir(0) / "train" / "sessions" / "s2" / "skills"
    _write_skill(successful, body="Observe again after a failed grasp, then replan.")
    _write_skill(unsupported, body="This change has no objective evidence.")
    batch = {
        "success_count": 2,
        "fail_count": 0,
        "need_human_count": 0,
        "outcomes": [
            _outcome("ep-1", "s1", reward=1.0),
            _outcome("ep-2", "s2", reward=0.0),
        ],
    }

    manifest = collect_skill_candidates(
        generation_dir=experiment.generation_dir(0),
        phase="train",
        batch_payload=batch,
    )

    assert manifest["candidate_count"] == 1
    assert manifest["candidates"][0]["supporting_episode_ids"] == ["ep-1"]
    assert Path(manifest["candidates"][0]["candidate_path"]).is_file()
    assert objective_batch_metrics(batch)["objective_success_count"] == 1


def test_experiment_propagates_reviewed_exact_task_playbook_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "skills"
    _write_skill(source, body="Observe, then pick.")
    experiment = ExperimentWorkspace.create("playbooks", root=tmp_path / "experiments")
    experiment.initialize_generation(0, source_skills=source)
    task = "pick up test bowl and place it in the basket"
    session_id = "session-1"
    episode_id = "episode-1"
    candidate = {
        "schema_version": "openeta.task_playbook.v1",
        "status": "candidate",
        "playbook_id": "auto-test-task-1",
        "revision": 1,
        "scope": {
            "environment_id": "openeta/test-v0",
            "suite": "test_suite",
            "task_index": 1,
            "task_text_sha256": task_text_sha256(task),
        },
        "compatibility": {"calibration_ids": []},
        "guidance": {
            "task_summary": task,
            "rules": ["Re-observe before using this prior."],
        },
        "evidence": {
            "source_session_ids": [session_id],
            "official_rewards": [1.0],
        },
    }
    staged = (
        experiment.generation_dir(0)
        / "train"
        / "sessions"
        / session_id
        / "memory"
        / "task_playbook_reviews"
        / "candidate"
    )
    staged.mkdir(parents=True)
    (staged / "candidate.json").write_text(
        json.dumps(candidate), encoding="utf-8"
    )
    outcome = {
        "episode_id": episode_id,
        "session_id": session_id,
        "env_id": "openeta/test-v0",
        "status": "success",
        "episode": {
            "task": task,
            "steps": [
                {
                    "observation": {
                        "metadata": {
                            "env_id": "openeta/test-v0",
                            "suite": "test_suite",
                            "task_index": 1,
                        }
                    },
                    "step_result": {"reward": 1.0, "info": {}},
                }
            ],
        },
    }

    manifest = collect_task_playbook_candidates(
        generation_dir=experiment.generation_dir(0),
        phase="train",
        batch_payload={"outcomes": [outcome]},
    )
    selected = select_supported_task_playbook_candidates(manifest)
    proposed = build_proposed_task_playbook_tree(
        baseline_task_playbooks=experiment.task_playbook_baseline(0),
        destination=experiment.generation_dir(0) / "proposed_task_playbooks",
        candidates=selected,
    )

    assert manifest["candidate_count"] == 1
    assert [item["playbook_id"] for item in selected] == ["auto-test-task-1"]
    assert (proposed / "candidate" / "auto-test-task-1.json").is_file()


def test_task_playbook_candidate_merges_with_curated_exact_scope(
    tmp_path: Path,
) -> None:
    experiment = ExperimentWorkspace.create("merge-playbook", root=tmp_path / "experiments")
    skills = tmp_path / "skills"
    _write_skill(skills, body="Observe, then pick.")
    experiment.initialize_generation(0, source_skills=skills)
    source = tmp_path / "incoming.json"
    incoming = {
        "schema_version": "openeta.task_playbook.v1",
        "status": "candidate",
        "playbook_id": "auto-alphabet-success",
        "revision": 1,
        "scope": {
            "environment_id": "openeta/libero_libero_object_task0-v0",
            "suite": "libero_object",
            "task_index": 0,
            "task_text_sha256": task_text_sha256(
                "pick up the alphabet soup and place it in the basket"
            ),
        },
        "compatibility": {"calibration_ids": ["graspnet-eef-panda-p8"]},
        "guidance": {"observed_object_queries": ["alphabet soup"]},
        "evidence": {
            "source_session_ids": ["new-success"],
            "official_rewards": [1.0],
        },
    }
    source.write_text(json.dumps(incoming), encoding="utf-8")

    proposed = build_proposed_task_playbook_tree(
        baseline_task_playbooks=experiment.task_playbook_baseline(0),
        destination=experiment.generation_dir(0) / "merged_task_playbooks",
        candidates=[{"candidate_path": str(source)}],
    )
    curated_path = (
        proposed / "candidate" / "libero-object-task0-alphabet-soup.json"
    )
    merged = json.loads(curated_path.read_text(encoding="utf-8"))

    assert not (proposed / "candidate" / "auto-alphabet-success.json").exists()
    assert merged["revision"] == 2
    assert merged["guidance"]["object_priors"][0]["canonical_asset_key"] == (
        "libero/alphabet_soup"
    )
    assert "new-success" in merged["evidence"]["source_session_ids"]


def test_experiment_collects_reviewed_session_strategy_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_skill(source, body="Observe, then pick.")
    experiment = ExperimentWorkspace.create("strategy-exp", root=tmp_path / "experiments")
    experiment.initialize_generation(0, source_skills=source)
    session = "strategy-session"
    staged = (
        experiment.generation_dir(0)
        / "train"
        / "sessions"
        / session
        / "memory"
        / "grasp_strategy_lifecycle"
        / "staged"
        / "candidate"
    )
    staged.mkdir(parents=True)
    strategy = load_grasp_strategies(DEFAULT_GRASP_STRATEGY_ROOT)[0]
    strategy = {key: value for key, value in strategy.items() if not key.startswith("_")}
    strategy.update(
        {
            "strategy_id": "top-down-vertical-panda-p9",
            "revision": 2,
            "supersedes": "top-down-vertical-panda-p8",
        }
    )
    (staged / "top-down-vertical-panda-p9.json").write_text(
        json.dumps(strategy),
        encoding="utf-8",
    )
    batch = {
        "outcomes": [_outcome("ep-1", session, reward=1.0)],
        "success_count": 1,
        "fail_count": 0,
        "need_human_count": 0,
    }

    manifest = collect_grasp_strategy_candidates(
        generation_dir=experiment.generation_dir(0),
        phase="train",
        batch_payload=batch,
    )
    selected = select_supported_grasp_strategy_candidate(manifest)

    assert manifest["candidate_count"] == 1
    assert selected is not None
    proposed = build_proposed_grasp_strategy_tree(
        baseline_strategies=experiment.grasp_strategy_baseline(0),
        destination=experiment.generation_dir(0) / "proposed",
        candidate_path=selected["candidate_path"],
    )
    assert grasp_strategy_tree_sha256(proposed) != grasp_strategy_tree_sha256(
        experiment.grasp_strategy_baseline(0)
    )


def test_strategy_evidence_records_paired_provenance(tmp_path: Path) -> None:
    strategy = load_grasp_strategies(DEFAULT_GRASP_STRATEGY_ROOT)[0]
    strategy_sha = grasp_strategy_sha256(strategy)
    strategy_tree_sha = grasp_strategy_tree_sha256(DEFAULT_GRASP_STRATEGY_ROOT)
    calibration = json.loads(
        DEFAULT_GRASP_PROFILE.read_text(encoding="utf-8")
    )
    calibration_sha = calibration_profile_sha256(calibration)
    baseline_outcome = _outcome("ep-1", "baseline", reward=1.0)
    candidate_outcome = _outcome("ep-1", "candidate", reward=1.0)
    candidate_outcome.update({"env_id": "libero-task", "seed": 7})
    candidate_outcome["episode"]["task"] = "pick bowl"
    candidate_outcome["episode"]["metadata"] = {
        "grasp_strategy_tree_sha256": strategy_tree_sha,
        "calibration_profile_sha256": calibration_sha,
    }
    baseline = {
        "outcomes": [baseline_outcome],
        "success_count": 1,
        "fail_count": 0,
        "need_human_count": 0,
    }
    candidate = {
        "outcomes": [candidate_outcome],
        "success_count": 1,
        "fail_count": 0,
        "need_human_count": 0,
    }

    path = write_grasp_strategy_evidence(
        tmp_path / "evidence.json",
        split="held_out",
        strategy_sha256=strategy_sha,
        calibration_profile_sha256=calibration_sha,
        baseline=baseline,
        candidate=candidate,
        expected_strategy_tree_sha256=strategy_tree_sha,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["successes"] == 1
    assert payload["baseline_successes"] == 1
    assert payload["task_ids"] == ["pick bowl"]
    assert payload["contract_violations"] == 0


def test_reviewed_autonomy_strategy_iteration_accepts_paired_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    skills = tmp_path / "skills"
    _write_skill(skills, body="Observe, then pick.")
    experiment = ExperimentWorkspace.create("auto-strategy", root=tmp_path / "experiments")
    baseline_skills = experiment.initialize_generation(0, source_skills=skills)
    baseline_strategies = experiment.grasp_strategy_baseline(0)
    current = load_grasp_strategies(baseline_strategies)[0]
    authored = {key: value for key, value in current.items() if not key.startswith("_")}
    authored.update(
        {
            "status": "candidate",
            "strategy_id": "top-down-vertical-panda-p9",
            "revision": int(current["revision"]) + 1,
            "supersedes": current["strategy_id"],
        }
    )
    author_backend = StaticPlannerBackend(
        {
            "decision": "propose",
            "reason": "repeatable geometry-family evidence",
            "strategy": authored,
        }
    )
    reviewer_backend = StaticPlannerBackend(
        {"decision": "approve", "reason": "bounded strategy"}
    )
    monkeypatch.setattr(
        experiment_cli,
        "_new_experiment_backend",
        lambda _args, max_tokens=None: (
            author_backend if max_tokens == 4096 else reviewer_backend
        ),
    )
    calibration = json.loads(DEFAULT_GRASP_PROFILE.read_text(encoding="utf-8"))
    calibration_sha = calibration_profile_sha256(calibration)

    def fake_validation(
        _args,
        _experiment,
        _generation,
        specs,
        *,
        skills_root,
        grasp_strategies_root,
        phase,
    ):
        del skills_root, phase
        tree_hash = grasp_strategy_tree_sha256(grasp_strategies_root)
        outcomes = []
        for spec in specs:
            outcome = _outcome(spec.episode_id, f"session-{spec.episode_id}", reward=1.0)
            outcome.update({"env_id": spec.env_id, "seed": spec.seed})
            outcome["episode"]["task"] = spec.task
            outcome["episode"]["metadata"] = {
                "grasp_strategy_tree_sha256": tree_hash,
                "calibration_profile_sha256": calibration_sha,
            }
            outcomes.append(outcome)
        return {
            "outcomes": outcomes,
            "success_count": len(outcomes),
            "fail_count": 0,
            "need_human_count": 0,
        }

    monkeypatch.setattr(experiment_cli, "_run_validation", fake_validation)
    specs = [
        ParallelEpisodeSpec("ep-1", "pick bowl", "libero", seed=1),
        ParallelEpisodeSpec("ep-2", "pick apple", "libero", seed=2),
    ]
    train = fake_validation(
        None,
        experiment,
        0,
        specs,
        skills_root=baseline_skills,
        grasp_strategies_root=baseline_strategies,
        phase="train",
    )
    args = SimpleNamespace(
        calibration_profile=str(DEFAULT_GRASP_PROFILE),
        model="",
        publish_grasp_strategies=False,
        strategy_min_canary_attempts=2,
        strategy_min_held_out_attempts=2,
        strategy_min_held_out_success_rate=0.95,
        strategy_min_held_out_tasks=2,
    )

    accepted, result = experiment_cli._iterate_grasp_strategy(
        args,
        experiment=experiment,
        generation=0,
        train=train,
        train_specs=specs,
        validation_specs=specs,
        skills_root=baseline_skills,
        baseline_strategies=baseline_strategies,
        session_candidate=None,
    )

    assert result["accepted_for_next_generation"] is True
    assert result["proposal"]["review"]["decision"] == "approve"
    assert result["canary"]["passed"] is True
    assert result["held_out"]["passed"] is True
    assert grasp_strategy_tree_sha256(accepted) != grasp_strategy_tree_sha256(
        baseline_strategies
    )


def test_validation_rejects_paired_objective_regression() -> None:
    baseline = {
        "success_count": 2,
        "fail_count": 0,
        "need_human_count": 0,
        "outcomes": [
            _outcome("ep-1", "baseline-1", reward=1.0),
            _outcome("ep-2", "baseline-2", reward=1.0),
        ],
    }
    candidate = {
        "success_count": 2,
        "fail_count": 0,
        "need_human_count": 0,
        "outcomes": [
            _outcome("ep-1", "candidate-1", reward=1.0),
            _outcome("ep-2", "candidate-2", reward=0.0),
        ],
    }

    comparison = validation_has_no_regression(baseline, candidate)

    assert comparison["passed"] is False
    assert comparison["regressed_episode_ids"] == ["ep-2"]


def test_candidate_selection_fails_closed_on_equal_support_conflict() -> None:
    selected = select_supported_candidates(
        {
            "candidates": [
                {"skill_name": "pick", "sha256": "a", "support_count": 2},
                {"skill_name": "pick", "sha256": "b", "support_count": 2},
            ]
        }
    )

    assert selected == []


def test_validation_requires_candidate_objective_success() -> None:
    no_evidence = {
        "success_count": 1,
        "fail_count": 0,
        "need_human_count": 0,
        "outcomes": [_outcome("ep-1", "session", reward=0.0)],
    }

    comparison = validation_has_no_regression(no_evidence, no_evidence)

    assert comparison["passed"] is False
    assert comparison["candidate_has_objective_success"] is False


def test_strategy_validation_excludes_structured_infrastructure_failure() -> None:
    baseline = {
        "success_count": 1,
        "fail_count": 0,
        "need_human_count": 0,
        "outcomes": [_outcome("ep-1", "baseline", reward=1.0)],
    }
    candidate = {
        "success_count": 1,
        "fail_count": 1,
        "need_human_count": 0,
        "outcomes": [
            _outcome("ep-1", "candidate", reward=1.0),
            {
                "episode_id": "ep-infra",
                "status": "fail",
                "error": {
                    "code": "provider_queue_timeout",
                    "type": "ProviderQueueTimeoutError",
                },
            },
        ],
    }

    comparison = strategy_validation_has_no_regression(baseline, candidate)

    assert comparison["passed"] is True
    assert comparison["excluded_infrastructure"]["candidate"] == ["ep-infra"]
