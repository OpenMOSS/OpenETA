from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.cli.eval as eval_cli
from agent.backends.provider_config import PlannerProviderConfig
from agent.cli.eval import main as eval_main
from agent.evals.plan import (
    EvaluationExecution,
    EvaluationPlan,
    EvaluationVariant,
    compile_evaluation_plan,
    compiled_plan_payload,
    load_evaluation_plan,
)
from agent.evals.runner import (
    EvaluationScheduler,
    build_evaluation_report,
    classify_evaluation_failure,
)
from agent.evals.store import EvaluationRunStore
from agent.evals.visual_history_rollout import (
    _alternating_tool_cycle_count,
    extract_visual_history_rollouts,
)
from agent.runtime.episode import EpisodeResult
from agent.runtime.parallel import ParallelEpisodeSpec, ParallelEpisodeWorker
from agent.tools.grasp_geometry import DEFAULT_GRASP_PROFILE


def _plan() -> EvaluationPlan:
    return EvaluationPlan(
        plan_id="test-eval",
        description="test",
        episodes=(ParallelEpisodeSpec("episode", "pick cube", "env", seed=7),),
        variants=(
            EvaluationVariant("A", runtime={}),
            EvaluationVariant(
                "B",
                runtime={"visual_history": {"include_vdm": False}},
            ),
        ),
        repeats=2,
        shuffle_seed=13,
        execution=EvaluationExecution(concurrency=2, max_attempts=2),
    )


def test_plan_compilation_is_paired_deterministic_and_hashed() -> None:
    plan = _plan()
    first = compile_evaluation_plan(plan)
    second = compile_evaluation_plan(plan)

    assert [job.job_id for job in first] == [job.job_id for job in second]
    assert len(first) == 4
    assert {job.pair_id for job in first} == {"episode-r000", "episode-r001"}
    assert all(job.spec.seed == 7 for job in first)
    assert {job.variant_id for job in first} == {"A", "B"}
    compiled = compiled_plan_payload(plan, first)
    assert compiled["schema_version"] == "openeta.compiled_evaluation.v1"
    assert len(compiled["plan_sha256"]) == 64


def test_visual_history_abc_plan_validates_and_compiles(capsys) -> None:
    path = Path("evaluations/visual_history_abc.json")
    plan = load_evaluation_plan(path)
    jobs = compile_evaluation_plan(plan)

    assert plan.plan_id == "visual-history-abc-v1"
    assert len(jobs) == 30
    assert {job.variant_id for job in jobs} == {
        "A_current_main_only",
        "B_bounded_raw_no_vdm",
        "C_bounded_raw_main_vdm",
    }
    assert eval_main(["validate", "--plan", str(path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["job_count"] == 30


def test_visual_history_canary_has_a_bounded_episode_budget() -> None:
    plan = load_evaluation_plan(Path("evaluations/visual_history_abc_canary.json"))
    jobs = compile_evaluation_plan(plan)

    assert len(jobs) == 3
    assert all(job.spec.max_turns == 60 for job in jobs)
    assert all(job.spec.max_tool_calls == 120 for job in jobs)


def test_eval_preflight_marks_incompatible_anygrasp_unavailable(monkeypatch) -> None:
    assert "ik_preview_check" in eval_cli._REQUIRED_SIM_MCP_TOOLS

    class SimulatorCatalog:
        def __init__(self, _url: str) -> None:
            pass

        def list_tools(self, *, timeout_s=None):
            del timeout_s
            return {
                "tools": [
                    {"name": name}
                    for name in eval_cli._REQUIRED_SIM_MCP_TOOLS
                ]
            }

    monkeypatch.setattr(
        eval_cli,
        "load_planner_provider_config",
        lambda: PlannerProviderConfig(
            model="fixture",
            api_base="http://provider.example/v1",
            api_key="test",
        ),
    )
    monkeypatch.setattr(eval_cli, "SseSimulatorMcpTransport", SimulatorCatalog)
    monkeypatch.setattr(
        eval_cli,
        "probe_object_memory_bank",
        lambda *_args, **_kwargs: {
            "configured": True,
            "checked": True,
            "available": True,
            "endpoint": "http://10.11.18.197:8080",
            "status": "ok",
        },
    )
    monkeypatch.setattr(
        eval_cli,
        "check_anygrasp_compatibility",
        lambda **_kwargs: {
            "backend": "anygrasp",
            "configured": True,
            "available": False,
            "compatible": False,
            "reason": "gripper_width_mismatch",
            "message": "AnyGrasp is unavailable: redeploy with 0.08 m geometry.",
        },
    )
    args = SimpleNamespace(
        model="",
        sim_url="http://sim.example/sse",
        sam3_url="",
        depth_prior_url="",
        anygrasp_url="http://anygrasp.example/sse",
        anyplace_url="",
        graspgenx_url="",
        molmopoint_url="",
        calibration_profile=str(DEFAULT_GRASP_PROFILE),
        skip_mcp_check=False,
        mcp_timeout_s=1.0,
    )

    result = eval_cli._remote_preflight(args)

    assert result["ok"] is True
    assert result["mcp"]["anygrasp"]["reason"] == "gripper_width_mismatch"
    assert result["warnings"][-1] == (
        "AnyGrasp is unavailable: redeploy with 0.08 m geometry."
    )


def test_eval_preflight_requires_object_memory_bank_even_when_mcp_checks_are_skipped(
    monkeypatch,
) -> None:
    class SimulatorCatalog:
        def __init__(self, _url: str) -> None:
            pass

        def list_tools(self, *, timeout_s=None):
            del timeout_s
            return {
                "tools": [
                    {"name": name}
                    for name in eval_cli._REQUIRED_SIM_MCP_TOOLS
                ]
            }

    monkeypatch.setattr(
        eval_cli,
        "load_planner_provider_config",
        lambda: PlannerProviderConfig(
            model="fixture",
            api_base="http://provider.example/v1",
            api_key="test",
        ),
    )
    monkeypatch.setattr(eval_cli, "SseSimulatorMcpTransport", SimulatorCatalog)
    monkeypatch.setattr(
        eval_cli,
        "probe_object_memory_bank",
        lambda *_args, **_kwargs: {
            "configured": True,
            "checked": True,
            "available": False,
            "endpoint": "http://memory.example",
            "reason": "connection_refused",
        },
    )
    args = SimpleNamespace(
        model="",
        sim_url="http://sim.example/sse",
        sam3_url="",
        depth_prior_url="",
        anygrasp_url="",
        anyplace_url="",
        graspgenx_url="",
        molmopoint_url="",
        calibration_profile=str(DEFAULT_GRASP_PROFILE),
        skip_mcp_check=True,
        mcp_timeout_s=1.0,
    )

    result = eval_cli._remote_preflight(args)

    assert result["ok"] is False
    assert result["mcp"]["object_memory"]["reason"] == "connection_refused"
    assert all("Memory Bank" not in warning for warning in result["warnings"])
    assert result["errors"] == [
        "required Object Memory Bank is unavailable from the evaluation worker: "
        "connection_refused. Restore the configured service before starting the "
        "evaluation."
    ]


def test_failed_required_preflight_does_not_create_a_run(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        eval_cli,
        "_remote_preflight",
        lambda _args: {
            "ok": False,
            "errors": ["required Object Memory Bank is unavailable"],
        },
    )
    args = SimpleNamespace(
        plan="evaluations/agent_first_milk_canary.json",
        root=str(tmp_path),
        run_id="must-not-exist",
    )

    with pytest.raises(ValueError, match="required Object Memory Bank"):
        eval_cli.run_plan(args)

    assert not (tmp_path / "must-not-exist").exists()


def test_scheduler_persists_each_attempt_and_retries_infrastructure(tmp_path: Path) -> None:
    plan = EvaluationPlan(
        plan_id="retry-eval",
        description="",
        episodes=(ParallelEpisodeSpec("episode", "pick cube", "env"),),
        variants=(EvaluationVariant("A"),),
        execution=EvaluationExecution(concurrency=1, max_attempts=2),
    )
    jobs = compile_evaluation_plan(plan)
    store = EvaluationRunStore.create(
        "run",
        root=tmp_path,
        compiled_plan=compiled_plan_payload(plan, jobs),
    )
    calls = {"count": 0}

    class Runner:
        def run(self, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                error = RuntimeError("provider unavailable")
                error.code = "provider_unavailable"
                raise error
            return EpisodeResult(
                task=kwargs["task"],
                session_id="session-success",
                terminated=True,
                metadata={"stop_reason": "task_complete"},
            )

    worker_factory = lambda spec, batch_id: ParallelEpisodeWorker(  # noqa: E731
        runner=Runner(),
        close=lambda: {"ok": True},
    )
    report = EvaluationScheduler(
        store=store,
        jobs=jobs,
        execution=plan.execution,
        worker_factory=worker_factory,
    ).run()

    job_dir = store.job_dir(jobs[0].job_id)
    assert (job_dir / "attempts/001/result.json").is_file()
    assert (job_dir / "attempts/002/result.json").is_file()
    assert (job_dir / "final.json").is_file()
    assert report["attempt_count"] == 2
    assert report["retry_count"] == 1
    assert report["status_counts"] == {"success": 1}
    assert store.run_metadata()["status"] == "complete"


def test_resume_reuses_abandoned_attempt_number(tmp_path: Path) -> None:
    plan = EvaluationPlan(
        plan_id="resume-eval",
        description="",
        episodes=(ParallelEpisodeSpec("episode", "pick cube", "env"),),
        variants=(EvaluationVariant("A"),),
        execution=EvaluationExecution(concurrency=1, max_attempts=1),
    )
    jobs = compile_evaluation_plan(plan)
    store = EvaluationRunStore.create(
        "run",
        root=tmp_path,
        compiled_plan=compiled_plan_payload(plan, jobs),
    )
    store.start_attempt(jobs[0], attempt=1, spec={"episode_id": jobs[0].job_id})

    class Runner:
        def run(self, **kwargs):
            return EpisodeResult(
                task=kwargs["task"],
                session_id="session",
                terminated=True,
                metadata={"stop_reason": "task_complete"},
            )

    report = EvaluationScheduler(
        store=store,
        jobs=jobs,
        execution=plan.execution,
        worker_factory=lambda spec, batch_id: ParallelEpisodeWorker(
            runner=Runner(), close=lambda: {"ok": True}
        ),
    ).run(resume=True)

    assert report["completed_count"] == 1
    assert store.next_attempt(jobs[0].job_id) == 3
    assert (store.attempt_dir(jobs[0].job_id, 1) / "state.json").is_file()
    assert (store.attempt_dir(jobs[0].job_id, 2) / "result.json").is_file()
    abandoned = json.loads(
        (store.attempt_dir(jobs[0].job_id, 1) / "state.json").read_text()
    )
    assert abandoned["status"] == "abandoned"


def test_failure_classification_separates_provider_task_and_resource() -> None:
    assert classify_evaluation_failure(
        {"status": "fail", "error": {"code": "provider_queue_timeout"}}
    )["retryable"] is True
    assert classify_evaluation_failure(
        {
            "status": "fail",
            "episode": {"metadata": {"stop_reason": "task_complete"}},
        }
    )["class"] == "task_failure"
    assert classify_evaluation_failure(
        {
            "status": "fail",
            "episode": {
                "metadata": {"failure_reason": {"code": "token_limit_exceeded"}}
            },
        }
    )["class"] == "resource_limit"
    provider_pause = classify_evaluation_failure(
        {
            "status": "need_human",
            "episode": {
                "steps": [
                    {
                        "action": {
                            "request_name": "ask_human",
                            "request_parameters": {
                                "message": "Planner provider request failed.",
                                "error_type": "TimeoutError",
                                "provider_attempts": 3,
                            },
                        }
                    }
                ]
            },
        }
    )
    assert provider_pause == {
        "class": "infrastructure",
        "stage": "provider",
        "code": "planner_provider_request_failed",
        "retryable": True,
        "error_type": "TimeoutError",
        "provider_attempts": 3,
    }
    quota_pause = classify_evaluation_failure(
        {
            "status": "need_human",
            "episode": {
                "steps": [
                    {
                        "action": {
                            "request_name": "ask_human",
                            "request_parameters": {
                                "message": "Planner provider request failed.",
                                "error_type": "ProviderHttpError",
                                "provider_attempts": 2,
                                "provider_error_code": (
                                    "insufficient_provider_quota"
                                ),
                                "retryable": False,
                            },
                        }
                    }
                ]
            },
        }
    )
    assert quota_pause == {
        "class": "external_dependency",
        "stage": "provider",
        "code": "insufficient_provider_quota",
        "retryable": False,
        "error_type": "ProviderHttpError",
        "provider_attempts": 2,
    }


def test_visual_history_extractor_reads_rollout_without_touching_generic_report(
    tmp_path: Path,
) -> None:
    plan = EvaluationPlan(
        plan_id="extract-eval",
        description="",
        episodes=(ParallelEpisodeSpec("episode", "pick cube", "env"),),
        variants=(EvaluationVariant("C"),),
    )
    jobs = compile_evaluation_plan(plan)
    store = EvaluationRunStore.create(
        "run",
        root=tmp_path,
        compiled_plan=compiled_plan_payload(plan, jobs),
    )
    rollout = (
        store.attempt_dir(jobs[0].job_id, 1)
        / "sessions"
        / "agent-session"
        / "rollout"
    )
    rollout.mkdir(parents=True)
    planner_context = {
        "schema_version": "openeta.agent_context.v2",
        "decision_state": {"schema_version": "openeta.decision_state.v1"},
        "visual_history": {
            "raw_evidence": [{"path": "a"}, {"path": "b"}],
            "compressed_deltas": [{"delta_id": "d"}],
        },
    }
    _write_jsonl(
        rollout / "model_calls.jsonl",
        [
            {
                "semantic_request": {"tool_context": planner_context, "metadata": {}},
                "parsed_decision": {
                    "kind": "tool_call",
                    "name": "grasp_pose_estimate",
                    "parameters": {"bundle_id": "grasp:host-issued"},
                },
                "validation": {"accepted": True},
                "result": {
                    "details": {
                        "usage": {
                            "prompt_tokens": 20,
                            "completion_tokens": 3,
                            "total_tokens": 23,
                        },
                        "provider_concurrency": {
                            "schema_version": "openeta.provider_concurrency.v1",
                            "limit": 2,
                            "request_count": 7,
                            "queue_timeout_count": 0,
                            "active": 1,
                            "total_queue_wait_s": 1.5,
                        },
                    }
                },
            },
            {
                "semantic_request": {
                    "tool_context": {},
                    "metadata": {"role": "visual_differencing"},
                },
                "duration_s": 0.5,
                "validation": {"accepted": True},
                "result": {"details": {"usage": {"total_tokens": 12}}},
            },
        ],
    )
    _write_jsonl(
        rollout / "transitions.jsonl",
        [
            {
                "action": {"command": {"kind": "tool_call", "name": "move_to"}},
                "reward": 1.0,
                "terminated": True,
                "truncated": False,
                "info": {"task_success": True},
            }
        ],
    )
    store.start_attempt(jobs[0], attempt=1, spec={"episode_id": jobs[0].job_id})
    (store.attempt_dir(jobs[0].job_id, 1) / "session_index.json").write_text(
        json.dumps(
            {
                "sessions": {
                    "agent-session": {
                        "session_id": "agent-session",
                        "metadata": {
                            "episode_id": jobs[0].job_id,
                            "evaluation": {"job_id": jobs[0].job_id},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    outcome = {
        "episode_id": jobs[0].job_id,
        "seed": 0,
        "status": "success",
        "duration_s": 1.0,
        "cleanup": {"ok": True},
        "episode": {
            "metadata": {
                "stop_reason": "task_complete",
                "usage": {"total_tokens": 35},
            },
            "steps": [
                {
                    "step_result": {
                        "reward": 1.0,
                        "info": {"task_success": True},
                    }
                }
            ],
        },
    }
    store.record_attempt_result(
        jobs[0],
        attempt=1,
        outcome=outcome,
        failure={"class": "none", "stage": "complete", "retryable": False},
        retryable=False,
        max_attempts=1,
    )
    final_path = store.job_dir(jobs[0].job_id) / "final.json"
    final_payload = json.loads(final_path.read_text(encoding="utf-8"))
    final_payload.update(started_at_s=10.0, completed_at_s=25.0)
    final_path.write_text(json.dumps(final_payload), encoding="utf-8")

    report = extract_visual_history_rollouts(store)

    assert report["variants"]["C"]["vdm_call_count"] == 1
    assert report["variants"]["C"]["vdm_total_tokens"] == 12
    assert report["variants"]["C"]["status_counts"] == {"success": 1}
    assert report["variants"]["C"]["stop_reason_counts"] == {"task_complete": 1}
    assert report["variants"]["C"]["planner_prompt_tokens"] == 20
    assert report["variants"]["C"]["planner_completion_tokens"] == 3
    assert report["variants"]["C"]["planner_total_tokens"] == 23
    assert report["variants"]["C"]["episode_total_tokens"] == 35
    assert report["variants"]["C"]["mean_episode_duration_s"] == 1.0
    assert report["variants"]["C"]["max_raw_visual_evidence_per_turn"] == 2
    assert report["variants"]["C"]["mean_compressed_delta_count_per_turn"] == 1.0
    assert report["jobs"][0]["mean_raw_visual_evidence_per_turn"] == 2.0
    assert report["jobs"][0]["decision_state_coverage"] == 1.0
    assert report["jobs"][0]["gate_block_count"] == 0
    assert report["jobs"][0]["repair_bundle_count"] == 0
    assert report["jobs"][0]["grasp_bundle_call_count"] == 1
    assert report["jobs"][0]["manual_grasp_input_call_count"] == 0
    assert report["jobs"][0]["max_compressed_delta_count_per_turn"] == 1
    assert report["terminal_job_count"] == 1
    assert report["partial_job_count"] == 0
    assert Path(report["state_probe_cases_path"]).is_file()

    generic_report = build_evaluation_report(store, jobs)
    assert generic_report["wall_clock_s"] == 15.0
    assert generic_report["provider_concurrency"]["request_count"] == 7
    assert generic_report["provider_concurrency"]["active"] == 0


def test_visual_history_extractor_includes_interrupted_partial_rollouts(
    tmp_path: Path,
) -> None:
    plan = EvaluationPlan(
        plan_id="partial-eval",
        description="",
        episodes=(ParallelEpisodeSpec("episode", "pick cube", "env", seed=4),),
        variants=(EvaluationVariant("C"),),
    )
    jobs = compile_evaluation_plan(plan)
    store = EvaluationRunStore.create(
        "partial-run",
        root=tmp_path,
        compiled_plan=compiled_plan_payload(plan, jobs),
    )
    store.start_attempt(jobs[0], attempt=1, spec={"episode_id": jobs[0].job_id})
    attempt = store.attempt_dir(jobs[0].job_id, 1)
    rollout = attempt / "sessions" / "partial-session" / "rollout"
    rollout.mkdir(parents=True)
    (attempt / "session_index.json").write_text(
        json.dumps(
            {
                "sessions": {
                    "partial-session": {
                        "metadata": {"evaluation": {"job_id": jobs[0].job_id}}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        rollout / "model_calls.jsonl",
        [
            {
                "semantic_request": {
                    "tool_context": {
                        "schema_version": "openeta.agent_context.v2",
                        "visual_history": {
                            "raw_evidence": [{"path": "current.png"}],
                            "compressed_deltas": [],
                        },
                    },
                    "metadata": {},
                },
                "parsed_decision": {
                    "kind": "tool_call",
                    "name": "anyplace",
                    "parameters": {"bundle_id": "anyplace:frozen"},
                },
                "result": {"details": {"usage": {"total_tokens": 9}}},
            }
        ],
    )
    _write_jsonl(rollout / "transitions.jsonl", [])
    store.set_run_status("interrupted", error={"type": "KeyboardInterrupt"})

    report = extract_visual_history_rollouts(store)

    assert report["job_count"] == 1
    assert report["terminal_job_count"] == 0
    assert report["partial_job_count"] == 1
    assert report["complete_pair_count"] == 0
    assert report["jobs"][0]["record_kind"] == "partial"
    assert report["jobs"][0]["status"] == "interrupted"
    assert report["jobs"][0]["stop_reason"] == "scheduler_interrupted"
    assert report["jobs"][0]["anyplace_bundle_call_count"] == 1
    assert report["jobs"][0]["episode_total_tokens"] == 9


def test_alternating_tool_cycle_count_detects_semantic_loops() -> None:
    assert _alternating_tool_cycle_count(
        ["sam3", "select_sam3_detection", "sam3", "select_sam3_detection"]
    ) == 2
    assert _alternating_tool_cycle_count(["sam3", "sam3", "move_to"]) == 0


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
