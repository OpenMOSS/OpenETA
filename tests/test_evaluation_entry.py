from __future__ import annotations

import json
from pathlib import Path

from agent.cli.eval import main as eval_main
from agent.evals.plan import (
    EvaluationExecution,
    EvaluationPlan,
    EvaluationVariant,
    compile_evaluation_plan,
    compiled_plan_payload,
    load_evaluation_plan,
)
from agent.evals.runner import EvaluationScheduler, classify_evaluation_failure
from agent.evals.store import EvaluationRunStore
from agent.evals.visual_history_rollout import (
    _alternating_tool_cycle_count,
    extract_visual_history_rollouts,
)
from agent.runtime.episode import EpisodeResult
from agent.runtime.parallel import ParallelEpisodeSpec, ParallelEpisodeWorker


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
                "parsed_decision": {"tool": "move_to"},
                "validation": {"accepted": True},
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
            "metadata": {},
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

    report = extract_visual_history_rollouts(store)

    assert report["variants"]["C"]["vdm_call_count"] == 1
    assert report["variants"]["C"]["vdm_total_tokens"] == 12
    assert report["jobs"][0]["mean_raw_visual_evidence_per_turn"] == 2.0
    assert Path(report["state_probe_cases_path"]).is_file()


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
