from __future__ import annotations

import json
from pathlib import Path

from agent.backends.planner import StaticPlannerBackend
from agent.runtime.calibration import calibration_profile_sha256
from agent.runtime.grasp_strategy_lifecycle import (
    GRASP_STRATEGY_EVIDENCE_SCHEMA_VERSION,
    BackendGraspStrategyAuthor,
    BackendGraspStrategyReviewer,
    GraspStrategyLifecycleConfig,
    GraspStrategyLifecycleManager,
    collect_grasp_strategy_evidence,
    evaluate_grasp_strategy_evidence,
)
from agent.tools.grasp_geometry import DEFAULT_GRASP_PROFILE
from agent.tools.grasp_strategies import (
    DEFAULT_GRASP_STRATEGY_ROOT,
    grasp_strategy_sha256,
    load_grasp_strategies,
)
from agent.tools.registry import build_default_tool_registry


def _calibration() -> dict:
    return json.loads(DEFAULT_GRASP_PROFILE.read_text(encoding="utf-8"))


def _strategy() -> dict:
    return {
        "schema_version": "openeta.grasp_strategy.v1",
        "status": "candidate",
        "strategy_id": "side-grasp-round-panda-r1",
        "strategy_family_id": "side-grasp-round-panda",
        "revision": 1,
        "description": "Preserve estimator pose for round non-upright objects.",
        "compatibility": {"calibration_ids": ["graspnet-eef-panda-p8"]},
        "automatic_activation": {"target_geometry_families": ["bowl", "apple"]},
        "validated_scope": {"target_geometry_families": ["bowl", "apple"]},
        "constraints": {
            "grasp_width_bounds_m": [0.02, 0.08],
            "clearance_m": 0.01,
        },
        "pose_policy": {
            "approach_axis": "preserve_candidate",
            "orientation": "preserve_candidate",
        },
        "provenance": {"intended_use": "test candidate"},
    }


def _manager(
    tmp_path: Path,
    *,
    publication_mode: str = "independent_reviewer",
) -> GraspStrategyLifecycleManager:
    reviewer = BackendGraspStrategyReviewer(
        StaticPlannerBackend(
            [
                {"decision": "approve", "reason": "bounded proposal"},
                {"decision": "approve", "reason": "canary passes"},
                {"decision": "approve", "reason": "held-out passes"},
            ]
        )
    )
    return GraspStrategyLifecycleManager(
        config=GraspStrategyLifecycleConfig(
            root=tmp_path / "lifecycle",
            candidate_dir=tmp_path / "shared" / "candidate",
            validated_dir=tmp_path / "shared" / "validated",
            session_strategy_root=tmp_path / "session-strategies",
            evidence_roots=(tmp_path,),
            publication_mode=publication_mode,
            min_canary_attempts=2,
            min_held_out_attempts=4,
            min_held_out_success_rate=0.75,
            min_held_out_task_count=2,
        ),
        reviewer=reviewer,
    )


def _evidence(
    path: Path,
    *,
    strategy_sha256: str,
    split: str,
    attempts: int,
    successes: int,
    task_ids: list[str],
    regressed: list[str] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": GRASP_STRATEGY_EVIDENCE_SCHEMA_VERSION,
                "producer": "openeta_experiment_host",
                "strategy_sha256": strategy_sha256,
                "calibration_profile_sha256": (
                    calibration_profile_sha256(_calibration())
                ),
                "split": split,
                "attempts": attempts,
                "successes": successes,
                "baseline_attempts": attempts,
                "baseline_successes": successes,
                "task_ids": task_ids,
                "seeds": list(range(attempts)),
                "regressed_episode_ids": regressed or [],
                "safety_violations": 0,
                "contract_violations": 0,
                "human_interventions": 0,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_strategy_proposal_stages_session_local_then_promotes_with_evidence(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    tools = build_default_tool_registry()
    tools.bind_handler("propose_grasp_strategy", manager.propose_handler)
    tools.bind_handler("promote_grasp_strategy", manager.promote_handler)

    proposed = tools.call(
        "propose_grasp_strategy",
        {
            "strategy": _strategy(),
            "calibration_profile": _calibration(),
            "rationale": "Round objects need an estimator-preserving strategy.",
        },
        metadata={"session_id": "strategy-session"},
    )

    assert proposed.success is True
    outputs = proposed.details["outputs"]
    staged = Path(outputs["session_strategy_path"])
    assert staged.is_file()
    assert staged.is_relative_to(tmp_path / "session-strategies")
    strategy_sha = outputs["strategy_sha256"]
    canary = _evidence(
        tmp_path / "canary.json",
        strategy_sha256=strategy_sha,
        split="canary",
        attempts=2,
        successes=2,
        task_ids=["task-a"],
    )
    candidate = tools.call(
        "promote_grasp_strategy",
        {
            "proposal_id": outputs["proposal_id"],
            "target_status": "candidate",
            "evidence": [{"path": str(canary), "split": "canary"}],
        },
        metadata={"session_id": "strategy-session"},
    )
    assert candidate.success is True
    held_out = _evidence(
        tmp_path / "held-out.json",
        strategy_sha256=strategy_sha,
        split="held_out",
        attempts=4,
        successes=4,
        task_ids=["task-b", "task-c"],
    )
    validated = tools.call(
        "promote_grasp_strategy",
        {
            "proposal_id": outputs["proposal_id"],
            "target_status": "validated",
            "evidence": [
                {"path": str(canary), "split": "canary"},
                {"path": str(held_out), "split": "held_out"},
            ],
        },
        metadata={"session_id": "strategy-session"},
    )
    assert validated.success is True
    assert Path(validated.details["outputs"]["target_path"]).is_file()

    effective = load_grasp_strategies(tmp_path / "shared")
    selected = [item for item in effective if item["strategy_id"] == _strategy()["strategy_id"]]
    assert len(selected) == 1
    assert selected[0]["status"] == "validated"


def test_strategy_publication_is_blocked_in_standard_mode(tmp_path: Path) -> None:
    manager = _manager(tmp_path, publication_mode="runtime_session_only")
    proposal = manager.create_proposal(
        session_id="strategy-session",
        strategy=_strategy(),
        calibration_profile=_calibration(),
        rationale="test",
    )
    canary = _evidence(
        tmp_path / "canary.json",
        strategy_sha256=proposal["strategy_sha256"],
        split="canary",
        attempts=2,
        successes=2,
        task_ids=["task-a"],
    )

    try:
        manager.promote(
            proposal=proposal,
            target_status="candidate",
            evidence_references=[{"path": str(canary), "split": "canary"}],
        )
    except PermissionError as exc:
        assert "session-local" in str(exc)
    else:
        raise AssertionError("standard mode must not publish shared strategies")


def test_strategy_evidence_rejects_paired_regression(tmp_path: Path) -> None:
    strategy_sha = grasp_strategy_sha256(_strategy())
    evidence_path = _evidence(
        tmp_path / "regressed.json",
        strategy_sha256=strategy_sha,
        split="canary",
        attempts=2,
        successes=1,
        task_ids=["task-a"],
        regressed=["episode-1"],
    )
    evidence = collect_grasp_strategy_evidence(
        [{"path": str(evidence_path), "split": "canary"}],
        expected_strategy_sha256=strategy_sha,
        expected_calibration_sha256=(
            calibration_profile_sha256(_calibration())
        ),
        allowed_roots=(tmp_path,),
    )

    gates = evaluate_grasp_strategy_evidence(
        evidence,
        target_status="candidate",
        min_canary_attempts=2,
        min_held_out_attempts=4,
        min_held_out_success_rate=0.75,
        min_held_out_task_count=2,
    )
    assert gates["passed"] is False
    assert "paired objective episodes regressed" in gates["failures"]


def test_strategy_evidence_rejects_session_sandbox_file(tmp_path: Path) -> None:
    sandbox = tmp_path / "sessions" / "session-a" / "sandbox"
    sandbox.mkdir(parents=True)
    path = _evidence(
        sandbox / "forged.json",
        strategy_sha256=grasp_strategy_sha256(_strategy()),
        split="canary",
        attempts=2,
        successes=2,
        task_ids=["task-a"],
    )

    try:
        collect_grasp_strategy_evidence(
            [{"path": str(path), "split": "canary"}],
            expected_strategy_sha256=grasp_strategy_sha256(_strategy()),
            expected_calibration_sha256=calibration_profile_sha256(_calibration()),
            allowed_roots=(tmp_path,),
        )
    except PermissionError as exc:
        assert "host-owned" in str(exc)
    else:
        raise AssertionError("session sandbox evidence must fail closed")


def test_strategy_author_uses_clean_context_and_validates_revision() -> None:
    current = load_grasp_strategies(DEFAULT_GRASP_STRATEGY_ROOT)
    previous = current[0]
    authored = _strategy()
    authored.update(
        {
            "strategy_id": "top-down-vertical-panda-p9",
            "strategy_family_id": previous["strategy_family_id"],
            "revision": int(previous["revision"]) + 1,
            "supersedes": previous["strategy_id"],
        }
    )
    author = BackendGraspStrategyAuthor(
        StaticPlannerBackend(
            {
                "decision": "propose",
                "reason": "repeatable task-family evidence",
                "strategy": authored,
            }
        )
    )

    result = author.author(
        current_strategies=current,
        calibration_profile=_calibration(),
        rollout_summary={"objective_success_rate": 0.5},
    )

    assert result.decision == "propose"
    assert result.strategy is not None
    assert result.strategy["supersedes"] == previous["strategy_id"]


def test_strategy_author_accepts_json_string_backend_payload() -> None:
    author = BackendGraspStrategyAuthor(
        StaticPlannerBackend(
            json.dumps(
                {
                    "decision": "no_change",
                    "reason": "insufficient evidence",
                    "strategy": None,
                }
            )
        )
    )

    result = author.author(
        current_strategies=load_grasp_strategies(DEFAULT_GRASP_STRATEGY_ROOT),
        calibration_profile=_calibration(),
        rollout_summary={"objective_success_rate": 0.0},
    )

    assert result.decision == "no_change"
    assert result.strategy is None


def test_strategy_reviewer_accepts_json_string_backend_payload() -> None:
    reviewer = BackendGraspStrategyReviewer(
        StaticPlannerBackend(
            json.dumps({"decision": "approve", "reason": "bounded proposal"})
        )
    )

    result = reviewer.review(
        proposal={"strategy": _strategy()},
        requested_stage="candidate",
        deterministic_checks={"passed": True},
        evidence=None,
    )

    assert result.approved is True
    assert result.decision == "approve"
