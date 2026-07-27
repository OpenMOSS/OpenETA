from __future__ import annotations

import json
from pathlib import Path

from agent.backends.planner import StaticPlannerBackend
from agent.runtime.calibration import (
    CALIBRATION_EVIDENCE_SCHEMA_VERSION,
    BackendCalibrationReviewer,
    CalibrationLifecycleConfig,
    CalibrationLifecycleManager,
    calibration_profile_sha256,
    collect_calibration_evidence,
    evaluate_validation_gates,
)
from agent.tools.registry import build_default_tool_registry


def _profile() -> dict:
    path = (
        Path(__file__).resolve().parents[1]
        / "agent"
        / "calibrations"
        / "candidate"
        / "graspnet-eef-panda-p8.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def _fingerprint() -> dict:
    return {
        "robot_model": "Panda",
        "gripper_model": "PandaGripper",
        "controller": "OSC_POSE",
        "environment": "LIBERO",
        "camera_calibration_id": "libero-agentview-panda-v1",
        "backend_versions": {
            "robosuite": "1.4.1",
            "mujoco": "3.3.0",
        },
    }


def _manager(
    tmp_path: Path,
    *,
    publication_mode: str = "independent_reviewer",
    review_decisions: list[dict] | None = None,
) -> CalibrationLifecycleManager:
    decisions = review_decisions or [
        {"decision": "approve", "reason": "proposal is scoped and coherent"},
        {"decision": "approve", "reason": "evidence supports candidate publication"},
        {"decision": "approve", "reason": "all validation gates pass"},
    ]
    return CalibrationLifecycleManager(
        config=CalibrationLifecycleConfig(
            root=tmp_path / "session-calibrations",
            candidate_dir=tmp_path / "repo" / "candidate",
            validated_dir=tmp_path / "repo" / "validated",
            evidence_roots=(tmp_path,),
            publication_mode=publication_mode,
        ),
        reviewer=BackendCalibrationReviewer(StaticPlannerBackend(decisions)),
    )


def _bind(manager: CalibrationLifecycleManager):
    tools = build_default_tool_registry()
    tools.bind_handler("propose_calibration_profile", manager.propose_handler)
    tools.bind_handler("promote_calibration_profile", manager.promote_handler)
    return tools


def _propose(tools) -> dict:
    result = tools.call(
        "propose_calibration_profile",
        {
            "profile": _profile(),
            "profile_fingerprint": _fingerprint(),
            "rationale": "Calibrate GraspNet seeds to the LIBERO Panda EEF.",
            "ledger": [
                {
                    "variant": "p8",
                    "verdict": "PASS",
                    "metric": "rigid_transform_fit",
                }
            ],
        },
        metadata={"session_id": "explore-session"},
    )
    assert result.success is True
    return result.details["outputs"]


def _evidence_file(
    path: Path,
    *,
    profile_sha256: str,
    split: str,
    attempts: int,
    successes: int,
    metrics: dict | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": CALIBRATION_EVIDENCE_SCHEMA_VERSION,
                "profile_sha256": profile_sha256,
                "split": split,
                "attempts": attempts,
                "successes": successes,
                "assisted_attempts": 0,
                "metrics": metrics or {},
            }
        ),
        encoding="utf-8",
    )
    return path


def _passing_evidence(tmp_path: Path, profile_sha256: str) -> list[dict]:
    metrics = {
        "finger_center_p95_m": 0.004,
        "axis_p95_deg": 2.5,
    }
    canary = _evidence_file(
        tmp_path / "canary.json",
        profile_sha256=profile_sha256,
        split="canary",
        attempts=2,
        successes=2,
        metrics=metrics,
    )
    held_out = _evidence_file(
        tmp_path / "held-out.json",
        profile_sha256=profile_sha256,
        split="held_out",
        attempts=20,
        successes=19,
        metrics=metrics,
    )
    return [
        {"path": str(canary), "split": "canary"},
        {"path": str(held_out), "split": "held_out"},
    ]


def test_proposal_is_reviewed_and_stays_session_local(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    tools = _bind(manager)

    outputs = _propose(tools)

    assert outputs["status"] == "reviewed"
    assert outputs["review"]["decision"] == "approve"
    profile_path = Path(outputs["profile_path"])
    proposal_path = Path(outputs["proposal_path"])
    assert profile_path.is_file()
    assert proposal_path.is_file()
    assert tmp_path / "repo" / "candidate" not in profile_path.parents
    staged = json.loads(profile_path.read_text(encoding="utf-8"))
    assert staged["status"] == "candidate"
    assert staged["profile_fingerprint"]["controller"] == "OSC_POSE"
    assert calibration_profile_sha256(staged) == outputs["profile_sha256"]


def test_candidate_then_validated_promotion_requires_evidence_and_review(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    tools = _bind(manager)
    proposal = _propose(tools)
    evidence = _passing_evidence(tmp_path, proposal["profile_sha256"])

    candidate = tools.call(
        "promote_calibration_profile",
        {
            "proposal_id": proposal["proposal_id"],
            "target_status": "candidate",
            "evidence": evidence,
        },
        metadata={"session_id": "explore-session"},
    )

    assert candidate.success is True
    candidate_outputs = candidate.details["outputs"]
    assert candidate_outputs["target_status"] == "candidate"
    assert candidate_outputs["published_profile_sha256"] == proposal["profile_sha256"]
    assert candidate_outputs["evidence_summary"]["metrics"]["held_out_success_rate"] == 0.95
    candidate_path = Path(candidate_outputs["target_path"])
    assert candidate_path.is_file()
    assert json.loads(candidate_path.read_text(encoding="utf-8"))["status"] == "candidate"

    replay = tools.call(
        "promote_calibration_profile",
        {
            "proposal_id": proposal["proposal_id"],
            "target_status": "candidate",
            "evidence": evidence,
        },
        metadata={"session_id": "explore-session"},
    )
    assert replay.success is True
    assert replay.details["outputs"]["idempotent_replay"] is True
    assert replay.details["outputs"]["target_path"] == str(candidate_path)

    validated = tools.call(
        "promote_calibration_profile",
        {
            "proposal_id": proposal["proposal_id"],
            "target_status": "validated",
            "evidence": evidence,
        },
        metadata={"session_id": "explore-session"},
    )

    assert validated.success is True
    validated_outputs = validated.details["outputs"]
    assert validated_outputs["gate_report"]["passed"] is True
    assert validated_outputs["published_profile_sha256"] == proposal["profile_sha256"]
    validated_path = Path(validated_outputs["target_path"])
    assert validated_path.parent.name == "validated"
    assert json.loads(validated_path.read_text(encoding="utf-8"))["status"] == "validated"


def test_standard_profile_cannot_publish_shared_candidate(tmp_path: Path) -> None:
    manager = _manager(tmp_path, publication_mode="runtime_session_only")
    tools = _bind(manager)
    proposal = _propose(tools)
    evidence = _passing_evidence(tmp_path, proposal["profile_sha256"])

    result = tools.call(
        "promote_calibration_profile",
        {
            "proposal_id": proposal["proposal_id"],
            "target_status": "candidate",
            "evidence": evidence,
        },
        metadata={"session_id": "explore-session"},
    )

    assert result.success is False
    assert result.details["outputs"]["reason"] == "calibration_promotion_failed"
    assert "session-local proposals only" in result.content
    assert not (tmp_path / "repo" / "candidate").exists()


def test_validated_requires_candidate_and_all_machine_gates(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    tools = _bind(manager)
    proposal = _propose(tools)
    evidence = _passing_evidence(tmp_path, proposal["profile_sha256"])

    too_early = tools.call(
        "promote_calibration_profile",
        {
            "proposal_id": proposal["proposal_id"],
            "target_status": "validated",
            "evidence": evidence,
        },
        metadata={"session_id": "explore-session"},
    )

    assert too_early.success is False
    assert "published candidate first" in too_early.content

    candidate = tools.call(
        "promote_calibration_profile",
        {
            "proposal_id": proposal["proposal_id"],
            "target_status": "candidate",
            "evidence": evidence,
        },
        metadata={"session_id": "explore-session"},
    )
    assert candidate.success is True
    failed_held_out = _evidence_file(
        tmp_path / "failed-held-out.json",
        profile_sha256=proposal["profile_sha256"],
        split="held_out",
        attempts=20,
        successes=18,
        metrics={
            "finger_center_p95_m": 0.004,
            "axis_p95_deg": 2.5,
        },
    )
    failed_evidence = [
        evidence[0],
        {"path": str(failed_held_out), "split": "held_out"},
    ]

    rejected = tools.call(
        "promote_calibration_profile",
        {
            "proposal_id": proposal["proposal_id"],
            "target_status": "validated",
            "evidence": failed_evidence,
        },
        metadata={"session_id": "explore-session"},
    )

    assert rejected.success is False
    assert rejected.details["outputs"]["reason"] == "calibration_promotion_gate_failed"
    assert "held_out_success_rate=0.9" in rejected.content


def test_evidence_rejects_wrong_profile_hash_and_missing_coverage(tmp_path: Path) -> None:
    wrong = _evidence_file(
        tmp_path / "wrong.json",
        profile_sha256="0" * 64,
        split="canary",
        attempts=2,
        successes=2,
    )

    try:
        collect_calibration_evidence(
            [{"path": str(wrong), "split": "canary"}],
            expected_profile_sha256="1" * 64,
            validation_gates=[
                {
                    "metric": "held_out_success_rate",
                    "operator": ">=",
                    "value": 0.95,
                }
            ],
            allowed_roots=(tmp_path,),
        )
    except ValueError as exc:
        assert "does not match proposal" in str(exc)
    else:
        raise AssertionError("wrong profile hash must be rejected")


def test_evidence_cannot_duplicate_sources_or_override_host_metrics(
    tmp_path: Path,
) -> None:
    profile_sha256 = "1" * 64
    source = _evidence_file(
        tmp_path / "source.json",
        profile_sha256=profile_sha256,
        split="canary",
        attempts=2,
        successes=2,
    )
    gate = [
        {
            "metric": "held_out_success_rate",
            "operator": ">=",
            "value": 0.95,
        }
    ]

    try:
        collect_calibration_evidence(
            [
                {"path": str(source), "split": "canary"},
                {"path": str(source), "split": "canary"},
            ],
            expected_profile_sha256=profile_sha256,
            validation_gates=gate,
            allowed_roots=(tmp_path,),
        )
    except ValueError as exc:
        assert "duplicate calibration evidence path" in str(exc)
    else:
        raise AssertionError("duplicate evidence must not inflate attempt counts")

    reserved = _evidence_file(
        tmp_path / "reserved.json",
        profile_sha256=profile_sha256,
        split="held_out",
        attempts=1,
        successes=0,
        metrics={"held_out_success_rate": 1.0},
    )
    try:
        collect_calibration_evidence(
            [{"path": str(reserved), "split": "held_out"}],
            expected_profile_sha256=profile_sha256,
            validation_gates=gate,
            allowed_roots=(tmp_path,),
        )
    except ValueError as exc:
        assert "host-derived and reserved" in str(exc)
    else:
        raise AssertionError("custom metrics must not override host-derived rates")


def test_gate_evaluation_fails_closed_for_missing_metric() -> None:
    report = evaluate_validation_gates(
        [
            {"metric": "axis_p95_deg", "operator": "<=", "value": 3.0},
            {"metric": "held_out_success_rate", "operator": ">=", "value": 0.95},
        ],
        {
            "metrics": {
                "held_out_success_rate": 1.0,
            }
        },
    )

    assert report["passed"] is False
    assert report["results"][0]["observed"] is None
    assert report["failures"] == ["axis_p95_deg is missing"]


def test_reviewer_rejection_keeps_auditable_blocked_proposal(tmp_path: Path) -> None:
    manager = _manager(
        tmp_path,
        review_decisions=[
            {
                "decision": "reject",
                "reason": "camera calibration identity is contradictory",
            }
        ],
    )
    tools = _bind(manager)

    result = tools.call(
        "propose_calibration_profile",
        {
            "profile": _profile(),
            "profile_fingerprint": _fingerprint(),
            "rationale": "Test a candidate transform.",
        },
        metadata={"session_id": "explore-session"},
    )

    assert result.success is False
    outputs = result.details["outputs"]
    assert outputs["reason"] == "calibration_review_blocked"
    proposal = json.loads(Path(outputs["proposal_path"]).read_text(encoding="utf-8"))
    assert proposal["status"] == "review_blocked"
    assert proposal["review"]["decision"] == "reject"
