from __future__ import annotations

from sim.envs.univtac.seed_eligibility import (
    ResetEligibilityInput,
    select_first_three_valid,
    summarize_seed_eligibility,
)


def _record(seed: int, *, valid: bool, **extra) -> ResetEligibilityInput:
    payload = {
        "seed": seed,
        "reset_exception": False,
        "plan_success": valid,
        "tactile_sensor_count": 2 if valid else 0,
        "rgb_marker_valid": valid,
        "failure_stage": None if valid else "reset_pre_move_planner",
        "planner_status": None if valid else "IK_FAIL",
        "failed_move_call_index": None if valid else 4,
        "failed_planning_call_index": None if valid else 4,
    }
    payload.update(extra)
    return ResetEligibilityInput(**payload)


def test_first_three_valid_are_selected_in_ascending_candidate_order() -> None:
    records = [
        _record(0, valid=False),
        _record(1, valid=True),
        _record(2, valid=False),
        _record(3, valid=True),
        _record(4, valid=True),
        _record(5, valid=True),
    ]

    assert select_first_three_valid(records, range(10)) == [1, 3, 4]


def test_reset_invalid_seed_never_enters_selected_seeds() -> None:
    records = [
        _record(0, valid=False),
        _record(1, valid=True),
        _record(2, valid=True),
        _record(3, valid=True),
    ]
    summary = summarize_seed_eligibility(records, range(10))

    assert summary["selected_smoke_seeds"] == [1, 2, 3]
    assert 0 in summary["reset_invalid_seeds"]


def test_agent_success_and_check_success_do_not_affect_reset_eligibility() -> None:
    base = {
        "seed": 2,
        "reset_exception": False,
        "plan_success": True,
        "tactile_sensor_count": 2,
        "rgb_marker_valid": True,
        "agent_success": False,
        "check_success": False,
    }

    record = ResetEligibilityInput.from_mapping(base)
    assert record.reset_valid is True
    assert select_first_three_valid([record], [2], required=1) == [2]


def test_missing_tactile_packet_makes_otherwise_clean_reset_invalid() -> None:
    record = _record(
        0,
        valid=True,
        tactile_sensor_count=1,
        rgb_marker_valid=False,
        failure_stage="observation_contract",
    )

    assert record.reset_valid is False
    assert select_first_three_valid([record], [0], required=1) == []
