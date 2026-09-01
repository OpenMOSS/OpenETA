from __future__ import annotations

from sim.reachability import (
    ROBUST_EXECUTION_JOINT_MARGIN_RAD,
    _execution_seed_search_summary,
    _select_execution_seed,
)


def _candidate(*, margin: float, travel: float, marker: str) -> dict:
    return {
        "joint_margin_min_rad": margin,
        "joint_travel_l2_rad": travel,
        "marker": marker,
    }


def test_execution_seed_prefers_robust_branch_over_nearer_fragile_branch() -> None:
    fragile = _candidate(margin=0.005, travel=0.1, marker="fragile")
    robust = _candidate(
        margin=ROBUST_EXECUTION_JOINT_MARGIN_RAD,
        travel=0.8,
        marker="robust",
    )

    selected = _select_execution_seed([fragile, robust])

    assert selected["marker"] == "robust"


def test_execution_seed_minimizes_travel_after_robust_margin_is_met() -> None:
    farther = _candidate(margin=0.4, travel=1.2, marker="farther")
    nearer = _candidate(margin=0.2, travel=0.3, marker="nearer")

    selected = _select_execution_seed([farther, nearer])

    assert selected["marker"] == "nearer"


def test_execution_seed_maximizes_margin_when_all_branches_are_fragile() -> None:
    nearer = _candidate(margin=0.02, travel=0.1, marker="nearer")
    safer = _candidate(margin=0.08, travel=0.9, marker="safer")

    selected = _select_execution_seed([nearer, safer])
    summary = _execution_seed_search_summary(
        [nearer, safer],
        selected=selected,
        optional_search_timed_out=False,
    )

    assert selected["marker"] == "safer"
    assert summary["feasible_solution_count"] == 2
    assert summary["fragile_solution_count"] == 2
    assert summary["robust_solution_selected"] is False


def test_execution_seed_does_not_use_distant_robust_branch_as_local_posture_seed() -> None:
    local_fragile = _candidate(margin=0.08, travel=0.4, marker="local")
    distant_robust = _candidate(margin=0.4, travel=4.0, marker="distant")

    selected = _select_execution_seed([local_fragile, distant_robust])
    summary = _execution_seed_search_summary(
        [local_fragile, distant_robust],
        selected=selected,
        optional_search_timed_out=False,
    )

    assert selected["marker"] == "local"
    assert summary["robust_solution_selected"] is False
    assert summary["distant_robust_solution_count"] == 1
    assert summary["candidate_summaries"][1]["within_local_travel_envelope"] is False
