"""Reset-only seed eligibility rules for UniVTAC smoke qualification."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


SEED_ELIGIBILITY_SCHEMA_VERSION = "openeta.univtac.seed_eligibility.v1"
SELECTION_RULE = (
    "ascending first-three seeds with exception-free reset, native plan_success=true, "
    "exactly two tactile sensors, and valid rgb_marker packets"
)


@dataclass(frozen=True)
class ResetEligibilityInput:
    seed: int
    reset_exception: bool
    plan_success: bool
    tactile_sensor_count: int
    rgb_marker_valid: bool
    failure_stage: str | None = None
    planner_status: str | None = None
    failed_move_call_index: int | None = None
    failed_planning_call_index: int | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ResetEligibilityInput":
        return cls(
            seed=int(payload["seed"]),
            reset_exception=bool(payload.get("reset_exception", False)),
            plan_success=bool(payload.get("plan_success", False)),
            tactile_sensor_count=int(payload.get("tactile_sensor_count", 0)),
            rgb_marker_valid=bool(payload.get("rgb_marker_valid", False)),
            failure_stage=payload.get("failure_stage"),
            planner_status=payload.get("planner_status"),
            failed_move_call_index=payload.get("failed_move_call_index"),
            failed_planning_call_index=payload.get("failed_planning_call_index"),
        )

    @property
    def reset_valid(self) -> bool:
        return (
            not self.reset_exception
            and self.plan_success is True
            and self.tactile_sensor_count == 2
            and self.rgb_marker_valid is True
        )

    @property
    def failure_signature(self) -> tuple[Any, ...] | None:
        if self.reset_valid:
            return None
        return (
            self.failure_stage,
            self.planner_status,
            self.failed_move_call_index,
            self.failed_planning_call_index,
        )


def _validated_candidate_order(candidate_order: Iterable[int]) -> list[int]:
    values = [int(seed) for seed in candidate_order]
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError("candidate seeds must be unique and strictly ascending")
    return values


def select_first_three_valid(
    records: Sequence[ResetEligibilityInput | Mapping[str, Any]],
    candidate_order: Iterable[int],
    *,
    required: int = 3,
) -> list[int]:
    order = _validated_candidate_order(candidate_order)
    by_seed = {
        record.seed: record
        for record in (
            item
            if isinstance(item, ResetEligibilityInput)
            else ResetEligibilityInput.from_mapping(item)
            for item in records
        )
    }
    selected: list[int] = []
    for seed in order:
        record = by_seed.get(seed)
        if record is not None and record.reset_valid:
            selected.append(seed)
            if len(selected) == required:
                break
    return selected


def suspected_systematic_failure(
    records: Sequence[ResetEligibilityInput | Mapping[str, Any]],
    *,
    first_batch_size: int = 6,
) -> bool:
    normalized = [
        item
        if isinstance(item, ResetEligibilityInput)
        else ResetEligibilityInput.from_mapping(item)
        for item in records
    ]
    first_batch = normalized[:first_batch_size]
    if len(first_batch) < first_batch_size or any(item.reset_valid for item in first_batch):
        return False
    signatures = [item.failure_signature for item in first_batch]
    return len(set(signatures)) == 1 and signatures[0] is not None


def should_stop_scan(
    records: Sequence[ResetEligibilityInput | Mapping[str, Any]],
    candidate_order: Iterable[int],
    *,
    required: int = 3,
) -> bool:
    selected = select_first_three_valid(records, candidate_order, required=required)
    if len(selected) >= required:
        return True
    return suspected_systematic_failure(records)


def summarize_seed_eligibility(
    records: Sequence[ResetEligibilityInput | Mapping[str, Any]],
    candidate_order: Iterable[int],
    *,
    required: int = 3,
) -> dict[str, Any]:
    order = _validated_candidate_order(candidate_order)
    normalized = [
        item
        if isinstance(item, ResetEligibilityInput)
        else ResetEligibilityInput.from_mapping(item)
        for item in records
    ]
    by_seed = {item.seed: item for item in normalized}
    attempted = [seed for seed in order if seed in by_seed]
    valid = [seed for seed in attempted if by_seed[seed].reset_valid]
    invalid = [seed for seed in attempted if not by_seed[seed].reset_valid]
    selected = select_first_three_valid(normalized, order, required=required)
    failure_stages = Counter(
        by_seed[seed].failure_stage or "unknown" for seed in invalid
    )
    planner_statuses = Counter(
        by_seed[seed].planner_status or "unavailable" for seed in invalid
    )
    return {
        "schema_version": SEED_ELIGIBILITY_SCHEMA_VERSION,
        "candidate_order": order,
        "attempted_seeds": attempted,
        "reset_valid_seeds": valid,
        "reset_invalid_seeds": invalid,
        "selected_smoke_seeds": selected,
        "selection_rule": SELECTION_RULE,
        "failure_stage_histogram": dict(sorted(failure_stages.items())),
        "planner_status_histogram": dict(sorted(planner_statuses.items())),
        "suspected_systematic": suspected_systematic_failure(normalized),
        "required_valid_seed_count": int(required),
    }
