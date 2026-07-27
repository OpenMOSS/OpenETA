"""Structured, task-family grasp strategy loading and deterministic selection."""

from __future__ import annotations

import json
import hashlib
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict


GRASP_STRATEGY_SCHEMA_VERSION = "openeta.grasp_strategy.v1"
DEFAULT_GRASP_STRATEGY_ROOT = (
    Path(__file__).resolve().parents[1] / "strategies" / "grasp"
)


class GraspStrategyError(ValueError):
    """Raised when a task strategy is malformed or physically incompatible."""


def load_grasp_strategies(
    root: str | Path = DEFAULT_GRASP_STRATEGY_ROOT,
) -> list[JsonDict]:
    """Load candidate and validated strategies with stable ordering."""

    strategy_root = Path(root)
    strategies: list[JsonDict] = []
    seen: dict[str, str] = {}
    for status in ("validated", "candidate"):
        for path in sorted((strategy_root / status).glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            strategy = validate_grasp_strategy(payload)
            strategy_id = str(strategy["strategy_id"])
            previous_status = seen.get(strategy_id)
            if previous_status == "validated" and status == "candidate":
                continue
            if previous_status is not None:
                raise GraspStrategyError(f"duplicate grasp strategy id: {strategy_id}")
            seen[strategy_id] = status
            strategy["_source_path"] = str(path.resolve())
            strategies.append(strategy)
    return strategies


def validate_grasp_strategy(value: Mapping[str, Any]) -> JsonDict:
    """Validate one strategy independently of any semantic task label."""

    strategy = json.loads(json.dumps(dict(value)))
    if strategy.get("schema_version") != GRASP_STRATEGY_SCHEMA_VERSION:
        raise GraspStrategyError("unsupported grasp strategy schema")
    allowed_fields = {
        "schema_version",
        "status",
        "strategy_id",
        "strategy_family_id",
        "revision",
        "supersedes",
        "description",
        "compatibility",
        "automatic_activation",
        "validated_scope",
        "constraints",
        "candidate_filter",
        "alignment_policy",
        "motion_policy",
        "pose_policy",
        "provenance",
        "lifecycle",
        "_source_path",
    }
    unknown = set(strategy) - allowed_fields
    if unknown:
        raise GraspStrategyError(
            "grasp strategy has forbidden fields: " + ", ".join(sorted(unknown))
        )
    strategy_id = str(strategy.get("strategy_id") or "").strip()
    if not strategy_id:
        raise GraspStrategyError("strategy_id is required")
    if strategy.get("status") not in {"candidate", "validated"}:
        raise GraspStrategyError("strategy status must be candidate or validated")
    family_id = str(strategy.get("strategy_family_id") or strategy_id).strip()
    if not family_id:
        raise GraspStrategyError("strategy_family_id is required")
    revision = strategy.get("revision", 1)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise GraspStrategyError("strategy revision must be a positive integer")
    strategy["strategy_family_id"] = family_id
    strategy["revision"] = revision
    compatibility = _mapping(strategy.get("compatibility"), "compatibility")
    calibration_ids = compatibility.get("calibration_ids")
    if not isinstance(calibration_ids, list) or not all(
        isinstance(item, str) and item.strip() for item in calibration_ids
    ):
        raise GraspStrategyError("compatibility.calibration_ids must be strings")
    constraints = _mapping(strategy.get("constraints"), "constraints")
    width_bounds = _vector(
        constraints.get("grasp_width_bounds_m"),
        2,
        "constraints.grasp_width_bounds_m",
    )
    if not 0.0 <= width_bounds[0] < width_bounds[1] <= 0.2:
        raise GraspStrategyError("strategy grasp width bounds are invalid")
    pose_policy = _mapping(strategy.get("pose_policy"), "pose_policy")
    if pose_policy.get("orientation") not in {
        "preserve_candidate",
        "top_down",
        "top_down_preserve_yaw",
    }:
        raise GraspStrategyError("unsupported strategy orientation policy")
    if pose_policy.get("approach_axis") not in {
        "preserve_candidate",
        "world_-Z",
    }:
        raise GraspStrategyError("unsupported strategy approach policy")
    candidate_filter = strategy.get("candidate_filter", {})
    if not isinstance(candidate_filter, dict):
        raise GraspStrategyError("candidate_filter must be an object")
    if "min_downward_alignment" in candidate_filter:
        alignment = _finite_float(
            candidate_filter["min_downward_alignment"],
            "candidate_filter.min_downward_alignment",
        )
        if not 0.0 <= alignment <= 1.0:
            raise GraspStrategyError(
                "candidate_filter.min_downward_alignment must be in [0, 1]"
            )
    alignment_policy = strategy.get("alignment_policy", {})
    if not isinstance(alignment_policy, dict):
        raise GraspStrategyError("alignment_policy must be an object")
    if alignment_policy.get("target_region", "mask_centroid") not in {
        "mask_centroid",
        "nearest_shallow_surface",
    }:
        raise GraspStrategyError("unsupported alignment target region")
    motion_policy = strategy.get("motion_policy", {})
    if not isinstance(motion_policy, dict):
        raise GraspStrategyError("motion_policy must be an object")
    if "precontact_distance_m" in motion_policy:
        precontact = _finite_float(
            motion_policy["precontact_distance_m"],
            "motion_policy.precontact_distance_m",
        )
        if not 0.01 <= precontact <= 0.10:
            raise GraspStrategyError(
                "motion_policy.precontact_distance_m must be in [0.01, 0.10]"
            )
    automatic = strategy.get("automatic_activation", {})
    if not isinstance(automatic, dict):
        raise GraspStrategyError("automatic_activation must be an object")
    families = automatic.get("target_geometry_families", [])
    if not isinstance(families, list) or not all(
        isinstance(item, str) and item.strip() for item in families
    ):
        raise GraspStrategyError(
            "automatic_activation.target_geometry_families must be strings"
        )
    return strategy


def grasp_strategy_sha256(value: Mapping[str, Any]) -> str:
    """Hash strategy semantics independently of publication receipts."""

    strategy = validate_grasp_strategy(value)
    strategy.pop("_source_path", None)
    strategy.pop("lifecycle", None)
    strategy["status"] = "candidate"
    payload = json.dumps(
        strategy,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def grasp_strategy_tree_sha256(
    root: str | Path = DEFAULT_GRASP_STRATEGY_ROOT,
) -> str:
    """Hash all effective strategies after validated-over-candidate resolution."""

    digest = hashlib.sha256()
    for strategy in sorted(
        load_grasp_strategies(root),
        key=lambda item: str(item.get("strategy_id") or ""),
    ):
        digest.update(str(strategy["strategy_id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(grasp_strategy_sha256(strategy).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(strategy["status"]).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def select_grasp_strategy(
    strategies: Sequence[Mapping[str, Any]],
    *,
    calibration_id: str,
    target_geometry_family: str = "",
    strategy_id: str = "",
) -> tuple[JsonDict | None, str]:
    """Select an explicit strategy or one deterministic automatic match."""

    compatible = [
        dict(strategy)
        for strategy in strategies
        if _strategy_supports_calibration(strategy, calibration_id)
    ]
    if strategy_id:
        for strategy in compatible:
            if str(strategy.get("strategy_id") or "") == strategy_id:
                return strategy, "explicit"
        raise GraspStrategyError(
            f"unknown or incompatible grasp strategy: {strategy_id}"
        )
    family = target_geometry_family.strip()
    if not family:
        return None, "generic_fallback"
    matches = [
        strategy
        for strategy in compatible
        if family
        in _mapping(
            strategy.get("automatic_activation", {}),
            "automatic_activation",
        ).get("target_geometry_families", [])
    ]
    if not matches:
        return None, "generic_fallback"
    matches.sort(
        key=lambda item: (
            0 if item.get("status") == "validated" else 1,
            -int(item.get("revision") or 1),
            str(item.get("strategy_id") or ""),
        )
    )
    return matches[0], "automatic_geometry_family"


def strategy_grasp_width_bounds(strategy: Mapping[str, Any]) -> tuple[float, float]:
    constraints = _mapping(strategy.get("constraints"), "constraints")
    values = _vector(
        constraints.get("grasp_width_bounds_m"),
        2,
        "constraints.grasp_width_bounds_m",
    )
    return values[0], values[1]


def strategy_pose_policy(strategy: Mapping[str, Any]) -> JsonDict:
    return dict(_mapping(strategy.get("pose_policy"), "pose_policy"))


def strategy_candidate_filter(strategy: Mapping[str, Any]) -> JsonDict:
    return dict(_mapping(strategy.get("candidate_filter", {}), "candidate_filter"))


def strategy_alignment_policy(strategy: Mapping[str, Any]) -> JsonDict:
    return dict(_mapping(strategy.get("alignment_policy", {}), "alignment_policy"))


def strategy_motion_policy(strategy: Mapping[str, Any]) -> JsonDict:
    return dict(_mapping(strategy.get("motion_policy", {}), "motion_policy"))


def public_grasp_strategy(strategy: Mapping[str, Any] | None) -> JsonDict | None:
    if strategy is None:
        return None
    return {
        str(key): value
        for key, value in strategy.items()
        if not str(key).startswith("_")
    }


def _strategy_supports_calibration(
    strategy: Mapping[str, Any],
    calibration_id: str,
) -> bool:
    compatibility = _mapping(strategy.get("compatibility"), "compatibility")
    calibration_ids = compatibility.get("calibration_ids")
    return isinstance(calibration_ids, list) and calibration_id in calibration_ids


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GraspStrategyError(f"{label} must be an object")
    return value


def _vector(value: object, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise GraspStrategyError(f"{label} must contain {size} values")
    result: list[float] = []
    for item in value:
        try:
            parsed = float(item)
        except (TypeError, ValueError) as exc:
            raise GraspStrategyError(f"{label} must contain numbers") from exc
        if not math.isfinite(parsed):
            raise GraspStrategyError(f"{label} must contain finite numbers")
        result.append(parsed)
    return result


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise GraspStrategyError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GraspStrategyError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise GraspStrategyError(f"{label} must be finite")
    return parsed
