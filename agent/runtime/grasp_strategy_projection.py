"""Evaluation-only projections over a session-owned grasp strategy snapshot."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from adapter.protocol import JsonDict
from agent.tools.grasp_strategies import (
    grasp_strategy_tree_sha256,
    load_grasp_strategies,
)


_ALLOWED_FIELDS = {
    "exclude_strategy_ids",
    "strip_canary_evidence_strategy_ids",
}


def normalize_grasp_strategy_projection(value: object) -> JsonDict:
    """Validate and normalize one host-owned evaluation projection."""

    if value is None:
        return {
            "exclude_strategy_ids": [],
            "strip_canary_evidence_strategy_ids": [],
        }
    if not isinstance(value, Mapping):
        raise ValueError("runtime.grasp_strategies must be an object")
    unknown = sorted(str(key) for key in value if key not in _ALLOWED_FIELDS)
    if unknown:
        raise ValueError(
            "unsupported runtime.grasp_strategies field(s): " + ", ".join(unknown)
        )

    normalized: JsonDict = {}
    for field in sorted(_ALLOWED_FIELDS):
        observed = value.get(field, [])
        if not isinstance(observed, list) or any(
            not isinstance(item, str) or not item.strip() for item in observed
        ):
            raise ValueError(f"runtime.grasp_strategies.{field} must be a list of ids")
        ids = [item.strip() for item in observed]
        if len(ids) != len(set(ids)):
            raise ValueError(f"runtime.grasp_strategies.{field} contains duplicates")
        normalized[field] = sorted(ids)

    overlap = set(normalized["exclude_strategy_ids"]) & set(
        normalized["strip_canary_evidence_strategy_ids"]
    )
    if overlap:
        raise ValueError(
            "a grasp strategy cannot be both excluded and evidence-stripped: "
            + ", ".join(sorted(overlap))
        )
    return normalized


def apply_grasp_strategy_projection(
    root: str | Path,
    value: object,
) -> JsonDict:
    """Apply a projection to an isolated session tree and return its receipt."""

    strategy_root = Path(root)
    projection = normalize_grasp_strategy_projection(value)
    before_sha256 = grasp_strategy_tree_sha256(strategy_root)
    paths: dict[str, Path] = {}
    payloads: dict[str, JsonDict] = {}
    for path in sorted(strategy_root.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"grasp strategy must contain one object: {path}")
        strategy_id = str(payload.get("strategy_id") or "").strip()
        if not strategy_id:
            raise ValueError(f"grasp strategy is missing strategy_id: {path}")
        if strategy_id in paths:
            raise ValueError(f"duplicate grasp strategy id: {strategy_id}")
        paths[strategy_id] = path
        payloads[strategy_id] = payload

    requested = set(projection["exclude_strategy_ids"]) | set(
        projection["strip_canary_evidence_strategy_ids"]
    )
    missing = sorted(requested - set(paths))
    if missing:
        raise ValueError(
            "evaluation grasp strategy projection references unknown id(s): "
            + ", ".join(missing)
        )

    for strategy_id in projection["exclude_strategy_ids"]:
        paths[strategy_id].unlink()

    stripped: list[str] = []
    for strategy_id in projection["strip_canary_evidence_strategy_ids"]:
        payload = payloads[strategy_id]
        provenance = payload.get("provenance")
        if isinstance(provenance, dict) and "canary_evidence" in provenance:
            provenance.pop("canary_evidence")
            stripped.append(strategy_id)
        paths[strategy_id].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    effective = load_grasp_strategies(strategy_root)
    return {
        "schema_version": "openeta.grasp_strategy_projection.v1",
        **projection,
        "stripped_canary_evidence_ids": stripped,
        "effective_strategy_ids": sorted(
            str(item.get("strategy_id") or "") for item in effective
        ),
        "before_sha256": before_sha256,
        "after_sha256": grasp_strategy_tree_sha256(strategy_root),
    }
