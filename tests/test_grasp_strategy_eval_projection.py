from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from agent.runtime.grasp_strategy_projection import (
    apply_grasp_strategy_projection,
    normalize_grasp_strategy_projection,
)
from agent.tools.grasp_strategies import DEFAULT_GRASP_STRATEGY_ROOT


STRATEGY_ID = "top-down-vertical-panda-p8"


def _strategy_root(tmp_path: Path) -> Path:
    root = tmp_path / "grasp"
    shutil.copytree(DEFAULT_GRASP_STRATEGY_ROOT, root)
    return root


def test_projection_excludes_strategy_from_session_snapshot(tmp_path: Path) -> None:
    root = _strategy_root(tmp_path)

    receipt = apply_grasp_strategy_projection(
        root,
        {"exclude_strategy_ids": [STRATEGY_ID]},
    )

    assert STRATEGY_ID not in receipt["effective_strategy_ids"]
    assert receipt["before_sha256"] != receipt["after_sha256"]
    assert not any(
        json.loads(path.read_text(encoding="utf-8")).get("strategy_id") == STRATEGY_ID
        for path in root.glob("*/*.json")
    )


def test_projection_strips_only_task_specific_canary_evidence(tmp_path: Path) -> None:
    root = _strategy_root(tmp_path)

    receipt = apply_grasp_strategy_projection(
        root,
        {"strip_canary_evidence_strategy_ids": [STRATEGY_ID]},
    )

    target = next(
        json.loads(path.read_text(encoding="utf-8"))
        for path in root.glob("*/*.json")
        if json.loads(path.read_text(encoding="utf-8")).get("strategy_id") == STRATEGY_ID
    )
    assert STRATEGY_ID in receipt["effective_strategy_ids"]
    assert receipt["stripped_canary_evidence_ids"] == [STRATEGY_ID]
    assert "canary_evidence" not in target["provenance"]
    assert target["description"]
    assert target["pose_policy"]


def test_projection_rejects_conflicting_operations() -> None:
    with pytest.raises(ValueError, match="both excluded and evidence-stripped"):
        normalize_grasp_strategy_projection(
            {
                "exclude_strategy_ids": [STRATEGY_ID],
                "strip_canary_evidence_strategy_ids": [STRATEGY_ID],
            }
        )
