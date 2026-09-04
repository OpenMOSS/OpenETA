from __future__ import annotations

from pathlib import Path

from scripts.univtac.capture_grounding_skill_heldout import SEEDS


def test_heldout_capture_is_fixed_and_action_free() -> None:
    assert SEEDS == (1_000_003, 1_000_004, 1_000_005)
    source = (
        Path(__file__).resolve().parents[2]
        / "scripts/univtac/capture_grounding_skill_heldout.py"
    ).read_text(encoding="utf-8")
    assert '"play_once_count": 0' in source
    assert '"agent_action_count": 0' in source
    assert "check_success" not in source and "check_early_stop" not in source
