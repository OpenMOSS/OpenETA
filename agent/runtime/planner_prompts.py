"""Host-owned prompt composition for the main embodied planner."""

from __future__ import annotations

import hashlib
from pathlib import Path

from adapter.protocol import JsonDict


EMBODIED_CLOSED_LOOP_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / "embodied_closed_loop.md"
)


def load_embodied_closed_loop_prompt(
    path: str | Path = EMBODIED_CLOSED_LOOP_PROMPT_PATH,
) -> str:
    """Load the mandatory host-owned planner contract."""

    prompt_path = Path(path)
    content = prompt_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"planner prompt is empty: {prompt_path}")
    return content


def compose_main_planner_prompt(base_prompt: str) -> tuple[str, JsonDict]:
    """Append the closed-loop contract and return reproducibility metadata."""

    contract = load_embodied_closed_loop_prompt()
    composed = f"{base_prompt.strip()}\n\n{contract}\n"
    digest = hashlib.sha256(composed.encode("utf-8")).hexdigest()
    return composed, {
        "schema_version": "openeta.planner_prompt.v1",
        "sha256": digest,
        "source": str(EMBODIED_CLOSED_LOOP_PROMPT_PATH),
        "contract": "embodied_closed_loop",
    }
