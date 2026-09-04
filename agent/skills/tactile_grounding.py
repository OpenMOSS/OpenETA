"""Single-source loader for the read-only tactile grounding skill candidate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class TactileGroundingSkill:
    name: str
    version: str
    status: str
    mode: str
    tool_sequence: tuple[str, ...]
    requires_record_image_judgment: bool
    evidence_policy: dict[str, bool]
    forbidden_claims: tuple[str, ...]
    instruction: str
    output_schema: dict[str, str]

    def tools_for(self, ordering: Literal["image_first", "simultaneous"]) -> tuple[str, ...]:
        return self.tool_sequence if ordering == "image_first" else ("observe",)

    def prompt_for(self, ordering: Literal["image_first", "simultaneous"]) -> str:
        procedure = (
            "Call `observe_images`, then call `observe_structured_guidance`, then return the final "
            "JSON object."
            if ordering == "image_first"
            else "Call `observe` exactly once, then return the final JSON object."
        )
        schema = json.dumps(self.output_schema, indent=2)
        return (
            "You are a read-only multimodal tactile observer.\n\n"
            f"{procedure}\n\n{self.instruction.strip()}\n\n"
            f"Return exactly one JSON object matching this schema:\n{schema}\n\n"
            "Do not propose or execute an action. Do not output robot commands. Do not claim exact "
            "force, pressure, friction, grip stability, object pose, or task success.\n"
        )


def load_tactile_grounding_skill(path: Path | None = None) -> TactileGroundingSkill:
    source = path or Path(__file__).with_name("tactile_grounding_image_first.yaml")
    payload: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("tactile grounding skill must be a mapping")
    required = {
        "name",
        "version",
        "status",
        "mode",
        "tool_sequence",
        "requires_record_image_judgment",
        "evidence_policy",
        "forbidden_claims",
        "instruction",
        "output_schema",
    }
    if set(payload) != required:
        raise ValueError("tactile grounding skill fields are invalid")
    if payload["tool_sequence"] != ["observe_images", "observe_structured_guidance"]:
        raise ValueError("tactile grounding tool sequence must be image first")
    if payload["requires_record_image_judgment"] is not False:
        raise ValueError("tactile grounding candidate must not require a record tool")
    return TactileGroundingSkill(
        name=str(payload["name"]),
        version=str(payload["version"]),
        status=str(payload["status"]),
        mode=str(payload["mode"]),
        tool_sequence=tuple(payload["tool_sequence"]),
        requires_record_image_judgment=False,
        evidence_policy=dict(payload["evidence_policy"]),
        forbidden_claims=tuple(payload["forbidden_claims"]),
        instruction=str(payload["instruction"]),
        output_schema=dict(payload["output_schema"]),
    )


__all__ = ["TactileGroundingSkill", "load_tactile_grounding_skill"]
