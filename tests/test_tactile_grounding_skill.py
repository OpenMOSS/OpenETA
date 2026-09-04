from __future__ import annotations

from agent.skills.tactile_grounding import load_tactile_grounding_skill


def test_tactile_grounding_skill_is_image_first_read_only_candidate() -> None:
    skill = load_tactile_grounding_skill()
    assert skill.name == "tactile_grounding_image_first"
    assert skill.version == "0.1"
    assert skill.status == "candidate" and skill.mode == "read_only"
    assert skill.tools_for("image_first") == (
        "observe_images",
        "observe_structured_guidance",
    )
    assert skill.tools_for("simultaneous") == ("observe",)
    assert not skill.requires_record_image_judgment
    assert "record_image_judgment" not in skill.prompt_for("image_first")
    assert "Do not replace a clear image-supported judgment" in skill.prompt_for(
        "image_first"
    )


def test_control_and_skill_prompts_share_one_frozen_semantic_instruction() -> None:
    skill = load_tactile_grounding_skill()
    image_first = skill.prompt_for("image_first")
    simultaneous = skill.prompt_for("simultaneous")
    assert skill.instruction.strip() in image_first
    assert skill.instruction.strip() in simultaneous
    assert image_first != simultaneous
    assert "expected" not in image_first.lower()
    assert "baseline" not in image_first.lower()
