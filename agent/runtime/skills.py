"""Text-guidance skill registry for embodied tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from adapter.protocol import JsonDict


BUILTIN_SKILL_DIR = Path(__file__).resolve().parents[1] / "skills"


@dataclass(frozen=True, slots=True)
class SkillSpec:
    """Task-level guidance exposed to planners.

    Skills are editable text instructions that help the planner choose atomic
    tools. They are not executable macro actions and the runtime must not
    expand them into hidden tool calls.
    """

    name: str
    description: str
    content: str
    task_patterns: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    source: str = "builtin"
    version: str = "v1"
    editable: bool = True
    metadata: JsonDict = field(default_factory=dict)


class SkillRegistry:
    """Registry of task-level skill guidance documents."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillSpec] = {}

    def register(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            raise ValueError(f"Skill already registered: {spec.name}")
        self._skills[spec.name] = spec

    def update(self, spec: SkillSpec) -> None:
        if spec.name not in self._skills:
            raise KeyError(f"Unknown skill: {spec.name}")
        current = self._skills[spec.name]
        if not current.editable:
            raise ValueError(f"Skill is not editable: {spec.name}")
        self._skills[spec.name] = spec

    def upsert(self, spec: SkillSpec) -> None:
        if spec.name in self._skills:
            self.update(spec)
            return
        self.register(spec)

    def get(self, name: str) -> SkillSpec:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(f"Unknown skill: {name}") from exc

    def list(self) -> list[SkillSpec]:
        return list(self._skills.values())


def build_default_skill_registry() -> SkillRegistry:
    """Create the first text-guidance skill library."""

    registry = SkillRegistry()
    for path in sorted(BUILTIN_SKILL_DIR.glob("*.md")):
        registry.register(load_skill_markdown(path))

    for spec in [
        SkillSpec(
            name="place",
            description="Place a held object on or inside a target receptacle.",
            task_patterns=("place <object> on <target>", "put <object> into <target>"),
            allowed_tools=(
                "scene_detector",
                "sam3",
                "ik_preview_check",
                "obstacle_avoidance",
                "move_to",
                "gripper_control",
                "observe",
            ),
            content=(
                "Use this skill as guidance only. Locate the receptacle or surface, "
                "select a collision-free release pose, check reachability and path "
                "safety, move to the release pose, open the gripper, retreat, and "
                "observe to confirm the object is released at the intended place."
            ),
        ),
        SkillSpec(
            name="push",
            description="Draft guidance for short planar push manipulation.",
            task_patterns=("push <object>", "move <object> by pushing"),
            allowed_tools=(
                "scene_detector",
                "sam3",
                "ik_preview_check",
                "obstacle_avoidance",
                "move_to",
                "gripper_control",
                "observe",
            ),
            content=(
                "Use this skill as guidance only. This is placeholder guidance "
                "until dedicated push tools are connected. Identify the movable "
                "object, choose a short push contact segment, check IK and "
                "collision constraints, execute one short segment, then observe "
                "before continuing or correcting the motion."
            ),
        ),
        SkillSpec(
            name="pull",
            description="Draft guidance for short pull manipulation.",
            task_patterns=("pull <object>", "move <object> by pulling"),
            allowed_tools=(
                "scene_detector",
                "sam3",
                "ik_preview_check",
                "obstacle_avoidance",
                "move_to",
                "gripper_control",
                "observe",
            ),
            content=(
                "Use this skill as guidance only. This is placeholder guidance "
                "until dedicated pull tools are connected. Identify a reachable "
                "contact, edge, handle, or grasp-assisted pull mode, check IK and "
                "collision constraints, execute one short segment, then observe "
                "before continuing or correcting the motion."
            ),
        ),
        SkillSpec(
            name="stack",
            description="Guidance for stacking one object on another.",
            task_patterns=("stack <object> on <object>",),
            allowed_tools=(
                "scene_detector",
                "sam3",
                "anygrasp",
                "ik_preview_check",
                "obstacle_avoidance",
                "move_to",
                "gripper_control",
                "observe",
            ),
            content=(
                "Use this skill as guidance only. Combine pick and place guidance, "
                "but add stability checks before release and verify the stack after "
                "the gripper retreats."
            ),
        ),
    ]:
        if spec.name not in {skill.name for skill in registry.list()}:
            registry.register(spec)
    return registry


def load_skill_markdown(path: Path) -> SkillSpec:
    """Load a text-guidance skill from a markdown file with simple frontmatter."""

    text = path.read_text(encoding="utf-8")
    metadata, content = _split_frontmatter(text)
    name = _frontmatter_string(metadata, "name") or path.stem
    description = _frontmatter_string(metadata, "description") or name
    version = _frontmatter_string(metadata, "version") or "v1"
    editable = _frontmatter_bool(metadata, "editable", default=True)
    try:
        source_path = path.relative_to(BUILTIN_SKILL_DIR.parent)
    except ValueError:
        source_path = path
    return SkillSpec(
        name=name,
        description=description,
        content=content.strip(),
        task_patterns=_frontmatter_list(metadata, "task_patterns"),
        allowed_tools=_frontmatter_list(metadata, "allowed_tools"),
        source=f"markdown:{source_path}",
        version=version,
        editable=editable,
        metadata={"path": str(path)},
    )


def _split_frontmatter(text: str) -> tuple[dict[str, str | list[str]], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    try:
        end = next(idx for idx, line in enumerate(lines[1:], start=1) if line.strip() == "---")
    except StopIteration:
        return {}, text
    return _parse_frontmatter(lines[1:end]), "\n".join(lines[end + 1 :])


def _parse_frontmatter(lines: list[str]) -> dict[str, str | list[str]]:
    data: dict[str, str | list[str]] = {}
    current_list_key = ""
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- ") and current_list_key:
            value = stripped[2:].strip()
            current = data.setdefault(current_list_key, [])
            if isinstance(current, list):
                current.append(value)
            continue
        if ":" not in line:
            current_list_key = ""
            continue
        key, value = line.split(":", 1)
        current_list_key = key.strip()
        parsed_value = value.strip()
        data[current_list_key] = [] if not parsed_value else parsed_value
    return data


def _frontmatter_string(data: dict[str, str | list[str]], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else ""


def _frontmatter_bool(
    data: dict[str, str | list[str]],
    key: str,
    *,
    default: bool,
) -> bool:
    value = data.get(key)
    if not isinstance(value, str):
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _frontmatter_list(data: dict[str, str | list[str]], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if isinstance(value, list):
        return tuple(item for item in value if item)
    if isinstance(value, str) and value:
        return (value,)
    return ()
