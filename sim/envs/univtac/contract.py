"""JSON contracts for direct UniVTAC observation/action traces."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


SNAPSHOT_SCHEMA_VERSION = "openeta.univtac.task_snapshot.v1"
TRANSITION_SCHEMA_VERSION = "openeta.univtac.transition.v1"
SNAPSHOT_PHASES = frozenset({"pre_action", "post_action"})
FORBIDDEN_OPERATOR_KEYS = frozenset(
    {
        "tactile_pose",
        "actor_pose",
        "target_pose",
        "hole_pose",
        "exact_relative_object_pose",
        "relative_object_pose",
        "rel_pose",
        "check_success",
        "native_check_success",
        "eval_success",
        "plan_success",
    }
)
OPERATOR_TOP_LEVEL_KEYS = frozenset(
    {"cameras", "tactile", "proprio", "task_instruction", "step_identifiers"}
)
ARTIFACT_DESCRIPTOR_KEYS = frozenset(
    {"path", "shape", "dtype", "value_range", "encoding", "stored_dtype"}
)
STEP_IDENTIFIER_KEYS = frozenset(
    {"snapshot_id", "action_id", "phase", "simulator_step", "take_action_count"}
)


class UniVTACContractError(ValueError):
    """Raised when a UniVTAC trace violates the direct-smoke contract."""


class PrivilegedVisibilityError(UniVTACContractError):
    """Raised when privileged state leaks into ``operator_visible``."""


class IncompleteTactilePacketError(UniVTACContractError):
    """Raised when required tactile observations are absent or malformed."""


def _validate_relative_path(path: str) -> str:
    candidate = PurePosixPath(path)
    if not path or candidate.is_absolute() or ".." in candidate.parts:
        raise UniVTACContractError(f"artifact path must be relative: {path!r}")
    return candidate.as_posix()


def _walk_keys(value: Any, prefix: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            yield prefix + (key_text,)
            yield from _walk_keys(child, prefix + (key_text,))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_keys(child, prefix + (str(index),))


def validate_operator_visible(operator_visible: Mapping[str, Any]) -> None:
    actual_top_level = frozenset(str(key) for key in operator_visible)
    if actual_top_level != OPERATOR_TOP_LEVEL_KEYS:
        raise PrivilegedVisibilityError(
            "operator_visible top-level keys must be exactly "
            f"{sorted(OPERATOR_TOP_LEVEL_KEYS)}, got {sorted(actual_top_level)}"
        )
    for key_path in _walk_keys(operator_visible):
        if key_path[-1].lower() in FORBIDDEN_OPERATOR_KEYS:
            dotted = ".".join(key_path)
            raise PrivilegedVisibilityError(
                f"privileged key is forbidden in operator_visible: {dotted}"
            )

    cameras = operator_visible["cameras"]
    tactile = operator_visible["tactile"]
    proprio = operator_visible["proprio"]
    identifiers = operator_visible["step_identifiers"]
    if not isinstance(cameras, Mapping) or not isinstance(tactile, Mapping):
        raise PrivilegedVisibilityError("operator camera and tactile fields must be mappings")
    if not isinstance(proprio, Mapping) or frozenset(proprio) != {"joint", "ee"}:
        raise PrivilegedVisibilityError("operator proprio must contain exactly joint and ee")
    if not isinstance(operator_visible["task_instruction"], str):
        raise PrivilegedVisibilityError("operator task_instruction must be a string")
    if not isinstance(identifiers, Mapping) or frozenset(identifiers) != STEP_IDENTIFIER_KEYS:
        raise PrivilegedVisibilityError(
            "operator step_identifiers has fields outside the direct-smoke contract"
        )

    def validate_artifact_mapping(
        groups: Mapping[str, Any], *, allowed_payload_key: str
    ) -> None:
        for name, payload in groups.items():
            if not isinstance(name, str) or not isinstance(payload, Mapping):
                raise PrivilegedVisibilityError("operator sensor payloads must be named mappings")
            if frozenset(payload) != {allowed_payload_key}:
                raise PrivilegedVisibilityError(
                    f"operator {name!r} may contain only {allowed_payload_key}"
                )
            descriptor = payload[allowed_payload_key]
            if not isinstance(descriptor, Mapping) or "path" not in descriptor:
                raise PrivilegedVisibilityError(
                    f"operator {name!r}.{allowed_payload_key} must be an artifact reference"
                )
            unexpected = frozenset(descriptor) - ARTIFACT_DESCRIPTOR_KEYS
            if unexpected:
                raise PrivilegedVisibilityError(
                    f"operator artifact reference contains unexpected fields: {sorted(unexpected)}"
                )

    validate_artifact_mapping(cameras, allowed_payload_key="rgb")
    validate_artifact_mapping(tactile, allowed_payload_key="rgb_marker")


def _validate_artifact_paths(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key) == "path" and isinstance(child, str):
                _validate_relative_path(child)
            _validate_artifact_paths(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_artifact_paths(child)


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to an artifact stored outside JSON."""

    path: str
    shape: tuple[int, ...]
    dtype: str
    value_range: tuple[float, float]
    encoding: str
    stored_dtype: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _validate_relative_path(self.path))
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(
            self,
            "value_range",
            (float(self.value_range[0]), float(self.value_range[1])),
        )
        if any(dim <= 0 for dim in self.shape):
            raise UniVTACContractError(f"artifact shape must be positive: {self.shape}")
        if not self.dtype or not self.encoding or not self.stored_dtype:
            raise UniVTACContractError("artifact dtype and encoding fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UniVTACTaskSnapshot:
    """One pre- or post-action UniVTAC observation snapshot."""

    snapshot_id: str
    task_name: str
    seed: int
    phase: str
    action_id: str
    simulator_step: int
    take_action_count: int
    task_instruction: str
    external_camera: dict[str, Any]
    tactile_sensors: dict[str, Any]
    proprio: dict[str, Any]
    operator_visible: dict[str, Any]
    host_only: dict[str, Any]
    artifacts: dict[str, Any]
    schema_version: str = SNAPSHOT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SNAPSHOT_SCHEMA_VERSION:
            raise UniVTACContractError(f"unsupported snapshot schema: {self.schema_version}")
        if self.phase not in SNAPSHOT_PHASES:
            raise UniVTACContractError(f"invalid snapshot phase: {self.phase}")
        if not self.snapshot_id or not self.task_name or not self.action_id:
            raise UniVTACContractError("snapshot_id, task_name, and action_id are required")
        if self.simulator_step < 0 or self.take_action_count < 0:
            raise UniVTACContractError("step and action counts must be non-negative")

        for name in (
            "external_camera",
            "tactile_sensors",
            "proprio",
            "operator_visible",
            "host_only",
            "artifacts",
        ):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))

        validate_operator_visible(self.operator_visible)
        _validate_artifact_paths(self.to_dict())
        try:
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise UniVTACContractError(f"snapshot is not strict JSON serializable: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = None) -> str:
        separators = None if indent is not None else (",", ":")
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=indent,
            separators=separators,
            allow_nan=False,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UniVTACTaskSnapshot":
        return cls(**copy.deepcopy(dict(payload)))


@dataclass(frozen=True)
class TactileTransition:
    """One direct probe transition binding pre/post observations."""

    action_id: str
    action_type: str
    requested_displacement_mm: tuple[float, float, float]
    pre_snapshot_id: str
    post_snapshot_id: str
    pre_simulator_step: int
    post_simulator_step: int
    pre_take_action_count: int
    post_take_action_count: int
    tactile_differences: dict[str, Any]
    host_only: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRANSITION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRANSITION_SCHEMA_VERSION:
            raise UniVTACContractError(f"unsupported transition schema: {self.schema_version}")
        if not self.action_id or not self.action_type:
            raise UniVTACContractError("transition action_id and action_type are required")
        displacement = tuple(float(value) for value in self.requested_displacement_mm)
        if len(displacement) != 3:
            raise UniVTACContractError("requested_displacement_mm must have three values")
        object.__setattr__(self, "requested_displacement_mm", displacement)
        for name in ("tactile_differences", "host_only", "artifacts"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        _validate_artifact_paths(self.to_dict())
        try:
            json.dumps(self.to_dict(), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise UniVTACContractError(f"transition is not strict JSON serializable: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_transition_binding(
    pre: UniVTACTaskSnapshot,
    post: UniVTACTaskSnapshot,
    transition: TactileTransition | None = None,
) -> None:
    if pre.phase != "pre_action" or post.phase != "post_action":
        raise UniVTACContractError("transition requires pre_action then post_action snapshots")
    if pre.action_id != post.action_id:
        raise UniVTACContractError(
            f"pre/post action_id mismatch: {pre.action_id!r} != {post.action_id!r}"
        )
    if pre.task_name != post.task_name or pre.seed != post.seed:
        raise UniVTACContractError("pre/post task identity mismatch")
    if transition is not None:
        if transition.action_id != pre.action_id:
            raise UniVTACContractError("transition action_id does not bind both snapshots")
        if transition.pre_snapshot_id != pre.snapshot_id:
            raise UniVTACContractError("transition pre_snapshot_id mismatch")
        if transition.post_snapshot_id != post.snapshot_id:
            raise UniVTACContractError("transition post_snapshot_id mismatch")
        if transition.pre_simulator_step != pre.simulator_step:
            raise UniVTACContractError("transition pre_simulator_step mismatch")
        if transition.post_simulator_step != post.simulator_step:
            raise UniVTACContractError("transition post_simulator_step mismatch")
        if transition.pre_take_action_count != pre.take_action_count:
            raise UniVTACContractError("transition pre_take_action_count mismatch")
        if transition.post_take_action_count != post.take_action_count:
            raise UniVTACContractError("transition post_take_action_count mismatch")


def resolve_artifact_path(root: Path, relative_path: str) -> Path:
    """Resolve a validated relative artifact path beneath ``root``."""

    return root / _validate_relative_path(relative_path)
