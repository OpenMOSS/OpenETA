"""Types for observation-only Agent interactions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

READONLY_REPORT_SCHEMA = "openeta.readonly_observation_report.v1"
EXPECTED_IMAGE_LABELS = (
    "camera/head/rgb",
    "camera/wrist/rgb",
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
)
FORBIDDEN_RESPONSE_KEYS = frozenset(
    {
        "action",
        "actions",
        "command",
        "commands",
        "tool_calls",
        "function_call",
        "tool_name",
        "arguments",
        "trajectory",
        "waypoints",
        "control",
    }
)


class ReadOnlyObservationError(ValueError):
    """Raised when an observation-only request or response is invalid."""


@dataclass(frozen=True)
class ReadOnlyImageInput:
    label: str
    media_type: str
    path: Path
    width: int
    height: int
    channels: int
    dtype: str


@dataclass(frozen=True)
class ReadOnlyObservationContext:
    instruction: str
    step_identifiers: Mapping[str, Any]
    proprio: Mapping[str, Any]
    visual_descriptors: tuple[Mapping[str, Any], ...]
    images: tuple[ReadOnlyImageInput, ...]


@dataclass(frozen=True)
class ReadOnlyObservationReport:
    schema_version: str
    mode: str
    context_received: bool
    received_image_labels: tuple[str, ...]
    observation_notes: tuple[str, ...]
    uncertainties: tuple[str, ...]
    execution_requested: bool

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ReadOnlyObservationReport:
        forbidden = _find_forbidden_keys(payload)
        if forbidden:
            raise ReadOnlyObservationError(
                f"read-only response contains action fields: {sorted(forbidden)}"
            )
        expected = {
            "schema_version",
            "mode",
            "context_received",
            "received_image_labels",
            "observation_notes",
            "uncertainties",
            "execution_requested",
        }
        if set(payload) != expected:
            raise ReadOnlyObservationError("read-only response fields do not match the schema")
        report = cls(
            schema_version=str(payload["schema_version"]),
            mode=str(payload["mode"]),
            context_received=bool(payload["context_received"]),
            received_image_labels=tuple(str(value) for value in payload["received_image_labels"]),
            observation_notes=tuple(str(value) for value in payload["observation_notes"]),
            uncertainties=tuple(str(value) for value in payload["uncertainties"]),
            execution_requested=bool(payload["execution_requested"]),
        )
        validate_readonly_report(report)
        return report

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "context_received": self.context_received,
            "received_image_labels": list(self.received_image_labels),
            "observation_notes": list(self.observation_notes),
            "uncertainties": list(self.uncertainties),
            "execution_requested": self.execution_requested,
        }


@runtime_checkable
class ReadOnlyObservationBackend(Protocol):
    def observe(self, context: ReadOnlyObservationContext) -> ReadOnlyObservationReport: ...


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_RESPONSE_KEYS:
                found.add(str(key).lower())
            found.update(_find_forbidden_keys(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(_find_forbidden_keys(child))
    return found


def find_absolute_posix_paths(value: Any) -> set[str]:
    """Return structured string values that are absolute POSIX paths."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for child in value.values():
            found.update(find_absolute_posix_paths(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            found.update(find_absolute_posix_paths(child))
    elif isinstance(value, str) and PurePosixPath(value).is_absolute():
        found.add(value)
    return found


def validate_readonly_context(context: ReadOnlyObservationContext) -> None:
    labels = tuple(image.label for image in context.images)
    if labels != EXPECTED_IMAGE_LABELS:
        raise ReadOnlyObservationError(
            f"expected image labels {EXPECTED_IMAGE_LABELS}, got {labels}"
        )
    if not context.instruction:
        raise ReadOnlyObservationError("instruction must not be empty")
    if set(context.proprio) != {"joint", "ee"}:
        raise ReadOnlyObservationError("read-only proprio must contain exactly joint and ee")
    if len(context.visual_descriptors) != 4:
        raise ReadOnlyObservationError("read-only context requires four visual descriptors")
    descriptor_labels = tuple(str(item.get("label")) for item in context.visual_descriptors)
    if descriptor_labels != EXPECTED_IMAGE_LABELS:
        raise ReadOnlyObservationError("visual descriptor order does not match image order")
    for image in context.images:
        if image.media_type != "image/png" or image.dtype != "uint8":
            raise ReadOnlyObservationError("read-only images must be uint8 PNG")
        if image.channels != 3 or image.width <= 0 or image.height <= 0:
            raise ReadOnlyObservationError("read-only image dimensions are invalid")


def validate_readonly_report(report: ReadOnlyObservationReport) -> None:
    if report.schema_version != READONLY_REPORT_SCHEMA or report.mode != "read_only":
        raise ReadOnlyObservationError("backend response is not a read-only observation report")
    if not report.context_received:
        raise ReadOnlyObservationError("backend did not acknowledge the observation context")
    if report.received_image_labels != EXPECTED_IMAGE_LABELS:
        raise ReadOnlyObservationError("backend response image labels are incomplete or reordered")
    if report.execution_requested:
        raise ReadOnlyObservationError("read-only backend requested execution")
