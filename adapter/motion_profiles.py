"""Host-private motion profiles shared by Agent and simulator adapters.

The profile is selected by the host process, never by an Agent-authored tool
argument.  Keeping the switch outside the public tool schema lets the same
Agent request be replayed against the historical controller (A), the bounded
settling controller (B), and the sequential route executor (C).
"""

from __future__ import annotations

from dataclasses import dataclass
import os


MOTION_EXPERIMENT_CONDITION_ENV = "OPENETA_MOTION_EXPERIMENT_CONDITION"


@dataclass(frozen=True, slots=True)
class MotionControlProfile:
    condition: str
    stable_arrival_enabled: bool
    stable_steps_required: int
    joint_velocity_tolerance_rad_s: float
    progress_stall_enabled: bool
    progress_window_steps: int
    minimum_aligned_progress_m: float
    minimum_error_improvement_m: float
    cross_track_tolerance_m: float
    nominal_joint_velocity_limit_rad_s: float
    carrying_joint_velocity_limit_rad_s: float
    sequential_route_preview_enabled: bool

    @property
    def schema_version(self) -> str:
        return "openeta.motion_execution_profile.v1"

    def receipt(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "condition": self.condition,
            "stable_arrival_enabled": self.stable_arrival_enabled,
            "stable_steps_required": self.stable_steps_required,
            "joint_velocity_tolerance_rad_s": (
                self.joint_velocity_tolerance_rad_s
            ),
            "progress_stall_enabled": self.progress_stall_enabled,
            "progress_window_steps": self.progress_window_steps,
            "minimum_aligned_progress_m": self.minimum_aligned_progress_m,
            "minimum_error_improvement_m": self.minimum_error_improvement_m,
            "cross_track_tolerance_m": self.cross_track_tolerance_m,
            "nominal_joint_velocity_limit_rad_s": (
                self.nominal_joint_velocity_limit_rad_s
            ),
            "carrying_joint_velocity_limit_rad_s": (
                self.carrying_joint_velocity_limit_rad_s
            ),
            "sequential_route_preview_enabled": (
                self.sequential_route_preview_enabled
            ),
            "authority": "host_process_configuration",
        }


_PROFILES = {
    "A": MotionControlProfile(
        condition="A",
        stable_arrival_enabled=False,
        stable_steps_required=0,
        joint_velocity_tolerance_rad_s=0.05,
        progress_stall_enabled=False,
        progress_window_steps=0,
        minimum_aligned_progress_m=0.0,
        minimum_error_improvement_m=0.0,
        cross_track_tolerance_m=0.0,
        nominal_joint_velocity_limit_rad_s=0.5,
        carrying_joint_velocity_limit_rad_s=0.2,
        sequential_route_preview_enabled=False,
    ),
    "B": MotionControlProfile(
        condition="B",
        stable_arrival_enabled=True,
        stable_steps_required=3,
        joint_velocity_tolerance_rad_s=0.05,
        progress_stall_enabled=True,
        progress_window_steps=30,
        minimum_aligned_progress_m=0.001,
        minimum_error_improvement_m=0.001,
        cross_track_tolerance_m=0.01,
        nominal_joint_velocity_limit_rad_s=0.35,
        carrying_joint_velocity_limit_rad_s=0.15,
        sequential_route_preview_enabled=False,
    ),
    "C": MotionControlProfile(
        condition="C",
        stable_arrival_enabled=True,
        stable_steps_required=3,
        joint_velocity_tolerance_rad_s=0.05,
        progress_stall_enabled=True,
        progress_window_steps=30,
        minimum_aligned_progress_m=0.001,
        minimum_error_improvement_m=0.001,
        cross_track_tolerance_m=0.01,
        nominal_joint_velocity_limit_rad_s=0.35,
        carrying_joint_velocity_limit_rad_s=0.15,
        sequential_route_preview_enabled=True,
    ),
}


def normalize_motion_condition(value: object) -> str:
    condition = str(value or "A").strip().upper()
    if condition not in _PROFILES:
        raise ValueError(
            f"{MOTION_EXPERIMENT_CONDITION_ENV} must be A, B, or C; "
            f"received {condition!r}"
        )
    return condition


def motion_control_profile(condition: object | None = None) -> MotionControlProfile:
    selected = (
        os.environ.get(MOTION_EXPERIMENT_CONDITION_ENV, "A")
        if condition is None
        else condition
    )
    return _PROFILES[normalize_motion_condition(selected)]
