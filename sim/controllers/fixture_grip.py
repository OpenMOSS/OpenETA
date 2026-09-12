"""Opt-in worker control for already held objects or fixtures, without new authority."""

from dataclasses import replace
import math


ENABLE_ENV = 'OPENETA_LIBERO_GRIP_STABILIZATION'
LEGACY_ENABLE_ENV = 'OPENETA_LIBERO_FIXTURE_GRIP_STABILIZATION'


def stabilize_grip(profile, *, enabled, authorization, gripper_command,
                           contact, orientation_change_rad, seeded, attachment_proxy=None):
    """Stabilize seeded bilateral holds with host-resolved object/fixture scope.

    This does not prove target retention, authorize contact, select a target, or
    infer fixture travel. Other movements keep their original execution profile.
    """
    fixture = bool(isinstance(authorization, dict) and authorization.get('ok') is True
        and authorization.get('contact_kind') == 'articulated_fixture')
    attached = bool(isinstance(attachment_proxy, dict)
        and isinstance(attachment_proxy.get('object_name'), str)
        and attachment_proxy['object_name'].strip()
        and attachment_proxy.get('status') in {'tentative', 'confirmed'})
    active = bool(enabled and seeded and (fixture or attached)
        and gripper_command > 0
        and contact.get('available') is True
        and contact.get('left_fingerpad_contact') is True
        and contact.get('right_fingerpad_contact') is True
        and orientation_change_rad is not None
        and math.isfinite(orientation_change_rad) and orientation_change_rad >= 0
        and (not fixture or orientation_change_rad <= 0.05))
    if not active:
        return profile, 1.0, 'none'
    # Extra orientation weight stabilizes an existing hold during translation.
    # Applying it to a deliberately different orientation makes a velocity-
    # limited QP trade centimetres of translation for angular progress. Keep
    # the slow carry speed and stable arrival, but use the baseline angular
    # cost for requested reorientation beyond the existing 0.05 rad hold band.
    orientation_weight = 25.0 if orientation_change_rad <= 0.05 else 1.0
    return replace(profile,
        nominal_joint_velocity_limit_rad_s=min(profile.nominal_joint_velocity_limit_rad_s, 0.2),
        carrying_joint_velocity_limit_rad_s=min(profile.carrying_joint_velocity_limit_rad_s, 0.2),
        stable_arrival_enabled=True,
        stable_steps_required=max(profile.stable_steps_required, 3)), orientation_weight, 'fixture' if fixture else 'attached_object'
