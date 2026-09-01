"""Pure geometry predicates for worker-local controller recovery."""

from __future__ import annotations

from collections.abc import Sequence


def project_velocity_to_joint_limits(
    velocity: Sequence[float],
    current: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    dt: float,
    boundary_margin: float = 1e-6,
) -> tuple[list[float], list[int]]:
    """Project one emergency velocity step onto the hard joint-limit box.

    Mink's configuration-limit QP can become numerically infeasible near a
    boundary even though its collision-free fallback has a useful velocity.
    The fallback must never cross a hard limit, but rejecting the whole vector
    can strand every later Cartesian command.  Clip only the outward component
    of each affected joint and let the normal geometric preview validate the
    resulting full step.

    A joint that physics has already placed outside its range is restricted to
    non-worsening motion; incremental inward recovery remains possible.
    """

    if not (
        len(velocity) == len(current) == len(lower) == len(upper)
        and dt > 0.0
    ):
        raise ValueError("joint velocity projection inputs must have equal lengths and dt > 0")
    projected = [float(value) for value in velocity]
    clipped: list[int] = []
    for index, (speed, now, low, high) in enumerate(
        zip(projected, current, lower, upper)
    ):
        safe_low = float(low) + float(boundary_margin)
        safe_high = float(high) - float(boundary_margin)
        if safe_low > safe_high:
            raise ValueError("joint limit interval is narrower than the safety margin")
        if now < low:
            minimum_speed = 0.0
            maximum_speed = (safe_high - float(now)) / float(dt)
        elif now > high:
            minimum_speed = (safe_low - float(now)) / float(dt)
            maximum_speed = 0.0
        else:
            minimum_speed = (safe_low - float(now)) / float(dt)
            maximum_speed = (safe_high - float(now)) / float(dt)
        bounded = min(max(float(speed), minimum_speed), maximum_speed)
        if abs(bounded - float(speed)) > 1e-12:
            projected[index] = bounded
            clipped.append(index)
    return projected, clipped


def verified_collision_boundary_escape(
    current: dict[tuple[int, int], float],
    candidate: dict[tuple[int, int], float],
    *,
    hard_stop_distance_m: float,
    recovery_boundary_distance_m: float | None = None,
    nonworsening_epsilon_m: float = 1e-6,
    progress_epsilon_m: float = 1e-5,
) -> bool:
    """Allow only monotonic escape from a collision or active safety boundary."""

    boundary = (
        hard_stop_distance_m
        if recovery_boundary_distance_m is None
        else max(hard_stop_distance_m, recovery_boundary_distance_m)
    )
    violating = {
        pair: distance
        for pair, distance in current.items()
        if distance < boundary
    }
    if not violating or set(candidate) != set(current):
        return False
    if any(
        pair not in violating and distance < hard_stop_distance_m
        for pair, distance in candidate.items()
    ):
        return False
    if any(
        candidate[pair] < distance - nonworsening_epsilon_m
        for pair, distance in violating.items()
    ):
        return False
    return (
        min(candidate[pair] for pair in violating)
        > min(violating.values()) + progress_epsilon_m
        or all(candidate[pair] >= boundary for pair in violating)
    )


def verified_joint_limit_escape(
    current: Sequence[float],
    candidate: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    nonworsening_epsilon: float = 1e-8,
    progress_epsilon: float = 1e-7,
) -> bool:
    """Require an escape candidate to preserve or monotonically repair joint limits."""

    if not (len(current) == len(candidate) == len(lower) == len(upper)):
        return False
    progressed = False
    for now, proposed, low, high in zip(current, candidate, lower, upper):
        if now < low:
            if proposed < now - nonworsening_epsilon:
                return False
            progressed = progressed or proposed > now + progress_epsilon
        elif now > high:
            if proposed > now + nonworsening_epsilon:
                return False
            progressed = progressed or proposed < now - progress_epsilon
        elif proposed < low - nonworsening_epsilon or proposed > high + nonworsening_epsilon:
            return False
    return progressed or all(
        low - nonworsening_epsilon <= value <= high + nonworsening_epsilon
        for value, low, high in zip(candidate, lower, upper)
    )
