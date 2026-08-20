"""cuRobo GPU-accelerated collision checking for move_to.

``CollisionChecker`` wraps cuRobo's ``RobotWorld`` to validate joint
configurations against both self-collision and world obstacles.

Per-handle instances are cached so the cuRobo robot model (which has
significant init cost due to CUDA kernel compilation) is created only
once per environment.

Usage::

    from sim.mcp_server.collision import get_checker, remove_checker

    checker = get_checker(handle, "libero")
    in_collision, info = checker.check(joint_positions, objects)
"""

from __future__ import annotations

import logging
import math

_logger = logging.getLogger("openeta.collision")

# Penetration thresholds (metres).  A config counts as colliding only when it
# overlaps an obstacle deeper than this — sphere-model approximation and
# floating-point boundary contact stay below it, avoiding false positives.
_WORLD_PENETRATION_TOL = 0.005   # 5 mm
_SELF_PENETRATION_TOL = 0.0      # cuRobo self-distance is exactly 0 when valid

# Per-handle CollisionChecker cache
_checkers: dict[str, CollisionChecker] = {}

# ── Graceful-degrade sentinel for when cuRobo / CUDA is missing ──────
_curobo_available: bool | None = None

_RECEPTACLE_CATEGORIES = frozenset({"basket", "bin", "bowl", "tray", "container"})


def resolve_contact_authorization(
    authorization: object,
    objects: list[dict],
    *,
    max_anchor_distance_m: float = 0.15,
    ambiguity_margin_m: float = 0.01,
) -> tuple[dict | None, dict]:
    """Bind host grasp evidence to exactly one current simulator object.

    The Agent never supplies this authorization directly.  The harness resolves
    a current compiled grasp to an opaque host block; the simulator adapter then
    associates its 3-D target anchor with current privileged geometry.  This is
    an evidence-to-safety adapter, not task-stage tracking.
    """

    if not isinstance(authorization, dict):
        return None, {
            "ok": False,
            "code": "contact_authorization_missing",
            "message": "No host-resolved contact authorization was supplied.",
        }
    if authorization.get("schema_version") != "openeta.contact_authorization.v1":
        return None, {
            "ok": False,
            "code": "contact_authorization_schema_mismatch",
            "message": "Contact authorization has an unsupported schema version.",
        }
    if authorization.get("waypoint_role") != "grasp_contact":
        return None, {
            "ok": False,
            "code": "contact_authorization_role_mismatch",
            "message": "Only a host-resolved grasp_contact waypoint may authorize contact.",
        }
    anchor = authorization.get("target_anchor_world_xyz")
    if not _finite_xyz(anchor):
        return None, {
            "ok": False,
            "code": "contact_authorization_anchor_invalid",
            "message": "Contact authorization has no finite world-frame target anchor.",
        }
    anchor_xyz = [float(value) for value in anchor[:3]]
    candidates: list[tuple[float, float, dict]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        category = str(obj.get("category") or "").strip().lower()
        if category in _RECEPTACLE_CATEGORIES:
            continue
        position = obj.get("position")
        if not _finite_xyz(position):
            continue
        bounds = _object_aabb(obj)
        surface_distance = (
            _point_aabb_distance(anchor_xyz, *bounds)
            if bounds is not None
            else math.inf
        )
        center_distance = math.dist(
            anchor_xyz,
            [float(value) for value in position[:3]],
        )
        candidates.append((surface_distance, center_distance, obj))
    if not candidates:
        return None, {
            "ok": False,
            "code": "contact_target_geometry_unavailable",
            "message": "No non-receptacle scene object geometry can be associated with the grasp anchor.",
        }
    candidates.sort(key=lambda item: (item[0], item[1], str(item[2].get("name") or "")))
    best_surface, best_center, best = candidates[0]
    # Conservative rbound-derived AABBs may overlap, so surface distance alone
    # cannot disambiguate.  Among equally containing/nearby bounds, require a
    # clear centre-distance winner rather than silently choosing one object.
    near_surface = [item for item in candidates if item[0] <= best_surface + 1e-9]
    near_surface.sort(key=lambda item: item[1])
    if len(near_surface) > 1 and (
        near_surface[1][1] - near_surface[0][1] < ambiguity_margin_m
    ):
        return None, {
            "ok": False,
            "code": "contact_target_geometry_ambiguous",
            "message": (
                "The compiled grasp anchor is equally close to multiple scene objects; "
                "refresh target geometry instead of guessing which contact is intended."
            ),
            "candidate_objects": [
                str(item[2].get("name") or "") for item in near_surface[:4]
            ],
        }
    if best_surface > max_anchor_distance_m and best_center > max_anchor_distance_m:
        return None, {
            "ok": False,
            "code": "contact_target_geometry_not_found",
            "message": (
                "No scene object is within the contact-authorization association "
                f"radius ({max_anchor_distance_m:.3f} m) of the compiled grasp anchor."
            ),
            "nearest_object": str(best.get("name") or ""),
            "nearest_surface_distance_m": best_surface,
            "nearest_center_distance_m": best_center,
        }
    return best, {
        "ok": True,
        "schema_version": "openeta.contact_authorization_resolution.v1",
        "compiled_grasp_id": authorization.get("compiled_grasp_id"),
        "target_evidence_id": authorization.get("target_evidence_id"),
        "object_scene_epoch": authorization.get("object_scene_epoch"),
        "target_object_name": str(best.get("name") or ""),
        "target_object_category": str(best.get("category") or ""),
        "anchor_world_xyz": anchor_xyz,
        "surface_distance_m": best_surface,
        "center_distance_m": best_center,
    }


def _detect_curobo() -> bool:
    """Check whether cuRobo and CUDA are usable.  Result is cached."""
    global _curobo_available
    if _curobo_available is not None:
        return _curobo_available

    # cuRobo's setuptools_scm version detection gets confused by the parent
    # openeta git repo.  Pretend a version so it skips SCM lookup.
    import os as _os
    _os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.7.0")

    try:
        import curobo  # noqa: F401
    except ImportError:
        _logger.warning("cuRobo not installed. Collision checking disabled.")
        _curobo_available = False
        return False

    try:
        import torch
        if not torch.cuda.is_available():
            _logger.warning("No CUDA GPU. Collision checking disabled.")
            _curobo_available = False
            return False
    except ImportError:
        _logger.warning("torch not installed. Collision checking disabled.")
        _curobo_available = False
        return False

    _curobo_available = True
    return True


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _build_world_config(objects: list[dict]) -> object | None:
    """Convert observation *objects* list to a cuRobo ``WorldConfig``.

    Each object::

        {"name": str, "position": [x,y,z], "orientation": [qw,qx,qy,qz]|None}

    Objects are approximated as 5 cm cuboids.  More accurate geometry
    (mesh, bbox from simulator) can be added later.
    """
    from curobo.geom.types import Cuboid, WorldConfig

    cuboids: list[Cuboid] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = str(obj.get("name", "obj"))
        pos = obj.get("position", [0.0, 0.0, 0.0])
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            continue

        # Pose: [x, y, z, qw, qx, qy, qz]
        ori = obj.get("orientation")
        if ori and isinstance(ori, (list, tuple)) and len(ori) >= 4:
            pose = [float(pos[0]), float(pos[1]), float(pos[2]),
                    float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])]
        else:
            pose = [float(pos[0]), float(pos[1]), float(pos[2]),
                    1.0, 0.0, 0.0, 0.0]

        dims = obj.get("dims", [0.05, 0.05, 0.05])

        cuboids.append(Cuboid(name=name, pose=pose, dims=list(dims)))

    if not cuboids:
        return None
    return WorldConfig(cuboid=cuboids)


def check_attached_object_collision(
    attachment: dict,
    objects: list[dict],
    predicted_eef_xyz: list[float],
    *,
    baseline_eef_xyz: list[float] | None = None,
    margin_m: float = 0.005,
) -> tuple[bool, dict]:
    """Check a conservative attached-object AABB against scene obstacles.

    This is independent of cuRobo so the same attached-geometry contract can
    later be populated by a real-robot perception adapter.  Receptacles expose
    an interior placement corridor: entry is allowed only when the carried
    object's centre clears the rim in XY.
    """

    relative = attachment.get("relative_xyz")
    dims = attachment.get("dims")
    if not _finite_xyz(relative) or not _finite_xyz(dims) or not _finite_xyz(predicted_eef_xyz):
        return False, {"available": False, "reason": "attached_object_geometry_incomplete"}
    held_dims = [max(0.01, float(value)) for value in dims]
    held_center, held_min, held_max = _attached_aabb(
        predicted_eef_xyz,
        relative,
        held_dims,
        margin_m=margin_m,
    )
    baseline_bounds = (
        _attached_aabb(
            baseline_eef_xyz,
            relative,
            held_dims,
            margin_m=margin_m,
        )
        if _finite_xyz(baseline_eef_xyz)
        else None
    )
    attached_name = str(attachment.get("object_name") or "")
    egress_obstacles: list[str] = []

    for obstacle in objects:
        if not isinstance(obstacle, dict) or str(obstacle.get("name") or "") == attached_name:
            continue
        bounds = _object_aabb(obstacle)
        if bounds is None:
            continue
        obstacle_min, obstacle_max = bounds
        if not _aabb_intersects(held_min, held_max, obstacle_min, obstacle_max):
            continue
        category = str(obstacle.get("category") or "").strip().lower()
        if category in _RECEPTACLE_CATEGORIES and _inside_receptacle_corridor(
            held_center,
            held_dims,
            obstacle_min,
            obstacle_max,
            margin_m=margin_m,
        ):
            continue
        obstacle_name = str(obstacle.get("name") or category or "scene obstacle")
        predicted_overlap = _aabb_overlap_volume(
            held_min,
            held_max,
            obstacle_min,
            obstacle_max,
        )
        baseline_overlap = 0.0
        if baseline_bounds is not None:
            _, baseline_min, baseline_max = baseline_bounds
            baseline_overlap = _aabb_overlap_volume(
                baseline_min,
                baseline_max,
                obstacle_min,
                obstacle_max,
            )
        # A conservative proxy can already overlap a neighbouring object at the
        # instant attachment is confirmed.  Rejecting every still-overlapping
        # intermediate pose creates a deadlock in which even a vertical escape
        # cannot begin.  Permit only strict monotonic egress from that existing
        # overlap; new or unchanged/worsened overlap remains a hard stop.
        if baseline_overlap > 0.0 and predicted_overlap < baseline_overlap - 1e-12:
            egress_obstacles.append(obstacle_name)
            continue
        return True, {
            "available": True,
            "world_collision": True,
            "self_collision": False,
            "max_world_penetration": 0.0,
            "max_self_penetration": 0.0,
            "collision_type": "attached_object_world",
            "attached_object": attached_name or None,
            "obstacle": obstacle_name,
            "predicted_attached_center_xyz": held_center,
            "predicted_eef_xyz": [float(value) for value in predicted_eef_xyz],
            "overlap_volume_m3": predicted_overlap,
            "baseline_overlap_volume_m3": baseline_overlap,
            "new_or_worsened": True,
            "message": (
                f"Attached object {attached_name or '<unknown>'} would collide with "
                f"{obstacle_name}. Raise or reroute the carry waypoint; for a "
                "receptacle, centre the object inside its placement corridor before "
                "descending."
            ),
        }
    return False, {
        "available": True,
        "attached_object_world_collision": False,
        **(
            {
                "egress_from_initial_overlap": True,
                "egress_obstacles": egress_obstacles,
                "message": (
                    "The attached-object proxy still overlaps conservative scene "
                    "geometry, but this controller increment strictly reduces that "
                    "pre-existing overlap. Monotonic egress is allowed."
                ),
            }
            if egress_obstacles
            else {}
        ),
    }


def _attached_aabb(
    eef_xyz: list[float] | tuple[float, ...],
    relative_xyz: list[float] | tuple[float, ...],
    held_dims: list[float],
    *,
    margin_m: float,
) -> tuple[list[float], list[float], list[float]]:
    centre = [float(eef_xyz[i]) + float(relative_xyz[i]) for i in range(3)]
    lower = [centre[i] - held_dims[i] / 2.0 - margin_m for i in range(3)]
    upper = [centre[i] + held_dims[i] / 2.0 + margin_m for i in range(3)]
    return centre, lower, upper


def _finite_xyz(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 3
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value[:3]
        )
    )


def _object_aabb(obj: dict) -> tuple[list[float], list[float]] | None:
    lower = obj.get("aabb_min")
    upper = obj.get("aabb_max")
    if _finite_xyz(lower) and _finite_xyz(upper):
        return ([float(value) for value in lower[:3]], [float(value) for value in upper[:3]])
    position = obj.get("position")
    dims = obj.get("dims")
    if not _finite_xyz(position) or not _finite_xyz(dims):
        return None
    centre = [float(value) for value in position[:3]]
    size = [max(0.01, float(value)) for value in dims[:3]]
    return (
        [centre[i] - size[i] / 2.0 for i in range(3)],
        [centre[i] + size[i] / 2.0 for i in range(3)],
    )


def _aabb_intersects(
    left_min: list[float],
    left_max: list[float],
    right_min: list[float],
    right_max: list[float],
) -> bool:
    return all(left_min[i] <= right_max[i] and left_max[i] >= right_min[i] for i in range(3))


def _point_aabb_distance(
    point: list[float],
    lower: list[float],
    upper: list[float],
) -> float:
    offsets = [
        lower[index] - point[index]
        if point[index] < lower[index]
        else point[index] - upper[index]
        if point[index] > upper[index]
        else 0.0
        for index in range(3)
    ]
    return math.sqrt(sum(value * value for value in offsets))


def _aabb_overlap_volume(
    left_min: list[float],
    left_max: list[float],
    right_min: list[float],
    right_max: list[float],
) -> float:
    overlaps = [
        max(0.0, min(left_max[i], right_max[i]) - max(left_min[i], right_min[i]))
        for i in range(3)
    ]
    return overlaps[0] * overlaps[1] * overlaps[2]


def _inside_receptacle_corridor(
    held_center: list[float],
    held_dims: list[float],
    receptacle_min: list[float],
    receptacle_max: list[float],
    *,
    margin_m: float,
) -> bool:
    for axis in (0, 1):
        lower = receptacle_min[axis] + held_dims[axis] / 2.0 + margin_m
        upper = receptacle_max[axis] - held_dims[axis] / 2.0 - margin_m
        if lower > upper or not lower <= held_center[axis] <= upper:
            return False
    return True


# ══════════════════════════════════════════════════════════════════════
# CollisionChecker
# ══════════════════════════════════════════════════════════════════════

class CollisionChecker:
    """Per-handle cuRobo collision checker.

    Created lazily on first ``move_to`` call.  Implementation note:
    cuRobo's ``franka.yml`` model expects 7 DOF (fingers locked).
    """

    def __init__(self, backend: str) -> None:
        self._backend = backend
        self._available = _detect_curobo()
        self._robot_world: object | None = None
        self._arm_dof = 7          # Franka: 7 arm revolute joints
        self._state_dof: int = 7   # LIBERO: exactly 7
        self._last_objects_hash: int | None = None

        if backend == "maniskill":
            self._state_dof = 9    # 7 arm + 2 gripper, sliced to [:7]

        if backend == "metaworld":
            self._available = False
            _logger.info("MetaWorld backend — joint_positions not available; skipping collision")

        if not self._available:
            _logger.info("Collision checking unavailable for handle (backend=%s)", backend)

    # ── lazy init ──────────────────────────────────────────────────

    def _ensure_robot_world(self) -> object:
        """Create the cuRobo ``RobotWorld`` on first call (has JIT cost)."""
        if self._robot_world is not None:
            return self._robot_world

        import torch
        from curobo.types.base import TensorDeviceType
        from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
        from curobo.geom.sdf.world import CollisionCheckerType
        from curobo.geom.types import WorldConfig as CuroboWorldConfig

        tensor_args = TensorDeviceType(device=torch.device("cuda", index=0))

        # Must pass a non-None world_model so the collision checker gets
        # created at init time (otherwise update_world is a no-op).
        config = RobotWorldConfig.load_from_config(
            robot_config="franka.yml",
            world_model=CuroboWorldConfig(),  # empty, populated via update_world
            tensor_args=tensor_args,
            n_envs=1,
            collision_activation_distance=0.0,
            self_collision_activation_distance=0.0,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
        )
        _logger.info("Creating cuRobo RobotWorld (first call — JIT compiles CUDA kernels)...")
        self._robot_world = RobotWorld(config)
        _logger.info("cuRobo RobotWorld ready.")
        return self._robot_world

    # ── world update ───────────────────────────────────────────────

    def _update_world_if_changed(self, objects: list[dict]) -> bool:
        """Update the collision world if objects have changed.

        Returns ``True`` if the world was updated.
        """
        obj_hash = hash(tuple(
            (o.get("name"), tuple(o.get("position", [])),
             tuple(o.get("orientation", [])) if o.get("orientation") else None)
            for o in (objects or [])
        ))
        if obj_hash == self._last_objects_hash:
            return False
        self._last_objects_hash = obj_hash

        wc = _build_world_config(objects)
        if wc is not None:
            rw = self._robot_world
            if rw is not None:
                rw.update_world(wc)
            return True
        return False

    # ── penetration query ──────────────────────────────────────────

    def _max_world_penetration_esdf(self, rw: object, q: object) -> float:
        """Deepest world penetration (metres) at config *q*, via true ESDF.

        Returns the max signed distance of any robot collision sphere INTO a
        world obstacle: positive = penetrating, <= 0 = clear.  If the world
        has no obstacles, cuRobo would error on the query, so we short-circuit
        to 0.0 (nothing to collide with).
        """
        import torch  # noqa: F401 — parity with caller's device/dtype

        world = getattr(rw, "world_model", None)
        # No primitive obstacles loaded → nothing to penetrate.
        ctypes = getattr(world, "collision_types", {}) if world is not None else {}
        if not ctypes.get("primitive"):
            return 0.0

        from curobo.geom.sdf.world import CollisionQueryBuffer

        state = rw.get_kinematics(q)
        spheres = state.link_spheres_tensor.unsqueeze(1)
        buf = CollisionQueryBuffer()
        buf.update_buffer_shape(spheres.shape, rw.tensor_args, world.collision_types)
        weight = rw.tensor_args.to_device([1.0])
        act = rw.tensor_args.to_device([0.0])
        esdf = world.get_sphere_distance(
            spheres, buf, weight, act, compute_esdf=True,
        )
        # compute_esdf: positive INSIDE an obstacle, negative outside.  The
        # deepest penetration across all spheres is the max.
        return float(esdf.max().item())

    # ── public API ─────────────────────────────────────────────────

    def check(self, joint_positions: list[float],
              objects: list[dict] | None = None) -> tuple[bool, dict]:
        """Return ``(in_collision, info_dict)``.

        *joint_positions* is the raw joint angles from the observation
        (7 floats for LIBERO, 9 for ManiSkill).

        *objects* is the ``observation.objects`` list.

        On success, ``info`` contains keys:
          ``max_world_penetration``, ``max_self_penetration``,
          ``world_collision``, ``self_collision`` (positive = penetration).
        """
        objects = objects or []
        coverage = {
            "check_mode": "post_step_configuration",
            "trajectory_checked": False,
            "self_checked": bool(self._available),
            "world_checked": bool(self._available and objects),
            "world_object_count": len(objects),
        }
        if not self._available:
            return False, {"available": False,
                           "reason": "cuRobo not installed or CUDA unavailable",
                           **coverage}

        if self._backend == "metaworld":
            return False, {"available": False,
                           "reason": "joint_positions unavailable for MetaWorld",
                           "max_world_penetration": 0.0, "max_self_penetration": 0.0,
                           "world_collision": False, "self_collision": False}

        if not joint_positions or len(joint_positions) < self._arm_dof:
            return False, {"available": True,
                           "reason": f"Need >= {self._arm_dof} joint positions, got {len(joint_positions)}",
                           "max_world_penetration": 0.0, "max_self_penetration": 0.0,
                           "world_collision": False, "self_collision": False}

        try:
            rw = self._ensure_robot_world()
        except Exception as exc:
            _logger.error("Failed to create cuRobo RobotWorld: %s", exc)
            return False, {"available": False, "reason": f"RobotWorld init failed: {exc}"}

        # Update world obstacles if needed
        if objects:
            try:
                self._update_world_if_changed(objects)
            except Exception:
                pass  # best-effort; self-collision still works

        # Slice to arm DOF (ManiSkill: 9D → 7D)
        q_arm = joint_positions[:self._arm_dof]
        if len(q_arm) < self._arm_dof:
            return False, {"available": True,
                           "reason": f"Expected {self._arm_dof} arm joints, got {len(q_arm)}"}

        import torch
        try:
            q = torch.tensor(
                [q_arm],
                device=rw.tensor_args.device,
                dtype=rw.tensor_args.dtype,
            )
            # World penetration: use the TRUE Euclidean signed distance
            # (``compute_esdf=True``), NOT ``get_collision_distance`` — the
            # latter returns a weighted collision *cost* summed across every
            # robot sphere, which is positive whenever a sphere is merely
            # *near* an obstacle (within the activation distance) and is
            # insensitive to obstacle size.  That made every object in the
            # arm's workspace read as a "world penetration" even with no
            # contact (the reported false positive on ~step 3 of any reach).
            # ESDF is positive strictly inside an obstacle, negative outside,
            # so ``> tol`` means a real overlap.
            d_world_val = self._max_world_penetration_esdf(rw, q)
            # Self-collision uses cuRobo's dedicated self-collision distance,
            # which is 0 for valid configurations (verified on the LIBERO
            # reset pose) — keep it.
            _, d_self = rw.get_world_self_collision_distance_from_joints(q)
            d_self_val = float(d_self[0].item()) if d_self is not None else 0.0
        except Exception as exc:
            _logger.error("Collision check failed: %s", exc)
            return False, {"available": True, "error": str(exc),
                           "max_world_penetration": 0.0, "max_self_penetration": 0.0,
                           "world_collision": False, "self_collision": False}

        # Small tolerance so sphere-model / floating-point boundary contact
        # doesn't trip a collision — only a real overlap counts.
        world_coll = d_world_val > _WORLD_PENETRATION_TOL
        self_coll = d_self_val > _SELF_PENETRATION_TOL
        in_collision = world_coll or self_coll

        return in_collision, {
            "available": True,
            "max_world_penetration": d_world_val,
            "max_self_penetration": d_self_val,
            "world_collision": world_coll,
            "self_collision": self_coll,
            **coverage,
        }

    def close(self) -> None:
        """Release cuRobo GPU resources."""
        self._robot_world = None
        self._last_objects_hash = None


# ══════════════════════════════════════════════════════════════════════
# Module-level cache helpers
# ══════════════════════════════════════════════════════════════════════

def get_checker(handle: str, backend: str) -> CollisionChecker:
    """Get or lazily create a per-handle ``CollisionChecker``.

    Args:
        handle: MCP environment handle.
        backend: ``"libero"``, ``"maniskill"``, or ``"metaworld"``.

    Returns:
        Cached ``CollisionChecker`` instance, created on first call.
    """
    global _checkers
    if handle not in _checkers:
        _checkers[handle] = CollisionChecker(backend)
    return _checkers[handle]


def remove_checker(handle: str) -> None:
    """Clean up the collision checker for *handle*."""
    global _checkers
    checker = _checkers.pop(handle, None)
    if checker is not None:
        checker.close()
