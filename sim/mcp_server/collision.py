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

RECEPTACLE_CATEGORIES = frozenset({"basket", "bin", "bowl", "tray", "container"})

# Back-compat alias for in-module readers.
_RECEPTACLE_CATEGORIES = RECEPTACLE_CATEGORIES


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
    held_center = [float(predicted_eef_xyz[i]) + float(relative[i]) for i in range(3)]
    held_min = [held_center[i] - held_dims[i] / 2.0 - margin_m for i in range(3)]
    held_max = [held_center[i] + held_dims[i] / 2.0 + margin_m for i in range(3)]
    attached_name = str(attachment.get("object_name") or "")

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
            "message": (
                f"Attached object {attached_name or '<unknown>'} would collide with "
                f"{obstacle_name}. Raise or reroute the carry waypoint; for a "
                "receptacle, centre the object inside its placement corridor before "
                "descending."
            ),
        }
    return False, {"available": True, "attached_object_world_collision": False}


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
        self._robot_config = "franka.yml"
        self._last_objects_hash: int | None = None
        # Number of obstacles actually loaded into the cuRobo world.  This is
        # the authoritative "is there anything to hit" signal: cuRobo keeps
        # ``world_model.collision_types["primitive"]`` True for the lifetime of
        # the checker once the primitive type is registered, so that flag says
        # only that primitive checking is *enabled*, never that obstacles are
        # present.  Verified against cuRobo 0.7.7 on 2026-08-17.
        self._obstacle_count: int = 0
        # Cached observation->cuRobo joint permutation, plus the exact name list
        # it was derived from.  The permutation is only meaningful for that
        # ordering, so it is revalidated on every call rather than built once.
        self._joint_permutation: list[int] | None = None
        self._joint_permutation_names: list[str] | None = None

        if backend == "maniskill":
            self._state_dof = 9    # 7 arm + 2 gripper, sliced to [:7]

        if backend == "behavior":
            self._robot_config = "r1pro.yml"
            self._arm_dof = 22
            self._state_dof = 22
            # If the generated config is absent the checker must fail loudly at
            # this point rather than fall back to franka.yml, which would be a
            # silent wrong-robot check.
            try:
                from curobo.util_file import get_robot_configs_path, join_path
                import os
                if not os.path.exists(join_path(get_robot_configs_path(), "r1pro.yml")):
                    self._available = False
                    _logger.error("r1pro.yml missing — run scripts/gen_r1pro_curobo.py; "
                                  "BEHAVIOR collision disabled rather than using franka.yml")
            except Exception:
                pass

        if backend == "metaworld":
            self._available = False
            _logger.info("MetaWorld backend — joint_positions not available; skipping collision")

        if not self._available:
            _logger.info("Collision checking unavailable for handle (backend=%s)", backend)

    # ── joint mapping ──────────────────────────────────────────────

    def _needs_named_mapping(self) -> bool:
        """True when a positional slice cannot yield cuRobo's joint vector.

        LIBERO and ManiSkill report the arm as the leading entries in order, so
        slicing is correct there.  R1Pro does not: its vector starts with six
        base DOF and then interleaves the two arms
        (``left_arm_joint1, right_arm_joint1, left_arm_joint2, ...``), so the
        joints cuRobo wants are neither contiguous nor in order.
        """
        return self._backend == "behavior"

    def _map_by_name(self, joint_positions: list[float],
                     joint_names: list[str] | None,
                     ) -> tuple[list[float] | None, str]:
        """Permute *joint_positions* into the cuRobo model's joint order.

        Returns ``(q, "")`` on success or ``(None, reason)`` on failure.  Every
        failure path returns a reason rather than a best-effort vector: a
        mis-mapped q still yields a confident verdict, and a wrong "clear" is
        exactly what this module exists to prevent.
        """
        if not joint_names:
            return None, ("joint_names missing from observation; refusing to "
                          "guess R1Pro joint order (arms are interleaved)")
        if len(joint_names) != len(joint_positions):
            return None, (f"joint_names ({len(joint_names)}) does not match "
                          f"joint_positions ({len(joint_positions)})")

        try:
            rw = self._ensure_robot_world()
            target = [str(n) for n in rw.kinematics.joint_names]
        except Exception as exc:
            return None, f"could not read cuRobo joint order: {exc}"

        index = {str(n): i for i, n in enumerate(joint_names)}
        missing = [n for n in target if n not in index]
        if missing:
            return None, (f"observation lacks joints cuRobo needs: "
                          f"{missing[:4]}{'...' if len(missing) > 4 else ''}")

        # Rebuild when the incoming order changes, not only on first call.  A
        # reordering that still contains every required joint passes the
        # ``missing`` check above, so a cache keyed on nothing would keep
        # applying a stale permutation and report a confident verdict computed
        # from the wrong geometry -- the exact failure this mapping prevents.
        key = [str(n) for n in joint_names]
        if self._joint_permutation is None or self._joint_permutation_names != key:
            rebuilt = self._joint_permutation is not None
            self._joint_permutation = [index[n] for n in target]
            self._joint_permutation_names = key
            _logger.info("%s %d observation joints -> %d cuRobo joints by name",
                         "Remapped" if rebuilt else "Mapped",
                         len(joint_names), len(target))
        return [float(joint_positions[i]) for i in self._joint_permutation], ""

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
            robot_config=self._robot_config,
            world_model=CuroboWorldConfig(),  # empty, populated via update_world
            tensor_args=tensor_args,
            n_envs=1,
            collision_activation_distance=0.0,
            self_collision_activation_distance=0.0,
            collision_checker_type=CollisionCheckerType.PRIMITIVE,
        )
        _logger.info("Creating cuRobo RobotWorld (first call — JIT compiles CUDA kernels)...")
        self._robot_world = RobotWorld(config)

        # Reconcile the expected joint count against the model actually built.
        # Locked joints do not count as active DOF, so a hand-written constant
        # drifts from reality as soon as lock_joints changes; a mismatch here
        # would silently feed a wrongly-sized q to cuRobo.
        try:
            actual = int(self._robot_world.kinematics.get_dof())
            if actual != self._arm_dof:
                _logger.info("cuRobo active DOF is %d, expected %d (%s) — using %d",
                             actual, self._arm_dof, self._robot_config, actual)
                self._arm_dof = actual
        except Exception as exc:
            _logger.warning("Could not read cuRobo DOF: %s", exc)

        _logger.info("cuRobo RobotWorld ready (%s, dof=%d).",
                     self._robot_config, self._arm_dof)
        return self._robot_world

    # ── world update ───────────────────────────────────────────────

    def _update_world_if_changed(self, objects: list[dict]) -> bool:
        """Update the collision world if objects have changed.

        Returns ``True`` if the world was updated.

        An empty *objects* list explicitly CLEARS the world.  Returning early
        without clearing would leave the previous call's obstacles resident at
        their stale poses, so a later query would test the arm against
        last-frame geometry and report a stale verdict as authoritative.
        """
        # Must cover every field _build_world_config reads, dims included.  An
        # earlier version hashed only name/position/orientation, so an object
        # that changed size at a fixed pose -- a perception update refining a
        # bounding box, a resized proxy -- was silently checked against the
        # previous geometry: the same staleness class as skipping the clear.
        obj_hash = hash(tuple(
            (o.get("name"), tuple(o.get("position", [])),
             tuple(o.get("orientation", [])) if o.get("orientation") else None,
             tuple(o.get("dims", [])) if o.get("dims") else None)
            for o in (objects or [])
        ))
        if obj_hash == self._last_objects_hash:
            return False
        self._last_objects_hash = obj_hash

        rw = self._robot_world
        if rw is None:
            return False

        wc = _build_world_config(objects)
        if wc is None:
            # No usable geometry → drop every obstacle rather than keep the
            # previous world.  Recording the count as 0 is what actually makes
            # the next query safe: cuRobo leaves the primitive collision type
            # registered after an empty update, so its own state cannot tell us
            # the world is empty.
            self._obstacle_count = 0
            from curobo.geom.types import WorldConfig as CuroboWorldConfig

            rw.update_world(CuroboWorldConfig())
            return True

        rw.update_world(wc)
        self._obstacle_count = len(getattr(wc, "cuboid", None) or [])
        return True

    # ── penetration query ──────────────────────────────────────────

    def _max_world_penetration_esdf(self, rw: object, q: object) -> float:
        """Deepest world penetration (metres) at config *q*, via true ESDF.

        Returns the max signed distance of any robot collision sphere INTO a
        world obstacle: positive = penetrating, <= 0 = clear.  If the world
        has no obstacles, cuRobo would error on the query, so we short-circuit
        to 0.0 (nothing to collide with).
        """
        import torch  # noqa: F401 — parity with caller's device/dtype

        # Nothing loaded → nothing to penetrate.  Keyed on the count we recorded
        # rather than ``collision_types["primitive"]``, which stays True after an
        # empty update and would send an empty world through the ESDF query.
        if self._obstacle_count == 0:
            return 0.0

        from curobo.geom.sdf.world import CollisionQueryBuffer

        world = rw.world_model
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
              objects: list[dict] | None = None,
              joint_names: list[str] | None = None) -> tuple[bool, dict]:
        """Return ``(in_collision, info_dict)``.

        *joint_positions* is the raw joint angles from the observation
        (7 floats for LIBERO, 9 for ManiSkill, 28 for BEHAVIOR/R1Pro).

        *joint_names* names those angles positionally.  Required for BEHAVIOR:
        R1Pro reports arms interleaved left/right, so the subset cuRobo wants
        cannot be obtained by slicing.  Ignored for single-arm backends whose
        leading ``_arm_dof`` entries are already the arm in order.

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
            # Report *why*, per backend.  A caller that cannot tell "checked and
            # clear" from "never checked" will read an unsupported robot as a
            # safe one, which is the failure mode this whole path exists to
            # prevent.
            if self._backend == "metaworld":
                # This branch used to sit *below* the generic return and was
                # therefore dead: metaworld sets _available False in __init__,
                # so it reported "cuRobo not installed" even with cuRobo working.
                reason = "joint_positions unavailable for MetaWorld"
            elif self._backend == "behavior":
                reason = ("r1pro.yml not generated — run scripts/gen_r1pro_curobo.py")
            else:
                reason = "cuRobo not installed or CUDA unavailable"
            return False, {"available": False, "reason": reason,
                           "unsupported_robot": False,
                           **coverage,
                           "max_world_penetration": 0.0, "max_self_penetration": 0.0,
                           "world_collision": False, "self_collision": False}

        if not joint_positions:
            return False, {"available": True,
                           "reason": "No joint positions in observation",
                           "max_world_penetration": 0.0, "max_self_penetration": 0.0,
                           "world_collision": False, "self_collision": False}

        # Reorder to cuRobo's own joint order when names are available.  This is
        # mandatory for R1Pro and harmless elsewhere.  Failing closed here is
        # deliberate: an unmapped vector would still produce a verdict, just one
        # computed from the wrong joints.
        if self._needs_named_mapping():
            q_arm, map_err = self._map_by_name(joint_positions, joint_names)
            if map_err:
                return False, {"available": True, "reason": map_err,
                               "world_checked": False, "self_checked": False,
                               "max_world_penetration": 0.0,
                               "max_self_penetration": 0.0,
                               "world_collision": False, "self_collision": False}
        elif len(joint_positions) < self._arm_dof:
            return False, {"available": True,
                           "reason": f"Need >= {self._arm_dof} joint positions, got {len(joint_positions)}",
                           "max_world_penetration": 0.0, "max_self_penetration": 0.0,
                           "world_collision": False, "self_collision": False}
        else:
            q_arm = None  # resolved below by the existing slice path

        try:
            rw = self._ensure_robot_world()
        except Exception as exc:
            _logger.error("Failed to create cuRobo RobotWorld: %s", exc)
            return False, {"available": False, "reason": f"RobotWorld init failed: {exc}"}

        # Update world obstacles.  Called unconditionally — an empty list must
        # reach _update_world_if_changed so it can clear stale geometry.
        world_update_error = ""
        try:
            self._update_world_if_changed(objects or [])
        except Exception as exc:  # best-effort; self-collision still works
            world_update_error = str(exc)
            _logger.error("World update failed, world verdict is not authoritative: %s", exc)

        # Slice to arm DOF (ManiSkill: 9D → 7D).  Skipped when the name-based
        # mapping above already produced the vector in cuRobo's order.
        if q_arm is None:
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
                           "world_checked": False, "self_checked": False,
                           "obstacle_count": 0,
                           "max_world_penetration": 0.0, "max_self_penetration": 0.0,
                           "world_collision": False, "self_collision": False}

        # Small tolerance so sphere-model / floating-point boundary contact
        # doesn't trip a collision — only a real overlap counts.
        world_coll = d_world_val > _WORLD_PENETRATION_TOL
        self_coll = d_self_val > _SELF_PENETRATION_TOL
        in_collision = world_coll or self_coll

        # ``world_checked`` distinguishes "checked the world and it was clear"
        # from "never had any world geometry to check".  Without it a caller
        # cannot tell a real pass from an unpopulated world, and an
        # unconditional ``detected: False`` reads as a safety guarantee that
        # was never actually evaluated.
        info = {
            "available": True,
            "world_checked": self._obstacle_count > 0 and not world_update_error,
            "self_checked": True,
            "obstacle_count": len(objects or []),
            "max_world_penetration": d_world_val,
            "max_self_penetration": d_self_val,
            "world_collision": world_coll,
            "self_collision": self_coll,
            **coverage,
        }
        if world_update_error:
            info["world_update_error"] = world_update_error
        return in_collision, info

    @staticmethod
    def _world_has_obstacles(rw: object) -> bool:
        """Deprecated: cuRobo's ``collision_types`` cannot answer this.

        Retained only so external callers do not break.  Once the primitive
        collision type is registered cuRobo keeps the flag True for the
        checker's lifetime, including after an empty ``update_world``, so this
        reports whether primitive checking is *enabled* — not whether any
        obstacle is loaded.  Use ``self._obstacle_count`` instead.
        """
        world = getattr(rw, "world_model", None)
        ctypes = getattr(world, "collision_types", {}) if world is not None else {}
        return bool(ctypes.get("primitive"))

    def close(self) -> None:
        """Release cuRobo GPU resources."""
        self._robot_world = None
        self._last_objects_hash = None
        self._obstacle_count = 0


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
