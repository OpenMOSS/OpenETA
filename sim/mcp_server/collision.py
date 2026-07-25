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
        if not self._available:
            return False, {"available": False,
                           "reason": "cuRobo not installed or CUDA unavailable"}

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
