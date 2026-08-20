"""Prove the generated r1pro.yml loads in cuRobo and discriminates collisions.

Loading is necessary but not sufficient: a config can load and still return a
constant verdict, which from outside is indistinguishable from "always safe".
So this drives the *production* code path -- ``CollisionChecker`` -- and checks
that an enclosing obstacle reports penetration while an empty world does not.

Uses ESDF via CollisionChecker rather than ``get_collision_distance``: the
latter is a weighted proximity cost, which reads positive when a sphere is
merely near an obstacle.  See the comment at collision.py:465.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/yfzhang/nvme1/openeta_fix_wt")


def main() -> int:
    import torch
    from sim.mcp_server.collision import CollisionChecker

    checker = CollisionChecker("behavior")
    if not checker._available:
        print(f"FAIL: checker unavailable — {checker.check([0.0] * 22, [])[1]}")
        return 1

    rw = checker._ensure_robot_world()
    dof = rw.kinematics.get_dof()
    joints = list(rw.kinematics.joint_names)
    print(f"loaded: dof={dof}")
    print(f"  joints: {joints}")

    q = torch.zeros((1, dof), device=rw.tensor_args.device, dtype=rw.tensor_args.dtype)

    state = rw.get_kinematics(q)
    if not torch.isfinite(state.ee_position).all():
        print("FAIL: non-finite forward kinematics")
        return 1
    ee = state.ee_position.detach().cpu().numpy().reshape(-1)[:3]
    print(f"  ee_position (left): {ee.round(3).tolist()}")
    print(f"  spheres: {tuple(state.link_spheres_tensor.shape)}")

    zero = [0.0] * dof
    wall = [{"name": "wall", "position": [0.0, 0.0, 0.5], "dims": [4.0, 4.0, 4.0]}]

    det_empty, info_empty = checker.check(zero, [])
    det_wall, info_wall = checker.check(zero, wall)

    pen_empty = info_empty.get("max_world_penetration")
    pen_wall = info_wall.get("max_world_penetration")
    print(f"\n  empty world : detected={det_empty} pen={pen_empty} "
          f"count={info_empty.get('obstacle_count')} "
          f"world_checked={info_empty.get('world_checked')}")
    print(f"  enclosing   : detected={det_wall} pen={pen_wall} "
          f"count={info_wall.get('obstacle_count')} "
          f"world_checked={info_wall.get('world_checked')}")
    if info_wall.get("error"):
        print(f"  error: {info_wall['error']}")

    ok = (det_wall and not det_empty
          and isinstance(pen_wall, float) and pen_wall > 0.0)
    print(f"\n{'PASS' if ok else 'FAIL'}: enclosing box detected, empty world clear")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
