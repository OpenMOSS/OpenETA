"""Derive a URDF-loadable cuRobo config for R1Pro from the shipped OmniGibson one.

BEHAVIOR ships four generated cuRobo configs for R1Pro, but none can be loaded
through cuRobo's normal file path:

  * they carry no ``urdf_path`` / ``asset_root_path`` -- OmniGibson feeds cuRobo
    a live USD stage instead of loading from disk;
  * ``base_link: base_footprint_x`` and the six ``base_footprint_*_joint``
    entries exist only in the USD, injected at import time because
    ``use_holonomic_joints: true``.  All five R1Pro URDFs contain zero
    occurrences (verified by grep);
  * ``left_eef_link`` / ``right_eef_link`` are likewise USD-only Xforms.

What *is* reusable is the expensive part: 30 links of hand-tuned collision
spheres, all of whose link names do exist in ``r1pro.urdf`` (verified: the
set difference is empty).  So this rewrites the shipped config rather than
authoring spheres from scratch.

Transform applied:
  base_link      -> ``base_link``, the actual URDF root (nothing is a child of
                    it), instead of the USD-only ``base_footprint_x``.
  cspace         -> the six base_footprint joints are dropped, leaving the 22
                    real articulated joints.
  eef links      -> re-added as ``extra_links`` with the offset the importer
                    uses (position [0,0,-0.06], orientation [0,1,0,0] as wxyz),
                    so ``ee_link``/``link_names`` still resolve.
  usd_* keys     -> dropped; they only apply to use_usd_kinematics: True.

The base is deliberately *not* modelled as a joint.  A collision query is
evaluated at one configuration, and obstacle positions are already expressed in
the robot's frame, so a fixed root is the correct model for checking -- and it
avoids inventing base DOF that the URDF cannot support.  This config is for
collision checking, not for planning base motion.
"""
from __future__ import annotations

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

SRC = Path("/home/yfzhang/nvme1/BEHAVIOR-1K/datasets/omnigibson-robot-assets"
           "/models/r1pro/curobo/r1pro_description_curobo_default.yaml")
URDF_DIR = Path("/home/yfzhang/nvme1/BEHAVIOR-1K/datasets/omnigibson-robot-assets"
                "/models/r1pro/urdf")
DST = Path("/home/yfzhang/WorkSpace/openeta/third_party/curobo/src/curobo"
           "/content/configs/robot/r1pro.yml")
# Patched URDF lives beside the config, not in the read-only asset tree.
PATCHED_URDF = DST.parent / "r1pro_description" / "r1pro_curobo.urdf"

BASE_JOINT_PREFIX = "base_footprint_"
EEF_OFFSET = [0.0, 0.0, -0.06]
EEF_QUAT_WXYZ = [0.0, 1.0, 0.0, 0.0]  # source cfg lists orientation [0,1,0,0]
FINGER_OPEN_M = 0.05      # prismatic upper limit in r1pro.urdf


def write_patched_urdf() -> list[str]:
    """Copy r1pro.urdf, giving `continuous` joints the limits cuRobo demands.

    cuRobo's URDF parser reads ``joint.limit.effort`` unconditionally, but a
    ``continuous`` joint has no ``<limit>`` in URDF by spec -- so parsing dies
    with ``'NoneType' has no attribute 'effort'``.  R1Pro's three wheel joints
    are continuous, and they enter the tree because the wheel links are declared
    collision links.

    Dropping the wheels from collision_link_names would also fix the crash but
    would silently stop checking the base geometry.  Pinning them instead is
    lossless for collision purposes: a wheel is rotationally symmetric about its
    own axis, so its swept geometry does not depend on that angle.  The joints
    are additionally locked in the config so they are not planning DOF.
    """
    tree = ET.parse(URDF_DIR / "r1pro.urdf")
    root = tree.getroot()
    patched: list[str] = []
    for joint in root.findall("joint"):
        if joint.get("type") != "continuous":
            continue
        name = joint.get("name") or "?"
        joint.set("type", "revolute")
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("lower", "-3.14159")
        limit.set("upper", "3.14159")
        limit.set("effort", "100.0")
        limit.set("velocity", "10.0")
        patched.append(name)

    PATCHED_URDF.parent.mkdir(parents=True, exist_ok=True)
    tree.write(PATCHED_URDF, encoding="utf-8", xml_declaration=True)

    # Meshes are referenced relative to the URDF directory, so the asset root
    # must still resolve.  Symlink rather than copy: the mesh set is large and
    # the asset tree is read-only anyway.
    link = PATCHED_URDF.parent / "meshes"
    if not link.exists():
        try:
            link.symlink_to(URDF_DIR / "meshes")
        except OSError:
            shutil.copytree(URDF_DIR / "meshes", link)
    return patched


def main() -> int:
    cfg = yaml.safe_load(SRC.read_text())
    k = cfg["robot_cfg"]["kinematics"]

    patched_joints = write_patched_urdf()

    # ── point at the patched URDF ───────────────────────────────────
    k["urdf_path"] = str(PATCHED_URDF)
    k["asset_root_path"] = str(PATCHED_URDF.parent)
    k["use_usd_kinematics"] = False
    for key in ("usd_path", "usd_robot_root", "isaac_usd_path",
                "usd_flip_joints", "usd_flip_joint_limits"):
        k.pop(key, None)

    # ── root at the URDF's actual root ──────────────────────────────
    k["base_link"] = "base_link"

    # ── drop the USD-only virtual base DOF ──────────────────────────
    cs = k["cspace"]
    names = list(cs["joint_names"])
    keep = [i for i, n in enumerate(names) if not n.startswith(BASE_JOINT_PREFIX)]
    cs["joint_names"] = [names[i] for i in keep]
    for field in ("null_space_weight", "cspace_distance_weight", "retract_config"):
        val = cs.get(field)
        if isinstance(val, list) and len(val) == len(names):
            cs[field] = [val[i] for i in keep]
    if isinstance(k.get("lock_joints"), dict):
        k["lock_joints"] = {j: v for j, v in k["lock_joints"].items()
                            if not j.startswith(BASE_JOINT_PREFIX)}
    else:
        k["lock_joints"] = {}

    # OmniGibson writes `null` to mean "lock wherever the joint currently is",
    # and resolves it against the live articulation before calling cuRobo.  A
    # standalone config has no such articulation, and cuRobo's loader feeds
    # lock values straight to torch (None -> dtype object -> TypeError), so
    # every null needs a real number here.
    #
    # Fingers are pinned OPEN.  An open gripper is the wider of the two
    # envelopes, so checking against it cannot under-report a collision that a
    # closed gripper would have had -- the conservative direction for a safety
    # check.  Held objects are handled separately by the attachment proxy.
    for joint, value in list(k["lock_joints"].items()):
        if value is not None:
            continue
        k["lock_joints"][joint] = FINGER_OPEN_M if "finger" in joint else 0.0

    # Wheels are geometry, not planning DOF: pin them so they contribute their
    # collision spheres without adding controllable joints.  The steer joints
    # come along because they sit between base_link and the wheel links on the
    # path cuRobo walks to reach them -- so they become active DOF too, and
    # cuRobo requires every active joint to appear in cspace.  Locking is
    # correct for both: this config checks the arms against the world, and the
    # base is positioned by the simulator, not planned here.
    for joint in patched_joints + ["steer_motor_joint1", "steer_motor_joint2",
                                   "steer_motor_joint3"]:
        k["lock_joints"].setdefault(joint, 0.0)

    # ── re-add the eef frames the USD import would have created ─────
    extra = dict(k.get("extra_links") or {})
    for side in ("left", "right"):
        extra[f"{side}_eef_link"] = {
            "parent_link_name": f"{side}_gripper_link",
            "link_name": f"{side}_eef_link",
            "fixed_transform": EEF_OFFSET + EEF_QUAT_WXYZ,
            "joint_type": "FIXED",
            "joint_name": f"{side}_eef_joint",
        }
    k["extra_links"] = extra

    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(
        "# Generated by scripts/gen_r1pro_curobo.py -- do not hand-edit.\n"
        "# Derived from OmniGibson's r1pro_description_curobo_default.yaml,\n"
        "# rewritten to load from r1pro.urdf instead of a live USD stage.\n"
        + yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    )

    print(f"wrote {DST}")
    print(f"  urdf       : {PATCHED_URDF}")
    print(f"  patched    : {patched_joints} (continuous -> revolute, then locked)")
    print(f"  base_link  : {k['base_link']}")
    print(f"  ee_link    : {k.get('ee_link')}  link_names={k.get('link_names')}")
    print(f"  cspace     : {len(cs['joint_names'])} joints "
          f"(dropped {len(names) - len(keep)} base_footprint)")
    print(f"  spheres    : {len(k.get('collision_spheres') or {})} links")
    print(f"  extra_links: {sorted(extra)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
