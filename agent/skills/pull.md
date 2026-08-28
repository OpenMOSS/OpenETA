---
name: pull
description: Guidance for opening drawers and other short grasp-assisted pull manipulation.
version: v1
editable: true
task_patterns:
  - pull <object>
  - move <object> by pulling
  - drag <object>
  - open <drawer>
  - open the <drawer> of <cabinet>
  - pull open <drawer>
allowed_tools:
  - observe
  - sam3
  - select_sam3_detection
  - reject_sam3_detections
  - grasp_pose_estimate
  - compile_grasp_seed
  - compute_wrist_alignment
  - prepare_attachment_probe
  - assess_attachment_probe
  - move_to
  - follow_eef_trajectory
  - gripper_control
---
# Pull

Use this as task guidance, not an executable macro. The live tool contracts own
all exact request and result formats.

1. Observe the contact region, expected travel direction, support geometry, and
   nearby obstacles. For a drawer, treat the requested handle as the grasp target
   and the drawer travel as a closed-gripper pull, not a pick-and-place task.
2. Segment and confirm the intended handle or contact region when its boundary is
   unclear. Choose a grasp or hook-like contact consistent with the visible
   affordance and gripper width.
3. Compare compatible handle strategies with the estimator-native candidate.
   A geometry label is a strategy-matching hint, not attachment evidence, and a
   candidate strategy still requires an explicit Agent choice unless validated
   for automatic activation.
4. Approach and close using the reusable contact guidance from the pick skill.
   After close, prepare a short probe whose geometry matches the mechanism: a
   mostly linear probe for a drawer, or a small arc for a hinge. Assess attachment
   from fresh visual co-motion evidence before beginning the actual pull.
5. Preserve the gripper state and a mechanism-compatible EEF orientation. Execute
   one short pull segment along the observed travel direction, then inspect object
   displacement, attachment, and newly exposed collision geometry before deciding
   whether to continue.
6. Continue only while the handle remains attached and the mechanism moves as
   expected. Stop when current evidence or the official environment reward shows
   completion.

Recovery guidance:

- Lost contact or unexpected rotation: stop and replan from the new geometry.
- Unclear travel direction or hidden handle: obtain another view or ask for help.
- Motion command completion without mechanism displacement is not task progress;
  change contact or pull geometry rather than repeating the same segment.
