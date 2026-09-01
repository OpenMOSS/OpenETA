---
name: stack
description: Skeleton guidance for stacking one object on another.
version: v1
editable: true
task_patterns:
  - stack <object> on <object>
  - put <object> on top of <object>
allowed_tools:
  - observe
  - sam3
  - select_sam3_detection
  - reject_sam3_detections
  - grasp_pose_estimate
  - compile_grasp_seed
  - camera_pose_to_world
  - ik_preview_check
  - move_to
  - gripper_control
---
# Stack

Use this as task guidance, not an executable macro. Combine the reusable pick
and place workflows, then add stability reasoning before release.

1. Identify and visually confirm both the movable object and the support object.
   Inspect nearby obstacles and the usable top support area.
2. Acquire a stable grasp using the pick skill. Prefer contact geometry that
   leaves the carried object's base and the release view unobstructed.
3. Choose a placement reference that centres the carried object's support polygon
   over a level, sufficiently large region of the lower object while preserving
   finger clearance.
4. Carry above surrounding clutter, align without sweeping either object, and
   descend separately. Stop if the support shifts or the held object slips.
5. Release only after fresh evidence supports stable contact. Retreat enough to
   reveal the stack, then verify that both objects remain stationary and the
   official task condition is satisfied.

If the support is too small, tilted, moving, or visually ambiguous, obtain more
evidence or ask for help rather than forcing the same placement.
