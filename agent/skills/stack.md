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
  - move_to
  - gripper_control
---
# Stack

Use this skill as text guidance only. Do not treat `stack` as an executable
macro. This skeleton combines pick and place guidance while adding stability
checks before release.

## Recommended Tool Sequence

1. Call `observe` to identify the object to stack and the support object.
2. Confirm both objects and any obstacles around the support object from the
   current observation.
3. Call `sam3` to segment the grasped object or support object if visual
   boundaries are uncertain.
4. Call `grasp_pose_estimate` with the host-joined RGB, depth, intrinsics,
   complete target-mask artifact, camera frame id, and scene epoch when the
   object is not already held.
5. For an unheld object, follow the pick skill's explicit SAM3 selection,
   normalized grasp candidate, `compile_grasp_seed`, atomic `move_to`, and separate
   gripper-control sequence.
6. Use `move_to` and `gripper_control` for placement as individual atomic tool
   calls. The simulator controller owns reachability and path-collision checks;
   inspect each structured result and observe after every world mutation.
7. Before release, choose a pose that places the object's center over the
   support area and leaves gripper clearance.
8. Open the gripper with `gripper_control`, retreat with `move_to`, and
   call `observe` to verify the stack remains stable.

## Recovery Notes

- If the support object is too small, unstable, or tilted, ask for help or
  choose a different placement target.
- If grasp candidates are weak, return to pick guidance and acquire a more
  stable grasp before attempting the stack.
- If the stack shifts after release, do not immediately retry the same motion;
  observe, assess stability, and replan.
- Do not continue stacking from stale perception. Observe after every
  world-mutating tool call.
