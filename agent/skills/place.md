---
name: place
description: Guidance for placing a held object on or inside a target receptacle.
version: v1
editable: true
task_patterns:
  - place <object> on <target>
  - put <object> into <target>
  - place <object> in <target>
allowed_tools:
  - observe
  - scene_detector
  - sam3
  - move_to
  - gripper_control
---
# Place

Use this skill as text guidance only. Do not treat `place` as an executable
macro. After each tool result, inspect the returned observation or tool output
before choosing the next tool call.

## Recommended Tool Sequence

1. Call `observe` to confirm the robot is holding the object and to refresh the
   target receptacle or surface state.
2. Call `scene_detector` to identify the target surface, bin, basket, or
   receptacle and any nearby obstacles.
3. Call `sam3` if the target placement area needs visual segmentation.
4. Choose a candidate release pose above the target area with clearance for the
   held object and gripper.
5. Call `move_to` to move to a pre-release pose, then to the release
   pose if the previous observation/check result is still valid.
   The simulator controller owns reachability and path-collision checks; inspect
   its structured result before issuing another motion.
6. Call `gripper_control` to open the gripper.
7. Retreat with `move_to`, then call `observe` to verify the object was
   released at the intended place.

## Recovery Notes

- If the target receptacle or surface is ambiguous, call `ask_human` before
  moving.
- If the target is occluded, observe from another camera or request a broader
  scene query before choosing a release pose.
- If the simulator reports an unreachable or colliding path, choose a higher
  pre-place pose or a different approach direction.
- If the object remains in the gripper after opening, retry `gripper_control`
  once, observe, then ask for help or replan.
- Never release an object from stale perception. Observe again after every
  world-mutating tool call.
