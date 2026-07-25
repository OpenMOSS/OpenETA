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
  - sam3
  - select_sam3_detection
  - reject_sam3_detections
  - anyplace
  - camera_pose_to_world
  - move_to
  - gripper_control
---
# Place

Use this skill as text guidance only. Do not treat `place` as an executable
macro. After each tool result, inspect the returned observation or tool output
before choosing the next tool call.

## Recommended Tool Sequence

1. For a combined pick-and-place task, plan placement before the first grasp
   motion. AnyPlace requires the object and placement region from one aligned
   pre-grasp RGBD observation. Do not wait until the object has moved.
2. Retain the targeted `grasp_pose_estimate` result used for pickup, including its selected
   candidate and `details.outputs.source`. On the same original RGB image, call
   `sam3` for the basket, bin, or other placement region and resolve its
   selection obligation with `select_sam3_detection`. Do not segment or select
   the placement region before targeted grasp estimation succeeds: the runtime has one
   active SAM3 selection slot, and doing so would overwrite the selected object
   mask. After object selection, use the RGB, depth, intrinsics, and mask from
   that aligned observation directly; do not call `observe` merely to refresh
   unchanged artifact paths.
3. Call `anyplace` with the exact original RGB, depth, intrinsics, selected
   object mask, selected placement-region artifact, and
   `selected_grasp={candidate, source}` from that targeted grasp result.
   The planner context field `retained_targeted_grasp` contains the exact
   `candidate` and `source`; copy them directly without calling `get_memory`.
   Segment the placement region on `retained_targeted_grasp.source.rgb`, not a
   newer observation, and do not shorten or reconstruct any retained path.
   Never run grasp estimation on the receptacle as a substitute for AnyPlace.
4. Complete the pickup using the selected grasp. After closing the gripper,
   call `observe` and require positive evidence that the object moved with the
   end effector before starting placement.
5. Choose one complete `placement_candidates[i].place_grasp_pose` from the
   retained AnyPlace result as the placement reference. When matching
   intrinsics and a receptacle mask are available, prefer the compatible
   candidate whose projected gripper tip has the greatest interior mask
   clearance; score/rank remains a fallback, not proof of a safe release.
   Transform the selected pose with
   `camera_pose_to_world` using the matching original camera extrinsics; do not
   reuse a receptacle grasp pose or invent an unrelated world-frame coordinate.
6. Treat the transformed AnyPlace pose as the low release reference, not as a
   one-step carry trajectory. First move the closed gripper to the profile-derived
   pre-place hover over the same world X/Y. Raise to the supplied clearance before
   translating, then use the bounded horizontal waypoints from
   `placement_motion_guidance` rather than one long carry. Preserve the current
   EEF orientation and do not combine lateral carry with receptacle descent.
7. Inspect the fresh image after every carry waypoint. The earlier lift-probe
   PASS is stale after motion: continue only when the target is still co-located
   with the gripper and its source location remains vacant. If the target is
   visible elsewhere and the closed-gripper openness has collapsed to the empty
   threshold, follow the `attachment_lost` recovery action so the current grasp
   candidate is rejected before regrasping.
8. Treat the transformed AnyPlace Z as a low geometric reference. Descend only
   to the profile-derived `placement_motion_guidance.release_pose`, so the held
   object enters the receptacle without
   driving the gripper or object into its rim. A bounded world-frame adjustment
   is allowed when fresh visual feedback improves receptacle clearance.
9. Call `gripper_control` with `position=1` only after the vertical placement
   motion succeeds and fresh evidence still supports attachment over the
   receptacle.
10. Retreat with `move_to`, then call `observe` to verify the object was released
    in the intended place and check the official task reward.

## Recovery Notes

- If the target receptacle or surface is ambiguous, call `ask_human` before
  moving.
- If an already-held object has no retained targeted grasp-estimation provenance or
  pre-grasp aligned placement mask, do not fabricate AnyPlace inputs. Ask for a
  new supported plan or use an explicit task-provided release pose.
- If the target is occluded, observe from another camera or request a broader
  scene query before choosing a release pose.
- If the simulator reports an unreachable or colliding path, choose a higher
  pre-place pose or a different approach direction.
- If the object remains in the gripper after opening, retry `gripper_control`
  once, observe, then ask for help or replan.
- Never release an object from stale perception. Observe again after every
  world-mutating tool call.

For explicit clearance or controller-profile discovery, use the
`embodiment_explore` skill outside the benchmark episode. Do not copy a
successful value from another robot or environment into this task.
