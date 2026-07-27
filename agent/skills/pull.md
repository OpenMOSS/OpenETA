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
  - move_to
  - gripper_control
---
# Pull

Use this skill as text guidance only. Do not treat `pull` as an executable
macro. For drawers, treat the handle as the grasp target and the drawer travel
as a short closed-gripper pull. Do not treat opening as a pick-and-place task.

## Recommended Tool Sequence

1. Call `observe` to refresh object pose, pull direction, support surface
   geometry, and nearby obstacles.
2. Call `sam3` if the pull contact region or object boundary is unclear.
3. For a drawer, visually select the requested handle, estimate a handle grasp,
   label clear handle geometry as `drawer_handle`, compile the candidate, and
   follow the normal wrist-aligned contact and close sequence. The label selects
   only a candidate task strategy and is not attachment evidence. For other
   targets, choose gripper-width contact, hook-like contact, or grasp-assisted
   pull if appropriate.
4. Use `gripper_control` only when the pull strategy requires a specific
   gripper width or grasp state.
5. Call `move_to` to approach the contact pose. The simulator controller owns
   reachability and path-collision checks; stop on a structured rejection.
6. After verified handle attachment, preserve the closed gripper and EEF
   orientation. Execute one short world-frame pull segment along the drawer's
   travel axis, then call `observe` before deciding whether to continue.
7. Continue with bounded pull segments only while the handle remains attached.
   Stop as soon as the official reward or a clearly open drawer state is observed.

## Future Dedicated Tools

- A future pull planner tool should output contact/grasp mode, pull direction,
  segment length, and expected object displacement.
- A future pull execution tool should manage contact retention or grasp state
  inside the tool backend and still return control after a short observable
  segment.

## Recovery Notes

- Keep each pull segment short; re-observe between segments.
- If contact is lost or the object rotates unexpectedly, stop and replan from
  the new pose.
- If the pull requires a handle or reachable edge that is not visible, observe
  from another view or ask for clarification.
- Do not infer success from motion command completion alone. Verify object
  displacement in the next observation.
