---
name: pull
description: Draft guidance for short pull manipulation.
version: v1
editable: true
task_patterns:
  - pull <object>
  - move <object> by pulling
  - drag <object>
allowed_tools:
  - observe
  - scene_detector
  - sam3
  - move_to
  - gripper_control
---
# Pull

Use this skill as text guidance only. Do not treat `pull` as an executable
macro. This is draft placeholder guidance until OpenETA has dedicated pull
tools for hook/contact planning, grasp-assisted pulling, or controlled dragging.

## Recommended Tool Sequence

1. Call `observe` to refresh object pose, pull direction, support surface
   geometry, and nearby obstacles.
2. Call `scene_detector` to identify the movable object and any handles, lips,
   edges, or reachable contact regions.
3. Call `sam3` if the pull contact region or object boundary is unclear.
4. Choose a pull strategy: gripper-width contact, hook-like contact, or
   grasp-assisted pull if appropriate.
5. Use `gripper_control` only when the pull strategy requires a specific
   gripper width or grasp state.
6. Call `move_to` to approach the contact pose. The simulator controller owns
   reachability and path-collision checks; stop on a structured rejection.
7. Execute one short `move_to` pull segment, then call `observe`
   before deciding whether to continue.

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
