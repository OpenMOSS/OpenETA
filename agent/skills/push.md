---
name: push
description: Draft guidance for short planar push manipulation.
version: v1
editable: true
task_patterns:
  - push <object>
  - move <object> by pushing
  - slide <object>
allowed_tools:
  - observe
  - scene_detector
  - sam3
  - move_to
  - gripper_control
---
# Push

Use this skill as text guidance only. Do not treat `push` as an executable
macro. This is draft placeholder guidance until OpenETA has dedicated push
tools for contact planning and controlled planar displacement.

## Recommended Tool Sequence

1. Call `observe` to refresh object pose, support surface geometry, target
   displacement, and nearby obstacles.
2. Call `scene_detector` to list movable objects and relevant support surfaces.
3. Call `sam3` if the object boundary or contact region is unclear.
4. Choose a short push segment: approach pose, contact point, push direction,
   and distance.
5. Call `move_to` to approach the contact pose. The simulator controller owns
   reachability and path-collision checks; stop on a structured rejection.
6. Execute one short `move_to` push segment, then call `observe`
   before deciding whether to continue.

## Future Dedicated Tools

- A future push planner tool should output contact points, push direction,
  segment length, and expected object displacement.
- A future push execution tool should keep force/contact constraints inside the
  tool backend and still return control after a short observable segment.

## Recovery Notes

- Keep each push segment short; re-observe between segments.
- If the object rotates, slips, or catches on another object, stop and replan
  from the new pose.
- If the push path is blocked, choose a different contact side or ask for
  clarification.
- Do not infer success from motion command completion alone. Verify object
  displacement in the next observation.
