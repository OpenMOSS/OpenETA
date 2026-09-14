---
name: openeta-atomic
description: Operate an OpenETA atomic-profile robot episode using visual points, Cartesian motion and gripper control.
---

Use fresh RGB images and measured robot state to choose your own manipulation
strategy. SAM3, AnyGrasp and AnyPlace are absent from this profile.

`mark_point` measures the first visible RGB-D surface at original image pixels.
Read the adjacent source_packet_id/camera_frame_id label. Crops report their
mapping back to original pixels. Marks are immutable world positions; they do
not track moved objects, identify an object's centre, or define a grasp by
themselves. Remeasure geometry after contact or gripper actions.

`move_to` controls the Panda grip-site, not the wrist camera or an object centre.
All positions, offsets and deltas are metres. A surface point plus a chosen
offset can define a free-space waypoint. Consecutive moves define a path.
For transport near obstacles or a large orientation change, choose a short route
from the visible scene before moving: where to gain clearance, where to turn,
and from which side to approach. Consider the palm, arm and carried object's
swept space as well as the grip-site. The controller follows straight position
segments with gradual rotation; it does not plan a global detour. Adding a
collinear midpoint to a blocked segment does not create a detour.

Optional `waypoints` supplies up to four ordered intermediate poses before the
top-level final pose. Each pose uses `xyz_m`, `point_id` plus optional `offset_m`,
or orientation alone. Routes use strict orientation; omitted orientation inherits
the preceding target, and an orientation-only pose retains the preceding target
position. Relative deltas and contact grants are not supported within routes.
Use a separate call for intentional contact and gripper actuation. A short direct
move needs no extra waypoint. Use separate calls when a view/retention check is
needed before proceeding, particularly immediately after grasping.

Every segment gets fresh IK from the actual state and ordinary collision checks;
intermediate poses must be reached without corner smoothing. A failure stops the
route and returns completed_count, stopped_index, per-segment actual state and
remaining_indices (all indices are zero-based). Inspect and replan from actual
state; do not replay the original route after partial/unknown execution. Route
preview is only a numbered projected polyline, with no IK/path feasibility proof.
World direction `approach_world` is grip-site local +Z; `jaw_world` is local +X,
the jaw opening axis. For a fully specified orientation provide both nonparallel
directions; the Host normalizes and orthogonalizes them. For example, downward
approach [0,0,-1] with jaw [1,0,0] defines one top-down orientation. Arbitrary
approach directions are available. Omit both to preserve orientation.

For ordinary empty-handed pre-grasp orientation changes, prefer
`orientation_mode="parallel_jaw_symmetric"` when swapping the two fingers
preserves your intent. The Host checks two equivalent orientations, ranks IK
joint travel and limit margin, and reports the actual selected target. This can
avoid unnecessary half-turns. It requires an open, empty gripper without an
existing contact constraint. Use `strict` (the default) for held objects,
fixture manipulation, or a required wrist view. A rejected symmetric request
does not mean the strict pose is unreachable. The selection is endpoint-only;
`path_check=not_run` means local controller convergence remains unproven.

Preview shows the proposed grip-site in yellow, maximum jaw span in orange,
approach in green, actual grip-site in magenta, and the measured point in cyan.
Drawn geometry does not have its own depth. Preview does not move the robot or
prove grasp quality. Execution runs fresh IK and Mink collision checks; inspect
the actual endpoint, remaining error and images even after a tool error.

If a grasp approach stops at an iteration limit, collision, or controller error,
the intended grip-site may not have been reached. Do not close on the assumed
target. Inspect the actual pose and remaining error, remeasure the visible
surface, then correct the approach or retreat and choose another orientation.
Avoid repeating the same failed target unchanged. Large wrist rotations belong
in clear space before the final short approach.

A `rotation_corridor` failure means tested candidate orientations left the planned
rotation path; consider a different intermediate orientation or translate with
orientation held before turning. A `position_corridor` failure calls for a
materially different clearance segment. `candidate_obstacles` describes rejected
candidate steps, not a collision asserted at the current pose. Use its robot-part
and contact-target relation together with the returned images.

After a blocked withdrawal or reorientation, consider `motion_mode="recovery"`
with one short target within 0.15 m of the measured grip-site. You explicitly
permit up to 20 mm deviation from the position segment and 12 degrees from its
rotation path, so choose visible clearance for the whole arm, palm and any held
object. Final pose tolerances and collision/contact checks remain unchanged.
Recovery can re-solve local velocity and posture when a candidate is blocked;
it is not a global escape planner and can still fail. Use a separate call without
waypoints, `contact_point_id` or symmetric orientation. Keep ordinary intentional
contact moves in strict mode. Inspect the endpoint before continuing.

`candidate_trace` separates each candidate's tracking test from its geometry
test. `geometry_status="not_checked"` proves nothing about clearance, and
`checks_passed` does not establish physical execution. Reducing a step may fix
one bound while another still blocks it. Use `recent_motion_progress` to notice
repeated negligible measured movement and change approach; unavailable/unknown
motion remains unknown. A previously reached pose is a planning reference and
must be checked again after scene changes.

Read `interventions` together with the motion receipt: it explains Host rejections,
substituted operations, resolved orientation/horizon and controller step reductions.
`current_request` identifies this reply. `episode_status.previous_request` is an
explicitly historical receipt for recovering a lost reply, not a new execution.
Respect `not_started`, `partial` and `unknown`; never replay an unknown action.
Inspect `repair` for callable recovery hints when present.

For stalled motion, `stall_context.active_clearance_constraints` reports binding
QP rows at the last pre-actuation configuration; it does not prove the cause of
the entire stall. `measured_robot_contacts` describes actual final contact.
`collision.detected=false` only means the hard stop was not triggered, not that
contact or clearance constraints are absent. Unavailable diagnostics mean unknown.
If the palm is constrained or contacting outside the marked target, use the
images to choose a checked retreat and different approach; enlarging the target
grant or repeating the same segment does not resolve that obstruction.

When a goal names a compartment or an object's front/back, ground that relation
in the object's structure across the provided views. Image top/bottom alone does
not define the object's front/back. Compare openings and dividers; measure visible
edges when ambiguity would change the chosen placement. If an object appears
supported inside a compartment but official success remains false, reconsider
compartment identity and the other goal conditions before assuming it only needs
to be pushed deeper. A recovery grasp needs fresh geometry and an accessible
exposed edge; the previous gripper/object offset may no longer apply.

For intentional contact, provide contact_point_id naming a recently measured
surface point near the target grip-site. The Host binds only that target's
gripper contact; other collisions remain checked. Fixture contact normally binds
one collision piece. Experimental local-patch mode also permits adjacent pieces
within the marked moving part's local patch; it does not authorize the whole
cabinet, remote surfaces, or deep penetration. Follow the returned contact scope.
Closing may reuse the previous
contact move's binding or use a new contact_point_id. Check retention visually;
aperture or successful motion alone is not proof of grasp or placement.

Motion/gripper feedback can include final-configuration finger and fingerpad
contact booleans. These describe contact with external geometry, without target
identity. `finger_body_only`, `single_pad`, or `no_contact` after closing is a
reason to inspect and reposition, not to assume a secure grasp. Even
`bilateral_pads` does not prove retention: make a small checked lift/pull and
inspect whether the intended object or handle follows before a long transfer.
If the gripper moves but the target does not, release, remeasure and change the
grip geometry. Do not infer drawer opening from end-effector travel alone.

For pushing or sliding, reaching the grip-site target does not establish object
motion. Check a short contact move against the images. If the intended object
did not move, inspect contact height and approach side before attempting a longer
sweep; use the changed scene to plan the next segment.

Only the official task checker establishes task success. Geometry operations
and previews consume native request budget; each executed move consumes up to
three internal Host tool calls in strict mode, or five when comparing symmetric
orientations. A route consumes the same internal budget per segment, even though
it is one native request. Plan within the returned budgets. `contact_state` reports
whether a close binding or symmetric orientation is currently available; a stale
or depth-edge contact mark requires a fresh interior surface measurement.
`episode.remaining_budget` estimates remaining strict/symmetric move capacity
from both internal stage limits. Time, gripper/observation calls and route segments
also consume budget. `execution_accounting` separates preflight calls from physical
dispatch and known/unknown execution; it does not grant extra attempts.
