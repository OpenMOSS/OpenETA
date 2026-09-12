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

For intentional contact, provide contact_point_id naming a recently measured
surface point near the target grip-site. The Host binds only that target's
gripper contact; other collisions remain checked. Closing may reuse the previous
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

Only the official task checker establishes task success. Geometry operations
and previews consume native request budget; each executed move consumes up to
three internal Host tool calls in strict mode, or five when comparing symmetric
orientations. Plan within the returned budgets.
