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
  - ik_preview_check
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
2. Compile the targeted `grasp_pose_estimate` candidate chosen for pickup so the
   host evidence graph can bind that exact candidate to `details.outputs.source`.
   On the same original RGB image, call
   `sam3` for the basket, bin, or other placement region with
   `evidence_role="placement_region"`, then resolve its selection obligation with
   `select_sam3_detection`. Target-object and placement-region selections occupy
   separate semantic evidence slots; never reuse the default `target_object` role
   for a receptacle. After object selection, use the RGB, depth, intrinsics, and
   mask from that aligned observation directly; do not call `observe` merely to
   refresh unchanged artifact paths.
3. When `host_resolved_inputs.anyplace.status=ready`, call `anyplace` with only
   its exact `bundle_id`. The host resolves the frozen RGB, depth, intrinsics,
   selected object mask, placement-region mask, grasp candidate, and source
   provenance atomically. Never copy, shorten, reconstruct, or override those
   fields in model output. A ready bundle may report
   `placement_evidence_reuse.mode=fixed_camera_identity`: the host has safely
   retained the unchanged receptacle mask while rebinding RGB-D to a newer grasp
   source on the same fixed scene camera, so do not segment it again. If the bundle
   reports `grasp_rebase.mode=calibrated_world_invariant`, a wrist-refined grasp
   has been expressed in the prior fixed-camera geometry using exact packet-owned
   extrinsics. Call the ready bundle directly; do not discard the wrist candidate
   or rebuild a fixed-camera grasp. If the bundle
   reports `placement_source_mismatch`, execute its exact `repair_call` parameters
   instead of choosing a current or remembered image path yourself. The repair
   may request `placement_region` on the grasp source, or `target_object` on a
   fixed scene camera when the active grasp came from wrist. In the latter case,
   select the target, estimate and compile a grasp from the refreshed host bundle,
   and retain the existing placement selection so the resolver can form a
   same-camera bundle. Wrist/hand cameras move and therefore do not qualify for
   fixed-camera reuse.
   Never run grasp estimation on the receptacle as a substitute for AnyPlace.
4. Complete the pickup using the selected grasp. After closing the gripper,
   call `observe` and require positive evidence that the object moved with the
   end effector before starting placement.
5. Choose one complete `placement_candidates[i]` from the retained AnyPlace
   result as the placement reference. Copy only the result's `result_id` and
   that candidate's `id` into `camera_pose_to_world` as
   `placement_result_id` and `candidate_id`. The host atomically resolves the
   exact `place_grasp_pose`, original observation packet, and matching camera
   extrinsics. Never open the AnyPlace artifact with `python_exec` merely to
   reconstruct these fields, and never mix the two IDs with model-supplied pose
   or calibration overrides. If the returned candidates already include an
   interior-mask clearance metric, prefer the compatible candidate with the
   greatest clearance. Otherwise use the returned rank/score plus visual
   judgment; do not delay an executable handoff merely to reconstruct an
   unavailable clearance metric from artifacts. Rank remains a heuristic, not
   proof of a safe release.
   Do not reuse a receptacle grasp pose or invent an unrelated world-frame
   coordinate.
6. Treat the transformed AnyPlace pose as the low release reference, not as a
   one-step carry trajectory. Its `placement_reference.execution_authorized=false`
   is deliberate: use the current EEF receipt and fresh agentview/wrist evidence
   to propose a safe raised carry waypoint, followed by one or more bounded
   horizontal waypoints and a separate descent. Preserve the current EEF
   orientation unless an exact explicit-orientation IK check supports a change,
   and do not combine lateral carry with receptacle descent.
   Every planned endpoint must pass a current-epoch `ik_preview_check` for the
   exact xyz and the same explicit-or-preserve-current orientation policy before
   motion. Pass the returned `ik_receipt_id` to `move_to`; never reproduce the
   checked pose or rotation arrays. A gripper or robot mutation invalidates the previous receipt. Use its
   position/orientation residuals to adjust an unreachable waypoint; do not treat
   `unknown` as proof of safety. This is endpoint IK only. Keep
   `enable_collision_check=true` and require the motion receipt to report
   complete per-step trajectory/world coverage from the active controller. A
   confirmed held object participates in the collision envelope. If motion is
   rejected with `collision_type=attached_object_world`, use the named obstacle
   and predicted pose to choose a higher or more central IK-checked waypoint;
   do not repeat the same diagonal path. `steps_executed=0` means no motion
   occurred. If conservative geometry overlaps at the starting pose, choose a
   monotonic escape increment that reduces the overlap; never disable collision
   checking to force the carry.
7. Inspect the fresh image after every carry waypoint. The earlier lift-probe
   PASS is stale after motion: continue only when the target is still co-located
   with the gripper and its source location remains vacant. If the target is
   visible elsewhere and the closed-gripper openness has collapsed to the empty
   threshold, follow the `attachment_lost` recovery action so the current grasp
   candidate is rejected before regrasping.
8. Treat the transformed AnyPlace Z as a low geometric reference. Propose the
   final release pose from that reference plus fresh visual feedback; it remains
   subject to exact IK and attached-object collision checks. A bounded
   world-frame adjustment is allowed when visual evidence improves receptacle
   clearance. Centre the held object inside the receptacle before descending;
   the collision adapter may permit intentional corridor entry but will still
   reject rim overlap.
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
