---
name: pick
description: Guidance for acquiring a target object with atomic tools.
version: v1
editable: true
task_patterns:
  - pick <object>
  - grasp <object>
  - take <object>
  - 抓取 <object>
  - 抓起来 <object>
  - 拿起 <object>
allowed_tools:
  - observe
  - retrieve_asset_reference
  - sam3
  - select_sam3_detection
  - estimate_depth_prior
  - enhance_depth
  - grasp_pose_estimate
  - molmopoint
  - reject_sam3_detections
  - compile_grasp_seed
  - compute_wrist_alignment
  - propose_wrist_viewpoints
  - camera_pose_to_world
  - ik_preview_check
  - prepare_attachment_probe
  - assess_attachment_probe
  - python_exec
  - move_to
  - follow_eef_trajectory
  - gripper_control
---
# Pick

Use this as reusable task guidance, not an executable macro. It explains why
and when capabilities are useful; the live tool contracts exclusively define
their request fields, returned references, validity rules, and repair payloads.

## Evidence and target identity

1. Inspect the scene before choosing a grasp target. Normalize the task
   target into a concise English visual phrase when using text segmentation.
2. Segment the intended object and visually inspect the original image plus
   candidate overlays. Scores rank proposals but do not prove identity. Confirm
   the candidate that matches the task, or reject the set and obtain better
   evidence. Preserve the same physical instance across later views; do not
   silently relabel a nearby object when the target moves or becomes occluded.
3. When text segmentation misses an unusual known asset, use an available
   controlled asset reference or point-grounding capability to localize it in
   the original scene. Do not add category guesses that were not supported by
   the task or reference image.
4. Use depth enhancement only when sensor depth is too sparse for grasp
   estimation. Treat enhanced depth as candidate-generation evidence, not as a
   substitute for the sensor geometry required by collision checking.
5. An exact-task playbook may supply object appearance, likely scene region, or
   a previously useful strategy. Treat it only as a scoped prior: verify the
   current object and scene visually before using it. Similar language or a
   similar object from another task is not transferable evidence.

## Grasp estimation and selection

6. Estimate grasps only after target identity and aligned RGB-D evidence are
   coherent. If the host reports that its prepared input is stale or incomplete,
   refresh the relevant perception instead of reconstructing private paths or
   calibration data. Use the grasp-estimation facade rather than choosing a
   concrete backend directly. Estimator diversity is useful after physical
   evidence such as a confirmed slip, not after an unrelated transport failure.
7. Choose for transport stability, not merely the highest estimator score.
   Compare finger aperture, visible contact depth, object geometry, collision
   clearance, prior physical outcomes, and the projected contact location on
   the target mask. Backend scores are local to each estimator and are not
   directly comparable.
8. Treat visual grasp-advisor output as evidence, not authority. Prefer a
   candidate with broad, deep, symmetric contact on a stable body region. A
   contact on a cap, rim, shoulder, thin edge, or barely overlapping boundary is
   a slip risk. If every candidate has that weakness, obtain a materially
   different view or candidate set rather than choosing the least-bad reachable one.
9. Characterize the target geometry only when visually clear. Compare compatible
   grasp strategies exposed by the runtime with the estimator-native pose when
   their geometry assumptions match. Candidate strategies require an explicit
   Agent choice and remain experimental; validated strategies may be stronger
   priors, but neither bypasses visual, reachability, collision, contact, or
   attachment checks. Task- or episode-specific evidence belongs in a playbook
   or strategy record, not in this skill.
10. Compile the selected estimator candidate before robot motion. Compilation is
    the calibrated handoff from camera-frame grasp geometry to a world-frame EEF
    reference. Do not treat an uncompiled estimator pose as a robot target.

## Approach and near-field refinement

11. Treat clearance, alignment, and contact references as ordinary geometric
    waypoints rather than host-owned task phases. Choose the number and geometry
    of motion edges from the current EEF pose, visible obstacles, carried-object
    extent, and controller feedback.
12. Approach contact through the corridor behind the selected grasp direction.
    Avoid a long cross-axis sweep near the object: endpoint reachability alone
    does not prevent the gripper from pushing the target away on the way in.
    Establish a compatible contact orientation before the final inward motion.
13. Use one checked endpoint for a short clear motion. For a long transit,
    precision-sensitive approach, visible obstruction, or collision recovery,
    choose a small ordered route and observe from the actual endpoint before
    extending it. A midpoint on a failed straight path normally preserves the
    same collision; route around the named obstacle instead. Keep collision
    checking enabled and treat a controller stop as evidence to change geometry,
    not as permission to replay or disable checks.
14. Near the target, use the wrist view when it can materially improve contact:

    - Use calibrated wrist alignment when the approach orientation and axial
      contact depth remain credible and only a bounded lateral correction is
      needed.
    - Move to a target-facing wrist observation viewpoint when the target is
      clipped or poorly framed.
    - Run a fresh wrist-view grasp estimate when orientation, contact depth, or
      candidate identity is uncertain. The wrist estimate is a new candidate
      branch; it does not silently inherit or replace an earlier strategy.

    Keep visual adjustments inside the host-provided safety envelope.
15. A reachable endpoint proves kinematics, not controller convergence or path
    safety. Compare execution-seed quality and residual diagnostics. If a full
    contact orientation is infeasible, use a position-only reach only as a safe
    observation or retreat waypoint; do not close at the same position with an
    unrelated orientation and call it the same grasp.

## Contact, attachment, and transport

16. Close only after the actual motion receipt and fresh dual-view evidence show
    that the fingers reached the intended contact geometry. A reported collision
    may require visual reassessment, but collision with unrelated scenery does
    not by itself make a finger-only close unsafe when contact geometry is still
    valid. Follow the live gripper contract for the closed command.
17. A close acknowledgement or measured aperture is not proof of attachment.
    Keep the gripper latched and prepare a short, visually justified probe from
    the measured EEF pose. Execute the frozen probe geometry through the normal
    reachability and motion tools, then assess attachment from target/EEF
    co-motion plus vacancy at the source location.
18. Continue only on positive attachment evidence. Ambiguous evidence calls for
    another observation or reassessment; an empty close, visible slip, or failed
    co-motion calls for contact repair or a materially different candidate.
19. Once attachment is confirmed, include the whole held object in route
    clearance. Lift it clear of nearby clutter before lateral transport. Do not
    combine a long lateral carry with descent toward a receptacle, because a rim
    can strip an object from an otherwise latched gripper.

## Recovery choices

- Wrong or ambiguous target: reacquire identity before estimating another grasp.
- Weak candidate set: change view, estimator evidence, or strategy rather than
  cycling ranks from an unchanged result.
- Contact motion stopped or missed: reason from the actual endpoint and repair
  the approach corridor, orientation, or candidate.
- Empty close or failed probe: reopen when safe, reject the physical branch, and
  choose a materially different contact.
- Attachment lost during transport: stop placement, re-observe the displaced
  object, and begin a new evidence branch.
- Missing calibration, backend failure, timeout, or malformed response is not
  physical evidence against the candidate. Follow the structured tool recovery
  or report the capability gap.

For explicit robot, controller, sensor, or environment characterization, use
the `embodiment_explore` skill outside the benchmark episode.
