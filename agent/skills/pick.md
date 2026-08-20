---
name: pick
description: Guidance for acquiring a target object with atomic tools.
version: v1
editable: true
context_char_limit: 28000
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
  - gripper_control
---
# Pick

Use as text guidance only, not an executable macro. Inspect each result.

## Recommended Tool Sequence

1. Call `observe` to get the complete current scene observation.
2. Normalize the task target to a concise English visual phrase for `sam3`
   (for example, 牛奶盒 -> `milk box`, 方块 -> `cube`).
3. Call `sam3` with the exact short `source_packet_id` copied from visible
   observation evidence and the normalized `prompt`, for example `milk box` or
   `can`. The host resolves the session-owned local RGB-D paths, camera frame,
   and `source_observation`; never pass an image path to `sam3`. Add
   `camera_frame_id` only when the packet's default scene camera is not the
   intended view.
   Do not pass a non-English user phrase directly to `sam3` if a clear English
   object name is available.
   For an unusual asset that text segmentation misses, use
   `retrieve_asset_reference` with the exact task asset name plus the current
   scene `source_packet_id` and camera frame; never copy its local image path;
   do not append category guesses. Copy its original-image positive
   points unchanged into SAM3. Use the single bbox fallback only after the point
   mask and one dense grasp attempt produce no candidates.
4. Stop after `sam3`; dependent batched calls do not pass outputs. For every
   non-empty result, inspect the attached original/contact sheet and call
   `select_sam3_detection` with the exact `sam3_result_id` and `detection_id`.
   Score ranks candidates but does not prove identity. Gather another view when uncertain.
   The first confirmed target creates `target_identity_anchor.anchor_id`. For any
   later target-object selection from new detection evidence, copy that exact id
   as `identity_anchor_id` and explicitly set `identity_relation=same_instance`
   only after cross-view comparison. If fresh evidence proves the original target
   was wrong, use `identity_relation=replace_misidentified_anchor` with a concrete
   visual reason; never relabel a different object under the old anchor. Placement
   region selections do not use the target identity anchor.
5. For poor real-robot depth, optionally call `estimate_depth_prior` with only
   the current `source_packet_id` and intended `camera_frame_id`, then call
   `enhance_depth` with the same two short references. The host resolves aligned
   RGB-D, calibration, and the newest matching prior; never copy local depth or
   prior paths into either call. Use candidate depth only when its quality
   permits; collision checks must use sensor safety depth/cloud, never mono-filled
   geometry.
6. After selecting the target, inspect
   `host_resolved_inputs.grasp_pose_estimate`. When it reports `status=ready`,
   call `grasp_pose_estimate` with only its exact `bundle_id`. The host binds the
   selected mask, aligned RGB-D packet, intrinsics, camera frame, and object-scene
   epoch atomically; never copy or override those fields in model output. If the
   bundle is stale or incomplete, segment the target on a current aligned RGB-D
   observation instead of repairing paths by hand; never default to `detections[0]`.
   The explicit RGB-D parameter
   form remains a compatibility fallback only when no host bundle is available.
   Deployment fallback stays inside the facade; do not call a concrete grasp
   backend directly. After outcome evidence justifies estimator diversity—such
   as a visually confirmed physical slip—this same facade may be called with the
   exact `bundle_id` plus an Agent-owned `backend_preference` ordered subset
   (`anygrasp`, `graspgenx`). This changes attempt order only:
   unlisted configured backends remain structured fallbacks, and the returned
   `backend_policy` receipt reports the effective order and selected backend.
   Do not request diversity merely because a transport failed, and never use this
   field to infer that scores from different backends are comparable. Mono-filled
   depth is never collision evidence. Execution
   requires the active robot adapter/controller to return explicit trajectory,
   world, and attached-object collision coverage; if that capability is absent,
   do not pretend enhanced depth supplied it.
7. Read the normalized grasp candidate list. Candidate poses use the
   camera/OpenCV GraspNet convention and are sorted by backend-local score.
   Scores are backend-local. Choose using identity, width/calibration, collision,
   geometry, and prior outcomes. When the ToolResult includes
   `target_mask_candidate_projection`, compare each translation/tip pixel with
   the mask bbox and centroid. A candidate anchored at a thin top/side boundary
   is shallow-grasp evidence, especially after a prior slip; it is not an
   automatic rejection. Record the id and rationale in Agent memory;
   no host task phase chooses it.
   When `grasp_selection_advice` is present, treat it as read-only visual evidence:
   compare its recommendation, rejected-candidate reasons, confidence, and
   uncertainties with task-level constraints and prior outcomes. The advisor cannot
   activate a grasp. You still own the final candidate choice and must explicitly
   pass that exact candidate's short ids to `compile_grasp_seed`. If it abstains or has low
   confidence, inspect `grasp_selection_bundle.bundle_ref` or its preview images
   before choosing; do not silently fall back to rank 0.
   When selecting the SAM3 mask, include truthful
   `target_geometry_family` (`upright_can`, `upright_bottle`, `lying_bottle`, `boxed_item`,
   `bowl`, `apple`, `drawer_handle`, or `other`) only when visually clear. It is
   task evidence for strategy matching, not a calibration allowlist. Only a
   validated strategy may activate automatically from this hint. Candidate
   strategies are experimental evidence and require an explicit `strategy_id`;
   otherwise the compiler preserves the estimator pose.
8. Before grasp motion, call `compile_grasp_seed` with:
   - `grasp_result_id`: copy the exact estimator `result_id`.
   - `candidate_id`: copy the exact id of the candidate you selected from that
     result. The host resolves the complete candidate, its immutable source
     packet, camera extrinsics, frame id, and object-scene epoch. Never copy or
     reconstruct calibration matrices in the planner request.
   - `target_geometry_family`: optional truthful hint; omit when uncertain and
     never relabel an object to match a strategy.
   - `strategy_id`: optional session-local strategy backed by prior evidence.
   - `articulated_handle_options`: omit for every ordinary portable object. Only
     an explicitly verified `articulated_handle` or `drawer_handle` may set its
     nested `approach_mode` (`front`, `side`, or `top_down`).
   Calibration is not an object allowlist; no strategy match is required. Compiled
   poses are references. Do not use `camera_pose_to_world` for normalized grasps.
9. Plan one observed atomic edge at a time. Open only if not already open. A
   `hover_pose` is an ordinary collision-clearance waypoint, not an implicit
   phase and not a command to follow a fixed host sequence. Hover at least 0.15 m
   opposite world-frame `approach_world_xyz`, not fixed world `+Z`. Once there,
   use the evidence-triggered **Near-field Wrist Refinement** below when the
   wrist view can materially improve contact geometry. Preserve evidence lineage
   and move to contact only after visual evidence and deterministic checks support it.
   Compiled poses are anchors. Dual-view evidence may justify an xyz correction
   within host-derived 2 cm/call and 10 cm total residual caps: submit the adjusted
   pose to `ik_preview_check`, then pass its returned `ik_receipt_id` to `move_to`.
   Never copy the checked pose or rotation matrix into `move_to`; the host resolves
   them from the receipt. Preserve provenance and re-observe. These caps bound the offset from the compiled anchor;
   they are not a limit on how far the EEF may travel to reach it. An exact compiled
   hover/contact pose has zero residual and can be requested directly—do not split
   that approach into 2 cm increments. If a distinct far transit waypoint is useful,
   omit `compiled_grasp_id` and `waypoint_role`, keep it outside the contact safety
   envelope, observe there, and then use the compiled anchor. For normal compiled
   hover/contact reaches, call `ik_preview_check` with only the exact
   `compiled_grasp_id` and chosen `waypoint_role`; the host resolves the immutable
   pose. Then pass its `ik_receipt_id` to `move_to`, omit `num_steps`, and let the tool use its closed-loop
   default budget. `num_steps` is a maximum controller-iteration budget, not a
   distance or speed parameter; the environment's “3-5 steps for visible motion”
   hint applies to raw `step_env`, not to completing a `move_to`. Before committing
   to any endpoint, call `ik_preview_check` on that same world-frame
   xyz and the same orientation policy. A full-orientation preview authorizes
   only that full orientation; a `preserve_current_orientation=true` preview
   authorizes only the matching position-only move. For a compiled grasp,
   candidate orientation is functional contact geometry, not decoration. If
   the candidate's full 6-DoF contact pose is infeasible, a position-only
   `preserve_current_orientation` reach may be used as a non-contact observation
   or retreat waypoint, but it is not evidence that closing at the same xyz with
   an unrelated orientation will grasp the object. Select a materially different
   candidate, obtain a full wrist-view re-estimate, or visually justify and
   compile a new contact anchor instead of silently discarding orientation.
   Every robot or gripper
   mutation invalidates the prior receipt, so preview the next endpoint after
   the mutation rather than reusing it. A receipt with `steps_executed=0` and
   identical start/end pose is an acknowledged no-op, not a robot-motion epoch;
   it does not invalidate otherwise-current visual or IK evidence. `unreachable`
   means change the pose or candidate using its component residuals; `unknown`
   is solver uncertainty, not a safe approval. The specific
   `endpoint_collision_check_unavailable` result means IK succeeded but that
   optional backend is absent. Do not repeat the same IK with collision disabled:
   the earlier result already proved kinematics. Consume the deferred receipt only
   when controller capabilities explicitly own per-step trajectory/world checks,
   execute with `enable_collision_check=true`, and inspect the returned complete
   coverage receipt. Otherwise restore the collision backend or choose a safe
   non-execution recovery. Refreshing images or changing the pose does not repair
   that capability gap. Other `unknown` results require their stated recovery;
   `reachable` proves endpoint kinematics, but it is not always a positive local-
   controller recommendation. Inspect `execution_seed_quality` and the selected
   joint margin. If it reports `elevated` or `critical` risk and the grasp result
   still has untried candidates, normally compile and preview a materially different
   candidate before moving; compare the receipts rather than blindly following the
   visual advisor. Proceeding with an execution-fragile seed remains the Agent's
   choice when visual/task evidence justifies it, but state that tradeoff and never
   replay the same failed full-pose target. This comparison is read-only planning,
   not a host-selected grasp or a mandatory task phase.
   `reachable` covers endpoint kinematics only, while the motion receipt owns
   trajectory/world coverage. If a receipt says `reached_target=false`, do not pretend that the
   requested pose became the robot state. Refresh visual evidence and evaluate the
   next proposal from the reported actual EEF pose. A different directly checked
   motion remains allowed when its path from that actual pose is safe; there is no
   host-owned hover/contact phase transition.
   Treat contact as precision-critical rather than reusing a coarse clearance
   tolerance. Follow `execution_guidance.contact` from `compile_grasp_seed`
   (normally `tolerance=0.005` m and `ori_tolerance=0.10` rad), then compare the
   returned requested/actual EEF receipt and fresh wrist image before closing.
   `reached_target=true` uses the controller's declared metric (currently maximum
   per-axis error), so also read Euclidean `position_error_m`; neither kinematic
   number proves that the fingers straddle the object.
10. After contact, execute exactly binary `gripper_control position=0`;
   `0=closed`, `1=open`, fractions are invalid, and the command stays latched
   across every later motion. Keep three signals separate:
   - `commanded_state` is the last acknowledged binary latch command;
   - `measured_aperture.open_fraction` is continuous sensor feedback;
   - attachment remains unknown until post-lift co-motion/source-vacancy evidence.
   A held wide object may leave the measured aperture above 0.5; the compatibility
   `legacy_threshold_open=true` therefore does not mean that an open command was
   issued or that the grasp is empty. Its acknowledgement and observed aperture
   do not prove attachment; a static post-close image is not evidence.
   For a portable object, propose a visually justified 2-5 cm probe from the
   measured current EEF pose, normally with a lifting component. Do not reuse an
   earlier `grasp_clearance` or `grasp_precontact` pose: those are approach anchors
   and may be much longer or strongly lateral. Exact-IK-check this newly proposed
   probe pose, execute it with collision checking, observe both views, and decide
   attachment from co-motion/source-vacancy evidence before continuing. The probe
   direction remains Agent-owned; use current geometry rather than a fixed world-axis
   script. If the close receipt already reports an empty-close aperture or no active
   proxy, reopen and repair contact instead of probing. For an
   articulated handle, call `prepare_attachment_probe` with the current
   `compiled_grasp_id`, run every returned `ik_preview_request` in order, and pass
   only the resulting receipt id(s) to the indicated motion tool. Do not copy the
   frozen poses. Then call `assess_attachment_probe` with its `probe_id`. Treat
   PASS/FAIL/UNKNOWN as evidence; choose the recovery or continuation yourself.
   Once co-motion is established, treat the held object—not only the fingers—as
   part of the moving collision envelope. Keep enough vertical clearance for the
   full object extent. If `move_to` reports `collision_type=attached_object_world`,
   inspect its `attached_object`, `obstacle`, predicted pose, and recovery message;
   raise or reroute one IK-checked waypoint rather than repeating the rejected
   target. `steps_executed=0` means the controller never moved. When the
   conservative attached-object proxy already overlaps a neighbour, a vertical
   or lateral increment that strictly reduces that overlap is permitted as an
   egress; do not disable collision checking or move deeper into the overlap.
11. A simulator transport timeout means the action outcome is unknown, not failed.
    Observe the same handle and reconcile state before retry or a new action. A structured,
    candidate-linked rejection advances to the next candidate; calibration errors,
    unrelated failures, timeout, and interruption keep the current candidate active.
    After candidate-specific rejection, choose from evidence: another valid
    candidate, passive RGB-D, one checked clearance waypoint plus wrist
    re-estimation, or stop.
    Never invent a hover; safety, wrong-target, malformed-pose, stale-scene, and
    calibration rejections remain hard stops.

## Near-field Wrist Refinement

This is an optional visual correction opportunity, not a required task phase.
Use it near a collision-clearance/hover reference when the target is visible in
fresh wrist RGB-D and the initial scene-view pose has uncertain contact quality,
the target occupied too few scene-view pixels, or fresh wrist evidence shows the
gripper corridor is off the intended contact region. Skip it when current visual
and geometric evidence already supports the contact pose.

Consume `compile_grasp_seed.execution_guidance.near_field_refinement` explicitly.
In particular, a scene-view target mask below 2% image area or a single executable
candidate that the advisor could not compare is concrete evidence of uncertain
contact quality. After reaching a safe clearance view, prefer one of the wrist
refinements below before the first close when that guidance says `recommended=true`.
This remains Agent discretion, not a host phase: if fresh dual-view evidence makes
refinement unnecessary or unsafe, state that evidence and choose another action.

Choose the cheapest adequate refinement from the evidence:

- If the target is clipped, outside the wrist frame, or the current wrist
  orientation is not target-facing, do not treat a position-only clearance move
  as a valid observation viewpoint. Call `propose_wrist_viewpoints` with the exact
  current `compiled_grasp_id`, latest `source_packet_id`, and wrist
  `camera_frame_id`. It uses the live eye-in-hand transform to return several
  target-facing full EEF poses. Select one using workspace evidence, pass its
  proposal and candidate ids to `ik_preview_check`, and pass the returned
  `ik_receipt_id` to `move_to`.
  Position-only IK success proves endpoint position feasibility; it does not prove
  that the wrist camera will see the target.

- If the compiled approach direction, orientation, and contact depth remain
  credible and only lateral contact placement looks wrong, segment the target on
  the fresh wrist packet, explicitly select that wrist SAM3 detection, and call
  copy the exact `bundle_id` from
  `host_resolved_inputs.wrist_alignment` into `compute_wrist_alignment`. The host
  resolves its full-frame mask, aligned depth, intrinsics, extrinsics, measured
  EEF pose, compiled grasp, and scene epoch.
  Do not submit a desired pixel: the host projects the configured calibrated
  EEF-to-gripper-center point into that wrist image. The returned aligned hover,
  precontact, and contact poses are read-only translation references. This tool
  does not move, change grasp orientation, or independently repair axial contact
  depth. It emits those poses only when the selected mask is not clipped, the
  wrist packet matches both current object and robot epochs, the measured EEF is
  near the compiled clearance reference, the correction is not clamped, and every
  emitted reference fits the current residual budget. Otherwise the operationally
  successful result has `semantic_outcome=requires_better_view`, lists the failed
  geometric checks, and sets all executable pose fields to null. Use that evidence
  to choose a safe reposition/re-observation or a full wrist grasp re-estimate; do
  not execute a diagnostic correction vector as if it were an authorized pose.
- If the approach direction, orientation, surface, or contact depth is doubtful,
  do a full wrist-view re-estimation instead: call `sam3` with the fresh packet's
  exact `source_packet_id` and wrist `camera_frame_id`, select the intended mask,
  consume the resulting host `grasp_pose_estimate` bundle, inspect its candidates,
  then explicitly compile the chosen wrist candidate with that packet's matching
  camera extrinsics. The new compile is a new reference anchor; the host does not
  silently replace the earlier candidate.

Before executing either refined reference, call `ik_preview_check` and pass its
`ik_receipt_id` to `move_to`; keep endpoint
kinematics distinct from the motion controller's trajectory/world collision
receipt. A `compute_wrist_alignment` reference still uses the
original `compiled_grasp_id` and the ordinary residual budget. Re-observe after
each motion and verify that the gripper corridor/contact region actually improved.
Do not repeat refinement on an unchanged view merely to spend more turns. If the
target is occluded, the calibration chain fails, the mask touches an image boundary,
or the correction hits its clamp, retreat or gather a better view instead of
inventing pixels or accumulating blind residuals.

## Recovery Notes

- If the context reports `repeated_failed_motion_attractor`, the controller has
  repeatedly converged to nearly the same wrong EEF pose for the same target and
  orientation policy. Do not add iterations or replay that request. Use the
  reported actual EEF pose and change at least one material choice: grasp
  candidate, orientation policy, collision-clearance waypoint, or controller-
  compatible recovery path; then preview the new exact endpoint.
- After a visually confirmed attachment failure, preserve a compact failure note
  containing the attempted candidate geometry, actual contact endpoint, gripper
  openness, and observed failure mode. On the next estimate, compare candidates
  against that note and prefer a materially different approach/height/width when
  the scene geometry is otherwise unchanged. Do not repeatedly choose rank 0
  merely because candidate ids were regenerated; backend score is not recovery
  evidence. If no meaningfully different safe candidate exists, refresh from the
  wrist view or stop instead of replaying the same physical strategy.
  When the failed estimate's `selected_backend` is known, you may put a different
  estimator first through `grasp_pose_estimate.backend_preference`; inspect its
  `backend_policy` receipt rather than assuming the preferred backend ran.
- If exact-task `sam3` returns an empty mask and `retrieve_asset_reference` is
  executable, use reference localization before changing the prompt. Do not
  broaden an unusual asset name such as `alphabet soup` to `soup can`: that can
  segment another same-category instance. The point-prompt path may retry grasp
  estimation once in dense mode, then SAM3 once with bbox ROI attention.
- If the target disappears from the wrist view, do not repeat SAM3 or pointing
  tools on the unchanged wrist packet. Inspect the latest scene-primary
  `agentview` packet to recover global target visibility, or generate a new
  target-facing wrist viewpoint from the current compiled anchor. A clipped target
  mask has `grasp_input_bundle.status=requires_better_view` and must not be sent to
  targeted grasp estimation.
- A successful `propose_wrist_viewpoints` result is geometry evidence, not a
  one-packet lease. Pass its exact `proposal_id` as
  `viewpoint_proposal_id` plus the selected `candidate_id` to
  `ik_preview_check`; the host resolves the exact full pose. Do not copy only
  xyz or reconstruct the rotation matrix. Do not call the proposer again merely because the
  read-only tool turn produced a newer `source_packet_id`; reuse its
  `proposal_id` while the compiled grasp, object-scene epoch, robot-motion
  epoch, and camera mount are unchanged.
- If `sam3` returns multiple plausible masks, resolve the target identity before
  grasp estimation; confidence rank alone is not semantic identity.
- Do not treat transport errors, missing calibration, malformed parameters, or
  unrelated gripper failures as evidence against a candidate. Candidate changes
  are Agent decisions grounded in observed outcomes, not automatic host fallback.
- Never move from stale perception. Observe after every world-mutating tool call.
  Keep `scene_epoch` with artifact provenance; do not reuse old masks, depth, or poses.
- During transport, first raise the entire held object above intervening clutter,
  then translate with bounded horizontal waypoints. Do not combine a long lateral
  carry with descent toward a receptacle: the gripper can remain perfectly closed
  while a rim mechanically strips the object from the fingers.

For explicit robot/environment calibration or parameter discovery, use the
`embodiment_explore` skill outside the benchmark episode. This skill consumes
the resulting validated profile; it does not silently recalibrate one.
