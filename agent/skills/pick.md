---
name: pick
description: Guidance for acquiring a target object with atomic tools.
version: v1
editable: true
context_char_limit: 12000
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
  - reject_sam3_detections
  - compile_grasp_seed
  - compute_wrist_alignment
  - camera_pose_to_world
  - ik_preview_check
  - obstacle_avoidance
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
3. Call `sam3` on the exact local RGB path from `current_camera_artifacts` with
   the normalized `prompt`, for example `milk box` or `can`.
   Do not pass a non-English user phrase directly to `sam3` if a clear English
   object name is available.
   For an unusual asset that text segmentation misses, use
   `retrieve_asset_reference` with the exact task asset name and current scene
   image; do not append category guesses. Copy its original-image positive
   points unchanged into SAM3. Use the single bbox fallback only after the point
   mask and one dense grasp attempt produce no candidates.
4. Stop after `sam3`; dependent batched calls do not pass outputs. For every
   non-empty result, inspect the attached original/contact sheet and call
   `select_sam3_detection` with the exact `sam3_result_id` and `detection_id`.
   Score ranks candidates but does not prove identity. Gather another view when uncertain.
5. For poor real-robot depth, optionally estimate a prior and call
   `enhance_depth` on the same RGB-D packet. Use candidate depth only when its
   quality permits; collision checks must use sensor safety depth/cloud, never
   mono-filled geometry.
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
   backend directly. For enhanced depth, require candidate-linked sensor-only
   `obstacle_avoidance clear=true`; mono-filled depth is not collision evidence.
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
   pass that exact candidate to `compile_grasp_seed`. If it abstains or has low
   confidence, inspect `grasp_selection_bundle.bundle_ref` or its preview images
   before choosing; do not silently fall back to rank 0.
   When selecting the SAM3 mask, include truthful
   `target_geometry_family` (`upright_can`, `upright_bottle`, `boxed_item`,
   `bowl`, `apple`, `drawer_handle`, or `other`) only when visually clear. It is
   task evidence for strategy matching, not a calibration allowlist. Only a
   validated strategy may activate automatically from this hint. Candidate
   strategies are experimental evidence and require an explicit `strategy_id`;
   otherwise the compiler preserves the estimator pose.
8. Before grasp motion, call `compile_grasp_seed` with:
   - `camera_pose`: the complete candidate that you selected from the current
     grasp ToolResult or its `complete_outputs_artifact`, preserving its id,
     camera-frame rotation/translation, width, and dimensions. Use `python_exec`
     to inspect the complete candidate file when the inline preview is truncated.
   - `camera_extrinsics`: the matching `camera_packet.extrinsics` from the same
     observe/render camera used for RGB and depth.
   - `camera_frame_id`: the matching camera frame id, such as `agentview`.
   - `scene_epoch`: copy the current host object-scene epoch exactly. Robot-only
     motion advances `robot_motion_epoch` and does not by itself stale this pose.
   - `target_geometry_family`: optional truthful hint; omit when uncertain and
     never relabel an object to match a strategy.
   - `strategy_id`: optional session-local strategy backed by prior evidence.
   Calibration is not an object allowlist; no strategy match is required. Compiled
   poses are references. Do not use `camera_pose_to_world` for normalized grasps.
9. Plan one observed atomic edge at a time. Open only if not already open. A
   `hover_pose` is an ordinary collision-clearance waypoint, not an implicit
   phase and not a command to follow a fixed host sequence. Hover at least 0.15 m
   opposite world-frame `approach_world_xyz`, not fixed world `+Z`. Once there,
   prefer a full wrist-view grasp refresh when the target is visible: acquire
   fresh wrist RGB-D, run wrist-image SAM3, resolve its selection, call targeted
   `grasp_pose_estimate` on that same packet, and compile the refined candidate.
   `compute_wrist_alignment` remains an optional bounded correction; it is not a
   replacement for full grasp re-estimation. Preserve evidence lineage and move
   to contact only after visual evidence and deterministic checks support it.
   Compiled poses are anchors. Dual-view evidence may justify `move_to` xyz
   correction within host-derived 2 cm/call and 10 cm total residual caps. Preserve
   provenance and re-observe. These caps bound the offset from the compiled anchor;
   they are not a limit on how far the EEF may travel to reach it. An exact compiled
   hover/contact pose has zero residual and can be requested directly—do not split
   that approach into 2 cm increments. If a distinct far transit waypoint is useful,
   omit `compiled_grasp_id` and `waypoint_role`, keep it outside the contact safety
   envelope, observe there, and then use the compiled anchor. For normal compiled
   hover/contact reaches, omit `num_steps` and let `move_to` use its closed-loop
   default budget. `num_steps` is a maximum controller-iteration budget, not a
   distance or speed parameter; the environment's “3-5 steps for visible motion”
   hint applies to raw `step_env`, not to completing a `move_to`. Before committing
   to a compiled hover/contact endpoint, call `ik_preview_check` on that same
   world-frame pose. `unreachable` means change the pose or candidate using its
   component residuals; `unknown` is solver uncertainty, not a safe approval;
   `reachable` covers endpoint kinematics only, so keep path/collision evidence
   separate. If a receipt says
   `reached_target=false`, do not advance from hover to contact or from contact to
   close. Use the reported actual EEF pose plus fresh images to retry or replan.
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
   For a portable object, propose a small visually justified lift, observe both
   views, and decide attachment from co-motion evidence before continuing. For an
   articulated handle, call `prepare_attachment_probe` with the current
   `compiled_grasp_id`, execute the returned `frozen_action` exactly, then call
   `assess_attachment_probe` with its `probe_id`. Treat PASS/FAIL/UNKNOWN as
   evidence; choose the recovery or continuation yourself.
   Once co-motion is established, treat the held object—not only the fingers—as
   part of the moving collision envelope. Keep enough vertical clearance for the
   full object extent. If `move_to` reports `collision_type=attached_object_world`,
   inspect its `attached_object`, `obstacle`, predicted pose, and recovery message;
   raise or reroute one waypoint rather than repeating the rejected target.
11. A simulator transport timeout means the action outcome is unknown, not failed.
    Observe the same handle and reconcile state before retry or a new action. A structured,
    candidate-linked rejection advances to the next candidate; calibration errors,
    unrelated failures, timeout, and interruption keep the current candidate active.
    After candidate-specific rejection, choose from evidence: another valid
    candidate, passive RGB-D, one checked clearance waypoint plus wrist
    re-estimation, or stop.
    Never invent a hover; safety, wrong-target, malformed-pose, stale-scene, and
    calibration rejections remain hard stops.

## Recovery Notes

- After a visually confirmed attachment failure, preserve a compact failure note
  containing the attempted candidate geometry, actual contact endpoint, gripper
  openness, and observed failure mode. On the next estimate, compare candidates
  against that note and prefer a materially different approach/height/width when
  the scene geometry is otherwise unchanged. Do not repeatedly choose rank 0
  merely because candidate ids were regenerated; backend score is not recovery
  evidence. If no meaningfully different safe candidate exists, refresh from the
  wrist view or stop instead of replaying the same physical strategy.
- If exact-task `sam3` returns an empty mask and `retrieve_asset_reference` is
  executable, use reference localization before changing the prompt. Do not
  broaden an unusual asset name such as `alphabet soup` to `soup can`: that can
  segment another same-category instance. The point-prompt path may retry grasp
  estimation once in dense mode, then SAM3 once with bbox ROI attention.
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
