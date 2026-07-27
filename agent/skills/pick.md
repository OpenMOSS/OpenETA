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
  - reject_sam3_detections
  - activate_final_grasp_candidate
  - select_grasp_candidate
  - compile_grasp_seed
  - compute_wrist_alignment
  - camera_pose_to_world
  - ik_preview_check
  - obstacle_avoidance
  - move_to
  - gripper_control
---
# Pick

Use as text guidance only, not an executable macro. Inspect each result.

## Recommended Tool Sequence

1. Call `observe` to get the complete current scene observation.
2. Extract the target phrase from the user task and normalize it to a concise
   English visual object phrase for `sam3`. For example:
   - "please pick up milk box" -> `milk box`
   - "把桌上的罐子抓起来" -> `can`
   - "拿起牛奶盒" -> `milk box`
   - "抓取方块" -> `cube`
3. Call `sam3` on the exact local RGB path from `current_camera_artifacts` with
   the normalized `prompt`, for example `milk box` or `can`.
   Do not pass a non-English user phrase directly to `sam3` if a clear English object name is available.
   If direct text segmentation is empty or clearly fails to identify an unusual
   simulator asset, and `retrieve_asset_reference` is executable, call it with
   the active simulator `environment`, the exact target asset name from the task
   as `target_object`, and the exact local original RGB `scene_image`. Object
   memory resolves this task phrase to a canonical asset. Do not add
   visual category words such as `can`, `bottle`, or `box` to object-memory
   lookup. Its isolated localizer returns original-image `positive_points` and an
   audit image. Call `sam3` on that exact `scene_image`; copy points unchanged.
   Use `roi_bbox_xyxy` only for the runtime's single bbox fallback after the
   selected point mask and one dense grasp attempt produce no candidates.
4. Stop after `sam3` and inspect its result before calling
   `grasp_pose_estimate`. The
   runtime does not pass outputs between dependent batched calls. For every
   non-empty result, including a single detection, the runtime creates a
   `selection_obligation` and attaches the original RGB plus a candidate contact
   sheet to the next VLM planner request. Inspect those images and call
   `select_sam3_detection` with the exact `sam3_result_id` and `detection_id`.
   Score ranks candidates but does not prove identity. Gather another view when uncertain.
5. For real-robot RGB-D or poor depth, call `estimate_depth_prior` when
   executable, then `enhance_depth` with the same camera's exact `rgb`, `depth`,
   and `intrinsics`. Pass prior paths and confidence semantics unchanged, plus
   available registration, timestamps, scene epoch, and calibration hash. If
   no prior tool exists, sensor-only enhancement is diagnostic. Use
   `candidate_depth_png` for grasp generation only when its quality flag allows
   it. Collision evidence must use `safety_depth_png` or the safety point cloud,
   never mono-filled geometry.
6. Call `grasp_pose_estimate` with the exact
   `targeted_grasp_obligation.required_parameters`. The host joins:
   - `rgb`/`depth`: exact current artifact paths for the same camera.
   - `intrinsics`: same camera intrinsics with `fx`, `fy`, `cx`, `cy`, and `scale`.
   - `object_mask`: selected artifact with exact `mask_ref` and `source_image`;
     never pass a bare path or default to `detections[0]`.
   - `camera_frame_id` and `scene_epoch`: exact host provenance.
   - `extrinsics`: required; pass the same `camera_packet.extrinsics` used for
     `rgb`/`depth`. This makes every candidate carry a `world_pose` preview
     (rotation, translation, and `approach_world_xyz` in the world frame) plus a
     `world_downward_alignment` scalar (+1 straight down, 0 horizontal), which
     you need to compare candidate orientations in step 7. Without extrinsics the
     candidates have no world-frame pose and you cannot judge approach direction,
     so do not skip it.
   Backend-specific options and fallback are host-owned. Do not call AnyGrasp,
   Contact-GraspNet, or GraspGenX directly.
   If candidate depth was used, follow the host-generated
   `grasp_sensor_safety_obligation`: `obstacle_avoidance` must return
   `clear=true` for the exact candidate, scene epoch, report, and sensor-only
   safety artifacts before `compile_grasp_seed` becomes available.
7. Read the normalized grasp candidate list. It is sorted by backend-local
   score, but score does not measure approach quality and is not comparable
   across backends: the highest-scoring candidate is often a poor side grasp
   while a lower-ranked one is cleanly top-down. So do not just take rank 0.
   Review every candidate's `world_pose` (world frame) — judge approach
   orientation from `world_pose.approach_world_xyz` and `world_downward_alignment`
   (+1 straight down, 0 horizontal), never from the raw camera-frame
   `rotation_matrix`. `world_pose` is the grasp frame in world coordinates, a
   reference for orientation reasoning; `compile_grasp_seed` still layers the
   fixed grasp-to-EEF calibration on top to produce the executed pose. The
   camera-frame fields stay for the compiler and must not be edited by hand.
   Pick the candidate whose world-frame approach best fits the target and the
   robot's reachable envelope — for most tabletop targets that means the highest
   `world_downward_alignment` among the physically-valid candidates, not the
   highest score. If that candidate is not already the active rank-0 one, call
   `select_grasp_candidate` with its `candidate_id` and a concrete `reason`
   (cite the alignment values you compared) to make it active. Selection is
   limited to the physically-valid candidate list; only fall back to rank 0 when
   no candidate has a meaningfully better world-frame approach.
   When selecting the SAM3 mask, include truthful
   `target_geometry_family` (`upright_can`, `upright_bottle`, `boxed_item`,
   `bowl`, `apple`, `drawer_handle`, or `other`) only when visually clear. It is
   task evidence for strategy matching, not a calibration allowlist.
8. Before grasp motion, call `compile_grasp_seed` with:
   - `camera_pose`: the complete `grasp_candidate_policy.active_candidate`,
     preserving its id, camera-frame rotation/translation, width, and dimensions.
   - `camera_extrinsics`: the matching `camera_packet.extrinsics` from the same
     observe/render camera used for RGB and depth.
   - `camera_frame_id`: the matching camera frame id, such as `agentview`.
   - `scene_epoch`: copy the current host `scene_epoch` exactly.
   - `target_geometry_family`: optional truthful hint; omit when uncertain and
     never relabel an object to match a strategy.
   - `strategy_id`: optional session-local strategy backed by prior evidence.
   Calibration is not an object allowlist; no strategy match is required. Compiled
   poses are references. Do not use `camera_pose_to_world` for normalized grasps.
9. Follow `grasp_execution` one observed atomic edge at a time. The host opens only
   when the latched command is not already open. Hover is at least 0.25 m opposite
   world-frame `approach_world_xyz`, not unconditionally world `+Z`. At hover, use
   fresh matching wrist RGB-D to call `compute_wrist_alignment`; bounded feedback
   corrections must preserve world frame and candidate provenance. Accept only at contact.
10. After contact, execute binary `gripper_control position=0`; `0=closed`, `1=open`,
   and the command stays latched across motion. Its acknowledgement and observed
   openness do not prove attachment; a static post-close image is not evidence.
   Keep closed for the exact host lift probe.
   PASS requires target co-motion and source vacancy, then permits full lift. FAIL
   may reopen/reject only when the completed probe shows the target stayed at source.
   UNKNOWN requires observation and may not reopen or switch candidates.
11. A simulator transport timeout means the action outcome is unknown, not failed.
    Observe the same handle and reconcile state before retry or a new action. A structured,
    candidate-linked rejection advances to the next candidate; calibration errors,
    unrelated failures, timeout, and interruption keep the current candidate active.
    For host-classified `perception_refinable` or `uncertain_review` exhaustion,
    follow `grasp_estimation_fallback_obligation` exactly: passive RGB-D views,
    one IK/collision-checked hover plus fresh wrist re-estimation, then another
    backend. Never invent a hover. Safety, IK, collision, wrong-target, malformed
    pose, stale scene, and invalid calibration rejection remain hard stops.

## Recovery Notes

- If exact-task `sam3` returns an empty mask and `retrieve_asset_reference` is
  executable, use reference localization before changing the prompt. Do not
  broaden an unusual asset name such as `alphabet soup` to `soup can`: that can
  segment another same-category instance. The point-prompt path may retry grasp
  estimation once in dense mode, then SAM3 once with bbox ROI attention.
- If `sam3` returns multiple plausible masks, resolve the target identity before
  grasp estimation; confidence rank alone is not semantic identity.
- After all views/backends, the host activates one highest-score refinable pose
  for a final attempt. Pre-hover wrist images do not count as the wrist retry.
- Do not advance the grasp queue for transport errors, missing calibration,
  malformed parameters, or unrelated gripper failures. Automatic fallback is
  limited to rejection explicitly linked to the active grasp pose.
- Never move from stale perception. Observe after every world-mutating tool call.
  Keep `scene_epoch` with artifact provenance; do not reuse old masks, depth, or poses.

For explicit robot/environment calibration or parameter discovery, use the
`embodiment_explore` skill outside the benchmark episode. This skill consumes
the resulting validated profile; it does not silently recalibrate one.
