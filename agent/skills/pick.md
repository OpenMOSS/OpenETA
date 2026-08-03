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
  - scene_detector
  - sam3
  - select_sam3_detection
  - anygrasp
  - camera_pose_to_world
  - move_to
  - gripper_control
---
# Pick

Use this skill as text guidance only. Do not treat `pick` as an executable
macro. After each tool result, inspect the returned observation or tool output
before choosing the next tool call.

## Recommended Tool Sequence

1. Call `observe` to get the complete current scene observation.
2. Extract the target phrase from the user task, then normalize it to a visual
   object phrase for perception tools. Prefer concise English prompts for
   `sam3`, even when the user task is in another language. For example:
   - "please pick up milk box" -> `milk box`
   - "把桌上的罐子抓起来" -> `can`
   - "拿起牛奶盒" -> `milk box`
   - "抓取方块" -> `cube`
3. Call `sam3` to segment the target object or requested scene region. Use the
   normalized visual phrase as the `prompt`, for example `milk box` or `can`.
   Do not pass a non-English user phrase directly to `sam3` if a clear English
   object name is available.
4. Stop after `sam3` and inspect its result before calling `anygrasp`. The
   runtime does not automatically pass outputs between batched tool calls. For
   every non-empty result, including a single detection, the runtime creates a
   `selection_obligation` and attaches the original RGB plus a candidate contact
   sheet to the next VLM planner request. Inspect those images and call
   `select_sam3_detection` with the exact `sam3_result_id` and `detection_id`.
   Treat score as a ranking hint rather than proof of target identity. Rerun
   `sam3`, observe from another camera, or ask a human when uncertain.
5. Call `anygrasp` with concrete parameters:
   - `rgb`: local RGB image path from the same `observe` / render
     `camera_packet`.
   - `depth`: local depth image path from the same `camera_packet`.
   - `intrinsics`: copy `camera_packet.anygrasp_intrinsics` from the same
     camera as `rgb` and `depth`. It must include `fx`, `fy`, `cx`, `cy`, and
     `scale`; do not invent fallback values.
   - `target_mask`: the exact mask path returned by
     `select_sam3_detection`. Do not default to `detections[0]`, even when it is
     the only result; a detector score is not semantic confirmation. The runtime
     blocks targeted AnyGrasp until explicit VLM selection succeeds. Do not use
     invented aliases such as `latest_sam3_mask` or `mask_ref`.
6. Read the `anygrasp` grasp candidate list. Candidate poses are camera-frame
   grasp candidates sorted by score descending. The runtime records
   `grasp_candidate_policy`: rank 0 is the initial `active_candidate`; lower
   ranked candidates are fallbacks and must not be selected early.
7. Before grasp motion, call `camera_pose_to_world` with:
   - `camera_pose`: the complete `grasp_candidate_policy.active_candidate`,
     preserving its `id`, rank, score, pose, and dimensions.
   - `camera_extrinsics`: the matching `camera_packet.extrinsics` from the same
     observe/render camera used for RGB and depth.
   - `camera_frame_id`: the matching camera frame id, such as `agentview`.
8. Pass the complete `camera_pose_to_world` result
   `details.outputs.world_pose` directly as the world-frame `target_pose` for one
   atomic `move_to` call. Do not invent Cartesian offsets or tune the transformed
   pose in the planner. Only an explicit candidate-specific safety or failure-check
   rejection advances `grasp_candidate_policy.active_candidate`; ordinary motion
   failures, transport errors, interruption, and calibration failures keep the
   current candidate active for diagnosis. Transform a newly activated candidate
   on the next turn; do not reuse rejected poses. If the policy is exhausted,
   observe again and rerun AnyGrasp. After `move_to` succeeds, call
   `gripper_control` with `position=0` to close the gripper. Do not use Python
   filesystem helpers such as `os` or
   `glob` to discover AnyGrasp parameters; use the tool result or working
   memory artifact already in planner context.
9. After closing the gripper, observe and require positive grasp evidence
   before lifting. A closed gripper alone is not success. Verify that the
   requested object moved with the end effector, or use reward/termination or
   visual evidence. If the target stayed on the table, reopen, resegment or
   choose another candidate, and replan.

## Optional Scene Information

Do not call `scene_detector` in the default pick flow while its handler is a
dummy placeholder. If a real scene detector or simulator privileged object-list
tool is available in the runtime tool catalog, it may be used as optional
context before `sam3`; its output must not override the user-requested target
phrase unless it clearly resolves an ambiguity.

## Recovery Notes

- If `sam3` returns an empty or low-confidence mask, refine the text prompt or
  choose another camera frame.
- If `sam3` returns multiple plausible masks, resolve the target identity before
  AnyGrasp; confidence rank alone is not semantic identity.
- If `anygrasp` returns no feasible pose, retry with another segmented region or
  ask for a better view before issuing motion commands.
- Do not advance the AnyGrasp queue for transport errors, missing calibration,
  malformed parameters, or unrelated gripper failures. Automatic fallback is
  limited to rejection explicitly linked to the active grasp pose.
- Never continue through motion from stale perception. Observe again after every
  world-mutating tool call.
