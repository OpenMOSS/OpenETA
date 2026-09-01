# Motion-C five short canaries — 2026-08-24

Status: complete. Three control/permission boundaries passed, the read-only
freshness boundary passed semantically, and the wrist-refined grasp did not
complete attachment proof.

## Purpose

These canaries isolate the five boundaries recommended after the three-task
Motion-C run. They are intentionally shorter than another full pick-and-place
suite and distinguish controller, gate, evidence-lifecycle, and tool-handoff
failures.

The local simulator used the `mink_joint_velocity` controller with motion
condition C. Provider data transfer was authorized. AnyGrasp preflight matched
the 0.08 m Panda gripper calibration, and Memory Bank was healthy in the
`libero` namespace with 40 objects.

## Summary

| # | Boundary | Result | Main evidence |
| --- | --- | --- | --- |
| 1 | Agent-selected 2–4 point route using `follow_eef_trajectory` | Pass | Agent chose three points; 3/3 JIT previews and all waypoints reached |
| 2 | Preview-passed short segment executes consistently under Mink | Pass | 5/5 fresh-reset runs reached; mean endpoint error 1.69 mm |
| 3 | Clearance collision checks and target-contact authorization are separated | Pass | Unauthorized target contact stopped before actuation; the same contact was allowed only with target-scoped authorization |
| 4 | Wrist-refined grasp reaches close + lift + attachment proof | Fail | Wrist candidate compiled and contact attempted, but the probe handoff did not yield an executable lift chain |
| 5 | Read-only refresh preserves evidence and sandbox artifact inspection works | Semantic pass | Scene/motion epochs remained zero, selected evidence resolved after refresh, and sandbox output entered context; generic eval misclassified terminal `talk` |

## 1. Agent-selected Motion-C route

Run `motion-c-self-route-short-20260824-r1` completed successfully in 179.33 s
with six provider requests. The task specified the route constraint but did not
prescribe waypoint coordinates. The Agent selected:

1. `[0.00, -0.16, 0.42]`
2. `[0.05, -0.24, 0.40]`
3. `[0.10, -0.32, 0.40]`

It issued three `ik_preview_check` calls and one
`follow_eef_trajectory`. The controller reported:

- `waypoints_requested=3`, `waypoints_completed=3`;
- `sequential_route_preview.policy=just_in_time_from_actual_segment_end`;
- 3/3 segment receipts feasible with each preview seeded from the actual end
  of the preceding segment;
- final actual xyz `[0.099822, -0.319955, 0.397224]`;
- per-waypoint maximum-axis residuals 2.44, 2.49, and 2.78 mm;
- `reached_target=true`, `stop_reason=target_reached`.

This establishes that the distinctive condition-C path is usable and visible
to the Agent. It does not measure how often the Agent will choose that path in
an unconstrained full task.

## 2. IK-preview to Mink repeatability

The existing short-up controller canary was run on LIBERO Object task 2 with
fresh environments for seeds 2–6. All five IK previews were reachable and all
five executions reached the requested target without a reported collision.
Each execution used five controller steps and completed three stable-arrival
steps.

Endpoint position errors were 1.42, 1.87, 1.87, 1.87, and 1.42 mm: mean
1.69 mm, range 1.42–1.87 mm. These runs support preview/execution consistency
for short local translations. They do not overturn the earlier evidence that
longer motions or tightly constrained full-pose contacts can stall.

The task layout and kinematics were effectively identical across these reset
seeds, so this is a fresh-reset repeatability check rather than five diverse
geometries.

## 3. Clearance and intentional contact

The deterministic contact canary on salad dressing passed every assertion:

- clearance motion reached with full robot/world trajectory and endpoint
  checks, seven world objects, and no contact authorization;
- an unauthorized contact motion detected approximately -3.12 mm signed
  distance between the gripper and target and stopped before actuation;
- the same contact reached only after authorization was scoped to the gripper
  subtree and the selected salad-dressing target;
- closing remained a tentative target-bound proxy rather than attachment proof;
- the subsequent lift reached, with 113.62 mm object displacement and 42.34 mm
  object-to-EEF distance;
- carried-object predicted and actual geometry checks were both active.

This verifies the intended semantic split: collision checking remains strict
for clearance and non-target geometry, while expected target contact is
authorized only for the contact operation. This direct canary used a
host-private contact authorization fixture to isolate controller behavior; it
did not replay the full Agent grasp compilation chain.

## 4. Wrist-refined grasp

Run `wrist-grasp-short-20260824-r1` ran for 450.82 s with 26 provider requests
and did not complete the grasp.

The Agent behaved conservatively:

1. It compiled an initial scene-view grasp and moved to a proposed wrist
   viewpoint.
2. `compute_wrist_alignment` returned
   `wrist_alignment_outside_operating_region`; the Agent did not execute the
   non-authorized alignment.
3. It re-estimated from the wrist view. The advisor selected candidate
   `gpe-0f27bc1955264d7c-002` with confidence 0.87, preferring a broad upper-body
   grasp away from the carton edges.
4. It compiled that wrist candidate and attempted contact. Position converged
   to 0.29 mm, but orientation stalled at 0.169 rad versus the 0.10 rad motion
   tolerance. No collision was reported.
5. Gripper close was permitted by the reviewed 0.30 rad contact gate and
   returned aperture 0.466, but `attachment_proven=false`.
6. `prepare_attachment_probe` froze a 5 cm vertical lift, but the Agent never
   executed it and therefore obtained no co-motion or source-vacancy proof.

The stopping point exposes a tool-handoff defect. The tool response says to run
the returned ordered IK requests, but:

- `artifact_refs` is empty;
- the exact nested `ik_preview_requests[*].parameters.target_pose` is omitted
  from the bounded planner projection;
- artifact search by `probe_id` finds no durable file;
- the instruction correctly tells the Agent not to copy frozen poses.

The Agent could therefore see the probe id and frozen-path summary but could not
safely construct the next authorized IK request.

Post-canary implementation candidate: `ik_preview_check` now accepts
`probe_id + waypoint_index` and the host resolves the exact frozen pose. The
bounded conversation/latest-result projections preserve this short request and
omit the large frozen path. Unknown ids, invalid indices, completed probes,
epoch mismatch, or path-digest mismatch fail closed with actionable feedback.
Three-person review approved this additive ToolContract branch on 2026-08-24.
Shared RFC C.3 and the catalog authority receipts were updated before the live
rerun below.

The probe implementation also labels every prepared probe
`interaction_family=articulated_handle`, including this rigid milk-carton lift.
The generated linear geometry was still appropriate, but the semantic label is
misleading.

## 5. Read-only freshness and artifact access

Run `freshness-artifact-short-20260824-r1` completed the requested evidence
inspection in 125.19 s with eight provider requests:

1. `sam3` and `select_sam3_detection` created selected target evidence.
2. A read-only `observe` refreshed the packet.
3. `grasp_pose_estimate` successfully resolved the original selected bundle
   once after refresh; no second SAM3 call was needed.
4. `python_exec` read the returned artifact in the sandbox and its JSON result
   was returned to the planner context.

Both `robot_motion_epoch` and `object_scene_epoch` remained zero. The selected
evidence `sam3:20260824T104402802582Z-e9150684:detection_000` remained usable,
confirming that a packet refresh alone no longer invalidates scene evidence.
The first sandbox program used Python's `iter` builtin, which is not exposed;
the error was explicit and the Agent recovered with a normal `for` loop.

The universal evaluator recorded this run as `agent_failure / episode_failed`
only because the Agent ended with `talk`. The task was deliberately report-only
and required no physical objective, so this is an evaluator outcome-policy gap,
not a freshness failure.

## Conclusions and next review boundary

The short experiments narrow the remaining problem considerably:

- Motion-C's multi-waypoint JIT execution works when invoked.
- Short local IK-preview and Mink execution are consistent to millimetre scale.
- Clearance and intentional target contact have the intended collision-policy
  separation.
- Read-only packet refresh no longer causes the prior SAM3 freshness loop, and
  sandbox artifact output reaches the Agent.
- Wrist re-estimation and advisor selection are correctly conservative, but a
  stable-looking candidate still did not prove attachment.
- The immediate wrist-chain blocker was the incomplete
  `prepare_attachment_probe` handoff; a host-resolved short-ID candidate is now
  implemented. Full-pose orientation convergence and physical contact quality
  remain the next observed risks after live coverage.

Original review items before the follow-up rerun were:

1. approve or reject `probe_id + waypoint_index` as the host-resolved IK
   handoff for frozen attachment probes (approved on 2026-08-24);
2. generalize the attachment-probe interaction-family naming beyond articulated
   handles;
3. let report-only evaluation plans declare `talk` as a valid terminal outcome;
4. decide whether common safe builtins such as `iter` belong in the sandbox.

The original canaries did not change the shared RFC. The reviewed follow-up
subsequently synchronized the short-ID contract to RFC revision 2145.

## 6. Reviewed short-ID live rerun

Run `wrist-grasp-short-20260824-r2` reused the same grasp-only task after the
three-person review. Preflight passed for GPT-5.6 Luna, the local 17-tool LIBERO
simulator, AnyGrasp's 0.08 m deployment geometry, and the required LIBERO Memory
Bank.

The run was deliberately interrupted after 30 planner requests and 314.27 s of
provider time because a repeatable wrist-refinement cycle appeared before close:

1. Initial SAM3 selection, AnyGrasp, advisor-selected carton-body candidate,
   compile, clearance IK, and Mink move all succeeded.
2. Clearance reached within 1.3 mm and advanced `robot_motion_epoch` once.
3. Wrist alignment returned `requires_better_view`; the Agent proposed a new
   viewpoint and obtained feasible IK receipt `400cfcc7161b066eed06`.
4. The receipt remained visible in `decision_state.last_action_effect` and
   `world_evidence.ik_preview_receipts`, but no compact unresolved execution
   obligation was materialized.
5. Instead of calling `move_to` with that receipt, the Agent repeated point
   SAM3, target selection, and `compute_wrist_alignment`. Three alignment calls
   returned `requires_better_view` while the sampled wrist frames were
   pixel-identical and both semantic epochs stayed unchanged.

This run never reached `gripper_control`, `prepare_attachment_probe`, or
`assess_attachment_probe`, so it neither passes nor falsifies the reviewed
short-ID resolver in live Agent behavior. It exposes an earlier harness gap:
"feasible preview but not yet executed" is durable evidence but not an indexed
obligation, and the current no-progress detector does not recognize a
multi-tool semantic cycle with unchanged robot/object epochs.

## Artifacts

- Route run: `.openeta_eval/runs/motion-c-self-route-short-20260824-r1/`
- Wrist run: `.openeta_eval/runs/wrist-grasp-short-20260824-r1/`
- Reviewed wrist rerun: `.openeta_eval/runs/wrist-grasp-short-20260824-r2/`
- Pending-index/semantic-cycle rerun:
  `.openeta_eval/runs/wrist-grasp-short-20260824-r3/`
- Freshness run: `.openeta_eval/runs/freshness-artifact-short-20260824-r1/`
- IK/Mink receipts: `tmp/five-short-canaries/02-ik-mink-seed2.json` through
  `02-ik-mink-seed6.json`
- Clearance/contact receipt: `tmp/five-short-canaries/03-clearance-contact.json`
- Evaluation plans and manifests:
  `evaluations/motion_c_self_route_short.json`,
  `evaluations/wrist_grasp_short.json`, and
  `evaluations/freshness_artifact_short.json`

## 7. Pending execution index and semantic-cycle live rerun

Run `wrist-grasp-short-20260824-r3` exercised the two harness fixes prompted by
the r2 loop. Preflight again passed for GPT-5.6 Luna, the local 17-tool LIBERO
simulator, AnyGrasp's 0.08 m geometry, and the required Memory Bank. The run
ended at the 48-tool-call limit after 1027.22 s; all 48 tool calls executed
without a tool failure.

The pending execution index changed live behavior immediately. After
`ik_preview_check` returned feasible viewpoint receipt
`24bdb4a477fe77bcf6f3`, the next planner request explicitly cited the pending
receipt and dispatched `move_to`. The viewpoint reached with 0.7 mm position
error. Later, receipt `9066ec86ec53ae8d2884` was initially ignored while the
Agent repeated wrist perception and alignment. On the third
`requires_better_view` result, the new detector emitted
`semantic_state_cycle_without_world_change` with:

- unchanged visual signature `7a87651502432f8e3c92db38`;
- `object_scene_epoch=0` and `robot_motion_epoch=1`;
- three equivalent negative outcomes across a multi-tool read-only cycle;
- the still-pending execution receipt id.

The next action consumed that exact receipt and advanced the robot state. This
is advisory evidence rather than a stage machine: a later feasible viewpoint
receipt was intentionally superseded when the Agent judged the existing wrist
view sufficient and instead previewed the compiled contact endpoint.

The reviewed attachment-probe short reference also received live coverage.
The Agent called `prepare_attachment_probe`, then successfully used only
`probe_id=probe:70283b...` and `waypoint_index=0` for IK preview, and passed the
returned receipt to `move_to`. No frozen pose array was copied by the model.

The grasp itself was not stable. Contact reached within 1.14 mm, close returned
a tentative proxy with measured open fraction 0.4886, and an Agent-authored
3 cm lift reduced the aperture to 0.090. During the dedicated 5 cm attachment
probe the aperture collapsed further to 0.025 and the carried-object proxy was
retired with `aperture_collapsed_to_empty_close`. Direct inspection of
agentview frames 0046 through 0049 shows the gripper moving upward while the
milk carton remains approximately at its prior location rather than exhibiting
equal co-motion. The episode exhausted its turn budget before
`assess_attachment_probe`, but the receipt and visual evidence already indicate
that attachment would not have been proven.

Conclusion: both harness fixes and the reviewed short-id handoff work in a live
provider rollout and remove the r2 decision-loop blocker. The next task-level
bottleneck is again grasp/contact quality (candidate geometry, contact depth,
or simulated jaw-object interaction), not receipt visibility or probe argument
transport. Increasing the turn budget alone would allow the formal assessment
call but would not make this particular grasp successful.
