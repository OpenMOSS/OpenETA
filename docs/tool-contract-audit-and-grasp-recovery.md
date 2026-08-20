# Tool-contract audit and grasp recovery

Status: implemented locally; shared RFC schema changes pending three-person review.

## Purpose

OpenETA should help the Agent compose capable tools instead of encoding a grasp
script in host state. The host therefore owns evidence integrity, deterministic
safety checks, execution receipts, and actionable failure reporting. The Agent
owns task decomposition, tool selection, recovery, and progress assessment.

This document defines a reusable audit layer for the common failure mode in
which each tool appears individually callable but adjacent tools cannot consume
one another's outputs.

## Tool-result contract

Every planner-visible tool result uses `openeta.tool_result.v1` and exposes:

- transport/execution `success`;
- semantic `operational_success`;
- `semantic_outcome`;
- bounded but value-bearing `outputs`;
- durable artifact references for large payloads;
- diagnostics and recovery options;
- an environment receipt for acknowledged world mutation;
- explicit observation requirements.

`success=true` does not imply that a controller reached its requested target.
A `move_to` response with `reached_target=false` keeps transport success but now
sets `operational_success=false`, emits `target_not_reached`, returns requested
and actual EEF poses, and proposes replanning from the observed endpoint.

Memory compaction preserves every unknown output field under structural bounds.
It does not maintain a hand-written allowlist that turns a newly added tool into
an `output_keys`-only result. Large complete results still live in rollout
artifacts and can be queried with `python_exec`.

Large geometric payloads are host-resolved wherever a short identity is
available. `compile_grasp_seed` accepts `grasp_result_id + candidate_id`; the
host retrieves the complete candidate and the immutable source-packet camera
extrinsics. A wrist-viewpoint IK check accepts
`viewpoint_proposal_id + candidate_id`; the host injects the exact returned xyz
and orientation. These paths eliminate model transcription of matrices while
leaving generic `ik_preview_check(target_pose=...)` available for Agent-authored
motion hypotheses.

AnyPlace uses the same rule. A successful result returns a deterministic
`result_id` plus a compact `camera_pose_to_world_handoff`. The Agent selects one
candidate and supplies only `placement_result_id + candidate_id`; host memory
resolves the frozen `place_grasp_pose`, source observation packet, and camera
extrinsics. This removes artifact parsing and matrix transcription from the
main Agent without selecting a placement candidate or trajectory on its behalf.

The same short-reference rule now covers auxiliary perception. `molmopoint`
accepts an ordered list of packet/camera references; `estimate_depth_prior` and
`enhance_depth` accept one packet/camera reference. Host code resolves local
RGB-D paths, intrinsics, and the newest matching depth prior. Resolved paths and
calibration payloads are dispatched to tool handlers but are not retained in
the Agent action/transition ledger.

Legacy handlers that already return a structured `details.outputs` object are
flattened into the canonical `openeta.tool_result.v1.outputs` exactly once.
Double nesting under `outputs.outputs` is a contract error because downstream
memory extractors otherwise cannot see producer fields. Large artifacts remain
on disk and the canonical outputs contain their explicit paths and provenance.

Production runtime assembly applies a 120-second default deadline to remote
perception calls. A deadline returns a structured timeout diagnostic and an
actionable one-retry-or-alternative recovery instead of blocking the episode for
the old 600-second transport default. The bound can be changed per runtime for
known slower deployments.

## Packet provenance is not task progress

`source_packet_id` is the unique handle used to resolve session-local RGB-D and
calibration inputs. It proves provenance; it is not a one-turn validity lease.
A read-only tool turn may produce a newer observation packet even when the
object scene, robot pose, and camera mount did not change.

The planner receives `openeta.no_progress_tool_loop.v1` after two consecutive,
semantically equivalent tool requests whose only difference is packet
provenance. This is a reflection warning only:

- no tool is forced or blocked;
- the Agent may consume the previous result;
- it may use materially different inputs or another tool;
- it may justify a repeat when actual visual evidence changed.

The durable rollout auditor independently detects the same pattern as
`repeated_equivalent_read_only_call`.

## Wrist-view recovery

Targeted geometry now rejects a clipped or depth-poor selected mask with
`requires_better_view`. The Agent can recover globally from the scene-primary
camera or call `propose_wrist_viewpoints` for target-facing eye-in-hand poses.

`openeta.wrist_viewpoint_proposal.v1` contains:

- a stable `proposal_id`;
- the compiled grasp, object-scene, and robot-motion identities;
- exact full EEF pose candidates and their camera goals;
- a validity receipt stating that packet-id-only refresh does not invalidate
  the proposal;
- a next-action contract explaining how to copy one candidate into
  `ik_preview_check`.

The Agent chooses the candidate. Exact full-pose IK remains required before
motion, but no hover/contact task-stage state machine is introduced.

## Collision-coverage receipts

IK and motion tools emit `openeta.collision_coverage_receipt.v1`. The receipt
separates endpoint, trajectory, and world-object coverage. A remote
`collision.checked=true` on IK is interpreted as endpoint-configuration
coverage; it does not imply trajectory or world coverage. "No collision" is
never projected as general safety outside the explicitly checked scope.

Every `move_to` is additionally bound to a current-epoch `ik_preview_check`
receipt for the exact target xyz and orientation policy. A normal `feasible`
receipt authorizes the endpoint. If IK found a valid joint solution but its
optional endpoint collision backend was unavailable, the receipt is classified
`kinematically_feasible_collision_deferred`: it authorizes execution only when
`move_to` keeps `enable_collision_check=true` and the environment creation
capability explicitly declares a controller that owns per-step pre-actuation and
post-step trajectory/world collision checks. Missing capability, disabled checks,
or incomplete motion collision coverage remains fail-closed. Repeating the exact
IK with collision disabled adds no evidence and is flagged by the rollout audit.

An explicit full rotation and `preserve_current_orientation` are different
policies and cannot authorize one another. A receipt becomes stale after robot
motion or an object-scene change. This is an evidence-validity gate, not a task
phase: the Agent remains free to propose any pose and can repair an infeasible
pose with a position-only check, another orientation policy, another grasp
candidate, or another waypoint.

When the conservative AABB of a newly attached object already overlaps a
neighbour, the collision checker permits only motion whose predicted batches
strictly reduce that pre-existing overlap. New, unchanged, or worsening overlap
is still rejected. This gives a closed gripper a safe egress path without
globally disabling attached-object collision checks. Positive collision
evidence names the attached object and obstacle in the planner-visible result;
zero executed controller steps are reported explicitly.

A successful zero-step motion is not described as a mutation. When the current
EEF is already inside the requested tolerance, the ToolResult uses
`target_already_within_tolerance`, records `motion_outcome=no_state_change`, and
states that the robot and physical camera viewpoint did not change. Its recovery
options tell the Agent to consume existing visual evidence or propose a materially
different checked endpoint instead of treating a newly minted packet id as a new
view.

## Memory Bank preflight

Evaluation preflight probes the configured Object Memory Bank and reports its
endpoint, health, namespace, object count, or structured network failure. The
service is a required harness capability: a failed health probe is a preflight
error and no provider or simulator rollout is started. If the service becomes
unavailable after a run has started, the tool still returns structured service
diagnostics so the Agent can understand the interruption instead of receiving an
opaque tool failure; that runtime receipt does not make the capability optional.
The generic `--skip-mcp-check` debugging switch does not bypass this probe because
the HTTP Memory Bank is not an optional perception MCP backend.

## Post-hoc audit

Run the generic auditor against any durable rollout:

```bash
python -m agent.evals.tool_contract_audit \
  --tool-events path/to/rollout/tool_calls.jsonl
```

`--tool-events` also accepts a rollout, session, or attempt directory when it
contains exactly one `tool_calls.jsonl`. A run/job directory with multiple
rollouts fails as ambiguous instead of silently selecting one attempt.

The current audit checks missing diagnostics, missing operational semantics,
nonterminal outcomes without recovery, mutation receipts, motion pose feedback,
collision scope, packet lineage, repeated perception, repeated equivalent
read-only calls, redundant no-collision IK after an already valid kinematic
solution, failed IK receipts that incorrectly expose motion authority, unlabeled
zero-step motion, and target misses mislabeled as operational success.

Experiment-specific metrics remain separate extractors over the same rollout.
They are not added to the generic evaluation entry point.

### Ordered toolchain coverage

Contract correctness alone does not show whether the Agent consumed a producer's
handoff. Use the separate ordered-coverage extractor to scan one rollout or a
run root containing many rollouts:

```bash
python -m agent.evals.toolchain_coverage \
  --tool-events path/to/run-or-rollout \
  --spec evaluations/wrist_refinement_toolchain_coverage.json
```

The JSON spec declares ordered milestones, optional exact field predicates,
alternative predicates, cross-tool captured-ID references, and an optional
maximum gap. The report keeps matching
post-hoc and policy-neutral: it gives each rollout's longest matched prefix,
the next missing milestone, matched sequence numbers, aggregate completion
rate, and failure-frontier counts. It neither injects a scripted trajectory into
the Agent nor changes the universal eval reducer.

For quick inventory without field predicates, inline sequences are also
accepted:

```bash
python -m agent.evals.toolchain_coverage \
  --tool-events path/to/run \
  --sequence wrist=propose_wrist_viewpoints,ik_preview_check,move_to,sam3
```

The auditor also checks that every motion has a matching feasible current IK
receipt, detects repeated zero-step collision deadlocks, and detects repeated
attempts at the same target/orientation policy that converge to the same wrong
EEF pose. The runtime projects the last pattern as a non-blocking reflection
warning so the Agent can change candidate, orientation policy, waypoint, or
controller-compatible recovery strategy instead of retrying an attractor.

## 2026-08-19 canary evidence

Run `wrist-viewpoint-toolchain-20260818-r1` was intentionally interrupted after
the failure was reproduced. Before interruption it completed the expected
SAM3 → selection → AnyGrasp → compile → viewpoint proposal → exact IK → move
chain. The first viewpoint motion ended about 1.28 cm from its request. A later
contact motion stopped at the iteration limit about 3.89 cm from its request;
the result exposed the actual EEF pose and the Agent recovered through fresh
scene-primary segmentation.

The recovery then repeated an equivalent viewpoint proposal while packet ids
advanced but robot/object epochs and geometry remained unchanged. Re-auditing
the 29 completed tool calls found:

- one target miss incorrectly marked operationally successful;
- eight repeated equivalent read-only calls;
- two incomplete collision-coverage warnings.

The first two are fixed in the harness and covered by regression tests. The
remaining collision warnings truthfully describe simulator-service coverage:
trajectory/world collision evidence was not returned and must not be inferred
by the harness.

A short non-mutating regression (`wrist-viewpoint-consumption-20260819-r1`)
then called the proposer exactly once and consumed its candidate in the next IK
turn; seven completed tool calls produced zero findings under the original
auditor. It also exposed two payload-copy risks: the first compile attempt lost
the exponent in one extrinsics entry, and the IK request copied xyz without the
returned rotation. The ID-only host resolvers and new exact-viewpoint audit rule
were added from that evidence; the r1 diagnostic is not counted as benchmark
success because it intentionally ended with `talk` and did not execute the
LIBERO task.

The ID-only rerun `wrist-viewpoint-consumption-20260819-r2` completed the same
diagnostic in seven tool calls with zero audit findings. The Planner's compile
request contained only `grasp_result_id + candidate_id`; the IK request contained
only `viewpoint_proposal_id + candidate_id`. Both host resolutions succeeded on
the first attempt, the proposal was called exactly once, and exact full-pose IK
returned reachable. As in r1, the generic evaluation status is intentionally
not an objective-success score because the non-mutating diagnostic ends with
`talk` rather than executing the benchmark task.

As a historical coverage inventory, the auditor scanned the 20 most recent
local rollouts (491 completed calls) and classified 267 findings. The dominant
legacy categories were missing motion collision coverage (77), missing source
packet lineage (58), planner-side compile payload copying (48), missing IK
coverage (34), and target misses labeled operationally successful (27). These
counts describe old stored outputs, not the current harness: each harness-side
category now has a regression test, and the current ID-only r2 rollout has zero
findings. Incomplete trajectory/world collision coverage remains an explicit
simulator-service boundary.

### Full-chain r2/r3 follow-up

The full-chain run `wrist-viewpoint-toolchain-20260819-r2` reached and visually
confirmed a closed grasp, but exposed three adjacent-tool failures. The Agent
could execute a clearance pose whose full rotation did not match its feasible
IK receipt; the attached-object AABB then began in overlap and blocked every
escape motion at zero steps; and nested collision details were omitted from the
top-level planner feedback. The exact-IK gate, monotonic initial-overlap egress,
feedback promotion, and corresponding audit rules above were implemented from
this evidence.

The post-fix run `wrist-viewpoint-toolchain-20260819-r3` verified the intended
behavior through the contact attempt:

- the first packet id, compile short ids, wrist proposal short id, and every
  exact IK authorization were consumed correctly;
- the Agent did not close the gripper after a contact move reported that it had
  missed its target;
- endpoint-only collision scope and requested/actual EEF poses were visible in
  the decision context;
- no motion lacked a matching feasible exact IK receipt.

The run did not reach gripper closure, so monotonic attached-overlap egress has
unit-contract coverage but not live coverage yet. Its dominant blocker was
controller execution: candidate `gpe-18a43dde092246f8-004` was feasible under
IK, while two OSC contact attempts for the same target converged to nearly the
same wrong EEF pose, about 8.3 cm away, at the iteration limit. The improved
auditor reported this as `repeated_failed_motion_attractor`. This separates the
remaining problem from perception and evidence plumbing: current evidence
points to controller/trajectory compatibility rather than missing planner
context. Existing Mink canary results are promising, but production controller
replacement and its collision/contact/runtime boundaries remain a separate
architecture decision.

Full regression after these changes: `1176 passed, 12 skipped`.

### Mink Agent r1/r2 contract findings

The first worker-local Mink Agent runs exposed three general harness defects
that deterministic controller tests alone could not reveal:

- **bounded projection was mistaken for durable evidence:** the grasp advisor
  selected candidate 7 while working memory retained only the first five
  candidates. Compilation succeeded from the durable event, but contact
  authorization searched the bounded projection and rejected the same compiled
  id forever. Host provenance now verifies out-of-window candidates against the
  complete immutable structured artifact, including SHA-256, tool identity,
  result identity, and session-root checks. Agent-facing projections remain
  bounded.
- **opaque IDs were still too copy-fragile:** provider observation packet ids
  exceeded 80 characters and one model call omitted an internal hash segment.
  Current packets now receive short session-scoped aliases such as `obs-0006`;
  the host maps those aliases to camera paths, calibration, epochs, and the
  original provider id. Invalid aliases still fail closed with recent valid
  references.
- **controller exceptions erased recovery evidence:** a no-solution Mink QP
  surfaced only as `AssertionError:`. The worker now returns a named QP failure
  or performs only geometrically verified collision/joint boundary escape
  steps. The escape policy and step count are included in the collision receipt.

These are instances of one audit rule: anything the Agent can reference must
remain resolvable from complete host evidence, and every refusal must preserve
the actual failed contract plus an actionable recovery boundary.

### Mink Agent r8: contact-quality chain and systematic audit

Run `mink-agent-canary-20260819-r8-sol-system-audit` completed 60 Agent tool
calls without the prior packet-lineage, freshness, short-id, exact-IK,
collision-coverage, or opaque-controller findings. The Agent reached contact,
closed, lifted, detected a failed hold, opened, and began an autonomous
recovery. It stopped at the 60-turn budget rather than an infrastructure or
gate deadlock.

The remaining failure was one linked contact-quality chain:

- the compiled contact motion reported `reached_target=true` under a 10 mm
  maximum-axis tolerance, but its maximum-axis residual was 7.93 mm and its
  Euclidean residual was 12.05 mm;
- the old gripper-close response did not expose whether a carried-object proxy
  was armed or which object it represented;
- after a 5.62 cm lift, measured gripper openness fell from 0.1495 to 0.0399,
  which is strong aperture evidence of a slip/empty close and agreed with the
  dual-view images.

The generic rollout auditor now detects these as
`contact_execution_residual_exceeds_guidance`,
`gripper_close_without_attachment_proxy_receipt`, and
`probable_grasp_slip_after_lift_probe`. It also expires wrist-viewpoint
attribution after an unrelated tool, preventing a stale proposal from falsely
accusing a later grasp IK call.

The compiler now emits additive `openeta.compiled_grasp_motion_guidance.v1`:
contact starts at a recommended 5 mm maximum-axis position tolerance and 0.10
rad orientation tolerance, followed by explicit requested/actual-pose and
fresh-wrist inspection. This remains Agent guidance rather than a task-state
transition or a hard stage gate.

Gripper close now privately reuses active compiled target provenance. The
simulator resolves that host evidence to one live scene object before arming a
tentative carried-object collision proxy, and returns
`openeta.attachment_proxy_receipt.v1` with target name, binding source,
aperture, status, and `attachment_proven=false`. Aperture and a tentative proxy
remain insufficient; transport still requires post-lift co-motion and
source-vacancy evidence.

Deterministic result
`tmp/mink-contact-attachment-binding-mink-v2-20260819.json` passed all 15
checks: host-bound close target, no false attachment claim, exact contact
authorization, lift, object/EEF proximity, shape-aware bounds, and predicted
plus actual per-step carried-object collision coverage. Full repository
regression after these changes: `1220 passed, 12 skipped`.

### Mink Agent r9: semantic feedback consistency

Run `mink-agent-canary-20260819-r9-sol-contract-audit` completed 59 tool calls
without a gate, freshness, reference-resolution, controller-receipt, or
collision-coverage defect. The Agent moved to a checked wrist viewpoint,
resegmented the target, reran AnyGrasp from the near-field packet, and reached
the new clearance/contact pair with about 2.0 mm and 3.7 mm Euclidean EEF
error respectively. Gripper close then returned a decisive empty-close receipt:
`status=not_armed`, `reason=empty_close_or_no_measurable_contact`, and measured
open fraction `0.0319`.

The structured receipt was correct, but its old human-readable suffix always
recommended a small lift probe. The Agent followed that contradictory sentence
and spent four calls on two empty lifts before reopening. Gripper-close feedback
now has two explicit semantic branches:

- `requires_attachment_probe` only when a host-bound tentative proxy exists;
- `no_attachment_evidence` for `not_armed` or retired proxies, with explicit
  fresh-dual-view and reopen/repair-contact recovery and a warning not to lift.

The auditor now reports `lift_after_explicit_empty_close`. It also found a
point-prompt call that incorrectly combined `roi_bbox_xyxy`; the SAM3 contract
now states that ROI attention is text-mode-only and mutually exclusive with
point prompts, and the auditor reports
`mutually_exclusive_perception_prompts`.

The run ended at its turn budget while generating a third, visually grounded
recovery candidate, rather than at a harness deadlock. Its four audit findings
were exactly the two newly fixed classes above plus two unchanged-view
perception-repeat warnings. The remaining primary failure is grasp candidate /
contact quality: the controller reached the requested contact accurately, but
the selected pose still closed outside the object.

Post-fix deterministic result
`tmp/mink-contact-attachment-refresh-mink-v3-20260819.json` passed all 16
checks. A successful 11.76 cm lift retained the conservative tentative proxy,
returned its refresh receipt, preserved predicted/actual carried-object
trajectory coverage, and did not claim attachment proof. Full repository
regression after these fixes: `1225 passed, 12 skipped`.

### Systematic harness r2: deployment contract skew

Run `systematic-harness-milk-20260819-r2` executed 26 tools through the natural
Agent loop. It completed target and receptacle segmentation, grasp estimation,
compiled-seed resolution, AnyPlace, camera-to-world transformation, IK,
clearance, and contact. This confirms that the short-ID handoffs and repaired
compiled-grasp provenance survive the complete producer/consumer chain.

The first close was visually empty. The Agent reopened, requested a fresh wrist
grasp estimate, retried contact, closed again, and used a guarded lift to test
co-motion. The milk remained at its source, so the Agent reopened and started a
fresh segmentation rather than claiming success. The remaining failure was
grasp/contact quality, not a hidden stage or freshness deadlock.

The audit found that the simulator process at `127.0.0.1:8766` was older than
this checkout. Comparing all shared tool schemas found exactly three missing
parameters:

- `gripper_close.contact_authorization`;
- `move_to.contact_authorization`;
- `move_to.ik_execution_seed`.

Consequently the service could not bind the host-selected object, refresh an
attachment collision proxy, or bind previewed IK to execution. The host now
handles this skew at two levels:

1. MCP catalog discovery reports explicit compatibility diagnostics before the
   first action.
2. If an old backend accepts a call but omits the attachment receipt, the proxy
   returns `status=backend_contract_missing`, `attachment_proven=false`, and
   `collision_proxy_active=null`. It offers fresh dual-view inspection, a small
   guarded lift only when visually plausible, and simulator upgrade/restart as
   separate recovery options. A close without any active host authorization is
   instead reported as `not_armed/no_active_contact_authorization`.

These receipts describe capability uncertainty; they do not fabricate
attachment, block arbitrary Agent motion, or reintroduce a task-progress state
machine. Post-fix repository regression: `1251 passed, 12 skipped`.

### Mink natural Agent r4: numerically stable IK binding

A current-repository simulator was started separately on port 8769 with the
host-selected `mink_joint_velocity` profile. Its capability smoke reached a
2 cm upward target in 3/40 steps with 0.388 mm position error and a matching
worker-local controller receipt. Unlike the stale 8766 deployment, the service
exposed `move_to.contact_authorization`, `move_to.ik_execution_seed`, and
`gripper_close.contact_authorization`.

Run `systematic-harness-milk-mink-20260819-r3` exposed a host-side receipt
binding defect before contact: copying one previewed rotation matrix through
the model changed individual components by about `1e-10` to `7e-8`. The old
byte-exact pose signature treated that harmless JSON round trip as a different
orientation and demanded another IK preview.

IK receipt lookup now keeps the signature as its fast exact index, then falls
back to the receipt's stored pose using 1 micrometre position and `1e-6`
orientation-component absolute tolerances. These values are far below the
controller's millimetre/radian tolerances. A regression reproduces the exact r3
matrix differences and also proves that a 20 micrometre position change still
requires a new preview. This is numerical equivalence, not an Agent adjustment
allowance.

After the fix, run `systematic-harness-milk-mink-20260819-r4` crossed the same
boundary without a redundant preview:

- clearance and contact motions returned complete per-step trajectory/world
  collision coverage;
- contact reached in 21 steps with 1.64 mm Euclidean position error;
- close returned `tentative/non_empty_close_near_bound_target`, bound to
  `milk_1`, with measured open fraction 0.184;
- a four-step lift probe returned
  `retired/aperture_collapsed_to_empty_close` as openness fell to 0.046;
- the Agent reopened immediately, reacquired current target evidence, and did
  not transport an unproven object;
- a later poor GraspGenX clearance was stopped after five steps by an explicit
  `gripper0_hand_collision` versus `milk_1_g1` worker collision receipt. The
  Agent exited the reported boundary and resegmented instead of replaying it.

The 29-call audit passed the contract (`error_count=0`). Its only warnings were
the correctly diagnosed probable slip and one repeated equivalent read-only IK
call. The remaining task failure is candidate/contact quality: the harness now
coordinates the attempt, reports the slip, and supports recovery without a
hidden task-stage machine. Post-fix repository regression:
`1253 passed, 12 skipped`.

### Systematic Mink r5-r8: explicit adjacent-tool handoffs

Run `systematic-harness-milk-mink-20260819-r5` stopped before motion even
though its IK result had already found a valid joint solution. The endpoint
collision backend was unavailable, and the Planner did not combine that result
with a separate controller-capability block declaring Mink's per-step
trajectory/world collision coverage. IK results now include
`openeta.ik_motion_collision_delegation.v1` directly. A matching move is
actionable only when the current controller explicitly owns that coverage and
`enable_collision_check=true`; unknown controllers remain fail-closed.

Run r6 reproduced the same deferred-IK branch and immediately called the
guarded Mink move instead of `ask_human`. The controller then rejected a
constraint-escape preview with a named boundary receipt, proving that the fix
connected safety ownership rather than bypassing collision checks. Its 22-call
audit passed with no errors and one unchanged-view repetition warning.

r6 also showed that reaching a target-facing wrist viewpoint was being mistaken
for contact refinement: the Agent reused the older scene-view contact pose and
closed empty. Wrist viewpoint proposals and successful viewpoint motions now
emit an explicit post-reach evidence contract. It states that the move only
gathered evidence, identifies the wrist camera, and points the next turn to the
fresh `current_observation.source_packet_id` for segmentation, identity
selection, bounded wrist alignment, or a full wrist grasp estimate.

Run r7 consumed that handoff and segmented the wrist view, exposing three more
producer/consumer mismatches:

- an exact rotation matrix converted to an equivalent quaternion did not match
  the prior IK receipt;
- a ready wrist-alignment bundle was easy to overlook after target selection;
- current visual filesystem paths were still serialized into the main Agent's
  text context, allowing an obsolete `scene_image` argument to reappear.

IK fallback matching now canonicalizes matrix, quaternion, Euler, and rotation
vector representations and compares their angular difference while retaining
the 1 micrometre translation bound. Wrist target selection emits a direct
consumer handoff to `host_resolved_inputs.wrist_alignment`. The main Agent's
text prompt now removes transport-local visual paths while the backend still
loads the same files privately as image attachments; observation packet and
camera ids remain visible. Any Agent-supplied `scene_image` on asset retrieval
is removed, and the pipeline resolves the image privately from the packet.

Run r8 verified a single viewpoint IK-to-move path, a 0.3 mm viewpoint reach,
fresh wrist SAM3, and absence of local paths in Agent calls. Text SAM3 still
returned no detection on a visibly useful wrist view, after which the Agent
silently switched point grounding back to agentview. A no-detection result now
includes `openeta.same_view_perception_recovery.v1` with the exact packet,
camera, evidence role, SAM3 point template, and MolmoPoint source. This is
recovery evidence rather than a forced branch: the Agent may change viewpoints
when the target is actually absent or occluded, but no longer has to reconstruct
the current view identity from separate context fields.

Both r7 and r8 passed the generic tool-contract audit with zero findings. Full
repository regression after these adjacent-tool changes:
`1259 passed, 12 skipped`.

### Systematic Mink r11: coverage extractor and failed-clearance causality

The policy-neutral ordered toolchain extractor was applied retrospectively to
r6-r10. It distinguished three cases that manual inspection had conflated:

- r7 reached 5/6 wrist-alignment milestones and stopped only before
  `compute_wrist_alignment`;
- r8 reached the fresh wrist SAM3 milestone but did not select a target on that
  same wrist evidence;
- r9/r10 never chose the wrist-viewpoint branch and therefore count as
  unexercised, not failed handoffs.

The extractor accepts declarative field predicates, maximum call gaps, and
cross-tool captured-ID references. In particular, the same-view recovery spec
requires the exact `source_packet_id` produced by wrist SAM3 to be consumed by
MolmoPoint and point-mode SAM3. A later agentview call can no longer inflate
wrist-chain coverage.

Run `systematic-harness-milk-mink-20260819-r11` was stopped after 20 completed
tool calls once it produced two high-information defects. The first clearance
move ended after 100 Mink iterations with 3.96 mm position error but 0.197 rad
orientation error; a new grasp branch later ended its clearance with 3.01 mm
position error and 0.147 rad orientation error. The Agent then previewed and
executed contact even though the immediately preceding clearance receipt said
`reached_target=false`. Contact itself reached, but close was explicitly empty;
the Agent reopened rather than lifting an unproven object.

The runtime now persists `openeta.compiled_clearance_execution.v1`. Clearance
remains an optional, Agent-chosen ordinary waypoint: direct contact is not
forbidden. If the Agent did choose clearance for a compiled grasp, however, a
current-state failed receipt cannot authorize the dependent contact. The gate
returns actual EEF xyz, position/orientation error, step count, stop reason,
fresh observation access, and active bundle recovery rather than imposing a
next task phase. A successful retry, another compiled grasp, or a materially
different intervening route removes that exact dependency.

r11 also exposed an audit-only false positive: the IK and move matrices differed
in one component by about `1.9e-9`. The runtime already accepted them through
semantic rotation equivalence, while the offline auditor still compared exact
hashes. Both now share the same 1 micrometre / `1e-6` rad numerical-equivalence
rule. Re-auditing r11 leaves zero errors and one unchanged-view perception
warning. Full repository regression after these fixes:
`1265 passed, 12 skipped`.

### Systematic Mink r12: typed IK-to-motion handoff

Run `systematic-harness-milk-mink-20260819-r12` completed 20 tool calls before a
provider TLS handshake timeout produced `need_human`. The physical chain reached
compiled clearance in 63 Mink iterations (2.98 mm Euclidean position error,
0.099 rad orientation error), reached contact, closed, and completed a 4.7 cm
lift probe. The empty-close aperture and dual-view evidence correctly reported
`no_attachment_evidence`, so the Agent reopened instead of transporting an
unattached carton. The generic contract audit passed with zero errors and three
useful warnings: unchanged-view grasp re-estimation, repeated equivalent IK, and
probable grasp slip/empty close. The wrist-specific sequence was unexercised.

The two wasted recoveries had one common cause: the planner was required to copy
large structured values across tools. It first mistyped a repetitive
`compiled_grasp_id`, causing a valid clearance request to be rejected and the
same AnyGrasp/advisor computation to be rerun. Later it copied a 3x3 rotation
matrix from `ik_preview_check` into `move_to` with the third row concatenated to
the second row; the host rejected the malformed rotation, after which the Agent
repeated the identical IK call.

The Agent-facing motion contract now uses typed references:

- exact compiled anchors are previewed with `compiled_grasp_id + waypoint_role`;
  the host expands the immutable pose;
- generic or visually adjusted endpoints are still Agent-authored as full poses
  in `ik_preview_check`;
- every successful preview emits `ik_receipt_id` and
  `openeta.ik_motion_execution_ref.v1`;
- `move_to` accepts that receipt id plus execution controls, and the host resolves
  the exact checked xyz, orientation policy, provenance, tolerance defaults, and
  private IK seed;
- missing, unknown, stale, or mixed receipt/pose requests fail with recent valid
  ids and a direct repair instruction.

This is not a task state machine. The Agent still chooses the endpoint, waypoint,
candidate, adjustment, and recovery. The host merely prevents serialization
drift between a read-only safety check and the exact mutation it authorizes.

Run `systematic-harness-milk-mink-20260819-r13` was deliberately interrupted
after this handoff was covered live. The Agent called clearance IK with only
`compiled_grasp_id=074cdaa526856e5798af` and
`waypoint_role=grasp_clearance`. The tool returned
`ik_receipt_id=c8e641c949630d03758b`; the next `move_to` request contained that
id and execution controls but no pose. The host dispatched the exact stored pose,
Mink reached it in 61 iterations with 1.04 mm position error and 0.070 rad
orientation error, and the following contact IK again used the compiled
reference form. The generic contract audit reported zero errors and zero
warnings. This verifies both the public compact handoff and the private
full-fidelity execution expansion.

### Horizontal closure: trajectories, probes, and post-execution memory

A follow-up contract sweep found that the first typed migration covered direct
`move_to` calls but left two adjacent seams: `follow_eef_trajectory` still
accepted copied pose arrays, and `prepare_attachment_probe` still advertised a
raw `frozen_action`. Both now use the same typed boundary. The probe freezes only
host-owned geometry, emits one ordered IK preview request per waypoint, and
hands execution to `move_to` or `follow_eef_trajectory` by receipt id(s).

The sweep also separated the three parameter representations explicitly:

1. the public Agent request contains compact references and execution controls;
2. the pipeline tool call contains host-resolved geometry used by gates and
   execution;
3. the simulator request contains backend parameters only, with host receipt ids
   removed.

The first implementation assumed that the second representation survived in
`EnvAction.tool_calls[].parameters`. That was true while the handler ran and in
`tool_calls.jsonl`, but false after the pipeline intentionally replaced the call
parameters with the compact public request. The mismatch was not covered by the
original synthetic memory test, which manually constructed an action containing
the private pose.

Post-execution memory now reads an environment-authority
`openeta.resolved_tool_execution.v1` receipt stored as a top-level ToolResult
detail. It contains the exact target pose or ordered trajectory dispatched by
the host and remains outside bounded Agent conversation projections. Older
trusted motion results can fall back to
`outputs.pose_feedback.requested_eef_pose`; legacy/synthetic actions can still
use their recorded call parameters. Compiled clearance/contact receipts,
residual budgets, transport reconciliation, transition ledger entries, and
attachment probes therefore consume one durable execution fact without putting
raw geometry back into the Planner request.

Tests cover raw-trajectory rejection, ordered receipt expansion, infeasible
waypoint rejection, host-only id stripping, completion of a frozen articulated
probe after the robot epoch advances, and the real
`ActionPipeline -> SimulatorMcpToolProxy -> sanitized EnvAction -> AgentMemory`
path through contact and permitted gripper close.

### Systematic Mink r14: blocked calls were missing from the tool audit

Run `systematic-harness-milk-mink-20260819-r14` exercised the wrist-viewpoint
and typed-motion chain substantially farther than earlier canaries. The Agent
recovered from a MolmoPoint timeout, used fresh wrist SAM3 evidence, switched
from a rejected wrist-alignment operating region to wrist grasp estimation, and
reached two compiled contact endpoints. The first contact ended after 47 Mink
steps with 3.08 mm position error; the second ended after 78 steps with 0.616 mm
position error and 0.0997 rad orientation error. Both reported
`reached_target=true` with complete controller collision coverage.

Both subsequent binary close requests were nevertheless blocked with
`compiled_contact_receipt_missing`. The gate itself was correct: it refused to
treat a missing causal receipt as successful contact. The defect was the
representation mismatch above, so memory never saw the compiled id and
`waypoint_role=grasp_contact` that had actually been dispatched.

The original post-hoc audit misleadingly reported zero issues because
`tool_calls.jsonl` records only handlers that executed. Gate-rejected calls
exist only in the sibling durable `trace.jsonl`. The audit now reads both
sources, counts blocked pipeline plans, requires a repair code and actionable
next calls, warns on repeated identical blocks, and reports
`gate_rejected_close_after_successful_contact` when a trusted motion event
already reached the same compiled contact. Re-auditing r14 finds both lost
receipt failures and the repeated block instead of declaring the partial
tool-event stream clean.

### Systematic Mink r16: live contact-to-close recovery

Run `systematic-harness-milk-mink-20260819-r16` used the current local
simulator at `127.0.0.1:8769` and was intentionally interrupted immediately
after the repaired boundary was covered. Session
`32c31711-4c55-4df5-ab01-b387213914f6` completed 19 tool calls before cleanup.

The Agent independently followed the same high-information route as r14:
wrist-viewpoint proposal, typed IK and move, same-view point-SAM recovery after
a MolmoPoint timeout, wrist target selection, and a
`compute_wrist_alignment=requires_better_view` result. It then chose the
compiled clearance/contact route rather than being advanced by a host stage.

- wrist viewpoint: 19 steps, 0.782 mm position error;
- compiled clearance: 57 steps, 2.494 mm position error;
- compiled contact: 17 steps, 2.996 mm position error;
- all three motion ToolResults contain
  `openeta.resolved_tool_execution.v1`;
- the durable memory trace contains matching
  `openeta.compiled_clearance_execution.v1` and
  `openeta.compiled_contact_execution.v1`, with contact
  `reached_target=true` at robot-motion epoch 3;
- the next `gripper_control(position=0)` pipeline plan was `executed`, not
  blocked, and the simulator returned
  `tentative/non_empty_close_near_bound_target` with
  `requires_attachment_probe`.

The combined tool-event plus pipeline-trace audit reports 19 executed calls,
zero blocked plans, zero findings, and every quality check passing. The run was
not a task-success measurement: it was stopped while the Agent began the lift
probe, and the interrupted in-flight motion appears as an actionable failed
tool result. Explicit cleanup reported `already_closed=true`.

### Systematic Mink r17: exact same-view provenance and probe-size diagnosis

Run `systematic-harness-milk-mink-20260819-r17` completed its full 60-turn,
120-provider-request budget. It did not complete the task, but the generic
contract audit passed every hard quality check with zero errors. The Agent
recovered from infeasible full-pose IK, a zero-step collision rejection, a
MolmoPoint timeout, and repeated empty attachment evidence without a host-owned
task stage.

The wrist-viewpoint branch reached five of six alignment milestones and the
full-pose wrist fallback completed. The missing alignment consumer was not a
viewpoint or motion failure: after wrist SAM3 returned no detection on
`obs-0043`, the model silently substituted the newer read-only packets
`obs-0044` for MolmoPoint and `obs-0045` for point-mode SAM3. The physical view
was unchanged, but the immutable pixel provenance was no longer exact. The
pipeline now applies one evidence-integrity rule to MolmoPoint and point-mode
SAM3: same-camera recovery must include the exact active no-detection packet.
Switching to another camera or viewpoint remains Agent-owned. A mismatch returns
`same_view_packet_mismatch` plus a directly executable exact-packet repair call.

r17 also showed that both apparent lift slips used the wrong diagnostic motion.
After tentative close, the Agent reused the compiled `grasp_clearance` anchor:

- first displacement: 14.95 cm, post-close aperture 0.1799, then empty-close
  aperture 0.0175;
- second displacement: 14.24 cm, post-close aperture 0.7295, then empty-close
  aperture 0.0125.

The second close image showed the carton still standing beneath the gripper; the
later frame showed the gripper at clearance while the carton remained on the
table. This is not evidence that a continuously latched close dropped a proven
attachment. It is an oversized, strongly lateral first probe that cannot isolate
contact quality from slip. The portable-object guidance and gripper ToolResult
now recommend a new Agent-chosen 2-5 cm probe from the measured current EEF pose,
require an exact IK check, and explicitly reject reusing old clearance or
precontact anchors. Probe direction is still chosen from current dual-view
geometry rather than a fixed host script. The offline audit adds
`oversized_attachment_probe_after_close`; re-auditing r17 reports both oversized
clearance probes alongside the aperture-collapse evidence.

Focused regression for the exact-packet gate, simulator feedback, toolchain
coverage, and probe audit: `135 passed`.

### Systematic Mink r18: Agent-authored lift succeeds; transport slip and joint-boundary deadlock separated

Run `systematic-harness-milk-mink-20260819-r18`, session
`7a2fe548-a102-4409-8944-e8ec0d0640ca`, exhausted its 60-turn budget without
task success. Its 60 executed tool calls nevertheless provide two important
causal results. The generic contract audit passed with zero errors; its only two
warnings were repeated perception calls on unchanged views.

First, the short-probe guidance worked without a task stage. After contact
reached with 1.12 mm position error and close returned a non-empty tentative
aperture, the Agent independently read the measured EEF pose and authored an
approximately 3.1 cm, nearly vertical preserve-current lift. Mink executed 2.54
cm before satisfying the requested per-axis tolerance, the aperture remained
near 0.59, and both agentview and wrist images showed the milk carton co-moving
with the gripper. This is direct live evidence that complete tool feedback lets
the Planner choose a useful diagnostic action without a host `hover/lift` state.

The later 12.8 cm transport attempt is a different failure. The carton was held
after the intermediate move, but during the longer upward command the measured
aperture collapsed from 0.58 to 0.114 and the carton returned to the floor. The
worker's attached-object geometry then reported a real `milk_1_g1` versus
`floor` penetration. The collision receipt did not cause the drop; it detected
the physical slip after it occurred.

After reopening, a wrist-viewpoint move advanced 29 steps and approached within
10.8 mm, then the local controller entered a repeatable deadlock. Panda joint 6
was at 3.73498 rad, 0.0175 rad below its hard upper limit. The strict collision
and configuration-limit QPs were infeasible; the emergency velocity solved, but
its next full step crossed the joint boundary, so every later command was
rejected as `constraint_escape_preview_rejected` with zero execution. Robot and
world clearance remained 2 cm and there was no current joint violation, proving
that this was not a collision-recovery problem.

The controller/toolchain remediation is deliberately geometric rather than
task-specific:

- preserve-current IK receipts now carry the same private, exact-target,
  tolerance- and robot-epoch-bound joint seed as explicit full poses;
- the global IK posture target has enough null-space weight to keep local Mink
  in the previewed redundancy basin;
- when the configuration-limit QP is numerically infeasible, the emergency
  velocity is projected onto the hard joint box one joint at a time, then still
  undergoes the full predicted collision and joint-safety preview;
- near-limit IK results report the closest joint, boundary, and margin rather
  than presenting a fragile endpoint as an ordinary reachable pose;
- motion with an active tentative/confirmed attachment proxy uses the
  `attached_object_gentle` profile at 0.2 rad/s instead of the nominal 0.5
  rad/s. The policy depends only on host-owned physical attachment evidence,
  not pick/place stage.

Three live canaries passed against a freshly restarted current-checkout service
on port 8769 using the minimal Mink dependency overlay (which preserves the
LIBERO MuJoCo 3.3 runtime):

- controller capability: 2 cm free-space move reached in 3 steps with 0.388 mm
  error and a truthful nominal controller receipt;
- exact r18 wrist target plus preserve-current retreat: seeded full-pose motion
  reached in 70 steps, the following 5 cm retreat reached in 3 steps, both seed
  ids were bound, and no constraint-escape deadlock occurred;
- deterministic contact/close/12 cm lift: the gentle profile reached in 20
  steps, lifted the object 11.42 cm, retained a 0.464 aperture, kept the object
  4.22 cm from EEF, and reported complete predicted/actual attached-object
  trajectory coverage with no collision.

The first restart accidentally used a broad dependency directory containing
MuJoCo 3.11 and failed environment creation because old robosuite expected
`MjData.qM`. The deployment now uses `/tmp/openeta-mink-canary-min`, containing
only Mink, qpsolvers, and quadprog. This is recorded as an infra reproducibility
constraint, not hidden by retrying the controller.

Focused post-fix regression: `177 passed`; the later cross-camera AnyPlace
rebase change leaves the full repository at `1286 passed, 12 skipped`.

### Systematic Mink r20: wrist refinement exposed an AnyPlace camera-provenance loop

Run `systematic-harness-milk-mink-20260819-r20-sol`, session
`1ed58990-04da-4018-bf1c-a5d5078b8fc9`, reached the real grasp chain with
`gpt-5.6-sol`:

- the initial fixed-camera chain completed through AnyPlace, full-pose IK, and
  a 91-step seeded Mink clearance move with 3.1 mm Euclidean endpoint error;
- fresh wrist segmentation and the calibrated wrist-alignment bundle were used
  correctly.  The bundle rejected an 8.8 cm raw correction outside its 2 cm
  operating region and returned exact recovery evidence;
- the Agent followed that evidence and obtained a higher-confidence wrist grasp,
  but the AnyPlace resolver then reported `placement_source_mismatch` and forced
  a new fixed-camera grasp.  This discarded the wrist refinement;
- the resulting close initially retained 0.312 aperture, but a 3.5 cm gentle
  lift probe reduced it to 0.033.  The proxy correctly retired with
  `no_attachment_evidence`.  Mink reached the probe in 7 steps with the new
  0.2 rad/s attached-object profile, so this failure was contact quality rather
  than controller accuracy or gripper-latch loss.

The resolver now preserves the one-camera AnyPlace input contract without
discarding wrist geometry.  When a prior same-target, same-scene fixed-camera
bundle or durable compiled-grasp provenance exists, the host uses exact source
and target packet extrinsics to
re-express the wrist candidate in the fixed camera.  RGB-D, object mask, and
placement mask remain aligned to that fixed source; candidate world geometry is
unchanged and the bundle exposes
`grasp_rebase.mode=calibrated_world_invariant`.  Missing calibration, target
identity mismatch, or scene-epoch mismatch still fails closed to the explicit
repair path.

### Systematic Mink r21-r23: arbitrary-order rebase coverage and failed-motion epoch integrity

r21 showed that requiring a previously materialized AnyPlace bundle still
encoded an accidental action ordering: the Agent may legitimately refine a
wrist grasp before its first placement call. Compiled grasp provenance is now
recorded durably with its complete private geometry, and the resolver may use a
same-target, same-object-epoch fixed-camera provenance record as the calibrated
rebase anchor. This record remains off the normal Agent projection.

r22 reached the recovery branch but exposed a separate full-pose controller
plateau. Increasing the exact clearance command from 100 to 150 and 300 steps
left orientation error at approximately 0.197 rad, including with collision
checking disabled. The exact IK endpoint existed but had only 0.0329 rad joint
margin. The controller now reports local convergence stall evidence and IK
warns below 0.05 rad rather than suggesting that a larger iteration budget is a
generic repair.

r23 (`systematic-harness-milk-mink-20260819-r23-margin-aware`, session
`5e6eab09-0fbb-4102-9649-d97a233bf341`) ran 60 turns with `gpt-5.6-sol` and
provided the live arbitrary-order coverage:

- Memory Bank preflight was healthy (`libero`, 40 objects).
- The initial AnyGrasp chain reached contact at 5.0 mm Euclidean error. Close
  retained 0.208 aperture, but an Agent-authored 3 cm preserve-current probe
  collapsed to 0.049; the attachment proxy correctly retired as empty-close.
- The Agent reopened and recovered wrist perception through text-SAM no
  detection, exact-packet point-SAM, and explicit candidate selection.
- After wrist `grasp_pose_estimate` and `compile_grasp_seed`, the AnyPlace bundle
  remained `ready` with no repair call and
  `grasp_rebase.mode=calibrated_world_invariant`, from wrist `obs-0016` to fixed
  agentview `obs-0000`, under the unchanged target identity anchor. No forced
  fixed-camera grasp rebuild occurred.
- A later near-table wrist candidate entered a collision recovery boundary. The
  worker returned current and predicted signed distance plus
  `constraint_escape_preview_rejected`; the Agent chose a new checked wrist
  viewpoint and reached it instead of replaying the same command.
- MolmoPoint later timed out and several wrist/full-pose candidates either
  failed orientation convergence or had unsafe near-table geometry. The task
  therefore remained incomplete; the residual bottleneck is candidate/control
  quality, not the former AnyPlace freshness loop.

Post-hoc toolchain coverage was complete for the initial pick chain, placement
handoff, and wrist-recovery chain. The contract audit found one harness error:
a failed `move_to` had executed 11 steps and changed the EEF pose, but memory
advanced `robot_motion_epoch` only for calls whose top-level result was success.
That left a pre-failure wrist-viewpoint IK receipt apparently current and allowed
it to be reused later.

Robot epoch advancement now follows trusted physical execution evidence rather
than goal success. A failed motion advances the epoch when its structured
receipt proves positive executed steps or materially different start/end poses;
a blocked, skipped, or zero-step failed call does not. Exact tests cover partial
execution, no-receipt failure, zero-step success, and rejection of a preexisting
IK receipt after partial failure. Same-view packet mismatch and stale-contact
warnings in r23 were reviewed as correct fail-closed behavior: each returned an
exact executable repair, and the Agent subsequently recovered.

Focused contract regression: `161 passed`. Full repository regression after the
r23 epoch-integrity patch: `1290 passed, 12 skipped`.

### r23 candidate causality: endpoint-feasible is not an execution recommendation

A candidate-by-candidate reconstruction separated visual grasp selection from
controller executability. The read-only advisor abstained on the initial two
agentview candidates; the Agent knowingly used rank 0 and later recovered after
the shallow grasp slipped. On the two wrist estimates, the Agent copied the
advisor recommendations exactly:

- `gpe-d312b1fc45a7435a-001` had a 0.0810 rad minimum joint margin. Its contact
  motion reached 1.9 mm position error but stopped at 0.357 rad orientation
  error.
- `gpe-7be7539bcc00422a-002` had only 0.00534 rad margin on its first contact
  preview. The Agent nevertheless executed it; the controller stopped after 11
  steps at a constraint/collision recovery boundary. A later preview from a
  different robot configuration had 0.2119 rad margin, but the same contact
  geometry still failed locally.

The advisor had not violated its contract: it ranked 2-D object-relative jaw
geometry and explicitly declared that back-side contact, depth clearance, and
reachability were uncertain. The missing contract was downstream. The global IK
solver returned the first tolerance-satisfying branch and that branch became the
private Mink posture seed, so `reachable` was easy to misread as a positive
execution recommendation.

IK preview now performs bounded execution-seed selection without turning risk
into a gate. A first solution with at least 0.10 rad hard-limit margin returns
immediately. A more fragile first solution triggers at most seven additional
read-only starts. Candidate branches expose joint margin and L2/max joint travel.
Selection prefers robust branches only inside a 1.5 rad extra-travel envelope
around the nearest feasible branch; a mathematically robust but distant solution
is reported, not silently converted into a long local-controller posture route.
If no local robust branch exists, the endpoint remains `reachable` but returns
`execution_seed_quality=elevated|critical`, candidate summaries, and an explicit
recommendation to compare another waypoint, orientation, or grasp candidate.
This changes neither task order nor motion authorization.

A targeted task-7 replay reached the optional-search path. Seven IK branches
were feasible: the selected local branch had only 0.00173 rad margin and 1.80 rad
joint travel, while five higher-margin branches required 3.63-5.04 rad travel.
The tool correctly returned `critical`,
`distant_robust_solution_count=5`, and did not hide the local-control risk behind
global reachability. A separate difficult-pose canary retained the fast path:
one robust 0.397 rad solution was found in 0.16 s, and seeded Mink reached the
target in 88 steps with 0.30 mm position error and no collision.

The post-hoc contract auditor now emits
`failed_motion_after_execution_fragile_ik` when a failed motion consumed an
elevated/critical receipt. Re-auditing historical r23 identifies exactly the two
causal cases above at tool seq 50 and 82. This is a warning and does not make a
deliberately accepted risk a host rejection. The pick/simulator guidance now
asks the Agent to compare untried candidates when this evidence is available.

The first 36-turn live rerun, r24, stopped at turn 2 because the primary provider
returned an overload HTTP 500 and the fallback exceeded its 60 s read timeout;
it did not reach the behavior under test. Planner provider timeouts were raised
to 180 s while preserving `max_attempts=3`, and r25 was started with the same
task, seed, model, tools, and evaluation plan.

The r25 rerun (`systematic-harness-ik-seed-risk-20260820-r25-timeout180`,
session `08f3a197-c54c-49b6-9fa0-688cdd5a2a0e`) reached the intended behavior
boundary. The primary `gpt-5.6-sol` endpoint remained overloaded, so all 23
successful planner decisions came from the configured `gpt-5.6-luna` fallback.
The initial clearance and contact previews selected robust branches with 0.621
and 0.633 rad joint margin. Mink reached them in 75 and 28 steps with 2.61 and
1.41 mm position error, complete trajectory/world collision coverage, and no
collision. This is a direct positive contrast with the two r23 motions that
consumed fragile first-hit IK branches and then failed locally.

The first close remained a real grasp-quality failure, not a motion-contract
failure. It retained 0.227 aperture, but an Agent-authored 3 cm vertical probe
moved the EEF 2.63 cm and collapsed the aperture to 0.033. The attachment proxy
retired with `no_attachment_evidence`; fresh agentview and wrist images showed
the carton still on the floor. The Agent reopened, discarded the old grasp, and
started a new target-to-grasp chain. Text SAM returned no detection. A point-SAM
attempt on a newer packet was rejected with the exact immutable recovery anchor;
the Agent retried `obs-0016`, obtained a mask, selected it, and produced 20 new
GraspGenX candidates. The read-only advisor recommended candidate 0 with 0.88
confidence as a centered broad-body horizontal grasp, and the Agent explicitly
compiled that new candidate. Because the fixed-camera mask occupied only 0.86%
of the image, it then chose fresh wrist refinement; the run ended after wrist
text-SAM returned no detection and before the corresponding exact-packet point
recovery could be issued.

The terminal cause was provider exhaustion, not an Agent clarification or a
tool-contract failure: the fallback timed out, the primary remained overloaded,
and the final fallback attempt timed out (`provider_attempts=3`). This process
had been launched before the durable failure-classification patch was loaded,
so its stored report incorrectly says `human_intervention`. Response summaries
now retain a bounded provider-failure payload, and future evaluation workers
classify this exact terminal action as retryable
`infrastructure/provider/planner_provider_request_failed`. Evaluation attempts
are set to two. The 180 s timeout is per provider attempt; with three alternating
attempts, a planner turn can legitimately take more than 360 s, so a separate
whole-decision deadline remains an explicit follow-up rather than being confused
with the socket/request timeout.

The final r25 contract audit passed all twelve quality checks with no errors.
Its only warning was the expected
`probable_grasp_slip_after_lift_probe`, which the Agent correctly diagnosed and
recovered from. Typed toolchain extraction gave 100% coverage for both the
13-milestone initial pick/probe chain and the eight-milestone failed-pick to new
wrist-refinement chain. Evidence is retained in
`tmp/r25-tool-contract-audit-20260820.json` and
`tmp/r25-toolchain-coverage-recovery-20260820.json`.

Focused regression after the seed-selection/audit change: `196 passed`. After
provider defaults, durable failure evidence, and the final audit tests, full
repository regression is `1299 passed, 12 skipped`; `git diff --check` also
passes.

### Systematic Mink r26: estimator-order ownership and IK authorization integrity

The r26 canary (`systematic-harness-grasp-backend-preference-20260820-r26`,
session `29aad5c1-c3bf-4dbb-912c-71745e19bacd`) exercised the new
Agent-controlled estimator-order contract on LIBERO object task 7. The host still
owns source RGB-D, mask, calibration, and epoch provenance; the Agent may supply
only a validated `backend_preference` list. The facade retains the remaining
configured estimators as fallbacks and returns an explicit
`openeta.grasp_backend_policy.v1` receipt containing requested, configured,
effective, unavailable, and selected backends. This is an attempt-order choice,
not a host-selected recovery stage.

Memory Bank preflight was healthy (`libero`, 40 objects). AnyGrasp was externally
unavailable at its configured endpoint and Contact-GraspNet was not configured,
so the default call selected GraspGenX and returned 20 candidates. The external
availability gap prevented live coverage of a physical failure followed by an
Agent-requested estimator switch. The Agent correctly did not invent such a
switch before it had physical outcome evidence.

The 48-turn run did provide useful adjacent-tool coverage:

- the Agent compiled a GraspGenX candidate, proposed a calibrated wrist
  viewpoint, checked the exact full pose, and reached it;
- a MolmoPoint request timed out after about 149 seconds with structured
  `mcp_timeout` diagnostics, after which the Agent recovered through fresh
  scene-view segmentation instead of looping on the unchanged wrist packet;
- a clipped wrist mask was not forwarded to grasp estimation; the Agent requested
  another target-facing viewpoint;
- later wrist evidence produced a usable estimate and several candidate/IK
  comparisons, but no close or lift occurred before the turn limit. The terminal
  result was therefore `agent_failure/max_turns`, not a simulated task success.

The original post-hoc auditor missed the most important contradiction in this
rollout. Five full-pose IK checks returned `full_pose_infeasible`, yet each also
included a `motion_execution_ref` and text telling the Agent to pass that receipt
to `move_to`. The motion gate still rejected such authority, and the Agent did
not execute the failed poses, but the response encouraged materially identical
retries at tool sequences 48, 50, 66, 68, and 72.

IK results now carry `openeta.ik_execution_authorization.v1`. A receipt authorizes
`move_to` only when endpoint IK is feasible, or when endpoint kinematics are
feasible and collision checking is explicitly deferred to a verified
collision-owning controller. Rejected IK remains durable evidence through its
`ik_receipt_id`, but exposes no `motion_execution_ref`; its response says to
change the target pose, orientation policy, or candidate. Omitting a tolerance
while preserving the same xyz and explicit orientation is called out as a
non-repair. This constrains evidence causality without introducing a task-phase
state machine.

The generic auditor now checks the same invariant through
`ik_receipts_expose_consistent_execution_authority`. Re-auditing the immutable r26
rollout fails the contract on exactly those five historical calls; evidence is
retained in `tmp/r26-tool-contract-audit-ik-auth-20260820.json`. Focused IK proxy
and authorization-audit regression after the repair: `7 passed`. Full repository
regression: `1308 passed, 12 skipped`.
