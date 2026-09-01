# Agent Context and Memory Refactor

## Purpose

The Stage 2 runtime must let the Agent infer task state from current evidence
instead of receiving a host-authored task phase and required next action. The
runtime remains responsible for safety, authority, execution, and trusted
receipts; task decomposition and recovery sequencing belong to the Agent.

The governing rule is:

> Runtime may reject an action and explain which invariant it violates. Runtime
> must not prescribe the next task action merely to advance a host task phase.

## Production decision context

`ToolCallingPlanner` retains a full internal `openeta.planner_context.v1` for
runtime validation and migration compatibility. The model-facing projection is
`openeta.agent_context.v2`, organized as:

1. `objective`: current user/environment task and latest human input;
2. `current_observation`: compact structured state plus explicitly labelled
   visual evidence;
3. `recent_transitions`: elastic action, observation, receipt, and interaction
   history, projected together with the transition ledger and conversation;
4. `world_evidence`: selected targets, tool-produced candidates, gripper state,
   checker evidence, reconciliation evidence, and trusted environment receipts;
5. `open_questions`: unresolved perception or semantic-selection evidence;
6. `agent_working_state`: Agent-owned plans, hypotheses, and notes;
7. relevant skill guidance and executable atomic tools;
8. operational safety and observation-freshness constraints.

Host task-progress objects are absent from both the model-facing projection and
the runtime implementation. Grasp candidates, compiled anchors, verifier
outcomes, and tool receipts are evidence; none of them names the next task action.

Production uses one Agent-owned system prompt. It tells the model
to infer and persist its own plan, inspect current visual evidence before stale
memory, and choose grasp/recovery steps from evidence. There is no legacy prompt
branch or policy switch that can restore host task sequencing.

Each attached current image now carries a stable evidence id, camera frame and
role, observation step, optional timestamp, and `freshness=current`. The provider
message presents labelled images before the larger JSON decision context.

### Durable history and bounded model projection

The session workspace is the lossless source of truth. Event trace and
conversation records are appended incrementally, rich ToolResults and images are
materialized as immutable artifacts, and only derived working-memory snapshots
are atomically replaced. Building a Planner request never compacts or mutates
that durable history.

The main Planner sees three complementary projections rather than a replay of
the complete trace:

1. **Recent high-fidelity window.** Canonical conversation keeps all real user
   dialogue plus the latest four action/result groups. `recent_transitions` keeps
   the latest three observation turns and intervening recovery feedback; it does
   not duplicate action commands or environment receipts already represented by
   conversation and the ledger. The independently bounded visual-history policy
   still supplies the first main-view anchor, recent main views, current wrist
   view, and VDM bridges.
2. **Compact history ledger.** `transition_ledger` retains the complete compact
   tool/environment timeline without full ToolResult payloads.
3. **Current materialized state.** `current_observation`, `decision_state`,
   `world_evidence`, active bundles, open obligations, and artifact references
   describe what is valid now. They are rebuilt each turn rather than appended
   as another history stream.

Older raw action results, observations, response JSON, and images remain
queryable through the session memory/artifact interfaces. The total token budget
is a final overflow guard around this semantic projection, not the normal
mechanism for deciding which durable records become model-visible.

### Radix-cache-friendly provider layout

The canonical `PlannerBackendRequest.tool_context` remains complete for scripted
backends, rollout recording, replay, validation, and budgeting. The
OpenAI-compatible wire adapter partitions only the main
`openeta.agent_context.v2` serialization into a cache-stable system prefix and a
dynamic final user turn:

1. the Agent system prompt;
2. a deterministic `openeta.planner_static_context.v1` system message;
3. optional conversation summary and canonical growing conversation history;
4. the current dynamic context and labelled vision attachments in the final user
   message.

The stable message contains the full `available_tools` schemas, name-only
`tool_references`, selected `relevant_skills`, `skill_usage`, and the stable
`operational_constraints.rules`. JSON keys are recursively sorted and compactly
serialized so the same tool/skill contract is byte-identical across turns.
Current observation, visual history, transitions, evidence, artifacts,
freshness/reconciliation state, and open questions remain dynamic. A genuine
tool, skill, or execution-rule change intentionally changes the stable prefix and
invalidates the old cache entry.

The partition does not duplicate or remove semantic input: fields moved into the
stable system message are removed from the final user JSON. Isolated reviewers,
VDM, localization, grasp-advisor, and other sub-agent requests retain their
single-user-message representation. Each provider result includes a compact
`openeta.planner_prompt_layout.v1` diagnostic with stable/dynamic character
counts, field names, and the stable-prefix SHA-256, without copying prompt
content into the diagnostic.

## Memory ownership

Session trace remains append-only evidence. Working memory is split into:

- `facts.json`: runtime-owned evidence, epochs, bundles, receipts, and invariants;
- `agent_working_state.json`: facts written through Agent `save_memory`, marked
  `ownership=agent` and `freshness=agent_managed`;
- `artifacts.json`: stable tool-result and image references;
- `skill_notes.json`: Agent-authored notes scoped by skill.

Old sessions without `agent_working_state.json` are migrated in memory by
selecting legacy facts whose source is `save_memory`.

### One total context budget

Durable conversation, event, transition-ledger, artifact, and VDM records are
not shortened merely because they belong to different prompt sections. The
normal planner projection has no fixed 8-event, 4/12-transition, or 8/12-action
window. It first assembles the complete semantically bounded records available
to the session and estimates the combined input containing the system prompt,
Agent context, and canonical conversation.

The persisted transition ledger no longer rolls over at 32 entries, and normal
session resume no longer loads only the latest 64 events. Callers may still ask
for an explicit resume/event limit or an explicit durable compaction checkpoint;
those are opt-in operations rather than invisible production defaults.

When that combined input exceeds the configured fraction of the provider's
context window, after reserving the main Agent's output allowance, the host
removes the oldest elastic entries until the prompt fits. It removes redundant
event summaries before the compact transition ledger and conversation action
groups. Initial and current user instructions are protected ahead of old
action/result pairs. This projection never mutates append-only session history.
`context_budget.projection` records the initial/final token estimates and exact
per-source drop counts. Because provider image tokenization varies, the budget
also charges each attached image a configurable conservative estimate (2048
tokens by default) instead of pretending that image paths are the whole visual
cost.

Per-item limits remain only as structural abuse guards: inline/base64 images are
replaced by artifact references, high-cardinality structured outputs are
materialized to disk, and an individual textual tool summary cannot consume the
entire prompt. Raw visual inputs also keep the deliberately bounded
initial-plus-recent window; text-context capacity is not a reason to attach an
unbounded number of images.

RFC impact: `openeta.context_budget.v2`, the elastic-history projection fields,
and the shared 2048-token reasoning-subagent default are interface/configuration
changes. They remain implementation-local until the three-person RFC review
accepts the contract update; no shared RFC text is implied by this document.

Old task-policy facts are deleted by a one-way load migration and the cleaned
working memory is written back immediately. The production runtime contains no
reader, writer, transition function, or compatibility flag for them. Fresh grasp
candidate lists remain available as immutable tool-result evidence with source,
timestamp, object-scene epoch, and freshness. The Agent selects and records its
own candidate rationale; the host does not maintain an active candidate or
fallback queue.

Rank 0 is not promoted into `retained_targeted_grasp` until the Agent explicitly
selects and compiles that candidate. The former `grasp_compile_obligation`,
`candidate_fallback`, and `final_refinable_fallback` paths are physically
removed; legacy markers cannot bypass geometry validation.

### Complete structured outputs and coding access

High-cardinality tool results must not be made inaccessible merely because the
model-facing memory projection is bounded. Successful grasp and placement tools
therefore persist their complete structured result as an immutable, session-local
JSON artifact. Working memory retains the total count, a five-item preview, an
explicit `truncated` flag, and `complete_outputs_artifact` reference.

The `python_exec` AgentTool provides the corresponding coding-agent-style query
surface. Restricted code can list and read the entire current session tree and
can write derived files only below the session sandbox. It has neither network
nor Simulator MCP access: environment lifecycle and world mutation remain behind
stable AgentTools. Consequently `python_exec` has `planning` effect and does not
create a fresh-observation obligation. The Agent should query a referenced full
artifact before escalating merely because a needed candidate is outside the
working-memory preview.

## Physical removal of host task progress

The task-progress state machine has been physically removed rather than disabled.
`AgentMemory`, `ActionPipeline`, and `PlannerContextConfig` have no legacy policy
switches. Runtime contains no grasp execution stage, required-next-action packet,
host-selected grasp/fallback queue, forced hover/align/descend/close/placement
transition, or handler that depends on such a transition.

Compiled pose labels use geometric `waypoint_role` values only. They identify a
clearance, precontact, contact, or alignment reference for provenance and residual
calculation; they are not mutable progress state. Successful-rollout playbooks no
longer extract or publish an action-stage sequence.

The following host invariants remain active:

- automatic fresh observation after a world mutation without a canonical
  observation snapshot;
- observe-and-reconcile after transport-unknown world mutation;
- tool schema, handler authority, and artifact provenance checks;
- semantic selection/candidate provenance gates still needed to prevent an
  action from silently targeting different evidence;
- deterministic safety/checker hooks and supervision;
- official same-episode reward requirement for benchmark completion;
- execution budgets, cancellation, logging, and trusted environment receipts.

Perception selections are evidence, not task phases. SAM3 results are indexed by
an explicit semantic role: `target_object` and `placement_region`. The legacy
`selected_sam3_detection` field remains an alias for `target_object`; the
role-indexed `selected_sam3_detections` projection can retain both masks and
their source observation packets concurrently. A pending or failed placement
selection therefore does not invalidate target evidence. Role mismatch is
rejected at the selection boundary instead of being inferred from task stage.

The motion-reconciliation gate is always evaluated. This keeps unknown-outcome
handling independent from task planning: `observe` is allowed, while another
world mutation remains blocked until the same environment is reconciled.

## World versions, clearance waypoints, and provenance bundles

Freshness now separates two host-owned counters:

- `object_scene_epoch` invalidates evidence whose object-relative geometry may
  have changed. The legacy `scene_epoch` field is a compatibility alias for this
  value.
- `robot_motion_epoch` records EEF, gripper, base, and camera configuration
  changes. A pure `move_to`, trajectory, or lower-body motion advances this
  counter without making a world-frame target pose stale.

Gripper command acknowledgement advances only `robot_motion_epoch`. A close is
not attachment proof and an open is not release proof, so neither command alone
invalidates object-relative evidence. `object_scene_epoch` advances only from a
fresh host observation carrying `object_scene_change.changed=true` (or the
compact boolean equivalent), with an idempotent change id and an auditable
reason. Transport-unknown actions advance motion freshness only after
observation-based reconciliation confirms execution.

`compile_grasp_seed.hover_pose` is an ordinary `waypoint_role=grasp_clearance`
reference. Its role is geometric and does not imply a required successor.
After reaching a useful clearance pose, the Agent should normally use a fresh
wrist RGB-D packet for full target segmentation, targeted grasp estimation, and
candidate compilation. `compute_wrist_alignment` remains an optional bounded
correction, not the default replacement for wrist-view grasp estimation. After
a wrist target selection the host exposes one opaque
`wrist_alignment:<digest>` input bundle; the Planner never reconstructs mask,
depth, camera calibration, measured EEF pose, compiled grasp, or epochs. Its
desired gripper pixel is host-derived by projecting the configured
`eef_to_gripper_center_xyz` through the current EEF and wrist-camera transforms;
the optical principal point is not treated as the gripper location. The result
records `openeta.gripper_center_projection.v1` for calibration audit. Near-field
refinement is evidence-triggered rather than a host task phase: use alignment
when the original approach/orientation/contact depth remain credible and only
lateral contact placement needs correction; use a full wrist SAM3 → targeted
grasp estimate → explicit compile when orientation, surface, or axial contact
depth is uncertain. Neither path silently activates or replaces a candidate.

The opaque wrist bundle is bound to both object-scene and robot-motion epochs and
to target-identity continuity. `compute_wrist_alignment` publishes a nested
`openeta.wrist_alignment_operating_region.v1` receipt covering mask clipping,
epoch freshness, distance to the compiled clearance reference, correction clamp,
and the ordinary residual budget. If any check fails, it returns
`requires_better_view` with no aligned/adjusted poses. The Agent remains free to
choose whether to reposition, re-observe, resegment, compile a new grasp, or stop;
the host does not encode hover/align/descend transitions.

The host also maintains a read-only `openeta.provenance_evidence_graph.v1`.
Compiling a candidate binds its exact host-captured targeted grasp artifact to a
stable evidence id. A placement-region SAM3 selection on the same source image
then produces `openeta.anyplace_input_bundle.v1`. The model sees only:

The first explicitly confirmed target selection also creates an immutable
`openeta.target_identity_anchor.v1` containing its source packet, mask crop,
semantic phrase, and available asset-reference evidence. Later wrist or recovery
localizers must copy the active `identity_anchor_id` and explicitly declare
`identity_relation=same_instance` after cross-view comparison; the host never
silently binds new detection evidence to the old instance. If fresh evidence
shows that the first selection was wrong, the Agent may explicitly request
`replace_misidentified_anchor` with a concrete reason. That creates a new anchor
and leaves prior grasp evidence auditable/superseded rather than relabeling it.
An exact-instance verifier `mismatch` or `abstain` attached to the
same point prompt is a hard identity conflict; a generic class label cannot
override it, and an anchor with an exact verifier match cannot be replaced by
generic semantic evidence. Same-anchor wrist refinement does not supersede the original
compiled grasp provenance.

```json
{
  "status": "ready",
  "bundle_id": "anyplace:<digest>",
  "call_parameters": {"bundle_id": "anyplace:<digest>"}
}
```

`ActionPipeline` resolves that id to the frozen RGB, depth, object mask,
placement mask, intrinsics, grasp candidate, and source packet. It rejects an
unknown/superseded id, rejects AnyPlace inside `tool_batch`, and records the
grasp and placement evidence ids used for the resolution. The Agent-facing
AnyPlace ToolSpec therefore accepts only `bundle_id`; direct full packets remain
an internal handler compatibility surface and cannot be authored by the main
planner. This lets placement inference run before object motion without the old
attachment-stage gate or model-side path reconstruction.

The ToolSpec addition and bundle contract were approved in the three-person
review on 2026-08-14.

## Visual state counterfactual evaluation

`openeta-visual-state-eval` runs isolated, minimal visual probes. A case manifest
groups counterfactual variants under one task and fixed state-label vocabulary.
The request exposes both `allowed_state_labels` and `allowed_next_actions`, and
requires exact controlled values rather than scoring invented aliases.
Metrics cover:

- exact observable state classification;
- allowed next-action selection;
- citation of valid current-observation evidence ids;
- correct state changes across counterfactual image variants.
- resistance to deliberately contradictory facts marked as stale.

Example local inspection:

```bash
uv run --extra dev openeta-visual-state-eval \
  --manifest tests/fixtures/visual_state/scene_counterfactuals.json \
  --list-cases
```

Running the live evaluation sends the manifest images and minimal probe context
to the configured model provider. It therefore requires explicit data-egress
approval. Unit tests use deterministic backends and do not transmit data.

### Recorded 2026-08-13 baseline

After explicit approval, the three-case manifest was evaluated with the
configured AI Gateway using `gpt-5.6-luna`:

- v0 classified all three visual states correctly, cited current evidence for
  every visual fact, and flipped state across the image counterfactual. Its
  aggregate pass rate was incorrectly 0/3 because the scorer constrained
  `next_action` to manifest values that the request had not exposed; the model
  returned reasonable aliases instead.
- v1 exposed `allowed_next_actions` and required exact controlled values. Strict
  evaluation passed 3/3, with 1.0 evidence-citation rate, 1.0 stale-context
  resilience, and 1.0 counterfactual-flip rate.
- Reports are stored in `.openeta_eval/visual-state-baseline-v0.json` and
  `.openeta_eval/visual-state-baseline-v1.json`. Provider API-key fields are
  persisted only as `<redacted>`; no key fragments or base64 images are stored.

This small baseline verifies the context/evidence behavior under one target
visibility counterfactual and a no-image control. It is not evidence of
benchmark-grade manipulation success. Follow-up suites should add before/after
attachment scenes, occlusion, conflicting cameras, and longer memory histories.

## Completed migration and regression boundary

The useful parts of the former implementation have been separated into tool
evidence, verifier outputs, geometry roles, editable skill guidance, and the
retained host invariants listed above. A static regression test scans production
Python and skill files for removed state keys, switches, action-stage labels, and
the deleted fallback tool. Constructor-signature tests ensure a compatibility
flag cannot silently reintroduce the old path.

Any change to Section 5 command, observation, tool-result, or checker schemas
still requires three-person review. The context and eval schemas in this document
are Agent-runtime-local and do not replace those shared contracts.

## Gate and feedback contract remediation (2026-08-14)

The interrupted ABC run exposed a second-order failure: production disabled the
legacy host task policy and task-state tracker, but `ActionPipeline` still called
the reference/selection gate unconditionally. A pending reference localization
therefore behaved like a hidden required-next-action state even in the
Agent-owned path. In one completed B rollout this produced 83 blocked transitions.

The remediation applies six rules:

1. Gates retain only safety, permission, transport-unknown reconciliation,
   fresh-observation, parameter integrity, and evidence-freshness authority.
   Reference localization and pending SAM3 results are open evidence questions,
   not task-order gates. Provenance consumers still cannot consume an unverified
   mask or silently substitute a different selected mask.
2. Every retained pipeline rejection carries `openeta.gate_repair.v1`: the
   violated invariant, rejected call, evidence ids, stale evidence, and concrete
   allowed next calls with host-grounded parameters. A prose-only rejection is a
   contract defect.
3. `host_resolved_inputs.grasp_pose_estimate` and
   `host_resolved_inputs.wrist_alignment` generalize the AnyPlace pattern.
   After target selection, the host freezes the selected mask and its aligned
   source RGB-D packet, intrinsics, camera frame, and object-scene epoch behind a
   `grasp:<digest>` id. The preferred planner call contains only `bundle_id`; the
   explicit parameter form remains temporarily available for compatibility.
4. `agent_context.v2.decision_state` is a bounded operational index, not another
   state machine. It contains only the current packet, active bundle references,
   unresolved hard/open questions, the last action effect, and executable tool
   names. Full traces and artifacts remain queryable outside this projection.
5. `ToolResult v1` now separates `operational_success` from
   `semantic_outcome`. For example, a completed SAM3 request with zero masks is
   `success=true`, `operational_success=true`, and
   `semantic_outcome=no_detection`. `facts_produced` and `recovery_options` make
   this distinction survive context compaction.
6. Runtime assembly lints every loaded skill against the active ToolSpec catalog.
   Unknown allowed tools, imperative references to nonexistent tools, calls
   omitted from `allowed_tools`, and missing bundle-only guidance fail assembly.
   The pick and simulator skills were updated to remove manual provenance copying
   and raw reset instructions.

The ToolResult additions and the optional grasp bundle parameter are additive
implementation candidates. They must be recorded in the shared RFC and receive
the normal three-person schema review before the compatibility parameter form is
removed.

### Interrupted-run baseline for the next comparison

The run `visual-history-abc-20260814-host-provenance-run1` was intentionally
stopped after 9 complete jobs; no partial job was relabelled as a normal failure.
The experiment-specific extractor wrote its baseline to
`.openeta_eval/runs/visual-history-abc-20260814-host-provenance-run1/extractors/visual_history/metrics.json`.
Across the completed jobs it found:

- A/B/C gate blocks: 13 / 84 / 3;
- executable repair bundles: 0 in every variant;
- decision-state coverage: 0 in every variant;
- bundle-based grasp calls: 0; manual grasp input calls: 15 / 9 / 12;
- alternating A/B/A tool cycles averaged 11.67 / 42.0 / 61.0.

The next canary should compare these contract metrics before attempting a long
ABC run. At minimum, every new gate block must contain a non-empty executable
repair bundle, decision-state coverage must be 1.0, and targeted grasp calls
should use host bundles whenever they are ready.

## Bounded visual history follow-up

The model-facing visual context now has a separate design for a deterministic
raw-image window and task-conditioned fixed-main-camera visual deltas. See
[`bounded-visual-history-vdm.md`](bounded-visual-history-vdm.md). It preserves
the immutable trace and current-evidence priority established here; it does not
restore host task phases or required-next-action dispatch.

## Post-canary context and budget fixes (2026-08-14)

The first one-shot A/B/C canary used 40 planner turns and 80 tool calls. That
budget was sufficient to expose control and context defects, but it is too short
to estimate long-horizon task completion: A reached contact and closed the
gripper on its final turn. B instead spent its remaining budget repeatedly
querying the same structured artifact, while C stopped with six turns remaining
after provider exhaustion. Future outcome canaries therefore use 60 planner
turns, 120 tool calls, a 90-minute deadline, and 7.5M total tokens. A 40-turn
profile remains appropriate for cheaper contract smoke tests.

`python_exec` keeps its complete durable ToolResult, while the model-facing host
feedback now includes a bounded projection of `outputs.result`. Recent-transition
compaction treats this result as structured data, so lists of candidate objects
retain their field values instead of degrading to a list of key names. Large
source artifacts remain queryable through the session workspace.

The Object Memory path is explicitly split into retrieval and localization
failures. A service timeout produces `object_memory_retrieval_failed`; a VLM
localization failure produces `reference_localization_failed`. Exact-instance
review rejection still fails closed. When the reviewer only abstains because a
low-resolution simulation crop cannot establish exact geometry, the best
proposal may be returned as a confidence-capped provisional SAM3 point. This
point is not motion authority: downstream SAM3 segmentation and main-Agent
semantic selection remain mandatory before any provenance consumer or world
mutation can use it.

## Bounded transcript, direct calibration, and grasp lineage

The durable conversation remains append-only, but `model_messages()` projects
real user turns plus only the last eight action groups. Tool-result projection
retains `operational_success`, `semantic_outcome`, diagnostics, recovery options,
structured `python_exec` output, and exact artifact paths. Large values are
materialized and indexed rather than silently replaced by key names. A replay of
an actual 106-item milk session reduced the model-facing conversation from about
485k to 61k characters while preserving the complete stored transcript.

The current observation packet includes its exact camera calibration directly.
The Agent no longer needs to search source files to discover ordinary
camera-to-world transforms; source inspection remains available for exceptional
diagnosis through `python_exec`.

The provenance graph now binds a selected target evidence id to each compiled
grasp. A newer current-scene target selection marks an older compiled contact
pose as superseded. The gate rejects only stale contact/close actions near that
pose and returns both old/new evidence ids plus the current bundle repair. Safe
clearance, retreat, observation, and read-only recovery are still allowed. This
is an evidence-integrity invariant, not a required-next-action state.

## Motion acknowledgement versus target attainment

Simulator command execution and physical target attainment are separate facts.
A `move_to` receipt with `reached_target=false` remains
`operational_success=true` because the controller ran and may have changed the
world, but its ToolResult is now:

- `semantic_outcome=target_not_reached`;
- a `simulator_mcp_target_not_reached` or collision diagnostic;
- requested and actual XYZ plus Euclidean position error;
- an exact full-response artifact path;
- recovery options to inspect fresh evidence and replan from the actual pose.

The inline content explicitly warns the Agent not to assume the requested pose
was achieved. This avoids both unsafe false failure semantics and the earlier
ambiguous `success=true` message that encouraged repeated identical commands.
The host does not impose a clearance/contact successor relation. Every later
motion is checked from the actual robot state; a different direct proposal is
allowed when its own endpoint/path evidence is safe.

`ik_preview_check` additionally publishes an
`openeta.ik_preview_receipt.v1` bound to target xyz, orientation policy,
tolerances, and the current robot/object epochs. Results are classified as
`feasible`, `repairable`, `inconclusive`, or `hard_infeasible`. Repairable poses
remain task-space anchors and expose backend residuals/suggestions. Only an
unchanged hard-infeasible pose in the same epochs is replay-blocked; adjusted
xyz, a different orientation policy, a safe intermediate viewpoint, or changed
world/robot evidence receives a new check.

Motion execution uses that receipt as a typed handoff. The planner passes only
`ik_receipt_id` to `move_to`; the host resolves the exact checked pose and
orientation policy from durable working evidence. Exact compiled anchors may be
submitted to IK by `compiled_grasp_id + waypoint_role`, while visual corrections
remain Agent-authored poses at the IK boundary. This removes duplicated matrices
and coordinates from the conversational history without hiding the Agent's
choice or introducing task-progress state.

Compiled grasp execution now uses the same Agent-owned `move_to` primitive for
visual correction; there is no separate micro-adjust tool or host task phase. The
compiled pose is a reference anchor. For a provenance-preserving request, the host
computes the xyz residual, limits each change in residual to 0.02 m, and tracks a
0.10 m cumulative residual-path budget under the compiled grasp id. Comparing
residual vectors rather than absolute hover/contact coordinates means that carrying
the same correction from hover to contact consumes no artificial approach distance.
The current residual, remaining budget, and requested/actual EEF receipt are projected
to the Agent. These values support visual closed-loop reasoning; they are not
privileged object-relative contact truth.

## Attachment collision envelope and placement evidence reuse

The gripper command is a binary latched actuator (`0=closed`, `1=open`). The
planner context separates that command from `measured_aperture.open_fraction`;
the legacy threshold flag is compatibility data and never attachment proof.
After a non-empty close, the simulator safety adapter creates a tentative object
proxy and promotes it only after post-close EEF/object co-motion. A confirmed
proxy participates in carry collision checks as an EEF-relative conservative
box. Receptacles expose an interior placement corridor so centred insertion is
allowed while rim overlap is rejected with the attached object, obstacle, and
predicted pose in the diagnostic. Privileged simulator object geometry remains
inside the safety adapter; the Agent receives images, ordinary robot feedback,
and the resulting safety verdict. A real adapter can populate the same proxy
contract from segmentation/depth evidence.

New targeted-grasp perception no longer deactivates active provenance or a
materialized AnyPlace plan. It remains an immutable proposal until the Agent
explicitly compiles a candidate. When that switch requires a new AnyPlace
bundle, a selected placement-region mask may be reused across source images only
on the same fixed scene camera with identical intrinsics; wrist/hand cameras are
always refreshed. RGB-D and object-mask inputs still come from the new grasp
source. A mismatch that cannot be safely reused exposes an exact executable
`repair_call` rather than prose-only path guidance.

Evaluation turn budgets keep a fixed base limit. A distinct post-close grasp
branch switch can add a manifest-controlled allowance, bounded by a hard total;
the current canaries use 60 base turns, 12 turns per recovery branch, and at
most 36 recovery turns (96 total).

## Agent-first canary results

The milk task was not sufficient to isolate Object Memory because its target was
small and downstream grasp estimation repeatedly failed. A clearer salad-dressing
task therefore exercised the same harness. The recorded Luna and
provenance-verified Sol runs show bounded prompts, direct use of structured motion
feedback, correct stale-evidence recovery, and no host task-phase dependency.
Neither model consistently varied away from regenerated rank-0 grasp candidates,
and contact control commonly missed by 3–4 cm. These are now treated as grasp and
control quality problems rather than evidence that the Agent lacks task state.

The nominal first Sol run is explicitly invalid as a model comparison because
provider failover executed Luna. Full run-by-run metrics, corrected provider
identity, and conclusions are in
[`agent-first-canary-2026-08-14.md`](agent-first-canary-2026-08-14.md).

## Semantic evidence projection and session tool health

Long recovery sessions keep complete receipts, candidates, transition rows, and
ToolResults in durable memory and rollout artifacts. The planner projection does
not need to repeat every full structure on every turn. It now exposes:

- the latest full IK receipt plus a compact index of prior receipt ids, pose
  signatures, epochs, classifications, and target anchors;
- the grasp advisor recommendation, alternatives, and the corresponding visible
  candidates, with the complete candidate bank referenced by its artifact;
- all semantic tool transitions but only the latest repeated zero-reward
  environment receipt;
- a compact recovery outcome rather than a second copy of the full ToolResult;
- semantic structured-tool outputs in conversation history, while preserving
  direct `python_exec` results for coding-style inspection.

This is a model-facing projection only. It does not mutate durable history and
does not change gate authority. On the r48 final-turn semantic request, the
projection reduced Agent-context serialization from 319,175 to 203,670
characters (36.2%) and reduced `world_evidence` from 119,111 to 28,683
characters.

Infrastructure failures are tracked separately from task-semantic failures.
Timeout, transport, connection, or service-unavailable receipts mark a tool
`degraded` after one consecutive failure and `circuit_open` after two; a later
successful receipt restores `healthy`. `decision_state.tool_health` is advisory
evidence, not a gate. It tells the Agent not to repeat an unchanged call until a
preflight or explicit health change justifies retrying.

A wrist-view segmentation miss is likewise projected as optional refinement
failure. It does not invalidate a current-epoch scene-view compiled grasp by
itself. The Agent may inspect the fresh dual view, continue from the retained
anchor, apply a bounded visual residual adjustment, or pursue same-view point
grounding when the extra refinement is actually needed.
