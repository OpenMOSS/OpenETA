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
3. `recent_transitions`: bounded action, observation, receipt, and interaction
   events;
4. `world_evidence`: selected targets, tool-produced candidates, gripper state,
   checker evidence, reconciliation evidence, and trusted environment receipts;
5. `open_questions`: unresolved perception or semantic-selection evidence;
6. `agent_working_state`: Agent-owned plans, hypotheses, and notes;
7. relevant skill guidance and executable atomic tools;
8. operational safety and observation-freshness constraints.

Task-policy objects such as `grasp_execution.stage`, fallback stages, and
`required_action` are excluded from the model-facing projection.

Production also uses a dedicated Agent-owned system prompt. It tells the model
to infer and persist its own plan, inspect current visual evidence before stale
memory, and choose grasp/recovery steps from evidence. The legacy prompt remains
available only when `host_task_policy_enabled=True`; this prevents old prompt or
skill wording from silently recreating a task state machine after the context
projection has removed it.

Each attached current image now carries a stable evidence id, camera frame and
role, observation step, optional timestamp, and `freshness=current`. The provider
message presents labelled images before the larger JSON decision context.

## Memory ownership

Session trace remains append-only evidence. Working memory is split into:

- `facts.json`: legacy runtime facts retained during migration;
- `agent_working_state.json`: facts written through Agent `save_memory`, marked
  `ownership=agent` and `freshness=agent_managed`;
- `artifacts.json`: stable tool-result and image references;
- `skill_notes.json`: Agent-authored notes scoped by skill.

Old sessions without `agent_working_state.json` are migrated in memory by
selecting legacy facts whose source is `save_memory`.

When task-state tracking is disabled, legacy task-policy facts may remain on
disk for audit but are projected as inactive: they are absent from planner and
supervision context and cannot gate a new action. Fresh grasp candidate lists
remain available as tool-result evidence derived from artifacts, with source,
timestamp, scene epoch, and freshness, but without an `active_candidate` or
host-selected fallback stage.

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

## Policy migration switches

Direct `ToolCallingPlanner` and `ActionPipeline` construction retain legacy
defaults for tests and old callers. Shared production runtime assembly sets:

- `PlannerContextConfig.host_task_policy_enabled=False`;
- `ActionPipeline.task_execution_gate_enabled=False`;
- `AgentMemory.task_state_tracking_enabled=False`.

This disables host task-action dispatch, task-obligation planner validators,
grasp-stage action gating, and automatic task-stage transitions in the production
path.

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

The motion-reconciliation gate is evaluated even when the legacy task execution
gate is disabled. This keeps unknown-outcome handling independent from the task
state machine: `observe` is allowed, while another world mutation remains
blocked until the same environment is reconciled.

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

## Migration sequence

1. Establish counterfactual visual-state and context-ablation baselines.
2. Compare `agent_context.v2` with the legacy full planner context.
3. Keep production task-policy switches disabled and diagnose any behavior that
   still depends on legacy phases.
4. Replace useful legacy transitions with tool evidence, verifier outputs, or
   editable skill guidance.
5. Delete compatibility state machines after scenario evaluation demonstrates
   that no safety invariant depends on them.

Any change to Section 5 command, observation, tool-result, or checker schemas
still requires three-person review. The context and eval schemas in this document
are Agent-runtime-local and do not replace those shared contracts.

## Bounded visual history follow-up

The model-facing visual context now has a separate design for a deterministic
raw-image window and task-conditioned fixed-main-camera visual deltas. See
[`bounded-visual-history-vdm.md`](bounded-visual-history-vdm.md). It preserves
the immutable trace and current-evidence priority established here; it does not
restore host task phases or required-next-action dispatch.
