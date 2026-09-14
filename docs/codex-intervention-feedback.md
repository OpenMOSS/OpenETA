# Intervention feedback contract

Every Host, adapter or controller operation that changes how an Agent request
is interpreted, executed or allowed to continue must leave an Agent-visible
receipt. This includes successful execution under altered settings, not only
errors. The requirement applies across the system; the implementation and audit
below cover the isolated Codex plugin ingress and its current Mink execution path.
They do not certify all unrelated backends in the main harness.

## Required meaning

For each intervention, explain the requested operation and resolved difference,
the reason and supporting evidence, whether physical execution started, the actual
outcome, and what the Agent can do next. Report applicable budgets/settings and
remaining route segments. A generic error or successful tool transport does not
establish target arrival. An unavailable diagnostic must remain unknown.

Native replies add `current_request` (session-local sequence and tool) and an
`interventions` array. Entries use `source`, `effect`, `reason_code`, `message`,
and optional `execution_state` and allowlisted `details`. Existing motion,
robot, episode-limit, contact and repair receipts provide the supporting state.
Configured baseline settings are labeled `configured`; actual profile changes,
step reduction and joint-limit projection are labeled `adjusted`.

Aggregate failure states distinguish `not_started`, `partial` and `unknown`.
Any unknown physical stage dominates the aggregate; otherwise any executed
stage means partial progress when the request fails. These states describe the
current request, not completion of a previous transport-unknown action.

```json
{
  "source": "host",
  "effect": "substituted",
  "reason_code": "fresh_observation_required",
  "execution_state": "not_started",
  "details": {
    "requested_stage": "gripper_control",
    "executed_stage": "observe"
  },
  "message": "The previous mutation returned no fresh observation. Inspect the refreshed scene and submit a new plan."
}
```

## Audited boundaries

| Boundary / intervention | Agent-visible receipt |
|---|---|
| Inactive episode, native budget, schema or internal contract rejection | Current-request error and intervention; no stale motion feedback; episode state and limits; validation details where available |
| Interface-profile or evidence-bundle gate | Stable gate reason and recovery guidance; filtered repair hints for both native profiles |
| Forced observation refresh or stop after unknown transport outcome | Requested/executed stage, specific reason, execution state, no automatic replay; unknown remote completion requires environment retirement |
| Surface measurement, depth edge, stale or out-of-envelope contact mark | Existing typed measurement/contact errors; measured-to-target distance and allowed envelope for the latter; fresh measurement or shorter approach advice |
| Direction normalization and orthogonalization | Input and resolved world directions when changed |
| Parallel-jaw equivalent orientation selection | Existing selection receipt plus intervention; selected target and endpoint-only scope |
| Loaded reorientation horizon | Original/selected horizon and load-state reason |
| Fresh target / IK preflight | Existing stage receipts and failure recovery; IK acceptance is not controller completion |
| Controller speed profile, projected velocities and bounded step reduction | Reported speed, horizon, tolerances, profile, reduction counts/minimum scale; baseline versus changed settings distinguished |
| Endpoint / pre-actuation / post-step collision, joint or tracking rejection | Existing reason/stage, actual endpoint, candidate versus measured evidence; bounded obstacle and corridor classes |
| Legal but stalled QP motion | Last pre-actuation binding clearance classes and requested/commanded joint-speed maxima, plus measured end contacts; missing evidence explicitly unavailable |
| Gripper actuation and contact-binding retirement | Measured aperture/contact feedback; invalidation reason and fresh-mark requirement; no claim of retention from contact alone |
| Multi-segment route interruption, cancellation or budget exhaustion | Completed count, stopped index, remaining indices, per-segment receipts and aggregate execution state; no execution of later segments |
| Official success claim rejected | Explicit reason and official evidence status; tool success is not task success |
| Reply recovery and journaling | Per-request reset before early gates, public native journal, explicitly historical `episode_status.previous_request` including repair hints |

## Evidence and privacy

Stall context is projected through simulator response filtering, runtime motion
summaries and native feedback. Only bounded robot-part/contact-target relation
classes and control diagnostics cross these boundaries. Private geometry IDs,
object coordinates, witness points, constraint matrices and goal predicates do
not. Direction/distance feedback comes from the Agent's own requested or measured
points; it does not reveal hidden scene geometry.

A binding QP row is evidence of an active constraint at the last control tick,
not a proven cause of the whole stalled trajectory. The receipt explicitly sets
`causal_attribution=not_established`. Actual final contacts are a separate field.
The `requested_joint_speed_max_rad_s` value is the QP's proposed arm speed before
projection and step scaling; it is not an Agent-specified joint command.
`collision.detected=false` denotes no hard-stop trigger; it is not a declaration
of contact-free motion. Diagnostic failure cannot replace the known execution
outcome or authorize a previously blocked move.

This change does not modify motion control, contact grants or safety thresholds.
It does not introduce counterfactual actions, automatic retries, a global planner
or an oracle tool. It reports existing decisions and narrowly adds read-only
stall evidence.

## Delivery limits and integration

Feedback cannot be delivered over a broken connection. The native journal helps
operator diagnosis; `episode_status` can recover the most recent non-status
request receipt if the Host remains reachable. This is not a durable operation-ID
protocol, an exactly-once guarantee or permission to retry unknown actions.
Subsequent non-status requests replace that stored receipt. Cancellation still
stops the next segment rather than preempting an already running worker tick.

These are additive experimental interface changes. Three-person contract review
is required before integration into main. Validation and saved-state replay
evidence are recorded in [the diagnostic report](diagnostics/codex-feedback-contract-2026-09-13.md).
