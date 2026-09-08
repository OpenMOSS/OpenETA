# Transport-unknown motion: stop and retain the fence

2026-09-07, R0 eighth batch. Personal-branch candidate; retained memory fields,
gate compatibility and planner stop behavior require three-person review. This
is a safety correction, **not** acceptance of in-place timeout recovery.

## Reproduced defect

An in-memory diagnostic recorded a failed `move_to` with `motion_outcome=unknown`,
then supplied an observation whose EEF XYZ matched the requested target. Before
this fix, reconciliation became `completed` and the next-action gate opened,
although no remote operation-completion evidence was supplied. Three stable
target misses could similarly become `failed` and unlock the gate. Aperture-only
gripper agreement had the same operational-completion ambiguity.

Separately, MCP observation previously bypassed the complete per-handle control
lock. The worker's observation lock protects an individual physics step/render,
not the whole OSC loop: an observation could occur between steps while the
outstanding command continued afterwards.

Neither matching position nor apparent stillness proves a remote controller has
exited. Even observing after currently executing control exits cannot fence a
request that has not yet arrived at the server.

## Current behavior

- `observe_env` uses the same process-local, session/handle lock as complete MCP
  control calls and cleanup. It waits for current same-handle control to exit,
  rejects closing/close-failed/closed resources, and leaves other handles
  independent. Dashboard render/stream and direct worker endpoints are not
  promoted to this barrier.
- Any retained motion-reconciliation record fences Agent tools except `observe`.
  Old snapshot `completed`/`failed` values do not bypass it. A same-handle
  observation updates position diagnostics but keeps operational status unresolved.
- First observed position agreement advances the conservative motion epoch once;
  repeated matching observations do not repeatedly advance it. Gripper agreement
  still does not invent a latch, proxy or attachment PASS.
- The tool-call planner stops the current episode using the existing terminal
  `response/talk` path. This takes precedence over forced observation refresh,
  avoiding an endless observation loop. It does not report task success or accept
  human/model reassurance as an operation receipt. Manual read-only inspection
  remains available through the existing pipeline.

Example host fact value after a matching observation (additional provenance and
measurements omitted):

```json
{
  "tool": "move_to",
  "status": "unresolved",
  "position_reconciliation_status": "completed",
  "remote_completion_verified": false,
  "position_evidence_epoch_advanced": true
}
```

`position_reconciliation_status` preserves the old position-only diagnostic
vocabulary; `completed` there does not certify orientation, controller termination,
trajectory completion, or task success. `remote_completion_verified=false` is an
explicit limitation, not an implemented protocol or a writable unlock switch.
Existing `resolved_at_s` is a position-diagnostic timestamp, not a remote end time.

## Operator recovery boundary

Stop the affected episode; do not resend the action, reset the same unresolved
environment, delete its memory gate, or treat a new session ID as proof that old
physics stopped. The current implementation has no in-place unlock path. Use the
Host's [confirmed cleanup procedure](environment-cleanup.md), retaining the same
remote identity if cleanup fails. Only after confirmed retirement should a new
environment and session be started. Cleanup alone does not mutate the old session's
reconciliation fact. No automatic service restart or user-handle retirement is
performed by this planner response.

## Remaining design and validation

Proposed next boundary, **awaiting user design confirmation and collaborator
review**: Host-private operation IDs, terminal-state queries and rejection of
expired requests, initially scoped to single-process MCP. The Agent should not
fill these transport fields. Completion must be bound to the original operation
and environment lifetime before any in-place gate release. This is not a claim
of full distributed leases, crash recovery or fencing every worker entry point.

Regression tests cover real production memory/pipeline code with synthetic
observations, persisted legacy snapshots, bounded concurrent fake control, and
actual asynchronous lifecycle execution. They do not run a live timeout, late
dispatch, real robot or Human VLM task. The experimental milestone still requires
bounded live recovery/stop-and-cleanup validation on the deployed service version.

Entry points: `tests/test_motion_reconciliation.py`,
`tests/test_environment_lifecycle.py`, `tests/test_gripper_evidence.py`.
