# Episode guidance and postprocessing budget boundaries

2026-09-07, R0 ninth batch with tenth-batch tool-admission follow-up. Personal-branch
candidate, not a full unified budget implementation. New guidance usage fields,
tool-admission counters/gate, timeout events and review failure/skip metadata
require three-person review before integration.

## Guidance no longer runs outside the timed episode path

Previously, the timed planner/action worker returned before `InteractionResolver`
ran synchronously on the runner thread. A slow resolver could therefore keep
`run()` waiting past the episode deadline. The existing review recorded a
0.04-second episode taking approximately 0.128 seconds with a 0.12-second resolver.

The runner now waits for guidance only within the **remaining** episode budget:

- It checks resource exhaustion before guidance dispatch, after preparing the
  detached context, and before committing a returned answer. A new turn is not
  granted a fresh full episode timeout. Guidance context reports the actual
  completed turn index, not the following turn.
- The resolver receives deep-copied memory/observation data and runs in a tracked
  worker. Only the runner can record the resolution, charge usage, or write a
  guidance answer. The commit also checks cancellation and session generation.
- On deadline, the runner marks the episode truncated with `episode_timeout`,
  requests the existing bounded cleanup, and returns the already-completed
  planner/action step. It does not switch to `waiting_for_human` and freeze the
  expired budget. A late result or exception has no memory-commit path.
- The tracked worker remains visible to `wait_for_idle`. Starting another episode
  on that runner is refused while the previous tracked worker is still running.
  This guard does not cancel an external request or prove remote physics stopped.
- A timely ordinary resolver failure still falls back to human input. Existing
  provider-queue timeout propagation is preserved. Explicit human wait accounting
  remains separate; model guidance time is not subtracted as operator pause time.

Timeout event example:

```json
{
  "question": "Which cube?",
  "worker_pending": true,
  "usage_known": false
}
```

This is the payload of `guidance_resolution_timed_out`, not a provider cancellation
acknowledgement. Context preparation and local result serialization are synchronous;
the existing cleanup wait may add up to `INTERRUPT_CLOSE_GRACE_S` (0.25 seconds).
This is not a strict real-time bound on the entire `run()` call.

## Guidance usage is retained and charged

`BackendGuidanceResolver` copies `usage` and `usage_source` from a successful
backend result into resolution details. Both answers and abstentions consume the
episode token budget. `token_usage_sources` records call counts under the
`guidance:<source>` label, matching the existing source-accounting convention.

```json
{
  "usage": {"total_tokens": 12},
  "usage_source": "provider"
}
```

Counts must be non-negative integers (not bool, non-finite float or numeric text).
When a valid total is absent, two valid explicit prompt/completion counts can be
summed. Missing/malformed usage is not charged as a fabricated receipt: recorded
totals are then incomplete, not proof the call was free. Timed-out requests and
invalid backend responses may still incur provider cost without a returned usage
receipt. No result arriving after cancellation retroactively mutates an emitted
episode report. Full provider-side cost reconciliation remains open.

## Post-episode review cannot erase the task result

Follow-up (R0 eighteenth): catching an exception alone did not uphold this
boundary. The callback received the original `EpisodeResult` and could clear
steps or change outcome/metadata even before raising; its retained report could
also mutate published metadata later. Reproduction covered all three return
paths (success, exception, malformed report) and retained references.

The runner now passes a deep-copied result into review and takes an independent
copy of its returned report. Review still sees the complete result content, but
no longer owns those task/step/metadata references. Before publishing the review
event, the runner checks session generation, memory identity and execution ID
under the memory commit lock. A session replacement—including reopening the same
session ID—prevents insertion into the replacement session's trace. The old
returned episode retains its report with:

```json
{
  "memory_publication": {
    "recorded": false,
    "reason": "session_or_episode_changed_during_review"
  }
}
```

This publication marker is a local metadata candidate requiring collaborator
review. It does not mean staged proposals/skill changes were rolled back. Copying
is synchronous and is not a memory-size/time budget or an arbitrary-Python
sandbox: reviewer code holding other runtime references can still have independent
side effects. The live skill registry remains the explicit application target;
filesystem writes and approved auto-apply have not been moved into an abandoned
background worker. Input/report alias isolation is a prerequisite for further
compute/commit separation; it did not itself implement a review deadline. The
following R0 nineteenth-batch path now bounds supported preparation, not commit.

Regression: `tests/test_review_result_isolation.py`, alongside existing review
and guidance tests. No provider or live simulator is involved.

- A budget/interrupt stop does not start new automatic self-improvement review.
  The report records `reviewed=false`, empty returned proposals and trigger reason
  `episode_budget_or_interrupt_stop`. It is not a successful review or automatic
  deferred job; review can be requested separately after recovery.
- For an otherwise eligible episode, review exceptions and malformed reports are
  returned as `self_improvement_review.error`, preserving the task steps/outcome.
  `partial_effects_possible=true` explicitly warns that earlier proposal writes or
  an already approved application may have happened. An empty returned list does
  not prove no files were written; no rollback is claimed.
- The whole review still has no hard deadline/token budget. The supported path
  below separates preparation from commit; legacy custom callbacks remain
  synchronous because abandoning their combined compute/write methods is unsafe.

## Supported review preparation has its own deadline

R0 nineteenth batch splits `SelfImprovementReviewer` into `prepare_review` and
`commit_review`. `BackendReviewedSkillAutoApplier` similarly separates author/
independent reviewer calls from skill-file/registry updates. The old `maybe_review`
and `apply` methods remain synchronous compatibility wrappers.

The episode runner uses this split for the default reviewer with no auto-applier
or the built-in backend-reviewed applier. Preparation gets detached episode,
context/proposal and skill data. It creates no framework proposal or skill files.
A virtual skill registry preserves the order of multiple prepared updates without
mutating the live registry. Preparing an application error retains that error for
the subsequent proposal report; deadline/session cancellation instead stops the
whole preparation and does not commit its proposals.

`SelfImprovementConfig(preparation_timeout_s=120.0)` is the independent Host default;
positive finite numbers are accepted, bool/NaN/inf/nonpositive values rejected.
This is a configuration API, not a new Agent input or CLI flag. The runner waits
on a tracked preparation thread, checks expiration/session identity between
subagent stages, and rechecks under its memory commit lock immediately before
publishing the plan. Expiration returns the existing task outcome with review
error `ReviewPreparationTimeout` and, for example:

```json
{
  "preparation_budget": {
    "timeout_s": 120.0,
    "completed_in_time": false,
    "worker_pending": true,
    "commit_started": false
  }
}
```

Late preparation has no runner commit callback. Its thread stays visible to
`wait_for_idle`; a new episode cannot reuse that runner until it exits. After a
late author response, no new independent-review stage is dispatched. Timely
preparation reports `completed_in_time=true`; only the runner performs commit.
The built-in applier checks the current registry skill against its prepared
snapshot, including editability, before writing. Reviewer config/applier/store
replacement also rejects the old plan before proposal writes.

Limits remain explicit:

- Local copying, serialization and commit/file I/O are synchronous. The deadline
  is not a real-time bound on the entire run or an atomic transaction across
  proposal files, skill registry and task-playbook extraction. Partial commit
  failure is not rolled back. External file edits not reflected in the registry
  are not covered by the registry snapshot comparison.
- Provider requests and their internal retries cannot be forcibly cancelled;
  they may remain active until their own timeout. Preparation token usage and
  cancelled/late provider cost still need separate reconciliation.
- Custom reviewer subclasses or opaque custom auto-appliers retain the original
  synchronous path and do not receive a `preparation_budget` success claim. Custom
  subagents with the context-only review protocol receive detached inputs, but
  this is not a sandbox against arbitrary Python using its own external references.
- Prepared plans are Host-private values, not signed Agent authorization. The new
  phase methods, configuration and review metadata require collaborator review.
  Do not bypass the runner's ownership/deadline checks when using them directly.

Regression: `tests/test_review_preparation_budget.py` covers proposal/author/
reviewer timeouts, tracked-worker reuse blocking, late-result suppression,
generation changes, stale skills, configuration changes, timely commit and legacy
compatibility. These are local fixtures, not a real model/provider cancellation
or task experiment.

## Review proposal publication follow-up (R0 twentieth)

Review preparation does not itself authorize overwriting an existing proposal.
`SkillReviewProposalStore.save()` now validates the same filename boundary as
load, requires a canonical unsuffixed ID, and atomically publishes **only if the
destination does not exist**. Re-saving an existing pending/approved/rejected
record raises `FileExistsError`; it never resets that resolution or changes the
stored proposal. Replaying a committed preparation therefore stops at save before
invoking the auto-applier again.

Both creation and status updates use exclusive random temporary files instead of
a predictable `.json.tmp`. The complete JSON file is flushed/fsynced before
publication; creation uses an atomic hard link, updates use replacement. Failed
publication cleans up only its own random temporary file. Existing symlinks and
unrelated files are not followed or deleted through the old temporary name.

Loads require a regular non-symlink file, compare the opened file identity with
the pre-open stat, and require the embedded `proposal_id` to match the filename.
The returned `path` is derived by the store, not trusted from stored JSON. Save
rejects path separators, empty/dot IDs, control characters, overlong IDs and
noncanonical whitespace/`.json` suffixes before creating the store directory.
Load continues to accept the ordinary `fixture.json` filename alias.

This changes storage behavior and needs collaborator review. Existing invalid or
identity-mismatched records are not renamed/repaired automatically. Atomic
creation requires hard-link support and fails without overwriting if unavailable;
the regression suite also ran under the actual repository filesystem. Directory
ancestors/configured roots are trusted; this is not a directory-rename adversary
sandbox or a complete crash-durable journal (no parent-directory fsync guarantee).
Status approval/rejection still lacks a multi-process compare-and-swap transaction
with skill application. Multi-proposal commit can leave an earlier prefix saved
when a later proposal fails; no rollback or automatic partial-commit replay is
claimed. See `tests/test_review_store_boundaries.py`.

## Executable tools reserve quota before dispatch

R0 tenth batch adds `ToolCallBudget`, a Host-owned, locked ledger shared by a
runner's scoped tool calls. Before a registered executable tool enters its
authorization gate or handler, `ToolRegistry.call` reserves one slot. Failure,
supervision rejection, or cancellation after reservation does not refund it.
An already-cancelled call is not admitted; cancellation during authorization is
checked again before entering the handler. Missing handlers/unknown tools remain
legacy failed requests, not fabricated successful admissions.

At exhaustion, the registry returns `tool_call_budget_exhausted` with
`dispatched=false`. Agent parameters cannot raise the quota, and nested caller
metadata/scopes cannot remove the current Host ledger. Synchronous nested calls
inherit it even when the handler runs in the registry's cancellation worker.
Cooperative cancellation markers remain intact, including the Python sandbox's
requirement to reap before releasing execution ownership.

- Read-only batches reserve per call. An oversized batch returns earlier results
  plus explicit denied entries; it is not all-or-nothing and no rollback is claimed.
- An exactly exhausted quota still permits a terminal response or human pause.
  A subsequent attempted executable tool is refused and stops the episode with
  the existing `tool_call_limit_exceeded` failure code. Hidden nested denial also
  stops the episode even if its outer handler returns success.
- Existing `tool_call_count` retains requested/failed/blocked attempt accounting,
  with a conservative floor equal to carried usage plus actual admissions. The
  floor prevents nested calls from disappearing when pause/resume persists this
  legacy counter. It is not a pure handler-dispatch count and may exceed the limit
  when the extra attempts were denied before dispatch.
- `usage.tool_admission` and quota-failure `admission` expose the actual ledger.
  `admitted_this_run` counts admitted attempts, including authorization rejection,
  **not** successful motion, physics steps, or task progress. `carried_usage` is
  conservatively charged from the prior attempt count, not reconstructed historical
  dispatch evidence. Failure `observed` includes denied admission requests, so a
  nested quota failure is not reported with a misleading below-limit count.

Example after requesting three executable calls with a limit of two:

```json
{
  "limit": 2,
  "carried_usage": 0,
  "admitted_this_run": 2,
  "denied_this_run": 1,
  "remaining": 0
}
```

The ledger is process-local and bound to the execution scope. Direct simulator
calls, environment reset/cleanup, model calls inside a handler and unrelated
registries/raw threads without the Host scope are not counted as new admissions.
Host cleanup remains possible after quota exhaustion; this does not give the
Agent a quota bypass. A single `move_to` admission can execute many controller
steps, which still need their separate finite budgets. This ledger is neither a
distributed execution lease nor a sandbox against arbitrary trusted host code.

## Simulator RPC timeout is independent of provider waiting

R0 fourteenth-batch follow-up (2026-09-07) fixes the CLI and batch entry points'
`max(300, provider.timeout_s)` coupling. With a Human VLM provider timeout of
86400 seconds, that expression also allowed an individual simulator RPC to wait
86400 seconds. Increasing a human/model response deadline must not implicitly
increase the simulator deadline.

The Host now selects an independent positive finite timeout, default 300 seconds:

- `openeta`, `openeta-batch`, and experiment `preflight/run/iterate` parsers accept
  `--simulator-timeout-s`. Experiment execution forwards it to the shared worker
  factory; preflight's separate `--mcp-timeout-s` remains its discovery timeout.
- CLI constructor/state and `build_mcp_episode_worker_factory` accept
  `simulator_timeout_s`. Invalid values, including booleans, zero, negatives,
  NaN and infinity, are rejected before runtime/provider/workspace construction.
  Changing provider config or rebuilding the CLI runtime does not change it.
- Batch environment create/reset/render and simulator tool proxies receive the
  same limit. CLI simulator tools receive the selected limit. Catalog discovery
  retains its separate short timeout; close uses `min(simulator_timeout_s, 30)`.
- All five Human VLM launch scripts pass `OPENETA_SIMULATOR_TIMEOUT_S` (default
  300) explicitly to the CLI. This does not change their historical episode,
  provider or model-token defaults.
- Explicit batch resume accepts the same flag/API argument. This Host value is
  **not** added to persisted paused records in this batch: re-supply a non-default
  value when resuming; otherwise the independent 300-second default applies.
  The parameter does not restore an old RPC's deadline or prove its completion.

Example: `--simulator-timeout-s 45 --episode-timeout-s 600` means a 45-second
simulator execution RPC limit and a separate 600-second episode budget, not an
extra 45 seconds after that episode budget. The current runner may stop waiting
before the transport times out; unified remaining-deadline propagation and
remote cancellation/expired-request rejection remain open. A timed-out motion
still follows the [unknown-operation stop rule](motion-outcome-safety.md).

This is a Host configuration/API candidate requiring integration review, not a
new Agent tool parameter. No claim is made that all handlers, perception calls,
provider requests, shutdown operations or controller steps share one hard budget.

## Remaining acceptance

Still open: a shared deadline/cancellation context across all model/tool clients,
provider cancellation, all advisor usage accounting, independent review budgets,
and hard token admission. Token ceilings remain post-result checks and may be
exceeded by one model call. Legacy tool-attempt accounting also remains distinct
from the new pre-dispatch admission ceiling. Human VLM guidance requests may remain
pending remotely after local timeout; parent/child cancellation orchestration
remains open.

Tests exercise blocked real threads, detached-context mutation attempts, late
answers, actual pipeline/runner flow, generation changes, usage, postprocessing
errors, concurrent quota contention, nested calls, batch exhaustion and pause/resume
with local fake providers. They do not prove live provider cancellation or
real LIBERO experiment readiness. See [milestone](experiment-ready-milestone-2026-09-06.md)
and [progress log](harness-refactor-progress-2026-09-05.md).
