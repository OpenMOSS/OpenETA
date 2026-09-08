# Environment cleanup and retry ownership

Local refactor candidate, 2026-09-06. The acknowledgement and lifecycle fields
below are shared-interface changes requiring collaborator review. No live
simulator, robot, or external worker was stopped during implementation/tests.

## Confirmed cleanup, not optimistic deletion

The Agent client, MCP server, and bench worker now distinguish requested cleanup
from confirmed resource retirement. `adapter/environment_lifecycle.py` requires
an explicit `ok=true` or `already_closed=true` acknowledgement without error,
pending state, or contradictory failure fields. Empty objects, transport errors,
worker business errors, malformed responses, and pending requests are not success.

The server's `env_lifecycle.py` tracks these process-local phases:

`remote close acknowledged → worker reference released → local cache/checker retired → handle removed`

- A failed remote close retains the local handle, cache/checker, and worker
  reference. A subsequent `close_env` retries that same remote identity.
- A failure after remote acknowledgement retains phase markers. Retry does not
  repeat confirmed remote deletion or an already completed reference release.
- Local retirement removes the handle last. Repeated close of a retired handle
  reports `already_closed=true` without another release.
- Explicit close and TTL cleanup use the same per-environment lock. Weak lock
  ownership preserves the same lock while callers hold or wait for it; dropping
  a lock from the registry while waiters still use it is no longer allowed.
- Control requests reject resources whose cleanup is pending/failed. MCP and
  dashboard environment listings expose `lifecycle_state`; retained failed
  cleanup is not presented as an ordinary active resource.

Example retryable server response:

```json
{
  "ok": false,
  "already_closed": false,
  "close_state": "close_failed",
  "retryable": true,
  "error": "remote_close: RuntimeError: worker unavailable",
  "cleanup_errors": ["remote_close: RuntimeError: worker unavailable"],
  "remote": {}
}
```

The client keeps its bound handle until this acknowledgement is positive.
Concurrent close attempts return `ok=false, pending=true`; they do not report
successful no-ops while the first attempt is unresolved. New proxy calls/reset
cannot reuse a close-failed environment. Startup retry cannot clear an old handle
and create its replacement after failed cleanup. After confirmed cleanup, normal
new-environment creation remains available.

## Local startup admission (2026-09-07)

The tool creator and batch episode reset reserve a shared Host-private
`startup_in_progress` flag under the lifecycle lock before startup, releasing it
in `finally` after create/reset/post-processing. They do not hold the lock across
network calls. While reserved:

- A second create/reset is rejected, even before the first handle arrives.
- Tool, episode and CLI close return failure/pending, not a successful empty
  cleanup receipt. The caller must retry after the tracked startup exits.
- Proxy tools and CLI endpoint replacement are rejected. Startup's own reset
  and bounded retry cleanup remain allowed as internal calls.

The CLI now serializes its close entry with other local lifecycle entries,
retries failed cleanup instead of caching failure forever, and does not reuse
an old successful shutdown result when a new handle is bound. A known handle
without a cleanup transport is a failure, not a successful skip.

The tool creator retains a successfully returned identity before artifact
normalization/callbacks can fail. Cancelled startup attempts cleanup; confirmed
close removes the matching identity, and unconfirmed close retains it with
`close_failed`. Cancellation prose no longer asserts cleanup succeeded merely
because it was attempted. These diagnostic/behavior changes are local candidates
requiring collaborator review; no Agent input or remote operation protocol was
added.

This reservation proves only that one local startup callback is still running.
Once a create transport call raises without returning a handle, remote creation
can still be unknown; the flag is released, and historical transient startup
retry behavior remains. This is **not** orphan reconciliation, durable cleanup,
remote cancellation, or permission to treat a create timeout as confirmed
absence. A `pending` close is not automatically deferred or completed on behalf
of its caller. Direct clients and independently configured objects are outside
this shared-config admission boundary.

Regression: `tests/test_environment_startup_admission.py` uses event-controlled
threads at both create and reset, all three close entries, duplicate startup,
endpoint replacement, transport/callback failures, cancellation cleanup success
and failure, and CLI retry/cache handling. No actual service was started/stopped.

## Worker and TTL behavior

The bench worker executes close under its per-handle observation lock in the
existing simulator executor. It does not hold that blocking lock across an
event-loop await. `env.close()` exceptions retain the environment/cache and return
a failure. Missing handles acknowledge `already_closed=true`, allowing recovery
when a successful DELETE response was lost. This is a behavior change from the
old worker's ambiguous `ok=false` for absence; deploy/restart matching server and
worker revisions together.

BEHAVIOR retains its existing two-phase protocol: worker acknowledgement requests
process retirement, then the manager stops that single-use process. The manager
now verifies process exit before removing the worker from its pool; inability to
confirm exit propagates as cleanup failure with ownership retained.

TTL cleanup retires acknowledged handles independently and retains unsuccessful
ones plus their session activity registration for a later sweep. Its blocking
work runs off the asyncio loop. Stream cancellation is scheduled on the owning
loop. Explicit close is retryable by its caller; this does not introduce automatic
aggressive retries or close other active-session resources merely because one
explicit close failed.

## Limits and validation

- Follow-up (2026-09-07): MCP `observe_env` now shares the complete control lock,
  rather than relying on worker per-step observation serialization. This is not
  proof of remote operation completion or rejection of delayed requests. Unknown
  motion remains fenced even after position agreement, and the planner stops the
  episode. Confirmed cleanup is required before starting a new environment/session;
  it does not clear the old Agent memory fact. See [timeout boundaries](motion-outcome-safety.md).
- Phase markers live in server memory, not a durable distributed cleanup journal.
  Server/worker restart recovery, orphan discovery, and process-wide `stop_all`
  remain separate work. Reference release assumes the manager operation does not
  decrement and then raise; its normal decrement is atomic under its own lock.
- This is not cancellation of arbitrary daemon handlers, a transaction with
  remote physics, or a complete execution lease for every direct worker endpoint.
  A partially failed simulator close may need an operator to retire the worker;
  the system must not invent a successful acknowledgement to make it disappear.
- Existing remote transport timeouts still bound each request; long-running or
  unresponsive workers are not proven dead solely by timeout. Aggregate sweep
  budgets and cross-session creation/cleanup leases remain open.
- Tests use fake transports/managers/processes and local in-memory environments.
  They cover transport/business failures, phase retries, bound-handle retention,
  blocked reuse, startup failure, TTL retention, repeated close, weak lock
  identity, actual asynchronous executor behavior, and BEHAVIOR exit confirmation.
  Real deployed LIBERO/RoboCasa/BEHAVIOR resource-count checks have not run.
- The current outer sandbox forbids socketpair `send`, which prevents asyncio
  cross-thread wakeups. The asynchronous worker test probes this capability and
  explicitly skips there; the complete lifecycle module is also run outside
  that sandbox. A skip alone is never counted as a successful concurrency test.

Regression entry points: `tests/test_environment_lifecycle.py`,
`tests/test_simulator_mcp_proxy.py`, `tests/test_sim_control_codecs.py`, and
`tests/test_bench_worker_terminal_result.py`. See the
[refactor log](harness-refactor-progress-2026-09-05.md) for results and remaining
baseline failures.
