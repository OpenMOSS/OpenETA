# Manual-provider attribution (2026-09-07 candidate)

## R0 27: received assistance versus pending help

Fresh Long 9 evidence showed a provider read timeout becoming `ask_human`, with
0.019 s of waiting enough to mark `human_assisted=true` despite no operator
answer. The candidate now treats a Host backend `failed` status as
`planner_provider_failed`: the episode and its final step are truncated,
`waiting_for_human=false`, and batch error type is `EpisodePlannerFailure`.
Provider error code/type, retryability and attempt count are retained. Raw
backend fallback commands remain in the trace; their `ask_human` spelling does
not override the runner's failure status or trigger guidance/CLI questioning.
Model-authored error parameters without Host failed status cannot trigger it.

`human_assisted` now requires a nonempty Host `human_answer` event since the
current `episode_start`, or a recorded main-planner manual-provider response.
Waiting, calling resume without an answer, and empty answers are not assistance;
a zero-wait answer counts, and a later blank answer cannot erase earlier help.
The elapsed/wait budget clocks are unchanged. This is current-run attribution,
not proof that every earlier episode or isolated role was autonomous.

Failed backend calls without usage retain `token_usage_sources.unknown` rather
than disappearing from source accounting. `total_tokens` still sums available
usage, not unknown remote consumption; it is not a complete bill or a hard
provider-side spending cap. HTTP retry attempts within a backend result are not
separate entries in this source counter. Existing deterministic/Host-only
no-usage behavior is preserved.

This changes episode failure/assistance interpretation and requires collaborator
review. Validation includes a real local HTTP read-timeout fixture through the
production backend/planner/runner, plus deterministic answer/reset/spoofing
tests (`tests/test_provider_failure_episode.py`). No external provider outage or
new LIBERO run was induced for this repair; previous raw reports are unchanged.
Full local regression: 2336 passed, 21 skipped, one pre-existing historical
authority/catalog failure, 36 warnings (64.43 s). Real-model integration flags
were disabled. JUnit: `tmp/provider-failure-r0-27.oOJW3Z/local-tests.xml`.

The live observation smoke received operator responses but reported
`human_wait_s=0` / `human_assisted=false`. The former only measures the runner's
explicit interaction pauses; provider waiting previously had no reporting path.

The standalone console now adds server-generated completion-envelope metadata:

```json
{
  "provider_interaction": {
    "schema_version": "manual_vlm.provider_interaction.v1",
    "mode": "manual_console",
    "request_id": "the-console-request-id",
    "wait_s": 7.5
  }
}
```

`wait_s` uses the console's monotonic clock from queue insertion to accepted
response (including submission processing), not subtraction of wall timestamps.
It is a completed request's console latency, not proof of human thinking time.
It is recorded with the response trace and is not read from operator content,
model names, elapsed-time thresholds, or adapter-authored assistant messages.
The standalone console still does not import Agent implementation modules.

The backend validates the version, mode, bounded nonempty request ID and finite
nonnegative numeric duration; malformed/unknown envelopes are ignored. Main
planner validation retries carry every accepted envelope into decision metadata,
including attempts that failed action validation. The episode deduplicates
request IDs across recorded actions. Example result additions:

```json
{
  "usage": {
    "human_wait_s": 0.0,
    "manual_provider": {
      "scope": "recorded_main_planner_responses_this_run",
      "response_count": 2,
      "wait_s": 10.0,
      "source": "provider_reported"
    }
  },
  "assistance": {
    "manual_provider_assisted": true,
    "human_assisted": true
  }
}
```

`human_assisted` now includes a reported manual-console response even when its
duration is zero. This identifies use of the manual channel, not the operator's
biological identity: Codex-operated Human VLM is still manual-provider-assisted.
This additive envelope/metadata and broadened assistance flag need three-person
contract review before coordinated deployment. Existing success classifiers are
unchanged, but downstream assistance cohorts can change for new records.

## Deliberate limits

- `human_wait_s` remains runner-pause time. Neither it nor provider wait is
  subtracted differently from the existing episode budget. New wait totals must
  not be subtracted from wall time and advertised as autonomous model latency.
- Coverage is **recorded main-planner responses in this run**, not every provider
  call: nested advisors, guidance, visual delta, reviews, outstanding/cancelled
  requests and responses lost before an episode action is recorded remain open.
  Thus absent metadata or a false flag is not proof of fully autonomous execution.
- Provider metadata is reported telemetry, not a signed safety/authorization
  receipt. It cannot authorize motion or certify task success.
- Reset/resume starts this local ledger anew. Historical/pre-resume waits are
  not reconstructed or silently inferred; combine explicit run records when
  comparing assisted episodes. Old services do not emit the new envelope.
- Token estimates retain their existing source labels; manual text length is
  not actual billed model usage. No tokens or historical traces are rewritten.

Tests: `tests/test_manual_provider_accounting.py` covers the real backend →
planner → runtime → episode path with a fake provider transport, retry/dedup/reset,
strict metadata validation and monotonic console timing. The existing real
loopback test in `tests/test_manual_vlm_proxy.py` checks the HTTP envelope. These
are fixtures, not a replacement for a fresh Human VLM task/advisor regression.

## Explicit console cancellation is not transient (2026-09-07)

The existing console responds to operator cancellation **and** its own decision
timeout with HTTP 503 and this structured error:

```json
{
  "error": {
    "type": "human_cancelled",
    "code": "human_cancelled",
    "request_id": "the-console-request-id",
    "message": "Human response timed out."
  }
}
```

Previously the backend treated this as retryable service failure and could
resubmit or switch to a configured fallback. It now recognizes this exact
envelope in `ProviderHttpError`, suppresses both automatic retry and failover,
and reports `provider_error_code=manual_provider_cancelled`, `retryable=false`.
The main planner preserves the existing structured `ask_human` failure path;
this does not generate another provider request during validation retries.

Recognition requires HTTP 503, both matching code/type, a nonblank request ID
of at most 128 characters, and a parseable JSON body of at most 16,384 characters.
Plain-text mentions, malformed/oversized bodies, and other 503 errors retain
existing retry policy. The console wire format/status itself is unchanged, so
this fix can understand existing services without restarting them. The new
failure code and changed retry semantics require collaboration review.

The terminal result applies to this provider invocation, not an irrevocable
session-wide ban: an operator may explicitly start another attempt. It does not
cancel parent/sibling requests, propagate Agent cancellation to pending server
requests, abort simulator motion, distinguish operator cancellation from console
timeout, or count cancelled requests as completed manual responses. Those remain
separate work. A transport timeout without the explicit envelope still follows
the existing transient-failure policy.

Regression: `tests/test_manual_provider_cancellation.py` checks retry/failover,
ordinary 503 compatibility, urllib HTTP-error conversion, and main episode
propagation. Two actual loopback backend/console cases verify manual cancellation
and console timeout leave exactly one cancelled request, with a loopback-only
fallback configured. These fixtures use fixed text, not perception data or any
existing user request.
