# Gripper command, aperture, and attachment evidence

2026-09-06 HV-08 partial implementation. Personal-branch candidate: retained
receipt fields, latch semantics, invalidation reasons and gate compatibility
require three-person review before integration. No production authority change.

## Evidence boundaries

The host tracks distinct questions:

- Which binary gripper command was acknowledged? Ordinary arm motion does not
  replace this history.
- Does a supplied `openeta.gripper_actuation_receipt.v1` consistently acknowledge
  that command, a true latch, and positive integer settling steps?
- What does fresh measured aperture indicate? Aperture is neither a persistent
  command latch nor proof of non-empty contact or attachment.
- Is the target-bound contact/proxy/independent attachment evidence still valid?

`agent/tools/gripper_evidence.py` shares measurement and receipt interpretation
between memory and the probe gate. Finite aperture in `[0, 1]` takes precedence
over the coarse `open` boolean. The existing fully-open threshold remains `0.8`.
Thus `{"open": true, "openness": 0.4}` is not fully open; it cannot reconcile an
open request. `{"open": false, "openness": 0.95}` contradicts closed attachment.
Malformed aperture, including bool, NaN, infinity, and out-of-range values, is
unknown and cannot use a coarse boolean as a fallback. Boolean-only legacy
telemetry is still supported when aperture is absent/null.

## Memory and probe behavior

For successful gripper calls, memory retains a supplied actuation receipt alongside
the existing tentative proxy. A conflicting/malformed modern receipt is not
silently treated as legacy absence: memory records `latched: false` and
`receipt_error`, drops the proxy and prior attachment verdict, and invalidates
contact geometry whose latch evidence is contradictory. A successful old result
without an actuation receipt still uses the pre-existing acknowledgement path;
this compatibility path is not modern receipt validation.

Example retained host fact value (IDs/proxy/provenance omitted here):

```json
{
  "schema_version": "openeta.gripper_command_state.v1",
  "position": 0,
  "state": "closed",
  "latched": true,
  "gripper_actuation_receipt": {
    "schema_version": "openeta.gripper_actuation_receipt.v1",
    "command": "close",
    "command_latched": true,
    "steps_executed": 60,
    "measured_open_fraction": 0.4
  }
}
```

`position`/`state` describe the command-side binary condition, not an attachment
verdict. When an unknown action is reconciled only from aperture, memory no longer
invents `latched: true` or carries forward an old proxy/attachment PASS. A close
condition alone does not assert that EEF contact geometry moved. Reconciled open
uses the same contact/attachment invalidation path as acknowledged open.

Independent of command history, fresh fully-open telemetry changes an existing
attachment PASS to UNKNOWN with `invalidation_reason: "measured_gripper_open"`.
Historical gripper command remains unchanged. Existing aperture-collapse
invalidation also remains; neither invalidation asserts where the object went.

Probe preparation/assessment rejects an explicitly unconfirmed latch, an
inconsistent supplied actuation receipt, fresh fully-open telemetry, or malformed
aperture. Matching tentative proxy or the existing articulated-contact fallback
is still required. Legacy facts without modern receipt/latch fields retain their
existing compatibility behavior. No bare close command or narrow aperture becomes
an attachment PASS; no compiled-grasp binding, probe short-ID, IK or collision
gate is waived. The Agent still chooses whether and where to probe.

## Validation and remaining work

Tests exercise command → memory → unknown-result observation reconciliation,
receipt contradictions, evidence invalidation/reuse, and actual probe preparation.
Ordinary arm motion preserves acknowledged closure and attachment evidence.

Read-only inspection of the five representative supplied session traces did not
locate the exact reported last-close rejection in their prepare-probe calls.
Goal 0 contains a carried-proxy-inapplicable rejection (an already separate
articulated fallback); Long 9's two recorded probe preparations succeeded. This
does not disprove failures in other sessions or establish that all HV-08 causes
are fixed. No historical trace was modified.

Still open: exact provenance/freshness for modern receipts across environment
recreation, all attachment-evidence consumers, real interrupted-grasp recovery,
and live Spatial 0 / Long 9 regression. Follow-up on 2026-09-07 separates the old
position-only verdict into `position_reconciliation_status`: even when that is
`completed`, operational `status` remains `unresolved` and the next-action gate
stays closed. Old completed/failed snapshots are also fenced. The planner stops
the episode instead of repeatedly forcing observations. See
[unknown-motion safety and cleanup boundary](motion-outcome-safety.md).
Remote operation completion/leases and in-place unlock remain unimplemented.
Do not use these tests as real-experiment acceptance.
