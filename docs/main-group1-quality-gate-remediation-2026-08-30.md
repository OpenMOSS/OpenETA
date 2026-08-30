# Main feature integration: Group 1 quality-gate remediation

Date: 2026-08-30

## Quality-gate result

Group 1 integrates the upstream Python-exec fixes and human VLM operator
console. Its first fixed-seed LIBERO Object pass@1 quality gate combined:

- the valid outcomes from `libero-object-10-main-group1-20260829-r01`; and
- the six serial infrastructure supplements from
  `libero-object-6-main-group1-serial-gripper-contract-20260830-r05`.

The aggregate result is 1/10, below the frozen 2/10 baseline. The supplement
itself completed 6/6 jobs with six valid `agent_failure` outcomes. Two provider
capacity failures were retried and excluded from task outcomes. Group 2 must not
start until a new Group 1 quality gate reaches at least 2/10.

## Evidence-backed defects

The r05 rollout for Object task 4 exposed a deterministic false-positive path:

1. the last acknowledged close reported `attachment_proxy_status=not_armed`
   with `empty_close_or_no_measurable_contact`;
2. the gripper was subsequently opened and the current observation reported
   `open=true`, `openness≈0.998`;
3. `prepare_attachment_probe` nevertheless prepared a probe; and
4. the visual reviewer returned `PASS`, contradicting the host-observed aperture.

The same rollouts also reused old contact coordinates after an acknowledged
reopen. Target identity lineage is still useful, but released or abandoned
contact geometry is not current motion or close authorization.

## Runtime remediation

- A successful close now retains its compact `attachment_proxy_receipt` in the
  host-owned gripper command fact.
- Probe preparation and assessment fail closed unless the latest command is
  closed, the close receipt is `tentative`, and current measured aperture does
  not explicitly report the gripper open.
- A visual reviewer cannot override contradictory gripper command/aperture
  evidence.
- An acknowledged reopen after a close invalidates only the active contact
  geometry. It preserves the target identity anchor and immutable evidence, but
  blocks contact authorization, close, and probe reuse until the Agent compiles
  and executes a new branch.

These are evidence-lifetime checks, not task progress or a manipulation-stage
state machine. The Agent remains free to choose any new perception, candidate,
path, or recovery action.

## Contract review boundary

The compact Agent-facing descriptions now disclose the close-receipt and
measured-aperture requirement, and every rejection contains actionable evidence.
The reviewed canonical `openeta.tool_contract.v1` catalog is intentionally not
changed in this remediation commit. A future canonical update should add the
host gripper-latch receipt/current observation as explicit consumed facts and
declare the gripper-evidence audit output for `prepare_attachment_probe` and
`assess_attachment_probe`. That catalog change requires three-person schema
review before regenerating authority canaries or promotion decisions.

## Local verification

- focused attachment/memory tests: 63 passed;
- contract, fixture, no-state-machine, and migration tests: 137 passed;
- complete sandbox suite: 1502 passed, 12 skipped, with four expected local
  socket failures for the manual VLM proxy before the unrestricted rerun.

The next evidence step is a small live probe-rejection canary followed by a new
fixed-seed Object 10 pass@1 Group 1 quality gate.

## Live aperture-semantics correction

The first remediation run
`libero-object-10-main-group1-attachment-remediation-20260830-r01` was stopped
during task 1 and is invalid for aggregate scoring. It exposed a second,
narrower contract detail: the simulator can report coarse `open=true` while a
continuous aperture measurement around `0.526` and a tentative close receipt
show that an object is obstructing closure. The initial remediation incorrectly
treated the boolean as authoritative and rejected a valid probe.

The corrected verifier uses finite continuous `openness` as the primary
measurement (`>=0.8` is definitely open) and falls back to the boolean only when
continuous aperture is absent. The original false-positive case remains
rejected because it reported `openness≈0.998`. A dedicated regression test now
covers the mixed `open=true, openness≈0.526` case. After this correction the
full suite reports 1508 passed and 12 skipped.

## Completed Group 1 quality gate

The fixed-seed remediation run
`libero-object-10-main-group1-attachment-remediation-20260830-r02` completed all
ten jobs on `gpt-5.6-luna` at commit `1fad13f`:

- objective pass@1: **3/10 (30%)**, above the frozen 2/10 baseline;
- valid outcomes: 10/10, with no infrastructure-invalid jobs or retries;
- failures: seven `agent_failure` outcomes at the episode layer;
- provider concurrency: one, 1,707 requests, zero queue timeouts; and
- wall clock: 31,386.563 seconds.

Task indices 1, 3, and 6 succeeded with official environment reward at 128,
128, and 101 turns respectively. Tasks 2, 4, 5, 7, 8, and 9 exhausted the
160-turn budget. Task 0 ended at turn 105 with a valid
`remote_episode_terminated` agent failure. The successful task 6 recovered from
an initial failed grasp and completed a second grasp/transport/place branch,
which is positive evidence that the repair did not introduce a host-maintained
manipulation-stage machine.

The live run directly exercised the Group 1 Python capability. On Object task
0 the Agent used `python_exec` to read the immutable grasp-candidate artifact,
compare the complete candidate bank, and choose a materially different pose.
The call completed successfully; its 2,838-character stdout was materialized as
an artifact and returned with an explicit path and preview. This validates the
large-result handoff rather than merely unit-testing the sandbox.

The attachment repair also behaved as intended across multiple tasks:

- a mixed `open=true, openness≈0.526` tentative close was allowed to proceed to
  a probe instead of being falsely rejected;
- deterministic empty-close receipts caused the Agent to reopen and rebuild the
  grasp branch; and
- successful probe chains enabled transport and placement on three distinct
  Object tasks.

## Observed follow-up issues outside the Group 1 regression

Two costly behaviors predate the Group 1 commits and are retained as review
evidence rather than folded into this migration:

1. `attached_release_after_failed_motion` can retain an earlier PASS attachment
   after a failed carry motion even when fresh images suggest the object is back
   on the surface. The gate gives an actionable recovery—execute a distinct
   successful carrying motion before release—and the Agent used it, but the
   detour is expensive. `git blame` traces this gate to `aca54cd`, before Group
   1.
2. When AnyPlace receives an invalid placement bundle, the repair envelope can
   point to a `grasp:*` bundle that is valid for `grasp_pose_estimate` but still
   not a placement-ready AnyPlace input. In task 3 the Agent first fabricated a
   `grasp-selection:*` id, then retried AnyPlace with the suggested `grasp:*`
   id, and only later rebuilt a valid placement bundle. The task nevertheless
   succeeded, but the repair response should eventually identify an acceptable
   AnyPlace bundle or state exactly which evidence must be regenerated.

Neither issue changes the Group 1 acceptance result or the reviewed canonical
ToolContract catalog. They remain candidates for a separately reviewed harness
repair.
