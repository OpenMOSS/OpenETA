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
