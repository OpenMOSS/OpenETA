# Manipulation autonomy: policy gates become evidence

User authorized implementation and fresh experiments after reviewing the gate
audit. Branch: `dev/huaizezheng/luna-task-recovery-2026-09-08`. No commit/push,
review-approval hash changes or shared-document writes. OpenETA RFC collaboration
skill used; shared RFC access remains unavailable. Contract/checker semantics
below require collaborator review, including the default profile, not only the
experimental bundle projection.

## Implementation

Host owns reference integrity and checked execution; the Agent owns recovery
strategy. An actuator command is not a claim of grasp or task success.

| Former strategy veto | New behavior | Boundary retained |
| --- | --- | --- |
| Failed carrying motion forbids opening | Return prior failure as advisory; allow release | Unknown remote operation fencing, actual actuation receipt |
| Close requires old compiled contact arrival within fixed tolerances | Allow finger command and expose measured endpoint/residual; unavailable optional proxy binding becomes unbound actuation | No automatic attachment PASS; do not transfer contact binding between objects |
| Fixed clearance corridor / entry rotation / successful previous clearance | Agent may select another approach | Exact current IK receipt and controller collision checks |
| 2 cm incremental / 10 cm cumulative / 20 degree recovery budgets | Residuals are advisory; no global historical motion budget | Finite world-frame geometry, current object epoch, resolvable reference |
| Compiling branch B revokes branch A | Explicit same-instance current A remains resolvable | Invalidated geometry and cross-instance authorization stay rejected |
| Reference verifier mismatch or abstain forbids SAM3 selection | Main Agent may choose; assessment and selection rationale retained | Exact packet/camera/mask provenance and identity relations |
| Probe requires close latch / tentative proxy / reached contact beforehand | Bounded probe can measure uncertain attachment; return missing evidence as warnings | Frozen bounded probe path, exact execution receipts, before/after evidence and independent assessment |

`manipulation_advisories` is exposed in the canonical decision context and direct
motion/gripper results. Example (additive output, not an authorization token):

```json
{"code":"previous_motion_not_reached","blocking":false,"actual_eef_xyz":[0.1,0.2,0.3],"message":"The previous motion failed. Judge the current scene before choosing another pose, closing, or releasing; the old endpoint is not the actual state."}
```

Actual failures remain FAIL. Visual attachment PASS contradicted by a measured
open gripper, or lacking a usable aperture measurement, remains UNKNOWN; probing
is allowed but does not manufacture positive evidence. A known executed contact
for object A followed by a plan for B does not assign B to a later close.

Not removed: malformed/stale/bad-instance references; exact pose-policy and
joint-seed binding; controller physical checks; pending unknown remote operation
fencing; attachment assessment and terminal reward. `compute_wrist_alignment`
retains its local calibrated estimation domain, but no longer consumes the
historical cumulative execution budget. That estimator abstention is not a veto
on independently authored checked poses.

## Validation

Focused autonomy / contract remediation / decision context / probe / remote
reconciliation tests: **151 passed**. Advisor handler wording tests: **25 passed**.
Optional-binding dispatch regression: three cases (missing/stale/invalidated
planning evidence) exercise the actual runtime-to-simulator-proxy path, with no
borrowed contact authorization or attachment proof. Reviewer-context tests:
**9 passed**. Mink controlled-joint numerical tests: **7 passed**. Final full
suite: **2439 passed / 32 skipped / 1 failed** in 67.18 s, plus 36 existing Pillow
deprecation warnings. Failure is the existing reviewed-migration authority test
(`test_reviewed_tool_contract_migration_is_complete_and_narrowly_authoritative`):
the current catalog/Markdown/readiness/resolver projections are current and
structural issues are empty, but promotion/authority evidence no longer matches
the implementation. Approval hashes were not rewritten to make this green.
Logs: `tmp/luna-autonomy-tests.log`, `tmp/luna-autonomy-tests.xml`.
`git diff --check` passed. No historical traces sent to providers.

## Fresh standard Luna experiment: stopped before motion

Run root: `tmp/luna-autonomy-CMxgpj/`; session
`e10fa4cc-6361-4b79-ad2c-04fff9cbcfbd`. Standard `agent.cli.batch_eval`,
`bundle_stage3`, Object 0 / seed 0 / original task, main/advisor/fallback Luna.
15M known tokens / 160 turns / 320 tools / 10800 s; dedicated simulator 18766,
same Mink joint-velocity controller as the prior failed-contact sample. No task
oracle coordinates or target labels added. Official reward remains the task
success criterion; this one episode is diagnostic, not a success-rate estimate.

Initial launch was denied over object-bank data egress. Read-only code inspection
established `ObjectMemoryBankClient` sends only name/namespace GET queries and
downloads assets, blocks redirects, and never receives scene images. Re-review
approved the same command on that evidence. Simulated images go only to the
already authorized small-model services and Luna provider; no alternate route.
The standard entry also exposes MolmoPoint on the same authorized host (8777);
the Agent used it after text SAM3 returned no detections.

Eight recorded main calls, all `gpt-5.6-luna`, **269971 known main tokens**.
This excludes isolated advisor / other unrecorded in-flight provider accounting,
so it is not a billing total. One malformed XML response was repaired; a later
SAM3 `positive_points` / packet mismatch was repaired to `points` bound to the
original `obs-0002` / `agentview` source. No budget exhaustion or observed provider
overload caused the stop.

Outcome: Agent selected the foreground red/green can as alphabet soup, then
requested grasp estimation. The actual SAM3 contact sheet shows the wrong
instance. At selection, recorded provider attachments include both current views,
the actual selection sheet and source image, all three labeled object-reference
images, and candidate overlay/crop. Thus this sample is not the earlier missing-
selection-image bug; correct visible context did not guarantee correct identity.
Do not turn that model mistake into a new mandatory reference-verifier veto.

Stopped before any motion or finger actuation; no grasp, lift or official task
success. This episode did not reach the contact-recovery milestone and cannot
measure whether gate simplification improves autonomous grasping. Batch PID
2493324 exited 130; `episode_interrupt.cleanup` reports closed, `ok=true`, no
cleanup errors. No result-file success is inferred from an interrupted batch.

## Deterministic local simulator integration (not Agent performance)

Supplement: `tmp/luna-autonomy-CMxgpj/replay_close.py`, result
`tmp/luna-autonomy-CMxgpj/replay-close.json`. Loopback-only fresh Object 0 / seed 0;
replay the first two previously recorded exact seeded motions with unchanged
collision checks, then send one close and one open through the actual
`OpenEtaAgentRuntime` / planner / pipeline / simulator proxy. Static requests
isolate dispatch behavior; no LLM, perception service or historical trace upload.

- Clearance reached in **13 steps**, position error **2.037 mm**, orientation
  error **0.08451 rad**.
- Contact again stopped at **150 steps**, position error **22.421 mm**,
  orientation error **0.28195 rad**, `iteration_limit`, `reached_target=false`.
- Both close and open returned `executed` and tool success. Close received a
  tentative proxy with measured openness **0.46497** and
  `attachment_proven=false`; opening retired it.
- `previous_motion_not_reached` and `latest_contact_execution` remain visible;
  the contact receipt remains failed and `attachment_evidence` remains null.
- New environment cleanup reports closed, `ok=true`, no errors. Dedicated
  simulator PID 2490087 exited 0 after all environments closed.
  Port 18766 was verified free afterward; no paid/test service left running.

This validates actuation dispatch after a physically reproduced failed endpoint,
not a successful grasp or transport. No probe/lift/task success was asserted.
Independent review-context wording and optional-proxy fallback cleanup landed
after the paid batch process started; this local integration uses those final
changes. They were not exercised by the perception-only paid sample.
