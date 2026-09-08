# Bundle-only Stage 1 / Luna / standard batch — 2026-09-07

## Scope and authority

User approved staged interface experiments and external provider test calls.
Read-only RFC check: revision 2158. This is an explicitly selected experimental
profile, not promotion of the reviewed stable contract; no review hash changes,
shared-document writes, commits or pushes. OpenETA collaboration skill was used
to separate the experiment record from the shared schema authority.

`metadata.agent_interface_profile="bundle_stage1"` fixes the profile for the
episode. Unknown profiles fail closed. Default remains `legacy_compatible`.
Only `compile_grasp_seed` and `move_to` are bundle-only in this stage. IK still
exposes the existing native proposal/pose branches and its target-pose bundle
branch; placement, probe and trajectory interfaces are not fully migrated.
This is not an all-tools bundle-only evaluation.

The experimental public schema and admission check share one derived schema.
Planner and direct pipeline reject old/mixed fields; migrated tools must be
atomic, not hidden in batch calls. Host normalization still passes through all
native provenance, freshness, IK, controller and collision gates. No silent
fallback or Agent-selected mode switch. Chat history repeats the authored
bundle request; raw command/trace keeps the normalized native request. Output
receipt IDs remain evidence, not alternative invocation instructions. The
pending-execution index supplies the matching bundle reference. IK guidance is
contract-aware; Pick no longer suggests unsupported position-only execution.

## Run configuration

Run directory: `tmp/batch-luna-bundle-stage1-wUdCss/`.
Manifest: `manifest.json` in that directory; new isolated workspace, no imported
historical traces. Object 0 / seed 0, original benchmark task text. Standard
`agent.cli.batch_eval`, default tools, skills, vision history and supervision.
Single episode/provider concurrency 1; main and fallback both `gpt-5.6-luna` at
the user-authorized provider. Manifest `required_model` is now enforced against
both endpoints before workspace/environment creation, rather than merely logged.

User-confirmed historical per-task budget: 15,000,000 cumulative known tokens,
160 planner turns, 320 tool admissions, 10,800 seconds. This uses the formal
budget but a single interface experiment is not a performance estimate or a
controlled before/after comparison. Dedicated simulator on 127.0.0.1:18766,
Mink joint-velocity profile, one GPU 0 worker, outer lifetime 12,000 seconds.
Token accounting retains its existing cross-role/provider limitations.

```bash
env OPENETA_LLM_MODEL=gpt-5.6-luna OPENETA_LLM_FALLBACK_MODEL=gpt-5.6-luna \
  .venv/bin/python -u -m agent.cli.batch_eval \
  --manifest tmp/batch-luna-bundle-stage1-wUdCss/manifest.json \
  --concurrency 1 --provider-concurrency 1 --model gpt-5.6-luna \
  --sim-url http://127.0.0.1:18766/sse \
  --batch-id standard-luna-bundle-stage1-20260907 \
  --output tmp/batch-luna-bundle-stage1-wUdCss/result.json
```

## Validation and results

New tests cover public request/repair context, schema/admission agreement,
native-field rejection, live-gate retention, pending execution references and
authored-call history. Initial all-tests run caught a real skill projection
length regression (>8000 characters); shortened the corrected IK paragraph,
and the affected runtime seed-chain/skill tests now pass. Two old prose
assertions were updated to check the new non-execution warning.

Final local suite before paid calls: 2385 passed, 25 skipped, one pre-existing
reviewed-authority/catalog failure (67.51 s). Evidence:
`tmp/bundle-stage1-final-tests.xml` and `.log`. All five generated projections
remain current and structural issues are empty (`tmp/bundle-stage1-contract-audit.json`).
Numerical IK tests under LIBERO Python: 5 passed (1.91 s).

Live Agent session: `e1a6a1ab-64fe-4675-aac1-2643878aac72`.
Simulator session: `8ed3a2a3-dd0b-4d2b-a0c1-5b6cef8ba3fe`,
handle `ed1996cb-8f6`. Workspace lives under the run directory's `sessions/`.
Owned simulator PID 1628863, batch client PID 1631465.
First actual provider request verified Luna / 16384 output cap, strict compile
and move schemas in the system message, and Stage 1 scope in the user context.

New live defect (not fixed in the running process): the server added
`execution_seed_quality` / `joint_limit_proximity` prose asserting a feasible
endpoint to an `inconclusive` / `ik_search_no_solution` result. Its best numerical
iterate is not a feasible solution. Safety authorization remains negative, but
the contradictory recovery guidance must be fixed and regression-tested.

### Live outcome: deliberately interrupted, task incomplete

29 main requests / 29 HTTP attempts, 10 rejected candidate calls, 984,452 known
main-planner provider tokens. This is not complete advisor/cross-role billing.
18 completed tool calls: SAM3 5, selection 3, grasp estimate 1, compile 4, IK 4,
MolmoPoint 1. No move, trajectory or gripper calls. One accepted IK decision
arrived after interruption and was not dispatched. Main-model first-start to
last-completion span: 626.95 s; this is not the full batch lifetime.

Four compile calls and four IK calls successfully consumed actual registered
bundles. Initial compile used invented `grasp_candidates_bundle_id`; strict
feedback named the allowed fields and Luna corrected it to `bundle_id`.
All four executed IK results were `inconclusive / ik_search_no_solution`; no
motion authorization was granted. One old candidate bundle was re-previewed,
so a common ID field alone does not establish good recovery decisions.
The other nine parameter rejections were grasp estimate (1), SAM3 (4),
selection (2), MolmoPoint (1), reject-detections (1). Selection also encountered
a live identity-continuity gate and later supplied the required same-instance
evidence. The isolated grasp advisor returned an invalid decision and abstained
as unavailable; the main planner retained candidate choice. No provider retry,
overload or timeout was observed in the recorded main requests.

The operator stopped the experiment to fix contradictory Host feedback, not
because the restored budget was exhausted. This is neither a success nor a
formal performance failure sample. `move_to`'s strict bundle interface has
local gate/schema coverage but **no live execution coverage in this run**.

Cancellation at turn index 18 recorded `episode_interrupt` with
`close_state=closed`, remote `ok=true`, `cleanup_errors=[]`. Client exited 130;
dedicated simulator exited 0 after SIGINT. PIDs 1631465/1628863/1631629 and port
18766 were verified gone. No other user services were stopped.

CLI cancellation did not write normal `result.json` and left the rollout
manifest marked active. Raw trace preserves the close acknowledgement; use
`interrupted-run-summary.json` in the run directory for the explicitly labelled
post-run reconstruction, not a fabricated normal batch result. Durable
interrupted-run finalization remains a follow-up issue.

### Post-run repair

Only after client/environment/server cleanup, gated seed-quality/proximity
annotations on `kinematic_status=reachable`. Failed and timed-out search
iterates retain raw residuals/margins without positive feasibility prose.
Positive kinematic guidance now also states that live authorization gates still
apply. Added fixtures for both no-solution and timeout with near-limit best
iterates. This repair is locally tested; **no second paid/live run has validated
it yet**.

Post-repair full suite: **2387 passed, 25 skipped, one existing reviewed-authority
failure**, 67.23 s (`tmp/bundle-stage1-post-live-tests.xml` / `.log`). External
model integration tests disabled; loopback fixtures permitted. No approval hash
was edited. `git diff --check` passed.
