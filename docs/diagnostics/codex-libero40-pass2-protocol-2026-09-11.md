# LIBERO 40 tasks × 2 trials: frozen evaluation protocol

> **2026-09-12 schedule amendment:** The user changed the remaining evaluation to retry first-round failures only. The original protocol below is retained as history; the amendment at the end supersedes its requirements to execute all 80 slots and pool all attempts. Per-trial conditions and the frozen runtime remain unchanged.

User-approved scope: Astra high, all ten tasks in LIBERO Spatial, Object, Goal and Long (`libero_10`), two independent sessions per task. Both trials run even if the first succeeds. No cross-trial memory, operator action hints, additional perception models or changes to the atomic tool set.

## Fixed conditions

- 40 tasks, official order index 0; round 1 traverses all 40, then round 2 repeats all 40.
- Per task, both trials load **official initial state index 0**, with reset seed 0 and five zero-action settling steps. Zero actions use the actual Mink controller dimension (8). Initial scenes already satisfying success are rejected. Selected files and actual initialized simulator-state hashes are recorded privately and compared across repetitions.
- Model `gpt-6-astra`, reasoning effort `high`, subscription authentication, independent private Codex home and thread for each trial.
- Mink joint-velocity controller with the existing checked motion hook, atomic orientation selection, contact feedback and grip/empty-translation controls from the completed development campaign.
- Plugin version `0.1.0+codex.20260911080118`; its version change adds the operator-only official initialization path. No public tool/observation fields are added.
- 2400 seconds per episode; maximum 160 requests, Host turns and internal tool calls. The existing 150-step ordinary / 300-step eligible held-rotation motion limits remain.
- Original preserved remote model catalog. Code, plugin, benchmark assets/state files, dependency versions and Codex version are recorded. Runtime source and benchmark files are checked before/after every trial; resume rejects a changed protocol.
- `/tmp/LIBERO` was absent when preparing this batch. The official repository was restored privately to `tmp/libero-eval-source`, commit `8f1084e3132a39270c3a13ebe37270a43ece2a01`. An isolated `LIBERO_CONFIG_PATH` points to its assets/BDDL/initial states; the user's global LIBERO-PRO configuration is not modified.

## Scoring and failure handling

- Empirical pass@1 = tasks whose first trial passed / 40.
- Empirical pass@2 = tasks with at least one of their two trials passing / 40.
- Attempt success rate = successful attempts / 80 at completion.
- A scored pass requires official Host success, integration success, identical initial-state verification, source consistency and full owned-process/port/auth cleanup.
- All 80 scheduled slots are retained, including infrastructure failures; no replacement trials or successful-task skipping. Network errors and integration failures are separately reported. Until all slots finish, the task coverage numbers are lower bounds, not completed-batch rates.
- Integration or initialization/audit failures pause the queue for diagnosis; a robot task failure with valid integration proceeds to the next slot. Operator interruptions are preserved, not overwritten.
- Bugs discovered during the batch are recorded. Behavioral fixes belong to a subsequent version/batch, so the frozen evaluation is not mixed with development retries.
- This evaluates two attempts at one fixed official initial state per task. It is not the full 50-initial-state official benchmark and does not establish population-level 100% reliability. Official success may precede gripper release, as in the preceding campaign.

## Preparation evidence

`tmp/codex-libero40-pass2-20260911/preflight.json`: 40 official state files checked; one representative task per suite reset twice with identical actual state hashes; all four environments closed. No model calls in this physics preflight. 42 targeted tests passed for initial-state isolation, controller dimension, initial-success rejection, exactly 80 slots, pass@2 union/denominators, evidence gates and existing launcher/controller integration. Plugin validation passed.

A separate observe-only Codex high connection check is kept outside the formal slots. Full task traces and private states remain under the batch directory. The executable protocol is frozen in `protocol.json` immediately before launch.

Implementation: `scripts/codex_pass2.py`, `sim/libero_initial_state.py`, an opt-in reset hook in `sim/env_registry.py`. Without `OPENETA_LIBERO_INIT_STATE_INDEX`, procedural-reset behavior is unchanged. The initialization metadata is written only to the operator-private directory, not the model's observation or tool response. Review the initialization policy with collaborators before main merge. No main change, commit, push or external publication.


## Launch

The observe-only connection check passed with exactly episode_status → observe → finish_episode(false), no robot motion, and complete cleanup. The formal frozen batch then started on 2026-09-11 afternoon (Asia/Shanghai), beginning with Spatial task 0, repetition 1. Codex CLI is 0.154.0; its actual native binary is fingerprinted, including the nested platform package. Frozen input snapshot hash: `3e9a643138a5da625b68576f14831cede8d87240b9327f902f4e4e2341bef696`, covering 1562 files/metadata entries. All registered host/simulator package versions are recorded and checked in the snapshot.

Live human-readable results: `tmp/codex-libero40-pass2-20260911/results.md`; machine-readable results: `results.json`; status: `batch-status.json`. These update automatically after each finalized trial. No formal task result existed at launch.

## 2026-09-12: user-approved retry-only schedule

Round 1 completed all 40 tasks with 34 scored successes (Spatial 10/10, Object 8/10, Goal 9/10, Long 7/10). Empirical single-attempt success is estimated only from this round: **34/40 = 85%**. Long 9 trial 1 reached the 2400-second episode limit and then launcher timeout; it retains its original failed slot, integration flag and six stream-error events. Cleanup and source/initial-state audits passed. The first round recorded 18 stream-error events in total.

After the original queue resumed, Spatial 0 trial 2 completed successfully before the schedule changed. It is retained as an **extra repeat** and excluded from required-trial attempt totals. The queue was paused cleanly at that task boundary; no active trial was aborted.

The six required second sessions are Object 1, Object 4, Goal 5, Long 3, Long 5 and Long 9, in that order. Each uses the same official initial state as its first session, with no first-attempt transcript, operator recovery hint, or cross-session memory. The required evaluation therefore consists of **40 first attempts + 6 retries = 46 attempts**, plus the separately retained Spatial 0 repeat. Final empirical pass@2 is `(34 + successful second attempts on these six failed tasks) / 40`. Failed retries, including infrastructure failures, remain failed slots. No third attempt or replacement is scheduled. Aggregate success over the failure-conditioned attempts is not reported as a single-attempt success estimate.

The operator-only driver `tmp/codex-libero40-pass2-20260911/retry_failed.py` imports the frozen evaluator and changes only task selection and aggregate reporting. It directly reuses the original `run_trial` and all original execution/audit checks. Original `protocol.json`, `frozen-source.json`, the 40 first-round `result.json` files, and the extra repeat are immutable. `retry-policy-amendment.json` records user authorization, the exact retry list, and their hashes plus the new driver's hash. All are validated on resume and at each source check. Existing batch status/results were archived under `before-retry-amendment-*` before reporting changed. No frozen source or benchmark input was edited.

Validation: three focused scheduling/statistics tests cover the exact six-task selection, preservation and exclusion of the extra repeat, completion with an unsuccessful retry, unchanged first-round denominator, pass@2 union, and rejection of an unfinished first round. The normal frozen-protocol preparation check remains required before resuming.
