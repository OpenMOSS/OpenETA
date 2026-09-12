# GPT-6 Astra subscription harness experiment — 2026-09-09

## Result

The Astra run did **not** complete the bowl-on-plate task. It attempted two
grasps and never entered destination segmentation, transport or placement.
The first lift reached its requested EEF pose, but Astra judged from the images
that the bowl did not follow. The second lift was stopped before any controller
step by an `attached_object_world` collision verdict. The episode then exhausted
its 1,200-second deadline. Official task success remained false and every
recorded step reward was zero.

The subscription-backed native MCP integration worked. The launcher reported
`status=completed`, exit code 0, `integration_passed=true`, and
`task_success=false`. Integration success is not manipulation success.
This single attempt does not establish an improvement over Sol.

## Configuration and isolation

- Model: `gpt-6-astra`, medium reasoning effort, Codex CLI with ChatGPT login.
  Auxiliary model inference disabled; no paid-provider fallback.
- Task: pick up the black bowl between the plate and the ramekin and place it
  on the plate; environment `openeta/libero_libero_spatial_task0-v0`, seed 0.
- Same native tool subset as the Sol run: **no AnyPlace or
  camera_pose_to_world**. No tools or runtime code changed during this run.
- Limits: episode 1,200 s; requests, turns and tool calls 80 each; individual
  simulator call 120 s. Launcher elapsed time: 1,244.105 s including overhead.
- Both the intermediate motion-image optimization and the final-only gripper
  rendering fix were active. The earlier Sol full-task run had only the former;
  this is therefore not a strict model-only performance comparison.
- Checkout: `OpenETA-codex-plugin`, branch
  `dev/huaizezheng/codex-plugin-smoke-2026-09-08`, base HEAD
  `723bed12721910e6bcc8db783cb9c1c16cfddaa1` plus recorded inherited snapshot
  and isolated uncommitted prototype changes.
- Dedicated simulator port: 18778. SAM3 and AnyGrasp used the same configured
  services as the earlier experiment.
- Host session: `cef7f39f80a8481ba178bd886e268164`.
- Execution: `e015c910-a006-41a6-80ab-dcd3300e0e85`.
- Simulator session: `3e2445f0-6d48-48e7-9322-5b13267dd04b`.
- Evidence directory: `tmp/codex-plugin-astra-01/`.

The original `OpenETA` checkout and other agents' services were not modified.
All six runtime/skill source hashes captured before the experiment remained
unchanged. No commit, push or shared-document update was performed.

## Observed behavior

1. The first SAM3 request omitted required packet/camera references. Astra
   repaired the arguments. Text segmentation returned no detections; Astra
   then used a point prompt and selected the bowl between plate and ramekin.
2. It compared two compiled grasp candidates and reached a clearance pose.
   It observed that the bowl had settled upright and refreshed the grasp
   evidence instead of continuing with the earlier geometry.
3. Refreshing segmentation encountered the same kind of reference friction as
   Sol: two long simulator artifact identifiers were rejected as source packet
   IDs. An additional observation did not immediately resolve this. After
   `inspect_evidence`, Astra used the registered alias `obs-0013` successfully.
4. New grasp candidates included full-pose IK searches with no solution and
   feasible solutions with small joint margins. Two approaches passed the
   execution gate but failed to converge within 150 OSC steps.
5. Astra compared more candidates, then authored a vertical world-frame
   approach using the current estimated geometry. It added an unsupported
   `reason` argument to `propose_motion_target`, received a schema error, and
   repaired it. The vertical approach and descent reached their targets.
6. The first close and short lift returned. Astra reported that the bowl did
   not follow the lift, reopened the gripper, and changed the contact point
   and jaw direction for a second attempt.
7. The adjusted approach and descent reached their targets; the second close
   returned. The following lift was stopped at the initial configuration with
   `collision_detected`, class `attached_object_world`, and zero executed steps.
   This class is a collision-checker report, not independent proof that the
   target was securely grasped. No second lift co-motion evidence was obtained.
8. Astra requested `finish_episode(success=false)`. The deadline had already
   closed the Host, so that request returned `episode_not_active`. Its final
   answer correctly reported failure without claiming task success.

Every requested `move_to` retained `enable_collision_check=true`. Early motion
receipts nevertheless reported incomplete collision coverage and no available
collision result; the final lift did report a collision. These facts demonstrate
neither complete trajectory/world checking nor collision-free manipulation.
Unlike Sol's earlier run, Astra did not disable the check during recovery.

## Motion and gripper timings

Durations below are rollout tool start/end intervals, not inference latency.

| Action | Duration | Controller result |
| --- | ---: | --- |
| Initial clearance | 6.884 s | Reached; 1.9 mm position error |
| Refreshed approach 1 | 18.497 s | 150-step iteration limit; 60.8 mm error |
| Refreshed approach 2 | 19.097 s | 150-step iteration limit; 77.8 mm error |
| Vertical approach | 7.120 s | Reached; 2.8 mm error |
| First descent | 4.969 s | Reached per controller receipt; 5.1 mm Euclidean error |
| First close | 2.490 s | Returned |
| First lift | 3.553 s | Reached; 4.6 mm error; model judged no bowl co-motion |
| Reopen | 2.359 s | Returned |
| Adjusted approach | 6.877 s | Reached; 1.0 mm error |
| Second descent | 5.075 s | Reached; 2.7 mm error |
| Second close | 2.672 s | Returned |
| Second lift | 0.128 s | Collision stop; zero steps |

There was no individual simulator transport timeout. The failure reason was
`episode_timeout`, observed at 1,200.000235 s. The Host admitted **52 requests**
and completed **49 turns/tool calls**, below all count limits of 80. The late
rejected finish did not increase the admitted count.

Recorded tool start/end durations sum to 131.084 s. The remaining episode time
includes model inference, context construction, projection, transport, rejected
requests and orchestration. It cannot all be labeled model reasoning latency
from these logs. The large gap warrants measurement before further budget tuning.

## Comparison and next diagnostic priorities

- Astra repaired argument errors, refreshed moved-object evidence, changed
  approach orientation, recognized its first failed grasp, and retained motion
  collision checking. These are useful observed behaviors, not a success-rate
  estimate. The earlier Sol run reached an unacknowledged release request;
  Astra did not reach placement. Neither attempt had official task success.
- Faster gripper calls validate that the new path runs promptly in an actual
  model-driven task. Comparing 2.490 s here with Sol's historical 91.252 s does
  not isolate a model effect or an identical-contact paired speedup.
- Both models made the initial SAM3 argument error and supplied an unsupported
  `reason` to `propose_motion_target`. Packet aliases also confused both.
  Native schemas avoid the XML transport path but do not eliminate argument
  and reference mistakes. Audit exposed schemas, evidence IDs and repair
  payloads before attributing all failures to model capacity.
- Astra explicitly said that native tools did not provide a way to read the
  installed skill file and that it would use Host guidance. The launcher disables
  shell access and asks the model to use the skill. This is a potential skill
  delivery gap to verify; the model's statement alone does not establish exactly
  which skill content Codex injected. No skill-reading action was recorded.
- Diagnose the two OSC convergence failures and replay the final collision stop
  with the recorded state before deciding whether contact recovery needs a new
  tool. The final receipt already advertises `escape_current_collision_boundary`
  through existing preview/motion tools; there was no remaining time to test it.
  Do not label the collision a false positive without a replay.
- AnyPlace remains relevant to the later placement stage, but its absence was
  not exercised by Astra's attempt. Adding it alone cannot be credited with
  resolving the observed grasp and control failures. No new tool was added.

## Usage, cleanup and reproducibility

Codex reported 4,646,428 input tokens (4,475,264 cached), 7,968 output tokens,
and 645 reasoning output tokens. These emitted aggregate counters do not state
subscription-credit consumption. No shell, web or file-change actions were
recorded from the tested model process.

Final Host cleanup was `{"ok":true,"skipped":true}`. The environment close
implementation returns this when its handle is already empty; the launcher
accepted it as clean integration completion. Separately, the dedicated server
group 3314946 (server 3314948, worker 3316032) was terminated after the run.
Both processes exited and port 18778 was released. The temporary authentication
copy was deleted. These checks are recorded in `process-cleanup.json`.

Primary evidence: `summary.json`, `analysis.json`, `source-snapshot.json`,
`codex-events.jsonl`, `final.txt`, `host/host-status.json`,
`host/host-commands.jsonl`, and the session's `rollout/` and
`artifacts/responses/` directories. `analysis.json` is generated from the raw
events by the local `analyze.py`; it preserves motion receipts, tool durations,
IK outcomes, rejected calls and model statements.

No implementation code changed in this experiment, so no test suite was rerun.
Validation consisted of the live bounded task, unchanged-source checks,
inherited-snapshot checks, and final diff formatting validation.
