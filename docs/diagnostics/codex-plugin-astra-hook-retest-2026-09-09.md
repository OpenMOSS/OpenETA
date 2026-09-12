# Astra full-task retest with automatic IK — 2026-09-09

## Result

The task was unsuccessful. Astra voluntarily called `finish_episode(false)` at
567.631 seconds, after two post-motion SAM3 requests failed source-packet
resolution. It did not close the gripper or attempt placement. This was **not**
a timeout or a tool-count exhaustion: 15 native requests, 14 Host turns and 13
Host tool calls were recorded against limits of 1,200 seconds and 80 each.
The final failure-reason object is empty, termination is true and truncation is
false because the model explicitly ended the attempt.

The automatic IK hook was exercised successfully. Astra submitted three
`move_to(bundle_id)` requests without any standalone IK call. The first two
stopped after preflight returned `ik_search_no_solution`, with no motion
dispatched. The third passed fresh IK but reached the 150-step controller
iteration limit with 25.1 mm position error. The hook distinguishes these
outcomes; it did not eliminate controller nonconvergence.

`integration_passed=true`, `task_success=false`, Codex exit code 0. The Host
received an explicit successful environment-close acknowledgement. All recorded
step rewards were zero. Launcher elapsed time including overhead was 610.503 s.

## Configuration

- Model: `gpt-6-astra`, medium effort, saved ChatGPT subscription login.
  Auxiliary model backends disabled; no paid API fallback.
- Same bowl-between-plate-and-ramekin task, LIBERO spatial task 0, seed 0,
  OSC controller and dedicated simulator port 18778.
- Same SAM3/AnyGrasp endpoints as earlier runs; no AnyPlace added.
- Motion and gripper image optimizations retained. New automatic IK ingress
  and corresponding skill guidance active.
- Checkout `OpenETA-codex-plugin`, branch
  `dev/huaizezheng/codex-plugin-smoke-2026-09-08`, base HEAD
  `723bed12721910e6bcc8db783cb9c1c16cfddaa1` plus recorded snapshot/prototype.
- Host session `4d198ed4935f48c6916b0f35234dfb12`;
  execution `cf9797c8-45bb-4871-ae62-8cbd4f44bb4b`;
  simulator session `6ba96c2d-4808-4fe7-8ea3-9b84b0028195`.
- Output `tmp/codex-plugin-astra-02/`.

All seven captured implementation/skill hashes remained unchanged during the
run. The original checkout and other agents' services were untouched. No commit,
push, shared-document update or implementation change was made in this retest.
An initial launcher invocation rejected the pre-created output directory before
model execution. Preparation files were moved aside and the one actual model
attempt started with a new directory.

## Sequence and motion evidence

1. The first SAM3 text request omitted required references. Astra repaired it
   with `obs-0000`/`agentview`, selected `detection_002`, and estimated grasps.
   It described the target as the upside-down bowl between plate and ramekin.
2. Compiled candidates 0 and 1 were submitted directly to native `move_to`.
   Their full-pose IK searches found no solution within budget; neither command
   dispatched motion. For candidate 0 the best solution had only 1.5 mm
   position error but 0.208 rad orientation error, exceeding 0.05 rad tolerance.
3. Candidate 12 passed preflight. The IK result reported 0.232 mm position
   error, 0.0321 rad orientation error and 0.397 rad minimum joint margin.
   Preview and motion both used 0.002 m / 0.05 rad tolerances. Actual motion
   took 20.279 s and stopped after 150 steps with 25.126 mm position error and
   0.0895 rad orientation error. It was not a simulator transport timeout.
4. Astra reported that the plate shifted in the images and chose fresh wrist
   segmentation before contact. It supplied a long simulator observation
   artifact ID, received `invalid_source_packet`, called `observe`, and tried
   the new long render artifact ID. That reference was also rejected.
5. Astra ended the task, saying it could not establish current grasp geometry
   through the exposed interface. It did not try `inspect_evidence`, which it
   had used successfully to repair a similar reference problem in the earlier
   Astra attempt. The model's inability-to-continue claim is not proof that
   every available recovery path was exhausted.

The three native move requests consumed four internal tool steps: two rejected
previews, followed by a successful preview and a move. There were no separate
model calls to `propose_motion_target` or `ik_preview_check`. This confirms the
single-request interface was actually used, but one run does not establish a
task-success or latency improvement.

The actual move retained collision checking by default. Its receipt reported
incomplete collision coverage; no complete path/world clearance is established.
The plate-shift observation is the model's interpretation, not separately
measured object displacement in this report.

## Confirmed reference-feedback gap

The internal Host repair bundles did contain usable registered source IDs:

- First post-motion rejection: `recent_source_packets` included `obs-0006`
  through `obs-0011`, all listing agentview/render/wrist cameras.
- Second rejection after `observe`: the internal list included `obs-0008`
  through `obs-0013`.

However, the corresponding native MCP text responses contained **none of those
IDs**. The only short observation alias in those responses, and in the returned
move/observe contexts, was `obs-0000`, attached to earlier target/grasp evidence.
The long artifact IDs remained visible. This confirms a loss of actionable
reference-repair information between the internal Host record and the native
response projection, rather than absence of registered current frames.
`reference-diagnosis.json` records the paired internal/native comparisons.

Prioritize exposing the current registered packet/camera IDs and the relevant
repair fields, with consistent image labels. This run did not implement that
follow-up. Also retain the separate investigation of IK-approved OSC execution;
automatic preflight cannot by itself resolve its residuals. AnyPlace was not
reached or evaluated.

Astra again said it could not read the installed skill through native tools.
That remains a skill-delivery concern to audit, not a verified explanation for
all of this run's behavior.

## Usage, cleanup and artifacts

Codex reported 1,215,577 input tokens (1,092,864 cached), 2,608 output tokens and
187 reasoning output tokens. These counters are not subscription-credit costs.
No shell, web or file-change actions were recorded from the tested model.
Tool start/end intervals sum to 68.131 s; the rest is not precisely separable
into inference, context projection and orchestration from these logs.

The temporary authentication copy was deleted. Dedicated process group 3427607,
including server 3427609 and worker 3429413, was retired; all exited and port
18778 was released. `process-cleanup.json` records the verification.

Primary artifacts: `summary.json`, `analysis.json`, `source-snapshot.json`,
`reference-diagnosis.json`, `codex-events.jsonl`, `host/host-status.json`,
`host/host-commands.jsonl`, and the session rollout/response artifacts.
No test suite was rerun because implementation was unchanged; validation was
this authorized live task plus source/snapshot and final diff checks.
