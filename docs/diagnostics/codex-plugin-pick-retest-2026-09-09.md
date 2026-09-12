# Codex plugin full-task retest — 2026-09-09

## Result

The retest progressed through approach, gripper close, lift, destination
segmentation, transport and descent. It reached the release request but exhausted
the **1,200-second episode deadline** before that request returned. There was no
official task-success evidence. `task_success=false` and
`integration_passed=false`; the latter also reflects unconfirmed cleanup at
launcher exit. This is an unsuccessful bounded task attempt.

The previous motion transport timeout did not recur. Completed motion calls took
3.6–13.6 seconds. Two later OSC moves returned explicit `iteration_limit`
receipts rather than hanging, although they did not reach their requested poses.

Execution conditions changed during the model's recovery: the first trajectory
was blocked for missing collision authorization, after which Sol explicitly set
`enable_collision_check=false` on the executed trajectory and subsequent moves.
The Host accepted those requests. This run therefore does not demonstrate
collision-checked manipulation or verified collision clearance.

## Configuration and isolation

- Checkout: `OpenETA-codex-plugin`; branch
  `dev/huaizezheng/codex-plugin-smoke-2026-09-08`.
- Base HEAD: `723bed12721910e6bcc8db783cb9c1c16cfddaa1`, with the recorded
  inherited snapshot and uncommitted prototype/follow-up changes.
- Model: `gpt-5.6-sol`, medium effort, Codex CLI with saved ChatGPT login.
  Internal LLM backends disabled; no paid-provider fallback.
- Environment: `openeta/libero_libero_spatial_task0-v0`, seed 0, default OSC
  controller. Task: pick up the black bowl between the plate and ramekin and
  place it on the plate.
- Episode deadline: 1,200 s. Requests, turns and tool calls: 80 each.
  Single simulator-call timeout: 120 s.
- Dedicated simulator: `http://127.0.0.1:18778/sse`; existing explicitly
  configured SAM3/AnyGrasp services. Worker output logging enabled.
- Host session: `d1167f2baa614c56af7b5e8665d9ae22`.
- Execution: `07848cb9-32e0-48e7-bc5d-e9fcadc1db1d`.
- Simulator session: `0805635c-7af9-48bd-b93d-924c0057658d`.
- Output: `tmp/codex-plugin-pick-02/`. `source-snapshot.json` hashes the adapter,
  launcher, worker and skill used by the run; all hashes remained unchanged.

The original `OpenETA` checkout and other agents' services were not modified.
The test changed no implementation code. No commit, push or shared-doc update.

## Observed sequence

1. The initial SAM3 call omitted required image references. Sol repaired the
   arguments after an ingress error and selected `detection_001` from the
   `black bowl` query. AnyGrasp returned 19 candidates; Sol compiled candidate 0.
2. IK found solutions for clearance and contact. The first composed approach
   was blocked because the service could not provide requested collision proof.
   Sol repeated IK without endpoint collision checking, recomposed the path and
   executed it with motion collision checking disabled.
3. The approach reached the contact target with 4.6 mm position error. Gripper
   close returned normally, followed by a 5 cm lift. Sol reported visual
   co-motion and source vacancy; that model interpretation is not official
   task-success evidence or a separately validated attachment receipt.
4. Destination segmentation required several repairs: long artifact ID versus
   short packet alias, target-object versus placement-region role, and the
   placement planner's same-source-frame requirement. Sol eventually selected
   the plate mask from the required `obs-0001` frame.
5. The plugin does not expose the placement planner as a callable native tool.
   Sol authored a plate-center transit pose from the available calibration and
   image evidence. Initial IK failed at 0.15 rad orientation tolerance. Sol
   increased that tolerance to 0.30 rad and obtained a usable IK receipt.
6. Transport ran 100 internal steps and returned `iteration_limit`, with
   36.8 mm position error. Sol used the actual endpoint to author a lower
   placement pose; that move ran 80 steps and again returned `iteration_limit`,
   with 21.9 mm position error.
7. Sol judged the bowl visually over the plate and requested gripper opening.
   The total episode deadline cancelled the active execution before an
   acknowledged release result. Its final unsuccessful finish request was
   rejected with `episode_not_active`, since the Host had already truncated.

## Timings and budget outcome

| Action | Recorded tool duration | Receipt |
| --- | ---: | --- |
| Two-waypoint approach | 13.604 s | Executed; endpoint error 4.6 mm |
| Gripper close | 91.252 s | Executed |
| Lift | 3.559 s | Executed |
| Transport | 12.359 s | Target not reached; 100-step iteration limit |
| Descent | 10.811 s | Target not reached; 80-step iteration limit |
| Gripper open | 53.095 s until cancellation | `execution_cancelled`, `abandoned=true` |

At truncation the Host reported **42 admitted requests, 33 completed turns and
33 tool calls** against limits of 80. The open command has a start/end-cancel
event in the rollout but no completed runner step, so it is not counted as an
additional completed turn. The rejected finish after closure also did not
increase the admitted request counter.

The failure reason is exactly:

```json
{
  "code": "episode_timeout",
  "limit": 1200.0,
  "observed": 1200.0004341667518,
  "unit": "seconds"
}
```

Launcher elapsed time was 1,226.932 s, including setup and final response.
The gripper-open failure was `host_execution_error: episode turn exceeded
1200-second deadline`, not `simulator_mcp_transport_timeout` and not a request
or tool-call count limit. All completed step rewards were 0.

Recorded tool start/end durations sum to approximately 205.1 s, including the
cancelled release. The rest includes model inference, context construction,
transport, rejected requests and orchestration; the logs do not precisely
separate these costs. Simply counting completed motion seconds understates
the total episode time required by this integration.

## Remaining issues exposed

- **Gripper batch rendering:** `gripper_close` requests 60 internal steps and
  `gripper_open` requests 40. They leave `render=true`; the worker's batch loop
  performs image work on every step. The earlier fix optimizes intermediate
  steps explicitly marked `render=false`, so it does not cover this path.
  A follow-up should retain periodic/final/terminal images while avoiding full
  camera work on every internal gripper step, with separate validation.
- **Placement tool coverage:** the skill/context describes placement planning
  whose callable tool is absent from this minimal plugin subset. The resulting
  repairs and manually authored poses increased the planning burden.
- **OSC convergence:** transport and descent returned meaningful failures, but
  did not meet position tolerance. This remains a controller issue independent
  of the motion transport-performance fix.
- **Collision policy:** the native schema/Host allowed the model to turn off
  collision checking after the first block. A future collision-checked test must
  provide the capability and enforce the intended policy at the Host boundary.
- **Deadline cleanup:** at launcher exit, `closed=true` meant local Host
  closure, while the cleanup receipt still had `ok=false, pending=true` and
  `environment close already in progress`. The prototype lacks a confirmed
  final cleanup receipt for this concurrent deadline path. Do not reinterpret
  that state as clean episode completion.

These issues were recorded, not silently fixed during the model's run. No second
subscription run or extended deadline was started in this retest turn.

## Usage, cleanup and artifacts

Codex reported 7,613,641 input tokens, of which 7,375,232 were cached input;
12,500 output tokens, with 3,995 reasoning output tokens. These are emitted
aggregate counters, not a conversion into subscription credits or API charges.
No shell, web or file-change actions were recorded from the tested Codex process.

The temporary authentication copy was confirmed deleted. After the run, only
the dedicated process group `3246950` was retired (server `3246953`, worker
`3248163`); their exit and release of port 18778 were verified. This cleanup
does not retroactively change the run's pending receipt or task verdict.

Primary local evidence:

- `tmp/codex-plugin-pick-02/summary.json`, `final.txt`, `analysis.json`.
- `tmp/codex-plugin-pick-02/host/host-status.json`, `host-commands.jsonl`.
- Host workspace `sessions/d1167f2baa614c56af7b5e8665d9ae22/rollout/` and
  `artifacts/responses/` contain tool timings, endpoint receipts and images.
- `tmp/codex-plugin-pick-02/codex-events.jsonl` contains native MCP events;
  `tmp/codex-plugin-pick-02-worker-logs/` retains worker output.

No test suite was rerun because implementation was unchanged. This live task
attempt is the validation performed for the user's retest request.
