# Atomic tools + Astra + Mink — 2026-09-09

## Outcome

The atomic tool profile is implemented and exercised through the native Codex
plugin. Astra selected a bowl from RGB images, authored surface points and
Cartesian poses, closed on its rim, lifted it, translated toward the plate and
later descended. The post-lift image visibly shows the bowl carried by the
fingers. **The complete task did not succeed:** the 1200 s Host deadline expired
before release. Official task success is false. No finish claim was received.

Only one model episode was launched. It used ChatGPT subscription authentication,
gpt-6-astra medium, default Mink, LIBERO spatial task 0 / seed 0, 512 px images,
1200 s and 80 requests / Host turns / Host tool calls. There were no SAM3,
AnyGrasp, AnyPlace or auxiliary LLM calls. The independent clone remains on
`dev/huaizezheng/codex-plugin-smoke-2026-09-08`; main and its agents were untouched.

## Behavior and motion evidence

The model received the atomic skill body directly in its startup prompt and
acknowledged using it. Unlike the preceding attempt, it did not report inability
to read the skill. Native exposure was exactly six tools. The run used 16 native
active requests and 17 internal Host tool calls. Active native calls: episode_status 1,
observe 1, mark_point 7, move_to 6 (one geometric preview), gripper_control 1.
There were no native parameter/schema errors or atomic geometry rejections.
A seventeenth native call requested gripper open after the Host deadline; it
returned episode_not_active and executed no release. The closed-call rejection
does not increment the Host active-request counter.

| Motion | Control steps | Position error | Orientation error | Outcome |
|---|---:|---:|---:|---|
| Point + 120 mm upward approach | 42 | 0.238 mm | 0.0268 rad | Reached |
| Point − 8 mm, explicit point-contact authorization | 12 | 0.584 mm | 0.0149 rad | Reached |
| Lift 120 mm after closing | 25 | 0.414 mm | 0.0383 rad | Reached; bowl visibly carried |
| Translate toward plate | 57 | 0.355 mm | 0.1210 rad | local_convergence_stalled |
| Descend 60 mm after reconnection | 10 | 1.826 mm | 0.0423 rad | Reached |

Every physical move went through propose_motion_target, fresh ik_preview_check,
then move_to. Mink receipts report self/endpoint/trajectory/world coverage and
no detected collision on all five moves. This is the controller's reported
coverage, not a claim that all physical contact was absent: target-gripper
contact is explicitly allowed by the Host's narrow binding.

The translation missed its 0.05 rad orientation tolerance while position was
already close. Its gentle attached-object profile used a 0.2 rad/s joint limit
and stopped after 30 orientation-stall steps. The subsequent downward move
preserved the then-current measured orientation and reached its own target.
No controller tolerances, physics or iteration budgets were changed during the
run. The last 60 mm descent ended around episode 1110.65 s, leaving about 89 s;
no open/release command followed before deadline.

After close, measured openness was 0.22897. This did not itself establish grasp;
post-lift RGB supplied the observed co-motion evidence. Simulator attachment
proxies remain tentative safety metadata and do not establish task success.

## Connection and timing

The model resumed after a **552.939 s** gap from the end of the translation to
start of the next descent. During the gap, Codex logged reconnect stages 2/5
through 5/5: WebSocket request send failures, first Connection reset by peer
(os error 104), then Broken pipe (os error 32). This resembles the preceding
554.4 s gap but uses different reported errors. The logs do not identify the
underlying network/provider/client root cause.

All atomic operations, including their geometry/rendering and any composed Host
stages, totaled 89.448 s. This accounting differs from the preceding report's
underlying-tool-only timing and must not be compared as the same metric.
Native events retained top-level image blocks, with largest event line 616700
characters and no `chars truncated` markers. This removes the earlier event-log
truncation ambiguity for this run; it does not prove the cause of disconnections.
Model-catalog refresh child-process timeouts were also logged separately.

The user reported local network optimization. This experiment inherits process
network environment and system networking, but its private CODEX_HOME copies only
login credentials, not the main Codex configuration. An asynchronous clarification
was requested about which layer was changed; no network configuration was changed
or assumed in this experiment. Improvements cannot be isolated causally because
tool/context delivery also changed, and no paired success-rate study was run.

## Implementation and validation

See [atomic interface](../codex-atomic-tools.md) for parameters, point freshness,
contact provenance and the additive simulator contract requiring collaborator
review before shared integration. Host geometry uses calibrated RGB-D; simulator
object geometry remains privileged collision infrastructure and is not offered
to the model as grasp proposals or object poses. Orthographic multiview marking
is not included in this first profile; free-space waypoints use point offsets
or explicit coordinates, and observation supports local crops.

80 focused tests passed: 79 Host/atomic/motion/launcher/controller/collision/
attachment checks plus the real MCP stdio test. The stdio test hung under the
restricted sandbox and was interrupted; it passed under an explicit 60 s outer
bound outside that sandbox. No-model canary also passed (see interface document).
All 53 inherited snapshot files and all 12 frozen run source files remained
unchanged during the model episode. No commit, push or external-document update.

## Artifacts

- `tmp/codex-plugin-astra-atomic-01/analysis.json`, `summary.json`.
- `host/atomic-commands.jsonl`: exact model-authored points, poses and timings.
- `host/host-commands.jsonl` and workspace rollout: normal Host stages/receipts.
- `codex-events.jsonl`, `codex-stderr.log`: native calls and client errors.
- `lift-agentview.png`, `final-descent-agentview.png`: retained native feedback.
- `tmp/codex-plugin-astra-atomic-01-setup/source-snapshot.json`, `processes.json`.
- Host session: `5eaf67dde4634cbcaa1b59cf8b239462`.
- Execution: `87d5874d-ae1a-42b0-8a9f-48fff9987dd1`.
- Simulator session: `f34d0861-33fb-4fb8-807d-6add0a1afe13`.

The Host closed at 1200.243 s after episode_timeout. Codex did not finish in the
additional launcher grace period: status=launcher_timeout, elapsed=1410.310 s,
launcher exit=1, integration_passed=false, task_success=false. No final model
answer or completed-turn usage was recorded; empty usage does not mean zero
subscription consumption.

Server PID 3650795 and launcher PID 3650952 are gone; no owned Codex/MCP/worker
process remains, port 18779 is released, and the private auth copy was deleted.
The Host cleanup record says ok=true/skipped=true after the runner interruption;
independent process and port checks confirm termination. Evidence is in
summary.json and process-cleanup.json.
