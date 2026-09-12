# Gripper final-render optimization — 2026-09-09

## Change

Both `gripper_close` and `gripper_open` now send their whole settling batch with
`render=false`, then request one final render before returning. The existing
60-step close and 40-step open horizons, gripper command latch and physics
actions are retained. This builds on the intermediate camera-projection fix in
`sim/bench_worker.py`.

The implementation is `_step_gripper_with_final_observation` in
`sim/mcp_server/server.py`. It runs under the existing per-environment control
lock, preserving the step's reward, termination flags and info while replacing
only the final observation. The final image also updates the server cache.
Terminal steps already preserve their exact visual evidence and do not trigger
an extra render; failed steps are returned unchanged. A failed final render is
reported explicitly while retaining the actuation result, instead of claiming
that an old image is fresh.

This removes repeated explicit render injection, pixel conversion and PNG
encoding/transport. A backend may still render cameras internally during
`env.step()`; this change does not reconfigure backend sensor implementations.

All code remains in the independent `OpenETA-codex-plugin` checkout on
`dev/huaizezheng/codex-plugin-smoke-2026-09-08`. The original `OpenETA` checkout
was not modified. No model call, subscription inference, commit or push was
performed in this optimization turn.

## Validation

28 focused tests passed: gripper latch/horizon tests, final-observation tests,
worker intermediate-image and terminal-result tests, and simulator proxy tests
selected with `-k gripper`. The new tests cover:

- one camera-free batch followed by exactly one render;
- retained reward/info/terminal flags and updated final-image cache;
- terminal, truncated and failed steps bypassing the extra render;
- final-render failures preserving the actuation receipt and reporting error.

A no-model live check replayed the previous attempt's recorded approach targets
in a fresh LIBERO spatial task 0, seed 0, with 512×512 images, then closed and
opened the gripper. The approach retained the recorded collision-disabled
condition; this is a performance/transport check, not a collision-safety test
or an autonomous grasp-success evaluation.

| Operation | Duration | Settling steps | Final RGB images |
| --- | ---: | ---: | ---: |
| Recorded approach | 13.484 s | Recorded two-waypoint budget | — |
| Close | **2.599 s** | 60 | 3 decoded 512×512 images |
| Open | **2.189 s** | 40 | 3 decoded 512×512 images |

The proxy trace confirms that each gripper operation sent exactly one `/step`
request with `render=false`, followed by exactly one `/render` request. Reward
remained 0 and both operations returned their normal gripper actuation receipts.
Environment close returned `ok=true`.

The preceding full-task attempt spent 91.252 s on gripper close. That number is
a historical comparison, not a paired timing average: the final gripper aperture
and robot state differed slightly in this fresh replay, so these measurements
do not prove identical historical grasp contact. An additional attempt to set
up a matched legacy-render comparison failed to reach its approach target and
stopped **before gripper actuation**. It was excluded from the timing conclusion,
its environment closed successfully, and no paired speedup is claimed.

Primary artifacts:

- `tmp/codex-gripper-final-render-2026-09-09/report.json` and `replay.py`.
- The same directory's `proxy-events.jsonl`, worker logs and decoded final PNGs.
- `tmp/codex-gripper-render-ab-2026-09-09/legacy/report.json` records the aborted
  supplemental comparison, including its failed approach and cleanup.

Dedicated server groups `3302271` and `3306395` were retired after the checks;
their workers and port 18778 were verified released. No new full Codex task
attempt was run after this fix.

## Placement clarification

The previous full-task attempt did **not** call AnyPlace. The main runtime has
an `anyplace` handler integration, but this minimal plugin's `PICK_TOOLS`
admission subset does not expose it to Codex.

The observed route was SAM3 plate segmentation, model-authored world-frame
target poses, IK checks, `move_to`, and a gripper-open request. Sol estimated
the destination center and authored a transit target `(0.084, 0.187, 0.983)`
and a descent target `(0.084, 0.187, 0.94)`. These were not AnyPlace-generated
placement candidates or verified stable placement poses. The moves returned
36.8 mm and 21.9 mm position errors respectively; release was then cancelled
by the episode deadline. Final release/settling and task success were not
established. This gripper optimization does not add AnyPlace exposure.
