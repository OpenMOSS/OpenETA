# Codex move_to timeout diagnosis — 2026-09-09

## Finding and scope

Replaying the recorded first motion revealed two distinct problems:

1. Intermediate simulation steps were converting and encoding full camera
   images despite `render=false`. The unmodified 150-step motion took 170.130 s,
   longer than the Host's 120-second simulator-call timeout.
2. The OSC controller did not converge to this full-pose target. After 150 steps
   it reported `iteration_limit`, with maximum-axis position error 19.124 mm
   against a 10 mm tolerance. Increasing receive time alone did not reach it.

The intermediate-image fix reduced the same motion to **19.454 s**, a measured
**8.745× speedup**. It returned the explicit non-convergence receipt within the
unchanged 120-second timeout and closed the environment successfully. It does
not solve the controller's target-tracking problem or complete the pick task.

All work stayed in `OpenETA-codex-plugin`, branch
`dev/huaizezheng/codex-plugin-smoke-2026-09-08`. No changes were made in the
original `OpenETA` checkout. All 53 inherited snapshot hashes still match.
No model inference, paid API call or subscription allowance was used by these
three deterministic simulator replays. No commit, push or shared-doc write.

## Source evidence and reproduction conditions

The source is `tmp/codex-plugin-pick-01/host/host-commands.jsonl`, execution
`68409d08-c802-44d2-b00e-c3e6719f7fd4`, session
`537ae3ce990d480c88ebaafc62c02763`. Its sole dispatched `move_to` failed after
120.098 s with `simulator_mcp_transport_timeout`; the earlier blocked request
did not actuate. The original run did not retain worker progress logs, so this
follow-up does not claim to reconstruct every historical worker step.

Replays used a fresh LIBERO spatial task 0, seed 0, 512×512 images, identical
recorded target, tolerances and `enable_collision_check=true`. The initial EEF
position/orientation matched the original source evidence. Perception and IK
from the original run had not moved the robot and were not rerun. Target:

```json
{
  "x": -0.001673983204,
  "y": 0.200696597374,
  "z": 1.052078551395,
  "roll": -138.26310738572542,
  "pitch": 0.21383859201016858,
  "yaw": 101.07999746768694,
  "tolerance": 0.01,
  "ori_tolerance": 0.2,
  "enable_collision_check": true
}
```

The declared executor was `robosuite.osc_pose` /
`openeta.outer_closed_loop_cartesian.v1`, the default selected when
`OPENETA_LIBERO_CONTROLLER_PROFILE` is unset. It was not the worker-local Mink
executor and did not consume a preview joint seed. Its existing `num_steps=150`
default was retained. These are 150 internal simulation steps in **one** tool
call, not 150 agent tool calls. The collision receipt reported checking
unavailable because cuRobo was absent; a false detected flag is not proof of
collision clearance.

A dedicated loopback server at port 18778 logged worker HTTP request start/end
times, returned robot state and error status. Diagnostic-only wrappers enabled
periodic faulthandler stack snapshots without altering control computations.
The baseline and encoding-only ablation allowed the diagnostic receiver to
wait 360 s to observe the final receipt. The production timeout was not changed.
The final fixed-code replay used a 120 s receiver timeout.

## Measurements

| Condition | Motion wall time | Mean internal step | Internal steps | Final receipt |
| --- | ---: | ---: | ---: | --- |
| Original worker | 170.130 s | 1.124 s | 150 | `iteration_limit`, 19.124 mm max-axis error |
| Diagnostic ablation: omit intermediate PNG encoding only | 142.013 s | 0.937 s | 150 | Same |
| Fixed worker: omit intermediate pixel conversion and encoding | 19.454 s | 0.118 s | 150 | Same |

The original worker completed only 106 motion steps in the first 120 s. All
recorded worker step requests returned without an error. Baseline and fixed
trajectories matched to a maximum per-coordinate difference of
`2.550e-12 m` across all 150 EEF samples. All three replays returned reward 0
and successful environment-close receipts. These are controller diagnostics,
not autonomous task-performance samples or throughput guarantees under all loads.

Baseline convergence samples:

| Internal step | Maximum-axis position error | Orientation error |
| --- | ---: | ---: |
| 1 | 205.63 mm | 1.8235 rad |
| 15 | 82.97 mm | 0.8107 rad |
| 30 | 25.45 mm | 0.2281 rad |
| 60 | 19.13 mm | 0.0879 rad |
| 90 | 22.85 mm | 0.1037 rad |
| 120 | 18.91 mm | 0.0895 rad |
| 150 | 19.12 mm | 0.0910 rad |

Position error plateaus/oscillates rather than continuing steadily toward the
10 mm threshold; orientation meets its 0.2 rad threshold. This establishes OSC
non-convergence for the tested pose, but does not alone identify its physical
cause (coupling, contact, control gains or other controller limitations).
Changing controller policy or validating the Mink execution route is separate
work. This follow-up did not loosen tolerances or fabricate a successful reach.

## Why the timeout and close failure occur

`sim/bench_worker.py::_step_with_image` used `render=false` only to skip its
additional `_inject_render_frame` call. Backends can still include cameras in
the observation produced by `env.step()`. The worker always fed that observation
through `EnvObservation.from_dict` and `StepResult.to_mcp_dict`.

The baseline stack samples landed in camera PNG encoding and NumPy pixel-array
conversion under `adapter/protocol.py`. When the diagnostic ablation removed
only encoding, stack samples exposed `_to_int_list3d` in `CameraFrame.from_dict`:
intermediate RGB/depth pixels were still being converted one by one. This is why
encoding removal alone saved only part of the total cost.

The recorded cleanup timeout is also consistent with the nested lifetimes:
`move_to` holds `_serialized_env_control`'s per-environment lock, and `close_env`
requires that same lock. The blocking worker operation can continue after the
120 s MCP client wait expires (worker HTTP timeout defaults to 600 s). Episode
close waits at most 30 s, so a roughly 170 s motion can outlast both waits.
The replay proves continued worker progress; the exact historical close-lock
wait was not logged and remains an inference from the source and timings.

## Changes and validation

- `sim/bench_worker.py`: for non-rendering, nonterminal intermediate steps,
  remove cameras from a shallow transport-only observation **before**
  `EnvObservation.from_dict`. Keep the full raw `_last_obs` cache, robot state,
  rewards and step physics unchanged. Rendered, terminated and truncated
  observations keep images; cached terminal receipts remain replayable.
- `tools/codex_host.py` and `scripts/codex_plugin_smoke.py`: episode default
  600 → **1,200 s**; request/turn/tool-call defaults 40 → **80** (the original
  live pick used only 30). Record configured limits in status and summaries.
  Single simulator-call timeout stays **120 s**. CLI explicit overrides still
  take precedence; the observation-only example intentionally uses a small budget.
- `tests/test_worker_intermediate_observation.py`: four cases exercise skipped
  intermediate camera conversion, retained raw cache/robot state, periodic
  images and both terminal paths with no repeated actuation.

Validation performed: 21 Codex Host/launcher tests; 6 worker intermediate-image
and terminal-result tests; 143 existing simulator proxy, timeout and motion
reconciliation tests. **170 passed** in total. The MCP stdio test ran outside
the tool sandbox, as required by the existing environment limitation. No full
repository suite or new end-to-end Codex task run is claimed.

All three dedicated server process groups and workers were retired after the
replays; port 18778 was released. Other agents' processes/services were untouched.

## Local artifacts

Under the checkout's ignored `tmp/` directory:

- `codex-motion-diagnosis-2026-09-09/`: baseline replay report, exact move
  arguments, `server_probe.py`, `replay.py`, worker stack snapshots and proxy log.
- `codex-motion-ablation-2026-09-09/`: encoding-only ablation report and logs;
  its temporary worker hook is `codex-motion-diagnosis-2026-09-09/worker_probe.py`.
- `codex-motion-fixed-2026-09-09/`: final replay report and logs;
  `comparison.json` contains per-step errors/timing and trajectory comparisons.

The scripts are local diagnostic artifacts, not a new supported simulator
entry point. Replaying the pre-fix conditions requires the pre-fix worker source;
running their baseline script against the current checkout uses the fix.
The review patch in `tmp/codex-plugin-baseline/prototype.patch` contains the
prototype and follow-up changes, excluding the inherited unrelated edits.
