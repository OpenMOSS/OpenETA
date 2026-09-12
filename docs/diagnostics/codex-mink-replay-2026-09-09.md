# Mink replay of the Astra OSC failure — 2026-09-09

## Result

The existing Mink controller reached the recorded Astra02 candidate-12 target
in **84 of 150 steps**, with **0.481 mm Euclidean position error**, **0.471 mm
maximum-axis error**, and **0.049962 rad orientation error**. The original
criteria, less than 2 mm on every position axis and less than 0.05 rad angular
error, were both satisfied. No controller or native ingress code change was
needed. This is a no-model endpoint replay, not an autonomous manipulation task
or a full-episode success result.

| Execution | Steps | Position norm | Maximum-axis error | Orientation error | Reached |
| --- | ---: | ---: | ---: | ---: | --- |
| Recorded/replayed OSC | 150 | 25.126 mm | 21.363 mm | 0.089525 rad | No |
| OSC after continuation | 600 total | 4.161 mm | 3.445 mm | 0.016945 rad | No |
| Mink through native IK hook | 84 | 0.481 mm | 0.471 mm | 0.049962 rad | Yes |

The Mink angular residual is just below the requested threshold; this does not
show tighter orientation accuracy or stability after a holding interval. The
single target supports trying Mink for the next Codex harness episode, but does
not establish a general success rate. OSC evidence and limitations are recorded
in [the previous convergence diagnostic](codex-reference-and-convergence-2026-09-09.md).

## Comparable conditions and actual execution

- Isolated checkout `OpenETA-codex-plugin`, branch
  `dev/huaizezheng/codex-plugin-smoke-2026-09-08`, base HEAD
  `723bed12721910e6bcc8db783cb9c1c16cfddaa1` plus the recorded snapshot/prototype.
- LIBERO spatial task 0, seed 0, 512 x 512 images; original benchmark physics.
- Target world XYZ `[0.083477230853, 0.085563528763, 1.149682016687]`;
  quaternion xyzw `[-0.6753626981637774, -0.6842080618795365,
  -0.054252220451405625, 0.2698170687041185]` from the prior replay's IK record.
- Measured initial EEF position differs from the OSC replay by only
  `2.758e-13 m`. Same tolerance and initial 150-step budget; no continuation
  was needed. The prior OSC replay used direct simulator motion; this run used
  the native Host hook. OSC does not consume the IK joint seed, while Mink does.
  Thus this compares the existing execution paths, not solvers stripped of their
  normal integration behavior.
- The native `move_to(target_pose)` request executed ordinary
  `propose_motion_target -> ik_preview_check -> move_to` Host steps.
- Receipt: `controller_id=mink.robosuite_joint_velocity`,
  `goal_executor=openeta.worker_mink_goal.v1`, `configured_name=JOINT_VELOCITY`,
  `execution_policy=ik_preview_seeded_cartesian_goal`. The execution seed ID
  matches fresh IK receipt `f07b64c2cc961b270926`; no OSC fallback occurred.
- Collision checking stayed enabled. The worker reported self, endpoint,
  trajectory and world checks, no detected collision; the Host coverage receipt
  was `trajectory_and_world`, `coverage_complete=true`. This is the declared
  worker per-step check scope for this segment, not a guarantee for a later route.
- Nominal joint velocity limit was 0.5 rad/s. Default motion execution condition
  was retained; no special stabilization or physics ablation was introduced.

The complete native request took 20.230 s. Rollout start/end events give 0.308 s
for IK and 12.965 s for the move tool, including its transport. The original
OSC direct 150-step replay took 20.007 s. These are observed single-run timings,
not a controlled throughput benchmark. The simulator-call limit remained 120 s.
One native request used three Host turns/tool calls. No grasp close, lift,
placement, perception-service request or model inference was performed.

## Configuration and dependencies

The simulator process was started with:

```bash
OPENETA_LIBERO_CONTROLLER_PROFILE=mink_joint_velocity
OPENETA_LIBERO_MINK_DEPENDENCY_PATH=/media/user/B29202FA9202C2B91/Stage2-OpenETA/OpenETA-codex-plugin/tmp/codex-mink-deps
```

These are simulator/worker deployment settings, not Codex model arguments or
Agent-controlled tool parameters. A fresh environment is needed after selecting
the controller. The global/default OSC setting was not edited.

The minimal existing `/tmp/openeta-mink-canary-min` overlay was copied into the
isolated checkout. It contains Mink 0.0.6, qpsolvers 4.13.0 and quadprog 0.1.13;
MuJoCo 3.3.0 and NumPy 1.26.4 remain supplied by the clone's LIBERO environment.
Imports passed before execution. `dependency-manifest.json` records file hashes;
no packages were installed into the original checkout or shared overlay.

## Artifacts and validation

Output: `tmp/codex-mink-replay-01/`.

- `run.py`: bounded native Host replay and automatic environment/server cleanup.
- `source.json`: previous original-physics replay record containing target and
  initial-state provenance.
- `report.json`, `response-0.json`: measured outcome and complete native feedback.
- `verification.json`: exact controller/seed assertions, precision assertions,
  step timings, source hashes, inherited-file and process/port checks.
- `host/host-commands.jsonl` and session rollout: all three internal tool steps.
- Host session `284d64765f3d4b079235a55a8fe1fb8e`.

The result and matching seed were asserted, and all 53 inherited snapshot files
retain their hashes. The dedicated simulator and clone worker exited; port
18778 has no listener. No production code changed, so no code regression suite
was repeated: validation was dependency import plus the live integration replay.
Only local experiment/setup documentation and the review patch index were
updated. The original checkout, other agents' processes and shared collaboration
documents were not changed. No commit or push was made.
