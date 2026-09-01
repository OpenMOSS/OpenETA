# LIBERO Mink controller endpoint canary — 2026-08-17

## Outcome

This endpoint-only canary supports continuing the Mink controller exploration.
Mink driving robosuite `JOINT_VELOCITY` improved the final target precision over
the current `OSC_POSE` loop on the paired cases below. It is not yet a production
controller decision: collision avoidance, manipulation contact, dependency
upgrades, and sim-to-real geometry were deliberately excluded and remain the
four follow-up boundaries.

The Agent-facing `move_to` schema was not changed.

## Question and isolation

The question was whether changing only the motion solve/tracking layer improves
the final EEF endpoint. The experiment therefore excluded the Agent, camera
rendering, perception, grasp selection, object attachment, collision checking,
and reward-based task completion.

- Environment: LIBERO `libero_object`, task 2, `pick up the salad dressing and
  place it in the basket`.
- Object-layout seeds: 2, 3, and 4.
- Arm starts per seed: nominal, left-high, and right-high. The two elevated
  states were generated once and their complete flattened MuJoCo states were
  replayed unchanged to both controllers.
- Targets per arm start: two translation-only reaches and three coupled
  position/orientation reaches, including a pose from an earlier grasp rollout.
- Total: 45 paired cases per backend. Seeds change object layout but not the
  Panda start configuration; because this canary contains no contact or
  collision constraints, the control measurements repeat across the three
  seeds. They are retained to verify that object-layout reset differences do
  not leak into this isolated result, not counted as independent kinematic
  diversity.
- Control rate: 20 Hz; maximum 100 policy steps.
- Success: maximum per-axis position error below 2 mm and, when requested,
  orientation error below 0.05 rad.

Each pair begins from the exact same simulator state and uses the same absolute
target. Both controllers use actual EEF feedback after execution steps.

## Compared controllers

### OSC baseline

The baseline reproduces the current server behavior: robosuite `OSC_POSE`,
normalized translation scale 0.05 m, rotation scale 0.5 rad, pose refresh every
step for position-only commands, and every three steps for full-pose commands.

### Mink candidate

- Mink 0.0.6 with the environment's existing MuJoCo 3.3.0;
- `FrameTask(position_cost=1, orientation_cost=1, gain=0.8)`; orientation cost is
  zero for translation-only cases;
- weak initial-posture regularization with cost `1e-3`;
- configuration limits and 0.5 rad/s arm-joint velocity limits;
- `quadprog` through qpsolvers 4.13.0;
- robosuite `JOINT_VELOCITY` as the dynamics-level tracker.

Mink 0.0.6 is only a compatibility pin for this canary. The current LIBERO venv
uses MuJoCo 3.3.0, whereas recent Mink versions require a newer MuJoCo. This run
does not decide the production dependency strategy.

## Aggregate result

| Metric | OSC | Mink |
|---|---:|---:|
| Target reached | 39 / 45 | **42 / 45** |
| Mean position error, all cases | 3.361 mm | **2.792 mm** |
| Mean position error, cases both reached | 2.036 mm | **1.325 mm** |
| Mean position error, oriented cases | 4.283 mm | **3.655 mm** |
| Mean orientation error, oriented cases | 1.439° | **0.119°** |
| Mean orientation error, oriented cases both reached | 0.955° | **0.109°** |
| Maximum orientation error | 5.308° | **0.202°** |
| Mean control steps | 48.4 | **31.3** |
| Total controller runtime | 14.88 s | **9.99 s** |
| Solver/runtime errors | 0 | 0 |

Across the 45 exact pairs, Mink had lower position error in 36, lower
orientation error in all 27 oriented pairs, and fewer control steps in 39.
Thirty-nine pairs were reached by both controllers, three were reached only by
Mink, none only by OSC, and three by neither.

## Per-start result

The table shows one row per unique kinematic pair; the three object-layout seeds
produced the same values. Position is Euclidean error in millimetres. `—` means
orientation was not controlled.

| Start | Target | OSC pos | Mink pos | OSC ori | Mink ori | OSC / Mink steps | OSC / Mink reached |
|---|---|---:|---:|---:|---:|---:|---:|
| nominal | translation short | 0.959 | **0.602** | — | — | 10 / **8** | yes / yes |
| nominal | translation long | 2.162 | **0.926** | — | — | 32 / **18** | yes / yes |
| nominal | yaw 30° | 1.571 | **1.466** | 1.504° | **0.098°** | 33 / **32** | yes / yes |
| nominal | pitch 25° + yaw 35° | 2.438 | **0.333** | 0.614° | **0.081°** | 54 / **45** | yes / yes |
| nominal | recorded grasp pose | 2.422 | **1.435** | 0.536° | **0.108°** | 69 / **44** | yes / yes |
| left-high | translation short | 1.726 | **0.936** | — | — | 9 / **8** | yes / yes |
| left-high | translation long | 3.056 | **2.174** | — | — | 100 / **15** | no / **yes** |
| left-high | yaw 30° | 2.440 | **1.660** | 0.649° | **0.161°** | 51 / **29** | yes / yes |
| left-high | pitch 25° + yaw 35° | **20.895** | 22.478 | 5.308° | **0.202°** | 100 / 100 | no / no |
| left-high | recorded grasp pose | 2.466 | **1.861** | 0.549° | **0.192°** | 63 / **29** | yes / yes |
| right-high | translation short | **1.287** | 2.875 | — | — | 9 / **7** | yes / yes |
| right-high | translation long | 2.681 | **1.478** | — | — | 49 / **18** | yes / yes |
| right-high | yaw 30° | 2.614 | **1.296** | 0.668° | **0.126°** | 57 / **29** | yes / yes |
| right-high | pitch 25° + yaw 35° | **1.332** | 1.583 | 2.601° | **0.056°** | **30** / 58 | yes / yes |
| right-high | recorded grasp pose | 2.368 | **0.778** | 0.519° | **0.045°** | 60 / **29** | yes / yes |

## Reachability follow-up for the common failure

The left-high `pitch 25° + yaw 35°` target was checked independently of
both controllers against the exact MuJoCo forward kinematics and Panda joint
limits. This separates target feasibility from controller convergence:

- 64 joint-limit-respecting random-start least-squares solves could reach the
  requested position alone to numerical precision, and could separately reach
  the requested orientation to numerical precision.
- No random-start solve reached the full pose within the canary thresholds.
- A differential-evolution minimax search used 140,343 FK evaluations. Its best
  result still had 3.480 mm maximum-axis position error and 0.0870 rad (4.98°)
  orientation error, for a normalized worst-constraint score of 1.740; a score
  at or below 1 would satisfy both thresholds.
- Constrained optimization found that enforcing maximum-axis position error at
  2 mm leaves at least 0.0986 rad (5.65°) orientation error. Conversely,
  enforcing orientation error at 0.05 rad leaves at least 10.93 mm maximum-axis
  position error.

This is strong numerical evidence that the combined 6-DoF target is
kinematically infeasible at the requested tolerances, rather than merely short
of controller iterations. It is not a formal analytic infeasibility proof, but
the two constrained Pareto-bound checks make a missed local IK solution
unlikely. The result also exposes a current harness gap: `ik_preview_check` is
registered as an interface, but normal runtime assembly has no real simulator
reachability backend; the implementation in `bind_dummy_tool_handlers` always
returns feasible and is intentionally only a placeholder.

## Interpretation

1. The precision hypothesis is supported. The clearest improvement is
   orientation tracking: roughly a 12x reduction in mean oriented error, with
   every paired orientation result improved.
2. The improvement is not merely a larger step budget. Mink used fewer steps on
   most pairs and converted the left-high long translation from an iteration-limit
   failure into a reached target.
3. Mink is not uniformly better on position. Two right-high targets ended with
   slightly lower OSC position error, although both controllers reached them.
4. The common left-high coupled failure is important negative evidence. Mink
   drove orientation to 0.20° but retained 22.5 mm position error, showing how a
   local weighted IK task can trade objectives near an infeasible or poorly
   conditioned target. It does not remove the need for reachability, cost policy,
   or longer-range planning.
5. No conclusion about collision safety or grasp stability is licensed by this
   run. Those questions require the separately discussed collision-pair/contact
   policy, attached-object geometry, dependency, and sim-to-real work.

The next architectural discussion can therefore treat “Mink improves endpoint
tracking enough to justify integration work” as a positive canary result, while
keeping OSC available until the four omitted boundaries have been resolved and
an Agent-level manipulation A/B test passes.

The subsequent worker-local phase-1 integration and capability receipts are
documented in
[`controller-capability-and-mink-integration.md`](controller-capability-and-mink-integration.md).
It preserves OSC as the default and deliberately rejects collision-enabled or
confirmed-attached-object Mink motion until the phase-2 safety boundary is
implemented.

## Reproduction

Runner: [`scripts/libero_mink_canary.py`](../scripts/libero_mink_canary.py)

The canary used an isolated temporary dependency directory so the shared LIBERO
venv was not modified:

```bash
sim/venvs/libero/bin/python -m pip install \
  --target /tmp/openeta-mink-canary-min --no-deps \
  mink==0.0.6 qpsolvers==4.13.0 quadprog==0.1.13

PYTHONPATH=/tmp/openeta-mink-canary-min \
MPLCONFIGDIR=/tmp/openeta-mpl \
LIBERO_DIR=/tmp/LIBERO \
sim/venvs/libero/bin/python scripts/libero_mink_canary.py \
  --mink-path /tmp/openeta-mink-canary-min \
  --seeds 2 3 4 \
  --output /tmp/openeta-libero-mink-canary-profiles-20260817.json
```

The raw result produced by the recorded run had SHA-256
`32ba153a787a14da4b92dd204194877b906bd8786390e74d8b587b206027e5a7`.
