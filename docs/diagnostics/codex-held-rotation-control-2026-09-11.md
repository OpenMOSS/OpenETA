# Deliberate held-object rotation: task weight and bounded horizon

## Confirmed failure mechanism

Object 7 attempt 001 ultimately passed, but two intended held rotations stopped at the Cartesian divergence guard with 90.302 / 89.968 mm position errors after 10 / 15 steps. Object 4 attempt 003 had the same guard after 8 steps at 101.178 mm. All three failures were reproduced exactly from their private saved states with original target, IK seed, physics, collision checks, velocity limits and stable-arrival rules.

The earlier grip stabilization multiplied orientation task cost by 25 for every attached-object movement. This improves orientation retention during translation, but for a large *requested* rotation a velocity-limited QP trades excessive Cartesian displacement for angular progress. A matched nine-run ablation varied only the FrameTask orientation cost (production cost 500, intermediate 100, baseline 20). Worker receipts in that ablation still describe production multiplier 25; actual setter costs are explicitly recorded, so do not read those receipts as the ablated weight.

Evidence: `tmp/codex-held-rotation-weight-20260911/replay.py`, `report.json`, `replay.log`. No model calls or task-performance claims.

| Saved rotation | Old multiplier 25 | Baseline multiplier 1, same 150-step cap |
|---|---|---|
| Ketchup, Object 4 #54 | 8 steps; 101.178 mm drift; guard | 150-step cap; max position error 4.756 mm; 20.869 deg remaining |
| Milk, Object 7 #27 | 10 steps; 90.302 mm drift; guard | reached in 91 steps; max position error 3.532 mm; final 2.626 deg |
| Milk, Object 7 #30 | 15 steps; 89.968 mm drift; guard | reached in 42 steps; max position error 2.166 mm; final 1.904 deg |

All baseline-cost variants retained bilateral pad contact; this alone does not prove retention, so the ablation also measured held-object relative-position drift. The two milk runs had relative drift 0.227 / 0.082 mm.

## Ketchup was still converging at the cap

Its angular error decreased from 65.058 deg at step 50 to 40.824 at 100, 32.261 at 120, 24.486 at 140 and 20.869 at 150; actual minimum arm limit margin at step 150 was 0.449 rad. A separate 300-step probe using the **actual repaired controller without monkeypatches** reached in 203 steps: final position error 0.218 mm, angle error 2.429 deg, bilateral pads and stable arrival. Maximum position error stayed 4.756 mm. Final minimum joint margin was 0.01144 rad, so the endpoint remains near a limit. Reachability here means the existing 2 mm per-axis / 0.05 rad tolerances; it does not establish arbitrary stricter exact-pose feasibility.

## Implementation

- `sim/controllers/fixture_grip.py`: keep multiplier 25 only within the existing 0.05 rad orientation-hold band. Intentional attached-object reorientation uses baseline multiplier 1 while keeping the same low carry speed and stable arrival. Fixture gating remains unchanged.
- `sim/controllers/mink_goal.py`: active grip stabilization follows the resolved stabilization kind, since its speed/arrival controls can remain active when angular multiplier is 1.
- `tools/codex_atomic.py`: strict moves with existing possible-load state and orientation change >0.05 rad pass an explicit 300-step cap through the existing checked motion hook. Translation, empty-hand and symmetric moves retain the 150-step default. Possible-load state is not an assertion that an object is retained. No new tool, agent-settable authority, IK bypass, collision exemption or tolerance relaxation.

## Validation

Actual production integration: `tmp/codex-held-rotation-integrated-20260911/` contains six saved-state runs without task-cost monkeypatches: ketchup 150 and 300, both milk rotations, the earlier 130-step milk carry, and a drawer pull. All recorded expectations and cleanup passed. The carry still reached in 130 steps at 0.070 deg; the drawer pull still reached in 8 steps, bilateral contact and stable arrival. The 150-step ketchup case intentionally remains a cap failure in the report.

Final targeted tests: 94 passed in 6.68 s (`test_codex_atomic`, `test_codex_motion_hook`, `test_codex_orientation`, `test_fixture_grip_stabilization`, `test_codex_experiment_controller`, `test_codex_failure_feedback`); the LIBERO interpreter also passed both real-frame tests in 2.48 s. An earlier Python 3.13 suite had one MuJoCo-dependent skip, covered by the latter interpreter check. `git diff --check` and plugin validation passed.

Plugin version `0.1.0+codex.20260910174006`. Existing `num_steps` and receipt fields are reused; private branch only, no main merge, no external publication. Record the changed bounded execution policy for collaborator review before main merge. Campaign official completion remains Object 0–7 (8/20); these controller probes add no task successes. Resume Object 8–9 then Goal 0–9 with unchanged startup-wait fix and original remote model catalog. Pending user preference about retry memory remains unimplemented.
