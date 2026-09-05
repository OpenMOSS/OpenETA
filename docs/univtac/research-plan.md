# UniVTAC tactile-agent research plan

This is the canonical research-plan and status entry point for
`tactile-agent-for-univtac`, updated on 2026-09-05 against the current checkout
and retained run evidence. The design below is the next research direction;
it is not a claim that the full autonomous backend already works.

## Research question

Can a frozen embodied Agent, without fine-tuning, use a few successful
tactile–action–outcome examples in context to make better use of current touch,
operate autonomously, improve UniVTAC native task success, and reduce trial and
error?

Vision and proprioception remain normal operating inputs. Tactile ICL does not
require touch-only control. Visual–action ICL is a control for whether any gain
comes from historical touch rather than simply seeing successful operations.
Prompt ordering, left/right image-patch questions, and opaque button mappings
are not the paper's main problem.

## Target operating loop

Use OpenMOSS/OpenETA's `openeta-for-codex` branch as the design reference and
reuse the existing OpenETA tools, MCP/Gateway/worker separation, and replay
facilities. The target is direct operation:

```text
current vision + bilateral touch + proprioception + operation history
    -> Codex decides the next operation
    -> observe / mark_point / move_to and ordinary gripper commands
    -> executor performs the specified command
    -> new observations and execution feedback
    -> Codex continues, adjusts, retreats, or recovers
    -> UniVTAC native checker evaluates the episode
```

The Agent chooses motion targets, direction, magnitude, orientation, and gripper
operation. IK, trajectory interpolation, and ordinary low-level control may
remain in the executor. They implement the requested motion, not a task solution.
Retain the initialization already included in official evaluation reset/`pre_move`;
from the official policy handoff point, the Agent decides the task body.

The online executor must not call the task expert, use hidden object/hole poses
to compute a correct correction, or automatically perform an expert prefix,
correction formula, or final insertion. Historical `ember`/`slate` expert buttons
are not the main experiment interface. A reusable skill may teach the Agent how
to operate, but must not conceal an online expert solution or take over control.

Allow observation, probing, and recovery within the same episode and shared
budget. These are distinct from restarting failed episodes or selecting the
best attempt. Use OpenETA's minimal `check_task` success boolean; expose no
hidden target error, ground-truth target pose, or correct-action suggestion.
Ordinary feedback about the commanded motion and measured robot state remains
available. The R1.4 general-tool backend completed two unscored control-debug episodes
and three fresh no-demo Codex episodes: all three were natively evaluable,
with autonomous development success 0/3.

## What a demonstration contains

A demonstration is a real successful episode completed through the same
OpenETA operation interface:

```text
task goal + pre-action vision / bilateral tactile short history / proprioception
    -> actual tool call and parameters
    -> actual execution feedback
    -> post-action vision / touch / proprioception
    -> subsequent operations and final native outcome
```

In question–answer terms, Q is the current task, state, recent action, and tactile
change; A is the operation actually performed. The following observations and
outcome show its consequences. Failed attempts and recovery inside a successful
trajectory may remain; do not present every step as optimal.

Deliver actual images or short sequences to the Agent, not just file paths,
array shapes, or a caption. Align bilateral tactile history with each action.
State action units and coordinate frames explicitly, preserving the actual call
and its scene context. New targets must be grounded in the current observation;
examples must not encourage copying world coordinates from a different scene.

Start with a few examples, preferably successful Agent or human operations
through the same interface. Label historical expert and expert-assisted data
separately; do not rename them as autonomous demonstrations. Select examples
from development data, separate from formal test episodes. The current query
must never contain its future correct action, future images, or future outcome.
Historical demonstration outcomes are part of the example, not query answers.

## Core comparison: change only historical examples

All three conditions receive the same current vision, touch, proprioception,
and current operation history. They share the frozen Agent, normal tool
instructions, tools, executor, task starting point, and action/time/context
budget policy. A must understand tool functions without guessing button meanings.

| Condition | Historical context | Current input |
| --- | --- | --- |
| A — no examples | No historical operation demonstrations | Vision + touch + proprioception + operation history |
| B — visual–action control | Historical vision, proprioception, actual actions, execution feedback, and outcomes; no historical touch or touch-derived descriptions/features | Same as A |
| C — tactile–action ICL (main method) | Exactly B's trajectories and non-tactile content, plus action-aligned bilateral tactile history and post-action touch | Same as A |

B/C use identical example sources, counts, actions, outcomes, and non-tactile
content. C versus A tests whether complete successful examples help operation.
C versus B tests whether historical touch adds value. Do not remove B's current
touch: that would conflate the value of current sensing with historical tactile
examples. Actual context and inference costs are recorded, including C's extra
images, under the common budget policy.

A system with no current touch may be a later supplementary baseline; it does
not replace A/B/C. Removing tactile input does not disable contact physics or
change the controller. The older vision-only pilot arms are not automatically
this new B condition.

## Evidence and implementation status

The existing Isaac 5.1 harness provides reusable simulation startup, native
reset/`pre_move`, head/wrist and bilateral tactile capture, proprioception,
Codex MCP communication, actual action traces, and dashboard replay. Exact
Agent inputs are retained in `operator_context.jsonl`; host-only diagnostics
must not be substituted for what the Agent saw.

R1.4 adds a general-tool MCP/Gateway path with a synchronous UniVTAC worker,
robot-target control, four-image observation projection, native checking, and
control/physics counters. The original task-specific and expert-assisted paths
remain historical implementations. Two separate unscored control debug runs
completed: small translation/rotation and close reached; opening remained partial
and was reported as not reached. The focused suite passed 22 tests. Head-camera
marking is supported; wrist geometry is unavailable due to stale camera poses,
while all four RGB/tactile images remain available. The three fresh Terra medium
episodes all ended through native early stop, with native success 0/3. This is
a no-demo autonomous development result, not evidence of tactile ICL benefit. See [Architecture](../architecture.md) for implementation pointers.

### R1.4 autonomous no-demo development result

The two independent control-debug episodes are excluded from this table.
All three formal development episodes had valid reset, available native
checking, and no worker infrastructure error. Each made five non-preview
requests, all of which physically moved the robot. None reached the tool,
control-step, or wall-time budget; the native early-stop condition ended them.

| Seed | Native success | Tools | Control steps | Physics steps | Simulation seconds | Codex wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1000003 | false | 22 | 16 | 32 | 0.267 | 143.034 |
| 1000004 | false | 20 | 7 | 14 | 0.117 | 146.050 |
| 1000005 | false | 16 | 6 | 12 | 0.100 | 111.029 |

These counts start after official initialization. The Agent repeatedly chose
translations along world -Z (1000003/5) or its current TCP +Z (1000004), without
an orientation or gripper change in the formal episodes. The host-only native
checker recorded `inhand_bias` values of 45.77/46.10/46.93 mm, exceeding the
40 mm early-stop threshold. This native indicator is not a measured cumulative
slip path or proof that the object fully fell; failure attribution remains open. Motion-target arrival did not imply insertion success.
See the retained [summary](../../outputs/univtac-isaac51-r14/summary.json) and
[delivery metrics](../../outputs/univtac-isaac51-r14/delivery_metrics.json).

Workers saved and closed normally. All three Codex processes were stopped
after the frozen launcher's 15-second native-terminal grace period; complete
token usage is unavailable, not zero. Seed 1000004 did not call `finish_episode`;
the host finalized its native outcome. The post-batch launcher extends final
reporting grace to 300 seconds within the unchanged 3600-second Codex deadline.
This fix is not a rerun or recovered cost measurement; natural Agent finalization
and complete cost reporting still need verification in a later authorized run.

GPT-6 Pro accepted R1.4 delivery and the unchanged 0/3 result, with limits on
failure attribution. When `gripper` is omitted, the controller currently uses
the measured finger opening as its next target, rather than retaining the
previous commanded closing target. This is confirmed implementation behavior;
its effect on loaded grasp retention and these failures is unverified. Likewise,
pose arrival does not establish velocity settling or contact stability. The
formal trajectories contain no Agent-requested gripper changes, but that does
not exclude a gripper-control contribution to failure.

Touch is observed between tool calls. Control-step logs contain robot targets
and state, not a complete within-action tactile sequence. No real-time tactile
controller or fully reliable contact-control capability is claimed. These
limits are recorded for the next decision; no controller fix or rerun followed
the review. If the controller or sensing interface changes, future B/C must be
compared with A under that same version, not directly with this historical A.

The reusable command uses the existing r09 runtime and pinned Isaac51 source:

```bash
uv run --frozen --extra dev python scripts/univtac/run_autonomous_insert_hole.py \
  --config configs/univtac/autonomous_insert_hole.yaml \
  --mode batch --output-root outputs/univtac-isaac51-r14-new
```

Use a fresh output root. `--mode debug` runs separate unscored controls. These
commands are usage documentation, not authorization to repeat the completed
batch. Replay is at `http://127.0.0.1:9400/r14-autonomous`, served by the local
`univtac-r14-dashboard.service`; raw outputs remain local and are not in Git.

### Historical evidence, with its original scope

These are development results, not one autonomous-success leaderboard. Raw
results and unsuccessful episodes remain unchanged in their original output
roots; the links below refer to local retained artifacts, not a portable dataset.

- **R1.0, native expert:** Pull Out Key succeeded on 3/3 development seeds,
  with nine recorded action transitions
  ([expert summary](../../outputs/univtac-isaac51-r10/expert_summary.json)).
  **Separately, a restricted-skill Codex smoke** selected three native skills
  in one live episode and reached native success
  ([episode](../../outputs/univtac-isaac51-r10/agent/attempt_02/seed_1000000/episode.json)).
  This demonstrates real model-selected actions and feedback, not autonomous
  choice of motion parameters.
- **R1.1, restricted-skill ICL:** Pull Out Key continuation success counts were
  3/3 without examples, 3/3 with correct multimodal examples, 2/3 in the old
  visual-only condition, and 1/3 with swapped action labels. The recorded
  interpretation is `no_detectable_icl_signal`
  ([summary](../../outputs/univtac-isaac51-r11/summary.json)). Native outcomes
  and the report's selection-based gains are different metrics; neither
  establishes a positive tactile ICL result under the new A/B/C design.
- **R1.2, expert-assisted task qualification:** native experts succeeded 3/3
  on both Lift Bottle and Insert Hole. Insert Hole showed two correction
  classes; a one-seed correct/wrong continuation comparison succeeded/failed
  respectively ([summary](../../outputs/univtac-isaac51-r12/summary.json)).
  This motivates a development task, not an autonomous policy claim.
- **R1.3, expert-assisted Insert Hole continuation:** three query experts
  succeeded; 12 Codex choices and 12 continuations brought the total to 15
  simulator episodes. No-demo, correct multimodal, and old vision-only arms
  each succeeded 2/3; swapped labels succeeded 1/3
  ([summary](../../outputs/univtac-isaac51-r13/summary.json),
  [manifest](../../outputs/univtac-isaac51-r13/run_manifest.json)). The Agent
  chose between two opaque procedures; the harness supplied the initial
  downward move, native correction magnitude/z, and final insertion. The
  result does not establish action-ICL or tactile-specific benefit, and is not
  a general-tool autonomous Agent success rate.

R0.9.13–R0.9.18 remain observation/representation diagnostics. By project decision,
R0.9.19–R0.9.21 are historical exploration only: preserve their records, do not
use them to choose the next method, and do not expand that branch. R0.9.21's
capture-only record must not be relabelled as a completed transfer experiment.

## Next work and evaluation

1. R1.4 completed the direct-operation development baseline on fixed seeds
   `1000003`, `1000004`, `1000005`. Review the actual failures before the next
   authorized step. These remain development data, not formal held-out tests.
   The [configuration](../../configs/univtac/autonomous_insert_hole.yaml) keeps
   Terra medium, 30 non-preview move requests, 100 tool calls, and 3600 Codex
   seconds per episode alongside the native control budget.
2. Collect a small set of successful same-interface operations, retaining real
   images, calls, feedback, recovery, and outcomes. No-demo performance need
   not be high before examples may be introduced.
3. Run the A/B/C comparison. The same-interface successful example bank and
   this comparison remain unfinished. R1.4 produced no successful example and
   did not run B/C.
4. Use actual failures to improve tactile representation and experience
   organization, then extend task coverage.

The first representation is real tactile images and short action-aligned
history. Later candidates include baseline/current or difference images,
image-derived structured measurements, Octopi-1.5, VT-MUSE-inspired temporal
representations, and reusable operation skills. These enhance the same tactile
ICL question; none is a prerequisite for direct operation or the first ICL test.
Octopi has not been validated on our simulated touch, and VT-MUSE is not an
integrated, effective component. Image change alone is not verified force,
slip, or stable grasp.

The final target is main results on at least three UniVTAC-Isaac51 manipulation
tasks; candidates are Insert Hole, Pull Out Key, and Lift Bottle. Use a fixed
formal test batch independent of development and example selection. The primary
metric is native task success rate. Record action count, recovery behavior,
completion time, context usage, and inference cost as secondary metrics. Report
expert success, expert-assisted continuation success, and autonomous Agent
success separately, not in one main-result table.

All conditions use the same Isaac51 task implementation. These results cannot
be presented as direct reproductions of FTP-1/Isaac 4.5 numbers. See
[Isaac 5.1 compatibility boundary](isaac51_compatibility_boundary.md) and
[Vendor Notes](../vendor-notes.md). This documentation update does not launch or
authorize execution of the next experiments.
