# UniVTAC tactile-agent research plan

This is the canonical research-plan and status entry point for
`tactile-agent-for-univtac`, updated on 2026-09-06 against the current checkout
and retained run evidence. The design below is the next research direction;
it is not a claim that all planned capabilities already work.

[Related Work](related-work.md) and its [BibTeX](related-work.bib) collect the
primary references, reading depth, and design implications. R1.5 implements
segment-end tactile-change clip selection, with local image difference as a
control. This is neither slip recognition nor an online motion
interrupt. Event detection supports observations and example organization
within the tactile ICL question, without changing the A/B/C comparison below.

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
and complete cost reporting were subsequently verified in R1.5 below; R1.4
usage remains unavailable.

GPT-6 Pro accepted R1.4 delivery and the unchanged 0/3 result, with limits on
failure attribution. In the frozen R1.4 implementation, when `gripper` was omitted, the controller used
the measured finger opening as its next target, rather than retaining the
previous commanded closing target. This is confirmed implementation behavior;
its effect on loaded grasp retention and these failures is unverified. Likewise,
pose arrival does not establish velocity settling or contact stability. The
formal trajectories contain no Agent-requested gripper changes, but that does
not exclude a gripper-control contribution to failure.

In R1.4, touch was observed between tool calls. Control-step logs contain robot targets
and state, not a complete within-action tactile sequence. No real-time tactile
controller or fully reliable contact-control capability is claimed. These
limits describe R1.4. R1.5 changes command retention and observation delivery as
recorded below; the old batch has not been rerun or relabelled. If the controller or sensing interface changes, future B/C must be
compared with A under that same version, not directly with this historical A.

The reusable command uses the existing r09 runtime and pinned Isaac51 source:

```bash
uv run --frozen --extra dev python scripts/univtac/run_autonomous_insert_hole.py \
  --config configs/univtac/autonomous_insert_hole.yaml \
  --mode batch --output-root outputs/univtac-isaac51-r15-new
```

Use a fresh output root. `--mode debug` runs separate unscored controls. These
commands are usage documentation, not authorization to repeat the completed
batch. Replay is at `http://127.0.0.1:9400/r14-autonomous`, served by the local
`univtac-r14-dashboard.service`; raw outputs remain local and are not in Git.

### R1.5 grasp commands and segment-end tactile history

R1.5 reuses the general tools, synchronous worker, native dynamic controller,
and r09 runtime. `GripperTargets` reads the last submitted two-finger command
from `Articulation._joint_pos_target_sim` after official reset returns. A narrow
adapter replaces only the scalar gripper dispatch inside the counted native
qpos action. It preserves the two targets independently; omitted gripper
commands never replace them with measured opening. Explicit close/open sets
and retains the native targets. No extra closing motion happens at takeover.

`arm_reached` is independent of gripper closure. A new gripper command gets a
finite wait: target error at most 1 mm, or both positions vary by at most 0.2 mm
in 0.10 simulation seconds, with a 40-control-step maximum wait. These settings
were retained after unscored debugging, not selected from formal outcomes.
Waiting ends without claiming stable grasp; subsequent moves retain the target.

One fresh no-Codex debug episode completed in
`outputs/univtac-isaac51-r15/debug/seed_1000003`: seven physical requests,
28 native control steps, 56 physics steps, and 29 four-view samples including
takeover. Inherited targets were 0.005721318535506725 and
0.005722052417695522 m in native joint1/joint2 order. Translation/rotation
preserved them. Close used six steps, and its zero targets persisted through
an ensuing translation; open used 16 steps in the disposable debug state.
Loaded stiffness/damping/effort limits were read as 2000/100/200 for both fingers
and were not changed. This debug is not part of task success evaluation.

Sampling reads the existing refreshed buffers after each native qpos control
step, copies arrays before buffer reuse, and writes raw frames independently
of offline video encoding. Debug intervals were 1/60 simulation second, with
matching four-sensor timestamps and consecutive frame counters. This is not
60 wall-clock frames/second or proof of zero renderer latency. No additional
physics was used for recording. The four-panel H.264 video and raw frame index
are retained for review; model thinking occurs while physics is paused.

The implemented detector is small fixed-ROI integer patch matching plus a
same-frame image-difference control, not a reproduction of a full slip method.
Each pad has independent quality and scores; low quality is explicitly marked
as image-difference fallback. Debug had one such left-pad transition among 28.
The initial thresholds (2 original-image pixels, difference 0.015, quality 0.35)
were retained after inspecting debug data. At segment end, up to four shared
times select before/preceding-peak/peak/latest frames, sorted and deduplicated.
Two labelled tactile strips plus current head/wrist use native MCP images;
`operator_context.jsonl` records the actual delivered payload. No demonstrations,
slip labels, automatic correction or tactile interrupts are added.

The frozen formal controller/observation version is commit `66afcc6`; one fresh
no-demo episode per seed completed under `outputs/univtac-isaac51-r15/batch`.
All three had valid reset, available native checking, and no infrastructure
error: **autonomous development success is 0/3**. The single debug above is
excluded. No formal episode was retried or used to tune the detector/controller.

| Seed | Native success / ending | MCP calls | Non-preview requests received / admitted / physical moves | Control / physics steps | Simulation seconds | Four-view samples | Codex wall seconds |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| 1000003 | false / native early stop | 15 | 7 / 7 / 7 | 27 / 54 | 0.450 | 28 | 147.784 |
| 1000004 | false / Agent finish | 17 | 9 / 9 / 9 | 72 / 144 | 1.200 | 73 | 166.735 |
| 1000005 | false / native early stop | 14 | 6 / 5 / 5 | 6 / 12 | 0.100 | 7 | 120.584 |

Counts start after official reset. Samples include takeover. Seed 1000005's
sixth non-preview request arrived after native termination and was rejected
before the session's admission counter; it caused no additional physical step.
No episode exhausted a development or native step budget.

The worker remained available for final reading and `finish_episode` until host
cleanup after Codex exit. All three models exited naturally with return code 0
and real usage. Seeds 1000003/5 continued for 20.767/32.568 wall seconds after
the launcher observed native terminal, within the 300-second grace and original
3600-second total limit. Seed 1000004 finished voluntarily; it is not a native
terminal-grace witness. Live runs did not exercise full grace/total-time expiry;
focused tests cover those cleanup branches. Results and recording indices were
saved before `SimulationApp.close`.

| Seed | Input tokens | Cached input tokens (included in input) | Output tokens | Reported reasoning tokens |
| --- | ---: | ---: | ---: | ---: |
| 1000003 | 581,584 | 522,880 | 2,947 | 1,870 |
| 1000004 | 660,481 | 592,256 | 4,410 | 3,034 |
| 1000005 | 482,666 | 433,408 | 2,195 | 1,149 |

These are reported usage fields, not additive independent cost categories.
Total input/output were 1,724,731/9,552 tokens; no monetary cost is inferred.

**What the recordings show:**

- **1000003:** the Agent requested four downward world-Z translations, opened
  the gripper, moved down again, then requested a 50 mm retreat. The retreat
  executed one control step before native early stop and did not reach its
  target. Host-only `inhand_bias` was 41.61 mm, above the unchanged 40 mm
  threshold. The final review image still shows the rod near the fixture;
  neither complete dropping nor a unique failure cause is established.
- **1000004:** downward probes were interleaved with open/close commands,
  followed by a 35 mm upward request with open, another close, and voluntary
  finish. Arm targets were reached, but native success remained false. The
  final head image shows the rod leaning diagonally out of the fixture below
  the gripper. Closing fingers to about 0.762 mm did not establish a retained
  object or successful insertion.
- **1000005:** three preview-resolved motions and two further downward moves
  preceded native early stop (`inhand_bias` 46.29 mm, host-only). The later
  15 mm retreat request was rejected without physics. The Agent never changed
  the inherited gripper command. The tactile images change substantially, but
  this is not a verified slip label or a correct-action diagnosis.

The native joint1/joint2 inherited target pairs, in mm, were respectively
(5.721921, 5.722739), (5.721828, 5.722760), and (5.721528, 5.722320). Command
changes were open; open/close/open/close; and none. Final target pairs were
(39, 39), (0, 0), and the retained initial pair; measured final openings were
(38.653348, 38.662773), (0.762617, 0.762174), and (5.722758, 5.721204) mm.
These names follow native finger-joint order; anatomical left/right mapping
has not been independently checked. Holding a command does not prove grasp
stability. All formal motion requests were translations; rotation was exercised
in the separate debug, not claimed as formal Agent behavior.

The formal recordings contain 108 four-view sample times and 105 control-step
transitions. Recorded sensor IDs were consecutive, with no unrefreshed sample,
no missing frame, matching four-sensor timestamps, and approximately 1/60-second
simulation intervals. This metadata agreement does not prove zero renderer lag.
Of 210 pad transitions, one used labelled image-difference fallback (the left
pad in seed 1000003); none did so in the other seeds. Tracking quality is not
semantic detection accuracy. The Agent received 7/9/5 segment histories with
17/24/11 selected shared time points: 104 tactile image cells in total, including
reused references. All delivered cells were checked against their raw PNGs and
matched pixel-for-pixel. Video decode produced 28/73/7 recorded frames, without
invented observations. Waiting for Codex does not advance simulation time.

Replay is at `http://127.0.0.1:9401/r15-autonomous`, served by the local
`univtac-r15-dashboard.service`. It separates full videos/raw frames and
host-only curves/diagnostics from actual Agent images, requests and responses;
it provides playback speed, frame stepping and both detectors' candidate windows.
The retained videos are [1000003](../../outputs/univtac-isaac51-r15/batch/seed_1000003/review.mp4),
[1000004](../../outputs/univtac-isaac51-r15/batch/seed_1000004/review.mp4),
[1000005](../../outputs/univtac-isaac51-r15/batch/seed_1000005/review.mp4), and
[unscored debug](../../outputs/univtac-isaac51-r15/debug/seed_1000003/review.mp4).
[Delivery metrics](../../outputs/univtac-isaac51-r15/delivery_metrics.json)
retain per-action feedback, command history, timing, sampling, usage and lifecycle
fields. These artifacts are local and excluded from Git. Post-batch replay
edits only clarify request counts, preview target annotations and display; they
do not change the frozen controller, selector, prompt or retained raw runs.

The 45 focused tests, scoped Ruff/compile checks, HTTP/range responses and video
decoding passed. No browser was available for an interactive playback acceptance
check; server/template and raw-frame checks do not replace that check.

This completes the R1.5 development scope, with no successful autonomous
trajectory to promote into a demonstration. No B/C runs or automatic example
selection were performed. Future A/B/C must share this controller and current
observation mechanism; a change relative to R1.4 cannot establish ICL benefit
or isolate the value of change selection. Integer patch tracking, contact
interpretation, reliable manipulation and success-example collection remain
limitations or future work, not reasons to retune this completed batch.

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

1. R1.5 completed command-retention and tactile-history validation on fixed
   seeds `1000003`, `1000004`, `1000005`. Review its actual failures before the next
   authorized step. These remain development data, not formal held-out tests.
   The [configuration](../../configs/univtac/autonomous_insert_hole.yaml) keeps
   Terra medium, 30 non-preview move requests, 100 tool calls, and 3600 Codex
   seconds per episode alongside the native control budget.
2. Collect a small set of successful same-interface operations, retaining real
   images, calls, feedback, recovery, and outcomes. No-demo performance need
   not be high before examples may be introduced.
3. Run the A/B/C comparison. The same-interface successful example bank and
   this comparison remain unfinished. Neither R1.4 nor R1.5 produced a successful
   autonomous example or ran B/C.
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
