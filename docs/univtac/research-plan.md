# UniVTAC tactile-agent research plan

This is the canonical research-plan and status entry point for
`tactile-agent-for-univtac`, updated on 2026-09-08 against the current checkout
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

Demonstrations must come from actual successful execution of the current
UniVTAC-Isaac51 native expert. Human operations, Agent successes and fixed-target
diagnostic replays do not substitute for this expert bank. Preserve the actual
commands when recorded, units, coordinate frame and timing. If only robot states
are available, describe measured expert motion; do not invent original commands
or execution feedback, and do not rename native actions as OpenETA calls:

```text
task goal + pre-action vision / bilateral tactile short history / proprioception
    -> recorded expert action, or explicitly labelled measured expert motion
    -> recorded execution feedback (or explicitly unavailable)
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

R1.7 uses official published `isaac51/insert_hole` episodes 0 and 1, selected
in numeric ID order. Metadata records success and seeds/source seeds 0 and 1,
separate from query development seeds `1000003`–`1000005`. A task-specific
train/test split and the data-producing commit are not recorded in the inspected
release metadata. R1.6 local support data remain historical evidence and are not
mixed into this package. The offline expert may use
its native ground-truth algorithm. Agent-visible examples contain historical
observations, actual actions, feedback and success outcomes, without executable
truth-based correction formulas, hidden-pose queries or future query answers.
Visual-action and tactile-action exports share the same trajectories and
non-tactile content; the latter adds only aligned historical touch. Historical
EE poses and motions are not current-scene OpenETA TCP commands to copy blindly.

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

### R1.6 fixed-target pacing comparison and expert curation

R1.6 started at `4161567`; the six replay runs used controller commit `96c85b2`.
Subsequent changes concern offline videos, expert exports and documentation.
There were exactly six fresh fixed-target episodes, zero new Codex decision
experiments and zero expert recaptures. This is a controller diagnostic, not an
Agent success batch or the ICL A/B/C experiment. All six resets/outcomes were
valid; no infrastructure retry or speed/configuration search occurred.

The [comparison configuration](../../configs/univtac/motion_pacing_comparison.yaml)
freezes the 7/9/5 admitted R1.5 commands, including resolved world TCP position,
orientation and explicit gripper changes. Preview IDs are resolved once from
recorded execution; the rejected post-terminal request is excluded. Both variants
receive identical absolute requests, without accumulating new relative targets.
Execution order was original/candidate for 1000003, candidate/original for
1000004, and original/candidate for 1000005. Prepared commands and variant
configs were saved before the first launch.

`original` retains R1.5 execution semantics and remains the default. The optional
`paced_candidate` advances an intermediate Cartesian reference using control
simulation time, then uses the same IK and dynamic actuators. Translation is
bounded by a 0.04 m/s reference and shortest-path rotation by 0.4 rad/s; both
share a duration so they arrive together. At 1/60-second control dt, these are
at most 0.667 mm and 0.006667 rad per reference step. A 50 mm request requires
75 reference steps, leaving only five of the unchanged 80 segment steps for
tracking. Reference velocity is not an actual-velocity guarantee.

Final-target arrival, 3 mm / 3 degree tolerances, IK correction limits, gripper
commands, physics dt/decimation, actuator settings, native thresholds and the
300-control-step budget are unchanged. A no-simulator test documents that an
otherwise unchanged 1 mm request can return reached with zero physical steps
under the existing tolerance; the tolerance was not fixed in this comparison.
Native terminal stops immediately. A non-reached segment budget would stop the
replay without hidden steps; none of these six reached that budget branch.

| Seed | Variant | Native success / ending | Arrived / attempted / planned targets | Control / physics steps | Simulation seconds | Sample times | After-reset wall seconds |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 1000003 | original | false / early stop | 6 / 6 / 7 | 24 / 48 | 0.400 | 25 | 7.752 |
| 1000003 | paced_candidate | true / success | 6 / 7 / 7 | 127 / 254 | 2.117 | 128 | 25.731 |
| 1000004 | original | true / success | 9 / 9 / 9 | 72 / 144 | 1.200 | 73 | 19.119 |
| 1000004 | paced_candidate | true / success | 7 / 8 / 9 | 104 / 208 | 1.733 | 105 | 24.396 |
| 1000005 | original | false / early stop | 5 / 5 / 5 | 6 / 12 | 0.100 | 7 | 4.969 |
| 1000005 | paced_candidate | false / early stop | 4 / 5 / 5 | 89 / 178 | 1.483 | 90 | 19.734 |

The after-reset wall clock includes synchronous tool handling, not initialization,
video export or model thinking. Full worker wall times in seed order,
original/candidate, were 88.078/101.005, 108.491/112.291 and 85.659/109.106 seconds.
Every sampled transition corresponds to one native control step and two physics
steps; each episode has takeover plus one sample per control, without missing
or unrefreshed samples in the retained records. The native success latch can end
a request before its arm target is reached: this happened during retreat in
1000003 candidate and during the upward/open request in 1000004 candidate.
Those partial targets are not counted as arrived. No replay success enters the
expert demonstration bank.

**Measured comparison and limitations.** Peak sampled-interval TCP speeds were
0.616/0.045, 0.962/0.042 and 0.785/0.042 m/s (original/candidate); these are norms
of net displacement per control interval, not instantaneous physics-step peaks.
The candidate changed actual timing rather than just a reference plot. Maximum
along-request positional overshoot was approximately 2.059/0.017,
8.946/0.008 and 2.293/0 mm; this is a directional projection, not every possible
contact instability. Candidate arrived endpoints commonly stop near the 3 mm
tolerance boundary, so smaller image changes alone are not comparable progress.

Before the first explicit gripper change, both variants completed the same first
four targets for 1000003 and first three for 1000004. Maximum host-only native
in-hand Z change over those prefixes was 32.58/27.44 and 15.41/10.89 mm.
Later explicit opens are retained, not attributed entirely to motion speed.
1000005 never changed gripper command: both variants stopped early. Its candidate
ended 7.07 mm from the final target versus original's 2.15 mm, so the smaller
40.36 versus 46.32 mm in-hand change does not establish better stability at
matched completion. This field is the native relative-pose indicator, not a
measured cumulative slip path.

Initial states were not bit-identical. Candidate-minus-original initial TCP
position differences (mm) were (0.031, 0.024, -0.295),
(-0.046, -0.001, -0.001), and (-0.087, -0.023, -0.320). Initial prism-relative-to-
gripper X differed by about +1.76, -0.97 and -2.48 mm; orientations, robot joints,
finger targets and measured openings are retained in the offline comparison.
These contact differences limit causal attribution from three single pairs.
1000004 succeeded in both variants; 1000005 failed in both, with more candidate
steps and less final completion. The observed native-success count 1/3 versus
2/3 is a development result for these commands, not evidence that speed is the
unique cause of historical failures or an ICL gain. Keep the candidate for user
review; do not replace the default on this evidence alone.

**How to watch.** The [local R1.6 page](http://127.0.0.1:9401/r16-motion-pacing)
provides Chinese explanations, separate four-view videos, side-by-side 1× and
0.05× simulation-time versions, and one-second-per-frame inspection copies.
Both sides share the takeover-relative clock and playback multiplier. A shorter
side explicitly holds its last frame; no independent length normalization or
synthetic intermediate observations are used. Display holds, including the final
one-second hold, are playback metadata rather than additional physical time.

- **1000003:** first compare the four-target prefix, then action 5 open and the
  retreat. Original stops during action 6; candidate reaches native success
  during action 7 before completing that retreat target. The final external
  views show a released rod near the fixture; native success is reported from
  the checker, not inferred from the image.
- **1000004:** both succeed. Compare the first three targets and action 8's
  upward/open operation. Candidate terminates earlier in that command; original
  completes the final close too. Do not present different termination points
  as equal completion of the full nine-command list.
- **1000005:** the 0.1-second original is especially short; use its frame-hold
  version, then the common-rate side-by-side video. Both show tactile-pattern
  changes, and both stop early. Candidate moves more slowly but has not finished
  the final target; the pattern difference is not a slip label.

**Expert bank.** Support seeds 1000000/1000001 reuse actual R1.2 native
`task.play_once()` successes under the pinned Isaac51 collect configuration:
120 Hz control/physics, decimation 1, native planner/dense moves (default
`force=True`). These are not the dynamic OpenETA replay controller and not
proof that the backends are equivalent. Each has three aligned before/after
vision, bilateral touch, robot-state and native-action pairs. Source steps were
395→560→646→724 (checker at 744) and 446→671→811→934 (checker at 954).
The timestamps derive from these collect step differences / 120; no dense
sensor frame IDs or instantaneous velocity is fabricated. Native save/video
frequency had been disabled in the R1.2 harness, so these are not official
full HDF5 collections. Existing data suffice for sparse boundary examples;
no expert recapture was required.

Each export preserves native EE/base position in metres, quaternion wxyz,
actual target, realized displacement, time-dilation parameter and outcome;
it does not claim the expert called OpenETA tools. Dense tactile motion,
per-frame sensor IDs and gripper target-buffer history remain unavailable.
The visual-action and tactile-action JSONs differ only by aligned touch, with
real PNGs copied from the same original trajectory. Provenance manifests remain
host-only. The separately labelled expert videos dwell on six boundary images,
not a continuous physical process. No human/Agent/fixed-replay success was
substituted and no ICL inference was run.

Artifacts live in `outputs/univtac-isaac51-r16/`: `commands.json`, per-variant
control/sample/contact logs, `comparison_summary.json`, three pair folders,
and `expert/seed_1000000` / `seed_1000001`. They are excluded from Git.
Reproduction uses a fresh output root; the second command is offline only:

```bash
uv run --frozen --extra dev python -m scripts.univtac.run_motion_pacing_comparison \
  --output-root outputs/univtac-isaac51-r16-new
uv run --frozen --extra dev python -m scripts.univtac.motion_pacing_review \
  --root outputs/univtac-isaac51-r16-new
```

The 47 focused tests, scoped Ruff/compile checks and Markdown links passed.
Video presentation timestamps were checked against the simulation-time mapping;
actual browser loading, playback and seeking were exercised, and key frames
were visually inspected. This is not a claim of an independent frame-by-frame
review of every video. Scoped neat reconciled the current expert-source rule
and retained the historical results.

This completed batch is not authorization to repeat it. The replay page's fixed
route points at the retained R1.6 root; another root's generated `review.html`
can be served through the existing artifact route. No new autonomous or ICL
batch starts until the user has reviewed the videos and decided.

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

## R1.7 autonomous tactile ICL pilot and next work

R1.7 completed nine fresh Insert Hole episodes with `gpt-5.6-terra` / medium,
the original controller, and identical current tactile-history mechanisms.
The task began at `b60aa97`; frozen experiment code was `51e6a0e`. There were
no added expert/debug episodes, task-failure retries, or controller changes.

Official HDF5 episodes 0/1 contain 189/166 recorded frames and three movement/
delay segments each. B received 12 before/after vision strips; C received the
same non-tactile content and vision strips plus 12 bilateral tactile-history
strips. Common boundaries and visual times use movement metadata, independently
of touch. HDF5 contains JPEG image streams, `embodiment/ee`, `embodiment/joint`,
`step`, and `atom/id` / `atom/tag`, but no independent raw action commands.
Measured EE is base-relative panda_hand xyz + wxyz, not current world TCP;
nominal seconds use recorded step differences / 120, without an embedded clock.
The documentation revision is not the data-producing commit and does not
upgrade our pinned runtime. The task-specific train/test split remains unknown.
Each real Codex context reviewed its fixed demonstration/empty result once.
A received no demonstration images, B received 12 native MCP images, and C
received 24, followed in all conditions by live observations and autonomous
commands. Actual delivery is recorded in each `operator_context.jsonl`.

A (`no_demo`), B (`visual_action_icl`), and C (`tactile_action_icl`) each achieved
**0/3 native task successes**. All nine episodes were evaluable, ended with
`native_early_stop`, and had natural Codex exits with recorded usage; no
infrastructure errors were reported. C−A, C−B and B−A were all zero percentage
points. This three-seed development pilot found no success-rate benefit; it
does not establish that tactile ICL is generally ineffective.

| Seed | Condition | Native success | Actual motions | Control steps | Simulation s | Codex wall s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1000003 | A | false | 4 | 11 | 0.183 | 348.21 |
| 1000003 | B | false | 4 | 7 | 0.117 | 97.18 |
| 1000003 | C | false | 3 | 11 | 0.183 | 144.83 |
| 1000004 | A | false | 12 | 78 | 1.300 | 253.90 |
| 1000004 | B | false | 3 | 15 | 0.250 | 124.53 |
| 1000004 | C | false | 3 | 6 | 0.100 | 94.53 |
| 1000005 | A | false | 5 | 24 | 0.400 | 170.19 |
| 1000005 | B | false | 4 | 11 | 0.183 | 129.03 |
| 1000005 | C | false | 2 | 6 | 0.100 | 110.93 |

Physics steps are twice the control steps in this batch. The
[results](../../outputs/univtac-isaac51-r17/results.json) retain per-episode
tool counts, worker time, raw input/cached/output/reasoning token usage, video
paths and errors; the [manifest](../../outputs/univtac-isaac51-r17/batch/run_manifest.json)
records execution. Earlier termination or fewer actions is not evidence of
greater task efficiency when the task failed.

Most B/C runs mainly attempted downward motions. Seed 1000004 A made twelve
motions including vertical probing and lateral adjustment; seed 1000005 A
opened the gripper and subsequently lifted before early termination. Inspect
these different trajectories individually rather than attributing all failures
to a single claim that the Agent did not understand touch.

The [episode dashboard](http://127.0.0.1:9401/r17-autonomous) shows actual
operator input, requested commands, execution feedback, outcomes and cost.
[Videos](http://127.0.0.1:9401/r17-videos) include all nine 1× recordings and
slow versions: 0.1× for seed 1000004 A, 0.05× for the other eight. Existing raw
frame views support frame-by-frame inspection. Slow playback repeats recorded
frames, without synthesizing observations.
[Demonstration previews](http://127.0.0.1:9401/r17-demonstrations) keep the
historical official experts separate from autonomous episodes.

Read the downloaded official files with the existing r09 Python (fresh output):

```bash
/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11 \
  -m scripts.univtac.prepare_official_demonstrations \
  --raw outputs/univtac-isaac51-r17/data/official \
  --output outputs/official-demonstrations-new
```

Use the [fixed configuration](../../configs/univtac/official_tactile_icl.yaml)
and a fresh batch output directory:

```bash
uv run --no-sync python -m scripts.univtac.run_official_tactile_icl \
  --config configs/univtac/official_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r17/demonstrations \
  --output-root outputs/univtac-isaac51-r17-new-batch
```

These commands describe reuse, not authorization to repeat this pilot. The
completed round stops for review of the actual inputs and videos. Additional
seeds, model/controller changes, or another round require a new decision.

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


## R1.8: public rules × tactile ICL with GPT-6 low

R1.8 completed all 18 fresh development episodes using execution commit
`902178e`, starting from `3ae61ee`. All 18 were natively evaluable, with no
infrastructure failures, natural Codex exits and recorded usage. New project operators and execution/review/neat agents use
`gpt-6-astra` with `model_reasoning_effort="low"` explicitly passed to Codex.
Historical R1.7 and earlier model labels remain unchanged.

U uses the pinned Isaac51 seen instruction: “Explore the inclined hole through
contact to determine its orientation, then insert the test tube into the hole.”
R adds static public success/early-stop rules, without current hidden state or
motion advice. R is the preselected full-information setting; U removes those
extra rules. The exact rule text and unchanged control budgets are in
[the R1.8 config](../../configs/univtac/task_rule_tactile_icl.yaml). Each frozen
condition prompt is saved before the first episode; initial prompt and observe
share the official instruction. U does not receive the additional numeric rules.

A has no demonstrations; B reuses R1.7 official Isaac51 `0.hdf5`/`1.hdf5`
visual–measured-motion examples; C adds their existing aligned bilateral touch.
The historical package stays intact. Only its generic task-goal sentence is
replaced at delivery with the same official sentence for B/C; examples,
segmentation, selected images and measured movements are unchanged. Every
condition retains current vision, tactile history, robot state and feedback.

The fixed order is seed 1000003: UA/RA/UB/RB/UC/RC; seed 1000004:
RB/UB/RC/UC/RA/UA; seed 1000005: UC/RC/UA/RA/UB/RB. Each cell is a fresh live
Codex episode under the original controller, native 300-control-step budget,
80-step segment budget, 30 admitted motion requests, 100 tools, 3600 seconds
Codex and the existing terminal grace. No extra simulator debug, expert replay
or model capability run is part of this round.

```bash
uv run --no-sync python scripts/univtac/run_official_tactile_icl.py \
  --config configs/univtac/task_rule_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r17/demonstrations \
  --output-root outputs/univtac-isaac51-r18/batch
```

Report six native successes/3 and evaluability separately. Fixed percentage-point
contrasts are RA−UA, RB−UB and RC−UC for rules; RB−RA and RC−RA for demonstrations;
RC−RB and UC−UB for historical touch. R1.7 differs in model and task wording and
cannot isolate either effect. Three development seeds do not establish held-out
generalization. Failures and incomplete cells remain visible.

The existing dashboard exposes `/r18-autonomous`, `/r18-videos` and
`/r18-demonstrations` on port 9401. It shows actual initial prompts and native MCP
inputs separately from host-only review videos. All 18 four-view recordings and their slow versions were generated from retained
frames. Browser playback reached the end for all 36 files; representative frames
were also visually inspected. This is playback verification, not a claim of
human inspection of every video frame.


### R1.8 observed results

| Task information | A: no demo | B: visual–motion | C: added historical touch |
|---|---:|---:|---:|
| U: official instruction | 1/3 | 1/3 | 3/3 |
| R: instruction + public rules | 1/3 | 0/3 | 2/3 |

Every cell is evaluable. The eight successes ended with `native_success`; the
other ten ended with `native_early_stop`. No native/control/tool/wall-clock budget
was extended and no failed cell was rerun. R remains the preselected full-input
setting, despite U's higher C score.

Rule contrasts RA−UA, RB−UB and RC−UC are respectively 0.0, −33.3 and −33.3
percentage points. Within R, RB−RA is −33.3 pp, RC−RA is +33.3 pp, and RC−RB
is +66.7 pp. UC−UB is also +66.7 pp. Thus this pilot observed a positive
historical-touch difference against matched visual examples in both rows, but
no success-rate improvement from supplying the additional rules. These are
three-seed descriptive differences, not proof of a general benefit or of rules
being harmful. Fresh resets showed initial-state differences; historical
examples were fixed, but current physical states were not pixel-identical.
R1.7 is not a same-model control.

| Seed | Cell | Native result | Actual motions | Control / physics steps | Sim s | Codex / worker wall s |
|---|---|---|---:|---:|---:|---:|
| 1000003 | UA | early stop | 15 | 77 / 154 | 1.283 | 191.4 / 311.1 |
| 1000003 | RA | early stop | 10 | 40 / 80 | 0.667 | 129.1 / 272.9 |
| 1000003 | UB | early stop | 4 | 14 / 28 | 0.233 | 69.3 / 195.0 |
| 1000003 | RB | early stop | 7 | 17 / 34 | 0.283 | 92.8 / 217.5 |
| 1000003 | UC | success | 5 | 18 / 36 | 0.300 | 77.6 / 206.3 |
| 1000003 | RC | success | 6 | 18 / 36 | 0.300 | 310.0 / 434.7 |
| 1000004 | RB | early stop | 5 | 14 / 28 | 0.233 | 182.2 / 315.9 |
| 1000004 | UB | early stop | 5 | 13 / 26 | 0.217 | 74.0 / 200.6 |
| 1000004 | RC | early stop | 6 | 17 / 34 | 0.283 | 96.5 / 230.1 |
| 1000004 | UC | success | 6 | 19 / 38 | 0.317 | 86.3 / 216.1 |
| 1000004 | RA | early stop | 13 | 64 / 128 | 1.067 | 209.1 / 337.7 |
| 1000004 | UA | early stop | 9 | 28 / 56 | 0.467 | 126.8 / 253.6 |
| 1000005 | UC | success | 6 | 18 / 36 | 0.300 | 102.7 / 230.4 |
| 1000005 | RC | success | 7 | 20 / 40 | 0.333 | 299.3 / 429.0 |
| 1000005 | UA | success | 8 | 41 / 82 | 0.683 | 152.1 / 278.9 |
| 1000005 | RA | success | 11 | 72 / 144 | 1.200 | 182.4 / 308.1 |
| 1000005 | UB | success | 5 | 18 / 36 | 0.300 | 70.0 / 193.8 |
| 1000005 | RB | early stop | 6 | 15 / 30 | 0.250 | 75.9 / 199.7 |

Worker wall time includes initialization and shutdown; it excludes offline
review encoding. Codex wall time includes model/tool waits. Two RC episodes
had long waits after demonstration delivery and eventually completed normally;
model-catalog refresh timeout messages in stderr did not become episode failures.
Do not interpret shorter failed episodes as higher efficiency.

| Cell | Input tokens | Cached input (subset) | Output tokens | Reasoning output (subset) |
|---|---:|---:|---:|---:|
| UA | 1,807,564 | 1,666,304 | 5,353 | 880 |
| UB | 1,041,113 | 915,200 | 2,427 | 218 |
| UC | 1,374,565 | 1,218,048 | 2,671 | 200 |
| RA | 2,173,043 | 1,994,624 | 6,189 | 1,566 |
| RB | 1,296,784 | 1,158,784 | 2,692 | 223 |
| RC | 1,476,663 | 1,299,328 | 2,959 | 259 |

These are raw CLI cumulative fields summed over three episodes per condition;
cached input is not added to input, nor reasoning output to output. All actual
commands record `gpt-6-astra`, `model_reasoning_effort="low"` and CLI 0.153.4.
A/B/C delivered 0/12/24 historical images per context, once each. The ordinary
model/effort, prompt, image-count and budget readback is retained in
`outputs/univtac-isaac51-r18/delivery_validation.json`.

**How to watch.** On [the video page](http://127.0.0.1:9401/r18-videos), start
with seed 1000003 UB versus UC: UB explored positive x before returning downward
and failed, while UC explored negative x before positive-x/downward moves and
succeeded. Then compare seed 1000004 RC with UC, where the exploration directions
also differed and only UC succeeded. Seed 1000005 UA/RA show successful explicit
orientation adjustment without examples. All six A episodes explicitly requested
orientation changes; B/C episodes requested translations without explicit orientation changes. Seed 1000005 UA and RA explicitly commanded close and later open with retreat;
native success arrived during that final action. The other sixteen episodes
retained inherited gripper commands. Seed 1000005 UA also recovered from one
`mark_point` rejection for a pixel without valid observed depth by choosing
other pixels; this was a recoverable tool response, not an infrastructure failure. These request patterns describe behavior, not evidence that a
particular tactile feature caused a choice. Actual robot orientation can change
even when no new orientation was requested.

Videos use 0.1× for seed 1000003 UA/RA, seed 1000004 RA, and seed 1000005 UA/RA;
the other thirteen use 0.05×. Each also has a 1× version. The
[dashboard](http://127.0.0.1:9401/r18-autonomous) provides original-frame stepping,
exact prompts, actual MCP images and responses, action feedback and separate
host-only diagnostics. The [demonstration preview](http://127.0.0.1:9401/r18-demonstrations)
shows the two fixed official experts separately from current Agent episodes.

Validation: 51 focused tests, scoped Ruff, compileall and diff checks passed.
No simulator/model capability pretest or extra physical attempt was added.
Pro subsequently accepted this result and assigned the frozen new-seed
validation below; no prompt or model comparison was added.


## R1.9: frozen method on twelve new query seeds

Pro accepted R1.8 as a positive development signal for historical touch, while
recognizing that extra public rules did not improve success. R1.9 completed
all 36 fresh episodes on seeds 1000006–1000017 with the preselected R input.
RA/RB/RC achieved **3/12, 10/12 and 11/12** native successes. The main
comparison RC−RB is **+8.3 percentage points**, only one discordant seed in
RC’s favor; RC−RA and RB−RA are +66.7 and +58.3 points. This is a small
positive new-seed signal for historical touch, much smaller than R1.8’s
three-seed difference. It does not establish a robust or universal tactile
advantage. R1.8 stays separate; its old seeds are not pooled into this table.

The scoped search of this checkout's 505 episode/manifest/summary JSON files
and documentation found no execution records for these seeds; only the R1.9
plan named them. Other historical worktrees and remotes were not searched.
This is project new-seed validation, not an official published test split.
No expert screening, model pretest or controller debug is part of this round.

[The fixed R1.9 config](../../configs/univtac/new_seed_tactile_icl.yaml) records
all 36 cells before launch, with six condition permutations repeated twice.
Generated prompts were compared directly with R1.8's saved R prompt; all three
are identical. Demonstration text/image references were also compared with
R1.8's actual delivered projections. The same R1.7 source package and unchanged
delivery transformation reproduce those projections; no data is reselected or
resegmented. Model, current observation/history, tools, original controller,
physical parameters and every budget remain unchanged.

```bash
uv run --no-sync python scripts/univtac/run_official_tactile_icl.py \
  --config configs/univtac/new_seed_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r17/demonstrations \
  --output-root outputs/univtac-isaac51-r19/batch
```

All 36 planned cells were evaluable: 24 native successes and 12 native early
stops, with no infrastructure failures, replacements, retries or extra debug
episodes. No cell ended on a budget limit. All Codex processes exited naturally
and provided usage. The execution version was `c22dade`, starting from `e6ae93a`;
no controller, prompt, demonstration, sampling or budget changed during the
batch. The unrelated untracked `heldout_grounding_skill.py` was preserved.

The existing dashboard adds `/r19-autonomous`, `/r19-videos`,
`/r19-demonstrations` and `/r19-pairs`. RB/RC paired videos reuse the recorded-frame
alignment approach: common simulation time and playback rate, with the shorter
side explicitly holding its final frame. Commands may differ because the Agent
chooses them. They do not launch or advance physics. Every episode has a 1× video and a
slow version (0.05× below 0.5 seconds, otherwise 0.1×); each RB/RC pair has
1× and 0.05× versions. Full playback validation is recorded separately below.


### R1.9 outcomes and costs

| Condition | Planned / evaluable | Native success | Native failure | Infrastructure |
|---|---:|---:|---:|---:|
| RA: no examples | 12 / 12 | 3/12 (25.0%) | 9 | 0 |
| RB: visual–measured-motion examples | 12 / 12 | 10/12 (83.3%) | 2 | 0 |
| RC: same examples plus touch | 12 / 12 | 11/12 (91.7%) | 1 | 0 |

Each entry below is **native outcome; actual motion requests / control steps**.
All failures are native early stops. Physics steps equal twice control steps;
post-takeover simulation time is control steps / 60 seconds. Initialization is
recorded separately and is not folded into these task-body counts.

| Query seed | RA | RB | RC |
|---|---|---|---|
| 1000006 | early stop; 9 / 24 | success; 7 / 20 | success; 7 / 19 |
| 1000007 | early stop; 26 / 202 | early stop; 6 / 18 | success; 7 / 19 |
| 1000008 | success; 14 / 91 | success; 6 / 19 | success; 7 / 19 |
| 1000009 | early stop; 13 / 52 | success; 6 / 23 | success; 11 / 97 |
| 1000010 | early stop; 13 / 58 | success; 6 / 18 | success; 7 / 19 |
| 1000011 | early stop; 16 / 81 | success; 7 / 17 | success; 5 / 18 |
| 1000012 | success; 13 / 74 | success; 6 / 19 | success; 6 / 19 |
| 1000013 | early stop; 18 / 92 | success; 6 / 16 | success; 6 / 13 |
| 1000014 | early stop; 17 / 93 | success; 7 / 19 | success; 6 / 19 |
| 1000015 | success; 20 / 96 | success; 6 / 18 | success; 7 / 20 |
| 1000016 | early stop; 17 / 69 | early stop; 6 / 15 | early stop; 6 / 18 |
| 1000017 | early stop; 14 / 66 | success; 5 / 17 | success; 7 / 20 |

| Pair (RC versus control) | RC only succeeds | Control only succeeds | Both succeed | Both fail | Unavailable |
|---|---:|---:|---:|---:|---:|
| RC / RB | 1 | 0 | 10 | 1 | 0 |
| RC / RA | 8 | 0 | 3 | 1 | 0 |

The sole RC/RB discordant seed is 1000007. Both fail on 1000016. Thus the
new batch supports a limited positive replication signal, not the large effect
size seen on the old development seeds. The larger RB−RA difference belongs
to visual–motion examples; it must not be called a tactile benefit. Current
bilateral touch remains available in every condition. One attempt per cell
cannot separate model variability from fresh-reset variability, and this remains
one task on project-selected new seeds rather than a multi-task benchmark.

The following totals cover twelve episodes per condition. Worker wall time
includes initialization and shutdown, excludes offline review encoding; Codex
wall time includes model/tool waiting. Shorter failure is not efficiency gain.

| Condition | Motion requests | MCP calls | Control / physics steps | Sim seconds | Codex / worker wall seconds |
|---|---:|---:|---:|---:|---:|
| RA | 190 | 283 | 998 / 1996 | 16.633 | 3018.3 / 4682.4 |
| RB | 74 | 123 | 219 / 438 | 3.650 | 1167.5 / 2784.2 |
| RC | 82 | 140 | 300 / 600 | 5.000 | 1318.4 / 2889.0 |

| Condition | Input tokens | Cached input (subset) | Output tokens | Reasoning output (subset) |
|---|---:|---:|---:|---:|
| RA | 12,283,829 | 11,498,496 | 35,175 | 9,666 |
| RB | 5,109,416 | 4,546,944 | 12,201 | 1,508 |
| RC | 6,690,215 | 6,096,128 | 13,649 | 1,797 |

These are actual cumulative CLI fields, not estimates; subset fields are not
added again. All 36 commands explicitly use `gpt-6-astra`,
`model_reasoning_effort="low"`, CLI 0.153.4. Saved prompts equal the R1.8 R
prompt, and each context delivers demonstrations once: RA/RB/RC 0/12/24 images.
Readback of actual model arguments, prompt, demonstration text/image references,
image existence and budget/step counters passed for all 36. Evidence:
`outputs/univtac-isaac51-r19/delivery_validation.json`; full per-cell costs and
outcomes are in `results.json` beside it. There are 1,553 synchronized recorded
sample sets, including the takeover frame in each episode; no recording step
was added to physics.

### What to watch

Start at [RB/RC paired replay](http://127.0.0.1:9401/r19-pairs), then use
[all individual videos](http://127.0.0.1:9401/r19-videos) and
[actual Agent inputs and actions](http://127.0.0.1:9401/r19-autonomous).
The [fixed historical expert preview](http://127.0.0.1:9401/r19-demonstrations)
is separate from these current Agent results.

- **1000007 RB/RC:** the only RC-only success. Compare the actual requested
  directions and returned observations; do not infer that a particular touch
  feature caused the different decisions merely from the outcome.
- **1000016 RB/RC:** both fail. RB explores positive x before reversing;
  RC first explores negative x, then positive x/downward. Both retain inherited
  gripper targets and request no explicit orientation change. Different
  exploration directions did not ensure success. Watch the early-stop frame
  and the labelled end-of-side hold at equal playback speed.
- **1000009 RC:** a longer 97-step success, including translation, explicit
  close/open and orientation requests. It is not the same short translation-only
  pattern as most C episodes.
- **1000012 RA and 1000015 RA:** successful no-example recovery attempts after
  different orientation probes. Seed 1000012 succeeds during an open plus
  pose-change request, before reaching that request’s final target. Seed
  1000015 closes, tilts toward negative x and makes repeated diagonal moves,
  succeeding with the close target retained. These are observed sequences,
  not proof that opening or closing alone caused success.
- **1000007 RA:** 26 actual motions and 202 control steps end in native early
  stop, not budget exhaustion. Keep this long failed attempt alongside the
  shorter successes.

There was one recoverable `mark_point` invalid-depth-pixel response on
1000012/RA; the Agent selected a different pixel and continued. This is not an
infrastructure failure. A tool execution field containing `native_success`
marks task termination even when the requested arm target was not reached;
command arrival and native success remain distinct.

Prelaunch validation passed 55 focused tests covering new ordering, frozen
inputs/model, paired outcomes/timeline and previous interfaces, plus scoped
Ruff, compileall and diff checks. No extra simulator/model test was launched.
Final GPT-6 low full-history scoped neat found no result/cost inconsistency.
All 96 files (36 individual episodes × two rates, 12 pairs × two rates) reached
browser `ended`; receipts are in
`outputs/univtac-isaac51-r19/browser_playback_check.json`. This verifies native
playback, not manual viewing of every frame. Representative paired frames were
also visually inspected. Two offline playback-check interruptions were resolved
without regenerating observations: one long browser wait, and one navigation
collision between page inspection and playback. Playback now uses its own browser
session. Neither affected the simulator, Agent input or episode results.
This round adds no mechanism ablation or task; subsequent experimental action
requires the next concrete Pro instruction under the continuing project goal.


## R1.10: historical versus current tactile input

Pro accepted R1.9 while tightening the claim: expert demonstrations helped on
this batch, but the single extra RC success does not establish a reliable
historical-touch benefit. RB still had current touch; its 10/12 is not evidence
that vision alone suffices. R1.10 is a supplementary modality ablation, not a
change to the original A/B/C definitions or the complete multimodal method.

The [fixed configuration](../../configs/univtac/current_tactile_ablation.yaml)
completed eight new seeds 1000018–1000025, four conditions each (32 fresh episodes):
B_live / C_live retain current bilateral tactile history; B_no_live / C_no_live
omit it from Agent-visible outputs. B has the same twelve historical visual
images and measured-motion text as R1.9; C adds the same twelve historical
bilateral-touch images. C_no_live retains those historical tactile images.
There is no no-example A condition or reuse of R1.9 outcomes in the new table.

`project_current_observation` filters current touch after all normal and
recoverable-error observation paths in the existing session. It leaves the
host observation/recorder unchanged, removes current tactile image descriptors
and the complete tactile-history selection metadata, and does not affect
historical review. All four groups retain current head/wrist, robot/gripper
state, execution feedback, budgets and minimal native checks. Sensors, sampling,
reset/pre_move adaptive grasp and contact physics remain active in no_live;
it is not a claim of a system that never used tactile sensing.

All four prompts are identical. The R1.9 rule, coordinate and budget text is
unchanged; tactile format statements gain “when provided” and a common missing-
modality explanation. Model remains `gpt-6-astra` / low, with original control
and unchanged 30 motion / 100 MCP / 3600 Codex-second / 300 native-step /
80 segment-step limits and terminal grace. No debugging simulator episode,
model pretest, expert screening/collection or parameter change is added.

```bash
uv run --no-sync python scripts/univtac/run_official_tactile_icl.py \
  --config configs/univtac/current_tactile_ablation.yaml \
  --demonstrations outputs/univtac-isaac51-r17/demonstrations \
  --output-root outputs/univtac-isaac51-r110/batch
```

Prelaunch offline checks used retained R1.9 observations: removing current touch
preserved other fields and the host payload; all four historical projections
matched actual R1.9 delivery and the four prompts were equal. An explicitly
labelled two-frame offline video test reused R1.9 data; it is not a new physical
episode. The no_live touch region reads “仅供用户审阅，本episode未送给Agent”.
Raw frame stepping is host-only; Agent Saw uses actual operator_context images.
The existing dashboard adds `/r110-autonomous`, `/r110-videos`,
`/r110-demonstrations` and `/r110-pairs`; paired views compare live/no_live within
B and within C on common simulation time, holding an ended side's last frame.

The 2×2 main table reports successes/8, evaluability, native failures and
infrastructure issues. Fixed contrasts are C_live−B_live, C_no_live−B_no_live,
B_live−B_no_live and C_live−C_no_live, each with paired outcomes. The difference
between the first two is descriptive, not proof of model internals. Input length
also changes when touch is omitted. If scores do not decrease, report that this
batch did not detect a performance drop, not that the model ignored touch.

Prelaunch validation passed 63 focused tests, scoped Ruff, compileall and diff
checks. The MCP file retains 19 pre-existing Ruff diagnostics; comparison with
the baseline found no new diagnostic from its changed tool description.
GPT-6 low full-history scoped neat found no documentation inconsistency.
The 32-cell batch completed on execution commit `601f02c`, starting from
`7757bea`. There were no replacements, retries, added debugging episodes or
mid-batch implementation changes. All 32 were evaluable, 26 succeeded and six
ended in native early stop. No episode exhausted its budget. All model processes
exited naturally with usage; no recoverable tool error or infrastructure failure
was recorded. Previous rounds remain separate.


### R1.10 results: historical-touch gain appears only under no_live in this batch

| Historical examples | Current touch available | Current touch omitted |
|---|---:|---:|
| B: visual–measured-motion | 7/8 (87.5%) | 5/8 (62.5%) |
| C: same plus historical touch | 7/8 (87.5%) | 7/8 (87.5%) |

Every cell has planned=8, evaluable=8 and infrastructure issues=0. Native
failures for B_live / C_live / B_no_live / C_no_live are 1 / 1 / 3 / 1.

| Fixed comparison | Difference (percentage points) | Only first succeeds | Only second succeeds | Both succeed | Both fail | Unavailable |
|---|---:|---:|---:|---:|---:|---:|
| B_live-B_no_live | +25.0 | 2 | 0 | 5 | 1 | 0 |
| C_live-B_live | +0.0 | 1 | 1 | 6 | 0 | 0 |
| C_live-C_no_live | +0.0 | 1 | 1 | 6 | 0 | 0 |
| C_no_live-B_no_live | +25.0 | 2 | 0 | 5 | 1 | 0 |

The descriptive difference-in-differences is **−25.0 percentage points**:
(C_live−B_live)−(C_no_live−B_no_live). This batch does not support the simple
claim that historical touch mainly improves the use of current touch. With B,
current touch adds two successes; with C, its aggregate difference is zero,
but one seed improves and another worsens. Historical touch adds no net success
under live input and two under no_live. These are limited input-condition
comparisons, not evidence that the model ignored a modality or learned a
particular internal mechanism. Removing touch also changes context length;
fresh resets and one model attempt per cell leave substantial uncertainty.
No new default or controller change follows from these results alone.

Each entry is **native outcome; actual motion requests / control steps**.
Physics steps are twice control steps, and task-body simulation seconds are
control steps / 60. Initialization counts remain separate.

| Seed | B_live | C_live | B_no_live | C_no_live |
|---|---|---|---|---|
| 1000018 | success; 6 / 17 | early stop; 6 / 18 | early stop; 6 / 18 | success; 6 / 16 |
| 1000019 | success; 6 / 17 | success; 6 / 17 | success; 6 / 19 | success; 7 / 17 |
| 1000020 | success; 7 / 17 | success; 7 / 17 | success; 6 / 15 | success; 7 / 19 |
| 1000021 | success; 6 / 18 | success; 6 / 18 | success; 6 / 15 | success; 6 / 18 |
| 1000022 | success; 7 / 16 | success; 6 / 14 | success; 7 / 19 | success; 5 / 14 |
| 1000023 | success; 6 / 14 | success; 6 / 19 | success; 6 / 16 | success; 5 / 18 |
| 1000024 | success; 8 / 20 | success; 5 / 17 | early stop; 6 / 18 | early stop; 6 / 17 |
| 1000025 | early stop; 6 / 17 | success; 6 / 19 | early stop; 5 / 14 | success; 6 / 19 |

The following are totals over eight episodes per condition. Worker wall time
includes startup/shutdown; offline review encoding is excluded. Codex wall time
includes model/tool waiting, not just physics. In particular, 1000022/C_live
waited after demonstrations before continuing and exited normally; its Codex
wall time was 348.8 seconds. Shorter failure is not an efficiency gain.

| Condition | Motion requests | MCP calls | Control / physics steps | Sim seconds | Codex / worker wall seconds |
|---|---:|---:|---:|---:|---:|
| B_live | 52 | 86 | 136 / 272 | 2.267 | 822.1 / 1887.1 |
| C_live | 48 | 82 | 139 / 278 | 2.317 | 1081.2 / 2146.1 |
| B_no_live | 48 | 82 | 134 / 268 | 2.233 | 855.9 / 1906.7 |
| C_no_live | 48 | 81 | 138 / 276 | 2.300 | 806.5 / 1835.5 |

| Condition | Input tokens | Cached input (subset) | Output tokens | Reasoning output (subset) |
|---|---:|---:|---:|---:|
| B_live | 3,625,932 | 3,162,368 | 8,189 | 915 |
| C_live | 3,871,966 | 3,463,296 | 7,677 | 711 |
| B_no_live | 3,195,812 | 2,840,832 | 7,981 | 833 |
| C_no_live | 3,595,349 | 3,247,488 | 7,475 | 643 |

These are actual CLI cumulative fields; cached and reasoning subsets are not
added again. All 32 command records explicitly use GPT-6 `gpt-6-astra` / low,
CLI 0.153.4, and the common frozen prompt. Demonstrations are delivered once
per context: B=12 and C=24 historical images. All current observation returns
contain four images under live and two head/wrist images under no_live;
no_live omits tactile-history selection metadata. Current touch remains in
host recordings, not Agent Saw. Readback passed for all 32 in
`outputs/univtac-isaac51-r110/delivery_validation.json`; full per-cell costs and
paired counts are in `results.json` beside it. Raw recording retains 579
synchronized sample sets, including each takeover frame, without added physics.

### R1.10 viewing guide

[All 32 videos](http://127.0.0.1:9401/r110-videos),
[live/no_live paired views](http://127.0.0.1:9401/r110-pairs),
[actual inputs/actions](http://127.0.0.1:9401/r110-autonomous), and
[historical expert examples](http://127.0.0.1:9401/r110-demonstrations) use the
existing dashboard. All current trajectories explicitly request only world
translations and preserve inherited gripper commands; this does not mean actual
orientation is perfectly constant.

- **1000018:** B_live succeeds and B_no_live fails; C_live fails and C_no_live
  succeeds. The two successful request sequences first explore negative x and
  then move positive-x/downward; failed sequences explore positive x first.
  Watch both B and C pairs, not just one favorable comparison.
- **1000024:** both live conditions succeed and both no_live conditions fail.
  Successful requests again explore negative x first; failed requests explore
  positive x first. These are action descriptions, not a tactile interpretation.
- **1000025:** both B conditions fail and both C conditions succeed, irrespective
  of current touch. It provides a counterpoint to a story requiring current
  touch for historical examples to help.

The same seed does not create identical initial robot states. For example,
1000024's observed initial approach-axis x component was approximately −0.0093
and −0.0102 under B_live/C_live, versus +0.0332 and +0.0348 under the two no_live
runs. These ordinary proprioceptive differences limit single-pair attribution;
no object or robot state was moved to force matching. Requested translation is
not measured translation, and native success is not target arrival. Some final
requests report arm arrival and native early stop together; others reach native
success before the requested target. Keep both feedback fields visible.

Every episode has 1× and 0.05× playback; all recorded task-body durations are
below 0.5 seconds. Sixteen same-seed B/C live/no_live pairs also have 1× and
0.05× versions, on shared simulation time with explicit ended-side frame holds.
No interpolated sensor observations are generated. The no_live tactile panels
are visibly labelled “仅供用户审阅，本episode未送给Agent”.
All 96 playback files reached browser `ended`; receipts are in
`outputs/univtac-isaac51-r110/browser_playback_check.json`. Representative paired
frames were visually inspected, including the no_live warning. This is not a
claim of manually reviewing every frame. Final GPT-6 low full-history scoped neat completed a read-only result/cost and
claim-boundary check; no numerical inconsistency was found.

## R1.11: second-task autonomous pilot (partial; reset blocker)

Pro accepted R1.10 and stopped further Insert Hole mechanism ablations. Expert
examples have helped autonomous insertion, but stable additional benefit from
historical touch, and positive interaction between historical and current touch,
remain unestablished. The descriptive −25 pp interaction is not proof of
negative synergy or substitution.

R1.11 extends the same method to `grasp_classify`, with fresh seeds
1000026–1000031 and A/B/C, 18 planned autonomous episodes. All groups retain
current head/wrist, bilateral tactile history, robot state and execution feedback,
GPT-6 `gpt-6-astra` / low, original control, 30 admitted motion requests,
100 MCP calls, 3600 Codex seconds, native 300 control steps, 80-step segments
and existing terminal grace. No controller or native task parameters change.
The task's own initialization has `use_adaptive_grasp=False`; its native friction
and grasp initialization are retained. It has no task-specific early-stop
predicate; wrong-goal placement does not create an extra failure rule.

The common prompt uses the official seen instruction, one active object,
rough → orange / plain → green, and static native placement tolerances:
object origin in its correct goal frame has strict |x|/|y| < 0.02 m and
|z| < 0.01 m, and positive-axis dot product > 0.965. No current class,
true target transform or target error enters the operator. Class is recorded
only in host finalization for post-run subgroup analysis. The goal must be
located from ordinary observations and supported head-camera marking.

Official `isaac51/grasp_classify` metadata records success for episodes 0 and 1.
They are the first plain and rough records respectively in ascending ID order;
only these two were downloaded/inspected. Metadata lacks class, so historical
active-actor/final-goal records establish class only for host selection. The
Agent-visible package contains no hidden class or asset-name answer label.
Both HDF5s have 48 aligned rows, steps 222–316 with gap 2, and one actual motion
segment. All 384 four-view images decoded. Metadata ends at step 337; the saved
sequence does not cover the expert's final unsaved delay. No continuous frames
are fabricated. Original action commands are absent: examples contain measured
EE/joint changes, not invented OpenETA calls. B delivers four visual strips;
C adds four bilateral tactile strips, with identical non-tactile content.

Configuration: `configs/univtac/grasp_classify_tactile_icl.yaml`.
Official source: [ModelScope UniVTAC](https://modelscope.cn/datasets/byml2024/UniVTAC),
[Isaac51 download script](https://github.com/univtac/UniVTAC/blob/isaac51/data/download.sh).
Data, matched projections and provenance are retained locally under
`outputs/univtac-isaac51-r111/{data/official,demonstrations}`. A maximum of one
unscored reset/observe/shutdown check at seed 999999 is allowed before the
18 formal episodes; it executes no task-body motion and no Codex.

```bash
uv run --no-sync python -m scripts.univtac.run_autonomous_insert_hole \
  --config configs/univtac/grasp_classify_tactile_icl.yaml --mode observe_only \
  --output-root outputs/univtac-isaac51-r111/debug
uv run --no-sync python -m scripts.univtac.run_official_tactile_icl \
  --config configs/univtac/grasp_classify_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r111/demonstrations \
  --output-root outputs/univtac-isaac51-r111/batch
```

These commands require fresh output subdirectories and do not authorize retries.
The reused runner filenames retain their historical names. The one allowed seed-999999 observation check completed: four images, head
marking available, wrist geometry unavailable, zero post-reset control/physics
steps and zero Codex processes. The worker exited normally in 184.98 seconds.
Success was deliberately not evaluated. Its retained termination label
`codex_exit` is an old generic host-close label, not evidence that Codex ran;
the subsequent label-only fix names future observe-only closure explicitly.
No physical rerun was needed. Formal native results and video delivery remain pending. Review entry: `http://127.0.0.1:9401/r111-autonomous`.

### R1.11 partial execution and initialization blocker

The frozen formal version is `370366a`. Two formal simulator invocations occurred,
plus the single debug above; 16 planned cells have not started. The runner stopped
on the second cell through its existing infrastructure-error mechanism.

- 1000026/A: native success, 3 physical motions, 24 control / 48 physics steps,
  0.4 simulation seconds, 11 MCP calls. GPT-6 low exited naturally after
  114.046 Codex seconds; worker wall time was 254.776 seconds. Actual usage:
  input 292,516, cached-input subset 260,864, output 1,021, reasoning subset 200.
  The host-only class was rough. Agent requested a 30 mm upward move with close,
  then an absolute placement target and a downward adjustment. Its final text
  attributes classification to touch; this self-report does not establish a
  tactile mechanism. Native checking, not the explanation, establishes success.
- 1000026/B: official reset raised `Timeout: reset exceed time limit of 120.0 s,
  cost 166.90325421496527 s.` in `_base_task.py:462`, during the initial five-step
  reset test. No Codex process, operator observation or task-body action occurred.
  `native_success_available=false`, `task_success=null`, not a native failure.
  The launcher observed return code 0 despite the persisted worker exception;
  the not-ready record still correctly makes the cell infrastructure/unavailable.

There is no completed A/B/C success-rate comparison: A has 1 evaluable success
of 6 planned, B has one unavailable attempt of 6 planned, C has no attempts.
No cell was retried, no reset-time limit changed, and no query was replaced.
The remaining 16 cells are pending Pro's decision about recovery within the
invocation budget. The reset failure is a startup wall-time issue, distinct from
native task control-step exhaustion or an Agent timeout; its low-level cause is
not yet established.

A's actual MCP input, prompt and counters passed readback. Its 1× and 0.05×
videos both reached browser ended; no B task-body video exists because takeover
was never reached. The historical expert preview and B/C matched images are
available at the R1.11 dashboard; missing formal videos/results remain labelled
pending rather than manufactured. A subsequent UI-only correction excludes the
observe-only manifest from formal group aggregation; it does not change the
frozen controller, prompt, examples or physical outcome.

### Explicit one-time recovery authorization

Pro reply `b42baf5b-0e18-467a-88e7-162a7a2149bf` accepts the partial report,
not a completed R1.11 result. It explicitly raises the simulator invocation cap
from 19 to 20: exactly one unchanged-configuration recovery for 1000026/B,
then the remaining 16 cells in original order. A and debug are never rerun.
Any subsequent worker-not-ready infrastructure error pauses the entire tail.
The native reset wall-time limit remains 120 seconds; worker total wall time
is not evidence that this reset limit is inadequate.

Use the existing runner with `--resume-r111` and the same config, demonstrations
and `--output-root outputs/univtac-isaac51-r111/batch`. The original B error stays
in its directory; recovery uses `seed_1000026/B/attempt_2`. A dedicated recovery
manifest records authorization, attempts and stop state. Review selects that
one authorized recovery regardless of success/failure, never the better result.
The original partial `batch/summary.json` remains retained; `results.json` and
`batch/recovery_manifest.json` describe the combined current state. Historical
infrastructure errors remain visible even if recovery succeeds. Paired differences
use only jointly evaluable seeds and report their count; missing cells are not
native failures. No new physical startup has yet occurred at this authorization
checkpoint.

### Recovery outcome: paused again at the required boundary

The one authorized 1000026/B recovery succeeded. Seven valid episodes completed
in the recovery continuation before 1000028/B hit the same reset-test limit:
120 seconds allowed, 133.95139663503505 seconds measured. Codex had not started.
The runner immediately stopped, as Pro required; no 1000029–1000031 episode
was launched. No limits or task/controller parameters changed.

Total startup count is **11**: one debug, eight evaluable autonomous episodes,
and two failed initialization attempts. The earlier 1000026/B failure remains
retained even though its sole recovery succeeded. Current 18-cell state is
8 successful/evaluable, 1 unavailable, 9 not run; no completed native failure.
There is no complete six-seed success-rate result.

| Seed | A | B | C |
|---|---|---|---|
| 1000026 | success; 3 moves / 24 control | success after one reset recovery; 1 / 11 | success; 1 / 9 |
| 1000027 | success; 5 / 39 | success; 3 / 22 | success; 2 / 11 |
| 1000028 | success; 6 / 58 | reset unavailable; no Codex | success; 1 / 10 |
| 1000029–1000031 | not run | not run | not run |

A=3 successes/3 evaluable/6 planned, B=2/2/6, C=3/3/6. C−B and B−A are
0 pp across two jointly evaluable seeds; C−A is 0 pp across three. These partial
pairs do not establish an ICL gain or general lack of value. Host-only classes
among evaluable episodes are A/C: two rough, one plain; B: one rough, one plain.
No missing episode was assigned a class from another run or removed from plan.

All eight operators exited naturally with actual usage; actual-input readback
passed, with 0/4/8 historical images for A/B/C and four current images per
observation return. There were no task-phase tool errors. The recordings retain
192 four-view sample sets, including each takeover, without added physics.
Costs below cover only evaluable episodes (unequal counts), not initialization
failures, so totals must not be compared as matched efficiency estimates.

| Condition (completed n) | Motions / MCP | Control / physics | Sim s | Codex / worker wall s | Input / cached subset | Output / reasoning subset |
|---|---:|---:|---:|---:|---:|---:|
| A (3) | 14 / 42 | 121 / 242 | 2.017 | 416.615 / 691.654 | 1,079,677 / 983,936 | 3,937 / 967 |
| B (2) | 4 / 16 | 33 / 66 | 0.550 | 172.101 / 413.511 | 517,324 / 459,904 | 1,241 / 136 |
| C (3) | 4 / 27 | 30 / 60 | 0.500 | 259.163 / 473.243 | 911,599 / 820,224 | 2,147 / 318 |

For viewing, 1000027/A includes an explicit open followed by native success;
1000028/A contains several position adjustments before success. B/C on the first
two seeds both succeed; no extra success is attributable to historical touch in
those pairs. Native success is not evidence that the Agent's stated texture
reasoning was necessary. Further initialization diagnosis or physical recovery
requires the next Pro instruction; no automatic additional attempts are made.

All 20 derived videos for completed cells reached browser `ended`: eight
individual episodes × 1×/slow playback, plus two B/C pairs × 1×/0.05×.
Individual slow playback is 0.1× for 1000027/A and 1000028/A, otherwise 0.05×.
Receipts are retained in `outputs/univtac-isaac51-r111/browser_playback_check.json`.
Representative frames and the partial-result page were visually inspected;
this is not a claim of manually watching every frame. No video was fabricated
for a failed initialization or an unrun cell. The final focused suite passed
56 tests, with scoped Ruff, compileall, diff checks and GPT-6 low scoped neat.

### Limited reset timing diagnosis (authorized, not a formal retry)

Pro `43fd5010-a8e6-4baa-b761-50b17c04c5db` keeps the formal batch paused and
allows at most one seed-1000028 reset-only timing diagnostic if retained logs
are insufficient. Current cumulative startup cap for this diagnostic stage is
12 (11 already used plus at most one); future formal recovery needs separate
Pro authorization. No timeout or behavior change is authorized in this stage.

Retained logs show first-step cumulative Running times 62.74/74.22 seconds in
the failures, with second-step increments 104.52/60.06 seconds. Successful
B26 recovery and A28/C28 have second-step increments 0.21/0.17/0.17 seconds.
B27 also had a slow first step (76.81 seconds) yet passed the reset test. Running
is cumulative from `start_time` set near `_reset_idx`'s end; the first value
includes surrounding reset work, not just one step. The exception checks a
separate cumulative clock starting immediately before the five-step loop.
Retained process samples have membership/libcuda mappings, not CPU/GPU activity
or stacks. They cannot distinguish computation from waiting inside `_step`.

The optional worker flag `--reset-timing-only` records existing first-five-step
boundaries and wall/process CPU time, plus one 30-second traceback per long step.
UIPC callbacks are wrapped before registration, with the same original call,
arguments and order. No extra render, physics or GPU synchronization is added.
After native reset returns it saves timing and closes, with no Codex, autonomous
session, task-body action or added success check. Native initialization still
moves the robot and physics. Logs are separate under
`outputs/univtac-isaac51-r111/reset-diagnostic/`; no diagnostic outcome enters
A/B/C statistics. Two focused tests verify call count/order and exception
preservation. This is instrumentation, not a fix or claim of a known root cause.

The single authorized diagnostic completed normally: **not reproduced, not
fixed**. Task construction took 16.346 seconds and `Task.reset` 39.703 seconds.
First-five `_step` durations were 6.294/0.167/0.167/0.125/0.106 seconds. In this
normal sample the first `UipcSim.step` (native `world.advance` + `world.retrieve`)
accounted for 6.223 seconds; `task._update_render` was 0.063 seconds. That does
not identify which subcall caused either historical timeout. First-step process
CPU was 7.283 seconds across all process threads; a contemporaneous device-wide
5-second GPU sample reported 93% utilization. Neither is a per-kernel trace or
evidence of the failed runs' resource state. No step exceeded 30 seconds, so no
timeout stack was captured. Nested spans overlap and must not be summed twice.

Native reset/pre_move advanced 240 physics steps. Zero Codex processes and zero
task-body actions occurred; no extra success check was invoked. The scoped
launcher exited normally in 70.017 seconds, cleanup complete. GPU sampling was
stopped after this diagnostic; no unrelated process was touched. Total R1.11
simulator invocations are now 12. The formal 8-success/1-unavailable/9-not-run
state is unchanged. No second diagnostic or formal continuation was launched.

No specific behavior fix is justified yet: the missing evidence is an abnormal
call's nested timing/stack and contemporaneous resource state. Retain the opt-in
instrumentation for a future explicitly authorized attempt; do not claim that
raising the timeout, clearing caches or changing runtime would solve it.
The separate human-readable report is
[reset diagnosis](http://127.0.0.1:9401/artifact?run=univtac-isaac51-r111/reset-diagnostic&path=report.html),
with `report.json`, `retained_timing_comparison.json`, `gpu_samples.csv` and raw
`seed_1000028/reset_timing.jsonl` beside it. This diagnostic is excluded from
all autonomous and ICL statistics.

### Timed formal recovery authorization

Pro `ea3a4ea7-4666-4444-99e0-eccce5c4a742` accepts the limited diagnosis as
not reproduced/unknown cause and authorizes exactly ten further formal starts:
1000028/B in its own `attempt_2`, then 1000029 A/C/B, 1000030 C/B/A,
1000031 B/A/C. The cumulative cap is now explicitly **22**, replacing 20.
No independent diagnostic, warm-up, or rerun of the eight valid episodes is
included. Any infrastructure failure again pauses the remaining tail.

`--resume-r111-timed` reuses the original frozen condition YAMLs, prompt,
demonstrations and budgets. Each worker receives `--reset-timing`, independently
of `--reset-timing-only`. After one native reset, timing/stacks are closed and
the same Task enters the ordinary ready/Codex path; no second initialization
occurs. Initialization measurements stay host-only. This adds timing overhead,
not a claimed behavior fix; the native 120-second guard remains unchanged.
`batch/timed_recovery_manifest.json` records this authorization separately from
the earlier recovery. The standalone reset-only diagnostic never fills B28.
Thirteen focused tests cover the split, exact ten-cell order, original calls,
exception preservation and accepting native failure without retry.

### Timed recovery reproduced the initialization stall

The first authorized timed continuation, 1000028/B `attempt_2`, again failed
before ready; no further cell started. The native five-step-loop check reported
120.782643 seconds against 120 seconds. Total `Task.reset` wall time was
121.133265 seconds. Both measured `_step` calls returned before the native
exception; the 30-second stack timer did not terminate either call.

| Reset test step | Whole `_step` s | UipcSim advance+retrieve s | `_update_render` s | Process CPU in UIPC span s |
|---|---:|---:|---:|---:|
| 1 | 66.515655 | 66.441903 | 0.064119 | 76.203689 |
| 2 | 54.266845 | 54.195770 | 0.064827 | 62.344447 |

Both 30-second main-thread stack samples point to
`tacex_uipc/sim/uipc_sim.py:241`, `self.world.advance()`. This identifies the
slow call interval and its sampled location; the wrapper times advance and
retrieve jointly, so it is not separate timing of every inner operation.
Render was approximately 0.017/0.019 seconds and tactile update 0.042/0.040
seconds, unlike the long physics callback. During the first long call, a saved
snapshot showed device GPU utilization 95% and process CPU 200% (lifetime
average); the interval CPU totals above sum all process threads. These are
consistent with computational activity, not proof of any specific solver,
compilation, contact or driver cause. Native internals remain unlocalized.

The formal grid is still eight successes, B28 unavailable and nine not run.
There are now **13 starts = eight valid + three failed initializations + two
non-scored starts**. No B28 Codex process was launched, no valid trajectory/video
was rerun, and no timeout/physics/input change was applied after the failure.
Failed startup worker wall costs are 198.161 s (B26 original), 164.059 s (B28
original), and 150.342 s (B28 timed recovery), separate from Agent costs. All
three retained worker exceptions despite launcher returncode 0; cleanup was
complete. Details are in `timed_recovery_pause.json` and
`batch/seed_1000028/B/attempt_2/{reset_timing.jsonl,reset_stacks.txt,worker_error.json}`.
The remaining formal recovery is paused for the next Pro decision.
