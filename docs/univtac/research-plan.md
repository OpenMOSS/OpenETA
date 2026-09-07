# UniVTAC tactile-agent research plan

This is the canonical research-plan and status entry point for
`tactile-agent-for-univtac`, updated on 2026-09-07 against the current checkout
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
