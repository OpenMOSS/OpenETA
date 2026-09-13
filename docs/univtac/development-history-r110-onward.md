# Development history: R1.10 onward and protocol acceptance

These are historical checkpoints, not current execution state or authorization to resume.
See the [research plan](research-plan.md) for the current scope and
[shot-scaling results](shot-scaling-results.md) for the completed campaign.
Old pause/recovery instructions below apply only to their recorded checkpoints.

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
success before the requested target. Both fields remain available in host records and user review; fresh
no-online-feedback runs expose ordinary arm-arrival feedback and neutral
episode termination to the Agent.

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

## R1.11: autonomous Grasp & Classify — completed development pilot

The 18 planned task cells are now natively evaluable: **A 4/6, B 6/6, C 6/6**.
The main comparison **C−B is 0 pp**, with all six pairs successful in both
conditions. B−A and C−A are each +33.3 pp (two method-only successes, four
shared successes). This supports a demonstration benefit in these six
development seeds, not an additional historical-touch success-rate benefit.
It is not a full benchmark or held-out generalization claim. Insert Hole
results remain separate; stable historical-touch benefit and positive
historical/current-touch interaction remain unestablished.

### Fixed method, task information and official examples

Task: pinned Isaac51 `grasp_classify`; seeds 1000026–1000031. A has no examples,
B has official expert visual–measured-motion–outcome examples, C adds only
historical bilateral touch to the same examples. All current inputs retain
head/wrist, bilateral tactile history, proprioception and actual feedback.
Model: explicit `gpt-6-astra` / `model_reasoning_effort="low"`, real Codex CLI
0.153.4, isolated context/memory/history and scoped MCP. Original controller,
retained gripper targets, 30 admitted move requests, 100 MCP calls, 3600 Codex
seconds, native 300 control steps, 80-step segments and terminal grace remain.

The official instruction is: “Touch the cylinders to perceive their surface
texture, classify each object, and place it at the goal region for its class.”
All groups also know one object is active, rough maps to orange, plain/smooth
to green, and static native success tolerances: object origin in its correct
goal frame has strict |x|/|y| < 0.02 m and |z| < 0.01 m; positive-axis dot
product > 0.965. No current class, target transform or target error is supplied.
The task has no special wrong-goal early stop. Its native initialization uses
`use_adaptive_grasp=False`; native friction and grasp setup were preserved.
Class is written only in host finalization for subgroup analysis. Head marking
uses current depth/calibration; wrist geometry remains explicitly unavailable.

Only official `isaac51/grasp_classify` HDF5 0/1 were downloaded, in ascending
ID order. Metadata records success and seed/source_seed 0/1. Class is absent
from metadata; historical active-actor/final-goal data establish 0 as the first
plain and 1 as the first rough success, solely for host selection. No hidden
class/asset-name answer label enters examples. Each has 48 aligned rows,
steps 222–316 at gap 2 and one actual motion segment; all 384 four-view images
decoded. Metadata ends at step 337: the expert's final unsaved delay is not
fabricated as video. Original commands are absent, so examples describe
measured EE/joint motion, not invented OpenETA calls. B has four visual strips;
C adds four tactile strips. The non-tactile JSON and image correspondence match.
The data producer commit and a task-specific formal split remain unrecorded.
The existing dataset entry in [Related Work](related-work.md) covers this reuse.

### Native outcomes and actual operations

Each table entry is native outcome; admitted requests / control steps.
All requests below also caused physical stepping. Success is the native checker,
not classification text, goal naming, command arrival or model self-report.

| Seed | A: no examples | B: visual–motion | C: plus historical touch |
|---|---|---|---|
| 1000026 | success; 3 / 24 | success; 1 / 11 | success; 1 / 9 |
| 1000027 | success; 5 / 39 | success; 3 / 22 | success; 2 / 11 |
| 1000028 | success; 6 / 58 | success; 2 / 12 | success; 1 / 10 |
| 1000029 | step-limit failure; 8 / 300 | success; 3 / 20 | success; 2 / 11 |
| 1000030 | success; 4 / 39 | success; 2 / 17 | success; 2 / 12 |
| 1000031 | step-limit failure; 7 / 300 | success; 2 / 12 | success; 2 / 11 |

Every group has six evaluable results and zero unresolved infrastructure cells.
This does **not** mean no infrastructure attempts failed: three earlier startup
failures are retained below. All 18 Codex processes exited naturally with real
usage. Actual operator-context readback passed: demonstrations delivered once
(0/4/8 images for A/B/C), four current images per observation return, common
frozen prompt, explicit model/effort, and separate control/physics counts.
There were no task-stage tool errors. The recordings contain 936 four-view
sample sets (including takeover), without added physics.

Actual classes were rough on 1000026/28/31 and plain on 1000027/29/30. A succeeded
2/3 within each class; B/C succeeded 3/3 within each. No query class screening or
replacement occurred. Class correctness alone was never scored as task success.
A29 and A31 first requested +65 mm lift with close, then lateral movement with
open. Inspected frames show the cylinder outside the fingers/on the table;
subsequent regrasp requests consumed the native 300-step budget. This is a
specific control sequence and failed recovery, not evidence that the Agent
simply “does not understand touch.” B/C succeeded on both seeds; their shared
success does not isolate touch from the rest of the examples.

### Costs (six evaluable episodes per condition)

Worker wall time includes initialization/cleanup and excludes offline review
encoding. Initialization failures and non-scored launches are listed separately.
Cached input is an input subset and reasoning is an output subset; do not add
them twice. A's larger cost includes two full-budget failures and is not a
matched-success efficiency comparison. C's lower control count does not establish
an extra success-rate gain or a general efficiency mechanism.

| Condition | Requests / MCP | Control / physics | Sim s | Codex / worker wall s | Input / cached subset | Output / reasoning subset |
|---|---:|---:|---:|---:|---:|---:|
| A | 33 / 89 | 760 / 1520 | 12.667 | 915.673 / 1380.357 | 2,347,941 / 2,085,888 | 7,945 / 1,678 |
| B | 13 / 52 | 94 / 188 | 1.567 | 503.452 / 1005.123 | 1,671,581 / 1,455,360 | 4,292 / 715 |
| C | 10 / 52 | 64 / 128 | 1.067 | 515.988 / 1386.798 | 1,853,972 / 1,654,144 | 4,175 / 602 |

### Initialization history and explicit mitigation

There were **23 simulator starts = 18 valid autonomous episodes + three failed
initializations + two non-scored starts**. The non-scored runs were debug999999
(reset/observe/close, zero post-reset steps) and one reset-only timing diagnostic
seed1000028 (240 native initialization physics steps, no Agent/task-body actions).
Neither was an expert collection or formal task result. Debug worker wall time
was 184.981 s; reset-only diagnostic worker wall time was 70.017 s.

| Startup record | Outcome / evidence | Worker wall s |
|---|---|---:|
| B26 original | reset five-step interval 166.903 s > 120; no Codex | 198.161 |
| B26 attempt_2 | unique valid task result, success; original error retained | included in B cost |
| B28 original | reset interval 133.951 s > 120; no Codex | 164.059 |
| reset-only diagnostic B28 | reset returned in 39.703 s; not reproduced, not fixed | 70.017 |
| B28 attempt_2 with timing | reset interval 120.783 s > 120; no Codex | 150.342 |
| B28 attempt_3 with 600 s | unique valid task result, success; both earlier errors retained | included in B cost |

Thus two cells ultimately recovered successfully, through three recovery starts
(one failed). No already-valid Agent episode was retried. Failures were recognized
from `worker_error`/not-ready even where launcher returncode was zero. All three
failed starts cleaned up. Every stop/resume was separately authorized by Pro;
retained instructions and partial reports remain under the R1.11 output root.

Timed B28 failure localized its two long `_step`s to UIPC: whole steps
66.516/54.267 s, UIPC advance+retrieve 66.442/54.196 s, render updates about
0.064/0.065 s. Both 30-second main-thread stacks were at `world.advance()`.
Interval process CPU was 76.204/62.344 s across all threads; a contemporaneous
GPU snapshot showed 95% utilization. This locates the slow call interval and
shows computational activity, but does not identify solver/compilation/contact/
driver internals. Nested spans overlap. The earlier normal diagnostic's
6.223-second UIPC first step was not used as proof of the failed-run cause.

Pro explicitly authorized **only the remaining ten starts** to use native
`cfg.reset_time_limit=600.0`, with the existing 900-second ready deadline.
The first eight valid episodes used the original 120-second native default.
This mid-batch startup configuration difference is disclosed; the batch is not
claimed to have an identical startup configuration throughout. Task body, input,
controller, physics and native outcome rules were unchanged. No pinned source,
global default, cache, driver, solver, sensor or thread configuration was changed.

New copies in `batch/reset_limit_recovery_configs` add only the explicit override;
`native_reset_limit.json` records default/requested/configured/actual Task values.
Timing closes before the same Task enters ready/Codex; no second reset occurs.
The native clock is read after original steps, never replaced. Small logging
wall-time overhead remains. Clock observations distinguish separate five/20/five
reset loops and the marker-calibration clock; whole reset is not a substitute.

**C29 is the actual mitigation witness:** first five-step interval 430.807 s
(over 120, under 600), post-actor 20-step interval 53.719 s, final five-step
interval 0.398 s; marker-calibration global clock 485.604 s; complete reset
507.953 s, then normal ready and native task success. These overlapping clocks
must not be added together. The extra window enabled this initialization to
complete; it did not make UIPC faster or establish a root-cause repair. Other
new recorded intervals remained below 120 seconds, so their normal recovery
alone is not evidence that the override helped them.

### Reproduction and review

Configuration: `configs/univtac/grasp_classify_tactile_icl.yaml`. Main source/data
artifacts: `outputs/univtac-isaac51-r111/{demonstrations,results.json,final_analysis.json}`.
Original order remains in `batch/run_manifest.json`; original partial summary
and failed attempts remain intact. The first recovery, timed recovery and
600-second recovery have separate manifests. Effective cells choose the latest
**authorized** attempt regardless of outcome, never whichever succeeded best.
Execution revisions and their backfill provenance are in `execution_versions.json`.

The original fresh batch used:

```bash
uv run --no-sync python -m scripts.univtac.run_official_tactile_icl \
  --config configs/univtac/grasp_classify_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r111/demonstrations \
  --output-root outputs/univtac-isaac51-r111/batch
```

The authorized final recovery used the same command with
`--resume-r111-reset-limit`. Its manifest is one-time and already consumed;
these commands document execution, not permission to repeat this completed pilot.
The runner reuses the existing generic worker/MCP and original controller.

[Actual Agent inputs and requests](http://127.0.0.1:9401/r111-autonomous),
[all individual videos](http://127.0.0.1:9401/r111-videos),
[B/C paired playback](http://127.0.0.1:9401/r111-pairs), and
[historical expert/B–C input previews](http://127.0.0.1:9401/r111-demonstrations)
use the existing dashboard. Start with A29/A31 to see release/regrasp failures,
then B/C pairs for those same seeds: both example conditions succeed. The
separate [initialization diagnostic](http://127.0.0.1:9401/artifact?run=univtac-isaac51-r111/reset-diagnostic&path=report.html)
is host-only and excluded from task success.

Validation: 65 focused tests passed; scoped Ruff, compileall and git diff checks passed. All 48 query video files (36 individual 1×/slow videos and 12 paired 1×/0.05× videos) reached browser playback end. Individual slow versions use 0.1× or 0.05× according to duration. This verifies playable delivery, not a human watching every frame; representative failure and success frames were inspected separately. No generated intermediate observations were used. Scoped GPT-6 low neat confirmed the result and cost statements.

### R1.12 Grasp & Classify current/history touch ablation — infrastructure pause

Pro accepted R1.11, including the descriptive secondary result that C used fewer
control steps than B in all six jointly successful pairs (94 versus 64 total).
This does not establish overall lower time/cost: C had more input tokens and
slightly more Codex wall time. The 23-start R1.11 ledger is closed.

R1.12 uses eight new development seeds 1000032–1000039 and four conditions:
B_live/B_no_live share the frozen R1.11 B package; C_live/C_no_live share C,
including C_no_live's historical touch. Only live receives current bilateral
tactile history. All still record four host views with unchanged sensors,
original controller and native task physics (`use_adaptive_grasp=False`).
No_live is omitted Agent input, not a no-contact measurement.

Configuration: `configs/univtac/grasp_classify_current_tactile_ablation.yaml`.
All prompts are identical, with R1.11 rules plus the explicit missing-modality
notice and “when provided” format wording. The official 0/1 demonstrations,
segmentation, image selection and measured-motion interpretation remain frozen.
No task hints, new examples, controller changes or classification quiz are added.

The primary comparison is B_live−B_no_live. Also report C_live−B_live,
C_live−C_no_live and C_no_live−B_no_live, evaluable paired n/outcomes, and the
historical-touch difference-in-differences descriptively. Native success remains
the main outcome. Report all-evaluable costs and jointly-successful paired costs;
missing usage remains unavailable and subsets are not added twice. Final success
does not imply correct first classification, since feedback-driven recovery is
allowed. Initialization/failed-start costs stay separate.

All 32 starts use native reset limit 600 s, outer ready 900 s, cleanup 300 s,
Codex 3600 s including terminal grace 300 s, native 300 control steps,
80 per segment, 30 admitted motion requests and 100 MCP calls. Exactly one
fresh Task reset and Codex context per cell. No debug, warmup, expert or retry
reserve. Any infrastructure error pauses the remaining batch; native failure
continues the frozen order. No UIPC performance diagnosis is added.

Historical prelaunch checkpoint: the following command had not yet run when frozen; execution and pause are recorded below.

```bash
uv run --no-sync python -m scripts.univtac.run_official_tactile_icl \
  --config configs/univtac/grasp_classify_current_tactile_ablation.yaml \
  --demonstrations outputs/univtac-isaac51-r111/demonstrations \
  --output-root outputs/univtac-isaac51-r112/batch
```

Existing replay routes: [R1.12 inputs/results](http://127.0.0.1:9401/r112-autonomous),
[videos](http://127.0.0.1:9401/r112-videos),
[live/no_live pairs](http://127.0.0.1:9401/r112-pairs).
Complete delivery is 64 individual and 32 paired video files. No_live touch is
labelled host-only; missing/unready episodes do not receive fabricated videos.
Implementation/configuration and offline checks precede all physical execution.

Historical prelaunch evidence records 32 passing offline tests, equal B/C non-tactile
content, and 4/8 historical images. Simulator and operator starts are both zero
at this checkpoint; runtime delivery remains unverified.


#### R1.12 infrastructure pause (2026-09-08)

Frozen implementation `7dd93c0` ran ten simulator starts and nine Codex contexts.
The tenth start, seed 1000034 / C_no_live, did not reach ready. The runner recorded
its outer 900-second ready timeout; the worker subsequently preserved the native
reset error at `_base_task.py:478`: second initialization interval 670.778 s >
600 s. The first five-step interval was 250.640 s; complete failed reset took
921.815 s and worker lifecycle 950.301 s. These overlapping durations are not
additive. Native checks occur after a step returns, so the interval can exceed
600 before the exception. Cleanup completed with no SIGTERM/SIGKILL; no Codex or
query action started for this cell. This is not a native task failure.

The remaining 22 cells were not started. No retry or new diagnostic was launched.
All nine evaluable trajectories succeeded and naturally returned model usage:

| Seed | B_live | B_no_live | C_live | C_no_live |
| --- | --- | --- | --- | --- |
| 1000032 | success, 15 steps | success, 10 | success, 11 | success, 9 |
| 1000033 | success, 21 steps | success, 11 | success, 14 | success, 10 |
| 1000034 | not run | success, 10 | not run | initialization failure, unavailable |
| 1000035–1000039 | not run | not run | not run | not run |

Planned denominators stay eight per condition. Evaluable/success counts are
B_live 2/2, B_no_live 3/3, C_live 2/2 and C_no_live 2/2; these are **partial counts,
not completed eight-seed success rates**. Each fixed contrast has only two valid
pairs, both jointly successful, paired difference 0 pp. Full-batch contrasts and
difference-in-differences remain unavailable. No modality conclusion is supported.

Nine actual operator contexts passed model/prompt/once-only demonstration,
current-image and metadata checks. Live delivered four current images; no_live
delivered two and no tactile selection metadata; C_no_live kept eight historical
images. All nine preserved four-view host recording and native control/physics
counts. `results.json` contains all-evaluable costs and joint-success paired
costs with missing values left unavailable. Failed initialization has no model
usage because no model was started. Old R1.11 results remain unchanged.

Delivery at this pause is 18 individual videos and 8 paired videos for the two
complete seeds. No task video is fabricated for the unready cell. Detailed
attempts, timing and actions are in `outputs/univtac-isaac51-r112/partial_analysis.json`.
All 26 delivered video files reached browser playback end (18 individual and
8 paired); this verifies playable delivery, not human inspection of every frame.
Receipts are in `browser_playback_check.json`.

#### Authorized UIPC internal-timer diagnostic (not formal recovery)

After the R1.12 pause, one reset-only grasp_classify/1000034 start is authorized,
with the failed C_no_live configuration and unchanged native600/ready900/cleanup300
limits. It is unscored, has no operator, task-body actions, or additional success-checker calls, and cannot be
reused as a formal attempt. The 9 valid/1 failed/22 unstarted records stay frozen.
Existing outputs contain no native Timer tree. The installed Timer export clears
its window; inclusive nested durations must not be added. UipcSim already enables
Timer (including its existing CUDA timing synchronization); this diagnostic adds
no Timer enable or explicit sync. It exports construction/before/after-step/final
windows and export overhead, covers the native 5/20/5 intervals, and changes only
the existing diagnostic process's UIPC logger from Error to Info. No solver fix
or additional retry is authorized. At the prelaunch checkpoint, runtime internal evidence was pending.


The sole diagnostic completed (implementation `e8999b5`): reset 155.887 s,
243 native initialization/physics steps, zero operator/task-body/additional checker
calls. Native test intervals were 134.529/1.662/0.333 s. Newton scope counts were
1024/1024/640 for the first three steps, 6/6 for steps4/5, then 2–8. Sustained slow
steps were not reproduced; normal completion does not prove a fix. First-step
Newton45.931 s includes line search31.953 s, including trajectory candidates16.918 s
and CCD13.690 s. Both repeated iterations and per-scope cost increased. These
inclusive times are not additive. Info logs confirm max-iteration exits for the
first two steps; native strict_mode0 still returns without proving convergence.

All 62 draining Timer windows exported successfully. Export calls totaled
0.007956 s; total export-and-write overhead was 0.025283 s. Existing Timer CUDA synchronization was unchanged;
Info overhead is unmeasured. Actual r09 binary paths, pyuipc0.9.0/tacex_uipc0.1.0,
effective config and workspace are retained. No specific collision pair or
validated repair was established; no solver/geometry/initialization fix applied.
No extra startup or formal recovery is authorized. Related R1.12 starts total11,
with formal9 valid/1 failed/22 unstarted unchanged. See the
[short diagnostic report](http://127.0.0.1:9401/artifact?run=univtac-isaac51-r112/uipc-diagnostic&path=report.html)
and `outputs/univtac-isaac51-r112/uipc-diagnostic/report.json` for raw references.

#### Goal-pad scene diagnostic (O/K/Z, unscored): completed

The next authorized diagnostic holds grasp_classify/1000034 fixed for nine fresh
processes: O/K/Z, K/Z/O, Z/O/K. O keeps both pads dynamic at z0.002; K changes only
the two pads to native kinematic; Z changes only their initial z to0.010. The
formal default, pinned checkout and other actors remain untouched. These are
scene interventions, not benchmark-equivalent performance fixes. No formal
recovery or candidate adoption is authorized.

Offline USD tetrahedral-boundary inspection finds metre units, identity authored
transforms and collision-surface minimum gap at the source-specified initial pose
(offline, before runtime verification) about+1mm to the UIPC
z0.001 plane (+9mm for Z). This is actual boundary-vertex geometry against a
plane, not just actor origins or AABB overlap. The visible/PhysX table surface is
a different surface; it must not be conflated with the UIPC implicit ground.
Runtime pad transforms, native fixed/contact attributes and surface gaps will
be recorded before reset and after native steps. Existing render-buffer sampling
will record initialization only, with no extra physics/render/sync or Agent.

The common native600/outer900/cleanup300 limits, max_iter1024 and prior Timer/Info
instrumentation remain. Expected initialization timeouts are comparison outcomes:
continue the next predeclared cell only after cleanup. Any unrelated startup,
configuration, CUDA or cleanup error pauses. No replacement attempts. Maximum9
new starts brings related R1.12 total to20; formal9 valid/1 failed/22not_run and
R1.11 remain unchanged. Runtime results were pending at that historical prelaunch checkpoint; final results follow.


All nine fixed starts completed reset; no task-body/checker/model runs occurred.
Related R1.12 starts are now20, while the formal9 valid/1 failed/22not_run ledger
is unchanged. No candidate was adopted and no formal recovery was started.

| Repeat | O reset seconds / first Newton count | K | Z |
| --- | --- | --- | --- |
| 1 | 48.684 / 140 | 47.777 / 87 | 45.712 / 60 |
| 2 | 46.692 / 118 | 45.828 / 59 | 46.955 / 69 |
| 3 | 46.547 / 152 | 120.370 / 1024 | 43.660 / 71 |

K3's second step also required844 scopes; native Info records one max-iteration
exit. K therefore did not reliably remove slow startup computation. Z reduced
first-step iterations in all three paired repetitions, but full-reset time was
not always lower and all O runs were normal. This does not establish a repair
for historical persistent slow steps or stability across seeds. Complete30-step
counts, nested timing and common-step differences remain in the diagnostic CSV.

Native fixed flags were1 only for K; contact element and collision surfaces
remained present. Recorded pad-to-UIPC-ground surface distances stayed positive:
K about1mm, O/Z settling near0.48mm. Z fell after creation; K stayed fixed and its
native target reference z became0.017000m versus about0.016486m for O/Z, with no
compensation. Thus these interventions change scene semantics. After construction,
other prism translations differed by up to0.681mm across conditions despite
unchanged inputs; robot joint arrays matched. Do not claim identical initial state.
Native initialization counters239–242 exclude two earlier construction UIPC
frames, which are separately visible in native logs (total241–244). Neither is
zero motion. Early construction camera frames were not fabricated.

The recorded render buffers are replayed at1× and0.1× simulation time; solver
wall time is shown separately in curves. Raw same-time refreshes are retained,
while video takes the last real image per timestamp. Recording/Info overhead and
resource variation limit precise wall-time attribution; some offline encoding
overlapped the later matrix cells. Twelve scoped tests passed; runtime configs
match across all nine, and all processes cleaned up. See
[goal-pad diagnostic table, curves and initialization replay](http://127.0.0.1:9401/artifact?run=univtac-isaac51-r112/pad-scene-diagnostic&path=report.html)
and `outputs/univtac-isaac51-r112/pad-scene-diagnostic/report.md` for full values.
All18 final videos with unavailable-camera warnings reached browser playback
end; this verifies playback only, not successful capture of dynamic camera views.

Post-matrix media inspection found head/wrist frame IDs stayed1 and sensor time0
throughout all nine runs: the passive `_data` reads did not trigger camera lazy
buffer updates. Only bilateral tactile frames refreshed. The raw camera caches
remain evidence but are not current visual trajectories; delivery masks those
panels as unavailable and preserves actual touch/state/geometry. Complete dynamic
four-view recording was not achieved. This was found after all nine had ended;
no physical restart or retrospective control change is authorized.


### Four-way Insert Hole capacity test — initialization failed

The unscored fixed batch used seeds 1000040–1000043 and the existing successful
C_live input: official instruction/public rules, two official visual/tactile
expert measured-motion demonstrations, and current four-view observations.
The original controller, benchmark, physics and r09 runtime stayed unchanged.
Native reset remained 120 s and outer ready 900 s; the Grasp & Classify 600-second
override was not imported. Four workers started within 0.037 s, but none reached
ready: zero Codex contexts and zero autonomous rollouts completed. This is an
infrastructure capacity failure, not Agent success 0/4.

Across 121 approximately one-second samples, GPU usage peaked at 32,061 MiB with
only 40 MiB free; system MemAvailable reached 22.96 GiB and swap grew from 3 MiB
to about 2 GiB. Seed 1000040 failed a 256 MiB PhysX GPU allocation and could not
create its physics scene; 1000040/1000041 logs explicitly report GPU OOM.
1000042 had entered native pre_move before cancellation. The batch aborted and
all four worker groups were cleaned; batch wall time was 120.53 s. No four-ready
residency interval, autonomous action overlap or evaluable task video exists.
No retry or reduced-concurrency fallback ran within that four-way batch.

The bounded entry point is `uv run --no-sync python scripts/univtac/run_fourway_capacity.py
--output-root outputs/<fresh-dir>`; it defaults to the retained successful C_live
configuration. The all-ready barrier and operator release are capacity-only;
normal single-worker control/observation semantics remain unchanged.
[Capacity report and raw evidence](http://127.0.0.1:9401/artifact?run=univtac-fourway-capacity&path=report.html)
record this attempt. R1.12 remains 9 valid/1 failed initialization/22 not_run;
the proposed D condition and eight-task rollout remain unstarted.


### Two-way Insert Hole capacity test — completed, unscored

A separate fixed batch reused `run_fourway_capacity.py --concurrency 2`
for seeds 1000040/1000041 with C_live and gpt-6-astra/low. Both workers
initialized concurrently, passed the all-ready barrier, and completed autonomous
episodes with native success and natural Codex exit. No infrastructure error
was recorded; both lanes cleaned up. Batch wall time was 385.17 s. Worker
lifetimes were 381.44/369.29 s and Codex lifetimes 100.08/87.98 s.
Each episode used 5 motion requests, 9 tool calls, 17 control steps and
34 physics steps (0.2833 s of task simulation, excluding initialization).

Across 385 resource samples, GPU memory peaked at 20,726 MiB with a minimum
11,376 MiB free; minimum system MemAvailable was 53.33 GiB. Swap started
at 2,147,438,592 bytes (about 2 GiB of pre-existing occupancy), with zero
sampled peak increase, and ended at 2,146,717,696 bytes. Conservative simultaneous
ready residency was 88.028 s, Codex lifetime overlap 87.975 s, and three
pairs of motion execution intervals overlapped for 4.325 s in total.
Each episode recorded 18 samples after existing control/render steps.
Both 1× and 0.05× labelled videos reached browser playback end without error.
Cleanup left GPU usage at 987 MiB and system MemAvailable at 68.96 GiB.

[Two-way capacity report, videos and actual inputs](http://127.0.0.1:9401/artifact?run=univtac-two-way-capacity&path=report.html)
remain separate from the retained four-way OOM evidence. This is an unscored
capacity test, excluded from paper success rates. It establishes only this
Insert Hole two-way run; without a serial control it establishes neither 2×
speedup nor long-term concurrency stability across tasks. D, the eight-task
rollout and the full batch remain unstarted.


### Query evaluation feedback removal — implemented; live acceptance completed

The completed two-way results above used the legacy online-feedback protocol;
they are not acceptance evidence for `native_eval_no_online_task_feedback_v1`.
The fresh-run autonomous, official-ICL and capacity entry points stamp the new
protocol into generated configuration and results. Existing configuration files
are unchanged; explicit historical recovery branches retain their old protocol.
The new protocol removes `check_task` from live UniVTAC registration and Codex
`enabled_tools`; other backends retain their own tool. A query-field allowlist
projects actual MCP returns, including nested execution/observation feedback
and errors. Historical demonstrations retain their recorded success outcome.
Native evaluation/control stopping remains unchanged; voluntary finish is final
and is not an infrastructure failure. Host raw responses are preserved in
`host_tool_trace.jsonl`, evaluator results remain in the usual host artifacts,
and `operator_context.jsonl` records the actual delivered text/images.

53 focused offline tests passed, including native stop timing/priority, no
post-terminal motion, live MCP output/context, tactile strip annotations and
protocol-separated result loading. The existing shared MCP file's unrelated
lint debt was not expanded. `outputs/univtac-no-online-feedback-offline/`
contains read-only projection checks over 18 historical tool responses;
original records and expert outcomes were retained. That implementation/offline
round started no simulator or operator Codex. The separate live acceptance below
subsequently exercised the new protocol. D, the eight-task rollout and the full
batch remain unstarted; R1.12 remains paused under its existing protocol.


### No-online-feedback live acceptance — completed, unscored

A fresh C_live Insert Hole batch used seeds 1000040/1000041, gpt-6-astra/low
and `native_eval_no_online_task_feedback_v1`, reusing
`run_fourway_capacity.py --protocol-smoke --concurrency 2`. This option allows
at most three pre-ready attempts per lane, retains each attempt separately,
and never replaces an accepted ready episode. Both lanes accepted attempt 1:
two simulator starts, two independent Codex contexts, no retries. Each ready
lane proceeded independently; this was protocol acceptance, not another
capacity scan. Batch wall time was 379.09 s.

| Seed | Motions / tools | Control / physics steps | Task simulation s | Codex / worker s | Host native result |
|---|---:|---:|---:|---:|---|
| 1000040 | 7 / 10 | 18 / 36 | 0.3000 | 86.276 / 375.231 | success |
| 1000041 | 4 / 7 | 17 / 34 | 0.2833 | 76.675 / 363.736 | success |

Both actual MCP lists contain only `review_demonstrations`, `observe`,
`mark_point`, `move_to`, `report_issue` and `finish_episode`; `check_task` is
absent. All 17 returns matched host response projection and actual
`operator_context` delivery. Historical demonstrations retained their outcomes
and delivered 24 images each; query observations and motion returns delivered
four images each. Ordinary execution feedback remained available. Both
automatic endings delivered neutral feedback, and both Codex finals explicitly
acknowledged that tools had not supplied task success.

Protocol delivery and the autonomous chain passed independently of host task
outcomes. The only actual post-terminal tool was `finish_episode`, without
additional physics. Post-terminal motion rejection and unobserved failure/error
branches remain offline-test coverage. The focused suite passed 55 tests.
Finals, usage and host results were saved; worker and Codex process trees were
empty after cleanup. The two 1× and two 0.05× videos all reached browser playback
end without error, from 19/18 real samples; final score labels are user-only.

Lightweight sampling recorded GPU peak use 20,951 MiB, minimum GPU free memory
11,151 MiB and minimum MemAvailable 55.26 GiB. Pre-existing swap was
2,143,350,784 bytes with zero sampled peak increase. Cleanup left GPU use at
988 MiB. No benchmark, controller motion, physics, demonstration or budget
change was made.

[New-protocol report, videos and actual inputs](http://127.0.0.1:9401/artifact?run=univtac-no-online-feedback-live-smoke&path=report.html)
remain separate from legacy two-way/four-way evidence and the paused R1.12
ledger. These two episodes do not enter paper success rates and establish
neither old/new protocol equivalence nor absence of performance loss. D,
eight-task rollout and the full batch remain unstarted.


### Eight-task new-protocol coverage — completed

`configs/univtac/eight_task_coverage.yaml` fixes eight new cells at seed
1000040: Insert Hole D and one C cell each for Grasp & Classify, Insert Tube,
Insert HDMI, Pull Out Key, Lift Bottle, Lift Can and Put Bottle in Shelf.
The completed new-protocol Insert Hole C episode is referenced without rerunning
it. All cells retain `native_eval_no_online_task_feedback_v1` and
`gpt-6-astra` / low.

Offline preparation completed for all eight cells. The 32 A/B/C/D demonstration
returns passed actual MCP image decoding and pixel comparisons. Existing Insert
Hole and Grasp & Classify packages were copied unchanged, with D added separately.
The six new task packages preserve recorded action-atom segments rather than
forcing three segments per example. Official episodes 0/1 are used throughout;
Shelf episode 1 records source seed 2, which is retained in provenance.

The runner reuses the existing Coordinator and episode runner, with two slots
including cleanup, at most three pre-ready initialization attempts per cell,
and limits of eight operator Codex starts and 24 simulator starts. Native task
control limits remain 600 for HDMI, 500 for Lift Bottle and 300 for the others.
The preparation suite passed 55 focused offline tests. The implementation was
committed and pushed as `415f56d` before the fixed queue started. R1.12 remains
paused; no full batch was resumed.

Prepare with `uv run --no-sync python scripts/univtac/run_eight_task_coverage.py
--phase prepare`, then run the frozen queue with the same command and
`--phase run`. Artifacts use `outputs/univtac-eight-task-coverage/`; persistent
attempt directories retain the initialization budget across invocations.

The frozen queue completed all eight new autonomous cells with eight operator
Codex processes and nine simulator starts in 1616.16 wall seconds. Lift Can
required one preserved pre-ready initialization retry: native reset timed out
after 153.77 s against its unchanged 120 s limit. Attempt 2 was accepted. No
accepted episode was rerun. All eight accepted episodes were evaluable; all
nine worker lifecycles report complete cleanup.

| New cell (seed 1000040) | Native outcome / ending | Motion requests / physical motions | Tools | Control / physics steps | Simulation s | Codex wall s | Input / output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Insert Hole D | success | 5 / 5 | 8 | 14 / 28 | 0.2333 | 117.18 | 345379 / 754 |
| Grasp & Classify C | success | 2 / 2 | 7 | 14 / 28 | 0.2333 | 121.73 | 269383 / 621 |
| Insert Tube C | early stop, failure | 6 / 6 | 9 | 12 / 24 | 0.2000 | 97.23 | 509914 / 976 |
| Insert HDMI C | success | 8 / 7 | 16 | 13 / 26 | 0.2167 | 150.18 | 870747 / 1634 |
| Pull Out Key C | success | 5 / 5 | 8 | 62 / 124 | 1.0333 | 99.98 | 413336 / 898 |
| Lift Bottle C | early stop, failure | 24 / 24 | 37 | 377 / 754 | 6.2833 | 876.82 | 3613599 / 4240 |
| Lift Can C | early stop, failure | 2 / 2 | 6 | 22 / 44 | 0.3667 | 130.63 | 460397 / 593 |
| Put Bottle in Shelf C | voluntary finish, failure | 9 / 9 | 16 | 299 / 598 | 4.9833 | 389.17 | 1137735 / 2326 |

All eight Codex processes exited naturally with final text and actual usage.
Cached input and reasoning output remain subsets in raw usage; they are not
added again to the input/output totals above. The 107 actual tool responses
matched host projection and recorded operator context. No `check_task` was
registered. Historical expert outcomes remained visible, current-query scoring
remained host-only, and all actual post-terminal calls preserved physics counts.
Unobserved terminal/error branches remain covered by offline tests.

Sampled GPU usage peaked at 20807 MiB; minimum MemAvailable was 53.28 GiB,
with no sampled swap increase. GPU use after cleanup was 908 MiB. The failed
initialization is retained separately from four unsuccessful task outcomes.
The prior Insert Hole C episode is a separate coverage reference, not a ninth
new result. This one-seed coverage check establishes neither ICL gains nor
multi-seed reliability.

[Eight-task report, real inputs and 1×/0.05× videos](http://127.0.0.1:9401/artifact?run=univtac-eight-task-coverage&path=report.html)
uses real recorded frames; slow playback adds display time, not observations.
Host outcome labels are user-only. No benchmark, motion algorithm, physical
parameter, solver setting or success threshold changed.


