# Bounded Visual History and Main-View VDM

## Status

- Decision date: 2026-08-13
- Scope: Agent-runtime-local context projection and derived visual memory
- Shared schema impact: none
- Implementation status: implemented; live benchmark ablation pending

This document defines the first production design for visual history in the
OpenETA Stage 2 Agent. It combines a bounded raw-image window with a
task-conditioned visual differencing module (VDM), while keeping the complete
multimodal trace available for audit, replay, and later reprocessing.

The design is deliberately deterministic. It does not infer a task phase, score
image importance, or ask the host to decide which manipulation state the Agent
is in.

## Decision

OpenETA will use:

> an immutable multimodal trace, a bounded camera-role-aware raw visual window,
> and a task-conditioned main-view visual-delta ledger.

For planner turn `t`, where the current observation is `O_t`:

- persist every observation and every image artifact;
- attach the fixed main view from `O_0` and the latest three observations,
  including `O_t`;
- attach the wrist RGB image only from `O_t`;
- do not attach depth as a default planner image;
- create one VDM record for every adjacent pair of observations using only the
  configured fixed main view;
- project VDM records that bridge the initial observation to the beginning of
  the recent raw-image window;
- treat VDM output as fallible derived evidence, never as a trusted environment
  receipt or sole task-completion verdict.

The shared `EnvObservation` schema remains unchanged. The raw observation stays
the adapter/runtime boundary; visual history and VDM are Agent-owned derived
records.

## Why the VDM uses only the fixed main view

A fixed external camera normally observes most workspace changes and has stable
extrinsics across turns. This makes before/after comparison well-defined.

A wrist camera has a small field of view and moves with the end effector. A
large fraction of its apparent difference can therefore be caused by camera
motion rather than a world-state change. It remains useful for current local
evidence such as grasp closure, slippage, contact, insertion, and occlusion, but
is not the default source for compressed long-term visual history.

The policy is fixed by camera role:

| Source | Raw planner window | VDM input | Persisted |
| --- | --- | --- | --- |
| Fixed main RGB | `O_0` plus latest three turns | Yes | Yes |
| Wrist RGB | Current turn only | No | Yes |
| Depth | Not attached by default | No | Yes; available to perception tools |

This is a static projection rule. The host must not enable wrist VDM based on a
grasp stage, expected action, task label, or any other task-policy state.

## Raw visual window

The latest-three count includes the current observation:

```text
main_view_turns(t) = dedupe([0] + range(max(0, t - 2), t + 1))
wrist_view_turns(t) = [t]
```

For example, at `O_5`:

```text
durable observations: O0  O1  O2  O3  O4  O5
main-view images:     O0          O3  O4  O5
wrist images:                             O5
VDM bridge:           D1  D2  D3
                      0→1 1→2 2→3
```

The boundary delta `O_2 -> O_3` is included. Without it, the compressed middle
history would not connect to the first observation in the recent raw window.

At most five raw images are attached for the standard two-camera setup: four
fixed-main images and the current wrist image. Missing or duplicate artifacts
are represented explicitly and are not silently replaced by another camera.

## Camera identity

The runtime must select cameras by a configured semantic role, not by list
position. `EnvObservation.cameras` is a list and `cameras[0]` is not a stable
contract for the main view.

Initial defaults:

```text
OPENETA_VDM_CAMERA_ROLE=agentview
OPENETA_VISUAL_RECENT_MAIN_TURNS=3
OPENETA_VISUAL_INCLUDE_INITIAL_MAIN=true
OPENETA_VISUAL_INCLUDE_CURRENT_WRIST=true
OPENETA_VISUAL_VDM_ENABLED=true
```

Role resolution may use the image artifact `role` first and `frame_id` as a
backward-compatible alias. It must emit the resolved role and frame id in every
derived record. If the configured main camera is unavailable, that VDM record
is `unavailable`; the runtime must not substitute the wrist camera.

## VDM lifecycle

After `O_t` is materialized and persisted, the runtime produces:

```text
D_t = VDM(task, main_rgb(O_{t-1}), main_rgb(O_t))
```

The VDM call is isolated from the main planner conversation. Its inputs are:

- the current task objective;
- a labelled previous fixed-main RGB image;
- a labelled current fixed-main RGB image;
- an instruction to report only observable changes, progress evidence,
  completion evidence, and uncertainty.

The VDM must not receive the main Agent's chain of thought, expected outcome,
task phase, or proposed next action. This avoids turning expected behavior into
confirmation bias.

The first implementation may execute the VDM synchronously for reproducibility.
It must use a separately constructed backend/client so its token limit, image
limit, timeout, retry policy, and usage can be measured independently. Later
pipelining is allowed only if the next planner context never observes a partial
or ambiguously ordered delta ledger.

## Derived record

The runtime owns the envelope and provenance. The VDM supplies only bounded
semantic fields.

```json
{
  "schema_version": "openeta.visual_delta.v1",
  "delta_id": "visual_delta:3:agentview",
  "from_observation": {
    "step": 2,
    "evidence_id": "observation:2:agentview",
    "frame_id": "agentview",
    "path": "..."
  },
  "to_observation": {
    "step": 3,
    "evidence_id": "observation:3:agentview",
    "frame_id": "agentview",
    "path": "..."
  },
  "source_camera_roles": ["agentview"],
  "excluded_camera_roles": ["wrist"],
  "status": "available",
  "visible_changes": ["..."],
  "task_progress_evidence": ["..."],
  "completion_evidence": ["..."],
  "uncertainties": ["..."],
  "model": "...",
  "prompt_sha256": "...",
  "created_at_s": 0.0
}
```

Valid statuses are initially:

- `available`: a valid bounded response was produced;
- `no_visible_change`: the VDM explicitly found no relevant visible change;
- `unavailable`: an input artifact was missing or unreadable;
- `failed`: the VDM request or response validation failed.

Provider errors and invalid output are recorded without raw secrets or base64
payloads. Full source images already exist in the immutable artifact trace and
can be used to regenerate a failed delta later.

## Model-facing projection

The planner receives image paths and evidence metadata in chronological order:

1. initial fixed-main anchor, if distinct from the recent window;
2. recent fixed-main images from oldest to current;
3. current wrist image, if available.

It also receives `visual_history`:

```json
{
  "policy": {
    "main_camera_role": "agentview",
    "include_initial_main": true,
    "recent_main_turns": 3,
    "include_current_wrist": true,
    "vdm_camera_roles": ["agentview"]
  },
  "raw_evidence": [],
  "compressed_deltas": [],
  "coverage": {
    "from_step": 0,
    "through_step": 5,
    "missing_delta_steps": []
  }
}
```

Every image remains individually labelled with observation step, camera role,
frame id, timestamp when present, freshness, and evidence id. The current image
must remain distinguishable from historical images. VDM records are labelled as
derived historical evidence and must not be presented as current scene truth.

Recent raw-image transitions do not need duplicate VDM text in the first
baseline. The deltas are still generated and persisted so they are ready when a
turn later leaves the raw window.

## Persistence and resume

The full history requirement applies to both raw and derived evidence:

- raw RGB/depth artifact files remain in the session workspace;
- each observation event retains stable artifact references and camera
  provenance;
- visual-delta records are append-only and keyed by session, adjacent
  observation steps, and main camera role;
- resume reconstructs the same visual window and delta bridge without rerunning
  already valid VDM calls;
- corrupt, missing, or duplicate delta entries fail visibly and never reorder
  the observation timeline.

The model-facing window is bounded; the durable trace is not.

## Trust and completion boundary

VDM output is useful but fallible. It is not equivalent to:

- an environment reward;
- a trusted environment receipt;
- a checker verdict;
- structured robot/object state;
- direct current visual evidence.

`completion_evidence` may support the Agent's reasoning but cannot by itself
authorize a successful terminal result. The existing completion contract still
requires official reward/checker evidence, structured state change, or fresh
direct visual evidence as appropriate to the task.

The VDM prompt must allow uncertainty and `no_visible_change`. It must not infer
unobserved causes, claim contact from motion alone, or turn an occluded target
into a confident completion claim.

## Failure and budget behavior

- Missing main image: persist `unavailable`; continue with the bounded raw
  window.
- VDM timeout/provider failure: persist `failed`; do not block or discard the
  episode trace.
- Invalid VDM response: persist a compact validation error and treat the delta
  as failed.
- Image exceeds provider limit: record the exact exclusion reason; do not
  silently attach a different camera.
- Planner image budget below the configured policy requirement: expose a
  configuration error or explicit degraded-policy record; do not truncate by
  list order.
- The VDM uses a 2048-token completion allowance so reasoning-capable providers
  can finish their hidden reasoning and still emit the required JSON. Its four
  semantic arrays remain schema-bounded. Older delta records are retained until
  the main planner's combined token projector needs to remove them; this does
  not change the bounded raw visual-window rule.

## Shared TUI and batch assembly

Interactive TUI and batch evaluation must receive the same defaults and the
same visual-history component through `assemble_runtime()`. Entry points may
override configuration values, but they must not implement their own history
selection, camera ordering, VDM prompt, or persistence behavior.

The visual-history manager belongs to the Agent runtime/assembly boundary and
must be usable by dummy backends in tests. The main planner, VDM backend, and
supervision/checker backends remain isolated roles even when they share the same
provider endpoint.

## Evaluation plan

### Deterministic tests

- window indices for turns 0 through at least 8;
- overlap and deduplication of `O_0` with the recent window;
- chronological ordering of raw image attachments;
- main view selected by role rather than camera-list position;
- current wrist included but historical wrist excluded;
- depth excluded from planner attachments;
- VDM receives exactly two fixed-main images and no wrist image;
- boundary delta connects compressed history to the recent raw window;
- VDM records persist and survive resume without duplicate calls;
- missing images, client failures, invalid JSON, and `no_visible_change` degrade
  explicitly;
- VDM completion text cannot masquerade as trusted completion evidence;
- TUI and batch runtime assembly yield the same visual-history configuration.

### Context probes

Extend the existing visual-state probes with:

- a long trajectory in which the decisive change leaves the raw window;
- a moving wrist view with an unchanged fixed main view;
- main-view occlusion with useful current wrist evidence;
- contradictory or hallucinated VDM text versus current raw images;
- a missing middle delta;
- resumed-session reconstruction.

### Ablations

Compare at least:

| Variant | Raw visual context | Compressed history |
| --- | --- | --- |
| A | Current main only | Existing text transitions |
| B | Initial plus latest-three main; current wrist | None |
| C | Same as B | Main-view VDM bridge |
| D | Current main only | Main-view VDM ledger |
| E | Same as C | Main plus wrist VDM, evaluation-only |

Measure task success, visual-state probe accuracy, evidence citation accuracy,
VDM hallucination/omission rate, image count, prompt tokens, VDM tokens,
provider calls, latency, and cost. Variant C is the proposed production default;
Variant E exists only to test the decision to exclude wrist VDM.

The deterministic A-D projection report can be generated from a persisted
session without any provider call or data egress:

```bash
openeta-visual-history-eval \
  --memory-root .openeta_memory \
  --session-id <session-id> \
  --strict
```

## Acceptance criteria

The implementation is complete when:

1. raw artifacts and observation references remain durable for all turns;
2. the planner receives exactly the configured bounded visual window;
3. only the configured fixed main view enters VDM;
4. evicted middle observations are connected by ordered, persisted VDM records;
5. VDM failure cannot corrupt, block, or falsely complete an episode;
6. TUI and batch share one implementation and one default configuration;
7. focused unit/integration tests and context probes pass;
8. ablation output exposes enough usage and accuracy data to compare the policy
   with current-only and no-VDM baselines.

## Deferred extensions

These are intentionally not part of the first implementation:

- host-selected camera importance;
- task-stage-triggered wrist VDM;
- object tracks or keyframe heuristics;
- VDM-authored task state or required next actions;
- automatic substitution of missing camera roles;
- unbounded hierarchical visual/text summarization;
- on-demand historical wrist inspection. The full trace preserves the option to
  add an explicit Agent inspection tool later if evaluation demonstrates a need.

## Influences

- [CaP-X](https://arxiv.org/abs/2603.22435) motivated the isolated,
  task-conditioned adjacent-turn visual differencing client and the comparison
  between raw visual feedback and VDM-mediated feedback.
- MAI-UI motivated retaining the complete trajectory while projecting a bounded
  recent visual window to the model.
