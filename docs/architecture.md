# OpenETA architecture

OpenETA separates the model-visible Operator interface from simulator control,
optional perception services, and retained evaluation artifacts.

## Operator boundary

A fresh Operator process receives the task-specific startup prompt and the six
tools declared by the active context profile. It runs in an isolated workspace
and Codex home without repository files, earlier sessions, memories, or
unrelated MCP servers.

The released OpenETA-Light profile is content-addressed. It resolves the
startup prompt, tool descriptions, result contracts, and renderer modes before
the Operator MCP server registers its public tools. Profile integrity failure
stops launch rather than falling back to another context.

## Gateway

`tools/embodied_mcp_server.py` exposes the public MCP tools.
`tools/embodied_gateway.py` owns one episode and translates each tool call into
simulator operations, geometry authoring, compact model-visible results, and
visual feedback.

The gateway keeps model-visible results separate from retained diagnostic
details. Privileged simulator state and full controller telemetry are written
to episode artifacts but are not injected into the Operator prompt or tool
results.

## Simulator boundary

`sim/mcp_server/` owns simulator sessions and routes each environment to an
isolated benchmark worker. Each evaluation attempt has its own environment
handle, seed, simulator ports, Gateway ports, and artifact root. Calls inside
one attempt remain serial; independent attempts may run concurrently.

Task success comes only from the simulator-native checker. The Gateway latches
terminal success so a later action cannot invalidate a completed task before
`finish_episode` records it.

### UniVTAC development boundary

The `tactile-agent-for-univtac` branch vendors the official UniVTAC source at
`third_party/ftp1-policy/UniVTAC` for legacy provenance. Current experiments use
a separately installed, pinned Isaac 5.1 source and a project-scoped direct
harness under `sim/envs/univtac/` and `scripts/univtac/`; they do not import the
vendored tree as a generic OpenETA backend.

### Implemented UniVTAC paths

The direct harness retains visual/tactile observations, exact Codex MCP inputs,
real action traces, native outcomes, and dashboard replay. It includes read-only
capture as well as executed R1.0–R1.3 experiments:

- [`tools/univtac_operation_mcp_server.py`](../tools/univtac_operation_mcp_server.py)
  exposes `observe / execute_skill / finish_episode` to a live Pull Out Key
  worker. Its skills wrap task-specific native operations; they do not let
  Codex choose arbitrary motion parameters.
- [`tools/univtac_insert_hole_icl_mcp_server.py`](../tools/univtac_insert_hole_icl_mcp_server.py)
  exposes demonstration review, query observation, and one opaque skill choice.
  [`probe_branching_task.py`](../scripts/univtac/probe_branching_task.py)
  executes the expert-assisted correction and final insertion afterward.

These are real action paths, not general-tool autonomous operation. In these
historical pilots the query's privileged state and native evaluation stay
host-side; demonstration outcome labels may be part of the example context.

### Direct-operation integration — autonomous development batch completed

The design reference is OpenMOSS/OpenETA's `openeta-for-codex` branch. R1.4 adds
`LiveBackendGateway` in the existing [Gateway module](../tools/embodied_gateway.py)
and a six-tool live backend mode in the [Operator MCP](../tools/embodied_mcp_server.py).
The original LIBERO Gateway remains separate.

```text
vision + bilateral tactile history + proprioception + operation history
    -> Codex -> observe / mark_point / move_to
    -> executor -> fresh observations and execution feedback -> Codex
    -> minimal check_task native success boolean
```

The [UniVTAC session](../sim/envs/univtac/autonomous_session.py) connects the
general tools to a synchronous [Isaac51 worker](../scripts/univtac/serve_autonomous_worker.py).
This direct worker route does not require generic simulator registry registration.
The [controller](../sim/envs/univtac/autonomous_operation.py) uses robot-state
Jacobian IK and native qpos control with `force=False`; target resolution supports
world/TCP translation, orientation, gripper commands and numeric previews.
Two separate unscored debug runs exercised translation, orientation, gripper
commands, measured feedback, fresh images, and native control/physics counters.
Small translations, rotations and gripper close reached their targets. Gripper
open moved the fingers but correctly reported `control_segment_not_reached`
after its 80-control-step segment. This does not prove every command reaches.

`mark_point` supports head-camera RGB-D; back-projection was checked against
retained sensor data. Wrist geometry is unavailable because its pose cache does
not track the articulation. Head/wrist RGB and both tactile images remain
available. Three fresh no-demo Terra medium episodes completed with native
success 0/3; all were evaluable and ended through native early stop. Workers
closed normally. In this frozen batch, Codex was stopped after a 15-second
terminal grace period, so complete token usage is unavailable. The post-batch
launcher allows 300 seconds for final reporting within the original 3600-second
Codex deadline; R1.5 subsequently verified natural finalization and real usage
for all three episodes without rerunning R1.4. Contact reliability remains
limited: the frozen R1.4 omitted-gripper path
reset the target to measured opening, not the prior commanded closing target.
Its effect on grasp retention is unverified. Pose arrival also does not prove
velocity settling. R1.4 touch observations were between tool calls; the inner IK
controller still uses robot state, not tactile feedback. R1.5 preserves the two
submitted gripper targets, samples existing four-view buffers after native
control steps, and returns segment-end tactile strips through the same MCP.
`arm_reached` and finite gripper-wait completion are separate; neither implies
stable contact. Full review video, raw frames and feature scores remain separate
from the exact model-visible context. See [tactile history](../sim/envs/univtac/tactile_history.py).

Preserve official reset/`pre_move` initialization. At the official policy handoff,
the Agent chooses targets, direction, magnitude, orientation, and gripper actions,
including observation, retreat, and recovery within the episode budget. The
executor may solve IK and interpolate the requested trajectory; it must not use
the task expert, hidden target poses, expert prefixes, correction formulas, or
automatic final insertion to decide the task body. `check_task` returns only
minimal native success feedback, with no hidden target error or action advice.

The [UniVTAC research plan](univtac/research-plan.md) owns the example format,
A/B/C controls, historical result boundaries, and next work. This architecture
section distinguishes historical R1.4 evidence from R1.5: one unscored debug
and three evaluable no-demo episodes, native success 0/3. Two native-terminal
episodes used 20.767/32.568 seconds of reporting grace; the third finished
voluntarily. No tactile controller or interruption is added. Holding gripper
targets and delivering real history do not establish stable grasp or ICL benefit.

## Geometry and visual feedback

`tools/pointcloud_pose_marking.py` back-projects calibrated RGB-D observations,
renders orthographic views, solves multi-view point constraints, and creates
current-only mark and pose-preview images. Agentview and wrist clicks use their
aligned RGB-D surface directly when valid.

The same target resolver is used for pose preview and execution. Close previews
are immutable and require an explicit preview ID to commit. Motion feedback
reports measured endpoints; LIBERO may additionally render small sampled robot
contact markers after `not_reached` without assigning a collision reason.

## Artifacts and evaluation

`logger/` retains append-only events, images, contracts, and episode
projections. Replay tools consume those artifacts without changing the live
episode.

`scripts/embodied/libero_operator_coverage.py` creates deterministic task and
seed manifests, launches independent Operators, validates provider and
reasoning-effort identity, and reports native-checker success separately from
infrastructure validity.

See [OpenETA-Light interface](openeta-light.md) and
[LIBERO evaluation](libero-evaluation.md) for the public contracts.
