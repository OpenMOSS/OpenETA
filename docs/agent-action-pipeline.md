# OpenETA Agent Command Pipeline

This document defines the first structured agent-command schema and execution
pipeline for the lightweight OpenETA agent runtime.

Cross-tool result projection, packet-validity semantics, no-progress reflection,
and the durable rollout auditor are specified in
[`tool-contract-audit-and-grasp-recovery.md`](tool-contract-audit-and-grasp-recovery.md).

## Goal

The agent should not send an unstructured dict to the simulator. The primary
OpenETA path is closed-loop tool calling: the agent receives observation,
memory, tool references, and skill references, then chooses one next
`tool_call` or `response`. Skill references are text guidance documents, not
executable task macros. The tool result and updated observation are fed back
before the next world-changing decision. That agent decision is compiled into a
stable command payload that can represent:

- ordinary tool calls such as `sam3`, `anygrasp`, or `move_to`,
- tool-like interfaces such as `skill_call`, `safe_check`, `code_policy`, and
  `sense`,
- response subtypes such as `ask_human`, `talk`, and `task_complete`.

The simulator does not need to execute all of these immediately. The first
contract is a structured plan that downstream MCP simulator tools can inspect
and incrementally support.

## Naming Convention

OpenETA reserves these terms to avoid mixing agent decisions with embodied
control actions:

| Layer | Preferred term | Current code mapping | Examples |
|---|---|---|---|
| Task-level guidance | `Skill` / `TaskSkill` | `SkillSpec` markdown | `pick`, `place`, `stack` |
| Agent-callable capability | `Tool` / `AgentTool` | `ToolSpec` | `sam3`, `anygrasp`, `move_to`, `save_memory` |
| Embodied physical primitive | `AtomAction` | subset of `ToolSpec(category="control", effect="world_mutating")` | `move_to`, `gripper_control` |
| Agent runtime decision | `AgentCommand` | `CommandRequest` plus `CommandKind` | `tool_call`, `response` |
| CLI user input command | `SlashCommand` | `agent.cli.openeta_cli` slash commands | `/provider`, `/run`, `/step` |
| Sim/env execution payload | `Action` / `EnvAction` | `adapter.protocol.EnvAction` | structured compatibility/logging payload; real simulator execution is routed through MCP tools |

The runtime schema uses `CommandKind`, `CommandRequest`, and
`CommandPipelinePlan`. Former `Action*` aliases have been removed from the
OpenETA runtime API.

## Agent Command Kinds

Target OpenETA wire schema version: `openeta.agent_command.v1`.

Collaboration-doc note: an earlier 5.2 draft described `code_policy` as the
main control path and exposed many top-level action kinds. That statement is
now superseded. The agreed `CommandKind` top level has only `tool_call` and
`response`; `code_policy` remains a bounded backend under `tool_call`.

Supported `AgentCommand.request.kind` values:

| Command kind | Meaning |
|---|---|
| `tool_call` | Invoke a named agent tool or tool-like interface. Examples: `sam3`, `anygrasp`, `move_to`, `skill_call`, `safe_check`, `code_policy`, `sense`. World-mutating control tools are `AtomAction` tools. A `tool_call` may carry a `calls` array for a restricted read-only/planning batch. `safe_check` can be requested explicitly by the agent for planning/preview, and can also be inserted automatically by the pipeline as an execution gate before selected tools. |
| `response` | End the current agent turn with text/status, a human-facing request, or explicit task completion. Examples: `ask_human`, `talk`, `task_complete`. |

Planner payloads must use exactly one of these two top-level kinds. Historical
top-level forms such as `skill_call`, `safe_check`, `code_policy`, `ask_human`,
`sense`, and `talk` are invalid; they must be represented as `tool_call` or
`response` with an explicit `name`. `noop` is not part of the planner-facing
command surface; use `talk` for a human-readable status or `task_complete` when
the task is finished.

## EnvAction Shape

`EnvAction` is the structured compatibility/logging payload for an agent turn.
`EnvAction.action_type` mirrors `AgentCommand.request.kind` for compatibility,
but adapter authors should treat the embedded `command.request` as an agent
decision, not as a raw robot actuator action.

`EnvAction.command` carries the `AgentCommand` payload and currently uses this
shape:

Primary `tool_call` example:

```json
{
  "schema_version": "openeta.agent_command.v1",
  "request": {
    "kind": "tool_call",
    "name": "sam3",
    "parameters": {
      "source_packet_id": "env-abc-observation-0001",
      "prompt": "red cube"
    },
    "reasoning": "Locate the target object before selecting a grasp."
  },
  "status": "pending",
  "safety_checks": [],
  "tool_calls": [
    {
      "kind": "tool_call",
      "name": "sam3",
      "parameters": {
        "source_packet_id": "env-abc-observation-0001",
        "prompt": "red cube"
      },
      "status": "pending",
      "reason": "Direct planner-requested tool call. No handler registered yet."
    }
  ],
  "metadata": {
    "interface": {
      "kind": "tool_call",
      "name": "tool_call",
      "implemented": false
    },
    "execution_rule": {
      "mode": "single_tool_closed_loop",
      "effect": "read_only",
      "batchable": true,
      "requires_observation_after_call": false
    }
  }
}
```

Restricted `tool_call` batch example:

```json
{
  "kind": "tool_call",
  "name": "tool_batch",
  "parameters": {
    "calls": [
      {"name": "sam3", "parameters": {"source_packet_id": "env-abc-observation-0001", "prompt": "cube"}},
      {"name": "hand_pose_database", "parameters": {"object": "cube", "task": "pick"}}
    ]
  }
}
```

This is valid because both tools are read-only or planning helpers. A batch that
contains a world-mutating tool such as `lower_body_control_policy` is compiled
as `blocked`.

The `EnvAction`/`AgentCommand` JSON above is the internal, post-parse logging
shape. The live main-planner wire format is XML:

```xml
<decision>
  <kind>tool_call</kind>
  <name>python_exec</name>
  <reasoning>Inspect the persisted result.</reasoning>
  <parameters>
    <code><![CDATA[
result = artifacts.read_json('/workspace/result.json')
]]></code>
  </parameters>
</decision>
```

The response must contain exactly one `<decision>` with `kind`, `name`,
`reasoning`, and `parameters`. Tool arguments are named children of
`parameters`; an argument-free call uses `<parameters/>`. Repeated `<item>`
children encode lists and named children encode nested objects. Scalars may use
`type="integer"`, `type="number"`, `type="boolean"`, or `type="null"`.
Multi-line code and quoted text should use CDATA. `tool_batch` uses repeated
`<call>` children under `<calls>`, each with its own `name` and `parameters`.

`ToolCallingPlanner` converts XML to the existing internal decision dict before
schema validation and rollout recording. Host-generated failure payloads and
deterministic test fixtures may still provide an already-parsed dict. Isolated
sub-agents retain their explicit JSON contracts and JSON response mode.

## Pipeline Stages

The default runtime stages are:

1. `Planner`: creates a one-step `PlannerDecision`.
2. `ToolCallingPlanner`: builds bounded `tool_context` from the current
   observation, session task, memory summary, tool references, skill metadata,
   selected markdown skill guidance, and execution rules.
3. `PlannerBackend`: returns one decision payload from a placeholder,
   deterministic fixture, callable SDK/API wrapper, future commercial API, or
   local LLM/VLM backend. The live main planner emits XML; fixture backends may
   return the parsed dict directly.
4. Backend validation: `ToolCallingPlanner` parses XML into the internal
   `{kind, name, parameters, reasoning}` shape, validates command kind,
   tool/skill names, parameters, and bounded code-policy requirements, then
   retries with validation feedback before falling back to
   `response::ask_human`.
5. `ActionPipeline`: normalizes that decision into a `CommandRequest`.
6. Compilation: registered tool handlers may execute immediately and return a
   structured `ToolResult`; missing handlers leave calls as `pending`.
7. Memory update: the compiled pipeline plan and final sim/env `EnvAction` are
   logged.

## Tool Handler Contract

`ToolRegistry.call()` is the adapter boundary for executable tools. Handlers
receive a `ToolExecutionContext` with the selected tool spec, planner
parameters, current observation, and pipeline metadata. They may return a
`ToolResult`, a dict, a string, or `None`; the registry normalizes every
successful or failed return into the same `ToolResult.details` envelope:

```json
{
  "schema_version": "openeta.tool_result.v1",
  "tool": "sam3",
  "category": "perception",
  "effect": "read_only",
  "result_type": "perception",
  "success": true,
  "parameters": {"source_packet_id": "env-abc-observation-0001", "prompt": "cube"},
  "outputs": {
    "masks": [{"mask_id": "mask-cube-001", "label": "cube", "score": 0.99}]
  },
  "artifacts": [{"type": "segmentation_mask", "id": "mask-cube-001"}],
  "state_delta": {},
  "diagnostics": [],
  "requires_observation_after_call": false
}
```

`result_type` is derived from the tool contract:

| Result type | Used for | Expected payload focus |
|---|---|---|
| `perception` | perception tools such as `scene_detector` or `sam3` | object lists, detections, masks, camera-derived artifacts |
| `planning` | planning/manipulation/navigation tools such as `anygrasp` | candidate poses, trajectories, plans, scores |
| `safety` | safety tools such as `ik_preview_check` | feasibility flags, blocked reasons, safety margins |
| `bookkeeping` | memory and other runtime bookkeeping tools | saved keys, loaded memory, compact summaries |
| `world_mutating` | control tools such as `move_to` or `gripper_control` | executed command summary and `state_delta` |

Development and CLI smoke runs can bind deterministic dummy handlers with
`bind_dummy_tool_handlers()`. Real simulator, perception, and control handlers
should replace those handlers while preserving the same result envelope.

### Trusted environment feedback

`state_delta` remains a generic tool-feedback channel. It can carry motion
summaries and planner-useful state changes, but it cannot establish benchmark
reward or episode termination. Simulator handlers are registered separately
with the host-owned `environment` authority. After the handler returns,
`ToolRegistry` removes handler-supplied provenance and stamps
`openeta.tool_result_provenance.v1` from the active execution scope.

An authorized Simulator MCP result may additionally carry
`openeta.environment_receipt.v1`. The receipt binds the remote response to the
Agent execution/session, simulator session, environment handle, backend, and
tool call. `ToolFeedbackEpisodeEnvironment` accepts reward and terminal fields
only when that host provenance and all identities match the active episode.
Rewards must be finite numeric values and terminal fields must be booleans.
LIBERO and other official-reward evaluations accept a positive reward only
from such a trusted same-execution receipt.

Fresh simulator state is represented by
`openeta.observation_snapshot.v1`. The snapshot retains robot/object state,
camera frame ids, intrinsics, extrinsics, timestamps, and session-local image
artifact references without embedding image pixels. The compact
`observation_summary` remains a display/planner-summary format and must not be
used to reconstruct `EnvObservation`.

After a successful or transport-unknown world mutation, a receipt without a
fresh snapshot causes the TUI bridge to remove current camera artifacts and
publish a host `fresh_observation_obligation`. The planner dispatches `observe`
without a model decision before another dependent action. Three consecutive
refresh attempts without a snapshot truncate the episode with
`fresh_observation_unavailable`, preventing an unbounded retry loop. Historical
images remain in trace/memory but are never exposed as the current frame.

### SAM3 detection selection evidence

The agent-facing `sam3` ToolSpec has two explicit modes. `mode="text"` (the
default) consumes one session-scoped `source_packet_id` plus a natural-language
`prompt`; `mode="points"` consumes the same packet reference plus 1–64 top-left pixel points
with normalized `{x, y, label}` fields, where label `1` is foreground and `0`
is background and at least one foreground point is required. Prompt and point
inputs are mutually exclusive. The host resolves the packet to an exact RGB
artifact and aligned source observation. Packet ids are unique within the
active Agent session; an individual image is identified internally by
`(source_packet_id, camera_frame_id, kind)`. Unknown ids, missing artifacts,
ambiguous cameras, duplicate ids, and path/id mismatches fail closed. MolmoPoint results are not passed through
verbatim: the planner selects `image_sources[image_index]` and maps
`pixel_x/pixel_y` to `x/y` before calling SAM3.

SAM3 detections are ranked by score while preserving `backend_index` and a
stable ranked detection id. Score is only a ranking hint. Every non-empty SAM3
result creates durable `pending_target_selection` evidence, including the single-candidate
case. The evidence contains the original image, candidate-specific overlay or
crop references, and a contact sheet so the main agent explicitly confirms the
mask before downstream use.

Every non-empty result reports `selection_required=true` and
`selected_detection=null`, including the single-candidate case. `AgentMemory`
creates the matching semantic-confirmation record before targeted grasping or
world-mutating execution; there is no handler-level auto-selection bypass.

Point mode always returns exactly three score-ranked mask candidates and never
auto-selects one. The handler verifies the echoed points, binary mask geometry,
candidate ranks and backend indices, overlays, and coordinate metadata before
materializing the result; any inconsistency rejects the complete response.

The next VLM planner request attaches the original image and contact sheet as
multimodal image parts. The main agent resolves the open question with
`select_sam3_detection(sam3_result_id, detection_id, ...)`. The handler validates
that both ids belong to the pending result and records the selected mask under
the SAM3 request's explicit `evidence_role`. `target_object` is the compatible
default; `placement_region` retains a receptacle independently. Starting or
resolving one role never deletes the other role's mask or source-observation
bundle, and a selection cannot change the role declared by its pending result.
The first target selection creates `openeta.target_identity_anchor.v1`. A later
target selection from different detection evidence must copy its exact
`identity_anchor_id` and declare either `identity_relation=same_instance` or
`identity_relation=replace_misidentified_anchor`. The former is an auditable
cross-view identity confirmation; the latter requires a concrete correction
reason, creates a new anchor, and cannot override an exact reference-verifier
match. Missing or mismatched identity parameters leave the SAM3 selection
pending and return executable correction guidance.
Targeted AnyGrasp and GraspGenX may consume only a mask whose semantic selection
has been explicitly recorded; an unresolved result is rejected as unverified
provenance, with the result id and reason returned to the Agent. After selection,
both predictors must use that exact mask. This is evidence integrity, not an
instruction to call a particular tool next.

Candidates are normalized in score-descending order and persisted in full. The
Agent chooses among them using current visual evidence, calibration, geometry,
checker results, and its own attempt notes. Candidate IDs survive compilation,
motion, review, and receipts. The host records outcomes but never advances a
queue, activates a fallback, or dispatches a retry sequence.

The independent agent-facing `graspgenx` ToolSpec is bound only when the
`openeta-graspgenx` (or `graspgenx`) MCP URL is configured. It requires local
RGB and aligned depth paths, a complete SAM3 mask artifact, finite intrinsics,
an exact advertised gripper name, and a camera-frame up direction. RGB remains
local; the handler sends only encoded depth and mask bytes to MCP. It validates
all returned normalized poses atomically, persists success and failure audit
records under `tmp/tool_result/graspgenx/`, and creates top-1/top-10 RGB
overlays. `list_graspgenx_grippers` is an independent read-only capability
query and does not load model weights or create audit runs.

## Simulator MCP Execution Boundary

The long-term OpenETA agent-runtime contract for simulator communication is
MCP-only. The simulator server may expose REST endpoints for dashboards,
debugging, or local tooling, but agent-side runtime code must not depend on
REST APIs because they are not the stable collaboration boundary.

For simulator-related tools, especially world-mutating control tools such as
`move_to`, `follow_eef_trajectory`, and `gripper_control`, the agent
runtime should proxy the `tool_call` to simulator-side MCP tools. The agent
chooses the tool and high-level parameters; the simulator machine owns
state-dependent execution details:

```text
Agent planner
  -> AgentCommand(tool_call: move_to)
  -> ActionPipeline validation / logging / optional checker hook
  -> MCP simulator tool call
  -> simulator-side IK, safety check, controller expansion, action clipping
  -> env.step(sim_internal_action)
  -> ToolResult + fresh observation
  -> image materialization + next closed-loop turn
```

`env.step(...)` actions and backend-specific action vectors are simulator-side
internal details. They should not become planner-facing parameters and should
not be computed by the agent runtime unless a future simulator MCP tool
explicitly delegates that work back to the agent.

Because simulator-side MCP tools own IK, safety checks, controller expansion,
action clipping, and backend-specific state reads, low-level simulator state is
not part of the required agent-facing MCP observation contract. Agent-side
runtime code should not require camera intrinsics/extrinsics, full end-effector
pose conventions, backend action-space metadata, or step counters merely to
execute control tools. Those fields may still appear as optional diagnostics,
but planner context should prefer high-level tool results and materialized image
references.

Agent-side code for this boundary lives in `agent.tools.sim_mcp`. It provides
MCP-only transports and `bind_simulator_mcp_tool_handlers()`, which binds
simulator-owned `ToolSpec`s to a proxy handler. The proxy supports explicit
tool-name mapping so the planner-facing name can stay stable even if the
simulator-side MCP tool is named differently. The current remote simulator MCP
server maps `observe -> render_env`, `move_to -> move_to`; and
`gripper_control` is routed to `gripper_open` or `gripper_close` according to
the requested binary position (`0=closed`, `1=open`); fractions are rejected.
The close/open command is latched across subsequent motion. Planner context
reports the binary commanded state separately from continuous measured aperture
and from attachment evidence.

Simulator safety requests privileged object geometry internally while preserving
the public `include_objects` boundary. After a non-empty close, object/EEF
co-motion promotes a tentative target to an attached collision proxy. The proxy
is checked against scene geometry before each carry batch, with a receptacle
interior corridor allowing centred insertion but rejecting rim overlap. These
safety details do not become task-stage state.

Environment creation is a stable AgentTool operation. `create_simulator_env`
is the only planner-facing creation path and owns the MCP
`create_env -> reset_env` sequence, 512x512 defaults, artifact materialization,
and active handle/session synchronization. Environment cleanup and simulator
control likewise use stable AgentTools. The restricted `python_exec` runtime has
no Simulator MCP client, so code cannot bypass those lifecycles.

`python_exec` is a `planning` tool for coding-agent-style inspection, filtering,
computation, and derived artifacts. It can read the full current session tree
(rollout, working memory, and artifacts), while writes are confined to the
session sandbox. Complete high-cardinality outputs such as grasp or placement
candidate lists are persisted as immutable JSON; working memory carries a
bounded preview, count, truncation flag, and queryable artifact reference. A
request for
`sandbox="outside_sandbox"` requires explicit approval for each invocation and
runs in a disposable host subprocess using the current OpenETA Python
interpreter and working directory. It has host-level imports, filesystem, and
network permissions, receives only JSON observation/parameter inputs, does not
receive in-process AgentTool or artifact helper objects, and is terminated when
its bounded timeout expires. Because normal `python_exec` has `planning` effect,
its feedback does not require a fresh simulator observation.

The remote simulator MCP server currently exposes stable environment-level MCP
tools such as `create_env`, `reset_env`, `step_env`, `render_env`, and
`close_env`. It also exposes high-level control tools: `move_to`,
`gripper_open`, and `gripper_close`. `step_env` still takes a backend action
vector and remains a lower-level escape hatch; planner-facing control tools
should prefer the high-level simulator MCP tools. The remote tool list does not
currently expose `observe_env`, so agent-side `observe` uses `render_env` for
fresh camera refresh.

Simulator MCP camera observations should return explicit calibration for agent
consumption. The current MuJoCo-backed MetaWorld/LIBERO contract is `pos + mat`:
`pos` is camera position in world coordinates and `mat` is a camera-to-world
rotation flattened row-major. Columns are camera-local axes in world
coordinates, with OpenGL-style camera axes (`+X` right, `+Y` up, camera looks
along local `-Z`, so look direction is `-col2`). Agent-side geometry helpers
such as `camera_pose_to_world` convert perception/grasp camera-frame poses into
world-frame poses before control, including OpenCV grasp pose to OpenGL sim
camera conversions when required. Simulator control tools such as `move_to`
should receive world-frame targets only.

For AnyPlace, the planner-facing conversion contract is reference-only:
`camera_pose_to_world({placement_result_id, candidate_id})`. A successful
AnyPlace call stores the five candidates and their immutable source observation
lineage in host memory. The pipeline resolves the chosen `place_grasp_pose` and
the original packet's camera extrinsics atomically. Mixed calls that combine
these IDs with model-authored pose or calibration fields are rejected. The
generic explicit-pose conversion form remains available for non-AnyPlace
geometry operations.

The resolved AnyPlace transform also returns
`openeta.placement_world_reference.v1`. It explicitly marks the world pose as a
low release geometric reference with `execution_authorized=false`; no fixed
carry script or fabricated clearance is implied. The Agent proposes waypoints
from current visual/EEF evidence, while exact-pose IK and controller-side
trajectory/attached-object collision checks remain the execution boundary.

Any test, smoke run, or integration runner that calls `create_env` against a
remote simulator MCP server must call `close_env` in a `finally` block once the
test is done. Remote env handles consume simulator resources on another
machine; leaking handles is treated as a test failure. Agent-side cleanup code
should use `close_simulator_mcp_env()` for best-effort structured cleanup.

## MCP Image Artifact References

Simulator MCP transport may return camera payloads with inline `rgb_base64`,
`depth_base64`, or `image_base64` fields. These fields are valid at the
transport boundary, but they must not be copied into planner context,
multi-turn memory, or downstream tool parameters because they can dominate the
context window.

Agent-side simulator facades and MCP-backed tool handlers should run the
host-internal `materialize_mcp_images()` utility before exposing an observation
to the planner. It is deliberately not registered as a main-Agent tool because
the Agent never receives raw MCP base64 payloads and therefore has no valid
reason to call it. The
materializer writes each image to `outputs/mcp_images/runs/<bundle_id>/` by
default, removes the inline base64 payload, and inserts lightweight references
such as:

```json
{
  "frame_id": "agentview",
  "rgb_ref": "cameras.0.agentview.rgb",
  "rgb_path": "/abs/path/outputs/mcp_images/runs/session-1/cameras.0.agentview.rgb.png",
  "rgb_base64_omitted": true,
  "width": 640,
  "height": 480
}
```

Later perception or manipulation tools should accept the reference or local
path, not the original base64 string. The `ToolResult.details.parameters` field
must also avoid echoing the original MCP payload.

## Checker Subagent Hooks

`ActionPipeline` can optionally run lightweight checker hooks around selected
tool calls through `CheckerSubagentConfig`. This hook path is intentionally
separate from planner-requested `safe_check` calls:

- Planner-requested `safe_check` is an ordinary `tool_call` subtype. It lets the
  agent preview a candidate action or query a safety tool before deciding what
  to do next.
- Hook-triggered safety checks are runtime execution gates. They protect
  configured high-risk tools even if the planner forgets to ask for a
  `safe_check`.

This is the current placeholder for future safety/failure sub-agents:

- `pre_safety_checks`: maps a target tool to a safety checker tool. Example:
  `{"move_to": "ik_preview_check"}` runs `ik_preview_check` before the
  world-mutating move. If the checker is pending, failed, or returns
  `success=false`, the target tool call is skipped and the command is marked
  `blocked`. The rejection always carries `openeta.gate_repair.v1`, including
  the complete checker call/result, violated invariant, rejected parameters,
  current evidence ids, and executable recovery calls. Empty skipped-call
  feedback is a contract failure. CLI pre-check gates are deliberately opt-in through the
  `OPENETA_PRE_SAFETY_CHECKS` JSON object because its default safety handlers
  are deterministic placeholders, not real safety backends.
- `post_failure_checks`: lists target tools that should trigger a post-tool
  failure checker. The CLI enables this feedback for registered tools; only a
  failed call emits a `failure_check` record with
  `schema_version=openeta.checker_result.v1` under
  `CommandPipelinePlan.metadata.checker_results.post_failure_checks`.
- A blocked or failed pipeline also writes a bounded `recovery_feedback` memory
  event. Its compact command status, request, tool result, and checker metadata
  are visible in the next planner turn for explicit replan/recovery.

Batch-level rejections use the same repair schema per blocked member. No gate
uses a manipulation-stage label as evidence: later motion is evaluated from the
actual EEF state reported by the previous receipt, not from the previous
requested target.

The hook output deliberately stays in `metadata.checker_results` and existing
`PipelineCall` records. A final standalone `SafetyVerdict` / `FailureVerdict`
schema should still be reviewed before becoming a hard cross-team contract.

`skill_call` is a guidance-only `tool_call` subtype:

1. If the request is `{"kind": "tool_call", "name": "skill_call"}`, the
   pipeline looks up the requested `SkillSpec`.
2. The returned `PipelineCall.result.content` contains the skill text guidance,
   plus metadata such as `allowed_tools`, source, version, and editability.
3. No safety checks or tool calls are inserted. The planner must choose the
   next atomic `tool_call` explicitly after reading the guidance.

## Reserved Interfaces

Current code has reserved Python interfaces under `agent/runtime/interfaces.py`
for command subtypes:

| Interface name | Target command kind | Default behavior |
|---|---|---|
| `skill_call` | `tool_call` | Placeholder, returns `pending`. |
| `tool_call` | `tool_call` | Placeholder, returns `pending`. |
| `safe_check` | `tool_call` | Agent-requested safety preview. In the pipeline hook path, safety tools can also be inserted automatically before selected target tools. |
| `code_policy` | `tool_call` | Placeholder, returns `pending`. |
| `sense` | `tool_call` | Placeholder, returns `pending`. |
| `ask_human` | `response` | Placeholder, returns `pending`. |
| `talk` | `response` | Placeholder, returns `pending`. |
| `task_complete` | `response` | Implemented episode-completion response. |

The default `OpenEtaAgentRuntime` creates an `ActionInterfaceRegistry` with all
of these interfaces registered. Replacing a placeholder later should only
require registering a new `ActionInterface` implementation for the same subtype
name.

Default runtime policy:

- `ToolCallingPlanner` is the default planner.
- `PlannerBackend` is the model boundary for closed-loop decisions. It returns
  JSON only; simulator/tool effects remain behind `ActionPipeline` and
  `ToolRegistry`.
- A planner turn should choose at most one state-changing atomic tool
  (`AtomAction`), then observe its result before planning the next
  state-changing step.
- `pick`, `place`, `push`, `pull`, and `stack` are skills, not tools. They are
  editable text guidance for choosing atomic tools.
- Planner context keeps all skills as a metadata index under
  `skill_references`, but only includes relevant markdown bodies under
  `selected_skill_guidance`. `PlannerContextConfig` bounds memory events,
  selected skill count, and skill content length.
- Batched tool calls are allowed only for `read_only`, `bookkeeping`, and
  `planning` tools. `world_mutating` tools are blocked from batches.
- Optional checker hooks can run configured pre-tool safety checks and
  post-tool failure checks. These hooks are sub-agent placeholders, not the
  final checker schema.
- `CodePolicyPlanner`, `CodePolicyBackend`, and `RlinfCodePolicySandbox` remain
  available for optional bounded code-policy snippets behind `tool_call`.
- `RuleBasedPlanner` is retained only for smoke tests and fallback debugging.
- `skill_call` remains available only as a guidance selection/inspection
  `tool_call` subtype. It must not hide a multi-tool executor behind the
  planner.
- `AgentMemory` keeps session events, working facts, artifacts, skill notes, and
  a compact summary. Runtime-owned memory tools (`save_memory`, `get_memory`,
  `delete_memory`, `compact_memory`) update that memory and expose it back
  through planner context.
- `OpenEtaEpisodeRunner` enforces separate budgets for planner turns, concrete
  tool calls, wall-clock time, and cumulative backend tokens. Defaults are 100,
  50, 600 seconds, and 5,000,000 tokens respectively. Resource exhaustion emits
  a structured `failure_reason`; the agent can still end an episode through
  `response::task_complete`, while environment/checker feedback can force
  `terminated=True` or `truncated=True` through `StepResult`.
- Evaluation specs may set `recovery_turns_per_branch` and
  `max_recovery_turns`. Only a distinct post-close compiled-grasp branch switch
  grants turns, so repeated perception and duplicate evidence cannot consume an
  unbounded extension.
- A runner deadline actively abandons a blocked turn, requests environment
  close, and prevents its late result from becoming an `EpisodeStep`. Missing
  provider token usage falls back to the shared TUI token estimator and records
  the accounting source.
- When attached, `JsonMemoryStore` persists each local session under
  `.openeta_memory/sessions/<session_id>/`: `trace.jsonl` stores the automatic
  event trace, `conversation.jsonl` stores resumable model-visible history,
  `working/*.json` stores mutable session memory, and `rollout/` stores
  evaluation evidence. Session-owned skills, tool artifacts, and the Python
  sandbox live under the same UUID root. This directory is local state and is
  ignored by git. The former `.openeta_memory/workspaces/<session_id>/` layout
  is read only as a legacy resume source.
- `agent/memory/` is reserved for curated, developer-accepted project memory.
  Ordinary memory tools must not write automatic runtime state there; future
  promoted-memory tools should make that transition explicit.

Minimal model-backed planner fixture:

```python
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.planner import ToolCallingPlanner

planner = ToolCallingPlanner(
    StaticPlannerBackend(
        {
            "kind": "tool_call",
            "name": "sam3",
            "parameters": {"source_packet_id": "env-abc-observation-0001", "prompt": "cube"},
            "reasoning": "Segment the target before grasp planning.",
        }
    )
)
```

Minimal callable backend wrapper:

```python
from agent.backends.planner import CallablePlannerBackend, PlannerBackendRequest
from agent.runtime.planner import ToolCallingPlanner


def call_model(request: PlannerBackendRequest) -> str:
    # Replace this with provider SDK/API code. Return main-planner XML or a dict.
    return (
        "<decision><kind>response</kind><name>talk</name>"
        "<reasoning>demo</reasoning>"
        "<parameters><message>demo</message></parameters></decision>"
    )


planner = ToolCallingPlanner(CallablePlannerBackend(call_model, provider="local"))
```

OpenAI-compatible provider configuration:

```python
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
)
from agent.backends.provider_config import load_planner_provider_config
from agent.runtime.planner import (
    ToolCallingPlanner,
)

provider_config = load_planner_provider_config()
planner = ToolCallingPlanner(
    OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig.from_provider_config(provider_config)
    )
)
```

`PlannerProviderConfig` is the future GUI-facing configuration model. It exposes
`missing_fields()`, `redacted()`, and `write_env_file()` helpers so a UI can
collect provider/model/base-url/key settings without leaking secrets into logs.
For OpenAI-compatible providers, `list_openai_compatible_models()` can populate
a model selector from `/v1/models`. The provider backend defaults to three total
attempts with exponential backoff starting at 0.5 seconds. Transient transport
failures and HTTP 408/429/500/502/503/504 are retried. An optional fallback
endpoint can be configured with `OPENETA_LLM_FALLBACK_PROVIDER`,
`OPENETA_LLM_FALLBACK_MODEL`, `OPENETA_LLM_FALLBACK_API_BASE`,
`OPENETA_LLM_FALLBACK_API_KEY`, and `OPENETA_LLM_FALLBACK_TIMEOUT_S`. A timeout,
connection failure, HTTP 401/403/408/429, or Cloudflare 520-527 response switches
the next attempt to the other endpoint. Consecutive switch-eligible failures
alternate endpoints, so the default three attempts use
`primary → fallback → primary`. Other retryable 5xx responses stay on the active
endpoint. Configure the retry policy with `OPENETA_LLM_MAX_ATTEMPTS` and
`OPENETA_LLM_RETRY_BACKOFF_S`; exhausted or non-transient failures remain
structured `response::ask_human` results.

The first local GUI entry point is:

```bash
conda run -n openeta python -m agent.gui.provider_config_app
```

It is intentionally small and local-only. It shares `PlannerProviderConfig`,
`OpenAICompatiblePlannerBackend`, and the same planner smoke path used by
`examples/openai_compatible_planner_smoke.py`.

The terminal control-console entry point is:

```bash
uv run openeta
```

This is the closer match to a Codex-style workflow. Users can enter tasks
directly, type `/` to open a keyboard-navigable slash-command popup, and observe
every planner request, safety check, tool call, compact tool-result summary,
and token-usage record in the terminal. The display layer follows the Codex TUI
five-row tool-output bound: it prioritizes failure diagnostics, semantic output
fields, and artifact paths before width-aware truncation. This does not truncate
the persisted `ToolResult`, session trace, memory, or planner context.
`response::ask_human` and dummy world-mutating tool
calls are handled as inline terminal prompts.

Minimal executable tool handler:

```python
from agent.tools.registry import ToolExecutionContext, ToolResult, build_default_tool_registry

tools = build_default_tool_registry()


def sam3_handler(context: ToolExecutionContext) -> ToolResult:
    return ToolResult(
        True,
        content="segmented target",
        details={"prompt": context.parameters["prompt"]},
    )


tools.bind_handler("sam3", sam3_handler)
```

Compatibility example:

```python
from agent.runtime.actions import CommandKind, PipelineStatus
from agent.runtime.interfaces import ActionExecutionResult, ActionInterface


class RealTalkInterface(ActionInterface):
    kind = CommandKind.RESPONSE
    name = "talk"
    description = "Send text to the robot speech system."

    def execute(self, context):
        message = context.request.parameters["message"]
        send_to_tts(message)
        return ActionExecutionResult(
            status=PipelineStatus.EXECUTED,
            content="Speech sent.",
            details={"message": message},
        )
```

Compiled `EnvAction.command.metadata.interface` includes the selected interface
descriptor so downstream logs can tell whether a command subtype is backed by a
real implementation or a placeholder. After the schema rename, `talk` should be
represented as `CommandKind.RESPONSE` with `request.name == "talk"`.

## Status Values

Pipeline calls and whole plans use:

| Status | Meaning |
|---|---|
| `planned` | Structurally valid and ready for a downstream executor. |
| `pending` | Registered but no handler exists yet, or deliberately deferred. |
| `executed` | Handler ran successfully. |
| `blocked` | Skill execution is blocked by safety failure. |
| `failed` | Tool, safety check, or skill lookup failed. |
| `skipped` | Deliberately skipped. Reserved for later use. |

The default tool registry currently has metadata for tools and safety checks but
does not bind real simulator MCP proxy handlers automatically, so many compiled
calls are `pending`. This is intentional: OpenETA can pass structured intent
through the bridge before every backend implementation exists. Tests and local
smoke runs can bind deterministic handlers with `ToolRegistry.bind_handler()`,
and simulator integration runs can bind MCP proxy handlers with
`bind_simulator_mcp_tool_handlers()`.

## Next Implementation Targets

1. Add schema validation to `ToolSpec` and `SkillSpec`.
2. Replace opt-in placeholder pre-checks with real safety checker backends;
   failure/recovery feedback is already wired into CLI planner memory.
3. Extend the validated LIBERO → RGBD → SAM3 → AnyGrasp → control/cleanup smoke
   into a model-planned, long-horizon task with fresh observations and recovery.
4. Implement bounded `CodePolicyInterface` sandbox/dry-run/execution for
   optional atomic-tool backends.
5. Add offline episode reconstruction on top of the persisted JSONL trace.
6. Port selected RAS VLM perception prompts as references, not as a fixed
   planner flow.
