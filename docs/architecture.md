# OpenETA Architecture

OpenETA separates simulator code, agent runtime code, and the protocol between
them.

## Modules

- `sim/`: simulator implementations and RLinf-derived environment code.
- `agent/`: lightweight Python embodied-agent runtime.
- `adapter/`: OpenETA-owned communication contract.

## simulator MCP interface

OpenETA's long-term simulator boundary is MCP-only. The agent runtime does not
call raw simulator APIs or depend on REST endpoints. REST may exist for the
simulator dashboard or debugging, but it is not the agent-runtime contract.

The agent chooses planner-facing tools and high-level parameters. Simulator-side
MCP tools own state-dependent execution details such as IK, safety checks,
controller expansion, action clipping, and backend action vectors. Low-level
simulator state is therefore not required in the agent-facing observation
contract merely to execute control tools.

Current remote simulator MCP mappings used by the agent runtime:

- `create_simulator_env` -> `create_env` then `reset_env`
- `close_simulator_env` -> `close_env`
- `observe` -> `render_env`
- `move_to` -> `move_to`
- `gripper_control(position >= 0.5)` -> `gripper_open`
- `gripper_control(position < 0.5)` -> `gripper_close`

MCP `isError` envelopes preserve their original text so remote failures remain
diagnosable instead of being collapsed into a generic invalid response.

MCP transport may return RGB/depth frames as base64 payloads. Agent-side code
must materialize those images to local artifact files before exposing them to
planner context or memory. Planner/tool parameters should use image refs or
paths, not inline base64 strings.

Simulator MCP camera payloads should expose explicit calibration and camera
axis conventions. The current MuJoCo-backed MetaWorld/LIBERO contract is
`pos + mat`, where `pos` is camera position in world coordinates and `mat` is a
camera-to-world rotation flattened row-major. Its camera frame is
OpenGL-style: `+X` right, `+Y` up, and the camera looks along local `-Z`
(`-col2` in world). Agent-side geometry tools must treat this `pos + mat`
format as row-major/OpenGL by default. 4x4 `camera_to_world` / `pose_mat`
payloads may still be used when their docstring or fields specify the camera
frame convention.

Simulator control tools should accept world-frame targets. Camera-frame grasp
poses must be converted with an agent geometry tool such as
`camera_pose_to_world` before calling `move_to`.

Grasping is a skill composition, not a separate Agent Tool. The same complete
world-frame normalized grasp result from `camera_pose_to_world` is sent to one atomic
`move_to` call without planner-side pose adjustment. Gripper close remains a
separate atomic `gripper_control` call so the planner can observe between
actions.

At the simulator boundary, ranked GraspNet-family candidates default to
translation-only control while retaining the controller's current EEF
orientation. A deployment with a calibrated GraspNet-to-Panda mapping can opt
in to forwarding candidate orientation: EEF x is GraspNet y (closing), EEF y
is GraspNet z (binormal), and EEF z is GraspNet x (approach). Non-candidate
world poses continue to forward their explicit orientation unchanged. This is
an adapter policy and does not change the remote MCP schema.

AnyGrasp candidates use an agent-owned greedy fallback state: MCP supplies
score-ranked camera-frame poses, the agent starts at rank 0, and only a
candidate-linked safety or failure-check rejection advances to the next rank.
Candidate IDs remain attached through frame conversion and downstream checks so
fallback decisions are attributable and cannot silently skip ranks.

GraspGenX candidates share that agent-owned greedy fallback state and preserve
their predictor/gripper provenance through camera-to-world conversion. Only a
candidate-linked safety or failure-check rejection advances either predictor's
ranked queue.

Any smoke test or integration runner that creates a remote MCP environment must
close it in a `finally` block with `close_env`. Leaking remote simulator handles
is treated as a test failure because it consumes resources on the simulator
machine.

`EnvObservation` and `EnvAction` remain useful local protocol/data structures
for tests, logs, and adapter compatibility:

- `EnvObservation`: task text, camera references/frames, optional robot/object
  summaries, metadata.
- `EnvAction`: structured compatibility/logging payload for one agent turn.

`EnvAction.command` now follows the first structured agent-command schema
documented in `docs/agent-action-pipeline.md`. The primary code names are
`CommandRequest` / `CommandKind`. `AgentCommand` is a runtime decision whose
`CommandKind` is either `tool_call` or `response`; older top-level command
kinds are not accepted. `skill_call`, `safe_check`, `code_policy`, and `sense`
are represented as named `tool_call` capabilities; `ask_human`, `talk`, and
task-completion/no-op responses are represented as `response` subtypes. This is
separate from CLI `SlashCommand` inputs such as `/provider` or `/run`, and
separate from simulator-side `EnvAction` payloads.

## tools interface

Tools are the primary agent execution surface. The main loop is closed-loop:
the agent observes the current state, chooses one `tool(parameter)` or skill,
receives the tool result plus updated environment feedback, then chooses the
next step. This avoids treating embodied control as a long open-loop generated
program.

Tool calls carry side-effect metadata:

- read-only sensing/query tools may be batched before the next observation,
- bookkeeping and pure planning helpers may be batched,
- world-mutating actuator/control tools are `AtomAction` tools; they must run
  one at a time and force a fresh observation before the next state-changing
  action.

- sam3
- select_sam3_detection
- anygrasp
- anydexgrasp
- SLAM
- Lower body control policy
- hand pose database

For SAM3 results with multiple candidates, the runtime inserts an explicit
detection-selection obligation. The VLM receives the original image and a
candidate contact sheet, then selects one stable detection id through
`select_sam3_detection`. Runtime gating prevents targeted grasp planning or
physical control from bypassing this step.

For simulator assets that text-prompt SAM3 cannot identify, the read-only
`retrieve_asset_reference` tool may use the configured object memory bank. A
dedicated clean-context VLM compares the original scene with front/side/top
asset views and returns one validated foreground pixel. Host code renders an
audit marker, then memory requires the main planner to copy the exact point into
`sam3`; the handler routes that call to the deployed `segment_points` MCP tool.
This localization sub-agent does not receive arbitrary code execution and does
not replace the downstream SAM3 candidate-selection obligation.

## safety interface and future sub-agent

Safety checks are represented as named `safe_check` capabilities under
`tool_call` in the command pipeline. They can later be promoted to a dedicated
safety sub-agent, but the current repository does not implement a separate
safety process.

- IK preview check
- obstacle avoidance

## foundation skills lab

Foundation skills are editable text guidance documents. They can describe a
recommended sequence of atomic tools and recovery checks for a full task, but
the runtime must not auto-expand a skill into hidden tool calls.
Built-in skill markdown files are stored under `agent/skills/*.md`.

- pick
- place
- push
- pull
- stack

Robot/environment characterization is isolated in `embodiment_explore`.
Calibration output follows the reviewed proposal, evidence, and promotion
contract in `docs/calibration-lifecycle.md`. Normal task skills consume staged
profiles and cannot silently write `agent/calibrations`.

## agent framework

- `agent/runtime/` carries the OpenETA Python runtime. Closed-loop `tool_call`
  is the primary execution mode: the agent selects one atomic tool from the
  current observation, memory, tool references, and skill references, then
  replans after receiving execution feedback.
- Session traces and session-local working memory are stored under
  `.openeta_memory/sessions/<session_id>/`. The tracked `agent/memory/`
  directory is reserved for reviewed, reusable project memory.
- `code_policy` 降级为 `tool_call` 下的可选 atomic tool backend：只用于短
  horizon、可局部验证、内部能 checkpoint/observe 的代码片段，不作为整任务主循环。
- `code_policy` remains optional and must not bypass the simulator MCP boundary
  for real execution. If a future code-policy backend needs simulator effects,
  it should call bounded tools or simulator MCP tools rather than taking raw
  env objects as its primary control surface.
- `agent.tools.sim_mcp` owns the current simulator MCP proxy implementation.
  It binds planner-facing tools to remote MCP tools and normalizes simulator
  tool results into the standard `ToolResult` envelope.

See `docs/code-policy-runtime.md` for the backend/sandbox boundary.


The current bridge is synchronous and single-step at the agent decision level.
This is deliberate: it gives the simulator and agent teams a stable shared
contract before adding richer session management, promoted project-memory
workflows, or online RL.
