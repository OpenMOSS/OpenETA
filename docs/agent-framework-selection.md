# Agent Framework Selection

This note records the current OpenETA agent-framework decision after comparing
the two local candidates under `third_party/`.

## Decision

OpenETA should use `third_party/RAS_interactivate_planner` as the first agent
framework reference and treat `third_party/pi` as a design reference.

The implementation direction is:

1. Build a lightweight Python agent runtime in `agent/`.
2. Reuse the RAS planner's robotics-oriented concepts: visual observations,
   single-step planning, action categories, executor feedback, humanoid/arm
   task split, and VLM-friendly prompts.
3. Borrow selected Pi ideas without adopting the Pi runtime wholesale: typed
   tool schemas, event logs, skill packaging, session state, and streaming-style
   lifecycle events.
4. Keep Codex only as a legacy/reference submodule for now. It is no longer the
   primary agent substrate.

## Candidate Comparison

| Dimension | RAS interactive planner | Pi agent harness |
|---|---|---|
| Language | Python | TypeScript / Node >= 22.19 |
| Fit with OpenETA/RLinf | High. OpenETA adapter and RLinf env code are Python. | Medium to low. Requires a Python/Node boundary. |
| Robotics semantics | Already has humanoid and arm planners, VLM observation loops, executors, and robot actions. | General coding-agent harness with no embodied robotics semantics. |
| Action model | Already has `talk`, `tool`, `act`, and `sense` action categories. | Tool calls are generic and would need an embodied schema from scratch. |
| Vision support | Direct VLM image planning is already in the planner. | Supports images, but through a coding-agent session model. |
| Runtime maturity | Application prototype; needs refactoring. | Mature agent runtime with tools, sessions, events, and skills. |
| Weight | Small Python codebase. | Large TypeScript monorepo with CLI/TUI/coding-agent layers. |
| Integration cost | Low to medium. | High. |

## Why Not Use Pi Directly

Pi is useful, but it is still too large for the current OpenETA agent layer. Its
core strengths are tool calling, state management, event streaming, sessions,
and skills. Those ideas are valuable, but the full runtime is oriented around a
coding-agent CLI rather than embodied simulator control.

Adopting Pi directly would add:

- a Node/TypeScript runtime beside the Python simulator stack,
- an additional packaging and build system,
- generic coding-agent concepts that still need robotics-specific adaptation,
- tool and session machinery that may be more complex than the first OpenETA
  bridge needs.

## Why Use RAS as the First Reference

RAS is closer to the target robotics loop:

```text
observation image + task + history
  -> VLM planner
  -> one high-level action
  -> executor
  -> new observation / feedback
  -> next planning turn
```

That shape matches OpenETA's intended closed-loop embodied tool-calling loop
better than a general coding CLI.

However, RAS should not be imported as-is. It currently has several prototype
assumptions:

- action schemas are embedded in prompt templates,
- humanoid and arm planners duplicate runtime logic,
- memory is simple conversation/execution-history lists,
- half-open-loop execution assumes success,
- real hardware, camera, and Docker paths are embedded in executor code,
- tools and skills are not first-class registries.

OpenETA should extract the useful architecture while replacing those assumptions
with a small, explicit runtime.

## Target OpenETA Agent Runtime

The first runtime should provide these pieces:

1. `AgentMemory`: session metadata, observations, actions, tool results,
   simulator feedback, and compact planning context.
2. `ToolRegistry`: first-class definitions for tools such as `sam3`,
   `scene_detector`, `anygrasp`, `anydexgrasp`, `move_to`,
   `gripper_control`, hand-pose database, IK preview check, and obstacle
   avoidance. Tools are stable atomic capabilities.
3. `SkillRegistry`: editable text-guidance documents such as pick, place,
   push, pull, and stack. Skills may recommend tool sequences, but they are not
   executable macros.
4. `Planner` + `PlannerBackend`: one-step planning interface that converts an
   `EnvObservation` and memory context into a structured decision, with a
   backend boundary for future LLM/VLM providers.
5. `AgentRuntime`: owns memory, tools, skills, planner, and conversion into
   `EnvAction`.
6. `AgentAdapter`: adapts the runtime to the existing OpenETA simulator-agent
   bridge.

The first implementation now supports deterministic fixtures, callable
SDK/API wrappers, and a placeholder LLM/VLM backend boundary. A real VLM
planner can replace the placeholder backend without changing the bridge
contract.

## Near-Term Migration Path

1. Keep the existing dummy loop working.
2. Add a lightweight OpenETA agent runtime under `agent/`.
3. Add a new adapter that wraps this runtime and produces `EnvAction`.
4. Keep tools as registered executable metadata, keep skills as registered text
   guidance, and bind `ToolExecutionContext -> ToolResult` handlers as
   simulator-backed implementations become available.
5. Replace half-open-loop success assumptions with simulator `StepResult`
   feedback and explicit safety-check results.
6. Port selected RAS planner prompts into an OpenETA VLM planner backend.
