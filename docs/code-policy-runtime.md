# OpenETA Code-Policy Runtime

This document records the boundary for optional bounded Code-as-Policy
execution. OpenETA's main embodied execution path is closed-loop tool calling;
code policy is retained as an optional backend for atomic tools that need
short, locally-verifiable generated snippets.

## Core Boundary

OpenETA should keep two responsibilities separate:

| Layer | Responsibility | Current location |
|---|---|---|
| Agent backend | Generate bounded Code-as-Policy snippets from observation, memory, tool references, skill references, and env API references. | `agent/runtime/backends.py` |
| Code-policy sandbox | Execute or dry-run generated policy code against an explicit OpenETA env facade backed by RLinf environments. | `agent/runtime/sandbox.py` |
| OpenETA env facade | Narrow runtime API exposed to generated policies: construct/own env, normalize `reset`, `step`, `chunk_step`, and `close`. | `agent/runtime/env_facade.py` |
| Code-policy API | The object actually exposed to generated code. It calls the env facade and records a trace. | `agent/runtime/policy_api.py` |
| RLinf-derived env registry | Resolve an `env_type` to an environment class. This is a class resolver, not a full execution sandbox. | `sim/envs/__init__.py` |
| Recording instrumentation | Adapter-owned rollout/video recording; there is no shared wrapper package in the current `sim.envs` tree. | simulator adapter |

The agent backend may use commercial APIs or local models. It should not own
simulator execution. The sandbox should own simulator access and execution
constraints. The default agent still plans in closed-loop tool calls; this
boundary exists for the cases where an atomic tool delegates a bounded local
subroutine to generated code.

## RLinf Sandbox

The intended sandbox root is:

```text
sim/
```

The local RLinf-derived environment registry is:

```text
sim/envs/__init__.py
```

It exposes:

| Symbol | Role |
|---|---|
| `SupportedEnvType` | Enumerates the supported backend families such as `maniskill`, `libero`, `isaaclab`, `robocasa`, and `genesis`. |
| `get_env_cls(env_type, env_cfg=None)` | Maps an env type to the corresponding RLinf environment class. |

This is the right starting point for environment construction. It is not, by
itself, a safe execution interface for bounded generated Code-as-Policy snippets:

- it returns classes, not constructed environments;
- constructor signatures differ across env families;
- action spaces and observation schemas differ across backends;
- code-policy execution still needs a narrow set of allowed operations.

The old `rlinf.envs.wrappers` path is not present in this repository.
`RlinfCodePolicySandbox.wrap_env()` therefore rejects wrapper flags explicitly
instead of importing a nonexistent module or pretending recording is active.
Rollout/video instrumentation should be supplied by the active simulator
adapter.

`RlinfCodePolicySandbox` is currently a reserved interface. It does not yet
construct all simulator envs or execute arbitrary generated Python. Real
execution still needs:

- a narrow env API surface,
- an OpenETA env facade that normalizes `reset`, `step`, `chunk_step`, and
  observation/action conversion,
- timeout and resource limits,
- safety preflight checks,
- dry-run/preview support,
- trajectory and code logging,
- clear error feedback for the next agent turn.

## OpenETA Env Facade

`agent/runtime/env_facade.py` provides the first interface example:

| Class | Role |
|---|---|
| `RlinfEnvSpec` | Declarative request for constructing an RLinf env through `get_env_cls`. |
| `RlinfEnvFacade` | Narrow wrapper around a constructed env with `reset`, `step`, `chunk_step`, `close`, and `descriptor`. |
| `RlinfStepResult` | Normalized Gym/Gymnasium step result container. |

Generated Code-as-Policy snippets should receive a controlled object shaped
like `RlinfEnvFacade`, not a raw RLinf env. This gives OpenETA one place to
enforce timeouts, action validation, observation conversion, dry-run behavior,
and logging.

## Code-Policy Execution Model

RLinf environments are ultimately controlled through `step(action)` or
`chunk_step(actions)`. Code-as-Policy does not replace that actuator boundary or
the outer closed-loop agent turn. Instead, generated code runs inside a sandbox
with a narrow API object and must remain short-horizon:

```python
obs = api.observe()
target = detect_target(obs, "red cube")
action = plan_next_action(obs, target)
step = api.step(action)

if not step.terminated:
    next_action = refine_action(step.observation)
    api.step(next_action)
```

The policy code can perform perception, branching, retries, safety checks, and
limited multi-step composition, but every physical/simulated transition still
goes through the env facade. The snippet must return checkpointed results so the
outer agent can observe feedback before selecting the next state-changing tool.
This gives OpenETA a single place to validate actions, record traces, interrupt
unsafe behavior, and convert results back into agent memory.

`agent/runtime/policy_api.py` currently provides `CodePolicyApi` with:

- `reset(**kwargs)`
- `observe()`
- `step(action, **kwargs)`
- `chunk_step(actions, **kwargs)`
- `descriptor()`

Future higher-level helpers such as `move_arm`, `open_gripper`, `close_gripper`,
or `call_tool` should be implemented on top of this API and eventually compile
down to one or more env actions.

## Vector Environment Compatibility

The extracted `sim/envs` tree is the supported import surface. A limited
compatibility copy of RLinf vector-environment helpers remains under
`sim/rlinf/rlinf/envs/venv`, but new OpenETA code should not import the old
`rlinf.envs` registry. Each simulator family still needs dependency-specific
acceptance testing before its vector execution path can be considered ready.

## Agent Backend

`CodePolicyBackend` is the optional generation boundary. It accepts a
`CodePolicyGenerationRequest` and returns a `CodePolicyGenerationResult`.

`PlaceholderCodePolicyBackend` produces a pending placeholder program. Future
backends can call commercial APIs behind the same interface. This backend is not
used by the default `OpenEtaAgentRuntime` unless a caller explicitly selects
`CodePolicyPlanner` or an atomic tool backend delegates a bounded subroutine to
it.

Pi reference points:

| Pi area | Useful idea for OpenETA |
|---|---|
| `third_party/pi/packages/ai` | Multi-provider model registry, auth resolution, provider-specific streaming/completion adapters. |
| `third_party/pi/packages/agent/src/agent-loop.ts` | Message loop that separates provider calls, tool execution, and turn lifecycle. |
| `third_party/pi/packages/agent/src/harness/agent-harness.ts` | Harness pattern with resources, active tools, event hooks, and session state. |
| `third_party/pi/packages/coding-agent/src/core/agent-session-runtime.ts` | Runtime/session ownership separated from provider logic. |

OpenETA should copy the separation of concerns, not the TypeScript runtime.

## Policy Context

`CodePolicyPlanner` builds `policy_context` with:

- `task`
- `observation`
- `memory`
- `tool_references`
- `skill_references`
- `env_api_reference`
- `safety_constraints`

The generated code may compose env APIs and tools inside a bounded local
subroutine. It should not attempt to solve a whole task in one generated
program. `pick`, `place`, and similar skills are text guidance for choosing
atomic tools, not mandatory OpenETA-authored executor flows.

## Next Steps

1. Define the narrow Python API exposed to bounded generated policies.
2. Validate each dependency-specific `sim.envs` backend and its vector path.
3. Implement a dry-run executor inside `RlinfCodePolicySandbox`.
4. Add a commercial API backend implementation behind `CodePolicyBackend`.
5. Feed sandbox execution errors, checkpoints, and observations back into agent
   memory.
6. Record code, actions, videos, and rewards through RLinf wrappers for RLVR.
