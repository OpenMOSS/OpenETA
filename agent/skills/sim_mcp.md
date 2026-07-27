---
name: sim_mcp
description: Guidance for using the remote simulator MCP tools in closed-loop episodes.
version: v1
editable: true
task_patterns:
  - simulator
  - simulation
  - sim mcp
  - create environment
  - create a libero environment
  - create a metaworld environment
  - move robot arm
  - move the end effector
  - 机械臂
  - 仿真环境
  - 创建环境
allowed_tools:
  - create_simulator_env
  - close_simulator_env
  - python_exec
  - observe
  - compile_grasp_seed
  - compute_wrist_alignment
  - camera_pose_to_world
  - move_to
  - gripper_control
---
# Simulator MCP

Use this skill as boundary guidance only. Do not treat it as an executable
macro, schema reference, or source of exact MCP parameter names.

The simulator boundary is MCP-only. Do not call REST endpoints or raw simulator
APIs from the agent loop. The planner must choose one atomic `tool_call` at a
time, inspect the result, and then decide the next tool call.

Use `create_simulator_env` as the only environment-creation operation. It owns
the simulator MCP `create_env -> reset_env` sequence, default render resolution,
artifact materialization, and active handle/session synchronization. Do not call
`create_env` through `python_exec`, raw MCP clients, REST, or `code_policy`.

Use `close_simulator_env` as the only environment-cleanup operation. It closes
the currently bound handle and clears runtime state atomically. Do not route
`close_env` through `python_exec` or ask the planner to rediscover the handle.

The MCP server's live catalog, tool docstrings, and input schemas are
authoritative for simulator operations that do not have a stable AgentTool.
Use `python_exec` to inspect `mcp.list_tools()` and read the saved full catalog
with `artifacts.read_json(...)` when exact fields matter. If this skill
conflicts with the MCP catalog/docstring/schema, follow the MCP documentation.
If a simulator MCP call fails, first inspect the saved error response and the
relevant MCP catalog/docstring/schema before changing parameters or retrying.
`remote_capability_missing` means the configured server does not expose the
required MCP tool. It is not a grasp-candidate rejection: do not retry the
same action or advance to another grasp candidate. Stop the workflow until a
compatible simulator MCP deployment is available.

Use `python_exec` for simulator MCP operations that are not exposed as stable
OpenETA atom tools. The coding tool exposes:

When `observe`, `move_to`, or `gripper_control` has a
registered handler, call that stable atom tool directly. Do not route those
operations through `python_exec`; direct atom tools preserve structured
observation, motion, collision, and memory artifacts for the next planner turn.

```python
mcp.list_tools()
mcp.call_tool(name, arguments)
artifacts.materialize_images(payload, bundle_id="optional-id")
artifacts.list_images(limit=20)
artifacts.read_json(path)
artifacts.grep_text(path, pattern, max_matches=20)
```

Set a JSON-serializable `result` variable before the code exits. Use sandboxed
execution by default. `outside_sandbox` is a general-purpose, separately
approved host subprocess and is not needed for configured MCP helpers.

`mcp.call_tool` returns a lightweight reference. It automatically materializes
MCP image payloads into local files and saves full JSON responses under
`response_path`. Do not pass base64 images through planner context. Use returned
paths, response artifacts, or `artifacts.list_images(...)` instead.

The current simulator MCP camera calibration contract for MuJoCo-backed
MetaWorld/LIBERO observations is `pos + mat`:

- `pos`: camera position in world coordinates, metres, not relative to the end
  effector.
- `mat`: camera-to-world rotation flattened row-major:
  `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`.
- Columns are camera-local axes expressed in world coordinates. This payload
  uses OpenGL-style camera axes: `+X` right, `+Y` up, and the camera looks along
  local `-Z`, so world look direction is `-col2`.
- Transform formula: `p_world = mat @ p_cam + pos`.

ManiSkill/SAPIEN camera metadata may use `pos + quat_xyzw` from
`CameraConfig.pose`; follow the live MCP docstring/schema for the exact fields.

If a non-grasp perception tool returns a camera-frame pose, treat it as OpenCV
camera frame unless the result says otherwise and use `camera_pose_to_world` with
the matching current camera calibration. A normalized `grasp_pose_estimate`
candidate is a stricter case: pass its complete camera-frame candidate to
`compile_grasp_seed`, which combines the
camera transform with the staged GraspNet-to-Panda-EEF calibration and, when
applicable, a session-local task-family strategy. Calibration selection is based
on environment/robot identity and is not an object-class allowlist. If no strategy
matches an honestly reported geometry family, compilation preserves the estimator
orientation and approach under the physical gripper limits. Do not send a normalized
grasp pose directly to `camera_pose_to_world` or simulator control tools.

Simulator control tools should accept world-frame targets. If a `move_to`
argument carries `target_pose.frame`, it must be `world`.

For normalized grasps, `grasp_execution` has two condition-bearing control states:
hover at least 0.25 m opposite world-frame `approach_world_xyz` (not fixed world
`+Z`), and binary latched close (`gripper_control position=0`). Alignment, contact,
probe, attachment verdict, and lift are ordered one-shot obligations/evidence gates.
Each edge remains one ordinary `move_to` or `gripper_control`. Compiled/aligned poses are references;
fresh visual feedback may justify a bounded world-frame pose adjustment accepted by
the runtime envelope. Fixed lift/attachment probes and gripper commands remain exact.
A close acknowledgement or numeric openness cannot replace post-probe co-motion
evidence. Close stays latched until binary `position=1`.
A transport timeout requires observation on the same handle before retry.

Classify failures before retrying:

- Transport timeout means unknown world state: observe and reconcile.
- Structured `reached_target=false` is motion evidence and may reject the
  linked candidate.
- Empty mask or no grasp candidates is a perception/planning outcome and may
  use the bounded fallback sequence in the active task skill.
- `model_inference_failed`, OOM, unavailable model, or incompatible deployment
  is infrastructure failure. Let `grasp_pose_estimate` perform structured
  backend fallback; do not invoke or retry a concrete estimator directly.

Never convert infrastructure failure into candidate rejection, and never spend
the episode turn budget repeatedly calling an unchanged failing backend.

When creating a simulator environment, call `create_simulator_env` with an
explicit `env_id`. It defaults camera renders to 512x512 and seed 0, then resets
the new handle and returns `initial_observation`. Use that result as the first
observation frame. Only call `reset_env` explicitly when a new episode reset is
intentionally requested.

If the user asks for local execution result image paths, inspect the latest MCP
tool result or list materialized images:

```python
images = artifacts.list_images(limit=10)
result = {
    "image_count": images["image_count"],
    "latest_image_path": images["latest_image_path"],
    "paths": images["paths"],
}
```

Do not run package installation commands from `python_exec`. If an import fails,
report the missing dependency to the user in natural language or switch to an
already configured dedicated tool/server. Heavy simulator or robotics
dependencies such as LIBERO, robosuite, torch, or OpenCV should live in their
own simulator/tool runtime rather than inside the lightweight planner sandbox.

## Cleanup

When a planner-created simulator environment is no longer needed, release it
with `close_simulator_env`. Remote MCP environments must not be left open waiting
for TTL cleanup.

For explicit robot, controller, sensor, or environment characterization, use
the `embodiment_explore` skill. Normal simulator tasks consume the active
validated profile and must not silently recalibrate it.
