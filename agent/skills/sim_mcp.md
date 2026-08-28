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
  - prepare_attachment_probe
  - camera_pose_to_world
  - ik_preview_check
  - move_to
  - gripper_control
---
# Simulator MCP

Use this as simulator-domain guidance, not an executable macro or interface
reference. Live AgentTool contracts exclusively define parameters, returns,
semantic limits, receipts, opaque references, and repair payloads.

## Episode boundary

- Use the environment lifecycle capabilities once at the episode boundaries.
  Continue from the initial observation instead of resetting redundantly, and
  release a planner-created environment when the episode is finished.
- Use registered atomic capabilities for perception and world mutation. Use
  Python for analysis and derived artifacts, not as an alternative robot-control
  path. A missing or incompatible capability is an infrastructure problem to
  report, not a reason to invent a private interface.
- Treat live structured results as the current execution evidence. If the remote
  deployment disagrees with the advertised capability, preserve the physical
  hypothesis and report the adapter or deployment mismatch.

## Geometry and control

- Keep frame semantics explicit and use current matching calibration before
  treating camera-frame geometry as a world reference. Compile normalized grasp
  candidates through the embodiment geometry layer before robot motion.
- Check chosen motion geometry, then reason from the actual execution result and
  fresh observation. Endpoint feasibility alone is not evidence of controller
  convergence or a clear path.
- The Agent owns route geometry. Use short observable edges near contact and
  route around named obstacles when a straight path fails. Do not disable
  safety checks or replay unchanged geometry without new evidence.
- Use fresh visual co-motion evidence before treating an object as attached and
  planning transport around its full extent.

## Operational use

- A remote capability error, model outage, OOM, or incompatible deployment is an
  infrastructure failure. Preserve the current physical hypothesis and report
  the missing capability after bounded configured fallback.
- When execution returns an unknown transport outcome, reconcile the same remote
  environment before any retry or new mutation.
- Do not install heavy simulator or robotics dependencies inside the lightweight
  planner sandbox. They belong in the simulator or dedicated model service.

For explicit robot, controller, sensor, or environment characterization, use
the `embodiment_explore` skill. Normal episodes consume the active validated
profile and do not silently recalibrate it.
