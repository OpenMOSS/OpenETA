---
name: embodiment_explore
description: Guidance for safely characterizing a new robot, controller, sensor setup, or environment and producing evidence-backed session-local profiles and skills.
version: v1
editable: true
task_patterns:
  - calibrate a new robot profile
  - characterize a new embodiment
  - discover controller parameters
  - adapt OpenETA to a new environment
  - explore robot and environment parameters
  - 标定新机型
  - 适配新机器人
  - 探索机型参数
  - 建立具身 profile
allowed_tools:
  - create_simulator_env
  - close_simulator_env
  - observe
  - python_exec
  - sam3
  - select_sam3_detection
  - reject_sam3_detections
  - grasp_pose_estimate
  - anyplace
  - camera_pose_to_world
  - compile_grasp_seed
  - compute_wrist_alignment
  - move_to
  - follow_eef_trajectory
  - gripper_control
  - save_memory
  - get_memory
  - compact_memory
  - propose_calibration_profile
  - promote_calibration_profile
  - register_skill
  - update_skill
---
# Embodiment Profile Exploration

Use this skill only for an explicit adaptation, calibration, or reviewed
self-improvement task. Do not enter exploration during an ordinary benchmark
episode merely because a task attempt failed.

The goal is to produce evidence-backed candidate profiles and reusable
heuristics, not one successful coordinate trajectory. The protocol applies to
simulation and real hardware; the available safety envelope and approval policy
may differ.

## Exploration Protocol

1. Define a profile fingerprint before moving:
   - robot and gripper model;
   - controller and control frequency;
   - tool/TCP and frame conventions;
   - camera models, mounting, and calibration version;
   - simulator/backend or real-hardware version;
   - object/receptacle family and task assumptions.
2. Inventory bound atomic tools and live schemas. Treat unbound capabilities as
   unavailable. Do not modify tool definitions or remote MCP contracts.
3. Establish the deterministic safety envelope. On real hardware, start with
   the most conservative approved limits and require the active supervision
   profile for world mutation.
4. Select one uncertain parameter family and one objective metric. Keep all
   other profile fields fixed. Useful families include:
   - frame and quaternion conventions;
   - TCP/grasp-to-EEF transform;
   - gripper command direction, usable width, and contact behavior;
   - controller response, tolerance, and bounded step size;
   - camera alignment and wrist-servo gain;
   - approach/contact/lift offsets;
   - carry, rim, release, and retreat clearances;
   - timeout reconciliation and settling time.
5. Probe with the smallest observable atomic transition. Observe before and
   after every mutation. Never infer success from tool-call success alone.
6. Record a structured ledger entry for every variant:
   profile fingerprint, parameter value, seed/state, input provenance, action,
   motion receipt, safety/checker verdict, visual evidence, objective reward,
   and failure classification.
7. Distinguish `PASS`, `FAIL`, and `UNKNOWN`. Infrastructure, provider,
   deployment, and resource failures do not score the tested physical
   parameter. Repeated deterministic backend errors must respect the runtime
   circuit breaker.
8. Call `propose_calibration_profile` with the complete candidate profile,
   fingerprint, machine-readable validation gates, rationale, and bounded
   ledger. This performs deterministic checks and independent review, then
   writes only to the current session.
9. Run repeated canaries and held-out episodes using that exact staged profile.
   Preserve its SHA-256 in every result. Infrastructure failures are excluded,
   not counted as physical failures.
10. Call `promote_calibration_profile` with local result references. Candidate
    publication requires profile-hash-linked canary and held-out coverage plus
    the active supervision policy. Validated promotion additionally requires
    every metric gate to pass and a previously published candidate.

Update skills only through the independent skill author/reviewer path. Never
edit a shared skill, calibration, or tool with `python_exec`.

## Promotion Rules

Keep a result as `candidate` until it reproduces across repeated canaries and
held-out states. Promotion requires objective task evidence, no safety or
runtime regression, and an independent review of the profile/skill diff.

Promote scoped facts such as "controller X on robot Y supports this bounded
range", not universal magic numbers. Include applicability and invalidation
conditions so a changed robot, controller, camera calibration, payload family,
or environment version forces revalidation.

## Stop Conditions

Stop exploration and report a structured blocker when:

- the next probe is outside the approved safety envelope;
- required state or provenance cannot be observed;
- a live tool/schema/profile contract is incompatible;
- the same deterministic infrastructure failure exhausts its retry budget;
- evidence remains ambiguous after the allowed observation budget;
- objective improvement cannot be separated from uncontrolled changes.

Normal task skills consume validated profiles. They may record a candidate
lesson after failure, but they must not silently recalibrate the embodiment
inside a benchmark episode.
