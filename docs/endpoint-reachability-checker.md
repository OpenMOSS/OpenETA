# Endpoint reachability checker

## Scope

`ik_preview_check(target_pose)` is a read-only geometric preflight. It answers
whether the requested world-frame EEF pose has a joint-limit-respecting IK
solution. It does not move the robot, maintain task progress, choose a grasp
candidate, or infer a manipulation stage.

When `target_pose` omits orientation, the checker defaults to the current EEF
orientation. This matches `move_to`'s position-only behavior instead of
claiming that a position is reachable using an arbitrary wrist orientation.
An explicitly unconstrained exploratory query may set
`preserve_current_orientation=false`.

The current real backend supports LIBERO's 7-DoF Panda. Unsupported backends
return `unknown/backend_unsupported`; they do not fall back to the dummy
`feasible=true` handler.

## Layered semantics

- Endpoint kinematics: owned by `ik_preview_check` and always attempted.
- Endpoint collision: optional in `ik_preview_check`; disabled by default until
  intended-contact allowances and scene geometry are calibrated.
- Path feasibility: not checked here and remains owned by
  `obstacle_avoidance`.
- Execution attainment: reported by the later `move_to` environment receipt.

The top-level status is tri-state:

- `reachable`: a candidate satisfies the requested position/orientation
  tolerances and joint limits, plus any explicitly requested available endpoint
  collision check.
- `unreachable`: a completed bounded multi-start search found no feasible
  candidate, or an explicitly requested endpoint check found a collision.
- `unknown`: the backend is unsupported, the solver timed out/failed, or a
  requested collision check was unavailable.

Only `unreachable` is emitted as a failed `ToolResult` and therefore blocks an
opt-in pre-tool checker hook. `unknown` stays visible to the Agent but does not
become a false hard rejection.

## Implementation

The bench worker owns the live simulator model, so it performs IK against an
independent MuJoCo `MjData`. The live environment is never stepped and its
`qpos` is never modified. The solver uses deterministic bounded multi-start
least squares over the seven Panda joints. When a full 6-DoF target fails, it
also checks position and orientation separately so the Agent can distinguish a
coupled-pose conflict from a wholly unreachable position.

Default tolerances and budget:

- maximum per-axis position residual: 2 mm;
- orientation residual: 0.05 rad;
- 24 joint-space starts, at most 300 evaluations each;
- 10 second wall-clock budget.

The compact Agent-visible result retains status, reason code, component
reachability, best residuals, collision/path coverage, solver coverage, and
recovery suggestions. The complete response is also materialized as a JSON
artifact by the normal simulator proxy.

## Provisional result shape

```json
{
  "status": "unreachable",
  "kinematic_status": "unreachable",
  "feasible": false,
  "reason_code": "full_pose_infeasible",
  "position_only_reachable": true,
  "orientation_only_reachable": true,
  "best_candidate": {
    "position_error_m": 0.00087,
    "max_axis_position_error_m": 0.00074,
    "orientation_error_rad": 0.10987,
    "joint_margin_min_rad": 0.397
  },
  "collision": {"checked": false},
  "path": {
    "checked": false,
    "reason": "path feasibility is owned by obstacle_avoidance"
  },
  "solver": {
    "method": "bounded_multistart_least_squares",
    "timed_out": false,
    "formal_infeasibility_proof": false
  },
  "suggestions": [
    "relax_target_orientation",
    "select_another_grasp_candidate"
  ]
}
```

This is a backward-compatible extension of the existing `feasible` output,
not yet the final shared `SafetyVerdict` schema. Promoting or renaming these
fields requires the three-person RFC review defined in the shared architecture
document.

## Canary evidence

On the LIBERO Object task 2 Panda model, the current EEF pose was recognized as
reachable in one evaluation. The previously common left-high coupled target
was classified as `full_pose_infeasible` under the default budget after 26
full/component attempts and 3,625 FK evaluations: position and orientation were each separately reachable,
while the best joint candidate retained 0.1099 rad orientation error at 0.739
mm maximum-axis position error. This matches the independent global/constrained
diagnosis recorded in
[`libero-mink-controller-canary-2026-08-17.md`](libero-mink-controller-canary-2026-08-17.md).
