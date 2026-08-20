# BEHAVIOR-1K interface proposal

Everything below is measured on this machine (2026-08-18), not inferred.
Reproduce with `scripts/setup_behavior.sh --verify` and the probes described at
the end.

## 1. What R1Pro actually is

Booting Isaac Sim headless and reading `robot.controller_action_idx` gives two
different layouts depending on config:

| config | `action_dim` | arm dims | arm controller |
|---|---|---|---|
| R1Pro defaults (no `controller_config`) | **21** | 6 = 3 pos + 3 rot | `InverseKinematicsController` |
| `configs/r1pro_behavior.yaml` | **23** | 7 joint targets | `JointController` |

Slot map for the 21-dim default — verified complete, no gaps, no overlaps:

```
base           0,1,2        HolonomicBaseJointController  (velocity)
trunk          3,4,5,6      JointController               (position)
arm_left       7..12        InverseKinematicsController   (3 pos + 3 rot)
gripper_left   13           MultiFingerGripperController  (1 scalar)
arm_right      14..19       InverseKinematicsController
gripper_right  20           MultiFingerGripperController
```

`arm_names = ['left','right']`, `default_arm = 'left'`,
`eef_link_names = {left: left_eef_link, right: right_eef_link}`.

Two traps worth naming, because both mislead if you read the URDF instead of
the live robot:

- The URDF has **7** movable joints per arm, but the IK controller commands
  **6**. Sizing an arm buffer from joint count overruns into the gripper slot.
- Each gripper has 2 finger joints but **1** command dim.

## 2. Why this needs its own interface, not the Franka one

The existing `move_to(handle, x, y, z)` encodes an assumption that holds for
LIBERO and ManiSkill and breaks here: *one* arm, *fixed* base, so a target pose
fully determines the action. R1Pro violates all three:

1. **Two arms.** `move_to` has no way to say which.
2. **Mobile base.** A target 2 m away is not an arm motion; it is drive-then-reach.
   Reachability is no longer a property of the target alone.
3. **Trunk.** 4 extra DOF that change what the arms can reach, with no slot in
   the Cartesian call.
4. **`action_dim` is config-dependent** (21 vs 23). A codec keyed on the string
   `"behavior"` cannot know which.

Point 4 is already handled correctly by accident of good design:
`_DEFAULT_ACTION_DIMS` in `sim/mcp_server/action_codecs.py` has no `behavior`
entry, and `_declared_behavior_layout` raises unless
`meta["control_spec"]["cartesian_delta"]["supported"]` is declared. The fix is
to *populate* that spec from the live robot, never to hardcode a dim.

## 3. Proposed shape: one codec, two agent-facing profiles

Keep a single worker protocol (the existing 7 HTTP routes on flat action
vectors — `move_to`/`gripper_*`/collision all live server-side on top of
`step`). Split only the **agent-facing** surface, which is what your
"two interfaces with different prompts" instinct is pointing at.

### 3.1 `control_spec`, published by the worker at `create_env`

The worker reads the live robot and publishes ground truth. No hardcoded dims.

```python
# sim/bench_worker.py, inside create_env for backend == "behavior"
robot = env.robots[0]
idx = {k: [int(i) for i in v] for k, v in robot.controller_action_idx.items()}
arm_ctrl = type(robot._controller_config["arm_left"]["name"])  # resolved name

control_spec = {
    "profile": "mobile_bimanual",
    "action_dim": int(robot.action_dim),          # 21 or 23, measured
    "groups": idx,                                # slot map, verbatim
    "arms": list(robot.arm_names),                # ['left','right']
    "default_arm": str(robot.default_arm),        # 'left'
    "base": {"type": "holonomic", "indices": idx["base"],
             "command": "velocity", "dims": ["vx", "vy", "wz"]},
    "trunk": {"indices": idx["trunk"], "command": "position"},
    "cartesian_delta": {
        # supported only when arms are IK-controlled; joint config sets False
        "supported": arm_ctrl == "InverseKinematicsController",
        "per_arm": {
            "left":  {"position_indices": idx["arm_left"][:3],
                      "rotation_indices": idx["arm_left"][3:6],
                      "gripper_index": idx["gripper_left"][0]},
            "right": {"position_indices": idx["arm_right"][:3],
                      "rotation_indices": idx["arm_right"][3:6],
                      "gripper_index": idx["gripper_right"][0]},
        },
        "frame": "eef_delta",
        "pos_scale_m": 0.05,    # metres per normalized 1.0 — needs calibration
        "rot_scale_rad": 0.15,  # ditto
    },
}
```

`pos_scale_m` / `rot_scale_rad` are placeholders. They must be calibrated by
commanding a unit action and measuring actual eef displacement, the same way
the other backends were. Shipping guesses here produces `move_to` calls that
silently undershoot.

### 3.2 Profile A — `mobile_bimanual` (default, for capable agents)

Explicit about the things that actually vary. `arm` is required; there is no
default, because a silent default arm is the kind of bug that looks like bad
policy rather than a wrong actuator.

```python
await move_to(handle, x, y, z, arm="left",           # required
              roll=None, pitch=None, yaw=None,        # optional orientation
              num_steps=120)
await gripper_open(handle, arm="left")
await gripper_close(handle, arm="right")

# base and trunk get first-class calls rather than being smuggled into move_to
await drive_base(handle, vx=0.2, vy=0.0, wz=0.0, num_steps=60)
await set_trunk(handle, targets=[0.1, 0.0, -0.2, 0.0], num_steps=40)
```

Returns gain a reachability field, because with a mobile base "can't reach" is
a normal outcome an agent must be able to act on:

```json
{
  "reached": false,
  "reason": "out_of_reach",
  "eef_error_m": 0.34,
  "hint": "target 0.34m beyond arm envelope; drive_base first",
  "collision": {"detected": false, "world_checked": true, "obstacle_count": 12}
}
```

### 3.3 Profile B — `joint_direct` (for the 23-dim joint config, and for VLA policies)

When arms are `JointController`, Cartesian is unavailable and the codec must
say so rather than fake it. This profile is also the right surface for a
policy that emits raw vectors.

```python
await step_joints(handle, action=[...23 floats...])   # validated against action_dim
await get_joint_state(handle)   # -> {"arm_left": [...7], "trunk": [...4], ...}
```

`move_to` on this profile returns
`ControlCodecError("unsupported_cartesian_control")` — the existing fail-closed
path, unchanged.

### 3.4 Prompt selection

`create_env` returns `control_spec.profile`; the MCP client picks the system
prompt from it. One sentence of difference that matters:

- `mobile_bimanual`: "You control a two-armed mobile robot. Name an arm on
  every reach. If a target is out of reach, drive the base toward it, then
  reach again."
- `joint_direct`: "You emit 23-dim joint vectors. Cartesian targets are not
  available."

## 4. Collision checking

One-line gate change in `sim/mcp_server/server.py:1090`:

```python
if enable_collision_check and backend in ("libero", "maniskill", "behavior"):
```

Two things must be settled before that line is worth adding, and neither is
free:

- **Arm DOF.** `CollisionChecker` assumes 7. R1Pro arms *are* 7 joints, so the
  cuRobo side is fine — but the *command* is 6 (IK), so the checker must be fed
  measured joint state, not the action vector. Feeding it the 6-dim command
  would check a pose the robot is not in.
- **Which arm.** cuRobo checks one kinematic chain at a time. Check the arm
  being commanded; left-vs-right contact falls out as self-collision, which is
  already covered.

Scene geometry: OmniGibson exposes `env.scene.objects` with `.get_position()`
and AABB, so `_capture_internal_objects` is a short comprehension. This is
untested — I have not run it — so treat the effort number below as an estimate,
not a measurement.

## 5. Concurrency: yes, with two caveats

Measured by booting two instances as separate subprocesses, staggered 3 s:

| | result |
|---|---|
| both survived | yes, `action_dim` 21 each |
| VRAM GPU0 | 3301 MB → 8666 MB peak = **~2.7 GB per instance** |
| step rate | 22–40 Hz |
| boot time | 31 s first, 129 s second (bootstrap contention) |

Caveats:

1. **Both instances pinned to GPU0.** GPU1 went 2364 → 2900 MB, i.e. unused.
   Isaac Sim does not spread on its own; set `CUDA_VISIBLE_DEVICES` per worker
   process.
2. **That 2.7 GB is an empty scene.** Real activities load full interactive
   houses from the 37 GB asset set, so per-instance VRAM will be materially
   higher. Do **not** divide 32.6 GB by 2.7 GB and conclude ~12 instances —
   that measurement has not been taken. Measure on a real activity first.

The pooling machinery already exists: `BehaviorProcessPool.acquire_shared`
leases slices via `_find_lease_offset`, and `BehaviorEnv` takes
`total_num_processes` + `group_world_size` and shards through `chunk_step`.

## 6. Open questions for you

1. **Base policy.** Should `move_to` auto-drive the base when a target is out
   of reach, or refuse and let the agent call `drive_base`? I lean refuse:
   auto-driving turns one tool call into an unbounded navigation problem, and a
   failed reach becomes indistinguishable from a failed drive.
2. **Which config is canonical for eval?** The 1016 registered activities pair
   with `r1pro_behavior.yaml` (23-dim joint). If eval runs that config,
   Profile A is unreachable and the Cartesian work is speculative until
   someone rewires the arms to IK.
3. **Trunk exposure.** First-class `set_trunk`, or fold it into a posture
   preset (`tuck`/`reach_high`)? Presets are far easier to prompt against.

Question 2 decides the effort: if eval is joint-space, Profile B alone is a
few hours (validate dim, passthrough, joint-state readback) and the Cartesian
half should be deferred rather than built on spec.

## 7. Reproducing the measurements

```bash
scripts/setup_behavior.sh --verify        # deps + symlink + versions
```

Probes used (each in its own subprocess, because `gm.HEADLESS` is write-once
and `og.shutdown()` hard-exits — write results *before* shutdown):

- action layout / slot map: boot `Scene` + R1Pro, dump
  `robot.controller_action_idx`, check for gaps and overlaps.
- 21-vs-23: same probe with and without `r1pro_behavior.yaml`'s
  `controller_config`, one process each.
- concurrency: N subprocesses staggered 3 s, sample `nvidia-smi` until exit.
