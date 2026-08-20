# Simulator MCP control surface

What `sim/mcp_server` exposes to an agent today, and what each control actually
does to the robot. Signatures are transcribed from `sim/mcp_server/server.py`;
the semantics notes are the parts that are not visible from a signature and have
already cost debugging time.

Complements [`behavior1k_interface_tiers.md`](behavior1k_interface_tiers.md),
which is a design document about how the surface *should* be structured per
morphology. This one is descriptive: it records the surface as built.

Backends: `libero`, `metaworld`, `maniskill`, `robocasa`, `behavior` (R1Pro),
`dummy`. Every tool is registered by the `_blocking_tool` decorator, runs over
`/mcp` or `/sse`, and takes an optional `session_id` (defaults to the calling
SSE session).

---

## 1. Tools

17 tools, grouped by what they are for. `handle` is the env id returned by
`create_env`.

### Discovery and lifecycle

| tool | arguments |
|---|---|
| `list_available_benches` | — |
| `hot_activate` | `bench` |
| `list_envs` | `env_type=""` |
| `search_envs` | `query` |
| `create_env` | `env_id`, `render_mode="rgb_array"`, `seed=0`, `task=""`, `image_width=None`, `image_height=None`, `include_objects=False`, `robot=""` |
| `reset_env` | `handle`, `seed=None` |
| `close_env` | `handle` |
| `list_active_envs` | — |

### Observation

| tool | arguments |
|---|---|
| `observe_env` | `handle` |
| `render_env` | `handle` |

`observe_env` does not step. `render_env` forces a fresh frame.

### Control

| tool | arguments |
|---|---|
| `step_env` | `handle`, `action=None`, `num_steps=1` |
| `move_to` | `handle`, `x`, `y`, `z`, `roll=None`, `pitch=None`, `yaw=None`, `num_steps=100`, `tolerance=0.002`, `ori_tolerance=0.05`, `enable_collision_check=True` |
| `follow_eef_trajectory` | `handle`, `trajectory`, `num_steps_per_waypoint=60`, `tolerance=0.002`, `ori_tolerance=0.05`, `enable_collision_check=True` |
| `gripper_open` / `gripper_close` | `handle` |
| `base_control` | `handle`, `forward=0.0`, `lateral=0.0`, `yaw=0.0`, `torso=0.0`, `trunk=None`, `command=""`, `num_steps=10` |
| `ik_preview_check` | `handle`, `x`, `y`, `z`, `roll=None`, `pitch=None`, `yaw=None`, `position_tolerance_m=0.002`, `orientation_tolerance_rad=0.05`, `max_attempts=24`, `max_nfev_per_attempt=300`, `timeout_s=10.0`, `preserve_current_orientation=True`, `check_endpoint_collision=False`, `include_scene_objects=False` |

`ik_preview_check` answers "is this pose reachable" without moving anything.

---

## 2. Two command types, and why the defaults differ

The single most load-bearing distinction on this surface. Controllers come in
two kinds, and **omitting an argument means opposite things** for each:

| | commands | omitted means | stops when |
|---|---|---|---|
| velocity | base `forward` / `lateral` / `yaw` | no motion on that axis | commands stop |
| position | trunk, arm joints, gripper | **not** "no motion" | never — it is a target |

A velocity command of `0.0` is genuinely neutral. A position command of `0.0` is
not: `Controller._preprocess_command` (OmniGibson `controller_base.py:304`)
scales the `[-1, 1]` input onto the joint's limits, so

```
command = 0.0  ->  (lower + upper) / 2
```

For R1Pro's `torso_joint1` (limits `-1.1345 .. 1.8326`) that midpoint is
**0.349 rad**. And because these controllers run with
`use_delta_commands: false`, it is an absolute target, not an increment.

The practical consequence, which was a real bug: any action assembled as
`[0.0] * action_dim` silently commands the trunk to a mid-range pose. That is
what `make_cartesian_action` produces before the arm slots are filled, so every
`move_to` was dragging the torso. Holding a position-mode joint requires
re-normalising its *current* angle through the inverse of that scaling — which
is why `control_spec.trunk` publishes joint limits.

**If you assemble raw action vectors for `step_env`, this is your problem too.**
`[0.0] * 21` is not "do nothing".

---

## 3. `base_control`

```python
base_control(
    handle,
    forward=0.0,      # base: normalized forward velocity  [-1, 1]
    lateral=0.0,      # base: normalized lateral velocity
    yaw=0.0,          # base: normalized CCW angular velocity
    trunk=None,       # trunk: position target; None = hold current pose
    torso=0.0,        # RoboCasa's original spelling, kept for compatibility
    command="",       # named base command, replaces forward/lateral/yaw
    num_steps=10,
)
```

One call emits **one action** covering base and trunk together, executed for
`num_steps`. They are not separate calls and cannot be sequenced within a call.

Per §2, `trunk` defaults to `None` rather than `0.0`, meaning *hold*: the trunk
pose is read back from the observation and re-normalised. Passing `trunk`
explicitly commands it.

```python
base_control(h, forward=1.0)                     # drive; trunk stays put
base_control(h, trunk=[0.2, 0.0, -0.1, 0.0])     # 4 explicit torso joints
base_control(h, forward=0.5, trunk=0.3)          # both at once; scalar broadcasts
base_control(h, command="forward", trunk=0.3)
```

A scalar broadcasts to every trunk joint. A list must match the joint count
exactly — a wrong length is an `invalid_trunk_command` error naming the expected
joints, never padded or truncated.

Named commands (`command=`) set only the three base velocities and leave the
trunk alone, so `command="stop"` means "base stops, trunk holds", not "zero
everything":

| | `forward` | `lateral` | `yaw` |
|---|---|---|---|
| `forward` | +1 | 0 | 0 |
| `backward` / `back` | −1 | 0 | 0 |
| `left` | 0 | +1 | 0 |
| `right` | 0 | −1 | 0 |
| `turn_left` | 0 | 0 | +1 |
| `turn_right` | 0 | 0 | −1 |
| `stop` | 0 | 0 | 0 |

Per backend:

| | base | trunk |
|---|---|---|
| RoboCasa PandaOmron | slots 7-9, velocity | slot 10, 1 dim |
| BEHAVIOR R1Pro | slots 0-2, holonomic velocity | slots 3-6, 4-joint chain |

R1Pro slots come from the declared `control_spec`, not from constants: its
`action_dim` is 21 under our IK overrides but 23 under the raw
`r1pro_behavior.yaml` joint controllers, so hard-coded indices would drive the
wrong actuators in one of the two.

Any other backend returns `unsupported_base_control` rather than silently
becoming a no-op. Same for a BEHAVIOR robot that declares no base.

The response reports what was actually sent:

```python
{"forward": 1.0, "lateral": 0.0, "yaw": 0.0,
 "command_type": "velocity",
 "num_steps": 10,
 "trunk": {"mode": "held", "values": [-0.234, 0.049, -0.083, 0.0]}}
```

`trunk.mode` is `"commanded"`, `"held"`, or `"unknown"`. The last one means the
pose could not be resolved, so the slots kept their mid-range default and the
torso may move — surfaced with a `reason` instead of being sent silently.

Not available: a single-scalar coordinated trunk *height*. `trunk=0.3`
broadcasts one normalized value to 4 joints, which is not "raise by 0.3 m" — a
4-DOF serial chain leans and bends. A true height command needs FK derivation.

---

## 4. `move_to` and `follow_eef_trajectory`

Absolute **world-frame** EEF pose, driven closed-loop. Not a delta.

```python
move_to(handle, x, y, z, roll=None, pitch=None, yaw=None,
        num_steps=100, tolerance=0.002, ori_tolerance=0.05,
        enable_collision_check=True)
```

Orientation is opt-in as a set: all three of `roll`/`pitch`/`yaw` or none.
Position-only motion preserves the current orientation. MetaWorld rejects
orientation outright (4-dim action, no rotation).

Internally this is a loop of small normalized deltas, so the per-backend scale
matters — these are controller command scales, not observed displacement:

| backend | position (m at 1.0) | rotation (rad at 1.0) |
|---|---|---|
| BEHAVIOR R1Pro | 0.05 | 0.25 |
| LIBERO | 0.05 | 0.5 |
| RoboCasa | 0.05 | 0.05 |
| MetaWorld | 0.005 | 0.05 |
| ManiSkill | 0.003 | 0.05 |

Two frame conversions happen inside, both driven by
`control_spec.cartesian_delta.command_frame`: RoboCasa and R1Pro consume deltas
in the **robot base frame**, so the world-frame error vector is rotated before
encoding. This needs `base_pose.quat_xyzw` in the observation; without it the
call fails with `missing_base_pose` rather than sending an unrotated vector.

`follow_eef_trajectory` runs 1-5 world-frame waypoints sequentially, each
`{"xyz": [...], "euler_xyz_deg": [...]}` with `frame: "world"`.

Returns `target` / `start` / `end` / `steps_executed` / `terminated` / `reward`,
plus a `collision` block. A skipped collision check reports its `reason`; treat
`world_checked: false` as "never looked", not as "clear".

Gripper state is latched: after an explicit `gripper_open` / `gripper_close`,
that ±1.0 is re-sent on every subsequent motion substep so the fingers keep
their grip instead of relaxing. Before the first gripper call, the slot is left
untouched.

---

## 5. Observations

`observe_env` and every stepping tool return the same shape. The `robot` block
for R1Pro:

| field | notes |
|---|---|
| `joint_positions` | 28 values |
| `joint_names` | **same order** as `joint_positions` |
| `joint_velocities` | |
| `base_pose` | `{"xyz": [...], "quat_xyzw": [...]}` — note the spelling |
| `ee_pose` | primary arm, `[x,y,z, qx,qy,qz,qw]` |
| `gripper_open` | fraction in `[0, 1]` |
| `metadata.arms` | per-arm `ee_pose` + `gripper_joint_positions` |

**Select joints by name, never by slicing.** R1Pro's vector is `base(0-5)`,
`torso(6-9)`, arms **interleaved** left/right `(10-23)`, `fingers(24-27)` — so
`left_arm_joint2` is at index 12, not 11. A positional slice puts base DOF into
torso slots and right-arm angles into left-arm slots, and produces a confident
wrong answer rather than an error. `joint_names` exists precisely so consumers
can permute; both the collision checker and the trunk-hold path use it, and both
fail closed when it is absent.

Quaternions are `xyzw` throughout. Poses are world-frame unless a field says
otherwise.

---

## 6. `control_spec`

Published per env at `create_env`, read off the live robot. This is how a client
discovers layout instead of assuming it.

```python
{
  "schema_version": "openeta.sim_control.v1",
  "profile": "mobile_manipulation",
  "arms": ["left", "right"], "default_arm": "left", "arm_required": true,
  "action_dim": 21,
  "cartesian_delta": {
    "supported": true, "position_indices": [7,8,9], "rotation_indices": [10,11,12],
    "command_frame": "robot_base",
    "position_scale_m": 0.05, "rotation_scale_rad": 0.25,
    "per_arm": {"left": {...}, "right": {...}},
  },
  "gripper": {"supported": true, "indices": [13], "open_value": 1.0, "close_value": -1.0},
  "base":  {"supported": true, "indices": [0,1,2], "command": "velocity",
            "dims": ["vx","vy","wz"], "type": "holonomic"},
  "trunk": {"supported": true, "indices": [3,4,5,6], "command": "position",
            "joint_names": ["torso_joint1", ...],
            "limits_lower": [...], "limits_upper": [...]},
  "groups": {...},
}
```

`trunk`'s names and limits are what make "hold position" expressible: without
limits the current angle cannot be re-normalised. They are published once here
rather than fetched per substep, since they are constant for the robot's life.

A robot missing any one of names/limits/indices publishes **none** of them while
still advertising `supported: true`. A half mapping is worse than none — it
would let a caller believe it can hold position while writing the wrong slot.

Absent sections are meaningful: no `base` means fixed-base, and
`cartesian_delta.supported: false` means `move_to` fails closed rather than
guessing that XYZ lives in slots 0-2.

---

## 7. Failure modes

Everything here fails closed rather than guessing. Codes worth handling:

| code | meaning |
|---|---|
| `unsupported_cartesian_control` | no declared IK layout; `move_to` refuses |
| `unsupported_base_control` | backend or robot has no mobile base |
| `invalid_trunk_command` | trunk value count ≠ joint count |
| `missing_base_pose` | base-frame conversion needs `base_pose.quat_xyzw` |
| `unknown_action_layout` | no declared `action_dim` |

Collision results carry `world_checked` / `self_checked` plus a `reason` when
skipped. A bare `available: false` reads like "clear" and is not enough to act
on — check the flags.

---

## 8. Known gaps

- **Coordinated trunk height.** No single-scalar "raise the torso" (§3).
- **`drive_base` / `set_trunk`.** The tier design in
  `behavior1k_interface_tiers.md` proposes these as separate morphology-gated
  tools; today the functionality lives in `base_control`.
- **Swept collision checking on BEHAVIOR.** `move_to` checks endpoints, not the
  path between them.
- **Trunk displacement under the old code was never measured.** The mechanism in
  §2 is established from the controller source and config, but the earlier drive
  test recorded only EEF and finger joints, so it could not have caught this
  either way. To confirm, log `torso_joint1..4` across a `move_to`.
