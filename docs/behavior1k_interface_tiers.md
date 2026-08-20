# Robot interface tiers

Supersedes §3 ("two profiles") of `behavior1k_interface_proposal.md`. That
section framed this as two parallel interfaces to build for BEHAVIOR, which
understates the design: the split is not per-backend, it is per-morphology, and
it applies to every bench we already have.

The measurements in §1, §5 and §7 of the original doc still stand and are not
repeated here.

## The dispatch key is measured capability, not backend name

A tier is selected from the `control_spec` the worker publishes at
`create_env`, read off the live robot:

| condition | consequence |
|---|---|
| `len(spec["arms"]) > 1` | `arm=` becomes a required parameter |
| `spec["base"]` present | register `drive_base` |
| `spec["trunk"]` present | register `set_trunk` |
| `cartesian_delta.supported == false` | Tier 3; `move_to` fails closed |

Keying on morphology rather than on the string `"behavior"` is not stylistic.
R1Pro's `action_dim` is 21 under its default IK controllers and 23 under
`r1pro_behavior.yaml`'s `JointController` override — the backend name cannot
distinguish them, so a name-keyed codec would drive the wrong actuators. It
also makes RoboCasa's PandaOmron land in Tier 2 automatically, because it has a
base, without anyone adding a special case for it.

## Tier 1 — `eef_only`

Franka-likes: LIBERO, ManiSkill, MetaWorld, and RoboCasa's arm.

```python
move_to(handle, x, y, z, roll=None, pitch=None, yaw=None, num_steps=100)
gripper_open(handle)
gripper_close(handle)
```

No `arm=` parameter: there is one arm, and an optional-with-default arm
argument is how you get a wrong-actuator bug that reads as bad policy.

This is today's surface, unchanged. Keeping it byte-identical matters because
it is the path whose swept-collision and attachment-proxy chain was verified
firing in a live episode; a rewrite would put that behind a new implementation
nobody has watched work.

## Tier 2 — `mobile_manipulation`

R1Pro, PandaOmron. Tier 1 plus the degrees of freedom a fixed-base interface
has nowhere to put.

```python
move_to(handle, x, y, z, arm="left", ...)      # arm now required
drive_base(handle, vx=0.2, vy=0.0, wz=0.0, num_steps=60)
set_trunk(handle, targets=[0.1, 0.0, -0.2, 0.0], num_steps=40)
```

Reach failure becomes a return value rather than an error:

```json
{
  "reached": false,
  "reason": "out_of_reach",
  "eef_error_m": 0.34,
  "hint": "target 0.34m beyond arm envelope; drive_base first",
  "collision": {"detected": false, "world_checked": true, "obstacle_count": 12}
}
```

With a mobile base, "cannot reach from here" is a normal intermediate state the
agent resolves by driving, not an exceptional condition. Raising on it forces
the agent to treat a routine situation as a failure.

Open design question: `move_to` should **not** auto-drive the base when a
target is out of reach. Auto-driving turns one tool call into an unbounded
navigation problem and makes a failed reach indistinguishable from a failed
drive. Refuse, report `out_of_reach`, let the agent call `drive_base`.

## Tier 3 — `joint_direct`

Any robot whose arms are joint-controlled, including R1Pro under
`r1pro_behavior.yaml` (23-dim). Also the right surface for a policy that emits
raw vectors.

```python
step_joints(handle, action=[...])   # validated against measured action_dim
get_joint_state(handle)             # -> {"arm_left": [...], "trunk": [...], ...}
```

`move_to` raises `ControlCodecError("unsupported_cartesian_control")` here.
That path already exists in `sim/mcp_server/action_codecs.py` and needs no
change.

## One implementation, not three

Tiers decide which parameters are required and which tools get registered. They
must not become separate implementations of `move_to`.

`sim/mcp_server/server.py:843` currently holds the only `move_to`, with zero
`backend ==` branches in its body; all backend variation already flows through
three codec calls (`cartesian_scales`, `cartesian_command_frame`,
`make_cartesian_action`). Hanging off that single function are the swept
collision check, the attachment proxy, and the cuRobo world. Forking per tier
duplicates that safety chain, and only one copy is the one that has been
observed working.

Note that no agent-level primitive lives under `sim/envs/` for any bench —
neither `sim/envs/behavior/` nor `sim/envs/libero/` defines `move_to`. The
agent-facing layer has always been server-side and backend-agnostic. Adding
BEHAVIOR support therefore means populating `control_spec`, not writing a
`move_to` inside `sim/envs/behavior/`.

## Resolved: which config, and what already exists

Traced, not guessed. The registry's behavior entry point
(`sim/env_registry.py:307`) builds `BehaviorDirectEnv` from
`sim/envs/behavior/direct_env.py` — **not** the `BehaviorEnv` process-pool class
in `behavior_env.py`, which serves the RLinf path. `direct_env.py` loads
`r1pro_behavior.yaml` and then overrides both arms via
`_configure_agent_cartesian_control`:

```python
"name": "InverseKinematicsController",
"mode": "pose_delta_ori",
"command_output_limits": [[-0.05]*3 + [-0.25]*3, [0.05]*3 + [0.25]*3],
```

So the agent-facing path is **21-dim IK**, and the scale factors are already
chosen and bounded (`_IK_POSITION_SCALE_M = 0.05`,
`_IK_ROTATION_SCALE_RAD = 0.25`) — they are not placeholders, correcting what
the original proposal said. The earlier 23-dim reading came from the raw yaml
before this override.

`BehaviorDirectEnv.openeta_control_spec` already exists and is already consumed
by `sim/bench_worker.py:726`, publishing `cartesian_delta` (with real scales,
`command_frame: robot_base`) and `gripper` (open `1.0`, close `-1.0`). Tier 1
equivalence for BEHAVIOR is therefore **already built**, not pending.

## Status after this change

Implemented and verified against a live R1Pro on a real scene
(`picking_up_trash`, 77 s boot):

1. **Both arms published.** `control_spec.cartesian_delta.per_arm` now carries
   `left` (pos 7-9, rot 10-12, gripper 13) and `right` (pos 14-16, rot 17-19,
   gripper 20). Previously only one arm was exposed, so the other was
   unreachable through MCP on a bimanual robot. Arms whose slot count is not 6
   are omitted rather than declared, so a joint-controlled config fails closed.
2. **`base` and `trunk` declared.** `base` = slots 0-2, holonomic, velocity
   command; `trunk` = slots 3-6, position. Both keyed off
   `controller_action_idx`, so a config lacking them advertises nothing.
3. **`arms` / `arm_required` / `action_dim` / `groups` added** so the tier and
   the `arm=` requirement are derivable without a backend-name lookup.
   Asserted: all 21 slots covered, no overlap, no index outside `action_dim`,
   left and right distinct.
4. **Collision is checked, against R1Pro's own geometry.** `server.py` routes
   behavior into the checker, which loads `r1pro.yml` — generated by
   `scripts/gen_r1pro_curobo.py` from OmniGibson's shipped cuRobo config (18
   active DOF, 143 spheres). Joint angles are permuted into cuRobo's order **by
   name**: R1Pro's 28-value observation interleaves the two arms, so a
   positional slice would compute a confident verdict from the wrong geometry.
   Absent or mismatched `joint_names` fails closed with a reason rather than
   returning a verdict, and the cached permutation is revalidated against the
   names on every call so a reordered vector cannot reuse a stale mapping. If
   `r1pro.yml` is missing the checker reports `available: false` naming the
   generator, rather than falling back to `franka.yml` — a wrong-robot check is
   worse than no check.

   Verified live by `scripts/verify_r1pro_live.py`, which orchestrates both
   interpreters itself (preflight, then the BEHAVIOR dump, then the cuRobo
   check; `--via-mcp` adds a real server on the deployed path). cuRobo's FK for
   `left_eef_link` agrees with the pose OmniGibson reports independently to
   ~1.3e-5 m. Two negative controls keep that from being vacuous: perturbing
   `right_arm_joint1` moves the left eef exactly 0.0 m while `left_arm_joint1`
   moves it 0.24 m (so a left/right swap would be caught), and swapping two
   names changes the mapped vector (so names drive the permutation).

### Two bugs found on the way

**BEHAVIOR envs could not be constructed at all.**
`_configure_agent_cartesian_control` passed `kv` and `joint_range_tolerance` to
`InverseKinematicsController`, neither of which is in its OmniGibson 3.9.1
signature; `create_controller` binds kwargs strictly, so every env raised
`TypeError: got an unexpected keyword argument 'kv'` before the first step.
Present since `1e8a5f4`, not introduced here. Both keys are dropped rather than
remapped — the surviving gains (`pos_damping_ratio`, `vel_kp`) are different
quantities, so substituting a value would change controller behaviour while
appearing to restore it.

**MetaWorld reported the wrong reason.** The generic `not self._available`
return preceded the metaworld-specific branch, making that branch dead code:
metaworld sets `_available = False` in `__init__`, so it claimed "cuRobo not
installed" on a machine where cuRobo works. Now reports
"joint_positions unavailable for MetaWorld".

## Still missing

**cuRobo has no R1Pro model.** `collision.py` hardcodes
`robot_config="franka.yml"` and cuRobo ships only `franka.yml` /
`franka_mobile.yml`. Enabling the check without a correct model would apply
Franka link geometry to R1Pro joint angles — a confident wrong verdict, worse
than an honest skip. So BEHAVIOR `move_to` still performs **no** swept check,
no attachment proxy, and no world check; it now says so instead of returning a
bare `available: false` that reads like "clear".

Closing this needs an R1Pro cuRobo config (URDF plus collision spheres) and
`_arm_dof` / `robot_config` made per-backend rather than Franka constants. When
wiring it, feed the checker **measured joint state**: the arms command 6 dims
over a 7-joint chain, so the action vector describes a pose the robot is not in.

The proposed `drive_base` / `set_trunk` split was **not** built as two tools.
Both slot groups are consumed by one `base_control` call, which emits a single
action covering base and trunk together — they share one action vector, so
splitting them would mean two steps where the robot can do one. The `trunk`
argument defaults to *hold*, not to `0.0`, because a position-mode slot at `0.0`
scales onto the middle of the joint range. See
[`sim-mcp-control-surface.md`](sim-mcp-control-surface.md) for the built surface.
