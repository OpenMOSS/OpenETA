# Grasp-quality canary record (2026-08-25)

## Scope

These experiments target the remaining contact-quality bottleneck on LIBERO
object task 7 seed 7 (`pick up the milk and place it in the basket`). The changes
do not add host-owned task phases or a fixed action script. The Agent still
selects candidates, strategies, viewpoints, probes, waypoints, and recovery
actions. The host only preserves evidence, resolves short identifiers, validates
freshness/safety, and reports execution receipts.

## Implemented evidence flow

- Grasp estimators expose compatible experimental strategies as explicit options;
  no option is automatically activated.
- An explicitly compiled strategy remains a named contact branch across viewpoint
  changes. A later wrist estimate is separate raw evidence and cannot silently
  inherit or replace that strategy.
- `prepare_attachment_probe` is available for ordinary portable objects. A close
  receipt remains tentative until a short Agent-chosen lift and independent
  `assess_attachment_probe` evidence support attachment.
- Successful pre-grasp `camera_pose_to_world` results are retained as a compact
  placement-world reference. A failed carrying motion blocks immediate release
  and returns the actual EEF/collision evidence; a later successful carrying
  motion clears that causal check.
- Strategy options may project bounded experimental provenance so the Agent can
  compare prior attachment evidence with a raw visual Advisor recommendation.

## Same-scene evidence

| Run | Model | Selected contact branch | Result |
| --- | --- | --- | --- |
| deterministic r21 | scripted geometry canary, not an Agent rollout | world-vertical top-down at the simulator-observed object centre | 15 checks passed; milk lifted 0.1059 m; EEF-object distance 0.0281 m |
| r22 | `gpt-5.6-sol` | raw wrist branch first, then Agent-explicit `top-down-vertical-panda-p8` | raw branch collapsed to empty close; top-down branch closed at 0.6750, passed attachment assessment, and transported the milk across the table; placement reference loss and unsafe release prevented completion |
| r23 | `gpt-5.6-sol` | initial top-down choice was replaced by a high-confidence wrist raw branch | raw close 0.4921 collapsed during a short probe and tipped/moved the carton; late top-down recovery no longer matched the changed posture |
| r24 | `gpt-5.6-sol` | Advisor-selected raw broad-side branch | contact endpoint error 0.0009 m; close openness 0.1593; 5 cm probe returned `aperture_collapsed_to_empty_close`; provider later failed after recovery began |
| r25/r26 | `gpt-5.6-sol` | no useful strategy comparison completed | provider HTTP failure before or immediately after the first perception call |
| r27 | `gpt-5.6-luna` diagnostic | Agent explicitly selected `top-down-vertical-panda-p8` after comparing the raw Advisor result with projected canary evidence | strategy choice and continuity worked; clearance reached with 0.0013 m error; rollout stopped on provider HTTP 403 insufficient quota before contact |

## Physical pick/place isolation canary

`scripts/mink_pick_place_canary.py` is explicitly diagnostic-only and is not
registered as an Agent tool or imported into the runtime. It uses privileged
simulator object coordinates to isolate embodiment physics from perception and
planning. The route semantics are therefore not a task policy.

Three independent task7/seed7 repetitions completed:

| Run | Contact | Carry | Release | Official result |
| --- | --- | --- | --- | --- |
| physical r1 | top-down, close openness 0.6616 | lift plus raised transit and receptacle-centre waypoint; all collision receipts clear | stationary raised release over open basket | reward 1, terminated; target/basket XY distance 4.7 mm |
| physical r2 | top-down | same geometric family | stationary raised release | reward 1, terminated; XY distance 7.4 mm |
| physical r3 | top-down | same geometric family | stationary raised release | reward 1, terminated; XY distance 9.8 mm |

This 3/3 result proves that Mink control, continuous gripper closure, carried-object
collision geometry, open-basket containment, and the official LIBERO predicate can
complete the task. It does not prove Agent completion. It also provides a useful
placement alternative: after collision-checked arrival above a visibly open
container, a bounded stationary drop may be safer than descending a carried-object
proxy through a rim. The Agent must decide from current geometry; the host does not
assign this option a task phase or fixed height.

The r22/r24 comparison shows that non-empty close aperture and visually deep 2-D
overlap are only hypotheses. They do not substitute for a short attachment probe.
It also shows that the raw Advisor preference can be systematically wrong for this
upright carton even when endpoint control is accurate. The evidence-bearing option
in r27 changed the Agent's choice without host auto-selection.

## New-provider canary

Runs r29-r32 used the replacement OpenAI-compatible primary model
`deepseek-v4-flash-vision-exp`. The primary endpoint remained stable throughout
r32: 141 recorded model/subagent calls, no provider retry error, and no failover.
The model supports the current multimodal request shape and the cache-stable prompt
layout. Main calls averaged about 69.5k prompt tokens; the provider reported about
19.5k cached tokens per main call.

Two compatibility defects were repaired before the final run:

- AnyPlace result materialization now happens before the placement bundle refresh,
  and cross-camera rebasing prefers the durable fixed-camera provenance event. This
  keeps a successful AnyPlace result resolvable and prevents unrelated actions from
  recursively changing the bundle identity.
- The previous 2,048-token reasoning-subagent cap truncated or invalidated results
  from reasoning-capable models. Attachment review, grasp Advisor, and reference
  localization now have an 8,192-token ceiling; VDM has 4,096 tokens. The main
  planner ceiling is 16,384 tokens. These are output ceilings, not required output
  lengths.

The final r32 Agent rollout did not complete the task (official reward remained
zero). It nevertheless exercised the complete perception and contact-recovery
flow without provider failure:

- It segmented both task objects, materialized AnyPlace candidates, tried scene and
  wrist AnyGrasp estimates, explicitly selected the experimental top-down strategy,
  and autonomously changed wrist viewpoint after repeated bad contacts.
- Five close attempts produced three tentative non-empty apertures (0.3528, 0.6735,
  and 0.6607) and two clear empty closes (0.0125). None produced valid visual
  attachment evidence. The 0.6735 contact remained open during its short lift but
  the independent reviewer correctly returned FAIL because the carton remained on
  the floor. The 0.6607 contact collapsed to an empty close during its probe.
- The model consumed many turns repairing contact and never began a carrying or
  placement path. It also produced 12 invalid or contract-rejected planner outputs,
  most commonly an extra `openeta_action` wrapper instead of the requested flat
  decision schema; corrective retries recovered, but reduced efficiency compared
  with `gpt-5.6-sol`.
- Late in the run the simulator returned `executing action in terminated episode`,
  while the projected environment receipt still implied `terminated=false`. The
  model inspected the full artifact and reasonably treated the contradiction as a
  transient controller failure. Subsequent recovery reached a Mink
  `constraint_escape_preview_rejected` boundary. The rollout was stopped after 102
  transitions because further recovery turns could not establish useful evidence.

The canary therefore establishes that the replacement model can operate the
harness, interpret rich tool feedback, and perform autonomous recovery, but this
run does not establish pick/place completion. The primary remaining physical
bottleneck is contact quality; two secondary efficiency/robustness issues are flat
decision-schema adherence and authoritative propagation of simulator termination.

Success criteria used for the subsequent full run were:

1. the Agent explicitly compares and selects a contact branch;
2. close plus the ordinary short attachment probe returns PASS;
3. the pre-grasp placement world reference remains visible after transport;
4. carry and descent use Agent-chosen collision-checked waypoints;
5. no release follows a failed attached motion without successful recovery;
6. the official LIBERO reward/termination reports task success.

## Luna recovery and first Agent-driven success

Runs r33-r38 returned to `gpt-5.6-luna` for lower-cost harness iteration. Sol
was not part of the normal or fallback provider chain.

Three harness defects were isolated before the successful run:

- `reset_env` claimed to reopen the gripper but removed the actuator latch and
  left subsequent motion at neutral command `0`. In LIBERO this allowed the
  fingers to drift closed before an Agent-requested close. Reset now establishes
  and preserves the binary OPEN command (`-1`) until an explicit close. A live
  reset/move probe measured openness `0.8514 -> 0.9533`, with the test motion
  reaching its target at `0.00039 m` residual.
- Finger-only close authorization used a `0.005 m` position envelope despite
  centimetre-scale controller accuracy. The envelope is now `0.01 m`; freshness,
  bundle/receipt identity, orientation (`0.30 rad`), and the independent
  attachment probe remain mandatory. A live failed grasp was still rejected when
  its aperture collapsed during the probe, so the relaxation did not create false
  transport evidence.
- The top-down strategy could overwrite an estimator candidate whose native
  approach pointed upward or sideways. It now requires native downward alignment
  of at least `0.5`; rejected candidates produce structured feedback and leave the
  other estimator candidates available for Agent choice. Receptacle collision
  feedback also exposes the signed XY correction into the feasible carried-object
  centre corridor instead of only saying that the object is outside the basket.

The final run
`grasp-quality-full-milk-20260825-luna-r38-open-latch` completed task 7 seed 7
without human or guidance intervention:

1. Luna recovered an initial empty SAM3 result with point mode, compared the raw
   Advisor evidence with the experimental top-down option, and prepared both grasp
   and placement evidence.
2. Its first close was non-empty (`0.4660`) but collapsed during the short lift;
   the attachment proxy retired it. Luna reopened, acquired a near-field wrist
   view, tried GraspGenX/AnyGrasp alternatives, handled two full-pose IK failures,
   and explicitly compiled another top-down wrist branch.
3. The second close measured `0.6710`; the Agent-chosen lift and independent
   attachment assessment returned `PASS`.
4. Luna chose collision-checked carry waypoints at a raised transit and then over
   the basket. The final raised endpoint was reached with `0.0018 m` position
   error. Fresh dual-view evidence showed the carton aligned inside the visibly
   open basket, so Luna explicitly opened the gripper.
5. The same environment execution returned official reward `1`,
   `terminated=true`, and evaluation status `success`.

The episode used 67 Agent turns and 66 tool calls. One 180-second primary Luna
request timed out and the configured Luna fallback completed that turn; there was
no Sol call. The evaluator reported objective success `1/1`, no retry, and clean
environment cleanup. This is the first Agent-driven end-to-end success in this
record, distinct from the privileged diagnostic-only physical canaries above.

## Agent-safe simulator feedback projection (2026-08-26)

Run `grasp-quality-full-milk-20260826-luna-agent-safe-feedback-r39` tested a
strict privilege boundary around simulator safety feedback. The simulator and
host were still allowed to use private object geometry for collision checks and
the evaluator still consumed official LIBERO reward. Before a simulator response
entered `ToolResult`, planner memory/context, or a `python_exec`-readable response
artifact, the adapter removed object instance/geom names, object dimensions and
relative coordinates, exact clearance distances, world-object counts, and raw
contact-authorization payloads. It retained collision verdict/scope, controller
outcome, requested/actual EEF pose, gripper proprioception, and visual recovery
instructions.

The Luna-only rollout completed task 7 seed 7 with official reward `1`:

1. The first non-empty close lost aperture during its short lift, so the Agent
   reopened and reacquired wrist evidence instead of treating the host proxy as
   attachment proof.
2. After rejecting misaligned and full-pose-infeasible candidates, the second
   close retained `0.6743` openness and the independent dual-view attachment
   assessment returned `PASS`.
3. During carry/place, one motion was rejected by the host-private collision
   checker. The Agent received no simulator geometry identity or exact clearance;
   it used fresh images plus actual EEF feedback, previewed a replacement endpoint,
   reached approximately `[-0.025, 0.275, 0.32]`, and opened the gripper.
4. The environment then returned official reward `1` and terminated successfully.

The episode used 72 Agent turns and 71 executed tool calls. There were 88
successful Luna provider requests (72 planner decisions and 16 VDM calls), no
failed provider attempt, no fallback, no Sol call, no human/guidance intervention,
and clean environment cleanup. Total recorded provider token usage was 4,453,366,
so this run validates the privilege boundary but also reinforces that context cost
remains a separate optimization target.

A structural leak audit covered 38 Agent-readable simulator response artifacts,
88 model-call records, 140 tool-call event records, 145 conversation records, and
five working-memory files. It found zero occurrences of the forbidden private
keys/values, including simulator object instance names, MuJoCo geom names, exact
minimum-distance fields, object-relative coordinates/dimensions, and world-object
counts. Observations exposed `object_count=0` and `objects=[]`; attachment remained
an independent visual verdict rather than a simulator truth label.

The shared RFC currently asks collision feedback to name obstacles and predicted
poses. This experiment intentionally replaces that with a non-privileged verdict,
checked scope, actual robot pose, and visual recovery class. The local result is
positive, but changing that shared interface contract still requires three-person
review before the RFC is updated.
