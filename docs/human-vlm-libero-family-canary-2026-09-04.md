# Human VLM LIBERO family canary — 2026-09-04

## Scope

- `openeta/libero_libero_spatial_task0-v0`, seed 0
- `openeta/libero_libero_object_task0-v0`, seed 0
- `openeta/libero_libero_10_task0-v0`, seed 0
- Human VLM decisions use only provider-visible observations, session artifacts,
  the Python sandbox, and registered tools.
- Success requires trusted environment evidence reporting positive reward and a
  terminal episode.

## Current result

| Task | Status | Evidence / blocker |
|---|---|---|
| Spatial 0 | Tool-incomplete | The bowl transport attempt exposed an attached-object sweep call regression, followed by a repeated environment-rebuild fresh-observation loop. The sweep keyword mismatch is fixed and covered by a regression test; the rebuild/freshness behavior remains a separate harness gap. |
| Object 0 | Pass | Session `e4e91a6a-4670-40e4-918f-6a713c47b6a5` completed the alphabet-soup pick and basket placement. The stationary gripper-open action at episode turn 48 returned trusted `reward=1`, `terminated=true`. |
| Long 0 | Pass | Session `3e3580c6-934e-40a9-9071-f2b80ab63d1e` independently picked and verified both cans, placed them at offset basket locations, and received trusted `reward=1.0`, `terminated=true` on the second stationary release at episode turn 99. |

## Object 0 evidence

- Mink reached the wrist-viewpoint target with approximately 1 mm translation
  error.
- It reached the grasp clearance within approximately 2.6 mm and contact within
  approximately 4.6 mm.
- The closed gripper retained the alphabet-soup can through a 5 cm upward probe.
- Independent visual review reported co-motion, source vacancy, and continued
  wrist-view engagement.
- A later resumed run completed placement and received trusted terminal reward
  in session `e4e91a6a-4670-40e4-918f-6a713c47b6a5`.

## Long 0 terminal evidence

- The alphabet-soup and tomato-sauce cans were each accepted only after a
  host-frozen 5 cm attachment probe and independent visual co-motion review.
- The alphabet-soup can was released near the basket centre. For the second
  can, fresh same-packet SAM3 evidence and an AnyPlace bundle produced the
  low placement reference. The Agent retained its x/y geometry but raised the
  release EEF to `z=0.65 m` and offset x by about 4 cm from the first can.
- Fresh agentview and wrist images showed the tomato-sauce can inside the
  basket opening with rim clearance immediately before stationary release.
- The second `gripper_control(position=1)` returned trusted `reward=1.0` and
  `terminated=true`; the harness then closed the active MCP environment.

## Harness defects found

### Reused render artifact path

Repeated read-only renders may return the same immutable local image path while
the Agent assigns a new compact `obs-*` packet ID. Reverse lookup previously
required exactly one packet owner and therefore returned no owner for this
valid history. This produced the contradictory AnyPlace projection in which
`required_source_image` and `provided_source_image` were identical while the
status was `source_packet_unresolvable`.

The resolver now selects the newest observation packet when every match refers
to the same camera frame. It still fails closed for cross-camera ambiguity or
conflicting owners at the newest observation index.

### Isolated advisor requests are easy to miss in manual orchestration

The grasp advisor uses an `inferred-*` session ID instead of the main Agent
session ID. Filtering pending Human VLM console requests by the main session
therefore hides the advisor even though the facade is correctly waiting for
it. Both cancelled retries had already materialized successful backend results;
they were waiting for manual advisor responses, not blocked in model inference.
Manual orchestration must filter by `request_type=grasp_selection_advice` (and
task or receive time), rather than requiring the main session ID. A future
console improvement should explicitly link isolated requests to their parent
Agent session.

### Wrist-viewpoint proposal values disappear from the planner projection

`propose_wrist_viewpoints` tells the Agent to copy an exact candidate
`target_pose`, but the ordinary planner projection replaces its numeric xyz and
rotation matrix with `<omitted>`. During Long 0, the exact calibrated oblique
viewpoint remained recoverable only from the append-only trace. The run could
continue through the coding-agent artifact interface, but the main planner
projection should instead preserve the selected/copy-required numeric pose or
return an opaque receipt that the IK tool can consume directly.

### Same-source AnyPlace repair remains turn-expensive

After attachment, placement still required segmenting the held object and the
receptacle on one fresh packet, re-running grasp estimation, compiling a
placement-only grasp representation, then invoking AnyPlace. This was correct
and ultimately successful, but it duplicated expensive grasp work and helped
push Long 0 to 99 episode turns. The evidence graph should be able to derive a
same-instance placement representation without implying that the secure grasp
must be selected or executed again.

## Resume condition

Object 0 and Long 0 now have trusted terminal success. Spatial 0 remains marked
tool-incomplete until its environment-rebuild freshness loop is reproduced and
repaired. The next requested coverage point is LIBERO Long 9.
