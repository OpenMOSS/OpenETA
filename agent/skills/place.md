---
name: place
description: Guidance for placing a held object on or inside a target receptacle.
version: v1
editable: true
task_patterns:
  - place <object> on <target>
  - put <object> into <target>
  - place <object> in <target>
allowed_tools:
  - observe
  - sam3
  - select_sam3_detection
  - reject_sam3_detections
  - anyplace
  - camera_pose_to_world
  - ik_preview_check
  - move_to
  - follow_eef_trajectory
  - gripper_control
---
# Place

Use this as reusable placement guidance, not an executable macro. It explains
placement decisions; the live tool contracts exclusively define request fields,
returned references, validity rules, and repair payloads.

## Plan placement evidence early

1. In a combined pick-and-place task, establish placement evidence before the
   first grasp motion when the placement estimator requires the object and
   receptacle from the same aligned pre-grasp scene. Do not wait until the held
   object occludes or changes that evidence.
2. Segment and visually confirm the receptacle or support region separately from
   the grasp target. A high segmentation score does not prove that the region is
   the requested destination. Keep object identity and placement-region identity
   as distinct evidence.
3. Use host-prepared placement evidence when it is ready. If it is stale or
   mismatched, follow the structured repair offered by the tool rather than
   choosing remembered image paths or rebuilding calibration. Fixed scene-camera
   evidence may remain reusable when the host proves its identity and geometry
   unchanged; moving wrist-camera evidence does not receive that assumption.
4. An exact-task playbook may identify the destination or describe a previously
   successful carry pattern. Treat that as a scoped prior only and re-verify the
   current scene, free space, attachment, and destination before acting.

## Choose a release reference

5. Select a complete placement candidate using visible containment, clearance
   from receptacle walls or support edges, object footprint, and estimator
   ranking. Prefer an interior candidate with robust clearance when that evidence
   is available. Ranking is a heuristic, not proof that the carried object will
   fit or remain stable.
6. Transform the selected placement reference with the matching camera
   calibration before motion. Treat the transformed pose as a low geometric
   release reference, not as a one-step carry trajectory or immutable command.
   Never substitute a grasp pose on the receptacle for a placement estimate.

## Carry and approach

7. Begin placement only after fresh evidence still supports attachment. Keep the
   gripper latched throughout the carry and reason about collisions using the
   full held-object extent, not only the fingers.
8. Decompose transport according to current geometry: first obtain vertical
   clearance, then use bounded lateral motion above clutter, and perform descent
   as a separate edge. Preserve the current orientation unless an explicitly
   checked change is useful. Avoid diagonal motion through a receptacle rim or
   nearby object.
9. Check every chosen endpoint before execution and inspect the actual motion
   result. Endpoint reachability does not prove path clearance. When attached-
   object collision is reported, use the named obstacle and predicted geometry
   to choose a higher, more central, or lateral detour. If the conservative
   envelope starts in overlap, choose a monotonic escape that reduces it rather
   than disabling collision checks.
10. Reassess attachment after carry waypoints. A probe result from before a long
    motion is historical evidence; visible separation, source reappearance, or
    empty-gripper evidence requires stopping placement and reacquiring the object.

## Release

11. Refine the low placement reference with current visual evidence while staying
    inside the runtime safety envelope. Choose the release geometry from the
    object, receptacle, and corridor rather than treating one method as a fixed
    phase:

    - Use controlled descent when the opening or support surface is clear and
      the carried object can enter without rim contact.
    - A bounded raised drop can be reasonable for a visibly open container and a
      non-fragile object that clearly fits when descent adds more rim risk. Do not
      use it for surface placement, fragile objects, narrow openings, or uncertain
      containment.

12. Stop lateral motion before release. Open only after the selected endpoint was
    actually reached and fresh evidence still shows the object attached over the
    intended destination. A failed carry or descent is not a valid release premise.
13. Retreat to reveal the result, observe, and verify that the object is no
    longer held and is stably on or inside the intended target. In benchmark
    episodes, use the official environment reward as the completion authority.

## Recovery choices

- Ambiguous destination: obtain another view or ask for clarification.
- Missing compatible pre-grasp placement evidence: use the structured repair or
  report the capability gap; do not fabricate a placement pose.
- Blocked carry: raise or route around the named obstacle using newly checked
  geometry rather than replaying the failed diagonal path.
- Object lost in transit: stop, re-ground the target, and return to pick guidance.
- Object remains held after release: observe the contact and receptacle geometry,
  then retry only if evidence supports a safe release adjustment.

For explicit controller-profile or clearance characterization, use the
`embodiment_explore` skill outside the benchmark episode.
