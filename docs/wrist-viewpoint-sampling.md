# Bounded wrist-viewpoint geometry

2026-09-06, HV-03 partial candidate on the personal refactor branch. Sampling,
additional output fields, proposal hashing, result count bound and handoff hints
require three-person contract review. This is **not** a pre-SAM3 recovery entry.

## Existing entry, expanded sampling

The public input remains `compiled_grasp_id`, `source_packet_id`, and wrist
`camera_frame_id`. The host resolves compiled target geometry, current EEF pose
and calibrated camera extrinsics. There is no Agent-specified candidate count or
new camera-moving tool.

The sampler now returns eight independent choices: four azimuths around the
target, at each of two elevations (40° and 65° above the world XY plane). The
azimuth origin follows the current target-to-camera bearing; directly above or
coincident positions use projected camera X, then world X as a degenerate fallback.
World +Z remains the assumed up direction, as in the previous implementation.

Host-private `standoff_m` keeps its old meaning of **height above the target**,
not radial distance. Defaults remain `[0.18, 0.22]` m, cycled through the bounded
samples. `camera_goal.target_distance_m` separately reports Euclidean target-to-
camera distance. `lateral_offset_m` retains its old world-X offset meaning; the
new full `offset_world_xyz` expresses the missing Y component without redefining
that legacy field. Up to three private height values are still accepted, but
cannot increase the output above eight. Public result schema now declares the cap.

There is no hidden scan sequence. The Agent may choose one, none, or gather other
evidence. This batch does not prefilter workspace feasibility: the result explicitly
reports `workspace_prefilter: "not_available"` and each candidate requires further
workspace, exact IK and collision checks. It must not be described as an already
safe 4–8-view set. Future filtering may return fewer candidates; it should not
fill a quota by adding unsafe or duplicate poses.

## Geometry and quality semantics

The existing rigid camera mount is preserved: solve target-facing camera pose,
then recover EEF pose using the full camera-to-EEF transform. Tests reconstruct
that mount and check camera origin, orthonormal rotation and target optical axis.

Example additive candidate hints (not an executable request):

```json
{
  "candidate_id": "wrist_view_00",
  "view_quality_estimates": {
    "view_angle_change_deg": 25.0,
    "target_anchor_on_optical_axis": true,
    "target_visibility": "unverified",
    "occlusion_quality": "unknown",
    "projected_target_size_px": null,
    "interpretation": "Geometric ranking hints only; not safety authorization."
  },
  "requires_workspace_check": true,
  "requires_collision_check": true,
  "requires_ik_preview": true
}
```

Angle change compares old/new target-to-camera rays, not measured visual quality
or required wrist rotation. Coincident current target/camera gives `null`, not a
fabricated angle. Target-on-optical-axis describes the ideal point geometry;
neither calibrated image coverage of the whole object nor actual visibility is
established. Without target extent/depth occlusion evidence, size and occlusion
quality remain unknown. Camera goal reachability is also unverified.

## Identity and execution references

Proposal hashing now includes full EEF and mount rotations plus the actual bounded
candidate geometry and sampling-policy version. Previously rotation-only changes
could retain the same ID despite producing different poses. Geometry-equivalent
packet-only refreshes still share the same ID; epochs remain part of the hash.
This identifies the generated geometry; it does not itself establish freshness.
The resolver now requires a successful production result, exact integer epochs,
and complete compiled/source/camera/mount bindings. It checks the referenced
compiled artifact against current object epoch and, when present, active grasp
provenance/target identity. Superseded or invalidated target evidence is rejected.

Original and latest same-camera session packets must still resolve with calibrated
inputs. A bounded pure-geometry calculation checks their recovered rigid mount
against the stored mount (absolute tolerance `1e-6` for the rounded coordinates).
The selected pose is not regenerated or substituted. Changes in target anchor or
camera mounting invalidate the old reference even if a producer failed to advance
the robot epoch. Equivalent packet-only refreshes remain reusable.

Missing provenance graphs are not upgraded to independently verified identity;
artifact-only host geometry remains subject to the existing target provenance
model. Environment incarnation/operation lifetime auditing is still separate.
Latest same-camera packet means the latest available matching camera in the index,
not proof that every newer observation contained that camera.

Candidate IDs must occur exactly once within the proposal. Resolved nested pose
data is deep-copied so a consumer cannot accidentally alter the historical source.
Legacy minimal records without source or mount bindings are no longer resolvable;
regenerate from current evidence instead of inventing missing calibration. The
result schema now declares these already-produced binding fields as required.

Handoff guidance now prefers the already-supported short-ID request:

```json
{
  "viewpoint_proposal_id": "wrist_viewpoint:<actual-proposal-id>",
  "candidate_id": "wrist_view_00"
}
```

Use actual IDs in `ik_preview_check`; the host resolves the exact full pose.
Then execute only the applicable `ik_receipt_id`. Neither sample score nor
proposal ID grants motion authority. The previous raw-pose compatibility path
has not been removed, but the new guidance does not ask the Agent to copy matrices.

## Unfinished HV-03 requirements

Pre-segmentation weak localization/identity entry and actual post-motion packet
projection remain unimplemented pending design decisions in
[active perception](active-perception-design-options-2026-09-05.md). No new
identity anchor semantics or localization resource was introduced in this batch.
Quality-trigger guidance, working-domain filtering, target extent/occlusion
estimates, calibration freshness, and real Spatial 0 / Long 9 tests remain open.
Geometry and short-ID unit tests do not prove that a previously hidden cup handle
is visible or that any sampled robot trajectory is executable.
