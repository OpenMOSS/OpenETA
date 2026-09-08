# Experimental bundle interface stages 2 and 3

User authorized continuous migration and Luna/API testing on 2026-09-07.
RFC collaboration constraints re-read at revision 2158. This is work on the
personal refactor branch, not a stable-contract promotion or three-person signoff.
No default profile, historical review hash, commit or shared document changed.

## Scope

| Profile | Strict manifest consumers added | Capability kept available |
| --- | --- | --- |
| Stage 1 | compile_grasp_seed, move_to | Existing default IK forms |
| Stage 2 | ik_preview_check | propose_motion_target authors all five prior native target sources |
| Stage 3 | follow_eef_trajectory, select_sam3_detection, reject_sam3_detections, grasp_pose_estimate, anyplace | compose_ik_trajectory groups 1–5 ordered IK result bundles; identity and candidate choices remain explicit |

Stage 2 auto-publishes target bundles from compile, wrist viewpoints, calibrated
wrist alignment, frozen attachment probes and camera-to-world transformation.
`propose_motion_target` is the manual geometry authoring exception: world pose,
compiled role/path sample, viewpoint candidate or probe waypoint. It performs
no IK, motion, task-stage transition or geometry relaxation. Existing producer
bundles can go straight to IK; authoring is not a mandatory extra step.

Stage 3 unifies two existing ready native-input formats behind immutable
manifest IDs. Native grasp/placement provenance resolvers are still authoritative;
the manifest does not duplicate RGB-D or make missing native evidence valid.
SAM3 prompting, observation packet/camera selection, semantic identity fields,
camera transforms, probe construction and refinement/viewpoint proposal inputs
still have explicit fields. This is not a claim that every ID in the repository
is now a single bundle format or that legacy implementations can be deleted.

Examples below use placeholders, not executable references:

```json
{"tool":"propose_motion_target","parameters":{"target_pose":{"frame":"world","xyz":[0.1,0.2,0.3]},"preserve_current_orientation":true}}
{"tool":"ik_preview_check","parameters":{"bundle_id":"<target-pose-manifest>"}}
{"tool":"move_to","parameters":{"bundle_id":"<ik-result-manifest>"}}
{"tool":"compose_ik_trajectory","parameters":{"bundle_ids":["<first-ik-result>","<second-ik-result>"]}}
{"tool":"follow_eef_trajectory","parameters":{"bundle_id":"<ordered-trajectory-manifest>"}}
{"tool":"select_sam3_detection","parameters":{"bundle_id":"<sam3-detections-manifest>","detection_id":"<chosen-detection>"}}
```

## Safety and context

The experimental request projection and admission use the same derived schema.
Planner and pipeline profiles must agree. Native/mixed references are rejected
on migrated public consumers; default legacy-compatible behavior is retained.
Schema validation supports the registered manifest ID pattern for every migrated
consumer and trajectory composition; whitespace and native IDs fail admission.

Proposal handlers only return read-only data; registration occurs later in the
runtime-owned memory commit. A cancelled/late handler cannot register evidence
in a new session. Ordered trajectory bundles contain exact native receipt IDs,
not unchecked poses or newly selected seeds. Duplicate receipts are rejected;
the motion compiler still checks all receipts and all existing trajectory gates.

Inspectable proposal summaries include purpose and original typed reference;
handoffs never authorize execution. Raw command/trace remains audit evidence.
The bounded discovery window retains the latest current bundle of each kind
before filling remaining slots by recency, so repeated IK/compile output cannot
hide an otherwise current source bundle. IK summaries retain their target pose,
signature and parent manifest association. This improves discovery, not authority.
Callable guidance in the Agent projection points at current manifests, or at
the authoring tool for new geometry; hints without current manifest bindings
are explicitly non-callable. Historical native requests are not silently
rewritten into new execution authorizations. Fixed profile is Host-owned.

No physical-quality threshold, collision policy, gripper gate, attachment
criterion or success definition is weakened. Compiled-grasp-free active
perception and orientation-changing grasp refinement are still separate
unfinished work, not hidden inside this interface migration.

## Validation plan and records

Local fixtures cover every IK source, producer handoffs, current/stale/session
boundaries, immutable registration, restored-session staleness, identity choice,
native input gate retention, ordered trajectory references, schema projection
and shared runtime assembly. Full suite and actual batch results are recorded
in the separate experiment report; no synthetic result counts as task success.

Fresh standard batch manifest: `tmp/batch-luna-stages23-MJYJuG/manifest.json`.
Two independent Object 0/seed 0 episodes, concurrency 1, main/fallback Luna only.
Each restores 15M known cumulative tokens / 160 turns / 320 tools / 10800 s.
New workspaces do not import historical traces. The comparison is a staged
interface smoke/regression experiment, not a success-rate estimate or a
controlled causal attribution study. Keep runtime code frozen during each run.
