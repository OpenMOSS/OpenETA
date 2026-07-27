# Real-Robot Depth and Point Cloud Enhancement

This document proposes a real-robot-only observation enhancement module for
OpenETA. The goal is to improve RGB-D depth completeness and point-cloud
continuity without treating monocular depth as a replacement for calibrated
sensor depth.

## Motivation

Real robot RGB-D observations often contain missing depth, edge noise, flying
points, reflective-surface failures, and sparse or incomplete point clouds. This
hurts downstream modules such as segmentation-guided grasp generation,
placement reasoning, scene monitoring, and operator debugging.

UniDepth V2 is a useful first backend for this problem because it predicts
metric depth, camera-coordinate points, intrinsics, and per-pixel confidence from
a single RGB image. It also accepts known camera models and intrinsics at
inference time. In OpenETA, it should be used as a dense geometric prior that is
calibrated against real RGB-D measurements, not as a source of ground truth.

## Non-Goals

- Do not replace reliable RGB-D sensor depth with monocular predictions.
- Do not use monocular-only geometry for final collision clearance, free-space
  carving, or precise grasp/contact height without confirmation.
- Do not change the simulator observation contract for this feature. The module
  is enabled by real-robot deployment profiles.
- Do not require a local GPU dependency in the default agent runtime. UniDepth
  should be pluggable and may run behind a remote MCP service.

## Existing OpenETA Touchpoints

OpenETA already has the right low-level contracts to add this as an observation
post-processor:

- `adapter.protocol.CameraFrame` carries `rgb`, `depth`, `intrinsics`,
  `extrinsics`, and `timestamp_s`.
- `adapter.protocol.EnvObservation.metadata` can carry compatible enhancement
  references without changing existing camera fields.
- `tools.anygrasp_core.build_point_cloud_from_rgbd` already backprojects depth
  with real intrinsics.
- `tools.graspgenx_core.build_targeted_point_clouds` follows the same
  RGB-D-to-point-cloud pattern.
- `docs/rollout-data-contract.md` records RGB and depth as immutable evidence.
- `docs/calibration-lifecycle.md` treats camera and robot calibration as
  reviewed profiles, which is the right place to bind real camera intrinsics and
  extrinsics.

The enhancement module should therefore preserve raw camera observations and
emit additional artifacts plus provenance metadata.

## Proposed Module

Add a post-observation module:

```text
real robot camera capture
  -> RGB/depth timestamp and registration checks
  -> depth enhancement post-processor
  -> enhanced depth and point-cloud artifacts
  -> planner/tool-visible metadata references
  -> grasp, placement, monitoring, and logging tools
```

Suggested implementation locations:

- `agent/runtime/depth_enhancement.py`
- `agent/runtime/depth_prior.py`
- `agent/runtime/pointcloud_artifacts.py`
- `tests/runtime/test_depth_enhancement.py`
- `docs/real-robot-depth-pointcloud-enhancement.md`

If UniDepth is exposed as a remote service, add only a thin client in the agent
runtime. The model server itself should live outside the default runtime path or
behind an MCP tool.

## Agent Tool Entry

The first agent-visible entry points are:

- `estimate_depth_prior`: optional remote metric monocular depth-prior call.
- `enhance_depth`: local sensor-first fusion and artifact materialization.

Configure the optional remote prior service in `.mcp.json` as one of
`openeta-depth-prior`, `depth-prior`, `depth_prior`, or `unidepth`:

```json
{
  "mcpServers": {
    "openeta-depth-prior": {"url": "http://<host>:<port>/sse"}
  }
}
```

`estimate_depth_prior` accepts local artifact paths instead of inline arrays:

```json
{
  "rgb": "tmp/image/rgb/<session>/<bundle>/wrist.png",
  "intrinsics": {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0},
  "camera_id": "wrist",
  "camera_model": "pinhole",
  "calibration_profile_id": "real_robot_camera_profile"
}
```

It materializes `prior_depth` and optional `prior_confidence` as local files.
The accompanying `prior_confidence_semantics` is mandatory whenever confidence
exists: UniDepth V2 uses `higher_is_better`; services returning uncertainty use
`lower_is_better`.
Then call `enhance_depth` with the same RGB-D observation:

```json
{
  "rgb": "tmp/image/rgb/<session>/<bundle>/wrist.png",
  "depth": "tmp/image/depth/<session>/<bundle>/wrist-sensor.png",
  "intrinsics": {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0},
  "prior_depth": "tmp/tool_result/depth_prior/<session>/<bundle>/wrist-depth.npy",
  "prior_confidence": "tmp/tool_result/depth_prior/<session>/<bundle>/wrist-confidence.npy",
  "prior_confidence_semantics": "higher_is_better",
  "sensor_confidence": "tmp/image/depth/<session>/<bundle>/wrist-confidence.png",
  "camera_id": "wrist",
  "calibration_profile_id": "real_robot_camera_profile",
  "calibration_hash": "sha256-of-reviewed-profile",
  "registration_status": "registered",
  "rgb_timestamp_s": 0.0,
  "depth_timestamp_s": 0.0,
  "scene_epoch": 0,
  "bundle_id": "optional-stable-bundle-id"
}
```

This keeps the model backend decoupled from fusion. `estimate_depth_prior`
materializes model outputs; `enhance_depth` remains the single place that
applies sensor-first alignment, conservative filling, point-cloud
backprojection, and provenance reporting. When the `enhance_depth` quality gate
allows grasp candidate generation, the host planner context can route
`grasp_pose_estimate` to the candidate depth PNG while preserving the
provenance mask and sensor-only safety artifacts in hints. AnyGrasp collision
filtering is disabled for this channel; backends without candidate-only
semantics are ineligible.

### SAM3-Triggered Depth Prefetch

When both SAM3 and the remote depth-prior service are configured, the host
starts `estimate_depth_prior` in a background worker immediately before sending
the validated RGB request to SAM3. This is host-owned latency hiding, not a
planner-visible parallel tool batch:

- The prefetch uses the same local RGB artifact, calibrated intrinsics, camera
  id, camera model, and calibration profile as the current observation.
- Requests are deduplicated by session, RGB file identity, intrinsics, camera
  metadata, and UniDepth resolution level.
- A later explicit `estimate_depth_prior` call reuses the in-flight or completed
  result. It waits only for any unfinished portion of the inference.
- Missing intrinsics or a failed prefetch never fail or delay the SAM3 result.
- The planner must still call `estimate_depth_prior`, then pass its materialized
  outputs to `enhance_depth`. AnyGrasp must consume the sensor-first enhanced
  artifacts, not the raw monocular prediction.

The coordinator serializes remote depth inference for a configured CLI runtime
to avoid concurrent GPU pressure while still overlapping it with SAM3 network
and model latency.

## Backend Interface

The enhancer should depend on an interface rather than importing UniDepth
directly.

```python
@dataclass(slots=True)
class DepthPriorPrediction:
    depth_m: np.ndarray
    confidence: np.ndarray | None
    points_camera: np.ndarray | None
    intrinsics: dict[str, Any] | None
    metadata: dict[str, Any]


class DepthPriorBackend(Protocol):
    def infer(
        self,
        *,
        rgb: np.ndarray,
        intrinsics: dict[str, float],
        camera_model: str,
        profile_id: str,
    ) -> DepthPriorPrediction:
        ...
```

The first production backend can be `unidepth_v2_mcp`, where the local runtime
sends RGB plus the calibrated camera model to a remote MCP service and receives
metric depth, confidence, optional model points, and backend metadata.

The default backend should be `noop`, which returns no enhancement and preserves
current behavior.

## Required Inputs

For each RGB-D frame:

- RGB image in the target color camera frame.
- Depth image registered into the RGB image coordinate frame.
- Real RGB camera intrinsics: `fx`, `fy`, `cx`, `cy`, image size, distortion
  model, and depth scale.
- Camera extrinsics relative to the robot or world frame.
- Timestamp and synchronization diagnostics.
- Optional hardware confidence or validity mask.
- Calibration profile id and hash.

Depth must be in metres internally. If the wire format is uint16 millimetres,
the module should decode it with `depth_m = depth_uint16 / 1000.0`.

## Fusion Pipeline

### 1. Validate Alignment

Before using RGB edges or monocular predictions, validate that RGB and depth are
registered to the same image plane. If the camera has separate RGB and depth
sensors, this requires:

- RGB intrinsics.
- Depth intrinsics.
- RGB-to-depth extrinsics.
- Distortion parameters.
- A depth-to-RGB resampling policy.

If alignment is missing or stale, the enhancer must return `enabled=false` and
record a diagnostic. Misaligned RGB-D causes doubled object edges and flying
points.

### 2. Build Sensor Masks

Compute at least these masks:

- `sensor_valid`: finite positive depth inside configured range.
- `sensor_reliable`: valid pixels after removing obvious edge, mixed-pixel,
  reflective, temporal-noise, or hardware-low-confidence regions.
- `edge_guard`: RGB/depth edge neighborhood where cross-boundary smoothing or
  filling is unsafe.
- `mono_candidate`: pixels where the monocular prior may be used.
- `unknown`: pixels that should remain unavailable.

The mask logic should be conservative near robot contact regions and object
boundaries.

### 3. Run the Monocular Prior

Run UniDepth V2 or another metric depth prior with the real camera model:

```python
prediction = depth_prior.infer(
    rgb=rgb,
    intrinsics=rgb_intrinsics,
    camera_model="pinhole",
    profile_id=calibration_profile_id,
)
```

Do not prefer model-predicted intrinsics when calibrated intrinsics are
available. The model intrinsics can be recorded for debugging but should not be
used to backproject the final OpenETA point cloud.

UniDepth confidence should be treated as an intra-image relative uncertainty
signal. It is not an absolute variance and should not be converted directly into
physical covariance.

### 4. Calibrate Model Depth to Sensor Depth

Even metric monocular depth can have scene-specific scale or local bias. Fit a
robust alignment on reliable sensor pixels:

```text
mono_aligned(u) = a * mono_depth(u)
```

Start with scale-only alignment. Add an offset term only after repeated real
robot evidence shows a stable offset:

```text
mono_aligned(u) = a * mono_depth(u) + b
```

Use Huber, RANSAC, or trimmed least squares over `sensor_reliable` pixels.
Reject samples near `edge_guard`, outside workspace depth limits, or with large
RGB-D temporal disagreement. A later version may fit separate scale values by
depth band or image region.

### 5. Fuse Depth Conservatively

Use a sensor-first rule:

```text
if sensor_reliable:
    fused_depth = sensor_depth
elif sensor_valid but noisy:
    fused_depth = denoised_sensor_depth or unknown
elif mono_candidate and mono_confident and not edge_guard:
    fused_depth = mono_aligned
else:
    fused_depth = unknown
```

Avoid uniform averaging such as `0.5 * sensor + 0.5 * mono`. It degrades good
sensor pixels and can create physically invalid surfaces between foreground and
background at occlusion edges.

If sensor and model disagree beyond a configured threshold on a reliable pixel,
the safe default is to keep the sensor value and record the disagreement. If the
pixel is important for manipulation, request another view or a guarded robot
motion before acting on model-only geometry.

### 6. Backproject With Real Intrinsics

After fusion, backproject using the calibrated RGB-D camera intrinsics:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = fused_depth(u, v)
```

Do not concatenate a sensor point cloud with UniDepth's model point cloud. Their
camera rays may differ. Model points are useful for debugging, but the final
OpenETA point cloud should be generated from fused depth and real intrinsics.

The implementation emits two explicit geometry channels:

- `candidate_depth` and candidate point cloud: reliable sensor measurements plus
  accepted monocular hole fills. These may generate grasp candidates.
- `safety_depth` and safety point cloud: reliable sensor measurements only.
  These are the only enhancement artifacts eligible for downstream safety
  confirmation.

Each candidate point should carry provenance and weight:

- `sensor`: high weight, usable for grasp/contact and collision checks.
- `mono_filled`: low weight, usable for coarse scene completion, segmentation
  support, and candidate generation.
- `unknown`: no point.

## Artifact Contract

Emit artifacts under the existing repository-local temporary artifact root:

```text
tmp/tool_result/depth_enhancement/<session-id>/<bundle-id>/<camera-id>-depth-enhancement.json
tmp/tool_result/depth_enhancement/<session-id>/<bundle-id>/<camera-id>-fused-depth.npy
tmp/tool_result/depth_enhancement/<session-id>/<bundle-id>/<camera-id>-safety-depth.npy
tmp/tool_result/depth_enhancement/<session-id>/<bundle-id>/<camera-id>-point-cloud.npz
tmp/tool_result/depth_enhancement/<session-id>/<bundle-id>/<camera-id>-safety-point-cloud.npz
tmp/image/depth/<session-id>/<bundle-id>/<camera-id>-sensor.png
tmp/image/depth/<session-id>/<bundle-id>/<camera-id>-fused.png
tmp/image/depth/<session-id>/<bundle-id>/<camera-id>-safety.png
tmp/image/mask/<session-id>/<bundle-id>/<camera-id>-provenance.png
```

Suggested JSON schema:

```json
{
  "schema_version": "openeta.depth_enhancement.v1",
  "enabled": true,
  "camera_id": "wrist_rgbd",
  "calibration_profile_id": "real_robot_camera_profile",
  "source": {
    "rgb_path": "tmp/image/rgb/...",
    "sensor_depth_path": "tmp/image/depth/...",
    "intrinsics": {"fx": 0.0, "fy": 0.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0}
  },
  "prior": {
    "backend": "unidepth_v2_mcp",
    "model": "UniDepthV2",
    "used_calibrated_camera": true,
    "confidence_semantics": "higher_is_better"
  },
  "alignment": {
    "mode": "scale_only",
    "scale": 1.0,
    "offset_m": 0.0,
    "reliable_pixel_count": 0,
    "robust_loss": "huber"
  },
  "outputs": {
    "fused_depth_npy": "tmp/tool_result/...",
    "candidate_depth_npy": "tmp/tool_result/...",
    "safety_depth_npy": "tmp/tool_result/...",
    "point_cloud_npz": "tmp/tool_result/...",
    "provenance_mask_png": "tmp/tool_result/..."
  },
  "quality": {
    "sensor_valid_ratio": 0.0,
    "filled_ratio": 0.0,
    "mono_only_ratio": 0.0,
    "large_disagreement_ratio": 0.0,
    "use_for_grasp_candidate_generation": false,
    "use_for_collision_clearance": false
  },
  "diagnostics": []
}
```

`fused_*` remains a backward-compatible alias for `candidate_*`. All
materialized PNG depth artifacts are canonical uint16 millimetres and publish
`depth_scale=1000`; metric NPY arrays remain float32 metres. The report stores
source paths, SHA-256 digests, intrinsics, timestamps, scene epoch, registration
status, and calibration identity so same-path rewrites and stale frames can be
rejected.

Attach this JSON reference to `EnvObservation.metadata["depth_enhancement"]`
or to the relevant tool result artifacts. Keep raw camera fields unchanged.

## Downstream Policy

### Grasp Tools

AnyGrasp, GraspGenX, and Contact-GraspNet should accept enhanced depth only when
the enhancement report explicitly marks it usable for candidate generation. Even
then:

- Rank or filter grasps using sensor-confirmed local points when possible.
- Down-weight candidates whose contact patch is mostly `mono_filled`.
- Reject candidates whose collision clearance depends on `mono_filled` points.
- Store the enhancement report path in tool metadata for audit.

The current host implementation permits candidate depth only through AnyGrasp
with `collision_detection=false`. Contact-GraspNet and GraspGenX are skipped
until they expose an equivalent candidate-only contract. A returned grasp keeps
`requires_sensor_safety_check=true`. Before grasp compilation, the host
dispatches `obstacle_avoidance` with the exact candidate id, scene epoch,
enhancement report, `safety_depth_png`, and sensor-only safety point cloud.
Only a matching `clear=true` result unlocks `compile_grasp_seed`; an unavailable
checker, missing artifact, stale epoch, or non-clear result fails closed.
Model-filled geometry is never represented as collision-free evidence.

## UniDepth V2 MCP Service

The repository includes:

- `tools/unidepth_v2_core.py`: lazy official UniDepth V2 backend.
- `tools/unidepth_v2_mcp_server.py`: `estimate_depth` over stdio or SSE.
- `tools/requirements-unidepth-v2.txt`: isolated service dependencies.

The default checkpoint is the official V2 ViT-L model
`lpiccinelli/unidepth-v2-vitl14`. Start it through the service manager:

```bash
python scripts/openeta_mcp_services.py start unidepth_v2 \
  --unidepth-v2-python /path/to/unidepth-env/bin/python \
  --unidepth-v2-device cuda \
  --unidepth-v2-resolution-level 4
```

Then configure the Agent-side MCP alias:

```json
{
  "mcpServers": {
    "openeta-depth-prior": {"url": "http://<gpu-host>:8779/sse"}
  }
}
```

### Segmentation and Grounding

SAM-style tools may use enhanced point clouds to improve object extent,
background separation, and table/object continuity. This is lower risk than
using model-only geometry for contact decisions.

### Motion and Safety

Motion planning may use enhanced geometry for coarse scene awareness, but final
clearance and contact decisions must rely on sensor-confirmed points or an
explicit human/robot confirmation step. Monocular-only points should not carve
free space.

## Evaluation Plan

Start with offline replay before enabling the module in closed-loop control.

Metrics:

- Sensor valid depth ratio before and after enhancement.
- Fill ratio in known sensor holes.
- Median and P95 disagreement on reliable sensor pixels.
- Edge leakage rate around foreground/background boundaries.
- Flying-point count after workspace filtering.
- Target object point count and stability across frames.
- Grasp candidate stability and sensor-confirmed contact ratio.
- Real robot task success, intervention, and near-miss rate.

Canary scenes:

- Matte cube or box on a table.
- Cluttered tabletop with known object sizes.
- Thin object or edge-heavy object.
- Transparent cup, reflective metal, and glossy packaging as negative cases.
- Multi-view confirmation after small wrist or arm motion.

The first acceptance gate should not be "more points is better". It should be
"more usable points without increasing unsafe or physically inconsistent
geometry".

## Rollout Plan

1. Add the `noop` enhancer and artifact schema tests.
2. Add an offline replay CLI that reads recorded RGB-D frames and writes
   enhancement artifacts.
3. Add the remote UniDepth V2 backend behind configuration.
4. Enable enhancement only for a real-robot calibration profile.
5. Add downstream tool flags so grasp tools can opt into enhanced depth.
6. Run offline and guarded real-robot canaries.
7. Promote the profile only after reviewed calibration and safety evidence.

## Risks

- UniDepth's license is CC BY-NC 4.0, so deployment use must be reviewed before
  any commercial or external production use.
- Transparent, reflective, mirror-like, and visually ambiguous surfaces can
  produce plausible but false geometry.
- Model confidence is relative within one image, not a calibrated physical
  uncertainty.
- Poor RGB/depth registration can make enhancement worse than raw depth.
- Over-trusting filled geometry near the gripper can create unsafe grasps.

## References

- UniDepth V2 paper: https://arxiv.org/html/2502.20110v2
- UniDepth repository and usage examples:
  https://github.com/lpiccinelli-eth/UniDepth
- UniDepth V2 notes on confidence, camera support, and ONNX:
  https://github.com/lpiccinelli-eth/UniDepth/blob/main/assets/docs/V2_README.md
- Marigold-DC depth completion prior work:
  https://arxiv.org/abs/2412.13389
