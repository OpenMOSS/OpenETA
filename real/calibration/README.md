# Hand-eye calibration (eye-to-hand + eye-in-hand)

Offline hand-eye calibration for the UR5e multi-view bench. Solves the rigid
transform between each camera and the robot from teleop recordings of a
checkerboard, using `cv2.calibrateHandEye`.

Two mounting types are supported and are calibrated **separately** because the
data-collection procedure differs:

| Camera        | serial        | mount   | mode        | solves         | board is...        |
|---------------|---------------|---------|-------------|----------------|--------------------|
| `d435i_fixed` | not committed | fixed   | eye-to-hand | `T_base_cam`   | mounted on the EE  |
| `l515`        | not committed | fixed   | eye-to-hand | `T_base_cam`   | mounted on the EE  |
| `d435i_wrist` | not committed | wrist   | eye-in-hand | `T_gripper_cam`| fixed in the scene |

- **Fixed camera (eye-to-hand):** the checkerboard is rigidly attached to the
  end-effector; the camera watches it from a fixed vantage. Result is the
  camera pose in the robot base frame, `T_base_cam`.
- **Wrist camera (eye-in-hand):** the checkerboard is fixed in the scene; the
  camera rides on the wrist and views it from many angles. Result is the camera
  pose relative to the end-effector, `T_gripper_cam`.

The math difference is only in how robot motion is fed to OpenCV; see
`run_hand_eye` in `eye_to_hand.py`. The `mount` field in
`real/config/ur5e_bench.json` records which is which.

## Prerequisites

- Dependencies from the `real` extra installed (`uv sync --extra real`):
  `opencv-python` (4.x — needs `cv2.calibrateHandEye`), `pyrealsense2`.
- A checkerboard with known geometry. Default assumes an 11x8 inner-corner grid
  (a 12x9-square board). **Measure the actual square edge length in metres** and
  pass it as `--square-size-m`; a wrong value scales every translation result.
- No process holding the RealSense devices. `realsense-viewer` grabbing a camera
  causes `Device or resource busy` during collection. Kill it first.

## Step 1 — Collect data (two separate batches)

Record with the normal teleop pipeline. Aim for **15–20+ poses** per batch with
large variation in *both* translation and rotation — degenerate motion (e.g.
pure translation) makes hand-eye ill-conditioned.

**Batch A — fixed cameras.** Rigidly attach the checkerboard to the EE. Teleop
the arm through varied poses, keeping the board visible to `d435i_fixed` and
`l515` simultaneously.

**Batch B — wrist camera.** Lay the checkerboard flat and fixed on the table.
Teleop so the wrist camera (`d435i_wrist`) views it from many positions/angles.

Note each batch's session date/id — you pass it as `--match` below.

## Step 2 — Build calibration datasets

Run from the local teleop checkout. This extracts frames
from every camera video and pairs each with the proprio pose at the same frame
index (frame `i` of every video is count-aligned with `data[i]`).

```bash
cd "$TELEOP_REPO"

# Batch A -> fixed cameras only
"$OPENETA_REPO/.venv/bin/python" tool/prepare_multiview_eye_to_hand_dataset.py \
  --source-root <your recordings root> \
  --match <batch-A session date> \
  --output-dir artifacts/calib_fixed \
  --cameras d435i_fixed,l515 \
  --frame-stride 10

# Batch B -> wrist camera only
"$OPENETA_REPO/.venv/bin/python" tool/prepare_multiview_eye_to_hand_dataset.py \
  --source-root <your recordings root> \
  --match <batch-B session date> \
  --output-dir artifacts/calib_wrist \
  --cameras d435i_wrist \
  --frame-stride 10
```

Each output dir gets one subdirectory per camera containing
`<sample>_frame.png` + `<sample>_proprio.json` pairs, plus `_manifest.json`.
Lower `--frame-stride` yields more samples; `--max-frames-per-session` caps them.

## Step 3 — Run calibration

Run from the local OpenETA checkout. Intrinsics are read automatically from the
RealSense SDK, matched by each camera's serial in `--config`. Set
`--inner-corners` and `--square-size-m` to your actual board.

```bash
cd "$OPENETA_REPO"

# Fixed cameras: eye-to-hand -> T_base_cam
./.venv/bin/python -m real.calibration.run_offline_calibration \
  --dataset-dir artifacts/calib_fixed \
  --config real/config/ur5e_bench.json \
  --inner-corners 11x8 --square-size-m 0.02 \
  --arm-key arm_left

# Wrist camera: eye-in-hand -> T_gripper_cam
./.venv/bin/python -m real.calibration.run_offline_calibration \
  --dataset-dir artifacts/calib_wrist \
  --config real/config/ur5e_bench.json \
  --inner-corners 11x8 --square-size-m 0.02 \
  --arm-key arm_left \
  --eye-in-hand d435i_wrist
```

Run **without** `--write-config` first and inspect the quality metrics. When
satisfied, re-run each with `--write-config` to patch the solved extrinsics into
`ur5e_bench.json` (a `.bak` is written first). Fixed cameras get
`extrinsics.T_base_cam`; the wrist camera gets `extrinsics.T_gripper_cam`.

## Step 4 — Read the quality metrics

Printed per camera and stored in `<dataset-dir>/eye_to_hand_report.json`:

- `detection_reprojection_error_px` — checkerboard corner fit. Sub-pixel to a
  few px is normal; large values mean wrong `--inner-corners` or bad focus.
- `pose_consistency_translation_error_mm` / `..._rotation_error_deg` — the
  primary accuracy signal: predicted vs observed board pose across samples.
  **Good:** translation mean within a few mm, rotation within ~1°. Large or
  high-variance values mean too few/degenerate poses, a wrong `--square-size-m`,
  a board that moved when it shouldn't have (or was rigid when it shouldn't),
  or the wrong `--eye-in-hand` assignment.
- `counts` — how many samples survived. Two-pass outlier rejection drops
  inconsistent samples; if too many are rejected, collect more/better poses.

## Troubleshooting

- **`only 0 valid detections`** — board not detected. Check `--inner-corners`
  (inner corners = squares − 1), lighting/focus, and that the board is actually
  in frame. Verify the dataset is the calibration recording, not task data.
- **`Device or resource busy` while collecting** — `realsense-viewer` or another
  process holds the camera. Close it.
- **Wrong/huge translation** — almost always a wrong `--square-size-m` or the
  wrong mode (a wrist camera solved as eye-to-hand). Confirm `--eye-in-hand`.
- **High rotation error only** — usually insufficient rotational variety in the
  collected poses.

## Notes

- `--method` selects the OpenCV solver (`TSAI`, `PARK`, `HORAUD`, `ANDREFF`,
  `DANIILIDIS`); `PARK` is the default and a good general choice.
- `--arm-key` selects which arm's proprio to read (`arm_left` here).
- The core solver is mode-aware and covered by synthetic round-trip tests: both
  eye-to-hand and eye-in-hand recover ground truth to ~0 error.
