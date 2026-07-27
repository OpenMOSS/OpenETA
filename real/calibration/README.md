# Hand-Eye Calibration

The offline calibration tools solve camera transforms for fixed
eye-to-hand cameras and wrist-mounted eye-in-hand cameras using
`cv2.calibrateHandEye`.

Deployment calibration is private operational data. Do not commit camera
serials, recorded frames, controller addresses, or solved transforms. Keep
them in the ignored local file `real/config/ur5e_bench.json`.

## Prepare A Local Configuration

```bash
cp real/config/ur5e_bench.example.json real/config/ur5e_bench.json
```

Set each local camera's serial and `mount`:

- `fixed`: checkerboard attached to the end effector; solves `T_base_cam`.
- `wrist`: checkerboard fixed in the scene; solves `T_gripper_cam`.

Leave `extrinsics` empty until calibration succeeds.

## Dataset Layout

Create one directory per camera. Each sample consists of a synchronized image
and robot proprioception payload:

```text
calibration_dataset/
  fixed_camera/
    000001_frame.png
    000001_proprio.json
  wrist_camera/
    000001_frame.png
    000001_proprio.json
```

Collect at least 15 to 20 poses with substantial translation and rotation
variation. Pure translation or a narrow range of orientations produces an
ill-conditioned solve.

## Run Calibration

Run without `--write-config` first:

```bash
uv run python -m real.calibration.run_offline_calibration \
  --dataset-dir /path/to/calibration_dataset \
  --config real/config/ur5e_bench.json \
  --inner-corners 11x8 \
  --square-size-m 0.02
```

For wrist cameras, add their configured names:

```bash
uv run python -m real.calibration.run_offline_calibration \
  --dataset-dir /path/to/calibration_dataset \
  --config real/config/ur5e_bench.json \
  --inner-corners 11x8 \
  --square-size-m 0.02 \
  --eye-in-hand wrist_camera
```

Inspect reprojection, translation, and rotation consistency metrics. Re-run
with `--write-config` only after the result is acceptable. The command writes a
local backup before updating the ignored bench configuration.

## Troubleshooting

- No valid detections: verify inner-corner dimensions, lighting, focus, and
  board visibility.
- Large translation error: verify square size and unit conversion.
- Large rotation error: collect poses with greater rotational diversity.
- Inconsistent transform: confirm that fixed and wrist cameras use the correct
  calibration mode and that the checkerboard did not move unexpectedly.
