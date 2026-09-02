# UniVTAC Isaac 5.1 compatibility boundary

This document records the static boundary between the FTP-1-era UniVTAC stack and the pinned Isaac 5.1 migration. It does not report a simulator run.

## Pinned sources

- Legacy paper-parity source: `michaelyuancb/ftp1-policy@89fa681d6c014cce28300946b7526db808e0b1c1`, task root `UniVTAC/`.
- Current official source: `univtac/UniVTAC@371fac67917307026be8f00869fcc1b61c623a9f`.

The audit uses a detached checkout for the Isaac 5.1 source. It never substitutes the moving `isaac51` branch head for the pinned commit.

## What changed

Isaac 5.1 is a new benchmark implementation boundary, not a transparent runtime upgrade. The pinned source changes Python 3.10 to 3.11, Isaac Sim 4.5 to 5.1, Isaac Lab 2.1.1 to 2.3.0, CUDA 12.4 to 12.6, and GCC 11 to GCC 12. It pins TacEx, libuipc, muda, SymEigen, cuRobo, and vcpkg revisions that the legacy installation path leaves implicit.

The six FTP-1 task modules remain present and preserve their high-level goals. Their executed semantics nevertheless change. Static scene actors move from high density to explicit kinematic motion, several initial heights change, Pull Out Key composes reset rotations differently, and adaptive grasp moves from camera-distance thresholds to positive indentation depth. Isaac 5.1 also changes reset stabilization, camera resolution, render synchronization, and evaluation decimation. Insert Hole and Insert Tube retain both `place_actor` calls, their `constraint_pose`, target construction, and task-level success thresholds.

The tactile interface still exposes GelSight Mini `rgb_marker` as HWC `uint8` in the 0–255 range. Isaac 5.1 adds `press_depth` as a separate positive-indentation field while retaining legacy raw camera distance under `depth`. That distinction supports new E1/E2/E3 experiments after runtime validation, but it prevents blind reuse of depth-based preprocessing.

## RTX 5090 boundary

The README claims RTX 40- and 50-series support. The installer defaults to `UNIVTAC_CUDA_ARCH=89`, and the installation guide documents `89` or `8.9` for the RTX 40-series phase-one machine. The pinned public source does not name RTX 5090, `sm_120`, `compute_120`, Blackwell, or an explicit PTX-forward-compatibility policy. We therefore classify the RTX 5090 recipe as `claimed_but_not_fully_documented`, not unsupported.

## Benchmark interpretation

Within one pinned and validated Isaac 5.1 environment, comparing OpenETA E0/E1/E2/E3 remains a valid experimental design because every variant shares the same physics, sensor, action, and evaluator implementation. Those results must be labeled `UniVTAC-Isaac51`.

Cross-version numeric comparability is not preserved. The official README declares Isaac 4.5 and 5.1 data non-cross-compatible, and the source changes control, physics, sensor timing, and parts of the success or early-stop path. An Isaac 5.1 success rate cannot serve as a direct reproduction of the FTP-1 paper's Isaac 4.5 number.

We therefore keep two tracks. Track A uses a separately installed, pinned Isaac 5.1 environment for the current OpenETA research after an isolated smoke test. Track B retains the frozen FTP-1 stack on an author-confirmed compatible GPU/runtime for legacy checkpoint and paper-number parity. Track B does not block the within-version causal comparisons in Track A.

## Next gate

The next step is `clarify_rtx5090_build_recipe_then_isolated_smoke`. Before installing anything, confirm the RTX 5090 CUDA architecture recipe and the six-task validation scope with the UniVTAC authors. Any Isaac 5.1 environment must be new and isolated; it must not overwrite the legacy environment.
