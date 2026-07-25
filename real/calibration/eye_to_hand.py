"""Eye-to-hand calibration core: checkerboard detection + cv2.calibrateHandEye.

This is an *eye-to-hand* configuration: a fixed camera observes a checkerboard
mounted on the robot end-effector. Per sample we know:

    T_base_ee   (robot proprio, forward kinematics)
    T_cam_board (from checkerboard solvePnP)

The unknown constants are:

    T_base_cam  (fixed camera pose in the robot base frame) -- what we want
    T_ee_board  (board pose relative to the end-effector)

We solve with ``cv2.calibrateHandEye`` by feeding the inverse robot pose
(``T_ee_base = inv(T_base_ee)``) as the gripper-to-base motion and the board
pose ``T_cam_board`` as the target-to-camera motion. With that mapping the
returned transform numerically corresponds to ``T_base_cam``.

Adapted from a teleoperation project's offline eye-to-hand calibration helper,
with ArUco single-marker detection replaced by checkerboard detection and the
solver reworked to operate on in-memory sample lists so it can serve multiple
cameras from one dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from real.calibration.geometry import (
    invert_transform,
    mean_transform,
    rotation_error_deg,
    transform_from_rt,
)


HAND_EYE_METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class CheckerboardSpec:
    """Checkerboard geometry.

    ``inner_corners`` is (cols, rows) of *inner* corners, i.e. (squares-1).
    A board of 12x9 squares has inner_corners (11, 8). ``square_size_m`` is the
    side length of one square in meters.
    """

    inner_corners_x: int = 11
    inner_corners_y: int = 8
    square_size_m: float = 0.02

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return (self.inner_corners_x, self.inner_corners_y)

    def object_points(self) -> np.ndarray:
        """Board-frame 3D coordinates of every inner corner (row-major)."""
        cols, rows = self.pattern_size
        grid = np.zeros((rows * cols, 3), dtype=np.float64)
        grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        grid *= self.square_size_m
        return grid


@dataclass
class Sample:
    """One (image, robot-pose) observation for a single camera."""

    sample_id: str
    T_base_ee: np.ndarray
    frame_id: str = ""
    image: Optional[np.ndarray] = None
    T_cam_board: Optional[np.ndarray] = None
    reprojection_error_px: Optional[float] = None
    reject_reason: Optional[str] = None


def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def detect_checkerboard_pose(
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    board: CheckerboardSpec,
) -> Tuple[np.ndarray, float]:
    """Detect the checkerboard and return (T_cam_board, reprojection_error_px).

    Raises ValueError with a machine-readable reason on failure so the caller
    can record it as a per-sample reject reason.
    """
    if image is None:
        raise ValueError("failed_to_load_image")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # findChessboardCornersSB gives a direction-consistent, sub-pixel corner
    # ordering. The legacy findChessboardCorners flips the board orientation
    # per-frame on wide, board-fills-frame views (e.g. the fixed front camera),
    # which mis-pairs image/object points and injects a systematic hand-eye
    # error (reprojection 3-6px). SB keeps reprojection ~0.3px.
    found, corners = cv2.findChessboardCornersSB(
        gray, board.pattern_size, cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise ValueError("checkerboard_not_found")

    # SB already returns sub-pixel corners; no cornerSubPix refinement needed.
    corners = np.asarray(corners, dtype=np.float64).reshape(-1, 2)

    object_points = board.object_points()
    success, rvec, tvec = cv2.solvePnP(
        object_points, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise ValueError("solvepnp_failed")

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    projected = projected.reshape(-1, 2)
    reproj_error = float(np.mean(np.linalg.norm(projected - corners, axis=1)))

    R_cam_board, _ = cv2.Rodrigues(rvec)
    T_cam_board = transform_from_rt(R_cam_board, tvec.reshape(3))
    return T_cam_board, reproj_error


EYE_TO_HAND = "eye_to_hand"
EYE_IN_HAND = "eye_in_hand"


def run_hand_eye(
    samples: Sequence[Sample], method_name: str, mode: str = EYE_TO_HAND
) -> np.ndarray:
    """Solve the camera transform from valid samples via cv2.calibrateHandEye.

    Returns ``T_base_cam`` for ``eye_to_hand`` (fixed camera, board on gripper)
    or ``T_gripper_cam`` for ``eye_in_hand`` (camera on gripper, board fixed).

    The only numeric difference is the robot-motion input fed to OpenCV:
      * eye_to_hand: feed ``T_ee_base = inv(T_base_ee)``; the returned transform
        is interpreted as ``T_base_cam``.
      * eye_in_hand: feed ``T_base_ee`` directly; OpenCV returns the standard
        cam-to-gripper transform, i.e. ``T_gripper_cam``.
    """
    if mode not in (EYE_TO_HAND, EYE_IN_HAND):
        raise ValueError(f"unknown calibration mode: {mode}")

    valid = [s for s in samples if s.reject_reason is None and s.T_cam_board is not None]
    if len(valid) < 3:
        raise ValueError("Need at least 3 valid samples for hand-eye calibration.")

    R_input, t_input = [], []
    R_target2cam, t_target2cam = [], []
    for sample in valid:
        T_robot = (
            invert_transform(sample.T_base_ee)
            if mode == EYE_TO_HAND
            else sample.T_base_ee
        )
        R_input.append(T_robot[:3, :3])
        t_input.append(T_robot[:3, 3].reshape(3, 1))
        R_target2cam.append(sample.T_cam_board[:3, :3])
        t_target2cam.append(sample.T_cam_board[:3, 3].reshape(3, 1))

    R_out, t_out = cv2.calibrateHandEye(
        R_input, t_input, R_target2cam, t_target2cam,
        method=HAND_EYE_METHODS[method_name],
    )
    return transform_from_rt(
        np.asarray(R_out, dtype=np.float64),
        np.asarray(t_out, dtype=np.float64).reshape(3),
    )


def estimate_board_invariant(
    samples: Sequence[Sample], T_cam: np.ndarray, mode: str = EYE_TO_HAND
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Estimate the fixed board transform and per-sample deviation from its mean.

    Returns ``T_ee_board`` (board relative to end-effector) for eye_to_hand,
    or ``T_base_board`` (board fixed in the base frame) for eye_in_hand. Both
    are constant across samples if the calibration is consistent, so the spread
    of the per-sample estimate is a direct quality signal.
    """
    valid = [s for s in samples if s.reject_reason is None and s.T_cam_board is not None]
    if mode == EYE_TO_HAND:
        # T_ee_board = inv(T_base_ee) @ T_base_cam @ T_cam_board
        implied = [
            invert_transform(s.T_base_ee) @ T_cam @ s.T_cam_board for s in valid
        ]
    else:
        # eye_in_hand: T_base_board = T_base_ee @ T_gripper_cam @ T_cam_board
        implied = [s.T_base_ee @ T_cam @ s.T_cam_board for s in valid]
    T_mean = mean_transform(implied)

    deviations = []
    for sample, T_i in zip(valid, implied):
        deviations.append(
            {
                "sample_id": sample.sample_id,
                "translation_error_m": float(
                    np.linalg.norm(T_i[:3, 3] - T_mean[:3, 3])
                ),
                "rotation_error_deg": rotation_error_deg(
                    T_mean[:3, :3], T_i[:3, :3]
                ),
            }
        )
    return T_mean, deviations


def apply_rejection_rules(
    samples: List[Sample],
    max_reprojection_px: float,
    max_consistency_translation_mm: float,
    max_consistency_rotation_deg: float,
    deviations: Sequence[Dict[str, float]],
) -> None:
    """Flag outlier samples in place based on reprojection and consistency."""
    dev_by_id = {item["sample_id"]: item for item in deviations}
    for sample in samples:
        if sample.reject_reason is not None:
            continue
        if (
            sample.reprojection_error_px is not None
            and sample.reprojection_error_px > max_reprojection_px
        ):
            sample.reject_reason = (
                f"reprojection_error_{sample.reprojection_error_px:.3f}"
                f"_gt_{max_reprojection_px:.3f}"
            )
            continue
        dev = dev_by_id.get(sample.sample_id)
        if dev is None:
            sample.reject_reason = "missing_consistency"
            continue
        if dev["translation_error_m"] * 1000.0 > max_consistency_translation_mm:
            sample.reject_reason = (
                f"consistency_translation_{dev['translation_error_m'] * 1000.0:.3f}"
                f"_mm_gt_{max_consistency_translation_mm:.3f}_mm"
            )
            continue
        if dev["rotation_error_deg"] > max_consistency_rotation_deg:
            sample.reject_reason = (
                f"consistency_rotation_{dev['rotation_error_deg']:.3f}"
                f"_deg_gt_{max_consistency_rotation_deg:.3f}_deg"
            )


def compute_pose_consistency(
    samples: Sequence[Sample], T_cam: np.ndarray, T_board: np.ndarray, mode: str = EYE_TO_HAND
) -> List[Dict[str, float]]:
    """Per-sample prediction error: predicted vs observed T_cam_board.

    For eye_to_hand, T_cam is T_base_cam and T_board is T_ee_board.
    For eye_in_hand, T_cam is T_gripper_cam and T_board is T_base_board.
    """
    metrics = []
    for sample in samples:
        if sample.reject_reason is not None or sample.T_cam_board is None:
            continue
        if mode == EYE_TO_HAND:
            # T_cam_board = inv(T_base_cam) @ T_base_ee @ T_ee_board
            T_pred = invert_transform(T_cam) @ sample.T_base_ee @ T_board
        else:
            # eye_in_hand: T_cam_board = inv(T_gripper_cam) @ inv(T_base_ee) @ T_base_board
            T_pred = invert_transform(T_cam) @ invert_transform(sample.T_base_ee) @ T_board
        metrics.append(
            {
                "sample_id": sample.sample_id,
                "translation_error_m": float(
                    np.linalg.norm(T_pred[:3, 3] - sample.T_cam_board[:3, 3])
                ),
                "rotation_error_deg": rotation_error_deg(
                    sample.T_cam_board[:3, :3], T_pred[:3, :3]
                ),
                "reprojection_error_px": sample.reprojection_error_px,
            }
        )
    return metrics


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "max": float(np.max(arr)),
    }


def transform_to_jsonable(T: np.ndarray) -> Dict[str, list]:
    return {
        "matrix": np.asarray(T, dtype=float).round(12).tolist(),
        "translation_m": np.asarray(T[:3, 3], dtype=float).round(12).tolist(),
        "rotation_matrix": np.asarray(T[:3, :3], dtype=float).round(12).tolist(),
    }


@dataclass
class CalibrationResult:
    """Result of calibrating one camera.

    ``T_cam`` is the solved camera transform: ``T_base_cam`` for eye_to_hand
    (fixed camera) or ``T_gripper_cam`` for eye_in_hand (wrist camera).
    ``T_board`` is the corresponding fixed board transform (``T_ee_board`` or
    ``T_base_board`` respectively).
    """

    camera_name: str
    mode: str
    T_cam: np.ndarray
    T_board: np.ndarray
    samples: List[Sample] = field(default_factory=list)
    report: Dict = field(default_factory=dict)


def calibrate_camera(
    camera_name: str,
    samples: List[Sample],
    K: np.ndarray,
    D: np.ndarray,
    board: CheckerboardSpec,
    method: str = "PARK",
    mode: str = EYE_TO_HAND,
    max_reprojection_px: float = 3.0,
    max_consistency_translation_mm: float = 40.0,
    max_consistency_rotation_deg: float = 6.0,
    expected_frame_id: Optional[str] = "base_link",
) -> CalibrationResult:
    """Full pipeline for one camera: detect, solve, reject outliers, re-solve.

    ``mode`` selects EYE_TO_HAND (fixed camera, board on gripper -> T_base_cam)
    or EYE_IN_HAND (wrist camera, board fixed in scene -> T_gripper_cam).

    Runs a two-pass solve: an initial hand-eye estimate is used to flag
    outliers, then the calibration is recomputed on the surviving samples.
    """
    cam_key = "T_base_cam" if mode == EYE_TO_HAND else "T_gripper_cam"
    board_key = "T_ee_board" if mode == EYE_TO_HAND else "T_base_board"

    # Detect the board in every sample image.
    for sample in samples:
        if sample.reject_reason is not None:
            continue
        if expected_frame_id is not None and sample.frame_id != expected_frame_id:
            sample.reject_reason = f"unexpected_ee_frame_{sample.frame_id}"
            continue
        try:
            T_cam_board, reproj = detect_checkerboard_pose(sample.image, K, D, board)
            sample.T_cam_board = T_cam_board
            sample.reprojection_error_px = reproj
        except ValueError as exc:
            sample.reject_reason = str(exc)

    initially_valid = [s for s in samples if s.reject_reason is None]
    if len(initially_valid) < 3:
        raise ValueError(
            f"[{camera_name}] only {len(initially_valid)} valid detections; need >= 3."
        )

    # First pass: estimate and flag outliers.
    T_cam_initial = run_hand_eye(samples, method, mode=mode)
    _, deviations = estimate_board_invariant(samples, T_cam_initial, mode=mode)
    apply_rejection_rules(
        samples,
        max_reprojection_px,
        max_consistency_translation_mm,
        max_consistency_rotation_deg,
        deviations,
    )

    if len([s for s in samples if s.reject_reason is None]) < 3:
        raise ValueError(f"[{camera_name}] outlier rejection left fewer than 3 samples.")

    # Second pass on surviving samples.
    T_cam = run_hand_eye(samples, method, mode=mode)
    T_board, deviations_final = estimate_board_invariant(samples, T_cam, mode=mode)
    pose_consistency = compute_pose_consistency(samples, T_cam, T_board, mode=mode)

    valid = [s for s in samples if s.reject_reason is None]
    rejected = [s for s in samples if s.reject_reason is not None]

    report = {
        "camera_name": camera_name,
        "mode": mode,
        "camera_intrinsics": {"K": K.round(12).tolist(), "D": D.reshape(-1).round(12).tolist()},
        "checkerboard": {
            "inner_corners": list(board.pattern_size),
            "square_size_m": board.square_size_m,
        },
        "method": {"opencv_function": "cv2.calibrateHandEye", "method": method},
        "counts": {
            "total_samples": len(samples),
            "initially_valid_samples": len(initially_valid),
            "final_valid_samples": len(valid),
            "rejected_samples": len(rejected),
        },
        "results": {
            cam_key: transform_to_jsonable(T_cam),
            board_key: transform_to_jsonable(T_board),
            f"{cam_key}_initial": transform_to_jsonable(T_cam_initial),
        },
        "quality_metrics": {
            "detection_reprojection_error_px": summarize_metric(
                [s.reprojection_error_px for s in valid if s.reprojection_error_px is not None]
            ),
            f"implied_{board_key}_translation_error_mm": summarize_metric(
                [d["translation_error_m"] * 1000.0 for d in deviations_final]
            ),
            f"implied_{board_key}_rotation_error_deg": summarize_metric(
                [d["rotation_error_deg"] for d in deviations_final]
            ),
            "pose_consistency_translation_error_mm": summarize_metric(
                [m["translation_error_m"] * 1000.0 for m in pose_consistency]
            ),
            "pose_consistency_rotation_error_deg": summarize_metric(
                [m["rotation_error_deg"] for m in pose_consistency]
            ),
        },
        "rejected_samples": [
            {"sample_id": s.sample_id, "reject_reason": s.reject_reason} for s in rejected
        ],
        "per_sample_pose_consistency": pose_consistency,
    }

    return CalibrationResult(
        camera_name=camera_name,
        mode=mode,
        T_cam=T_cam,
        T_board=T_board,
        samples=samples,
        report=report,
    )
