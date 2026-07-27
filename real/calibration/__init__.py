"""Eye-to-hand calibration for fixed (third-person) cameras.

Solves ``T_base_cam`` for cameras observing a checkerboard mounted on the robot
end-effector, from teleop-recorded datasets. See run_offline_calibration.py.
"""

from real.calibration.eye_to_hand import (
    EYE_IN_HAND,
    EYE_TO_HAND,
    CalibrationResult,
    CheckerboardSpec,
    Sample,
    calibrate_camera,
)

__all__ = [
    "CalibrationResult",
    "CheckerboardSpec",
    "Sample",
    "calibrate_camera",
    "EYE_TO_HAND",
    "EYE_IN_HAND",
]
