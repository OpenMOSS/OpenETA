from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg
from .mani_skill_sim import ManiSkillSimulator

"""Configuration for marker motion simulation via FEM simulation (like ManiSkill-ViTac used)."""

from typing import Literal

@configclass
class ManiSkillSimulatorCfg(GelSightSimulatorCfg):
    sensor_type:Literal['gsmini', 'xsensews', 'gf225'] = None

    simulation_approach_class: type = ManiSkillSimulator

    calib_folder_path: str = ""

    device: str = "cpu"

    marker_shape: tuple[int, int] = (7, 9)

    marker_interval: tuple[float, float] = (2.40625, 2.45833)

    marker_rotation_range: float = 0.0

    marker_translation_range: tuple[float, float] = (0.0, 0.0)

    marker_pos_shift_range: tuple[float, float] = (0.0, 0.0)

    marker_random_noise: float = 0.0

    marker_lose_tracking_probability: float = 0.0

    normalize: bool = False

    camera_to_surface: float = 0.0283,

    real_size: tuple[float, float] = (0.0266, 0.0209),

    tactile_img_res: tuple[int, int] = (320, 240)
    """Resolution of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is going to be up/down sampled.
    """
    
    sub_marker_num: int = 0

    marker_radius: float = 0.01 # in mm

    @configclass
    class MarkerParams:
        """Dimensions here are in mm (we assume that the world units are meters)"""

        # num_markers_col: int = 63
        # num_markers_row: int = 63
        num_markers: int = 63
        x0: float = 0
        y0: float = 0
        dx: float = 0
        dy: float = 0

    marker_params: MarkerParams = MarkerParams()

    init_marker_pos: tuple = ([[]], [[]])
    """Initial Marker positions.

    Tuple (xx_init pos, yy_init pos):
    - xx_init = initial position of each marker along the "height" of the tactile img (top-down)
        -> for each marker the initial x pos. Shape: (num_markers_row, num_marker_column)
    - yy_init = initial position of each marker along the "width" of the tactile img (left-right)
        -> for each marker the initial y pos. Shape: (num_markers_row, num_marker_column)
    """
