from dataclasses import MISSING

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from ..gelsight_simulator_cfg import GelSightSimulatorCfg

# from .fots_marker_sim_frame_transformer import FOTSMarkerFrameTransformerSimulator
from .fots_marker_sim import FOTSMarkerSimulator

"""Configuration for a tactile RGB simulation with Taxim."""


@configclass
class FOTSMarkerSimulatorCfg(GelSightSimulatorCfg):
    simulation_approach_class: type = FOTSMarkerSimulator

    calib_folder_path: str = ""

    device: str = None

    with_shadow: bool = False

    tactile_img_res: tuple = (240, 320)
    """Resolution of the Tactile Image.

    Can be different from the Sensor Camera.
    If this is the case, then height map from camera is going to be up/down sampled.
    """

    lamb: list[float] = []
    """Parameters for exponential functions used by FOTS for marker simulation"""

    # experimental params
    ball_radius = 4.70 / 2  # mm
    mm_to_pixel = 19.58  # units = pix/mm

    # optical simulation params
    pyramid_kernel_size: list[int] = []
    kernel_size: int = 0

    @configclass
    class MarkerParams:
        """Dimensions here are in mm (we assume that the world units are meters)"""

        num_markers_col: int = 11
        num_markers_row: int = 9
        num_markers: int = 99
        x0: float = 15.0
        y0: float = 26.0
        dx: float = 26.0
        dy: float = 29.0

    marker_params: MarkerParams = MarkerParams()

    init_marker_pos: tuple = ([[]], [[]])
    """Initial Marker positions.

    Tuple (xx_init pos, yy_init pos):
    - xx_init = initial position of each marker along the "height" of the tactile img (top-down)
        -> for each marker the initial x pos. Shape: (num_markers_row, num_marker_column)
    - yy_init = initial position of each marker along the "width" of the tactile img (left-right)
        -> for each marker the initial y pos. Shape: (num_markers_row, num_marker_column)
    """
    # gelpad_height: float = MISSING
    # """Used for computing indentation depth from height map"""

    # #### Camera Data
    # gelpad_to_camera_min_distance: float = MISSING
    # """Min distance of camera to the gelpad.
    # Used for computing the indentation depth out of the
    # camera height map.
    # """

    frame_transformer_cfg: FrameTransformerCfg = MISSING
