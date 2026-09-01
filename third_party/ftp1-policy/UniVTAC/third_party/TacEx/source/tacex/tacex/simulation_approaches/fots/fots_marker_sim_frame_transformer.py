from __future__ import annotations

import copy
import math
import numpy as np
import torch
from typing import TYPE_CHECKING

import cv2
import omni
import torchvision.transforms.functional as F

from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import euler_xyz_from_quat

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from ..gpu_taxim import TaximSimulator
from ..gpu_taxim.sim import TaximTorch
from .sim import MarkerMotion

if TYPE_CHECKING:
    from .fots_marker_sim_cfg import FOTSMarkerSimulatorCfg


class FOTSMarkerFrameTransformerSimulator(GelSightSimulator):
    """Wraps around the FOTS simulation for marker simulation of GelSight sensors inside Isaac Sim.

    The class uses an instance of the gpu_taxim simulator for generating the deformed height map.
    """

    cfg: FOTSMarkerSimulatorCfg

    def __init__(self, sensor: GelSightSensor, cfg: FOTSMarkerSimulatorCfg):
        self.sensor: GelSightSensor = sensor

        super().__init__(sensor=sensor, cfg=cfg)

        # use IsaacLab FrameTransformer for keeping track of relative position/rotation
        self.frame_transformer: FrameTransformer = FrameTransformer(self.cfg.frame_transformer_cfg)

    def _initialize_impl(self):
        if self.cfg.device is None:
            # use same device as simulation
            self._device = self.sensor.device
        else:
            self._device = self.cfg.device

        self._num_envs = self.sensor._num_envs

        # todo make size adaptable? I mean with env_ids. This way we would always simulate everything
        self._indentation_depth = torch.zeros((self.sensor._num_envs), device=self.sensor._device)
        """Indentation depth, i.e. how deep the object is pressed into the gelpad.
        Values are in mm.

        Indentation depth is equal to the maximum pressing depth of the object in the gelpad.
        It is used for shifting the height map for the Taxim simulation.
        """

        # use Taxim for gpu based operations
        if (self.sensor.optical_simulator is not None) and (type(self.sensor.optical_simulator) is TaximSimulator):
            self._taxim: TaximTorch = self.sensor.optical_simulator._taxim
        else:
            raise RuntimeError(
                "Currently FOTS simulation approach has to be used in combination with GPU-Taxim as optical-simulator."
            )

        # tactile rgb image without indentation
        bg_img = self._taxim.background_img.movedim(0, 2).cpu().numpy()
        self.marker_motion_sim = MarkerMotion(
            frame0_blur=bg_img,
            mm2pix=self.cfg.mm_to_pixel,
            num_markers_col=self.cfg.marker_params.num_markers_col,  # 20, #11
            num_markers_row=self.cfg.marker_params.num_markers_row,  # 15, #9
            tactile_img_width=self.cfg.tactile_img_res[0],
            tactile_img_height=self.cfg.tactile_img_res[1],
            lamb=[0.00125, 0.0021, 0.0038],  # [0.00125,0.00021,0.00038],
            x0=self.cfg.marker_params.x0,
            y0=self.cfg.marker_params.y0,
        )

        self.init_marker_pos = np.stack(
            (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos), axis=-1
        )
        self.init_marker_pos = self.init_marker_pos.reshape(-1, 2)
        # if camera resolution is different than the tactile RGB res, scale img
        self.img_res = self.cfg.tactile_img_res

        # create buffers
        self.marker_data = torch.zeros(
            (self.sensor._num_envs, 2, self.cfg.marker_params.num_markers, 2), device=self._device
        )
        """Marker flow data. Shape is [num_envs, 2, num_markers, 2]

        dim=1: [initial, current] marker positions
        dim=3: [x,y] values of the markers in the image of the sensor.
        """
        # set initial marker pos
        self.marker_data[:, 0] = torch.tensor(self.init_marker_pos, device=self._device)

        self.sensor._data.output["traj"] = []
        for _ in range(self.sensor._num_envs):
            self.sensor._data.output["traj"].append([])
        self.theta = torch.zeros((self.sensor._num_envs), device=self._device)

        # need to initialize manually
        self.frame_transformer._initialize_impl()
        self.frame_transformer._is_initialized = True
        print("Frame transformer for FOTS: ", self.frame_transformer)

        # for visualization of the markers
        self.patch_array_dict = copy.deepcopy(generate_patch_array())

    def marker_motion_simulation(self):
        self._indentation_depth = self.sensor._indentation_depth
        height_map = self.sensor._data.output[
            "height_map"
        ]  # height map has shape (height, width) cause row-column format

        # up/downscale height map if camera res different than tactile img res
        if (height_map.shape[1], height_map.shape[2]) != (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]):
            height_map = F.resize(height_map, (self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]))

        if self._device == "cpu":
            height_map = height_map.cpu()
            self._indentation_depth = self.sensor._indentation_depth.cpu()

        height_map_shifted = self._taxim._TaximTorch__get_shifted_height_map(self._indentation_depth, height_map)
        deformed_gel, contact_mask = self._taxim._TaximTorch__compute_gel_pad_deformation(height_map_shifted)
        deformed_gel = deformed_gel.max() - deformed_gel

        for env_id in range(deformed_gel.shape[0]):
            if self._indentation_depth[env_id].item() > 0.0:
                # compute contact center based on contact_mask
                contact_points = torch.argwhere(contact_mask[env_id])
                mean = torch.mean(contact_points.float(), dim=0).cpu().numpy()
                # print("should be pix ", mean[1], mean[0])
                # rows = height = y values
                mean[0] = (mean[0] - self.marker_motion_sim.tactile_img_height / 2) / self.marker_motion_sim.mm2pix
                # columns = width = x values
                mean[1] = (mean[1] - self.marker_motion_sim.tactile_img_width / 2) / self.marker_motion_sim.mm2pix

                # rel position/orientation of obj to sensor
                self.frame_transformer.update(dt=0.001)
                # rel_pos = self.frame_transformer.data.target_pos_source.cpu().numpy()[env_id, 0, :] # target_pos_source shape is (num_envs, num_targets, 3)
                # rel_pos *= 1000 # convert to mm

                # print("rel_pos in pix ", self.cfg.mm_to_pixel*rel_pos[0] + self.marker_motion_sim.tactile_img_width/2, self.cfg.mm_to_pixel*rel_pos[1] + self.marker_motion_sim.tactile_img_height/2)
                # print("rel_pos ", rel_pos)
                rel_orient = self.frame_transformer.data.target_quat_source[env_id]
                roll, pitch, yaw = euler_xyz_from_quat(rel_orient)

                theta = yaw.cpu().numpy()[0]  # -> only use the first target_frame #TODO fix that?
                print("rel yaw in deg ", np.rad2deg(theta))

                # order of traj depends on the source frame. With our camera placement, we need [-x,-y,theta]
                # self.sensor._data.output["traj"][env_id].append([-rel_pos[0], -rel_pos[1], theta])
                print("")

                # traj takes [x,y,theta] values
                self.sensor._data.output["traj"][env_id].append([mean[1], mean[0], theta])

                # todo vectorize marker_sim method with pytorch
                marker_x_pos, marker_y_pos = self.marker_motion_sim.marker_sim(
                    deformed_gel[env_id].cpu().numpy(),
                    contact_mask[env_id].cpu().numpy(),
                    self.sensor._data.output["traj"][env_id],
                )
            else:
                self.sensor._data.output["traj"][env_id] = []
                marker_x_pos = self.marker_motion_sim.init_marker_x_pos
                marker_y_pos = self.marker_motion_sim.init_marker_y_pos

            marker_pos = np.stack((marker_x_pos, marker_y_pos), axis=-1).reshape(-1, 2)
            self.marker_data[env_id, 1] = torch.tensor(marker_pos, device=self._device)

        return self.marker_data

    # def compute_indentation_depth(self):
    #     self.height_map = self.height_map / 1000 # convert height map from mm to meter
    #     min_distance_obj = self.height_map.amin((1,2))
    #     # smallest distance between object and sensor case
    #     dist_obj_sensor_case = min_distance_obj - self.cfg.gelpad_to_camera_min_distance

    #     # print("dist_obj_sensor_case", dist_obj_sensor_case)
    #     # if (dist_obj_sensor_case < 0):  # object is "inside the sensor", cause the object is closer to the camera than the edge of the sensor
    #     #     # print("Object is inside the sensor!!! Gelpad would be broken!!!")
    #     #     dist_obj_sensor_case = 0
    #     dist_obj_sensor_case = torch.where(dist_obj_sensor_case < 0, 0, dist_obj_sensor_case)

    #     self._indentation_depth[:] = torch.where(
    #         dist_obj_sensor_case <= self.cfg.gelpad_height,
    #         (self.cfg.gelpad_height - dist_obj_sensor_case)*1000,
    #         0
    #     )

    #     return self._indentation_depth

    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        self.init_marker_pos = (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos)

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Creates an USD attribute for the sensor asset, which can visualize the tactile image.

        Select the GelSight sensor case whose output you want to see in the Isaac Sim GUI,
        i.e. the `gelsight_mini_case` Xform (not the mesh!).
        Scroll down in the properties panel to "Raw Usd Properties" and click "Extra Properties".
        There is an attribute called "show_tactile_image".
        Toggle it on to show the sensor output in the GUI.

        If only optical simulation is used, then only an optical img is displayed.
        If only the marker simulatios is used, then only an image displaying the marker positions is displayed.
        If both, optical and marker simulation, are used, then the images are overlaid.
        """
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "_debug_windows"):
                # dict of windows that show the simulated tactile images, if the attribute of the sensor asset is turned on
                self._debug_windows = {}
                self._debug_img_providers = {}
                # todo check if we can make implementation more efficient than dict of dicts
                if "marker_motion" in self.sensor.cfg.data_types:
                    self._debug_windows = {}
                    self._debug_img_providers = {}
        else:
            pass

    def _debug_vis_callback(self, event):
        if self.sensor._prim_view is None:
            return

        # Update the GUI windows_prim_view
        for i, prim in enumerate(self.sensor._prim_view.prims):
            if "marker_motion" in self.sensor.cfg.data_types:
                show_img = prim.GetAttribute("debug_marker_motion").Get()
                if show_img:
                    if str(i) not in self._debug_windows:
                        # create a window
                        window = omni.ui.Window(
                            self.sensor._prim_view.prim_paths[i] + "/fots_marker",
                            width=self.cfg.tactile_img_res[0],
                            height=self.cfg.tactile_img_res[1],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    marker_flow_i = self.sensor.data.output["marker_motion"][i]

                    frame = self._create_marker_img(marker_flow_i)

                    # draw current marker positions like ManiSkill-ViTac does
                    # frame = self.draw_markers(marker_flow_i[1].cpu().numpy(), img_w=self.cfg.tactile_img_res[0], img_h=self.cfg.tactile_img_res[1])

                    # create tactile rgb img with markers
                    if "tactile_rgb" in self.sensor.cfg.data_types:
                        if (
                            self.sensor.cfg.optical_sim_cfg.tactile_img_res
                            == self.sensor.cfg.marker_motion_sim_cfg.tactile_img_res
                        ):
                            # todo add upscaling of tactile_rgb, if not same size
                            tactile_rgb = self.sensor.data.output["tactile_rgb"][i].cpu().numpy() * 255
                            frame = tactile_rgb * np.dstack([frame.astype(np.float64) / 255] * 3)

                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
                        # self._img_providers[str(i)].set_data_array(frame, [width, height, channels]) #method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        self._debug_img_providers[str(i)].set_bytes_data(
                            frame.flatten().data, [width, height]
                        )  # method signature: (numpy.ndarray[numpy.uint8], (width, height))
                        omni.ui.ImageWithProvider(
                            self._debug_img_providers[str(i)]
                        )  # , fill_policy=omni.ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT -> fill_policy by default: specifying the width and height of the item causes the image to be scaled to that size
                elif str(i) in self._debug_windows:
                    # remove window/img_provider from dictionary and destroy them
                    self._debug_windows.pop(str(i)).destroy()
                    self._debug_img_providers.pop(str(i)).destroy()

    def _create_marker_img(self, marker_data):
        """Visualization of marker flow, like in the original FOTS simulation.

        Marker data needs to have the shape [2, num_markers, 2]
        - dim=0: init and current markers
        - dim=2: x and y values of the marker position

        Args:
            marker_data: marker flow data with shape [2, num_markers, 2]
        """
        # for visualization -> white background with black dots
        color = (0, 0, 0)
        arrow_scale = 1

        frame = np.ones((self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0])).astype(np.uint8)

        # marker data has shape [2, num_markers, 2], where first dim = init and current marker position
        init_marker_pos = marker_data[0].cpu().numpy()
        current_marker_pos = marker_data[1].cpu().numpy()

        num_markers = marker_data.shape[1]

        for marker_index in range(num_markers):
            init_x_pos = int(init_marker_pos[marker_index][0])
            init_y_pos = int(init_marker_pos[marker_index][1])

            x_pos = int(current_marker_pos[marker_index][0])
            y_pos = int(current_marker_pos[marker_index][1])

            if (x_pos >= frame.shape[1]) or (x_pos < 0) or (y_pos >= frame.shape[0]) or (y_pos < 0):
                continue
            # cv2.circle(frame,(init_y_pos,init_x_pos), 6, (255,255,255), 1, lineType=8)

            pt1 = (init_x_pos, init_y_pos)
            pt2 = (x_pos + arrow_scale * int(x_pos - init_x_pos), y_pos + arrow_scale * int(y_pos - init_y_pos))
            # pt2 = (x_pos, y_pos)
            cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)

        # draw current contact point
        # if len(self.sensor._data.output["traj"][0]) > 0:
        #     # traj = self.sensor._data.output["traj"][0]
        #     center_x = int(
        #         self.sensor._data.output["traj"][0][-1][0] * self.marker_motion_sim.mm2pix
        #         + self.marker_motion_sim.tactile_img_width / 2
        #     )
        #     center_y = int(
        #         self.sensor._data.output["traj"][0][-1][1] * self.marker_motion_sim.mm2pix
        #         + self.marker_motion_sim.tactile_img_height / 2
        #     )
        #     cv2.circle(frame, (center_x, center_y), 6, color, 1, lineType=8)

        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # frame = frame[:self.cfg.tactile_img_res[1], :self.cfg.tactile_img_res[0]]
        return frame

    def draw_markers(self, marker_uv: np.array, marker_size=3, img_w=320, img_h=240) -> np.array:
        """Visualize the marker flow like the ManiSkill-ViTac Simulator does.

        Reference:
        https://github.com/chuanyune/ManiSkill-ViTac2025/blob/a3d7df54bca9a2e57f34b37be3a3df36dc218915/Track_1/envs/tactile_sensor_sapienipc.py

        Args:
            marker_uv: Marker flow of a sensor. Shape is (2, num_markers, 2).
            marker_size: The size of the markers in the image. Defaults to 3.
            img_w: Width of the tactile image. Defaults to 320.
            img_h: Height of the tactile image. Defaults to 240.

        Returns:
            Image with the markers visualized as dots.
        """
        marker_uv_compensated = marker_uv + np.array([0.5, 0.5])

        marker_image = np.ones((img_h + 24, img_w + 24), dtype=np.uint8) * 255
        for i in range(marker_uv_compensated.shape[0]):
            uv = marker_uv_compensated[i]
            u = uv[0] + 12
            v = uv[1] + 12
            patch_id_u = math.floor((u - math.floor(u)) * self.patch_array_dict["super_resolution_ratio"])
            patch_id_v = math.floor((v - math.floor(v)) * self.patch_array_dict["super_resolution_ratio"])
            patch_id_w = math.floor(
                (marker_size - self.patch_array_dict["base_circle_radius"])
                * self.patch_array_dict["super_resolution_ratio"]
            )
            current_patch = self.patch_array_dict["patch_array"][patch_id_u, patch_id_v, patch_id_w]
            patch_coord_u = math.floor(u) - 6
            patch_coord_v = math.floor(v) - 6
            if marker_image.shape[1] - 12 > patch_coord_u >= 0 and marker_image.shape[0] - 12 > patch_coord_v >= 0:
                marker_image[
                    patch_coord_v : patch_coord_v + 12,
                    patch_coord_u : patch_coord_u + 12,
                ] = current_patch
        marker_image = marker_image[12:-12, 12:-12]

        return marker_image


def generate_patch_array(super_resolution_ratio=10):
    circle_radius = 3
    size_slot_num = 50
    base_circle_radius = 1.5

    patch_array = np.zeros(
        (
            super_resolution_ratio,
            super_resolution_ratio,
            size_slot_num,
            4 * circle_radius,
            4 * circle_radius,
        ),
        dtype=np.uint8,
    )
    for u in range(super_resolution_ratio):
        for v in range(super_resolution_ratio):
            for w in range(size_slot_num):
                img_highres = (
                    np.ones(
                        (
                            4 * circle_radius * super_resolution_ratio,
                            4 * circle_radius * super_resolution_ratio,
                        ),
                        dtype=np.uint8,
                    )
                    * 255
                )
                center = np.array(
                    [
                        circle_radius * super_resolution_ratio * 2,
                        circle_radius * super_resolution_ratio * 2,
                    ],
                    dtype=np.uint8,
                )
                center_offseted = center + np.array([u, v])
                radius = round(base_circle_radius * super_resolution_ratio + w)
                img_highres = cv2.circle(
                    img_highres,
                    tuple(center_offseted),
                    radius,
                    (0, 0, 0),
                    thickness=cv2.FILLED,
                    lineType=cv2.LINE_AA,
                )
                img_highres = cv2.GaussianBlur(img_highres, (17, 17), 15)
                img_lowres = cv2.resize(
                    img_highres,
                    (4 * circle_radius, 4 * circle_radius),
                    interpolation=cv2.INTER_CUBIC,
                )
                patch_array[u, v, w, ...] = img_lowres

    return {
        "base_circle_radius": base_circle_radius,
        "circle_radius": circle_radius,
        "size_slot_num": size_slot_num,
        "patch_array": patch_array,
        "super_resolution_ratio": super_resolution_ratio,
    }
