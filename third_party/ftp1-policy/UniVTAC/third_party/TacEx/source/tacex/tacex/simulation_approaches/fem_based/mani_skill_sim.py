from __future__ import annotations

import time
import copy
import math
from re import M
import numpy as np
import torch
from typing import TYPE_CHECKING

import cv2
import omni.usd

from tacex_uipc import UipcObject

from ...gelsight_sensor import GelSightSensor
from ..gelsight_simulator import GelSightSimulator
from .sim import VisionTactileSensorUIPC

if TYPE_CHECKING:
    from .mani_skill_sim_cfg import ManiSkillSimulatorCfg

class ManiSkillSimulator(GelSightSimulator):
    """Wrapper for ManiSkill-ViTac simulator for GelSight sensors.

    Instead of IPC, we use UIPC.
    The original ManiSkill-ViTac simulator can be found here https://github.com/chuanyune/ManiSkill-ViTac2025.git
    """

    cfg: ManiSkillSimulatorCfg
    patch_array = None

    def __init__(self, sensor: GelSightSensor, cfg: ManiSkillSimulatorCfg):
        self.sensor: GelSightSensor = sensor

        # needed for VisionTactileSensorUIPC class
        self.camera = None
        self.gelpad_uipc: UipcObject = self.sensor.gelpad_obj
        self.radius = cfg.marker_radius
        self.draw_patch_array()

        super().__init__(sensor=sensor, cfg=cfg)

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

        self.camera = self.sensor.camera
        self.marker_motion_sim: VisionTactileSensorUIPC = VisionTactileSensorUIPC(
            self.gelpad_uipc,
            self.camera,
            sensor_type=self.cfg.sensor_type,
            tactile_img_width=self.cfg.tactile_img_res[0],
            tactile_img_height=self.cfg.tactile_img_res[1],
            marker_shape=self.cfg.marker_shape,
            marker_interval=self.cfg.marker_interval,
            sub_marker_num=self.cfg.sub_marker_num,
            marker_radius=self.radius,
            num_markers=self.cfg.marker_params.num_markers,
        )

        self.marker_motion_sim._gen_marker_grid()

        # create buffers
        self.marker_data = torch.zeros(
            (self.sensor._num_envs, 2, self.cfg.marker_params.num_markers, 2), device=self._device
        )
        """Marker flow data. Shape is [num_envs, 2, num_markers, 2]

        dim=1: [initial, current] marker positions
        dim=3: [x,y] values of the markers
        """

    def marker_motion_simulation(self):
        marker_flow = self.marker_motion_sim.gen_marker_flow()
        # todo do it properly for multi env, currently marker flow has shape [2, num_markers, 2] and we want [num_envs, 2, num_markers, 2]
        self.marker_data[0] = marker_flow
        return self.marker_data

    def reset(self):
        self._indentation_depth = torch.zeros((self._num_envs), device=self._device)
        # self.init_marker_pos = (self.marker_motion_sim.init_marker_x_pos, self.marker_motion_sim.init_marker_y_pos)

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
                            self.sensor._prim_view.prim_paths[i] + "/fem_marker",
                            width=self.cfg.tactile_img_res[0],
                            height=self.cfg.tactile_img_res[1],
                        )
                        self._debug_windows[str(i)] = window
                        # create image provider
                        self._debug_img_providers[str(i)] = (
                            omni.ui.ByteImageProvider()
                        )  # default format omni.ui.TextureFormat.RGBA8_UNORM

                    tactile_rgb = self.sensor.data.output["tactile_rgb"][i] / 255.0
                    marker_motion = self.sensor.data.output["marker_motion"][0, 1]
                    marker_img = self.draw_markers(marker_uv=marker_motion)
                    tactile_rgb *= torch.dstack([marker_img / 255] * 3)
                    frame = (tactile_rgb * 255).to(dtype=torch.uint8).cpu().numpy()
                    # marker_flow_i = self.sensor.data.output["marker_motion"][i]

                    # frame = self._create_marker_img(marker_flow_i)
                    # draw current marker positions like ManiSkill-ViTac does
                    # frame = self.draw_markers(
                    #     marker_flow_i[1].cpu().numpy(),
                    #     img_w=self.cfg.tactile_img_res[0],
                    #     img_h=self.cfg.tactile_img_res[1],
                    # )

                    # create tactile rgb img with markers
                    # if "tactile_rgb" in self.sensor.cfg.data_types:
                    #     if (
                    #         self.sensor.cfg.optical_sim_cfg.tactile_img_res
                    #         == self.sensor.cfg.marker_motion_sim_cfg.tactile_img_res
                    #     ):
                    #         # todo add upscaling of tactile_rgb, if not same size
                    #         tactile_rgb = self.sensor.data.output["tactile_rgb"][i].cpu().numpy() * 255
                    #         frame = tactile_rgb * np.dstack([frame.astype(np.float64) / 255] * 3)

                    frame = frame.astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)

                    height, width, channels = frame.shape

                    with self._debug_windows[str(i)].frame:
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
        """Visualization of marker flow like in the original FOTS simulation.

        Marker data needs to have the shape [2, num_markers, 2]
        - dim=0: init and current markers
        - dim=2: x and y values of the marker position

        Args:
            marker_data: marker flow data with shape [2, num_markers, 2]
        """
        # for visualization -> white background with black dots
        color = (0, 0, 0)
        arrow_scale = 1  # 10 #0.0001 #0.25

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
            # cv2.circle(frame,(column,row), 6, (255,255,255), 1, lineType=8)

            pt1 = (init_x_pos, init_y_pos)
            pt2 = (x_pos + arrow_scale * int(x_pos - init_x_pos), y_pos + arrow_scale * int(y_pos - init_y_pos))

            cv2.arrowedLine(frame, pt1, pt2, color, 2, tipLength=0.2)

        frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        return frame

    def draw_patch_array(self, super_resolution_ratio:int=10):
        if ManiSkillSimulator.patch_array is not None:
            return
        
        circle_radius = self.radius
        size_slot_num = 50
        base_circle_radius = circle_radius * 0.5  # 与原始实现对齐: 1.5 / 3 = 0.5 比例
        blur_k, blur_sigma = (17, 17), 15 
        
        # patch 尺寸为 4 * circle_radius，用于覆盖 [-2r, 2r] 范围
        patch_size = 4 * circle_radius
        patch_array = np.zeros(
            (
                super_resolution_ratio,
                super_resolution_ratio,
                size_slot_num,
                patch_size,
                patch_size,
            ),
            dtype=np.uint8,
        )
        for u in range(super_resolution_ratio):
            for v in range(super_resolution_ratio):
                for w in range(size_slot_num):
                    # 高分辨率图像用于亚像素渲染
                    img_highres = (
                        np.ones(
                            (
                                patch_size * super_resolution_ratio,
                                patch_size * super_resolution_ratio,
                            ),
                            dtype=np.uint8,
                        )
                        * 255
                    )
                    # 圆心在高分辨率图像中心
                    center = np.array(
                        [
                            circle_radius * super_resolution_ratio * 2,
                            circle_radius * super_resolution_ratio * 2,
                        ],
                        dtype=np.int32,
                    )
                    # 亚像素偏移: u, v 控制圆心的亚像素位置
                    center_offseted = center + np.array([u, v])
                    # w 控制圆的大小变化
                    draw_radius = round(base_circle_radius * super_resolution_ratio + w)
                    img_highres = cv2.circle(
                        img_highres,
                        tuple(center_offseted),
                        draw_radius,
                        (0, 0, 0),
                        thickness=cv2.FILLED,
                        lineType=cv2.LINE_AA,
                    )
                    img_highres = cv2.GaussianBlur(img_highres, blur_k, blur_sigma)
                    # 下采样到目标尺寸
                    img_lowres = cv2.resize(
                        img_highres,
                        (patch_size, patch_size),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    patch_array[u, v, w, ...] = img_lowres

        ManiSkillSimulator.patch_array = torch.tensor(
            patch_array, dtype=torch.uint8, device="cuda:0"
        )
        # 保存参数供 draw_markers 使用
        ManiSkillSimulator.patch_params = {
            "circle_radius": circle_radius,
            "base_circle_radius": base_circle_radius,
            "super_resolution_ratio": super_resolution_ratio,
        }

    def draw_markers(self, marker_uv: torch.Tensor) -> torch.Tensor:
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
        device = "cuda:0"
        params = ManiSkillSimulator.patch_params
        circle_radius = params["circle_radius"]
        base_circle_radius = params["base_circle_radius"]
        super_resolution_ratio = params["super_resolution_ratio"]
        
        pad_size = 2 * circle_radius
        patch_size = 4 * circle_radius
        
        marker_image = torch.ones((self.cfg.tactile_img_res[1], self.cfg.tactile_img_res[0]),
                               dtype=torch.float32, device=device) * 255
        marker_image = torch.nn.functional.pad(
            marker_image, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=255)
        
        uvs = marker_uv + 0.5 + pad_size
        u_floor = torch.floor(uvs[:, 0]).long()
        v_floor = torch.floor(uvs[:, 1]).long()
        u_frac = uvs[:, 0] - u_floor
        v_frac = uvs[:, 1] - v_floor
        
        patch_id_u = torch.floor(u_frac * super_resolution_ratio).long().clamp(0, super_resolution_ratio - 1)
        patch_id_v = torch.floor(v_frac * super_resolution_ratio).long().clamp(0, super_resolution_ratio - 1)
        marker_size = circle_radius
        patch_id_w = int((marker_size - base_circle_radius) * super_resolution_ratio)
        patch_id_w = max(0, min(patch_id_w, 49))  # clamp to [0, size_slot_num - 1]

        patches = ManiSkillSimulator.patch_array[patch_id_u, patch_id_v, patch_id_w, :, :]
        
        for i in range(len(patches)):
            u_start = u_floor[i].item() - pad_size
            v_start = v_floor[i].item() - pad_size
            if marker_image.shape[1] - patch_size > u_start >= 0 and marker_image.shape[0] - patch_size > v_start >= 0:
                replace_idx = (
                    slice(v_start, v_start + patch_size),
                    slice(u_start, u_start + patch_size)
                )
                old_status = marker_image[replace_idx]
                new_status = torch.minimum(old_status, patches[i].float())
                marker_image[replace_idx] = new_status

        marker_image = marker_image[pad_size:-pad_size, pad_size:-pad_size]
        
        noise_level = 80
        noise = torch.rand_like(marker_image) * noise_level
        marker_mask = marker_image < 255
        marker_image = torch.where(
            marker_mask,
            torch.clamp(marker_image + noise, 0, 255),
            marker_image)
        
        return marker_image
