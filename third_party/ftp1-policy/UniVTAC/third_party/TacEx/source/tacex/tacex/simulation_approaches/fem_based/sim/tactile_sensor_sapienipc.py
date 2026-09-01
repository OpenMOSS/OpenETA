###
# Original code is can be found here https://github.com/chuanyune/ManiSkill-ViTac2025/blob/a3d7df54bca9a2e57f34b37be3a3df36dc218915/Track_1/envs/tactile_sensor_sapienipc.py
##

import json
import math
import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

import copy
import numpy as np
from scipy.ndimage import gaussian_filter

import cv2
import sapien
import transforms3d as t3d
from envs.phong_shading import PhongShadingRenderer
from path import Path
from sapienipc.ipc_component import IPCFEMComponent
from sapienipc.ipc_system import IPCSystem
from sapienipc.ipc_utils.ipc_mesh import IPCTetMesh
from sapienipc.ipc_utils.user_utils import ipc_update_render_all
from sklearn.neighbors import NearestNeighbors
from utils.common import generate_patch_array
from utils.geometry import in_hull, transform_pts
from utils.sapienipc_utils import cv2ex2pose

patch_array_dict_global = generate_patch_array()


class TactileSensorSapienIPC:
    def __init__(
        self,
        scene: sapien.Scene,
        ipc_system: IPCSystem,
        meta_file,
        init_pos,
        init_rot,
        elastic_modulus=1e5,
        poisson_ratio=0.3,
        density: float = 1000,
        friction: float = 0.5,
        name: str = "tactile_sensor",
        no_render: bool = False,
        logger=None,
    ):
        self.ipc_system = ipc_system
        self.scene = scene
        self.time_step = ipc_system.config.time_step
        self.init_pos = init_pos
        self.init_rot = init_rot
        self.current_pos = init_pos
        self.current_rot = init_rot
        self.name = name
        self.logger = logger

        meta_file = Path(track_path) / "assets" / meta_file
        with open(meta_file) as f:
            config = json.load(f)

        meta_dir = meta_file.dirname()
        tet_mesh = IPCTetMesh(filename=meta_dir / config["tet_mesh"])

        # create IPC component
        self.fem_component = IPCFEMComponent()
        self.fem_component.set_tet_mesh(tet_mesh)
        self.fem_component.set_material(density=density, young=elastic_modulus, poisson=poisson_ratio)
        self.fem_component.set_friction(friction)

        if not no_render:
            # create render component
            self.render_component = sapien.render.RenderCudaMeshComponent(
                tet_mesh.n_vertices, tet_mesh.n_surface_triangles
            )
            self.render_component.set_vertex_count(tet_mesh.n_vertices)
            self.render_component.set_triangles(tet_mesh.surface_triangles)
            self.render_component.set_triangle_count(tet_mesh.n_surface_triangles)

            # Set material
            mat = sapien.render.RenderMaterial(
                base_color=[0.3, 1.0, 1.0, 1.0],  # light cyan
                specular=0.8,
                roughness=0.5,
                metallic=0.1,
            )
            self.render_component.set_material(mat)

        # create sapien entity
        self.entity = sapien.Entity()
        self.entity.add_component(self.render_component)
        self.entity.add_component(self.fem_component)
        self.entity.set_pose(sapien.Pose(p=init_pos, q=init_rot))
        self.entity.set_name(name)
        self.scene.add_entity(self.entity)
        self.active = np.loadtxt(meta_dir / config["active"]).astype(bool)
        self.on_surface = np.loadtxt(meta_dir / config["on_surface"]).astype(bool)
        self.faces = np.loadtxt(meta_dir / config["faces"]).astype(np.int32)
        self.boundary_idx = []
        for i in range(len(self.active)):
            if self.active[i] > 0:
                self.boundary_idx.append(i)

        self.boundary_idx = np.array(self.boundary_idx)
        boundary_num = len(self.boundary_idx)
        assert boundary_num >= 6
        self.transform_calculation_ids = [
            self.boundary_idx[0],
            self.boundary_idx[boundary_num // 6],
            self.boundary_idx[2 * boundary_num // 6],
            self.boundary_idx[3 * boundary_num // 6],
            self.boundary_idx[4 * boundary_num // 6],
            self.boundary_idx[5 * boundary_num // 6],
        ]
        self.init_boundary_pts = self.get_vertices_world()[self.transform_calculation_ids, :]
        self.vel_set = False
        self.init_surface_vertices = self.get_surface_vertices_world()

    def get_vertices_world(self):
        v = self.fem_component.get_positions().cpu().numpy()[:, :3]
        return v.copy()

    def get_surface_vertices_world(self):
        return self.get_vertices_world()[self.on_surface].copy()

    def get_surface_vertices_sensor(self):
        v = self.get_surface_vertices_world()
        v_cv = self.transform_to_sensor_frame(v)
        return v_cv

    def get_boundary_vertices_world(self):
        return self.get_vertices_world()[self.boundary_idx].copy()

    def get_pose(self):
        return self.current_pos, self.current_rot

    def transform_to_sensor_frame(self, input_vertices):
        current_pose_transform = np.eye(4)
        current_pose_transform[:3, :3] = t3d.quaternions.quat2mat(self.current_rot)
        current_pose_transform[:3, 3] = self.current_pos
        v_cv = transform_pts(input_vertices, np.linalg.inv(current_pose_transform))
        return v_cv


class VisionTactileSensorSapienIPC(TactileSensorSapienIPC):
    def __init__(
        self,
        marker_interval_range: tuple[float, float] = (2.0625, 2.0625),
        marker_rotation_range: float = 0.0,
        marker_translation_range: tuple[float, float] = (0.0, 0.0),
        marker_pos_shift_range: tuple[float, float] = (0.0, 0.0),
        marker_random_noise: float = 0.0,
        marker_lose_tracking_probability: float = 0.0,
        normalize: bool = False,
        marker_flow_size: int = 128,
        camera_params: tuple[float, float, float, float, float] = (
            340,
            325,
            160,
            125,
            0.0,
        ),
        **kwargs,
    ):
        """
        param: marker_interval_rang, in mm.
        param: marker_rotation_range: overall marker rotation, in radian.
        param: marker_translation_range: overall marker translation, in mm. first two elements: x-axis; last two elements: y-xis.
        param: marker_pos_shift_range: independent marker position shift, in mm, in x- and y-axis. caused by fabrication errors.
        param: marker_random_noise: std of Gaussian marker noise, in pixel. caused by CMOS noise and image processing.
        param: loss_tracking_probability: the probability of losing tracking, appled to each marker
        param: normalize: whether to normalize the output marker flow
        param: marker_flow_size: the size of the output marker flow
        param: camera_params: (fx, fy, cx, cy, distortion)
        """

        super().__init__(**kwargs)

        self.marker_interval_range = marker_interval_range
        self.marker_rotation_range = marker_rotation_range
        self.marker_translation_range = marker_translation_range
        self.marker_pos_shift_range = marker_pos_shift_range
        self.marker_random_noise = marker_random_noise
        self.marker_lose_tracking_probability = marker_lose_tracking_probability
        self.normalize = normalize
        self.marker_flow_size = marker_flow_size
        # camera frame to gel center
        # NOTE: camera frame follows opencv coordinate system
        self.camera2gel = np.eye(4)
        self.camera2gel[:3, :3] = t3d.euler.euler2mat(0.0, 0.0, -np.pi, axes="sxyz")
        self.camera2gel[:3, 3] = (0.0, 0.0, -0.02)
        self.gel2camera = np.linalg.inv(self.camera2gel)
        self.camera_params = camera_params
        self.camera_intrinsic = np.array(
            [
                [camera_params[0], 0, camera_params[2]],
                [0, camera_params[1], camera_params[3]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        self.camera_distort_coeffs = np.array([camera_params[4], 0, 0, 0, 0], dtype=np.float32)
        self.init_vertices_camera = self.get_vertices_camera()
        self.init_surface_vertices_camera = self.get_surface_vertices_camera()
        self.reference_surface_vertices_camera = self.get_surface_vertices_camera()

        self.cam_entity = sapien.Entity()
        self.cam = cam = sapien.render.RenderCameraComponent(320, 240)
        cam.set_perspective_parameters(
            0.0001,
            0.1,
            camera_params[0],
            camera_params[1],
            camera_params[2],
            camera_params[3],
            0,
        )
        self.cam_entity.add_component(cam)
        self.cam_entity.name = self.name + "_cam"
        self.scene.add_entity(self.cam_entity)
        self.phong_shading_renderer = PhongShadingRenderer()
        self.patch_array_dict = copy.deepcopy(patch_array_dict_global)

    def transform_to_camera_frame(self, input_vertices):
        current_pose_transform = np.eye(4)
        current_pose_transform[:3, :3] = t3d.quaternions.quat2mat(self.current_rot)
        current_pose_transform[:3, 3] = self.current_pos
        v_cv = transform_pts(input_vertices, self.gel2camera @ np.linalg.inv(current_pose_transform))
        return v_cv

    def get_vertices_camera(self):
        v = self.get_vertices_world()
        v_cv = self.transform_to_camera_frame(v)
        return v_cv

    def get_camera_pose(self):
        current_pose_transform = np.eye(4)
        current_pose_transform[:3, :3] = t3d.quaternions.quat2mat(self.current_rot)
        current_pose_transform[:3, 3] = self.current_pos
        return np.linalg.inv(self.gel2camera @ np.linalg.inv(current_pose_transform))

    def get_surface_vertices_camera(self):
        v = self.get_surface_vertices_world()
        v_cv = self.transform_to_camera_frame(v)
        return v_cv

    def get_init_surface_vertices_camera(self):
        return self.init_surface_vertices_camera.copy()

    def set_reference_surface_vertices_camera(self):
        self.reference_surface_vertices_camera = self.get_surface_vertices_camera().copy()

    def _gen_marker_grid(self):
        marker_interval = (self.marker_interval_range[1] - self.marker_interval_range[0]) * np.random.rand(1)[
            0
        ] + self.marker_interval_range[0]
        marker_rotation_angle = 2 * self.marker_rotation_range * np.random.rand(1) - self.marker_rotation_range
        marker_translation_x = (
            2 * self.marker_translation_range[0] * np.random.rand(1)[0] - self.marker_translation_range[0]
        )
        marker_translation_y = (
            2 * self.marker_translation_range[1] * np.random.rand(1)[0] - self.marker_translation_range[1]
        )

        marker_x_start = (
            -math.ceil((8 + marker_translation_x) / marker_interval) * marker_interval + marker_translation_x  # 16.5
        )
        marker_x_end = math.ceil((8 - marker_translation_x) / marker_interval) * marker_interval + marker_translation_x
        marker_y_start = (
            -math.ceil((6 + marker_translation_y) / marker_interval) * marker_interval + marker_translation_y
        )
        marker_y_end = math.ceil((6 - marker_translation_y) / marker_interval) * marker_interval + marker_translation_y

        marker_x = np.linspace(
            marker_x_start,
            marker_x_end,
            round((marker_x_end - marker_x_start) / marker_interval) + 1,
            True,
        )
        marker_y = np.linspace(
            marker_y_start,
            marker_y_end,
            round((marker_y_end - marker_y_start) / marker_interval) + 1,
            True,
        )

        marker_xy = np.array(np.meshgrid(marker_x, marker_y)).reshape((2, -1)).T
        marker_num = marker_xy.shape[0]
        # print(marker_num)

        marker_pos_shift_x = (
            np.random.rand(marker_num) * self.marker_pos_shift_range[0] * 2 - self.marker_pos_shift_range[0]
        )

        marker_pos_shift_y = (
            np.random.rand(marker_num) * self.marker_pos_shift_range[1] * 2 - self.marker_pos_shift_range[1]
        )

        marker_xy[:, 0] += marker_pos_shift_x
        marker_xy[:, 1] += marker_pos_shift_y

        rot_mat = np.array([
            [math.cos(marker_rotation_angle), -math.sin(marker_rotation_angle)],
            [math.sin(marker_rotation_angle), math.cos(marker_rotation_angle)],
        ])

        marker_rotated_xy = marker_xy @ rot_mat.T
        return marker_rotated_xy / 1000.0

    def _gen_marker_weight(self, marker_pts):
        # determine facet that each marker belongs to
        surface_pts = self.get_init_surface_vertices_camera()[:, :2]
        marker_on_surface = in_hull(marker_pts, surface_pts)
        marker_pts = marker_pts[marker_on_surface]

        f_v_on_surface = self.on_surface[self.faces]
        f_on_surface = self.faces[np.sum(f_v_on_surface, axis=1) == 3]
        global_id_to_surface_id = np.cumsum(self.on_surface) - 1
        f_on_surface_on_surface_id = global_id_to_surface_id[f_on_surface]
        f_center_on_surface = np.mean(self.init_vertices_camera[f_on_surface][:, :, :2], axis=1)

        nbrs = NearestNeighbors(n_neighbors=4, algorithm="ball_tree").fit(f_center_on_surface)
        distances, idx = nbrs.kneighbors(marker_pts)

        marker_pts_surface_idx = []
        marker_pts_surface_weight = []
        valid_marker_idx = []

        # compute barycentric weight of each vertex
        for i in range(marker_pts.shape[0]):
            possible_face_ids = idx[i]
            p = marker_pts[i]
            for possible_face_id in possible_face_ids.tolist():
                face_vertices_idx = f_on_surface_on_surface_id[possible_face_id]
                closet_pts = surface_pts[face_vertices_idx][:, :2]
                p0, p1, p2 = closet_pts
                A = np.stack([p1 - p0, p2 - p0], axis=1)
                w12 = np.linalg.inv(A) @ (p - p0)
                if possible_face_id == possible_face_ids[0]:
                    marker_pts_surface_idx.append(face_vertices_idx)
                    marker_pts_surface_weight.append(np.array([1 - w12.sum(), w12[0], w12[1]]))
                    valid_marker_idx.append(i)
                    if w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                        break
                elif w12[0] >= 0 and w12[1] >= 0 and w12[0] + w12[1] <= 1:
                    marker_pts_surface_idx[-1] = face_vertices_idx
                    marker_pts_surface_weight[-1] = np.array([1 - w12.sum(), w12[0], w12[1]])
                    valid_marker_idx[-1] = i
                    break

        valid_marker_idx = np.array(valid_marker_idx).astype(np.int32)
        marker_pts = marker_pts[valid_marker_idx]
        marker_pts_surface_idx = np.stack(marker_pts_surface_idx)
        marker_pts_surface_weight = np.stack(marker_pts_surface_weight)
        assert np.allclose(
            (surface_pts[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]).sum(1),
            marker_pts,
        ), (
            "max err:"
            f" {np.abs((surface_pts[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]).sum(1) - marker_pts).max()}"
        )

        return marker_pts_surface_idx, marker_pts_surface_weight

    def gen_marker_uv(self, marker_pts):
        marker_uv = cv2.projectPoints(
            marker_pts,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            self.camera_intrinsic,
            self.camera_distort_coeffs,
        )[0].squeeze(1)

        return marker_uv

    def gen_marker_flow(self):
        marker_grid = self._gen_marker_grid()
        marker_pts_surface_idx, marker_pts_surface_weight = self._gen_marker_weight(marker_grid)
        init_marker_pts = (
            self.reference_surface_vertices_camera[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]
        ).sum(1)
        curr_marker_pts = (
            self.get_surface_vertices_camera()[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]
        ).sum(1)

        init_marker_uv = self.gen_marker_uv(init_marker_pts)
        curr_marker_uv = self.gen_marker_uv(curr_marker_pts)
        marker_mask = np.logical_and.reduce([
            init_marker_uv[:, 0] > 5,
            init_marker_uv[:, 0] < 320,
            init_marker_uv[:, 1] > 5,
            init_marker_uv[:, 1] < 240,
        ])
        marker_flow = np.stack([init_marker_uv, curr_marker_uv], axis=0)
        marker_flow = marker_flow[:, marker_mask]

        # post processing
        no_lose_tracking_mask = np.random.rand(marker_flow.shape[1]) > self.marker_lose_tracking_probability
        marker_flow = marker_flow[:, no_lose_tracking_mask, :]
        noise = np.random.randn(*marker_flow.shape) * self.marker_random_noise
        marker_flow += noise

        original_point_num = marker_flow.shape[1]

        if original_point_num >= self.marker_flow_size:
            chosen = np.random.choice(original_point_num, self.marker_flow_size, replace=False)
            ret = marker_flow[:, chosen, ...]
        else:
            ret = np.zeros((marker_flow.shape[0], self.marker_flow_size, marker_flow.shape[-1]))
            ret[:, :original_point_num, :] = marker_flow.copy()
            ret[:, original_point_num:, :] = ret[:, original_point_num - 1 : original_point_num, :]

        if self.normalize:
            ret /= 160.0
            ret -= 1.0
        return ret

    def gen_rgb_image(self):
        # generate RGB image from depth
        depth = self._gen_depth()
        rgb = self.phong_shading_renderer.generate(depth)
        rgb = rgb.astype(np.float64)

        # generate markers
        marker_grid = self._gen_marker_grid()
        marker_pts_surface_idx, marker_pts_surface_weight = self._gen_marker_weight(marker_grid)
        curr_marker_pts = (
            self.get_surface_vertices_camera()[marker_pts_surface_idx] * marker_pts_surface_weight[..., None]
        ).sum(1)
        curr_marker_uv = self.gen_marker_uv(curr_marker_pts)

        curr_marker = self.draw_marker(curr_marker_uv)
        rgb = rgb.astype(np.float64)
        rgb *= np.dstack([curr_marker.astype(np.float64) / 255] * 3)
        rgb = rgb.astype(np.uint8)
        return rgb

    def _gen_depth(self):
        # hide the gel to get the depth of the object in contact
        self.render_component.disable()
        self.cam_entity.set_pose(cv2ex2pose(self.get_camera_pose()))
        self.scene.update_render()
        ipc_update_render_all(self.scene)
        self.cam.take_picture()
        position = self.cam.get_picture("Position")  # [H, W, 4]
        depth = -position[..., 2]  # float in meter
        fem_smooth_sigma = 2
        depth = gaussian_filter(depth, fem_smooth_sigma)
        self.render_component.enable()

        return depth

    def draw_marker(self, marker_uv, marker_size=3, img_w=320, img_h=240):
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
                replace_idx = (
                    slice(patch_coord_v, patch_coord_v + 12),
                    slice(patch_coord_u, patch_coord_u + 12)
                )
                marker_image[replace_idx] = np.where(
                    marker_image[replace_idx] < current_patch, marker_image[replace_idx], current_patch)
        marker_image = marker_image[12:-12, 12:-12]

        return marker_image
