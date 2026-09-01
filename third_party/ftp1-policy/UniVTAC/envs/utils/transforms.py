import torch
import scipy
import numpy as np
import transforms3d as t3d
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from .._base_task import BaseTask


class Pose:
    def __init__(self, p=(0, 0, 0), q=(1, 0, 0, 0)):
        self.p = np.array(p).copy()
        self.q = np.array(q).copy()
    
    @classmethod
    def from_matrix(cls, mat: np.ndarray):
        p = mat[:3, 3]
        q = t3d.quaternions.mat2quat(mat[:3, :3])
        return cls(p=p, q=q)
    
    @classmethod
    def from_list(cls, lst: list | np.ndarray | torch.Tensor):
        if isinstance(lst, (torch.Tensor, np.ndarray)):
            lst = lst.tolist()
        assert len(lst) in [3, 7], "List must be of length 3 or 7"
        p, q = (0, 0, 0), (1, 0, 0, 0)
        if len(lst) == 3:
            p = lst[:3]
        elif len(lst) == 7:
            p = lst[:3]
            q = lst[3:7]
        return cls(p=p, q=q)

    @property
    def R(self):
        ''' Rotation matrix '''
        return t3d.quaternions.quat2mat(self.q)
    @property
    def euler(self):
        ''' Euler angles '''
        return np.array(t3d.euler.quat2euler(self.q))

    def __eq__(self, other:'Pose'):
        return np.allclose(self.p, other.p) and np.allclose(self.q, other.q)
    
    def __getitem__(self, key):
        return np.concatenate([self.p, self.q])[key]
    
    def __setitem__(self, key, value):
        data = np.concatenate([self.p, self.q])
        data[key] = value
        self.p, self.q = data[:3], data[3:7]
        # Normalize quaternion
        norm_q = np.linalg.norm(self.q)
        if norm_q > 1e-8:
            self.q = self.q / norm_q
    
    def __str__(self):
        def format_array(a):
            return ', '.join([f"{x:.3f}" for x in a])
        return f"Pose(p={format_array(self.p)}, q={format_array(self.q)})"
    
    def __add__(self, other:'Pose|np.ndarray|list'):
        if isinstance(other, Pose):
            return Pose(self.p + other.p, t3d.quaternions.qmult(self.q, other.q))
        else:
            other = np.array(other).reshape(-1)
            assert len(other) in [3, 7], "List must be of length 3 or 7"
            p, q = (0, 0, 0), (1, 0, 0, 0)
            if len(other) == 3:
                p = other[:3]
            elif len(other) == 7:
                p = other[:3]
                q = other[3:7]
            return Pose(self.p + p, t3d.quaternions.qmult(self.q, q))

    def tolist(self):
        return self.p.tolist() + self.q.tolist()
    
    def totensor(self, dtype=torch.float32, device='cpu'):
        return torch.tensor(self.tolist(), dtype=dtype, device=device)

    def to_transformation_matrix(self):
        mat = np.eye(4)
        mat[:3, :3] = t3d.quaternions.quat2mat(self.q)
        mat[:3, 3] = self.p
        return mat
    
    def clone(self):
        return Pose(self.p, self.q)

    @staticmethod
    def create_noise(vec=[0.0, 0.0, 0.0], euler=[0.0, 0.0, 0.0], rng=None):
        assert len(vec) == 3 and len(euler) == 3
        if rng is None:
            rng = np.random
        for i, axis in enumerate(vec):
            if isinstance(axis, (list, np.ndarray)):
                vec[i] = rng.uniform(axis[0], axis[1])
            else:
                vec[i] = rng.uniform(-axis, axis)
        for i, axis in enumerate(euler):
            if isinstance(axis, (list, np.ndarray)):
                euler[i] = rng.uniform(axis[0], axis[1])
            else:
                euler[i] = rng.uniform(-axis, axis)
        return Pose(p=vec, q=t3d.euler.euler2quat(*euler))

    def add_bias(self, vec, coord:Literal['world', 'local']|'Pose'='local', clone=True):
        vec = np.array(vec).reshape(3)
        if isinstance(coord, Pose):
            vec = coord.to_transformation_matrix()[:3, :3] @ np.array(vec).reshape(3, 1)
            vec = vec.reshape(3)
        elif coord == 'local':
            vec = t3d.quaternions.quat2mat(self.q) @ np.array(vec).reshape(3, 1)
            vec = vec.reshape(3)

        if clone:
            return Pose(self.p+vec, self.q)
        else:
            self.p += vec
            return self
    
    def add_rotation(self, euler, coord:Literal['world', 'local']|'Pose'='local', clone=True):
        new_rotation = t3d.euler.euler2quat(*euler)
        if isinstance(coord, Pose):
            rebased_pose = self.rebase(to_coord=coord)
            new_rotation_mat = t3d.quaternions.quat2mat(new_rotation)
            rebased_new_q = t3d.quaternions.qmult(new_rotation, rebased_pose.q)
            rebased_new_p = (new_rotation_mat @ rebased_pose.p.reshape(3, 1)).reshape(3)
            new_pose = Pose(rebased_new_p, rebased_new_q).rebase(from_coord=coord, clone=False)
            new_p, new_q = new_pose.p, new_pose.q
        elif coord == 'local':
            new_q = t3d.quaternions.qmult(new_rotation, self.q)
            new_p = self.p
        else: # coord == 'world'
            new_rotation_mat = t3d.quaternions.quat2mat(new_rotation)
            new_q = t3d.quaternions.qmult(new_rotation, self.q)
            new_p = (new_rotation_mat @ self.p.reshape(3, 1)).reshape(3)

        if clone:
            return Pose(new_p, new_q)
        else:
            self.p, self.q = new_p, new_q
            return self
    
    def add_offset(self, other:'Pose', coord:Literal['world', 'local']='local', clone=True):
        '''
            self.p = self.p + other.p
            self.q = other.q * self.q
        '''
        if coord == 'local':
            new_p = self.p + (t3d.quaternions.quat2mat(self.q) @ other.p.reshape(3, 1)).reshape(3)
            new_q = t3d.quaternions.qmult(other.q, self.q)
        else:
            new_p = self.p + other.p
            new_q = t3d.quaternions.qmult(other.q, self.q)

        if clone:
            return Pose(new_p, new_q)
        else:
            self.p = new_p
            self.q = new_q
            return self
    
    def inv(self):
        inv_mat = np.linalg.inv(self.to_transformation_matrix())
        return Pose.from_matrix(inv_mat)
    
    def rebase(
        self,
        to_coord:Literal['world']|'Pose'='world',
        from_coord:Literal['world']|'Pose'='world',
        clone=True
    ):
        '''
            convert self (in from_coord) to be in to_coord
        '''
        T_self = self.to_transformation_matrix()
        if isinstance(from_coord, Pose):
            T_self_world = from_coord.to_transformation_matrix() @ T_self
        else:
            T_self_world = T_self
 
        if isinstance(to_coord, Pose):
            T_target = np.linalg.inv(to_coord.to_transformation_matrix()) @ T_self_world
        else:
            T_target = T_self_world

        if clone:
            return Pose.from_matrix(T_target)
        else:
            self.p = T_target[:3, 3]
            self.q = t3d.quaternions.mat2quat(T_target[:3, :3])
            return self

    
def estimate_rigid_transform(P:np.ndarray, Q:np.ndarray):
    '''
        estimate rigid transform from point list P to point list Q
    '''
    assert P.shape == Q.shape
    n, dim = P.shape
    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)
    C = np.dot(np.transpose(centeredP), centeredQ) / n

    try:
        V, S, W = np.linalg.svd(C)
        d = np.linalg.det(V) * np.linalg.det(W)
        D = np.eye(3)
        D[2, 2] = d
        R = np.dot(np.dot(V, D), W)
    except Exception as e:
        print(e)
        try:
            V, S, W = scipy.linalg.svd(C, lapack_driver='gesvd')
            d = np.linalg.det(V) * np.linalg.det(W)
            D = np.eye(3)
            D[2, 2] = d
            R = np.dot(np.dot(V, D), W)
        except Exception as e2:
            print(e2)
            R = np.eye(3)

    # P @ R + t = Q
    t = Q.mean(axis=0) - P.mean(axis=0).dot(R)

    mat = np.eye(4)
    mat[:3, :3], mat[:3, 3] = R.T, t
    return mat


def rotate_cone(new_pt: np.ndarray, origin: np.ndarray, z_dir: np.ndarray = [0, 0, 1]):
    x = origin - new_pt
    x = x / np.linalg.norm(x)
    bx_ = np.array(z_dir).reshape(3)
    z = bx_ - np.dot(x, bx_) * x
    z = z / np.linalg.norm(z)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)

def rotate_along_axis(
    target_pose:Pose,
    center_pose:Pose,
    axis,
    theta: float = np.pi / 2,
    axis_type: Literal["center", "target", "world"] = "center",
    towards=None,
    camera_face=None,
) -> list:
    """
    以 center 为中心，沿着指定轴旋转指定角度。通过 towards 可指定旋转方向（方向为 center->target 向量与 towards 向量乘积为正的方向）

    target_pose: 目标点（比如在物体正上方的预抓取点）
    center_pose: 中心点（比如物体的位置）
    axis: 旋转轴
    theta: 旋转角度（单位：弧度）
    axis_type: 旋转轴的类型（'center'：相对于 center_pose，'target'：相对于 target_pose，'world'：世界坐标系），默认是 'center'
    towards: 旋转方向（可选），如果指定了这个参数，则会根据这个参数来决定旋转的方向
    camera_face: 相机朝向（可选），会限制相机向量与该向量点积为正；如果设置为 None，只旋转不考虑相机朝向
    返回值：列表，前3个元素是坐标，后4个元素是四元数
    """
    if theta == 0:
        return target_pose.p.tolist() + target_pose.q.tolist()
    rotate_mat = t3d.axangles.axangle2mat(axis, theta)

    target_mat = target_pose.to_transformation_matrix()
    center_mat = center_pose.to_transformation_matrix()
    if axis_type == "center":
        world_axis = (center_mat[:3, :3] @ np.array(axis).reshape(3, 1)).reshape(3)
    elif axis_type == "target":
        world_axis = (target_mat[:3, :3] @ np.array(axis).reshape(3, 1)).reshape(3)
    else:
        world_axis = np.array(axis).reshape(3)

    rotate_mat = t3d.axangles.axangle2mat(world_axis, theta)
    p = (rotate_mat @ (target_pose.p - center_pose.p).reshape(3, 1)).reshape(3) + center_pose.p
    if towards is not None:
        towards = np.dot(p - center_pose.p, np.array(towards).reshape(3))
        if towards < 0:
            rotate_mat = t3d.axangles.axangle2mat(world_axis, -theta)
            p = (rotate_mat @ (target_pose.p - center_pose.p).reshape(3, 1)).reshape(3) + center_pose.p

    if camera_face is None:
        q = t3d.quaternions.mat2quat(rotate_mat @ target_mat[:3, :3])
    else:
        q = t3d.quaternions.mat2quat(rotate_cone(p, center_pose.p, camera_face))
    return p.tolist() + q.tolist()


def rotate2rob(target_pose:Pose, rob_pose:Pose, box_pose:Pose, theta: float = 0.5) -> list:
    """
    向指定的 rob_pose 偏移
    """
    target_mat = target_pose.to_transformation_matrix()
    v1 = (target_mat[:3, :3] @ np.array([[1, 0, 0]]).T).reshape(3)
    v2 = box_pose.p - rob_pose.p
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    return rotate_along_axis(
        target_pose=target_pose,
        center_pose=box_pose,
        axis=axis,
        theta=angle * theta,
        axis_type="world",
        towards=-v2,
    )


def choose_dirct(block_mat, base_pose: Pose):
    pts = block_mat[:3, :3] @ np.array([[1, -1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 0]])
    dirts = np.sum(np.power(pts - base_pose.p.reshape(3, 1), 2), axis=0)
    return pts[:, np.argmin(dirts)] + block_mat[:3, 3]


# def add_robot_visual_box(task, pose: Pose, name: str = "box"):
#     box_path = Path("./assets/objects/cube/textured.obj")
#     if not box_path.exists():
#         print("[WARNNING] cube not exists!")
#         return

#     pose = _toPose(pose)
#     scene: sapien.Scene = task.scene
#     builder = scene.create_actor_builder()
#     builder.set_physx_body_type("static")
#     builder.add_visual_from_file(
#         filename=str(box_path),
#         scale=[
#             0.04,
#         ] * 3,
#     )
#     builder.set_name(name)
#     builder.set_initial_pose(pose)
#     return builder.build()


def cal_quat_dis(quat1, quat2):
    qmult = t3d.quaternions.qmult
    qinv = t3d.quaternions.qinverse
    qnorm = t3d.quaternions.qnorm
    delta_quat = qmult(qinv(quat1), quat2)
    return 2 * np.arccos(np.fabs((delta_quat / qnorm(delta_quat))[0])) / np.pi


def get_align_matrix(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    获取从 v1 到 v2 的旋转矩阵
    """
    v1 = np.array(v1).reshape(3)
    v2 = np.array(v2).reshape(3)

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    else:
        return t3d.axangles.axangle2mat(axis, angle)


def generate_rotate_vectors(
    axis: Literal["x", "y", "z"] | np.ndarray | list,
    angle: np.ndarray | list | float,
    base: Pose = None,
    vector: np.ndarray | list = [1, 0, 0],
) -> np.ndarray:
    """
    获取从 base 到 axis 的旋转矩阵
    """
    if base is None:
        base = np.eye(4)
    else:
        base = base.to_transformation_matrix()

    if isinstance(axis, str):
        if axis == "x":
            axis = np.array([1, 0, 0])
        elif axis == "y":
            axis = np.array([0, 1, 0])
        elif axis == "z":
            axis = np.array([0, 0, 1])
        else:
            raise ValueError("axis must be x, y or z")
    else:
        axis = np.array(axis).reshape(3)

    axis = (base[:3, :3] @ axis.reshape(3, 1)).reshape(3)
    vector = (base[:3, :3] @ np.array(vector).reshape(3, 1)).reshape(3)

    vector = np.array(vector).reshape((3, 1))
    angle = np.array(angle).flatten()
    rotate_mat = np.zeros((3, angle.shape[0]))
    for idx, a in enumerate(angle):
        rotate_mat[:, idx] = (t3d.axangles.axangle2mat(axis, a) @ vector).reshape(3)
    return rotate_mat


def get_product_vector(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    获取 v2 在 v1 上的投影向量
    """
    v1 = np.array(v1).reshape(3)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.array(v2).reshape(3)
    return np.dot(v1, v2) * v1


def get_place_pose(
    actor_pose: Pose,
    target_pose: Pose,
    constrain: Literal["free", "align"] = "free",
    align_axis: list[np.ndarray] | np.ndarray | list = None,
    actor_axis: np.ndarray | list = [1, 0, 0],
    actor_axis_type: Literal["actor", "world"] = "actor",
) -> Pose:
    """
    获取物体应当被放置到的位置
    考虑因素：
        1. 三维坐标与给定坐标一致
        2. 物体的朝向合理
            - 物体 z 轴与给定坐标 z 轴一致
            - 满足在 xy 平面上的一定约束
                - 无约束（直接采用物体当前的 x,y 在 xOy 平面上的投影）
                - 物体的 x 轴对齐给定 x 轴
                - 选取物体的 x 轴与给定的世界轴单位向量集合中点积最小的方向

    actor_pose: 物体当前的 pose
    target_pose: 物体应当被放置到的位置
    constrain: 物体的约束类型
        - free: 无约束
        - align: 物体的 x 轴与给定的世界轴向量集合中点积最小的方向
    align_axis: 给定的世界轴向量集合，如果设置为 None，默认使用 target_pose 的 x 轴
    actor_axis: 计算点积的 actor 轴，默认使用 x 轴
    actor_axis_type: actor_axis 的类型，默认使用局部坐标系
        - actor: actor_pose 的局部坐标系
        - world: 世界坐标系
    """
    actor_pose_mat = actor_pose.to_transformation_matrix()
    target_pose_mat = target_pose.to_transformation_matrix()

    # 将物体的三维坐标与给定坐标对齐
    actor_pose_mat[:3, 3] = target_pose_mat[:3, 3]

    target_x = target_pose_mat[:3, 0]
    target_y = target_pose_mat[:3, 1]
    target_z = target_pose_mat[:3, 2]

    # 将物体的 z 轴与给定坐标的 z 轴对齐
    # actor2world = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]).T
    z_align_matrix = get_align_matrix(actor_pose_mat[:3, 2], target_z)
    actor_pose_mat[:3, :3] = z_align_matrix @ actor_pose_mat[:3, :3]

    if constrain == "align":
        if align_axis is None:
            align_axis = np.array(target_pose_mat[:3, :3] @ np.array([[1, 0, 0]]).T)
        elif isinstance(align_axis, list):
            align_axis = np.array(align_axis).reshape((-1, 3)).T
        else:
            align_axis = np.array(align_axis).reshape((3, -1))
        align_axis = align_axis / np.linalg.norm(align_axis, axis=0)

        if actor_axis_type == "actor":
            actor_axis = actor_pose_mat[:3, :3] @ np.array(actor_axis).reshape(3, 1)
        elif actor_axis_type == "world":
            actor_axis = np.array(actor_axis)
        closest_axis_id = np.argmax(actor_axis.reshape(3) @ align_axis)
        align_axis = align_axis[:, closest_axis_id]

        actor_axis_xOy = get_product_vector(target_x, actor_axis) + get_product_vector(target_y, actor_axis)
        align_axis_xOy = get_product_vector(target_x, align_axis) + get_product_vector(target_y, align_axis)
        align_mat_xOy = get_align_matrix(actor_axis_xOy, align_axis_xOy)
        actor_pose_mat[:3, :3] = align_mat_xOy @ actor_pose_mat[:3, :3]

    return Pose(actor_pose_mat[:3, 3], t3d.quaternions.mat2quat(actor_pose_mat[:3, :3]))


def get_face_prod(q, local_axis, target_axis):
    """
    get product of local_axis (under q world) and target_axis
    """
    q_mat = t3d.quaternions.quat2mat(q)
    face = q_mat @ np.array(local_axis).reshape(3, 1)
    face_prod = np.dot(face.reshape(3), np.array(target_axis))
    return face_prod

def construct_grasp_pose(p:np.ndarray, grasp_from:np.ndarray, camera_up:np.ndarray) -> Pose:
    '''
        construct grasp pose
        p: the position of the object
        grasp_from: grasping from (+y)
        camera_up: the camera up vector (+x)
    '''
    grasp_from = np.array(grasp_from).reshape(3)
    camera_up = np.array(camera_up).reshape(3)

    z = - grasp_from / np.linalg.norm(grasp_from)
    x = camera_up / np.linalg.norm(camera_up)
    rotate = np.stack([x, np.cross(z, x), z], axis=1)
    return Pose(p, t3d.quaternions.mat2quat(rotate))

def add_visual_box(pose:Pose, name='box', size=0.01, color=np.array([255.0, 0.0, 0.0])):
    from isaacsim.core.api.objects import VisualCuboid
    VisualCuboid(
        prim_path=f"/visualize/{name}",
        size=size,
        position=pose.p,
        orientation=pose.q,
        color=color,
    )

def calculate_target_pose(real_pose:Pose, set_pose:Pose, relative_real_pose:Pose):
    T_A = real_pose.to_transformation_matrix()
    T_B = set_pose.to_transformation_matrix()
    T_C = relative_real_pose.to_transformation_matrix()
    
    R_A = T_A[0:3, 0:3]
    t_A = T_A[0:3, 3]
    inv_A = np.eye(4)
    inv_A[0:3, 0:3] = R_A.T
    inv_A[0:3, 3] = -R_A.T @ t_A
    T_relative = inv_A @ T_C
    T_D = T_B @ T_relative
    relative_set_pose = Pose.from_matrix(T_D)
    
    return relative_set_pose