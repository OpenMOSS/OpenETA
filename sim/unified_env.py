"""Unified environment interface for agent evaluation.

Every env created via ``gym.make()`` with the OpenETA registry returns
the **same** observation structure regardless of simulator backend.

Canonical observation dict::

    {
        "cameras": {
            "<name>": {
                "rgb":        np.ndarray (H, W, 3) uint8,
                "depth":      np.ndarray (H, W) float32 | None,
                "intrinsics": {"fx": float, "fy": float, "cx": float, "cy": float} | None,
                "extrinsics": {"pos": [x,y,z], "quat_xyzw": [x,y,z,w]} | None,
            },
            ...
        },
        "proprio": {
            "joint_positions":  np.ndarray (N,) float32 | None,
            "joint_velocities": np.ndarray (N,) float32 | None,
            "ee_pose":          np.ndarray (7,) float32 | None,   # xyz + xyzw
            "gripper_open":     float | None,
        },
        "task_description": str,
        "objects": [{"name": str, "position": (3,), "orientation": (4,)|None}, ...],
    }

Keys are absent (not None) when a backend doesn't provide them.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


# ══════════════════════════════════════════════════════════════════════
# UnifiedEnv
# ══════════════════════════════════════════════════════════════════════

class UnifiedEnv(gym.Env):
    """Normalising wrapper — one observation schema across all backends.

    Parameters
    ----------
    env:
        An already-instantiated RLinf or dummy ``gym.Env``.
    render_mode:
        ``"human"`` – open an interactive viewer window.
        ``"rgb_array"`` – ``render()`` returns a numpy array.
        ``None`` – no rendering (fastest).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, env: gym.Env, render_mode: str | None = None) -> None:
        super().__init__()
        self._env = env
        self._render_mode = render_mode
        self._backend = self._detect_backend()
        self._include_objects: bool = False  # set by create_env via env_registry.make_env
        self._depth_znear: float = 0.0
        self._depth_zfar: float = 0.0
        self._apply_render_mode()
        self._cache_depth_clip()
        self._has_viewer = self._detect_viewer_support()
        self._obs_space: gym.spaces.Dict | None = None
        self._first_reset = True

    # ── backend detection ────────────────────────────────────────

    def _detect_backend(self) -> str:
        """Return a short string identifying the underlying env class."""
        env = self._unwrap()
        cls = type(env).__name__
        # Normalise common names
        if cls in ("DummySimEnv", "DummyAgentEnv"):
            return "dummy"
        if cls == "GenesisEnv":
            return "genesis"
        if cls == "ManiskillEnv":
            return "maniskill"
        if cls == "MetaWorldEnv":
            return "metaworld"
        if cls == "LiberoEnv":
            return "libero"
        if cls == "BehaviorEnv":
            return "behavior"
        if cls == "BehaviorDirectEnv":
            return "behavior"
        if cls == "RobocasaEnv":
            return "robocasa"
        if cls == "RoboCasaDirectEnv":
            return "robocasa"
        if cls == "CalvinEnv":
            return "calvin"
        if cls == "D4RLEnv":
            return "d4rl"
        # MetaWorld direct envs: class name varies per task
        if "MetaWorld" in cls or "metaworld" in str(type(env).__module__):
            return "metaworld"
        # ManiSkill 3 envs: sapien_env.BaseEnv subclass
        mod = str(type(env).__module__)
        if "mani_skill" in mod or "sapien" in mod:
            return "maniskill"
        # LIBERO envs: wrapped in _LibEnvWrapper from env_registry
        if cls == "_LibEnvWrapper" or hasattr(env, "_env") and "OffScreenRenderEnv" in str(type(getattr(env, "_env", None))):
            return "libero"
        if cls == "FrankaSimEnv":
            return "frankasim"
        if cls == "HabitatEnv":
            return "habitat"
        if cls == "RoboTwinEnv":
            return "robotwin"
        if cls == "RoboVerseEnv":
            return "roboverse"
        if cls == "EmbodiChainEnv":
            return "embodichain"
        return cls.lower()

    def _unwrap(self) -> Any:
        """Walk gymnasium Wrapper chain, but NOT internal vectorised envs."""
        env = self._env
        # Only unwrap gymnasium Wrapper subclasses (not internal .env attrs)
        import gymnasium as gym
        while isinstance(env, gym.Wrapper):
            env = env.env
        return env

    # ── depth helpers ────────────────────────────────────────────

    def _cache_depth_clip(self) -> None:
        if self._backend == "maniskill":
            self._depth_znear = 0.0
            self._depth_zfar = 1000.0

    def _depth_to_metres(self, depth_arr):
        """Convert MuJoCo z-buffer to linear depth (robosuite formula).

        Uses ``get_real_depth_map`` from robosuite::

            real = near / (1.0 - depth * (1.0 - near / far))

        where ``near = znear * extent``, ``far = zfar * extent``.
        Depth is already flipped by the normaliser before reaching here.

        Note: the default ``znear=0.001, zfar=50`` limits precision for
        close-range manipulation (20-30% error at ~0.3m).  This is the
        same calibration that cap-x uses.

        ManiSkill: int16 millimetres -> float32 metres.
        """
        import numpy as np
        arr = self._np(depth_arr)
        if self._backend == "maniskill":
            return arr.astype(np.float32) / 1000.0
        if self._backend in ("libero", "robocasa", "metaworld", "frankasim", "d4rl"):
            model = self._find_mj_model()
            if model is not None:
                extent = float(model.stat.extent)
                near = float(model.vis.map.znear) * extent
                far = float(model.vis.map.zfar) * extent
                a = arr.astype(np.float32)
                return near / (1.0 - a * (1.0 - near / far))
        return arr.astype(np.float32)

    @staticmethod
    def _sanitize_depth_metres(depth_arr: Any) -> np.ndarray:
        """Return finite, non-negative float32 metric depth.

        Zero is the transport-level invalid pixel. This keeps NaN/Inf values
        from leaking into JSON while retaining a fixed metres convention for
        every simulator backend.
        """
        arr = np.asarray(depth_arr, dtype=np.float32)
        return np.where(np.isfinite(arr) & (arr >= 0.0), arr, 0.0).astype(np.float32)

    def _extract_camera_params(self, camera_name: str = "",
                                image_width: int | None = None,
                                image_height: int | None = None) -> dict:
        """Try to extract intrinsics + extrinsics from the underlying env.

        Returns a dict with ``intrinsics`` and ``extrinsics`` keys, or
        empty dicts when the backend doesn't expose them.
        """
        result: dict[str, Any] = {"intrinsics": {}, "extrinsics": {}}

        if self._backend in ("metaworld", "libero", "robocasa", "frankasim", "d4rl"):
            try:
                params = self._mj_camera_params(camera_name, image_width, image_height)
                if params is not None:
                    result["intrinsics"] = {k: params[k] for k in
                        ("fx", "fy", "cx", "cy", "width", "height",
                         "znear", "zfar") if k in params}
                    result["extrinsics"] = params.get("extrinsics", {})
            except Exception:
                pass
        elif self._backend == "maniskill":
            try:
                params = self._ms_camera_params(camera_name)
                if params:
                    result["intrinsics"] = {k: params[k] for k in
                        ("fx", "fy", "cx", "cy", "width", "height") if k in params}
                    result["extrinsics"] = params.get("extrinsics", {})
            except Exception:
                pass

        return result

    def _mj_camera_params(self, camera_name: str = "",
                           image_width: int | None = None,
                           image_height: int | None = None) -> dict | None:
        """Extract camera intrinsics/extrinsics from a MuJoCo ``MjModel``.

        Works for metaworld, libero, frankasim, d4rl (all MuJoCo-based).

        In addition to intrinsics, returns ``znear`` and ``zfar`` — the
        near/far clip planes of the offscreen render context.  These are
        needed to convert the **normalised z-buffer** depth values (from
        ``mjr_readPixels``) to linear depth::

            z_linear = znear * zfar / (zfar - depth_norm * (zfar - znear))

        where ``depth_norm`` is the raw pixel value in [0, 1] from the
        renderer.
        """
        model = self._find_mj_model()
        if model is None:
            return None

        # Find the right camera id
        cam_id = 0
        nc = model.ncam
        if camera_name and nc > 0:
            # robosuite-style: model has camera_name2id / camera_names
            if hasattr(model, "camera_name2id"):
                try:
                    cam_id = model.camera_name2id(camera_name)
                except Exception:
                    pass
            elif hasattr(model, "camera_names"):
                for i, name in enumerate(model.camera_names):
                    if name == camera_name:
                        cam_id = i
                        break
            # fallback: try cam_nameid array
            else:
                for i in range(nc):
                    name = ""
                    try:
                        name_id = model.cam_nameid[i] if hasattr(model, "cam_nameid") else None
                        name = model.names[name_id] if name_id is not None else ""
                    except Exception:
                        pass
                    if name and camera_name in (name, name.decode() if isinstance(name, bytes) else ""):
                        cam_id = i
                        break

        # For MetaWorld's direct env, camera_id=2 is used
        if self._backend == "metaworld":
            cam_id = 2 if nc > 2 else 0

        if cam_id >= nc:
            cam_id = 0

        # Intrinsics from fovy (MuJoCo uses DEGREES!) + resolution
        fovy_deg = float(model.cam_fovy[cam_id])
        fovy = np.deg2rad(fovy_deg) if fovy_deg > 0 else np.pi / 4.0
        w = image_width or int(model.vis.global_.offwidth)
        h = image_height or int(model.vis.global_.offheight)
        fy = (h / 2.0) / np.tan(fovy / 2.0) if fovy > 0 else float(h)
        fx = fy

        extrinsics = {}
        try:
            pos_vals: list | None = None
            mat_vals: list | None = None
            # PREFER the *live* world pose from MjData: data.cam_xpos /
            # data.cam_xmat are the forward-kinematics world pose of the
            # camera, updated every mj_forward/mj_step.  This is REQUIRED for
            # cameras mounted on a moving body (wrist / eye-in-hand): their
            # model.cam_pos is only the static offset relative to the parent
            # body (e.g. [0.05,0,0] for LIBERO's Panda eye_in_hand), which is
            # constant regardless of arm pose — emitting that as world extrinsics
            # is wrong and misplaces the fused point cloud.  data.cam_x* are
            # world-frame for ALL cameras (worldbody-fixed agentview included),
            # so preferring them is always correct.  Fall back to the static
            # model.cam_pos/cam_mat only when MjData is unavailable.
            data = self._find_mj_data()
            if data is not None:
                _xpos = getattr(data, "cam_xpos", None)
                _xmat = getattr(data, "cam_xmat", None)
                try:
                    if _xpos is not None and hasattr(_xpos, "__getitem__"):
                        pos_vals = np.asarray(_xpos[cam_id]).flatten()[:3].tolist()
                    if _xmat is not None and hasattr(_xmat, "__getitem__"):
                        mat_vals = np.asarray(_xmat[cam_id]).flatten()[:9].tolist()
                except Exception:
                    pos_vals = None
                    mat_vals = None
            # MuJoCo 2.3+ ("mujoco" package): cam_pos / cam_mat are indexed arrays
            # (static mount pose — used only if live data pose is unavailable).
            _pos = getattr(model, "cam_pos", None)
            _mat = getattr(model, "cam_mat", None)
            if pos_vals is None and _pos is not None and hasattr(_pos, "__getitem__"):
                pos_vals = np.asarray(_pos[cam_id]).flatten()[:3].tolist()
            if mat_vals is None and _mat is not None and hasattr(_mat, "__getitem__"):
                mat_vals = np.asarray(_mat[cam_id]).flatten()[:9].tolist()
            # Fallback for some mujoco_py versions where fields are cam_posN / cam_matN
            if pos_vals is None:
                for candidate in (f"cam_pos{cam_id}", f"cam_pos0"):
                    raw = getattr(model, candidate, None)
                    if raw is not None:
                        arr = np.asarray(raw)
                        if arr.ndim >= 2 and arr.shape[0] > cam_id:
                            pos_vals = arr[cam_id].flatten()[:3].tolist()
                        else:
                            pos_vals = arr.flatten()[:3].tolist()
                        break
            if mat_vals is None:
                for candidate in (f"cam_mat{cam_id}", f"cam_mat0"):
                    raw = getattr(model, candidate, None)
                    if raw is not None:
                        arr = np.asarray(raw)
                        if arr.ndim >= 2 and arr.shape[0] > cam_id:
                            mat_vals = arr[cam_id].flatten()[:9].tolist()
                        else:
                            mat_vals = arr.flatten()[:9].tolist()
                        break
            if pos_vals is not None or mat_vals is not None:
                # MuJoCo stores cam_mat/cam_xmat **row-major** (C-order) and we
                # ``.flatten()`` it in C-order, so the emitted 9-vector is
                # row-major: ``np.array(mat).reshape(3, 3)`` yields the
                # camera->world rotation directly (columns = camera axes in
                # world).  Tag the layout explicitly so agents never guess.
                extrinsics = {
                    "pos": pos_vals,
                    "mat": mat_vals,
                    "matrix_layout": "row_major",
                    "frame_transform": "camera_to_world",
                    "camera_frame": "opengl",  # camera looks along -Z locally
                }
        except Exception as e:
            import logging
            logging.getLogger("openeta").warning(f"Failed to extract extrinsics: {e}")

        # MuJoCo stores znear/zfar as *fractions* of model.stat.extent, not
        # metres.  The returned depth (see ``_depth_to_metres``) is already
        # linearised into metres using the extent-scaled planes, so we must
        # expose the *scaled* (metric) planes here — otherwise an agent sees
        # ``zfar`` smaller than the max depth value and the numbers look
        # inconsistent.
        extent = float(model.stat.extent)
        return {
            "fx": float(fx), "fy": float(fy),
            "cx": float(w) / 2.0, "cy": float(h) / 2.0,
            "width": w, "height": h,
            "extrinsics": extrinsics,
            # Metric near/far clip planes (metres).  Depth is already
            # linearised to metres, so these bound the valid depth range.
            "znear": float(model.vis.map.znear) * extent,
            "zfar": float(model.vis.map.zfar) * extent,
        }

    def _find_mj_model(self) -> Any:
        """Walk the env/gym wrapper chain looking for a MuJoCo MjModel."""
        def _is_model(obj: Any) -> bool:
            return (hasattr(obj, "cam_fovy") and
                    (hasattr(obj, "ncam") or hasattr(obj, "nCam")))

        # Check direct attributes on the inner env
        for attr in ("model", "sim"):
            obj = getattr(self._env, attr, None)
            if obj is not None and _is_model(obj):
                return obj

        # Walk .env / .unwrapped / ._env chain
        env: Any = self._env
        for _ in range(15):
            # Direct model attribute
            if hasattr(env, "model") and _is_model(env.model):
                return env.model
            # robosuite-style: env.sim is a MjSim wrapper
            if hasattr(env, "sim"):
                sim_obj = env.sim
                if _is_model(sim_obj):
                    return sim_obj
                if hasattr(sim_obj, "model") and _is_model(sim_obj.model):
                    return sim_obj.model
            # Walk deeper — check _env BEFORE unwrapped since some
            # wrapper classes (e.g. _LibEnvWrapper) return self from
            # .unwrapped, creating an infinite loop.
            if hasattr(env, "_env"):
                env = env._env
            elif hasattr(env, "unwrapped"):
                env = env.unwrapped
            elif hasattr(env, "env"):
                env = env.env
            else:
                break
        return None

    def _find_mj_data(self) -> Any:
        """Walk the env/gym wrapper chain looking for the live MuJoCo MjData.

        MjData holds the *live* forward-kinematics state — in particular
        ``cam_xpos`` / ``cam_xmat`` (a camera's world pose given the current
        joint configuration).  This is what wrist / eye-in-hand cameras need;
        ``model.cam_pos`` only holds the static mount offset relative to the
        parent body.  Returns None when no MjData is reachable (callers then
        fall back to the static model pose).
        """
        def _is_data(obj: Any) -> bool:
            return hasattr(obj, "cam_xpos") and hasattr(obj, "cam_xmat")

        # Direct attributes on the inner env
        for attr in ("data", "sim"):
            obj = getattr(self._env, attr, None)
            if obj is not None:
                if _is_data(obj):
                    return obj
                # robosuite-style: env.sim.data
                inner = getattr(obj, "data", None)
                if inner is not None and _is_data(inner):
                    return inner

        # Walk .env / .unwrapped / ._env chain (mirrors _find_mj_model)
        env: Any = self._env
        for _ in range(15):
            if hasattr(env, "data") and _is_data(env.data):
                return env.data
            if hasattr(env, "sim"):
                sim_obj = env.sim
                if _is_data(sim_obj):
                    return sim_obj
                if hasattr(sim_obj, "data") and _is_data(sim_obj.data):
                    return sim_obj.data
            if hasattr(env, "_env"):
                env = env._env
            elif hasattr(env, "unwrapped"):
                env = env.unwrapped
            elif hasattr(env, "env"):
                env = env.env
            else:
                break
        return None

    def _ms_camera_params(self, camera_name: str = "") -> dict | None:
        """Extract camera parameters from a ManiSkill (SAPIEN) sensor."""
        try:
            inner = self._unwrap()
            if not hasattr(inner, "agent"):
                return None
            agent = inner.agent
            if not hasattr(agent, "scene") or not hasattr(agent.scene, "sensors"):
                return None
            sensors = agent.scene.sensors
            if camera_name:
                sensor = sensors.get(camera_name) if isinstance(sensors, dict) else None
            else:
                sensor = None
                for _name, _s in (sensors.items() if isinstance(sensors, dict) else []):
                    if hasattr(_s, "rgb") or "rgb" in str(type(_s).__name__).lower():
                        sensor = _s
                        break
            if sensor is None:
                return None
            # ManiSkill Sensor stores params in .config (CameraConfig)
            cfg = getattr(sensor, "config", sensor)
            near = getattr(cfg, "near", 0.01)
            far = getattr(cfg, "far", 100.0)
            fov = getattr(cfg, "fov", None)
            w = getattr(cfg, "width", None)
            h = getattr(cfg, "height", None)
            if fov is not None and h is not None and w is not None:
                fy = (h / 2.0) / np.tan(float(fov) / 2.0)
                fx = fy
                result: dict = {
                    "fx": float(fx), "fy": float(fy),
                    "cx": float(w) / 2.0, "cy": float(h) / 2.0,
                    "width": int(w), "height": int(h),
                }
                if near is not None:
                    result["near"] = float(near)
                if far is not None:
                    result["far"] = float(far)
                # Pose from sensor config (may be GPU tensors — use self._np)
                try:
                    pose = cfg.pose if hasattr(cfg, "pose") else None
                    if pose is not None:
                        p = pose.p if hasattr(pose, "p") else None
                        q = pose.q if hasattr(pose, "q") else None
                        ex: dict = {}
                        if p is not None:
                            ex["pos"] = self._np(p).flatten()[:3].tolist()
                        if q is not None:
                            # SAPIEN ``pose.q`` is **wxyz** and encodes the
                            # camera->world rotation.  The rest of OpenETA uses
                            # xyzw, so reorder to xyzw for a truthful field name.
                            wxyz = self._np(q).flatten()[:4]
                            ex["quat_xyzw"] = [float(wxyz[1]), float(wxyz[2]),
                                               float(wxyz[3]), float(wxyz[0])]
                            ex["frame_transform"] = "camera_to_world"
                            # SAPIEN/ROS camera: looks along local +X, +Z up.
                            ex["camera_frame"] = "ros"
                        if ex:
                            result["extrinsics"] = ex
                except Exception:
                    pass
                return result
        except Exception:
            pass
        return None

    # ── render_mode propagation ──────────────────────────────────

    def _apply_render_mode(self) -> None:
        if self._render_mode is None:
            return
        env = self._unwrap()
        b = self._backend
        if b == "genesis":
            if hasattr(env, "scene") and env.scene is not None:
                env.scene._show_viewer = (self._render_mode == "human")

    def _detect_viewer_support(self) -> bool:
        env = self._unwrap()
        cls = type(env).__name__
        if cls in ("DummySimEnv", "DummyAgentEnv"):
            return False
        if cls in ("GenesisEnv", "ManiskillEnv", "MetaWorldEnv", "FrankaSimEnv",
                    "D4RLEnv", "LiberoEnv", "CalvinEnv", "BehaviorEnv", "BehaviorDirectEnv",
                    "RobocasaEnv", "HabitatEnv"):
            return True
        if hasattr(env, "render") and callable(env.render):
            return True
        return False

    @property
    def has_viewer(self) -> bool:
        return self._has_viewer

    @property
    def render_mode(self) -> str | None:
        return self._render_mode

    # ══════════════════════════════════════════════════════════════
    # observation normalisation
    # ══════════════════════════════════════════════════════════════

    def _normalise_obs(self, raw: Any) -> dict[str, Any]:
        """Route to the correct backend-specific normaliser."""
        if isinstance(raw, tuple):
            raw = raw[0]
        if not isinstance(raw, dict):
            raw = {"_raw": raw}

        # Dispatch
        fn = getattr(self, f"_normalise_{self._backend}", self._normalise_generic)
        obs = fn(raw)

        # ── task_description: guaranteed key ─────────────────────
        if "task_description" not in obs:
            if "task" in raw:
                obs["task_description"] = str(raw["task"])
            elif "task_descriptions" in raw:
                td = raw["task_descriptions"]
                obs["task_description"] = str(td[0]) if isinstance(td, (list, tuple)) else str(td)
            elif hasattr(self._env, "_task_description"):
                obs["task_description"] = str(self._env._task_description)
            elif hasattr(self._env, "_env") and hasattr(self._env._env, "_task_description"):
                obs["task_description"] = str(self._env._env._task_description)
            else:
                obs["task_description"] = ""

        return obs

    # ── per-backend normalisers ──────────────────────────────────

    def _normalise_dummy(self, raw: dict) -> dict:
        """DummySimEnv provides CameraFrame, RobotState, and objects."""
        cameras: dict[str, dict] = {}
        for c in raw.get("cameras", []):
            if isinstance(c, dict):
                name = c.get("frame_id", "camera")
                cam: dict[str, Any] = {}
                rgb = c.get("rgb")
                if rgb is not None:
                    cam["rgb"] = self._np(rgb)
                depth = c.get("depth")
                if depth is not None:
                    cam["depth"] = self._np(depth)
                intr = c.get("intrinsics")
                if intr and isinstance(intr, dict):
                    cam["intrinsics"] = dict(intr)
                extr = c.get("extrinsics")
                if extr and isinstance(extr, dict):
                    cam["extrinsics"] = dict(extr)
                cameras[name] = cam

        obs: dict[str, Any] = {}
        if cameras:
            obs["cameras"] = cameras

        # proprio
        proprio: dict[str, Any] = {}
        jp = raw.get("robot_joint_positions")
        if jp is not None:
            proprio["joint_positions"] = self._np(jp)
        jv = raw.get("robot_joint_velocities")
        if jv is not None:
            proprio["joint_velocities"] = self._np(jv)
        ee = raw.get("end_effector_pose")
        if ee and isinstance(ee, dict):
            xyz = ee.get("xyz")
            quat = ee.get("quat_xyzw")
            if xyz is not None and quat is not None:
                proprio["ee_pose"] = np.concatenate([self._np(xyz), self._np(quat)])
            elif xyz is not None:
                proprio["ee_pose"] = self._np(xyz)
        gs = raw.get("gripper_state")
        if gs and isinstance(gs, dict):
            proprio["gripper_open"] = float(gs.get("open", 0.0))
        if proprio:
            obs["proprio"] = proprio

        # objects
        objs = raw.get("objects")
        if objs:
            obs["objects"] = [dict(o) if isinstance(o, dict) else o for o in objs]

        obs["task_description"] = raw.get("task", "")

        # metadata
        meta = raw.get("metadata")
        if meta:
            obs["metadata"] = dict(meta)
        return obs

    def _normalise_genesis(self, raw: dict) -> dict:
        obs: dict[str, Any] = {}

        # cameras
        img = raw.get("main_images")
        if img is not None:
            obs["cameras"] = {"head": {"rgb": self._np(img)}}

        # proprio
        state = raw.get("states")
        if state is not None:
            state = self._np(state)
            obs["proprio"] = {
                "joint_positions": state[:7] if len(state) >= 7 else state,
            }
            if len(state) >= 9:
                obs["proprio"]["gripper_open"] = float(state[7:9].mean())

        return obs

    def _normalise_maniskill(self, raw: dict) -> dict:
        """ManiSkill obs normaliser — supports rgbd, state, and raw-tensor modes."""
        obs: dict[str, Any] = {}

        # ── joint positions ─────────────────────────────────────────
        # rgbd mode: agent.qpos
        agent = raw.get("agent", {})
        if isinstance(agent, dict) and "qpos" in agent:
            state = self._np(agent["qpos"])
            joint_positions = state[..., :9] if state.shape[-1] >= 9 else state
            obs.setdefault("proprio", {})["joint_positions"] = joint_positions
        # state mode: flat tensor
        _raw = raw.get("_raw")
        if not obs.get("proprio"):
            state = None
            if isinstance(_raw, np.ndarray) or (hasattr(_raw, "detach") and not isinstance(_raw, dict)):
                state = self._np(_raw)
            else:
                state = raw.get("states")
            if state is not None:
                state = self._np(state)
                jp = state[..., :9] if state.shape[-1] >= 9 else state
                obs.setdefault("proprio", {})["joint_positions"] = jp

        # ── EE pose from robot FK ──────────────────────────────────
        try:
            inner = self._unwrap()
            if hasattr(inner, "agent") and hasattr(inner.agent, "robot"):
                tcp = inner.agent.robot.get_pose()
                if tcp is not None:
                    pos = self._np(tcp.p).flatten()[:3]
                    quat = self._np(tcp.q).flatten()[:4]
                    obs.setdefault("proprio", {})["ee_pose"] = np.concatenate([pos, quat])
        except Exception:
            pass

        # ── cameras (rgbd sensor_data) ──────────────────────────────
        sensor = raw.get("sensor_data", {})
        if sensor:
            cameras: dict[str, dict] = {}
            for cam_name, cam_data in sensor.items():
                if not isinstance(cam_data, dict):
                    continue
                cam: dict[str, Any] = {}
                rgb = cam_data.get("rgb")
                if rgb is not None:
                    arr = self._np(rgb)
                    cam["rgb"] = arr
                    h, w = arr.shape[:2]
                else:
                    h = w = None
                depth = cam_data.get("depth")
                if depth is not None:
                    cam["depth"] = self._depth_to_metres(np.squeeze(self._np(depth)))
                if cam:
                    cp = self._extract_camera_params(cam_name, image_width=w, image_height=h)
                    cam.update(cp)
                    cameras[cam_name] = cam
            if cameras:
                obs["cameras"] = cameras

        # Objects (privileged info)
        if self._include_objects:
            obs["objects"] = self._extract_maniskill_objects()

        if not obs:
            # Fallback: the raw obs was a tensor
            st = raw.get("_raw") if not isinstance(raw.get("_raw"), dict) else None
            if st is not None:
                obs["proprio"] = {"joint_positions": self._np(st)}

        return obs

    def _normalise_behavior(self, raw: dict) -> dict:
        obs: dict[str, Any] = {}

        # RLinf's BehaviorEnv._wrap_obs gives us:
        #   main_images: (H,W,3) zed head camera
        #   wrist_images: (2,H,W,3) left+right wrist realsense
        #   states: (32,) proprio
        img = raw.get("main_images")
        main_depth = raw.get("main_depth")
        wrist = raw.get("wrist_images")
        wrist_depths = raw.get("wrist_depths")
        camera_params = raw.get("_openeta_camera_params", {})
        if img is not None or wrist is not None:
            cameras: dict[str, dict] = {}
            if img is not None:
                cameras["zed_head"] = {"rgb": self._np(img)}
                if main_depth is not None:
                    cameras["zed_head"]["depth"] = self._sanitize_depth_metres(main_depth)
            if wrist is not None:
                wr = self._np(wrist)
                if wr.ndim == 4 and wr.shape[0] >= 2:
                    cameras["wrist_left"] = {"rgb": wr[0]}
                    cameras["wrist_right"] = {"rgb": wr[1]}
                    if wrist_depths is not None:
                        wd = self._np(wrist_depths).astype(np.float32)
                        if wd.ndim == 3 and wd.shape[0] >= 2:
                            cameras["wrist_left"]["depth"] = self._sanitize_depth_metres(wd[0])
                            cameras["wrist_right"]["depth"] = self._sanitize_depth_metres(wd[1])
                elif wr.ndim == 3:
                    cameras["wrist_left"] = {"rgb": wr}
            if isinstance(camera_params, dict):
                for name, params in camera_params.items():
                    if name in cameras and isinstance(params, dict):
                        cameras[name].update(params)
            if cameras:
                obs["cameras"] = cameras

        structured = raw.get("_openeta_proprio")
        if isinstance(structured, dict):
            obs["proprio"] = dict(structured)
        else:
            state = raw.get("states")
            if state is not None:
                state = self._np(state)
                obs["proprio"] = {"joint_positions": state}

        metadata = raw.get("_openeta_metadata")
        if isinstance(metadata, dict):
            obs["metadata"] = dict(metadata)

        return obs

    def _normalise_libero(self, raw: dict) -> dict:
        """LIBERO raw obs: agentview_image + robot0_eye_in_hand_image + depth + robot state."""

        # Lazy: patch MjrContext znear/zfar
        obs: dict[str, Any] = {}
        cameras: dict[str, dict] = {}

        # LIBERO provides images directly in the obs dict
        agentview = raw.get("agentview_image")
        agentview_depth = raw.get("agentview_depth")
        wrist = raw.get("robot0_eye_in_hand_image")
        wrist_depth = raw.get("robot0_eye_in_hand_depth")

        if agentview is not None:
            arr = np.flipud(self._np(agentview))
            h, w = arr.shape[:2]
            cam: dict[str, Any] = {"rgb": arr}
            if agentview_depth is not None:
                cam["depth"] = self._depth_to_metres(np.squeeze(np.flipud(self._np(agentview_depth))))
            cp = self._extract_camera_params("agentview", image_width=w, image_height=h)
            cam.update(cp)
            cameras["agentview"] = cam
        if wrist is not None:
            arr = np.flipud(self._np(wrist))
            h, w = arr.shape[:2]
            cam_w: dict[str, Any] = {"rgb": arr}
            if wrist_depth is not None:
                cam_w["depth"] = self._depth_to_metres(np.squeeze(np.flipud(self._np(wrist_depth))))
            cp = self._extract_camera_params("robot0_eye_in_hand", image_width=w, image_height=h)
            cam_w.update(cp)
            cameras["wrist"] = cam_w
        if cameras:
            obs["cameras"] = cameras

        # Proprio: eef_pos(3) + eef_quat(4) + gripper_qpos(2)
        eef_pos = raw.get("robot0_eef_pos")
        eef_quat = raw.get("robot0_eef_quat")
        gripper = raw.get("robot0_gripper_qpos")
        joint_pos = raw.get("robot0_joint_pos")

        proprio: dict[str, Any] = {}
        if eef_pos is not None and eef_quat is not None:
            ep = self._np(eef_pos)
            eq = self._np(eef_quat)
            # robosuite's robot0_eef_quat observable is already xyzw.
            proprio["ee_pose"] = np.concatenate([ep, eq])
        if gripper is not None:
            g = self._np(gripper)
            # LIBERO gripper_qpos is [left_finger, right_finger] symmetric
            # around 0, with full-open ≈ 0.04.  Normalise to [0, 1] so
            # the RobotState threshold (> 0.5) can detect open/closed.
            raw = float(np.abs(g).mean()) if g.size > 0 else 0.0
            proprio["gripper_open"] = min(1.0, raw * 25.0)
        if joint_pos is not None:
            proprio["joint_positions"] = self._np(joint_pos)
        if proprio:
            obs["proprio"] = proprio

        # Objects (privileged info)
        if self._include_objects:
            obs["objects"] = self._extract_libero_objects()

        return obs

    def _extract_libero_objects(self) -> list[dict]:
        """Extract scene objects from LIBERO's robosuite env."""
        try:
            # Walk: _LibEnvWrapper._env → OffScreenRenderEnv.env → robosuite env
            wrapper = self._env
            if hasattr(wrapper, "_env"):
                wrapper = wrapper._env  # _LibEnvWrapper → OffScreenRenderEnv
            inner_env = getattr(wrapper, "env", None)
            if inner_env is None or not hasattr(inner_env, "objects"):
                return []
            sim = getattr(wrapper, "sim", None)
            if sim is None:
                return []
            model = sim.model
            data = sim.data
            qpos = data.qpos.copy()
            objects: list[dict] = []
            for obj in inner_env.objects:
                root_body = getattr(obj, "root_body", "")
                if not root_body:
                    continue
                try:
                    body_id = model.body_name2id(root_body)
                    jnt_id = model.body_jntadr[body_id]
                    qpos_addr = model.jnt_qposadr[jnt_id]
                    pos = qpos[qpos_addr:qpos_addr + 3].tolist()
                    quat = qpos[qpos_addr + 3:qpos_addr + 7].tolist()
                except Exception:
                    pos, quat = None, None
                objects.append({
                    "name": getattr(obj, "name", root_body),
                    "category": getattr(obj, "category_name", ""),
                    "position": pos,
                    "orientation": quat,
                })
            return objects
        except Exception:
            return []

    def _normalise_metaworld(self, raw: dict) -> dict:
        """MetaWorld direct env returns numpy state array, not a dict."""
        obs: dict[str, Any] = {}

        # Handle three raw formats:
        # a) RLinf MetaWorldEnv._wrap_obs → {"full_image": ..., "state": ...}
        # b) Direct gym env → ndarray gets wrapped as {"_raw": ndarray}
        # c) Direct gym env dict → raw is already a dict from wrapped obs
        state_arr = raw.get("_raw") if isinstance(raw.get("_raw"), np.ndarray) else None
        if state_arr is None:
            state_arr = raw.get("state")

        # Render image — try rgbd_tuple for depth-aware render
        env_to_render = getattr(self._env, "env", self._env) if hasattr(self._env, "env") else self._env
        if hasattr(env_to_render, "render") and callable(env_to_render.render):
            try:
                img = None
                depth_img = None
                # Prefer rgbd_tuple from gymnasium's MujocoRenderer for depth
                if hasattr(env_to_render, "unwrapped"):
                    uw = env_to_render.unwrapped
                    mr = getattr(uw, "mujoco_renderer", None)
                    if mr is not None and hasattr(mr, "render"):
                        try:
                            img, depth_img = mr.render("rgbd_tuple")
                        except Exception:
                            pass
                if img is None:
                    img = env_to_render.render()
                if img is not None:
                    img = self._np(img)
                    img = np.flipud(img)  # MuJoCo OpenGL → top-left origin
                    h, w = img.shape[:2]
                    cam_dict: dict[str, Any] = {"rgb": img}
                    if depth_img is not None:
                        cam_dict["depth"] = self._depth_to_metres(np.flipud(self._np(depth_img)))
                    cp = self._extract_camera_params("view", image_width=w, image_height=h)
                    cam_dict.update(cp)
                    obs["cameras"] = {"view": cam_dict}
            except Exception:
                pass

        # Proprio — MetaWorld obs array: [0:3]=ee_pos, [3:7]=ee_quat (if full obs), [3]=gripper (if simple obs)
        if state_arr is not None:
            state_arr = self._np(state_arr)
            self._last_metaworld_state = state_arr  # cached for _extract_metaworld_objects
            if len(state_arr) >= 7:
                obs["proprio"] = {
                    "ee_pose": state_arr[:3] if len(state_arr) >= 3 else state_arr,
                    "gripper_open": float(state_arr[3]) if len(state_arr) >= 4 else None,
                }
            else:
                obs["proprio"] = {"joint_positions": state_arr}

        # Objects (privileged info)
        if self._include_objects:
            obs["objects"] = self._extract_metaworld_objects()

        # Task description — attached by _make_metaworld_direct
        if hasattr(self._env, "_task_description"):
            obs["task_description"] = self._env._task_description

        return obs

    def _extract_metaworld_objects(self) -> list[dict]:
        """Extract object/goal positions from MetaWorld's flat obs array.

        MetaWorld obs layout (consistent across ML1/MT1/ML45 tasks):
          [0:3]=hand_xyz, [3]=gripper, [4:7]=obj_xyz, [7:11]=obj_quat(wxyz),
          [11:14]=goal_xyz, [14:18]=goal_quat(wxyz), rest=zeros
        """
        arr = getattr(self, "_last_metaworld_state", None)
        if arr is None or len(arr) < 7:
            return []
        import numpy as np
        arr = np.asarray(arr)
        objects: list[dict] = []
        try:
            # Use the task name from the env
            if hasattr(self._env, "unwrapped"):
                uw = self._env.unwrapped
                model_name = getattr(uw, "model_name", "")
                if isinstance(model_name, str) and model_name:
                    name = model_name.rsplit(".xml", 1)[0].rsplit("/", 1)[-1]
                else:
                    name = "metaworld"
            else:
                name = "metaworld"
        except Exception:
            name = "metaworld"
        # Object at [4:7]
        if len(arr) >= 7:
            objects.append({
                "name": f"{name}/object",
                "position": arr[4:7].tolist(),
                "orientation": arr[7:11].tolist() if len(arr) >= 11 else None,
            })
        # Goal at [11:14]
        if len(arr) >= 14:
            objects.append({
                "name": f"{name}/goal",
                "position": arr[11:14].tolist(),
                "orientation": arr[14:18].tolist() if len(arr) >= 18 else None,
            })
        return objects

    def _extract_maniskill_objects(self) -> list[dict]:
        """Extract scene actors/articulations from ManiSkill's SAPIEN scene."""
        try:
            inner = self._unwrap()
            scene = getattr(inner, "scene", None)
            if scene is None:
                return []
            objects: list[dict] = []

            # Static actors (objects)
            actors = getattr(scene, "actors", {})
            for name, actor in (actors.items() if isinstance(actors, dict) else []):
                if name in ("ground", "goal_ee"):
                    continue  # skip ground plane and goal marker
                pose = getattr(actor, "pose", None)
                pos, quat = None, None
                if pose is not None:
                    if hasattr(pose, "p") and pose.p is not None:
                        pos = self._np(pose.p).flatten()[:3].tolist()
                    if hasattr(pose, "q") and pose.q is not None:
                        quat = self._np(pose.q).flatten()[:4].tolist()
                objects.append({"name": name, "position": pos, "orientation": quat})

            return objects
        except Exception:
            return []

    def _normalise_robocasa(self, raw: dict) -> dict:
        """Normalise official RoboCasa365 or legacy RLinf observations."""

        obs: dict[str, Any] = {}
        cameras: dict[str, dict] = {}

        official_cameras = [
            ("robot0_agentview_left", "agentview_left"),
            ("robot0_agentview_right", "agentview_right"),
            ("robot0_robotview", "agentview"),
            ("robot0_eye_in_hand", "wrist"),
        ]
        for raw_name, cam_name in official_cameras:
            image = raw.get(f"{raw_name}_image")
            if image is None:
                continue
            arr = np.flipud(self._np(image))
            h, w = arr.shape[:2]
            camera: dict[str, Any] = {"rgb": arr[..., :3]}
            depth = raw.get(f"{raw_name}_depth")
            if depth is not None:
                depth_arr = np.squeeze(np.flipud(self._np(depth)))
                camera["depth"] = self._sanitize_depth_metres(
                    self._depth_to_metres(depth_arr)
                )
            camera.update(
                self._extract_camera_params(raw_name, image_width=w, image_height=h)
            )
            cameras[cam_name] = camera

        # Backward-compatible support for the vendored RLinf vector wrapper.
        for raw_key, cam_name in [
            ("left_images", "agentview_left"),
            ("wrist_images", "wrist"),
            ("right_images", "agentview_right"),
        ]:
            if cam_name in cameras:
                continue
            images = raw.get(raw_key)
            if images is None:
                continue
            arr = self._np(images)
            if arr.ndim == 4:
                arr = arr[0]
            cameras[cam_name] = {"rgb": arr}
        if cameras:
            obs["cameras"] = cameras

        proprio: dict[str, Any] = {}
        joint_pos = raw.get("robot0_joint_pos")
        joint_vel = raw.get("robot0_joint_vel")
        eef_pos = raw.get("robot0_eef_pos")
        eef_quat = raw.get("robot0_eef_quat")
        gripper = raw.get("robot0_gripper_qpos")
        base_pos = raw.get("robot0_base_pos")
        base_quat = raw.get("robot0_base_quat")

        if joint_pos is not None:
            proprio["joint_positions"] = self._np(joint_pos)
        if joint_vel is not None:
            proprio["joint_velocities"] = self._np(joint_vel)
        if eef_pos is not None and eef_quat is not None:
            proprio["ee_pose"] = np.concatenate(
                [self._np(eef_pos).reshape(-1)[:3], self._np(eef_quat).reshape(-1)[:4]]
            )
        if gripper is not None:
            gripper_arr = self._np(gripper).reshape(-1)
            raw_open = float(np.abs(gripper_arr).mean()) if gripper_arr.size else 0.0
            proprio["gripper_open"] = min(1.0, raw_open * 25.0)
        if base_pos is not None or base_quat is not None:
            base_pose: dict[str, Any] = {}
            if base_pos is not None:
                base_pose["xyz"] = self._np(base_pos).reshape(-1)[:3].tolist()
            if base_quat is not None:
                base_pose["quat_xyzw"] = self._np(base_quat).reshape(-1)[:4].tolist()
            proprio["base_pose"] = base_pose

        state = raw.get("states")
        if not proprio and state is not None:
            state_arr = self._np(state)
            proprio["joint_positions"] = (
                state_arr[:9] if len(state_arr) >= 9 else state_arr
            )
        if proprio:
            obs["proprio"] = proprio

        task_description = raw.get("_openeta_task_description")
        if task_description is not None:
            obs["task_description"] = str(task_description)
        benchmark = raw.get("_openeta_benchmark")
        if isinstance(benchmark, dict):
            obs["metadata"] = {"benchmark": dict(benchmark)}

        if self._include_objects:
            obs["objects"] = self._extract_robocasa_objects()
        return obs

    def _extract_robocasa_objects(self) -> list[dict]:
        """Extract RoboCasa object poses for opt-in debugging only."""

        try:
            inner = getattr(self._env, "unwrapped_env", self._env)
            sim = getattr(inner, "sim", None)
            if sim is None:
                return []
            objects: list[dict] = []
            for obj in getattr(inner, "objects", []):
                root_body = getattr(obj, "root_body", "")
                if not root_body:
                    continue
                try:
                    body_id = sim.model.body_name2id(root_body)
                    position = np.asarray(sim.data.body_xpos[body_id]).reshape(-1)[:3].tolist()
                    orientation = np.asarray(sim.data.body_xquat[body_id]).reshape(-1)[:4].tolist()
                except Exception:
                    position, orientation = None, None
                objects.append(
                    {
                        "name": getattr(obj, "name", root_body),
                        "category": getattr(obj, "category_name", ""),
                        "position": position,
                        "orientation_wxyz": orientation,
                    }
                )
            return objects
        except Exception:
            return []

    def _normalise_calvin(self, raw: dict) -> dict:
        obs: dict[str, Any] = {}
        cameras: dict[str, dict] = {}

        full = raw.get("full_image")
        wrist = raw.get("wrist_image")
        if full is not None:
            cameras["static"] = {"rgb": self._np(full)}
        if wrist is not None:
            cameras["gripper"] = {"rgb": self._np(wrist)}
        if cameras:
            obs["cameras"] = cameras

        state = raw.get("state")
        if state is not None:
            state = self._np(state)
            obs["proprio"] = {"joint_positions": state}
        return obs

    def _normalise_d4rl(self, raw: dict) -> dict:
        # D4RL: states only, no images
        state = raw.get("states")
        if state is not None:
            return {"proprio": {"joint_positions": self._np(state)}}
        return {}

    def _normalise_frankasim(self, raw: dict) -> dict:
        # FrankaSim: MuJoCo states + render for image
        obs: dict[str, Any] = {}
        state = raw.get("state") or raw.get("states")
        if state is not None:
            obs["proprio"] = {"joint_positions": self._np(state)}
        # rendered image if available
        full = raw.get("full_image")
        if full is not None:
            obs["cameras"] = {"view": {"rgb": self._np(full)}}
        return obs

    def _normalise_habitat(self, raw: dict) -> dict:
        obs: dict[str, Any] = {}
        # Habitat: obs may contain rgb + depth + semantic per sensor
        cameras: dict[str, dict] = {}
        for key, val in raw.items():
            if not isinstance(val, (np.ndarray, list)):
                continue
            arr = self._np(val)
            if arr.ndim not in (3, 4):
                continue
            if arr.shape[-1] == 3 and "rgb" in key.lower():
                cameras[key] = {"rgb": arr}
            elif arr.shape[-1] == 1 or ("depth" in key.lower() and arr.ndim == 2):
                # find matching camera
                base = key.replace("_depth", "").replace("depth_", "").replace("depth", "")
                cam = cameras.get(base, {})
                cam["depth"] = arr
                cameras[base] = cam
        if cameras:
            obs["cameras"] = cameras

        state = raw.get("states") or raw.get("state")
        if state is not None:
            obs["proprio"] = {"joint_positions": self._np(state)}
        return obs

    def _normalise_generic(self, raw: dict) -> dict:
        """Fallback normaliser for unknown backends."""
        obs: dict[str, Any] = {}
        img = raw.get("main_images") or raw.get("full_image") or raw.get("image")
        if img is not None:
            obs["cameras"] = {"view": {"rgb": self._np(img)}}
        state = raw.get("states") or raw.get("state")
        if state is not None:
            obs["proprio"] = {"joint_positions": self._np(state)}
        return obs

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _np(value: Any) -> np.ndarray:
        """Convert tensor/array → numpy, squeezing batch dim 0 if present."""
        # Lazy import torch — only needed when value is a torch.Tensor
        if not isinstance(value, np.ndarray):
            if hasattr(value, "detach"):  # torch.Tensor duck-type check
                import torch
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if value.shape and value.shape[0] == 1:
            value = value[0]
        return value.copy()

    # ══════════════════════════════════════════════════════════════
    # gym.Env interface
    # ══════════════════════════════════════════════════════════════

    @property
    def observation_space(self) -> gym.spaces.Dict:
        if self._obs_space is not None:
            return self._obs_space
        return gym.spaces.Dict({
            "task_description": gym.spaces.Text(65536, min_length=0),
        })

    @property
    def action_space(self) -> gym.spaces.Space:
        import inspect
        try:
            sig = inspect.signature(self._unwrap().step)
            action_param = list(sig.parameters.values())[0]
            # Try to get action size from the inner env's action_space
            inner = self._unwrap()
            for attr in ("action_space", "act_space"):
                if hasattr(inner, attr):
                    return getattr(inner, attr)
        except Exception:
            pass
        return gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None,
              ) -> tuple[dict[str, Any], dict[str, Any]]:
        import inspect
        sig = inspect.signature(self._env.reset)
        kwargs: dict[str, Any] = {}
        if "seed" in sig.parameters and seed is not None:
            kwargs["seed"] = seed
        if "options" in sig.parameters and options is not None:
            kwargs["options"] = options
        result = self._env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            raw_obs, info = result
        else:
            raw_obs, info = result, {}
        obs = self._normalise_obs(raw_obs)
        if self._first_reset:
            self._obs_space = self._build_obs_space(obs)
            self._first_reset = False
        return obs, info

    def step(self, action: Any
             ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        # Convert list/tuple to numpy — some backends (ManiSkill) require it
        if isinstance(action, (list, tuple)):
            action = self._np(action).astype(np.float32)
        raw = self._env.step(action)
        if len(raw) == 5:
            raw_obs, reward, terminated, truncated, info = raw
        elif len(raw) == 4:
            raw_obs, reward, terminated, info = raw
            truncated = False
        else:
            raw_obs, reward, terminated, truncated, info = raw[0], raw[1], raw[2], raw[3], raw[4]
        obs = self._normalise_obs(raw_obs)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self) -> np.ndarray | None:
        if self._render_mode == "human":
            try:
                self._unwrap().render()
            except Exception:
                pass
            return None

        # rgb_array: try direct env render first
        raw = self._env
        if hasattr(raw, "render") and callable(raw.render):
            try:
                return self._finish_render(raw.render())
            except Exception:
                pass
        # Vector env path
        raw2 = self._unwrap()
        if hasattr(raw2, "env") and hasattr(raw2.env, "workers"):
            try:
                return self._finish_render(raw2.env.workers[0]._env.render())
            except Exception:
                pass
        return None

    def _finish_render(self, frame: Any) -> np.ndarray | None:
        """Post-process rendered frame: numpy cast, flip MuJoCo, return RGB."""
        if frame is None:
            return None
        frame = self._np(frame)
        if isinstance(frame, np.ndarray) and frame.ndim >= 3:
            frame = frame[..., :3] if frame.shape[-1] >= 3 else frame
            if self._backend in ("metaworld", "d4rl", "frankasim", "libero"):
                frame = np.flipud(frame)
            return frame
        return None

    def close(self) -> None:
        self._env.close()

    def _build_obs_space(self, obs: dict) -> gym.spaces.Dict:
        spaces: dict[str, gym.spaces.Space] = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                spaces[k] = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=v.shape, dtype=v.dtype)
            elif isinstance(v, str):
                spaces[k] = gym.spaces.Text(65536, min_length=0)
            elif isinstance(v, dict):
                # recursive for camera dict
                inner: dict[str, gym.spaces.Space] = {}
                for ik, iv in v.items():
                    if isinstance(iv, np.ndarray):
                        inner[ik] = gym.spaces.Box(
                            low=-np.inf, high=np.inf, shape=iv.shape, dtype=iv.dtype)
                    elif isinstance(iv, dict):
                        inner[ik] = gym.spaces.Dict({})
                    else:
                        inner[ik] = gym.spaces.Dict({})
                spaces[k] = gym.spaces.Dict(inner) if inner else gym.spaces.Dict({})
            elif isinstance(v, list):
                spaces[k] = gym.spaces.Sequence(gym.spaces.Dict({}))
            else:
                spaces[k] = gym.spaces.Dict({})
        return gym.spaces.Dict(spaces)


# ══════════════════════════════════════════════════════════════════════
# make_unified
# ══════════════════════════════════════════════════════════════════════

def make_unified(env_id: str, *, seed: int = 0, render_mode: str | None = None,
                 **kwargs: Any) -> UnifiedEnv:
    """Create a normalised environment in one call."""
    env = gym.make(env_id, seed=seed, render_mode=render_mode, **kwargs)
    if isinstance(env, UnifiedEnv):
        return env
    return UnifiedEnv(env, render_mode=render_mode)
