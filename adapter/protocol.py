"""Shared data structures between simulator adapters and code agents.

Every type in this module is a ``@dataclass(slots=True)`` with
``to_dict()`` / ``from_dict()`` for round-trip serialisation::

    # UnifiedEnv raw dict → typed observation
    obs = EnvObservation.from_dict(unified_env_obs)

    # Typed observation → JSON-serialisable dict
    payload = obs.to_dict()

    # JSON round-trip
    json_str = to_json(obs)
    obs2 = from_json(json_str, cls=EnvObservation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

JsonDict = dict[str, Any]

_T = TypeVar("_T")


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _to_float_list(value: Any) -> list[float]:
    """Convert array-like (numpy, torch, list) → ``list[float]``."""
    if value is None:
        return []
    # numpy array / torch tensor
    if hasattr(value, "tolist"):
        flat = value.tolist()
        if isinstance(flat, list):
            return [float(x) for x in _flatten_once(flat)]
        return [float(flat)]
    if isinstance(value, (list, tuple)):
        return [float(x) for x in _flatten_once(value)]
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _to_int_list3d(value: Any) -> list[list[list[int]]]:
    """Convert image array (H,W,3) → ``list[list[list[int]]]``."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    # Ensure 3-deep nesting of ints
    result: list[list[list[int]]] = []
    for row in value:
        if not isinstance(row, list):
            break
        out_row: list[list[int]] = []
        for pix in row:
            if isinstance(pix, list):
                out_row.append([int(c) for c in pix[:3]])
            else:
                out_row.append([int(pix)])
        if out_row:
            result.append(out_row)
    return result


def _to_float_list2d(value: Any) -> list[list[float]] | None:
    """Convert depth array (H,W) → ``list[list[float]]`` or ``None``."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return None
    result: list[list[float]] = []
    for row in value:
        if isinstance(row, list):
            result.append([float(x) for x in row])
        else:
            result.append([float(row)])
    return result


def _flatten_once(lst: list) -> list:
    """Flatten one nesting level: [[a],[b]] → [a,b]."""
    out: list = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            out.extend(item)
        else:
            out.append(item)
    return out


def _ensure_plain_dict(d: Any) -> JsonDict:
    """Copies a dict, converting numpy scalars to Python primitives."""
    if not isinstance(d, dict):
        return {}
    result: JsonDict = {}
    for k, v in d.items():
        if hasattr(v, "item"):           # numpy scalar
            result[k] = v.item()
        elif hasattr(v, "tolist"):       # numpy array
            result[k] = v.tolist()
        elif isinstance(v, dict):
            result[k] = _ensure_plain_dict(v)
        else:
            result[k] = v
    return result


def _encode_pixels_to_base64(pixels: Any, *, mode: str = "RGB") -> tuple[str, int, int] | tuple[None, None, None]:
    """Encode a pixel array (list-of-lists or numpy) to base64 PNG.

    *mode* ``"RGB"`` → 8-bit colour PNG.
    *mode* ``"depth"`` → 16-bit greyscale PNG, uint16 millimetres
    (``depth_m = pixel_value / 1000.0``).
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    arr = np.asarray(pixels)
    if arr.ndim < 2:
        return None, None, None
    h, w = arr.shape[:2]

    if mode == "depth":
        arr = arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] >= 1:
            arr = arr[:, :, 0]
        # Depth can contain NaN/inf when the physics diverges (e.g. after a
        # long move pushes the arm into an unstable configuration).  NaN/inf
        # survives ``round``+``clip`` and turns into garbage on the uint16
        # cast, corrupting the encoded PNG.  Replace non-finite pixels with 0
        # (= "no reading") BEFORE scaling so the depth image stays clean.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        # Encode as uint16 millimetres (fixed scale, NOT per-frame
        # normalisation).  depth_metres = pixel_value / 1000.0.
        # Max representable depth: 65.535 m.  Resolution: 1 mm.
        arr = np.clip(np.round(arr * 1000.0), 0, 65535).astype(np.uint16)
    else:
        arr = arr[..., :3].astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode(), w, h


def _depth_range(depth_pixels: Any) -> tuple[float, float]:
    """Return (min, max) of a depth array for dynamic-range hints.

    The depth PNG is encoded as uint16 millimetres::

        depth_m = pixel_value / 1000.0

    ``depth_min`` and ``depth_max`` in the MCP dict record the actual
    scene depth range in metres for informational purposes — they are
    **not** needed for reconstruction.
    """
    import numpy as np
    arr = np.asarray(depth_pixels, dtype=np.float32)
    # Ignore non-finite pixels so a single NaN/inf doesn't poison the whole
    # range (``arr.min()`` returns NaN if any element is NaN).  If every pixel
    # is non-finite, fall back to a degenerate 0.0–0.0 range.
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 0.0
    return float(finite.min()), float(finite.max())


# ══════════════════════════════════════════════════════════════════════
# Protocol data classes
# ══════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class CameraFrame:
    """RGBD camera frame in JSON-serializable form for the initial bridge.

    ``role`` is an optional backend-neutral semantic hint such as
    ``scene_primary`` or ``wrist_primary``.  ``frame_id`` remains the stable
    backend identifier and is never rewritten to emulate another simulator.
    """

    frame_id: str
    rgb: list[list[list[int]]]
    depth: list[list[float]] | None = None
    intrinsics: JsonDict = field(default_factory=dict)
    extrinsics: JsonDict = field(default_factory=dict)
    timestamp_s: float | None = None
    role: str = ""

    def to_dict(self) -> JsonDict:
        """Convert to a plain JSON-serialisable dict."""
        d: JsonDict = {
            "frame_id": self.frame_id,
            "rgb": self.rgb,
            "depth": self.depth,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
        }
        if self.timestamp_s is not None:
            d["timestamp_s"] = self.timestamp_s
        if self.role:
            d["role"] = self.role
        return d

    @classmethod
    def from_dict(cls, d: dict, *, frame_id: str = "") -> CameraFrame:
        """Create from a dict (serialised or UnifiedEnv camera entry).

        Parameters
        ----------
        d:
            Either a serialised ``CameraFrame`` dict with ``frame_id``,
            or a UnifiedEnv camera entry (``{"rgb": ndarray, ...}``).
        frame_id:
            Fallback camera name when *d* doesn't carry ``frame_id``.
        """
        fid: str = d.get("frame_id", frame_id)
        rgb = d.get("rgb")
        if rgb is not None:
            rgb = _to_int_list3d(rgb)
        else:
            rgb = []
        depth = _to_float_list2d(d.get("depth"))
        return cls(
            frame_id=fid,
            rgb=rgb,
            depth=depth,
            intrinsics=_ensure_plain_dict(d.get("intrinsics")),
            extrinsics=_ensure_plain_dict(d.get("extrinsics")),
            timestamp_s=d.get("timestamp_s"),
            role=str(d.get("role") or ""),
        )

    def to_mcp_dict(self) -> JsonDict:
        """Serialize for MCP transport: ``rgb`` → ``rgb_base64`` PNG.

        The ``rgb`` pixel list is encoded as a base64 PNG string so the
        JSON payload stays compact.

        **Depth is already linear metric depth in metres.**  The simulator
        layer linearises the raw MuJoCo z-buffer (and converts ManiSkill's
        int16 millimetres) *before* serialization — no client-side
        conversion is needed.  On the wire it is encoded as a uint16 PNG in
        millimetres, so recover metres with::

            depth_m = depth_uint16 / 1000.0

        Newly normalized OpenCV packets make that wire representation
        machine-readable with ``depth_encoding="uint16_png"`` and
        ``depth_scale=1000.0`` (encoded units per metre).  Legacy packets
        intentionally retain their existing shape for reproducibility.

        Values fall within ``[znear, zfar]`` from ``intrinsics`` (metric
        clip planes, in metres).

        **Extrinsics convention.**  Every dict is self-describing: consumers
        must read ``camera_frame`` instead of inferring a convention from the
        simulator name or matrix shape.

        Agent-facing RoboCasa and BEHAVIOR adapters normalize the pose to the
        same **OpenCV optical** frame used by their returned RGB, metric depth,
        and intrinsics (+X right, +Y down, +Z forward):

        * ``matrix_layout`` — ``"row_major"``
        * ``frame_transform`` — ``"camera_to_world"``
        * ``camera_frame`` — ``"opencv"``
        * ``image_origin`` — ``"top_left"``
        * ``pos`` — ``[x, y, z]`` camera position in **world** coordinates
          (metres), not relative to the end-effector
        * ``mat`` — 3×3 rotation matrix, **camera → world**, flattened
          row-major. ``camera_to_world`` also carries the equivalent 4×4
          homogeneous matrix.
        * ``normalized_from`` / ``raw_camera_convention`` — optional debug
          provenance for the renderer frame normalized by the adapter.

        For these normalized packets, pinhole deprojection is direct::

            x = (u - cx) * d / fx
            y = (v - cy) * d / fy
            p_opencv = np.array([x, y, d])
            R = np.array(mat).reshape(3, 3)
            p_world = R @ p_opencv + pos

        LIBERO is deliberately kept on its existing reproducible v1 packet:
        ``camera_frame="opengl"``, where local +Z is the renderer's backward
        axis and the camera looks along -Z.  MetaWorld and other legacy
        MuJoCo adapters currently use the same form.  For those packets,
        convert the OpenCV point with ``diag(1, -1, -1)`` before applying
        ``R``.  ManiSkill remains self-described as ``camera_frame="ros"``.
        The generic ``camera_pose_to_world`` tool accepts these legacy forms;
        no consumer should guess from the backend name.
        """
        d: JsonDict = {
            "frame_id": self.frame_id,
            "rgb_base64": None,
            "width": 0,
            "height": 0,
            "depth_base64": None,
            "intrinsics": self.intrinsics,
            "extrinsics": self.extrinsics,
        }
        if self.timestamp_s is not None:
            d["timestamp_s"] = self.timestamp_s
        if self.role:
            d["role"] = self.role

        # RGB → base64 PNG
        if self.rgb:
            try:
                enc, w, h = _encode_pixels_to_base64(self.rgb, mode="RGB")
                if enc:
                    d["rgb_base64"] = enc
                    d["width"] = w
                    d["height"] = h
            except Exception:
                pass

        # Depth → uint16 PNG in fixed millimetres. depth_min/depth_max remain
        # informational scene-range hints; reconstruction is always
        # depth_m = pixel / 1000.0.
        if self.depth:
            try:
                _dmin, _dmax = _depth_range(self.depth)
                denc, dw, dh = _encode_pixels_to_base64(self.depth, mode="depth")
                if denc:
                    d["depth_base64"] = denc
                    d["depth_min"] = _dmin
                    d["depth_max"] = _dmax
                    if (
                        self.extrinsics.get("camera_frame") == "opencv"
                        and self.extrinsics.get("normalized_from")
                    ):
                        # Additive metadata for the new canonical camera
                        # contract.  Do not alter LIBERO's legacy OpenGL
                        # packet, whose exact wire shape is reproducibility
                        # sensitive.
                        d["depth_encoding"] = "uint16_png"
                        d["depth_scale"] = 1000.0
                    if not d["width"]:
                        d["width"] = dw
                        d["height"] = dh
            except Exception:
                pass

        return d


@dataclass(slots=True)
class RobotState:
    """Robot proprioceptive state exposed to the agent."""

    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    end_effector_pose: JsonDict = field(default_factory=dict)
    gripper_state: JsonDict = field(default_factory=dict)
    base_pose: JsonDict | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Convert to a plain JSON-serialisable dict."""
        d: JsonDict = {
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "end_effector_pose": self.end_effector_pose,
            "gripper_state": self.gripper_state,
        }
        if self.base_pose is not None:
            d["base_pose"] = self.base_pose
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RobotState:
        """Create from a dict (serialised or UnifiedEnv ``proprio`` entry).

        Handles both::

            # Serialised form
            {"joint_positions": [0.1, ...], "end_effector_pose": {"xyz": [...], ...}}

            # UnifiedEnv proprio form
            {"joint_positions": np.ndarray, "ee_pose": np.ndarray, "gripper_open": 0.8}
        """
        # joint positions / velocities
        jp = _to_float_list(d.get("joint_positions", []))
        jv_raw = d.get("joint_velocities")
        jv: list[float] = _to_float_list(jv_raw) if jv_raw is not None else []

        # end-effector pose — may be a dict or a flat array
        ee = d.get("end_effector_pose") or d.get("ee_pose")
        ee_dict: JsonDict = {}
        if ee is not None:
            if isinstance(ee, dict):
                ee_dict = _ensure_plain_dict(ee)
            else:
                arr = _to_float_list(ee)
                if len(arr) >= 7:
                    ee_dict = {"xyz": arr[:3], "quat_xyzw": arr[3:7]}
                elif len(arr) >= 3:
                    ee_dict = {"xyz": arr[:3]}

        # gripper — may be a dict or a scalar.  We emit a *continuous*
        # ``openness`` in [0, 1] (0 = fully closed, 1 = fully open) so callers
        # can tell a half-grasp from a full one; ``open`` (bool, threshold 0.5)
        # is kept for backward compatibility.
        gs: JsonDict = {}
        gs_raw = d.get("gripper_state")
        if gs_raw and isinstance(gs_raw, dict):
            gs = _ensure_plain_dict(gs_raw)
            # If the upstream dict carries a numeric openness but no explicit
            # continuous field, normalise it so consumers get a float too.
            if "openness" not in gs:
                _src = gs.get("open")
                if isinstance(_src, (int, float)) and not isinstance(_src, bool):
                    ov = max(0.0, min(1.0, float(_src)))
                    gs["openness"] = ov
                    gs["open"] = ov > 0.5
        else:
            go = d.get("gripper_open")
            if go is not None:
                val = go.item() if hasattr(go, "item") else float(go)
                ov = max(0.0, min(1.0, float(val)))
                gs = {"openness": ov, "open": ov > 0.5}

        # base pose
        bp = d.get("base_pose")
        if bp is not None and isinstance(bp, dict):
            bp = _ensure_plain_dict(bp)

        # metadata
        meta = _ensure_plain_dict(d.get("metadata", {}))

        return cls(
            joint_positions=jp,
            joint_velocities=jv,
            end_effector_pose=ee_dict,
            gripper_state=gs,
            base_pose=bp,
            metadata=meta,
        )


@dataclass(slots=True)
class EnvObservation:
    """Observation packet sent from simulator to agent."""

    task: str
    cameras: list[CameraFrame]
    robot: RobotState
    objects: list[JsonDict] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Convert to a plain JSON-serialisable dict."""
        return {
            "task": self.task,
            "cameras": [c.to_dict() for c in self.cameras],
            "robot": self.robot.to_dict(),
            "objects": self.objects,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict, *, task: str | None = None) -> EnvObservation:
        """Create from a dict — accepts **both** serialised and UnifiedEnv formats.

        UnifiedEnv format::

            {"cameras": {"agentview": {"rgb": ndarray, ...}}, "proprio": {...},
             "task_description": "...", "objects": [...], "metadata": {...}}

        Serialised format::

            {"task": "...", "cameras": [{...}, ...], "robot": {...}, ...}
        """
        # ── cameras ────────────────────────────────────────────
        cameras_raw = d.get("cameras", {})
        cameras: list[CameraFrame] = []

        if isinstance(cameras_raw, dict):
            # UnifiedEnv format: camera name → {rgb, depth, ...}
            for cam_name, cam_data in cameras_raw.items():
                if isinstance(cam_data, dict):
                    cameras.append(CameraFrame.from_dict(cam_data, frame_id=cam_name))
        elif isinstance(cameras_raw, list):
            # Serialised format: list of CameraFrame dicts
            for cam_data in cameras_raw:
                if isinstance(cam_data, dict):
                    cameras.append(CameraFrame.from_dict(cam_data))

        # ── robot ──────────────────────────────────────────────
        # Serialised form uses "robot", UnifiedEnv uses "proprio"
        robot_raw = d.get("robot")
        if robot_raw is None:
            robot_raw = d.get("proprio", {})
        if isinstance(robot_raw, RobotState):
            robot = robot_raw
        elif isinstance(robot_raw, dict):
            robot = RobotState.from_dict(robot_raw)
        else:
            robot = RobotState()

        # ── task ───────────────────────────────────────────────
        if task is None:
            task = d.get("task") or d.get("task_description", "")

        # ── objects ────────────────────────────────────────────
        objs = d.get("objects", [])
        if isinstance(objs, list):
            objects = [_ensure_plain_dict(o) if isinstance(o, dict) else o for o in objs]
        else:
            objects = []

        # ── metadata ───────────────────────────────────────────
        metadata = _ensure_plain_dict(d.get("metadata", {}))

        return cls(
            task=task,
            cameras=cameras,
            robot=robot,
            objects=objects,
            metadata=metadata,
        )

    def to_mcp_dict(self) -> JsonDict:
        """Serialize for MCP transport — camera frames carry ``rgb_base64``.

        This is the preferred serialisation for network transport (MCP
        tools, REST API) because base64-encoded PNGs are far smaller than
        raw pixel lists.
        """
        return {
            "task": self.task,
            "cameras": [c.to_mcp_dict() for c in self.cameras],
            "robot": self.robot.to_dict(),
            "objects": self.objects,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EnvAction:
    """Action packet sent from agent to simulator."""

    action_type: str
    code: str | None = None
    command: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Convert to a plain JSON-serialisable dict."""
        d: JsonDict = {"action_type": self.action_type}
        if self.code is not None:
            d["code"] = self.code
        if self.command:
            d["command"] = self.command
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EnvAction:
        """Create from a serialised or API dict."""
        return cls(
            action_type=d.get("action_type", ""),
            code=d.get("code"),
            command=_ensure_plain_dict(d.get("command", {})),
            metadata=_ensure_plain_dict(d.get("metadata", {})),
        )


@dataclass(slots=True)
class StepResult:
    """Result of applying one action in the simulator."""

    observation: EnvObservation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Convert to a plain JSON-serialisable dict."""
        return {
            "observation": self.observation.to_dict(),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StepResult:
        """Create from a serialised dict."""
        obs_raw = d.get("observation", {})
        if isinstance(obs_raw, EnvObservation):
            obs = obs_raw
        elif isinstance(obs_raw, dict):
            obs = EnvObservation.from_dict(obs_raw)
        else:
            obs = EnvObservation(task="", cameras=[], robot=RobotState())

        return cls(
            observation=obs,
            reward=float(d.get("reward", 0.0)),
            terminated=bool(d.get("terminated", False)),
            truncated=bool(d.get("truncated", False)),
            info=_ensure_plain_dict(d.get("info", {})),
        )

    def to_mcp_dict(self) -> JsonDict:
        """Serialize for MCP transport — observation uses ``to_mcp_dict``."""
        return {
            "observation": self.observation.to_mcp_dict(),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }


# ══════════════════════════════════════════════════════════════════════
# JSON serialisation helpers
# ══════════════════════════════════════════════════════════════════════

_JSONABLE_TYPES = (CameraFrame, RobotState, EnvObservation, EnvAction, StepResult)


class _ProtocolEncoder(json.JSONEncoder):
    """JSON encoder that calls ``to_dict()`` on protocol data classes."""

    def default(self, obj: Any) -> Any:
        if hasattr(obj, "to_dict"):
            return obj.to_dict()  # type: ignore[union-attr]
        return super().default(obj)


def to_json(obj: Any, *, indent: int | None = None) -> str:
    """Serialize any protocol data class to a JSON string.

    Usage::

        json_str = to_json(obs)
        json_str = to_json(step_result, indent=2)
    """
    return json.dumps(obj, cls=_ProtocolEncoder, indent=indent)


def from_json(s: str, *, cls: type[_T] = EnvObservation) -> _T:  # type: ignore[type-var]
    """Deserialize a JSON string to a protocol data class.

    Usage::

        obs = from_json(json_str)                    # → EnvObservation
        action = from_json(json_str, cls=EnvAction)   # → EnvAction
    """
    d = json.loads(s)
    return cls.from_dict(d)  # type: ignore[attr-defined]
