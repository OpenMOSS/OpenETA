from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))

from openpi.models_pytorch.ftp1_model_config import FTP1_RESERVED_ACTION_DIM
from openpi.models_pytorch.ftp1_model_config import FTP1_SINGLE_ARM_ACTION_REP_DIM

from .._base_policy import BasePolicy


@dataclass(frozen=True)
class ActionMapping:
    arm_slice: tuple[int, int] | None = None
    gripper_index: int | None = None
    gripper_slot_idx: int = 28


def _canonicalize_action_joint_rep(rep: str | None) -> str | None:
    if rep is None:
        return None
    normalized = str(rep).strip().lower()
    if normalized in {"abs", "absolute"}:
        return "absolute"
    if normalized == "relative":
        return "relative"
    if normalized == "mix":
        return "mix"
    return None


def _resolve_univtac_abs_action_from_ftp1(
    action8_model: np.ndarray,
    qpos8_base: np.ndarray,
    action_joint_rep: str | None,
) -> np.ndarray:
    normalized = _canonicalize_action_joint_rep(action_joint_rep)
    if normalized is None:
        raise ValueError(f"Unsupported FTP1 action_joint_rep: {action_joint_rep!r}")

    action8 = np.asarray(action8_model, dtype=np.float32).reshape(-1)
    qpos8 = np.asarray(qpos8_base, dtype=np.float32).reshape(-1)
    if action8.shape[0] != 8:
        raise ValueError(f"Expected 8D action, got shape {action8.shape}")
    if qpos8.shape[0] != 8:
        raise ValueError(f"Expected 8D qpos base, got shape {qpos8.shape}")

    if normalized == "absolute":
        return action8.copy()
    if normalized == "relative":
        return qpos8 + action8

    resolved = action8.copy()
    resolved[:7] = qpos8[:7] + action8[:7]
    return resolved


def _parse_action_slice(s: str) -> tuple[int, int]:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice '{s}', expected 'start:end'")
    return int(parts[0]), int(parts[1])


def _infer_ftp1_layout(action_dim: int) -> str:
    single = FTP1_SINGLE_ARM_ACTION_REP_DIM
    dual = 2 * single + 9
    dual_with_reserved = dual + FTP1_RESERVED_ACTION_DIM

    if action_dim == single:
        return "single48"
    if action_dim in {dual, dual_with_reserved}:
        return "dual105"
    if action_dim == 91:
        return "legacy91"
    return "unknown"


def _build_ftp1_state_from_univtac_qpos(
    qpos8: np.ndarray,
    action_dim: int,
    mapping: ActionMapping,
) -> np.ndarray:
    qpos8 = np.asarray(qpos8, dtype=np.float32).reshape(-1)
    if qpos8.shape[0] != 8:
        raise ValueError(f"Expected qpos8 dim=8, got {qpos8.shape}")

    layout = _infer_ftp1_layout(action_dim)
    state = np.zeros((1, action_dim), dtype=np.float32)

    if layout in {"single48", "dual105"}:
        right_off = 0
        state[0, right_off + 9 : right_off + 16] = qpos8[:7]
        state[0, right_off + 16 + mapping.gripper_slot_idx] = qpos8[7]
        return state

    if layout == "legacy91":
        state[0, 9 + mapping.gripper_slot_idx] = qpos8[7]
        return state

    if action_dim > 9 + mapping.gripper_slot_idx:
        state[0, 9 + mapping.gripper_slot_idx] = qpos8[7]
    return state


def _extract_univtac_action_from_ftp1(
    action_vec: np.ndarray,
    action_dim: int,
    mapping: ActionMapping,
) -> np.ndarray:
    action_vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if action_vec.shape[0] != action_dim:
        raise ValueError(f"Action dim mismatch: got {action_vec.shape[0]} expected {action_dim}")

    if mapping.arm_slice is not None and mapping.gripper_index is not None:
        s0, s1 = mapping.arm_slice
        arm = action_vec[s0:s1]
        if arm.shape[0] != 7:
            raise ValueError(f"arm_slice must span 7 dims, got {arm.shape[0]}")
        out = np.zeros(8, dtype=np.float32)
        out[:7] = arm
        out[7] = float(action_vec[mapping.gripper_index])
        return out

    layout = _infer_ftp1_layout(action_dim)
    if layout in {"single48", "dual105"}:
        right_off = 0
        out = np.zeros(8, dtype=np.float32)
        out[:7] = action_vec[right_off + 9 : right_off + 16]
        out[7] = float(action_vec[right_off + 16 + mapping.gripper_slot_idx])
        return out

    if layout == "legacy91":
        out = np.zeros(8, dtype=np.float32)
        out[7] = float(action_vec[9 + mapping.gripper_slot_idx])
        return out

    raise ValueError(
        f"Unsupported action_dim={action_dim}. Provide arm_slice/gripper_index in deploy config."
    )


def _infer_action_rep_from_checkpoint(checkpoint_dir: str | Path) -> str | None:
    train_config_path = Path(checkpoint_dir) / "train_config.json"
    if not train_config_path.exists():
        return None
    try:
        with open(train_config_path) as f:
            train_cfg = json.load(f)
    except Exception:
        return None
    return _canonicalize_action_joint_rep(train_cfg.get("action_joint_rep"))


def _to_numpy_uint8(img: Any) -> np.ndarray:
    if isinstance(img, np.ndarray):
        arr = img
    elif isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC image, got {arr.shape}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 255.0)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _resize_hwc_uint8(img: np.ndarray, size: int = 224) -> np.ndarray:
    import cv2

    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


class Policy(BasePolicy):
    """
    FTP1 policy wrapper for UniVTAC's eval pipeline (UniVTAC/scripts/eval_policy.py).

    Example deploy YAML fields:
    - policy_name: FTP1
    - checkpoint_dir: /path/to/ckpt/7999
    - domain_name: YourDomainName
    - device: cuda
    - num_inference_steps: 10
    - chunk_len: 16
    - action_rep: auto
    - tactile_key: right_tactile_gripper
    - tactile_sensor: GelSightMini
    - gripper_slot_idx: 28
    - arm_slice: "9:16"            # optional override (for legacy 91D/custom)
    - gripper_index: 44            # optional override (for legacy 91D/custom)
    """

    def __init__(self, args: dict[str, Any]):
        from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

        self.checkpoint_dir = str(args["checkpoint_dir"])
        self.domain_name = str(args["domain_name"])
        self.device = str(args.get("device", "cuda"))
        self.num_inference_steps = int(args.get("num_inference_steps", 10))
        self.chunk_len = max(1, int(args.get("chunk_len", 16)))
        self.tactile_key = str(args.get("tactile_key", "right_tactile_gripper"))
        self.tactile_sensor = str(args.get("tactile_sensor", "GelSightMini"))
        configured_action_rep = str(args.get("action_rep", "auto"))

        if configured_action_rep == "auto":
            inferred_action_rep = _infer_action_rep_from_checkpoint(self.checkpoint_dir)
            if inferred_action_rep is None:
                self.action_rep = "relative"
                print(
                    "[FTP1 deploy] WARNING: action_rep=auto but train_config.json missing/invalid. "
                    "Fallback to 'relative'. Please set action_rep explicitly for absolute or mix checkpoints.",
                    flush=True,
                )
            else:
                self.action_rep = inferred_action_rep
                print(
                    f"[FTP1 deploy] Auto action_rep resolved to '{self.action_rep}' from "
                    f"{Path(self.checkpoint_dir) / 'train_config.json'}",
                    flush=True,
                )
        else:
            normalized_action_rep = _canonicalize_action_joint_rep(configured_action_rep)
            if normalized_action_rep is None:
                raise ValueError(f"Unsupported action_rep: {configured_action_rep!r}")
            self.action_rep = normalized_action_rep
            print(f"[FTP1 deploy] Using action_rep='{self.action_rep}'", flush=True)

        self.mapping = ActionMapping(
            arm_slice=_parse_action_slice(args["arm_slice"]) if args.get("arm_slice") else None,
            gripper_index=int(args["gripper_index"]) if args.get("gripper_index") is not None else None,
            gripper_slot_idx=int(args.get("gripper_slot_idx", 28)),
        )

        self.wrapper = FTP1InferenceWrapper(
            checkpoint_dir=self.checkpoint_dir,
            domain_name=self.domain_name,
            device=self.device,
            num_inference_steps=self.num_inference_steps,
        )
        self.model_action_dim = int(self.wrapper.get_action_dim())
        self.state_dim = int(self.wrapper.get_state_dim())
        self.action_dim = int(self.wrapper.get_action_dim())
        if self.state_dim != self.action_dim:
            raise ValueError(
                f"State/action dim mismatch: state_dim={self.state_dim}, action_dim={self.action_dim}"
            )
        print(
            "[FTP1 deploy] Loaded wrapper "
            f"(state_dim={self.state_dim}, action_dim={self.action_dim}, "
            f"model_action_dim={self.model_action_dim})",
            flush=True,
        )
        self._rtac_chunk_index_offset = 1

        self._cached_chunk: np.ndarray | None = None
        self._cached_chunk_base_qpos8: np.ndarray | None = None
        self._cached_idx: int = self._rtac_chunk_index_offset

    def reset(self):
        self._cached_chunk = None
        self._cached_chunk_base_qpos8 = None
        self._cached_idx = self._rtac_chunk_index_offset

    def _infer_chunk(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        images: dict[str, np.ndarray] = {}
        if "observation" in observation and "head" in observation["observation"]:
            head = _resize_hwc_uint8(_to_numpy_uint8(observation["observation"]["head"]["rgb"]), 224)
            images["camera_ego_rgb_0"] = head
        if "observation" in observation and "wrist" in observation["observation"]:
            wrist = _resize_hwc_uint8(_to_numpy_uint8(observation["observation"]["wrist"]["rgb"]), 224)
            images["right_wrist_camera_rgb_0"] = wrist

        qpos8 = observation["embodiment"]["joint"][:8]
        if isinstance(qpos8, torch.Tensor):
            qpos8 = qpos8.detach().cpu().numpy()
        qpos8 = np.asarray(qpos8, dtype=np.float32)
        self._cached_chunk_base_qpos8 = qpos8.reshape(8).copy()
        state = _build_ftp1_state_from_univtac_qpos(qpos8, self.state_dim, self.mapping)

        tactiles = None
        tactile_function_areas = None
        tactile_sensors = None
        left_key = "left_gsmini" if "left_gsmini" in observation.get("tactile", {}) else "left_tactile"
        right_key = "right_gsmini" if "right_gsmini" in observation.get("tactile", {}) else "right_tactile"
        if "tactile" in observation and left_key in observation["tactile"] and right_key in observation["tactile"]:
            left = _resize_hwc_uint8(_to_numpy_uint8(observation["tactile"][left_key]["rgb_marker"]), 224)
            right = _resize_hwc_uint8(_to_numpy_uint8(observation["tactile"][right_key]["rgb_marker"]), 224)
            tac = np.stack([left, right], axis=0)[None, ...].astype(np.float32)
            tactiles = {self.tactile_key: tac}
            tactile_function_areas = {self.tactile_key: [0, 1]}
            tactile_sensors = {self.tactile_key: self.tactile_sensor}

        chunk = self.wrapper.infer(
            images=images,
            state=state,
            prompt=prompt,
            tactiles=tactiles,
            tactile_function_areas=tactile_function_areas,
            tactile_sensors=tactile_sensors,
        )
        return np.asarray(chunk, dtype=np.float32)

    def get_action(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        max_chunk_idx_exclusive = None
        if self._cached_chunk is not None:
            max_chunk_idx_exclusive = min(self._rtac_chunk_index_offset + self.chunk_len, self._cached_chunk.shape[0])
        if (
            self._cached_chunk is None
            or self._cached_chunk_base_qpos8 is None
            or max_chunk_idx_exclusive is None
            or self._cached_idx >= max_chunk_idx_exclusive
        ):
            self._cached_chunk = self._infer_chunk(observation, prompt)
            if self._cached_chunk.ndim != 2 or self._cached_chunk.shape[0] <= self._rtac_chunk_index_offset:
                raise ValueError(
                    f"Unexpected FTP1 output shape {self._cached_chunk.shape}; need at least "
                    f"{self._rtac_chunk_index_offset + 1} steps to execute chunk[1]."
                )
            if self._cached_chunk.shape[1] != self.action_dim:
                raise ValueError(
                    f"Unexpected FTP1 action dim from wrapper.infer: {self._cached_chunk.shape[1]} "
                    f"(expected raw_action_dim={self.action_dim})"
                )
            self._cached_idx = self._rtac_chunk_index_offset
        action_vec = self._cached_chunk[self._cached_idx]
        self._cached_idx += 1
        action8_model = _extract_univtac_action_from_ftp1(action_vec, self.action_dim, self.mapping)
        return _resolve_univtac_abs_action_from_ftp1(
            action8_model,
            self._cached_chunk_base_qpos8,
            self.action_rep,
        )

    def eval(self, task, observation):
        prompt = str(getattr(task, "instruction", ""))
        action8 = self.get_action(observation, prompt)
        action_tensor = torch.from_numpy(action8).to(task.device).float()
        task.take_action(action_tensor, action_type="qpos")
