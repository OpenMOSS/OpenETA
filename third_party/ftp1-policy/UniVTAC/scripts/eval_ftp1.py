"""
Evaluate FTP1 policy on UniVTAC tasks (Isaac Lab + TacEx).

Key goals:
- Only load FTP1 model once per process (FTP1 checkpoint is large).
- Optionally evaluate multiple tasks in parallel (each worker process loads one FTP1).
- Keep preprocessing consistent with UniVTAC -> FTP1 training (see below).

Training vs Eval consistency (state / image / action / robot):
- State: absolute qpos 8D (7 arm + 1 gripper). Training uses proprioception_joint_rep='abs';
  eval builds state from observation["embodiment"]["joint"][:8] and fills [9:16]=arm, [16+28]=gripper,
  pose9d left zeros (UniVTAC zarr has no wrist_pose). Gripper uses the same absolute qpos
  convention in both data_processing and eval (no affine flip).
- Image: 224x224, BGR (training zarr from parse_data_univtac uses cv2.imdecode → BGR; eval converts
  sim RGB to BGR before resize so model sees same channel order).
- Tactile: (1,2,224,224,3) BGR; two pads: [left_tac (thumb), right_tac (index)] matching
  parse_data_univtac.py lines 147-166 which reads both left_gsmini AND right_gsmini separately.
- Action: action representation must match training (`action_joint_rep`).
  This script supports relative, absolute, and mix outputs.
  Use `--action_rep auto` (default) to infer from checkpoint `train_config.json`.
- Robot: one action per take_action(); task does set_arm(action[:-1]), set_gripper(action[-1]), _step().
  Decimation and step_lim come from task_config; control semantics match training (target qpos).
- Chunking (FTP1 vs TactileACT alignment):
  FTP1 chunk[0] is treated as current-timestamp placeholder (invalid for execution),
  so execution starts from chunk[1]. This matches the convention:
  chunk_ftp1[t] = chunk_tactileact[t-1].
"""

from __future__ import annotations

# Force JAX to CPU only before any import can load JAX (avoids conflict with Isaac Sim on GPU → process kill)
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import cv2
import importlib
import json
import re
import sys
import time
import traceback
import yaml
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from eval_result_utils import build_run_id, load_worker_metadata, summarize_worker_metadata, worker_name, write_json
from openpi.models_pytorch.ftp1_model_config import FTP1_RESERVED_ACTION_DIM
from openpi.models_pytorch.ftp1_model_config import FTP1_SINGLE_ARM_ACTION_REP_DIM
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper


# Keep sys.path consistent with other UniVTAC eval scripts
sys.path.insert(0, ".")
sys.path.insert(0, "./policy")

DEFAULT_ENSEMBLE_K = 0.01
DEFAULT_FTP1_SAVE_ROOT = "eval_result/FTP1"
_EPISODE_VIDEO_RESULT_RE = re.compile(r"^(?P<seed>\d+)_(?P<result>[A-Za-z0-9]+)\.mp4$")
_EPISODE_VIDEO_WITHOUT_RESULT_RE = re.compile(r"^(?P<seed>\d+)\.mp4$")
_COMPLETED_VIDEO_RESULTS = {"success", "failed"}


def _rewrite_cli_device_flags(argv: list[str]) -> list[str]:
    """Keep legacy FTP1 `--device` working while freeing `--device` for AppLauncher."""
    has_ftp1_device = any(arg == "--ftp1_device" or arg.startswith("--ftp1_device=") for arg in argv)
    has_sim_device = any(arg == "--sim_device" or arg.startswith("--sim_device=") for arg in argv)

    rewritten: list[str] = []
    rewrote_legacy_ftp1 = False
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--sim_device":
            rewritten.append("--device")
            if i + 1 >= len(argv):
                raise ValueError("--sim_device requires a value")
            rewritten.append(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--sim_device="):
            rewritten.append("--device=" + arg.split("=", 1)[1])
            i += 1
            continue
        if not has_ftp1_device and not has_sim_device and arg == "--device":
            rewritten.append("--ftp1_device")
            if i + 1 >= len(argv):
                raise ValueError("--device requires a value")
            rewritten.append(argv[i + 1])
            rewrote_legacy_ftp1 = True
            i += 2
            continue
        if not has_ftp1_device and not has_sim_device and arg.startswith("--device="):
            rewritten.append("--ftp1_device=" + arg.split("=", 1)[1])
            rewrote_legacy_ftp1 = True
            i += 1
            continue
        rewritten.append(arg)
        i += 1

    if rewrote_legacy_ftp1:
        print(
            "[eval_ftp1] WARNING: legacy --device was interpreted as FTP1 policy device. "
            "Use --ftp1_device for FTP1 and --sim_device for Isaac Sim.",
            flush=True,
        )
    return rewritten


def _canonicalize_action_joint_rep(rep: str | None) -> Literal["relative", "absolute", "mix"] | None:
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

def _parse_csv(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    if path.suffix in (".yml", ".yaml"):
        with open(path, "r") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config type: {path}")


def _resolve_task_config_path(task_config: str, *, must_exist: bool = True) -> Path | None:
    if not task_config:
        return None

    task_cfg_path = Path(task_config)
    if task_cfg_path.exists():
        return task_cfg_path

    candidate = Path(__file__).parent.parent / "task_config" / task_config
    if candidate.exists():
        return candidate

    if must_exist:
        raise FileNotFoundError(f"Task config not found: {task_config}")
    return candidate


def _infer_action_rep_from_checkpoint(checkpoint_dir: str | Path) -> str | None:
    """Infer action representation from checkpoint train_config.json."""
    train_config_path = Path(checkpoint_dir) / "train_config.json"
    if not train_config_path.exists():
        return None
    try:
        with open(train_config_path, "r") as f:
            train_cfg = json.load(f)
    except Exception:
        return None

    return _canonicalize_action_joint_rep(train_cfg.get("action_joint_rep"))


def _to_numpy_uint8(img: Any) -> np.ndarray:
    """Convert torch/np/PIL-like image to HWC uint8 numpy."""
    if img is None:
        raise ValueError("img is None")
    if isinstance(img, np.ndarray):
        arr = img
    else:
        try:
            if isinstance(img, torch.Tensor):
                arr = img.detach().cpu().numpy()
            else:
                arr = np.asarray(img)
        except Exception:
            arr = np.asarray(img)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {arr.shape}")
    if arr.dtype != np.uint8:
        # common case: float in [0,1] or [0,255]
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 255.0)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def _resize_hwc_uint8(img: np.ndarray, size: int = 224) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _save_infer_input_sample(
    save_dir: Path,
    images: dict[str, np.ndarray],
    state: np.ndarray,
    tactiles: dict[str, np.ndarray] | None,
    prompt: str,
) -> None:
    """Save one sample of the exact inputs passed to the model (224x224, BGR)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for key, img in images.items():
        if img is not None and img.size > 0:
            out = save_dir / f"{key.replace('/', '_')}.png"
            cv2.imwrite(str(out), img)
    tac_strips = []
    if tactiles is not None:
        for key, tac in tactiles.items():
            for i in range(tac.shape[1]):
                frame = tac[0, i]
                cv2.imwrite(str(save_dir / f"tactile_{key.replace('/', '_')}_pad{i}.png"), frame)
                tac_strips.append(frame)
    # One preview image: [head | wrist | tac0 | tac1] side by side (all 224 wide)
    parts = []
    for k in ["camera_ego_rgb_0", "right_wrist_camera_rgb_0"]:
        if images.get(k) is not None and images[k].size > 0:
            parts.append(images[k])
    parts.extend(tac_strips[:2])
    if parts:
        preview = np.concatenate(parts, axis=1)  # (224, 224*N, 3)
        cv2.imwrite(str(save_dir / "model_input_preview.png"), preview)
    (save_dir / "state.txt").write_text(f"state shape={state.shape}\nstate[0] (first 16)={state[0, :16].tolist()}")
    (save_dir / "prompt.txt").write_text(prompt or "")
    readme = """Images here are exactly what is fed to the FTP1 model (224x224, BGR).
- camera_ego_rgb_0.png = head camera
- right_wrist_camera_rgb_0.png = wrist camera
- tactile_*_pad0.png, pad1.png = two tactile pads
- model_input_preview.png = [head | wrist | tac0 | tac1] concatenated
Channel order is BGR (same as training from parse_data imdecode).
"""
    (save_dir / "README.txt").write_text(readme)


def _get_task_instruction(task_name: str) -> str:
    # Fallback instructions aligned with parse_data_univtac.py instruction_map
    instruction_map = {
        "grasp_classify": "use grasped tool for tactile sensing to move to target surface.",
        "insert_HDMI": "insert the HDMI to the fixed slot.",
        "insert_hole": "insert the stick to the hole.",
        "insert_tube": "insert the tube to the fixed slot.",
        "lift_can": "grasp the can and lifts it vertically without slippage.",
        "lift_bottle": "grasp the bottle and lift it vertically, keeping its final base within 5 cm of the wall.",
        "pull_out_key": "pull out the key.",
        "put_bottle_in_shelf": "grasp the bottle, then position it into the shelf cavity.",
    }
    return instruction_map[task_name]


@dataclass(frozen=True)
class ActionMapping:
    # If provided, override the default layout-based extraction
    arm_slice: tuple[int, int] | None = None  # [start, end) in FTP1 action vector for 7 arm joints
    gripper_index: int | None = None  # absolute index in FTP1 action vector for gripper scalar
    gripper_slot_idx: int = 28  # slot in 32-slot hand vector (parse_data_univtac uses 28)


def _parse_slice(s: str) -> tuple[int, int]:
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid slice '{s}', expected 'start:end'")
    return int(parts[0]), int(parts[1])


def _infer_layout(action_dim: int) -> Literal["single48", "dual105", "legacy91", "unknown"]:
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


def _build_state_from_univtac_qpos(
    qpos8: np.ndarray, action_dim: int, mapping: ActionMapping
) -> np.ndarray:
    qpos8 = np.asarray(qpos8, dtype=np.float32).reshape(-1)
    if qpos8.shape[0] != 8:
        raise ValueError(f"Expected qpos8 dim=8, got {qpos8.shape}")

    layout = _infer_layout(action_dim)
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
    action_vec: np.ndarray, action_dim: int, mapping: ActionMapping, action_rep: str
) -> np.ndarray:
    del action_rep  # semantics are applied when converting to absolute qpos targets

    action_vec = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    if action_vec.shape[0] != action_dim:
        raise ValueError(f"Action dim mismatch: got {action_vec.shape[0]} expected {action_dim}")

    if mapping.arm_slice is not None and mapping.gripper_index is not None:
        s0, s1 = mapping.arm_slice
        arm = action_vec[s0:s1]
        if arm.shape[0] != 7:
            raise ValueError(f"--arm_slice must span 7 dims, got {arm.shape[0]}")
        out = np.zeros(8, dtype=np.float32)
        out[:7] = arm
        out[7] = float(action_vec[mapping.gripper_index])
        return out

    layout = _infer_layout(action_dim)
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
        f"Unsupported action_dim={action_dim}. Provide --arm_slice and --gripper_index to map 91D/other to 8D."
    )


def _get_qpos8(observation: dict[str, Any]) -> np.ndarray:
    """Extract 8D qpos from observation as float32 numpy array."""
    qpos8 = observation["embodiment"]["joint"][:8]
    if isinstance(qpos8, torch.Tensor):
        qpos8 = qpos8.detach().cpu().numpy()
    return np.asarray(qpos8, dtype=np.float32).reshape(8)


def _load_task_settings() -> dict[str, Any]:
    """Load UniVTAC policy/task_settings.json; keys are task names, values have e.g. camera_type: 'head' | 'all'."""
    path = Path(__file__).resolve().parent.parent / "policy" / "task_settings.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _to_float_list(values: Any) -> list[float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return [float(x) for x in arr.tolist()]


def _jsonify_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify_value(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonify_value(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value

class FTP1ChunkPolicy:
    def __init__(
        self,
        checkpoint_dir: str,
        domain_name: str,
        device: str,
        num_inference_steps: int,
        tactile_key: str,
        tactile_sensor: str,
        mapping: ActionMapping,
        chunk_first_n: int = 0,
        action_rep: str = "relative",
        save_infer_input_dir: str | Path | None = None,
        use_temporal_ensemble: bool = False,
        ensemble_K: float = DEFAULT_ENSEMBLE_K,
    ):
        normalized_action_rep = _canonicalize_action_joint_rep(action_rep)
        if normalized_action_rep is None:
            raise ValueError(f"Unsupported action_rep: {action_rep!r}")
        self.action_rep = normalized_action_rep
        self.save_infer_input_dir = Path(save_infer_input_dir) if save_infer_input_dir else None
        self._saved_infer_input = False
        self._task_settings = _load_task_settings()
        self._current_task_name: str | None = None
        self.wrapper = FTP1InferenceWrapper(
            checkpoint_dir=checkpoint_dir,
            domain_name=domain_name,
            device=device,
            num_inference_steps=num_inference_steps,
        )
        self.model_action_dim = int(self.wrapper.get_action_dim())
        self.state_dim = int(self.wrapper.get_state_dim())
        self.action_dim = int(self.wrapper.get_action_dim())
        self.use_tactile_input = bool(getattr(self.wrapper.model_config, "use_tactile_input", True))
        if self.state_dim != self.action_dim:
            raise ValueError(
                f"State/action dim mismatch: state_dim={self.state_dim}, action_dim={self.action_dim}"
            )
        self.mapping = mapping
        self.tactile_key = tactile_key
        self.tactile_sensor = tactile_sensor
        self.chunk_first_n = max(0, int(chunk_first_n))
        self.use_temporal_ensemble = use_temporal_ensemble
        self.ensemble_K = ensemble_K
        # Align FTP1 chunk semantics to TactileACT convention:
        # FTP1 chunk[0] corresponds to current timestamp placeholder (invalid),
        # and first executable future action is chunk[1].
        self._rtac_chunk_index_offset = 1

        self._cached_chunk: np.ndarray | None = None
        self._cached_idx: int = 0
        # (predicted_chunk, infer_exec_step, qpos8_at_infer_step)
        self._chunk_history: list[tuple[np.ndarray, int, np.ndarray]] = []
        self._exec_step: int = 0
        self._last_action_debug: dict[str, Any] | None = None

        if not self.use_tactile_input:
            print(
                "[eval_ftp1] Checkpoint disables tactile input (use_tactile_input=False); runtime tactile observations will be ignored.",
                flush=True,
            )

    def reset(self):
        self._cached_chunk = None
        self._cached_idx = 0
        self._chunk_history = []
        self._exec_step = 0
        self._last_action_debug = None

    def set_task(self, task_name: str) -> None:
        """Set current task name so infer_chunk can use task_settings (e.g. camera_type) for this task."""
        self._current_task_name = task_name
        print("#"*80)
        print(f"[eval_ftp1] Set task {task_name}")

    def _use_wrist_camera(self) -> bool:
        """True if current task uses wrist camera (camera_type == 'all'); else head-only."""
        if self._current_task_name is None:
            return True  # default: use wrist if observation provides it (backward compatible)
        cfg = self._task_settings.get(self._current_task_name, {})
        return cfg.get("camera_type", "all") == "all"

    def infer_chunk(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        images: dict[str, np.ndarray] = {}
        if "observation" in observation and "head" in observation["observation"]:
            # TODO: verification
            head = _to_numpy_uint8(observation["observation"]["head"]["rgb"])
            head = _resize_hwc_uint8(head, 224)
            images["camera_ego_rgb_0"] = head
        if self._use_wrist_camera() and "observation" in observation and "wrist" in observation["observation"]:
            wrist = _to_numpy_uint8(observation["observation"]["wrist"]["rgb"])
            wrist = _resize_hwc_uint8(wrist, 224)
            images["right_wrist_camera_rgb_0"] = wrist

        # State from 8D qpos
        qpos8 = observation["embodiment"]["joint"][:8]
        if not isinstance(qpos8, np.ndarray):
            try:
                if isinstance(qpos8, torch.Tensor):
                    qpos8 = qpos8.detach().cpu().numpy()
                else:
                    qpos8 = np.asarray(qpos8)
            except Exception:
                qpos8 = np.asarray(qpos8)
        qpos8 = qpos8.astype(np.float32).reshape(-1)
        state = _build_state_from_univtac_qpos(qpos8, self.state_dim, self.mapping)

        # Tactile: stack two pads -> (1,2,224,224,3)
        # Both left AND right pads, matching parse_data_univtac.py lines 147-166.
        # Order: [left_tac (thumb), right_tac (index)] → areas [0, 1]
        # Support both key schemes: left_tactile/right_tactile (live sim) and left_gsmini/right_gsmini (replay)
        tactiles = None
        tactile_function_areas = None
        tactile_sensors = None
        left_key = "left_gsmini" if "left_gsmini" in observation.get("tactile", {}) else "left_tactile"
        right_key = "right_gsmini" if "right_gsmini" in observation.get("tactile", {}) else "right_tactile"
        if (
            self.use_tactile_input
            and "tactile" in observation
            and left_key in observation["tactile"]
            and right_key in observation["tactile"]
        ):
            left = _to_numpy_uint8(observation["tactile"][left_key]["rgb_marker"])
            left = _resize_hwc_uint8(left, 224)
            right = _to_numpy_uint8(observation["tactile"][right_key]["rgb_marker"])
            right = _resize_hwc_uint8(right, 224)
            tac = np.stack([left, right], axis=0)[None, ...]  # (1,2,H,W,3)
            tactiles = {self.tactile_key: tac.astype(np.float32)}
            tactile_function_areas = {self.tactile_key: [0, 1]}
            tactile_sensors = {self.tactile_key: self.tactile_sensor}
            
        # import pdb; pdb.set_trace()

        # if self.save_infer_input_dir and not self._saved_infer_input:
        #     _save_infer_input_sample(
        #         self.save_infer_input_dir,
        #         images=images,
        #         state=state,
        #         tactiles=tactiles,
        #         prompt=prompt,
        #     )
        #     self._saved_infer_input = True
        #     print(f"[eval_ftp1] Saved model input sample to {self.save_infer_input_dir}", flush=True)
        
        # from PIL import Image
        # images['camera_ego_rgb_0']
        # images['right_wrist_camera_rgb_0']
        # Image.fromarray(images['camera_ego_rgb_0']).save("camera_ego_rgb_0.png")
        # Image.fromarray(images['right_wrist_camera_rgb_0']).save("right_wrist_camera_rgb_0.png")
        # Image.fromarray(tactiles['right_tactile_gripper'][0,0]).save("left_tac.png")
        # Image.fromarray(tactiles['right_tactile_gripper'][0,1]).save("right_tac.png")

        chunk = self.wrapper.infer(
            images=images,
            state=state,
            prompt=prompt,
            tactiles=tactiles,
            tactile_function_areas=tactile_function_areas,
            tactile_sensors=tactile_sensors,
        )
        if chunk.ndim != 2 or chunk.shape[1] != self.action_dim:
            raise ValueError(f"Unexpected FTP1 output shape {chunk.shape}, expected (H,{self.action_dim})")
        return chunk

    def act(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        if self.use_temporal_ensemble:
            return self._act_temporal_ensemble(observation, prompt)
        raise ValueError("Please open the temporal ensemble feature by adding --temporal_ensemble to the command line.")

    def get_last_action_debug(self) -> dict[str, Any] | None:
        return _jsonify_value(self._last_action_debug)

    def _act_temporal_ensemble(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        # Re-infer every step (pure temporal ensemble, aligned with TactileACT).
        qpos8_curr = _get_qpos8(observation)
        chunk = self.infer_chunk(observation, prompt)  # (H, D) 
        self._chunk_history.append((chunk, self._exec_step, qpos8_curr.copy()))
        # Prune history: only keep chunks whose predictions still reach current step.
        # Using FTP1 offset semantics: valid index = delay + offset.
        pruned_history: list[tuple[np.ndarray, int, np.ndarray]] = []
        for c, s, qbase in self._chunk_history:
            pred_idx = self._exec_step - s + self._rtac_chunk_index_offset
            if not (1 <= pred_idx < c.shape[0]):
                continue
            if self.chunk_first_n > 0:
                max_pred_idx_exclusive = self._rtac_chunk_index_offset + self.chunk_first_n
                if pred_idx >= max_pred_idx_exclusive:
                    continue
            pruned_history.append((c, s, qbase))
        self._chunk_history = pruned_history

        # Ensemble: for exec step e, inference at step s contributes chunk_s[(e - s) + offset].
        # For FTP1, offset=1 so chunk[0] is skipped and chunk[1] is first valid future action.
        e = self._exec_step

        # Build valid actions in temporal order (old -> new), matching TactileACT policy path:
        # exp_weights = exp(-K * arange(len(valid_actions))).
        valid_abs_actions: list[np.ndarray] = []
        for cpred, s, qpos8_base in self._chunk_history:
            delay = e - s
            pred_idx = delay + self._rtac_chunk_index_offset
            if not (0 <= pred_idx < cpred.shape[0]):
                continue
            if self.chunk_first_n > 0:
                max_pred_idx_exclusive = self._rtac_chunk_index_offset + self.chunk_first_n
                if pred_idx >= max_pred_idx_exclusive:
                    continue
            # delta in FTP1 action space -> 8D delta -> absolute qpos target.
            delta8 = _extract_univtac_action_from_ftp1(
                cpred[pred_idx],
                self.action_dim,
                self.mapping,
                action_rep=self.action_rep,
            )
            abs8 = _resolve_univtac_abs_action_from_ftp1(delta8, qpos8_base, self.action_rep)
            valid_abs_actions.append(abs8)

        if valid_abs_actions:
            actions_stack = np.stack(valid_abs_actions, axis=0)  # (N, 8)
        else:
            raise ValueError(f"No valid actions to ensemble, current exec step {e}")
        exp_weights = np.exp(-self.ensemble_K * np.arange(actions_stack.shape[0], dtype=np.float32))
        if exp_weights.size > 0:
            exp_weights = exp_weights / exp_weights.sum()
        ensembled_abs = (actions_stack * exp_weights[:, None]).sum(axis=0)

        latest_delta8 = None
        latest_abs8 = None
        if chunk.shape[0] > self._rtac_chunk_index_offset:
            latest_delta8 = _extract_univtac_action_from_ftp1(
                chunk[self._rtac_chunk_index_offset],
                self.action_dim,
                self.mapping,
                action_rep=self.action_rep,
            )
            latest_abs8 = _resolve_univtac_abs_action_from_ftp1(latest_delta8, qpos8_curr, self.action_rep)
        self._last_action_debug = {
            "exec_step": int(e),
            "latest_chunk_shape": [int(chunk.shape[0]), int(chunk.shape[1])],
            "latest_chunk_first_delta8": _to_float_list(latest_delta8) if latest_delta8 is not None else None,
            "latest_chunk_first_abs8": _to_float_list(latest_abs8) if latest_abs8 is not None else None,
            "ensemble_candidates": int(actions_stack.shape[0]),
            "ensemble_weights": _to_float_list(exp_weights),
            "candidate_oldest_abs8": _to_float_list(actions_stack[0]),
            "candidate_newest_abs8": _to_float_list(actions_stack[-1]),
            "ensembled_abs8": _to_float_list(ensembled_abs),
        }

        self._exec_step += 1
        # 已在上面转换为绝对 joint 空间，这里直接返回 8D qpos 目标
        return ensembled_abs


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _parse_episode_result_from_video_name(filename: str) -> tuple[int, str] | None:
    match = _EPISODE_VIDEO_RESULT_RE.match(filename)
    if match is None:
        return None
    return int(match.group("seed")), str(match.group("result"))


def _collect_resume_state_from_videos(
    *,
    save_root: Path,
    worker_id: int,
    start_seed: int,
    total_num: int,
) -> tuple[dict[int, str], int, int]:
    """Return (completed_results, deleted_abnormal_videos, synced_metadata_entries)."""
    seed_set = set(range(start_seed, start_seed + total_num))
    worker = worker_name(worker_id)
    video_dir = Path(save_root) / "video" / worker
    metadata_path = Path(save_root) / "metadata" / f"{worker}.json"

    completed_candidates: dict[int, tuple[str, float, Path]] = {}
    deleted_abnormal_videos = 0
    if video_dir.exists():
        for video_path in sorted(video_dir.glob("*.mp4")):
            parsed = _parse_episode_result_from_video_name(video_path.name)
            if parsed is not None:
                seed, result = parsed
                if seed not in seed_set:
                    continue
                if result in _COMPLETED_VIDEO_RESULTS:
                    mtime = float(video_path.stat().st_mtime)
                    prev = completed_candidates.get(seed)
                    if prev is None or mtime >= prev[1]:
                        completed_candidates[seed] = (result, mtime, video_path)
                else:
                    # Non-success/fail files (for example *_error.mp4) are treated as abnormal and rerun.
                    video_path.unlink(missing_ok=True)
                    deleted_abnormal_videos += 1
                continue

            incomplete_match = _EPISODE_VIDEO_WITHOUT_RESULT_RE.match(video_path.name)
            if incomplete_match is None:
                continue
            seed = int(incomplete_match.group("seed"))
            if seed not in seed_set:
                continue
            video_path.unlink(missing_ok=True)
            deleted_abnormal_videos += 1

    completed_results = {seed: payload[0] for seed, payload in completed_candidates.items()}

    worker_metadata = _load_json_dict(metadata_path)
    synced_metadata_entries = 0
    for seed, result in completed_results.items():
        key = str(seed)
        old_entry = worker_metadata.get(key)
        entry = old_entry if isinstance(old_entry, dict) else {}
        changed = False
        if entry.get("result") != result:
            entry["result"] = result
            changed = True
        if entry.get("resume_from_video") is not True:
            entry["resume_from_video"] = True
            changed = True
        if changed or key not in worker_metadata:
            worker_metadata[key] = entry
            synced_metadata_entries += 1
    if synced_metadata_entries > 0:
        _write_json_atomic(metadata_path, worker_metadata)

    return completed_results, deleted_abnormal_videos, synced_metadata_entries


def _run_one_task(
    task_name: str,
    task_config: dict[str, Any],
    total_num: int,
    start_seed: int,
    instruction_type: Literal["seen", "unseen"],
    rtac: FTP1ChunkPolicy,
    save_root: Path,
    worker_id: int,
    no_video: bool = False,
    max_steps: int | None = None,
    video_frequency: int = 0,
    resume_from_videos: bool = False,
) -> dict[str, Any]:
    seeds = list(range(start_seed, start_seed + total_num))
    resumed_results: dict[int, str] = {}
    deleted_abnormal_videos = 0
    synced_metadata_entries = 0
    if resume_from_videos:
        resumed_results, deleted_abnormal_videos, synced_metadata_entries = _collect_resume_state_from_videos(
            save_root=save_root,
            worker_id=worker_id,
            start_seed=start_seed,
            total_num=total_num,
        )
    pending_seeds = [seed for seed in seeds if seed not in resumed_results]
    next_seed = pending_seeds[0] if pending_seeds else None
    print(
        "[eval_ftp1][resume] "
        f"task={task_name} worker={worker_name(worker_id)} "
        f"seed_range=[{start_seed},{start_seed + total_num - 1}] "
        f"completed={len(resumed_results)} pending={len(pending_seeds)} "
        f"next_seed={(next_seed if next_seed is not None else 'none')} "
        f"deleted_abnormal_videos={deleted_abnormal_videos} "
        f"synced_metadata_entries={synced_metadata_entries}",
        flush=True,
    )
    if not pending_seeds:
        succ = sum(1 for result in resumed_results.values() if result == "success")
        return {"task": task_name, "succ": succ, "done": len(resumed_results), "start_seed": start_seed}

    task_module = importlib.import_module(f"envs.{task_name}")
    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = save_root
    env_cfg.worker_name = worker_name(worker_id)
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    if max_steps is not None and max_steps > 0:
        env_cfg.step_lim = max_steps
    else:
        env_cfg.step_lim = task_config.get("step_lim", env_cfg.step_lim)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    if no_video:
        env_cfg.video_frequency = 0
    elif video_frequency > 0:
        env_cfg.video_frequency = video_frequency
    else:
        env_cfg.video_frequency = task_config.get("video_frequency", env_cfg.video_frequency)
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1

    task = task_module.Task(env_cfg, mode="eval")
    rtac.set_task(task_name)  # so policy uses task_settings (e.g. camera_type head vs all) for this task

    instructions = [_get_task_instruction(task_name)]
    succ = sum(1 for result in resumed_results.values() if result == "success")
    done = len(resumed_results)

    try:
        for seed in pending_seeds:
            t0 = time.perf_counter()
            rtac.reset()
            task.mode = "eval"
            task.reset(seed=seed, instructions=instructions)
            task.mean_steps = task.cfg.step_lim

            # Record initial frame (before any action) so the video shows the pre-action state.
            if task.cfg.video_frequency > 0:
                init_obs = task._get_observations()
                task.video_handler.write(task.get_frame_shot(init_obs))

            ok = False
            try:
                while task.take_action_cnt < task.cfg.step_lim:
                    # 按仿真步数上限停止（max_steps 表示 step_count，不是 action 次数）
                    if max_steps is not None and max_steps > 0 and task.step_count >= max_steps:
                        break
                    obs = task._get_observations()
                    prompt = instructions[0]

                    is_first_step = (task.take_action_cnt == 0)
                    qpos8_before = _get_qpos8(obs)
                    if is_first_step:
                        print(f"\n{'='*80}", flush=True)
                        print(f"[DEBUG] FIRST STEP  seed={seed}", flush=True)
                        print(f"[DEBUG] current qpos8 (from obs): {qpos8_before.tolist()}", flush=True)
                        print(f"[DEBUG] action_rep={rtac.action_rep}", flush=True)

                    action_index = int(task.take_action_cnt) + 1
                    step_count_before = int(task.step_count)
                    action8 = rtac.act(obs, prompt=prompt)
                    model_debug = rtac.get_last_action_debug()

                    if is_first_step:
                        last_chunk = rtac._chunk_history[-1][0] if rtac._chunk_history else None
                        if last_chunk is not None:
                            print(f"[DEBUG] raw chunk shape: {last_chunk.shape}", flush=True)
                            print(f"[DEBUG] chunk[0] (placeholder): arm={last_chunk[0, 9:16].tolist()}, gripper_slot28={last_chunk[0, 44]:.6f}", flush=True)
                            print(f"[DEBUG] chunk[1] (1st action):  arm={last_chunk[1, 9:16].tolist()}, gripper_slot28={last_chunk[1, 44]:.6f}", flush=True)
                        delta8_dbg = _extract_univtac_action_from_ftp1(
                            last_chunk[1] if last_chunk is not None else np.zeros(rtac.action_dim),
                            rtac.action_dim, rtac.mapping, action_rep=rtac.action_rep,
                        )
                        print(f"[DEBUG] extracted model-space 8D from chunk[1]: {delta8_dbg.tolist()}", flush=True)
                        print(
                            f"[DEBUG] resolved abs target from chunk[1]: "
                            f"{_resolve_univtac_abs_action_from_ftp1(delta8_dbg, qpos8_before, rtac.action_rep).tolist()}",
                            flush=True,
                        )
                        print(f"[DEBUG] ensembled action8 (sent to take_action): {action8.tolist()}", flush=True)
                        print(f"[DEBUG] action diff from current qpos: {(action8 - qpos8_before).tolist()}", flush=True)
                        print(f"{'='*80}\n", flush=True)

                    action_tensor = torch.from_numpy(action8).to(task.device).float()
                    task.take_action(action_tensor, action_type="qpos")
                    if task.eval_success:
                        ok = True
                        break
                    if task.check_early_stop():
                        break
            except Exception:
                task.clean_cache(result="error")
                raise
            else:
                cost = time.perf_counter() - t0
                task.clean_cache(result="success" if ok else "failed")
                done += 1
                if ok:
                    succ += 1
                print(
                    f"[{task_name}] seed={seed} {'success' if ok else 'failed'} cost={cost:.2f}s "
                    f"steps={task.step_count} actions={task.take_action_cnt} rate={succ}/{done} ({succ/done*100:.2f}%)"
                )
    finally:
        task.close()

    return {"task": task_name, "succ": succ, "done": done, "start_seed": start_seed}


def _write_task_run_summary(
    task_run_root: Path,
    *,
    task_name: str,
    run_id: str,
    checkpoint_dir: str | None,
    domain_name: str | None,
    instruction_type: str | None,
    task_config_path: Path | None,
    args_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    worker_payloads = load_worker_metadata(task_run_root / "metadata")
    summary = summarize_worker_metadata(worker_payloads)
    summary.update({
        "policy_name": "FTP1",
        "task_name": task_name,
        "run_id": run_id,
        "run_root": str(task_run_root),
        "checkpoint_dir": checkpoint_dir,
        "domain_name": domain_name,
        "instruction_type": instruction_type,
        "task_config_file": str(task_config_path) if task_config_path is not None else None,
        "worker_metadata_files": [f"metadata/{name}.json" for name in sorted(worker_payloads)],
        "eval_args": args_dict or {},
    })
    write_json(task_run_root / "metadata.json", summary)
    return summary


def _write_task_run_summaries(
    task_names: list[str],
    *,
    save_root: Path,
    task_run_ids: dict[str, str],
    checkpoint_dir: str | None,
    domain_name: str | None,
    instruction_type: str | None,
    task_config_path: Path | None,
    args_dict: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for task_name in task_names:
        run_id = task_run_ids[task_name]
        task_run_root = save_root / task_name / run_id
        summaries[task_name] = _write_task_run_summary(
            task_run_root,
            task_name=task_name,
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            domain_name=domain_name,
            instruction_type=instruction_type,
            task_config_path=task_config_path,
            args_dict=args_dict,
        )
    return summaries


def _backfill_task_run_summary(
    task_run_root: Path,
    *,
    checkpoint_dir: str | None,
    domain_name: str | None,
    instruction_type: str | None,
    task_config_path: Path | None,
    args_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    task_run_root = Path(task_run_root)
    return _write_task_run_summary(
        task_run_root,
        task_name=task_run_root.parent.name,
        run_id=task_run_root.name,
        checkpoint_dir=checkpoint_dir,
        domain_name=domain_name,
        instruction_type=instruction_type,
        task_config_path=task_config_path,
        args_dict=args_dict,
    )


def _split_tasks(tasks: list[str], workers: int) -> list[list[str]]:
    if workers <= 1:
        return [tasks]
    buckets = [[] for _ in range(workers)]
    for i, t in enumerate(tasks):
        buckets[i % workers].append(t)
    return buckets


def _split_devices(cuda_visible_devices: str, workers: int) -> list[list[str]]:
    devices = [d.strip() for d in cuda_visible_devices.split(",") if d.strip() != ""]
    if not devices:
        return [[""] for _ in range(workers)]
    return [[devices[i % len(devices)]] for i in range(workers)]


def _sanitize_run_id(value: str) -> str:
    return value.strip().replace("/", "_").replace("\\", "_")


def _resolve_task_run_ids(
    tasks: list[str],
    *,
    save_root: Path,
    run_suffix: str | None,
    run_id_override: str,
    resume: bool,
) -> dict[str, str]:
    if run_id_override:
        run_id = _sanitize_run_id(run_id_override)
        return {task_name: run_id for task_name in tasks}

    if not resume:
        run_id = build_run_id(commit=run_suffix)
        return {task_name: run_id for task_name in tasks}

    task_run_ids: dict[str, str] = {}
    fresh_run_id: str | None = None
    for task_name in tasks:
        task_root = save_root / task_name
        candidates: list[str] = []
        if task_root.exists():
            candidates = sorted([p.name for p in task_root.iterdir() if p.is_dir()])

        if len(candidates) == 1:
            task_run_ids[task_name] = candidates[0]
            print(
                f"[eval_ftp1][resume] task={task_name} resolved_run_id={candidates[0]}",
                flush=True,
            )
            continue

        if len(candidates) == 0:
            if fresh_run_id is None:
                fresh_run_id = build_run_id(commit=run_suffix)
            task_run_ids[task_name] = fresh_run_id
            print(
                f"[eval_ftp1][resume] task={task_name} no existing run under {task_root}; "
                f"start a new run_id={fresh_run_id}",
                flush=True,
            )
            continue

        # Keep strict behavior when multiple runs exist: caller must pick one explicitly.
        if len(candidates) > 1:
            raise ValueError(
                f"--resume found multiple existing runs under {task_root}: {candidates}. "
                "Please pass --run_id <existing_run_id> to select one run to resume."
            )
    return task_run_ids


def _worker_main(
    worker_id: int,
    task_list: list[str],
    args_dict: dict[str, Any],
    task_config_dict: dict[str, Any],
    device_list: list[str],
    total_num_for_worker: int,
):
    if device_list and device_list[0] != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_list)

    # 多 worker 时每个进程的 Isaac Sim 都会启动 HTTP transport server，默认绑定 8011，导致 "address already in use"。
    # 按 worker_id 为每个进程分配不同端口（必须在 AppLauncher 创建前注入到 sys.argv）。
    if worker_id > 0:
        _http_port = 8011 + worker_id
        sys.argv.append(f"--/exts/omni.services.transport.server.http/port={_http_port}")

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])  # defaults
    app_args.enable_cameras = True
    app_args.livestream = 2
    app_args.num_envs = 1
    # 强制 quality 渲染，减少 RTX 杂点（大显存机上 FTP1+Isaac 同 GPU 时易出现噪点）
    if hasattr(app_args, "rendering_mode"):
        app_args.rendering_mode = "quality"

    app_launcher = AppLauncher(app_args)
    simulation_app = app_launcher.app

    print(f"[eval_ftp1] Worker {worker_id}: App launched.", flush=True)
    try:
        mapping = ActionMapping(
            arm_slice=_parse_slice(args_dict["arm_slice"]) if args_dict.get("arm_slice") else None,
            gripper_index=int(args_dict["gripper_index"]) if args_dict.get("gripper_index") is not None else None,
            gripper_slot_idx=int(args_dict.get("gripper_slot_idx", 28)),
        )
        print(f"[eval_ftp1] Worker {worker_id}: Loading FTP1 from {args_dict['checkpoint_dir']} ...", flush=True)
        action_rep = str(args_dict.get("action_rep", "relative"))
        save_infer_dir = args_dict.get("save_infer_input_dir")
        rtac = FTP1ChunkPolicy(
            checkpoint_dir=args_dict["checkpoint_dir"],
            domain_name=args_dict["domain_name"],
            device=args_dict["ftp1_device"],
            num_inference_steps=int(args_dict["num_inference_steps"]),
            tactile_key=args_dict["tactile_key"],
            tactile_sensor=args_dict["tactile_sensor"],
            mapping=mapping,
            chunk_first_n=int(args_dict.get("chunk_first_n", 0)),
            action_rep=action_rep,
            save_infer_input_dir=save_infer_dir,
            use_temporal_ensemble=bool(args_dict.get("temporal_ensemble", False)),
            ensemble_K=float(args_dict.get("ensemble_K", DEFAULT_ENSEMBLE_K)),
        )
        print(
            f"[eval_ftp1] Worker {worker_id}: FTP1 loaded "
            f"(state_dim={rtac.state_dim}, action_dim={rtac.action_dim}, "
            f"model_action_dim={rtac.model_action_dim}, "
            f"action_rep={rtac.action_rep}, device={args_dict['ftp1_device']}, "
            f"temporal_ensemble={rtac.use_temporal_ensemble}).",
            flush=True,
        )

        # Each worker may be assigned a different number of episodes to run
        # (e.g., when parallelizing a single task across multiple workers).
        total = int(total_num_for_worker)
        start_seed = int(args_dict["start_seed"]) + worker_id * 100000  # avoid collisions across workers

        for task_name in task_list:
            task_run_id = str(args_dict["task_run_ids"][task_name])
            save_root = Path(args_dict["save_root"]) / task_name / task_run_id
            save_root.mkdir(parents=True, exist_ok=True)
            _run_one_task(
                task_name=task_name,
                task_config=task_config_dict,
                total_num=total,
                start_seed=start_seed,
                instruction_type=args_dict["instruction_type"],
                rtac=rtac,
                save_root=save_root,
                worker_id=worker_id,
                no_video=args_dict.get("no_video", False),
                max_steps=args_dict.get("max_steps"),
                video_frequency=int(args_dict.get("video_frequency", 0)),
                resume_from_videos=bool(args_dict.get("resume", False)),
            )
            if int(args_dict.get("workers", 1)) <= 1:
                _backfill_task_run_summary(
                    save_root,
                    checkpoint_dir=args_dict.get("checkpoint_dir"),
                    domain_name=args_dict.get("domain_name"),
                    instruction_type=args_dict.get("instruction_type"),
                    task_config_path=Path(args_dict["task_config_path"]) if args_dict.get("task_config_path") else None,
                    args_dict=args_dict,
                )
                print(
                    f"[eval_ftp1] Single-worker summary updated -> {save_root / 'metadata.json'}",
                    flush=True,
                )
    except Exception as e:
        print(f"[eval_ftp1] Worker {worker_id} ERROR: {e}\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        simulation_app.close()


def main():
    cli_argv = _rewrite_cli_device_flags(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Evaluate FTP1 on UniVTAC tasks (multi-task, optional parallel).")
    parser.add_argument("--checkpoint_dir", type=str, default="", help="FTP1 checkpoint step dir, e.g. .../7999")
    parser.add_argument("--domain_name", type=str, default="", help="Normalization domain name under checkpoint/normalization/")
    parser.add_argument("--task_list", type=str, default="", help="Comma-separated UniVTAC task names, e.g. lift_bottle,insert_hole")
    parser.add_argument("--task_config", type=str, default="contact.yml", help="Task config yaml under UniVTAC/task_config/ or an absolute path")
    parser.add_argument("--total_num", type=int, default=20, help="Number of eval episodes per task (per worker)")
    parser.add_argument("--start_seed", type=int, default=1000000, help="Base seed for evaluation")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (parallel by tasks)")
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", ""), help="CUDA_VISIBLE_DEVICES string to split")
    parser.add_argument(
        "--ftp1_device",
        type=str,
        default="cuda",
        help="Device for FTP1 inference (cuda/cpu). Use --sim_device for Isaac Sim if needed.",
    )
    parser.add_argument("--low_memory", action="store_true", help="Low-memory mode: run FTP1 model on CPU to save GPU VRAM for Isaac Sim physics/rendering.")
    parser.add_argument("--num_inference_steps", type=int, default=10, help="FTP1 diffusion denoise steps")
    parser.add_argument(
        "--chunk_first_n",
        type=int,
        default=0,
        help=(
            "Only use the first N executable steps from each predicted chunk. "
            "With FTP1 offset semantics, executable steps are chunk[1..N]. "
            "0 means use all predicted steps."
        ),
    )
    parser.add_argument("--instruction_type", type=str, default="seen", choices=["seen", "unseen"])
    parser.add_argument("--save_root", type=str, default=DEFAULT_FTP1_SAVE_ROOT, help="Root directory to save logs and summaries")
    parser.add_argument("--run_suffix", type=str, default="",
                        help="Optional label used as the run directory suffix. When set, output goes to <timestamp>_<run_suffix>; otherwise it falls back to <timestamp>_<git_commit>.")
    parser.add_argument("--run_id", type=str, default="",
                        help="Optional fixed run id under each task root (advanced override).")
    parser.add_argument("--tactile_key", type=str, default="right_tactile_gripper", help="FTP1 tactile input key")
    parser.add_argument("--tactile_sensor", type=str, default="GelSightMini", help="Tactile sensor name for FTP1 encoders")
    parser.add_argument("--no_video", action="store_true", help="Disable video recording (useful when ffmpeg is not installed)")
    parser.add_argument("--video_frequency", type=int, default=0,
                        help="Override video_frequency from task config. 1 = record every step. 0 = use task config default.")
    parser.add_argument("--gripper_slot_idx", type=int, default=28, help="Index into 32-slot hand vector for gripper (default 28)")

    # 91D/other dim override (recommended for dim91 checkpoint)
    parser.add_argument("--arm_slice", type=str, default=None, help="Override: slice for 7 arm dims in action/state, e.g. '9:16'")
    parser.add_argument("--gripper_index", type=int, default=None, help="Override: absolute index for gripper scalar in action/state")
    parser.add_argument("--action_rep", type=str, default="auto", choices=["auto", "relative", "absolute", "mix"],
                        help="Action representation: auto (infer from checkpoint train_config.json), relative (all deltas), absolute (all targets), or mix (arm delta + hand non-gripper relative + gripper(slot28) absolute)")
    parser.add_argument("--save_infer_input_dir", type=str, default=None,
                        help="If set, save one sample of the exact images/state/tactile fed to the model (224x224 BGR) to this dir for inspection.")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max simulation steps (step_count) per episode; stop when step_count >= this. Use 0 to only use task step_lim (max actions). .sh default is 160.")
    parser.add_argument("--temporal_ensemble", action="store_true",
                        help="Use temporal ensemble: re-infer every step and blend predictions with exponential decay.")
    parser.add_argument("--ensemble_K", type=float, default=DEFAULT_ENSEMBLE_K,
                        help="Temporal ensemble decay rate (default DEFAULT_ENSEMBLE_K).")
    parser.add_argument(
        "--summary_only_run_roots",
        type=str,
        default="",
        help=(
            "Comma-separated task run roots to summarize only, for example "
            "'eval_result/FTP1/lift_bottle/<run_id>,eval_result/FTP1/insert_hole/<run_id>'. "
            "When set, the script only regenerates root metadata.json and skips Isaac Sim / policy loading."
        ),
    )
    parser.add_argument(
        "--resume",
        "--resume_from_videos",
        dest="resume",
        action="store_true",
        help=(
            "Resume mode: when a task root contains exactly one existing run directory, "
            "the evaluator resumes that run by scanning video/worker_* files. "
            "If no run exists for a task, a new run_id is created for that task automatically. "
            "Completed '<seed>_(success|failed).mp4' files are skipped and synced into metadata; "
            "abnormal files (including '<seed>_error.mp4' and '<seed>.mp4') are deleted then rerun."
        ),
    )

    args = parser.parse_args(cli_argv)
    if args.chunk_first_n < 0:
        raise ValueError("--chunk_first_n must be >= 0")

    task_cfg_path = _resolve_task_config_path(args.task_config, must_exist=not bool(args.summary_only_run_roots))

    run_suffix = args.run_suffix.strip() or None

    summary_only_roots = [Path(p) for p in _parse_csv(args.summary_only_run_roots)]
    if summary_only_roots:
        args_dict = vars(args).copy()
        args_dict["task_config_path"] = str(task_cfg_path) if task_cfg_path is not None else None
        for task_run_root in summary_only_roots:
            summary = _backfill_task_run_summary(
                task_run_root,
                checkpoint_dir=args.checkpoint_dir or None,
                domain_name=args.domain_name or None,
                instruction_type=args.instruction_type or None,
                task_config_path=task_cfg_path,
                args_dict=args_dict,
            )
            print(
                f"[eval_ftp1] Summary-only updated {task_run_root / 'metadata.json'}: "
                f"{summary['success']}/{summary['total_episodes']} ({summary['success_rate'] * 100:.2f}%)",
                flush=True,
            )
        return

    if not args.checkpoint_dir:
        raise ValueError("--checkpoint_dir is required unless --summary_only_run_roots is set")
    if not args.domain_name:
        raise ValueError("--domain_name is required unless --summary_only_run_roots is set")

    if args.low_memory:
        args.ftp1_device = "cpu"
        print("[eval_ftp1] Low-memory mode: FTP1 inference on CPU.", flush=True)

    if args.action_rep == "auto":
        inferred_action_rep = _infer_action_rep_from_checkpoint(args.checkpoint_dir)
        if inferred_action_rep is None:
            args.action_rep = "relative"
            print(
                "[eval_ftp1] WARNING: --action_rep=auto but train_config.json missing/invalid. "
                "Fallback to 'relative'. Please pass --action_rep explicitly if your model was trained with absolute or mix actions.",
                flush=True,
            )
        else:
            args.action_rep = inferred_action_rep
            print(
                f"[eval_ftp1] Auto action_rep resolved to '{args.action_rep}' from "
                f"{Path(args.checkpoint_dir) / 'train_config.json'}",
                flush=True,
            )
    else:
        print(f"[eval_ftp1] Using user-specified action_rep='{args.action_rep}'", flush=True)

    tasks = _parse_csv(args.task_list)
    if not tasks:
        raise ValueError("--task_list is required unless --summary_only_run_roots is set")
    task_cfg = _read_yaml_or_json(task_cfg_path)

    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)
    task_run_ids = _resolve_task_run_ids(
        tasks,
        save_root=save_root,
        run_suffix=run_suffix,
        run_id_override=args.run_id,
        resume=bool(args.resume),
    )

    device_assign = _split_devices(args.gpu, args.workers)

    # For multi-task evaluation, keep original behavior: split tasks across workers,
    # each worker runs args.total_num episodes for its assigned tasks.
    # For single-task evaluation, parallelize that task across workers by splitting
    # the total_num episodes across workers.
    if args.workers <= 1:
        buckets = [tasks]
        per_worker_totals = [args.total_num]
    else:
        if len(tasks) == 1:
            # Single task: all workers get the same task_list, but fewer episodes each.
            buckets = [tasks for _ in range(args.workers)]
            base = args.total_num // args.workers
            extra = args.total_num % args.workers
            per_worker_totals = [
                base + (1 if wid < extra else 0) for wid in range(args.workers)
            ]
        else:
            buckets = _split_tasks(tasks, args.workers)
            per_worker_totals = [args.total_num for _ in range(args.workers)]

    args_dict = vars(args).copy()
    args_dict["save_root"] = str(save_root)
    args_dict["task_run_ids"] = task_run_ids
    args_dict["task_config_path"] = str(task_cfg_path)

    if args.workers <= 1:
        _worker_main(0, buckets[0], args_dict, task_cfg, device_assign[0], per_worker_totals[0])
    else:
        mp_ctx = get_context("spawn")
        procs = []
        for wid in range(args.workers):
            p = mp_ctx.Process(
                target=_worker_main,
                args=(wid, buckets[wid], args_dict, task_cfg, device_assign[wid], per_worker_totals[wid]),
                name=f"FTP1Worker-{wid}",
            )
            p.start()
            procs.append(p)

        failed = False
        for p in procs:
            p.join()
            if p.exitcode != 0:
                failed = True
                print(f"[ERROR] {p.name} exited with code {p.exitcode}")
        if failed:
            sys.exit(1)

    summaries = _write_task_run_summaries(
        tasks,
        save_root=save_root,
        task_run_ids=task_run_ids,
        checkpoint_dir=args.checkpoint_dir,
        domain_name=args.domain_name,
        instruction_type=args.instruction_type,
        task_config_path=task_cfg_path,
        args_dict=args_dict,
    )
    for task_name in tasks:
        summary = summaries[task_name]
        task_run_root = save_root / task_name / task_run_ids[task_name]
        print(
            f"[eval_ftp1] Summary for {task_name}: {summary['success']}/{summary['total_episodes']} "
            f"({summary['success_rate'] * 100:.2f}%) -> {task_run_root / 'metadata.json'}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}\n{traceback.format_exc()}")
        raise
