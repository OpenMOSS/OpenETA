"""
Run one UniVTAC task until N successful FTP1 episodes are saved.

This script is intentionally independent from the existing eval_ftp1.py / .sh
pipeline so the original evaluation path remains unchanged and reproducible.
"""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import argparse
import cv2
import importlib
import json
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from eval_result_utils import build_run_id, load_worker_metadata, summarize_worker_metadata, worker_name, write_json
from openpi.models_pytorch.ftp1_model_config import FTP1_RESERVED_ACTION_DIM
from openpi.models_pytorch.ftp1_model_config import FTP1_SINGLE_ARM_ACTION_REP_DIM
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

sys.path.insert(0, ".")
sys.path.insert(0, "./policy")

DEFAULT_ENSEMBLE_K = 0.01
DEFAULT_SAVE_ROOT = "eval_results/FTP1_success_runs"
DEFAULT_VIDEO_FPS = 10
SUCCESS_DIR_RE = re.compile(r"^success_(?P<idx>\d+)_seed_(?P<seed>\d+)$")


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
            "[eval_ftp1_until_success] WARNING: legacy --device was interpreted as FTP1 policy device. "
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


def _read_yaml_or_json(path: Path) -> dict[str, Any]:
    if path.suffix in {".yml", ".yaml"}:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config type: {path}")


def _resolve_task_config_path(task_config: str) -> Path:
    task_cfg_path = Path(task_config)
    if task_cfg_path.exists():
        return task_cfg_path

    candidate = Path(__file__).parent.parent / "task_config" / task_config
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Task config not found: {task_config}")


def _infer_action_rep_from_checkpoint(checkpoint_dir: str | Path) -> str | None:
    train_config_path = Path(checkpoint_dir) / "train_config.json"
    if not train_config_path.exists():
        return None
    try:
        with open(train_config_path, "r", encoding="utf-8") as f:
            train_cfg = json.load(f)
    except Exception:
        return None
    return _canonicalize_action_joint_rep(train_cfg.get("action_joint_rep"))


def _to_numpy_uint8(img: Any) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    if isinstance(img, np.ndarray):
        arr = img
    elif isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {arr.shape}")
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
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _get_task_instruction(task_name: str) -> str:
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
    arm_slice: tuple[int, int] | None = None
    gripper_index: int | None = None
    gripper_slot_idx: int = 28


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
        state[0, 9:16] = qpos8[:7]
        state[0, 9 + mapping.gripper_slot_idx] = qpos8[7]
        return state

    if action_dim > 9 + mapping.gripper_slot_idx:
        state[0, 9:16] = qpos8[:7]
        state[0, 9 + mapping.gripper_slot_idx] = qpos8[7]
    return state


def _extract_univtac_action_from_ftp1(
    action_vec: np.ndarray, action_dim: int, mapping: ActionMapping
) -> np.ndarray:
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
        out = np.zeros(8, dtype=np.float32)
        out[:7] = action_vec[9:16]
        out[7] = float(action_vec[16 + mapping.gripper_slot_idx])
        return out

    if layout == "legacy91":
        out = np.zeros(8, dtype=np.float32)
        out[:7] = action_vec[9:16]
        out[7] = float(action_vec[9 + mapping.gripper_slot_idx])
        return out

    raise ValueError(
        f"Unsupported action_dim={action_dim}. Provide --arm_slice and --gripper_index to map custom layouts."
    )


def _get_qpos8(observation: dict[str, Any]) -> np.ndarray:
    qpos8 = observation["embodiment"]["joint"][:8]
    if isinstance(qpos8, torch.Tensor):
        qpos8 = qpos8.detach().cpu().numpy()
    return np.asarray(qpos8, dtype=np.float32).reshape(8)


def _load_task_settings() -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "policy" / "task_settings.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
        ensemble_K: float = DEFAULT_ENSEMBLE_K,
    ):
        normalized_action_rep = _canonicalize_action_joint_rep(action_rep)
        if normalized_action_rep is None:
            raise ValueError(f"Unsupported action_rep: {action_rep!r}")
        self.action_rep = normalized_action_rep
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
        self.mapping = mapping
        self.tactile_key = tactile_key
        self.tactile_sensor = tactile_sensor
        self.chunk_first_n = max(0, int(chunk_first_n))
        self.ensemble_K = ensemble_K
        self._rtac_chunk_index_offset = 1
        self._chunk_history: list[tuple[np.ndarray, int, np.ndarray]] = []
        self._exec_step: int = 0

    def reset(self) -> None:
        self._chunk_history = []
        self._exec_step = 0

    def set_task(self, task_name: str) -> None:
        self._current_task_name = task_name

    def _use_wrist_camera(self) -> bool:
        if self._current_task_name is None:
            return True
        cfg = self._task_settings.get(self._current_task_name, {})
        return cfg.get("camera_type", "all") == "all"

    def infer_chunk(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        images: dict[str, np.ndarray] = {}
        if "observation" in observation and "head" in observation["observation"]:
            head = _to_numpy_uint8(observation["observation"]["head"]["rgb"])
            images["camera_ego_rgb_0"] = _resize_hwc_uint8(head, 224)
        if self._use_wrist_camera() and "observation" in observation and "wrist" in observation["observation"]:
            wrist = _to_numpy_uint8(observation["observation"]["wrist"]["rgb"])
            images["right_wrist_camera_rgb_0"] = _resize_hwc_uint8(wrist, 224)

        qpos8 = _get_qpos8(observation)
        state = _build_state_from_univtac_qpos(qpos8, self.state_dim, self.mapping)

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
            right = _to_numpy_uint8(observation["tactile"][right_key]["rgb_marker"])
            left = _resize_hwc_uint8(left, 224)
            right = _resize_hwc_uint8(right, 224)
            tac = np.stack([left, right], axis=0)[None, ...]
            tactiles = {self.tactile_key: tac.astype(np.float32)}
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
        if chunk.ndim != 2 or chunk.shape[1] != self.action_dim:
            raise ValueError(f"Unexpected FTP1 output shape {chunk.shape}, expected (H, {self.action_dim})")
        return chunk

    def act(self, observation: dict[str, Any], prompt: str) -> np.ndarray:
        qpos8_curr = _get_qpos8(observation)
        chunk = self.infer_chunk(observation, prompt)
        self._chunk_history.append((chunk, self._exec_step, qpos8_curr.copy()))

        pruned_history: list[tuple[np.ndarray, int, np.ndarray]] = []
        for candidate_chunk, infer_step, qbase in self._chunk_history:
            pred_idx = self._exec_step - infer_step + self._rtac_chunk_index_offset
            if not (1 <= pred_idx < candidate_chunk.shape[0]):
                continue
            if self.chunk_first_n > 0:
                max_pred_idx_exclusive = self._rtac_chunk_index_offset + self.chunk_first_n
                if pred_idx >= max_pred_idx_exclusive:
                    continue
            pruned_history.append((candidate_chunk, infer_step, qbase))
        self._chunk_history = pruned_history

        valid_abs_actions: list[np.ndarray] = []
        for candidate_chunk, infer_step, qpos8_base in self._chunk_history:
            delay = self._exec_step - infer_step
            pred_idx = delay + self._rtac_chunk_index_offset
            if not (0 <= pred_idx < candidate_chunk.shape[0]):
                continue
            if self.chunk_first_n > 0:
                max_pred_idx_exclusive = self._rtac_chunk_index_offset + self.chunk_first_n
                if pred_idx >= max_pred_idx_exclusive:
                    continue
            delta8 = _extract_univtac_action_from_ftp1(candidate_chunk[pred_idx], self.action_dim, self.mapping)
            abs8 = _resolve_univtac_abs_action_from_ftp1(delta8, qpos8_base, self.action_rep)
            valid_abs_actions.append(abs8)

        if not valid_abs_actions:
            raise ValueError(f"No valid actions to ensemble at exec_step={self._exec_step}")

        actions_stack = np.stack(valid_abs_actions, axis=0)
        exp_weights = np.exp(-self.ensemble_K * np.arange(actions_stack.shape[0], dtype=np.float32))
        exp_weights = exp_weights / exp_weights.sum()
        ensembled_abs = (actions_stack * exp_weights[:, None]).sum(axis=0)

        self._exec_step += 1
        return ensembled_abs


def _frame_from_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    if "observation" in observation and "head" in observation["observation"]:
        frames["main"] = _to_numpy_uint8(observation["observation"]["head"]["rgb"])
    if "observation" in observation and "wrist" in observation["observation"]:
        frames["wrist"] = _to_numpy_uint8(observation["observation"]["wrist"]["rgb"])
    left_key = "left_gsmini" if "left_gsmini" in observation.get("tactile", {}) else "left_tactile"
    right_key = "right_gsmini" if "right_gsmini" in observation.get("tactile", {}) else "right_tactile"
    if "tactile" in observation and left_key in observation["tactile"]:
        frames["left_tactile"] = _to_numpy_uint8(observation["tactile"][left_key]["rgb_marker"])
    if "tactile" in observation and right_key in observation["tactile"]:
        frames["right_tactile"] = _to_numpy_uint8(observation["tactile"][right_key]["rgb_marker"])
    return frames


def _append_episode_frames(frame_buffer: dict[str, list[np.ndarray]], observation: dict[str, Any]) -> None:
    current_frames = _frame_from_observation(observation)
    for key, frame in current_frames.items():
        frame_buffer.setdefault(key, []).append(frame)


def _write_video_ffmpeg(save_path: Path, frames: list[np.ndarray], fps: int) -> None:
    if not frames:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "18",
            "-movflags",
            "+faststart",
            str(save_path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            process.stdin.write(frame.tobytes())
    finally:
        assert process.stdin is not None
        process.stdin.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg failed when writing {save_path} (exit={return_code})")


def _scan_existing_success_records(success_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not success_root.exists():
        return records
    for path in sorted(p for p in success_root.iterdir() if p.is_dir()):
        match = SUCCESS_DIR_RE.match(path.name)
        if match is None:
            continue
        episode_json = path / "episode.json"
        payload: dict[str, Any] = {}
        if episode_json.exists():
            try:
                with open(episode_json, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
        records.append(
            {
                "success_index": int(match.group("idx")),
                "seed": int(match.group("seed")),
                "dir": str(path),
                "episode": payload,
            }
        )
    records.sort(key=lambda item: item["success_index"])
    return records


def _write_success_videos(
    success_dir: Path,
    frame_buffer: dict[str, list[np.ndarray]],
    *,
    fps: int,
    task_name: str,
    seed: int,
    success_index: int,
    step_count: int,
    action_count: int,
) -> dict[str, Any]:
    success_dir.mkdir(parents=True, exist_ok=True)
    video_files: dict[str, str] = {}
    for key in ["main", "wrist", "left_tactile", "right_tactile"]:
        frames = frame_buffer.get(key, [])
        if not frames:
            continue
        path = success_dir / f"{key}.mp4"
        _write_video_ffmpeg(path, frames, fps=fps)
        video_files[key] = str(path)

    payload = {
        "task_name": task_name,
        "seed": seed,
        "success_index": success_index,
        "step_count": step_count,
        "action_count": action_count,
        "fps": fps,
        "video_files": video_files,
        "frame_counts": {key: len(value) for key, value in frame_buffer.items()},
    }
    write_json(success_dir / "episode.json", payload)
    return payload


def _next_seed_from_metadata(metadata_dir: Path, fallback: int) -> int:
    worker_payload = load_worker_metadata(metadata_dir).get(worker_name(0), {})
    if not worker_payload:
        return fallback
    try:
        max_seed = max(int(seed) for seed in worker_payload.keys())
    except ValueError:
        return fallback
    return max_seed + 1


def _write_run_summary(
    run_root: Path,
    *,
    task_name: str,
    run_id: str,
    checkpoint_dir: str,
    domain_name: str,
    task_config_path: Path,
    args_dict: dict[str, Any],
    target_successes: int,
) -> dict[str, Any]:
    worker_payloads = load_worker_metadata(run_root / "metadata")
    summary = summarize_worker_metadata(worker_payloads)
    success_records = _scan_existing_success_records(run_root / "success_videos" / worker_name(0))
    summary.update(
        {
            "policy_name": "FTP1",
            "task_name": task_name,
            "run_id": run_id,
            "run_root": str(run_root),
            "checkpoint_dir": checkpoint_dir,
            "domain_name": domain_name,
            "task_config_file": str(task_config_path),
            "target_successes": target_successes,
            "saved_successes": len(success_records),
            "worker_metadata_files": [f"metadata/{name}.json" for name in sorted(worker_payloads)],
            "success_records": success_records,
            "eval_args": args_dict,
        }
    )
    write_json(run_root / "metadata.json", summary)
    write_json(run_root / "success_manifest.json", success_records)
    return summary


def _run_until_successes(args: argparse.Namespace, simulation_app: Any) -> None:
    del simulation_app
    task_config_path = _resolve_task_config_path(args.task_config)
    task_config = _read_yaml_or_json(task_config_path)

    action_rep = args.action_rep
    if action_rep == "auto":
        inferred = _infer_action_rep_from_checkpoint(args.checkpoint_dir)
        if inferred is None:
            action_rep = "relative"
            print(
                "[eval_ftp1_until_success] WARNING: auto action_rep inference failed; fallback to 'relative'.",
                flush=True,
            )
        else:
            action_rep = inferred
            print(f"[eval_ftp1_until_success] action_rep auto -> {action_rep}", flush=True)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.low_memory:
        args.ftp1_device = "cpu"

    if args.run_id:
        run_id = args.run_id.strip().replace("/", "_").replace("\\", "_")
    else:
        run_id = build_run_id(commit=args.run_suffix.strip() or None)

    run_root = Path(args.save_root) / args.task_name / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    success_root = run_root / "success_videos" / worker_name(0)
    success_root.mkdir(parents=True, exist_ok=True)

    existing_successes = _scan_existing_success_records(success_root)
    success_count = len(existing_successes)
    success_index = (existing_successes[-1]["success_index"] + 1) if existing_successes else 1
    seed = _next_seed_from_metadata(run_root / "metadata", args.start_seed)

    task_module = importlib.import_module(f"envs.{args.task_name}")
    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = run_root
    env_cfg.worker_name = worker_name(0)
    env_cfg.decimation = task_config.get("decimation", env_cfg.decimation)
    env_cfg.step_lim = args.max_steps if args.max_steps > 0 else task_config.get("step_lim", env_cfg.step_lim)
    env_cfg.obs_data_type = task_config.get("observations", {})
    env_cfg.save_frequency = task_config.get("save_frequency", env_cfg.save_frequency)
    env_cfg.video_frequency = 0
    env_cfg.random_texture = task_config.get("random_texture", False)
    env_cfg.scene.num_envs = 1

    mapping = ActionMapping(
        arm_slice=_parse_slice(args.arm_slice) if args.arm_slice else None,
        gripper_index=args.gripper_index,
        gripper_slot_idx=args.gripper_slot_idx,
    )
    policy = FTP1ChunkPolicy(
        checkpoint_dir=args.checkpoint_dir,
        domain_name=args.domain_name,
        device=args.ftp1_device,
        num_inference_steps=args.num_inference_steps,
        tactile_key=args.tactile_key,
        tactile_sensor=args.tactile_sensor,
        mapping=mapping,
        chunk_first_n=args.chunk_first_n,
        action_rep=action_rep,
        ensemble_K=args.ensemble_K,
    )
    policy.set_task(args.task_name)

    task = task_module.Task(env_cfg, mode="eval")
    instruction = _get_task_instruction(args.task_name)
    episodes_run = 0
    sample_every = args.video_frequency if args.video_frequency > 0 else int(task_config.get("video_frequency", 1))
    if sample_every <= 0:
        sample_every = 1

    try:
        while success_count < args.target_successes:
            if args.max_episodes > 0 and episodes_run >= args.max_episodes:
                raise RuntimeError(
                    f"Reached max_episodes={args.max_episodes} before target_successes={args.target_successes}."
                )

            policy.reset()
            task.mode = "eval"
            task.reset(seed=seed, instructions=[instruction])
            task.mean_steps = task.cfg.step_lim
            obs = task._get_observations()
            frame_buffer: dict[str, list[np.ndarray]] = {}
            _append_episode_frames(frame_buffer, obs)
            captured_steps = {int(task.step_count)}
            ok = False
            episode_start = time.perf_counter()

            try:
                while task.take_action_cnt < task.cfg.step_lim:
                    if args.max_steps > 0 and task.step_count >= args.max_steps:
                        break
                    action8 = policy.act(obs, prompt=instruction)
                    action_tensor = torch.from_numpy(action8).to(task.device).float()
                    task.take_action(action_tensor, action_type="qpos")
                    obs = task._get_observations()
                    should_capture = (
                        task.step_count % sample_every == 0
                        or task.eval_success
                        or task.check_early_stop()
                    )
                    if should_capture and int(task.step_count) not in captured_steps:
                        _append_episode_frames(frame_buffer, obs)
                        captured_steps.add(int(task.step_count))
                    if task.eval_success:
                        ok = True
                        break
                    if task.check_early_stop():
                        break
            except Exception:
                task.clean_cache(result="error")
                raise
            else:
                task.clean_cache(result="success" if ok else "failed")
                episode_cost = time.perf_counter() - episode_start
                episodes_run += 1

                if ok:
                    success_dir = success_root / f"success_{success_index:03d}_seed_{seed}"
                    episode_payload = _write_success_videos(
                        success_dir,
                        frame_buffer,
                        fps=args.video_fps,
                        task_name=args.task_name,
                        seed=seed,
                        success_index=success_index,
                        step_count=int(task.step_count),
                        action_count=int(task.take_action_cnt),
                    )
                    success_count += 1
                    success_index += 1
                    print(
                        f"[eval_ftp1_until_success] task={args.task_name} seed={seed} success "
                        f"{success_count}/{args.target_successes} cost={episode_cost:.2f}s "
                        f"saved={episode_payload['video_files']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[eval_ftp1_until_success] task={args.task_name} seed={seed} failed "
                        f"cost={episode_cost:.2f}s success={success_count}/{args.target_successes}",
                        flush=True,
                    )

                summary = _write_run_summary(
                    run_root,
                    task_name=args.task_name,
                    run_id=run_id,
                    checkpoint_dir=args.checkpoint_dir,
                    domain_name=args.domain_name,
                    task_config_path=task_config_path,
                    args_dict=vars(args),
                    target_successes=args.target_successes,
                )
                print(
                    f"[eval_ftp1_until_success] summary {summary['saved_successes']}/{summary['target_successes']} "
                    f"saved successes, total_episodes={summary['total_episodes']}",
                    flush=True,
                )
                seed += 1
    finally:
        task.close()


def main() -> None:
    from isaaclab.app import AppLauncher

    cli_argv = _rewrite_cli_device_flags(sys.argv[1:])
    parser = argparse.ArgumentParser(description="Run one UniVTAC FTP1 task until N successes are saved.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--domain_name", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_config", type=str, default="contact.yml")
    parser.add_argument("--target_successes", type=int, default=5)
    parser.add_argument("--max_episodes", type=int, default=0)
    parser.add_argument("--start_seed", type=int, default=1000000)
    parser.add_argument("--gpu", type=str, default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument(
        "--ftp1_device",
        type=str,
        default="cuda",
        help="Device for FTP1 inference (cuda/cpu). Use --sim_device for Isaac Sim if needed.",
    )
    parser.add_argument("--low_memory", action="store_true")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--chunk_first_n", type=int, default=20)
    parser.add_argument("--action_rep", type=str, default="auto", choices=["auto", "relative", "absolute", "mix"])
    parser.add_argument("--ensemble_K", type=float, default=DEFAULT_ENSEMBLE_K)
    parser.add_argument("--tactile_key", type=str, default="right_tactile_gripper")
    parser.add_argument("--tactile_sensor", type=str, default="GelSightMini")
    parser.add_argument("--gripper_slot_idx", type=int, default=28)
    parser.add_argument("--arm_slice", type=str, default=None)
    parser.add_argument("--gripper_index", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--video_frequency", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--save_root", type=str, default=DEFAULT_SAVE_ROOT)
    parser.add_argument("--run_suffix", type=str, default="")
    parser.add_argument("--run_id", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)

    args = parser.parse_args(cli_argv)
    args.enable_cameras = True
    args.livestream = 2
    args.num_envs = 1
    if hasattr(args, "rendering_mode"):
        args.rendering_mode = "quality"

    if args.target_successes <= 0:
        raise ValueError("--target_successes must be > 0")
    if args.chunk_first_n < 0:
        raise ValueError("--chunk_first_n must be >= 0")

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print(
        "[eval_ftp1_until_success] "
        f"task={args.task_name} checkpoint={args.checkpoint_dir} domain={args.domain_name} "
        f"target_successes={args.target_successes} run_id={args.run_id or '(auto)'} save_root={args.save_root}",
        flush=True,
    )
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        _run_until_successes(args, simulation_app)
    except Exception as exc:
        print(f"[FATAL] {exc}\n{traceback.format_exc()}", flush=True)
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
