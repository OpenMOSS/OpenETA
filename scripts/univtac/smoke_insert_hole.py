#!/usr/bin/env python3
"""Run a direct UniVTAC Insert Hole observation/action contract smoke test."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.contract import UniVTACContractError
from sim.envs.univtac.observation import capture_snapshot
from sim.envs.univtac.runtime import (
    assert_task_not_imported,
    import_task_after_launcher,
)
from sim.envs.univtac.trace import (
    build_transition,
    verify_artifacts,
    write_exception,
    write_json,
    write_snapshot,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "univtac" / "insert_hole_smoke.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "univtac-smoke" / "insert_hole"
VENDOR_TASK_MODULES = ("envs._base_task", "envs.insert_hole")
TACEX_PACKAGE_NAMES = ("tacex_assets", "tacex_uipc", "tacex_tasks", "tacex")
TACEX_HIGH_RES_ROBOT_ASSET = Path(
    "Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist.usd"
)
TACEX_ASSET_SENTINELS = (
    TACEX_HIGH_RES_ROBOT_ASSET,
    Path("Sensors/GelSight_Mini/Gelpad_high_res.usd"),
    Path("Sensors/GelSight_Mini/calibs/640x480/dataPack.npz"),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UniVTACContractError(f"smoke config must be a mapping: {path}")
    required = {
        "task",
        "seeds",
        "probe_delta_mm",
        "probe_time_dilation_factor",
        "save_host_only",
        "strict_two_tactile_sensors",
        "fail_on_missing_rgb_marker",
        "vendor_task_config",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise UniVTACContractError(f"smoke config is missing keys: {missing}")
    if payload["task"] != "insert_hole":
        raise UniVTACContractError("this direct smoke script only supports task=insert_hole")
    seeds = payload["seeds"]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise UniVTACContractError("seeds must be a non-empty list of integers")
    displacement = np.asarray(payload["probe_delta_mm"], dtype=np.float64)
    if displacement.shape != (3,):
        raise UniVTACContractError("probe_delta_mm must contain exactly three values")
    if displacement[2] >= 0:
        raise UniVTACContractError("probe_delta_mm must request a downward z displacement")
    if float(np.linalg.norm(displacement)) > 3.0:
        raise UniVTACContractError("probe displacement magnitude must not exceed 3 mm")
    time_dilation_factor = payload["probe_time_dilation_factor"]
    if not isinstance(time_dilation_factor, (int, float)) or not (
        0.0 < float(time_dilation_factor) <= 1.0
    ):
        raise UniVTACContractError(
            "probe_time_dilation_factor must be in the interval (0, 1]"
        )
    if payload["save_host_only"] is not True:
        raise UniVTACContractError("save_host_only must be true for this contract smoke")
    return payload


def _load_vendor_task_config(relative_path: str, task_root: Path) -> dict[str, Any]:
    path = (task_root / relative_path).resolve()
    if not path.is_relative_to(task_root.resolve()):
        raise UniVTACContractError(f"task config escapes task source root: {path}")
    if not path.is_file():
        raise UniVTACContractError(f"vendored task config does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UniVTACContractError(f"vendored task config must be a mapping: {path}")
    return payload


def _assert_vendor_task_not_imported() -> None:
    imported = [name for name in VENDOR_TASK_MODULES if name in sys.modules]
    if imported:
        raise RuntimeError(
            "UniVTAC task import occurred before AppLauncher initialization: "
            f"{imported}"
        )


def _resolve_tacex_asset_root(vendor_root: Path) -> Path:
    candidates: list[Path] = []
    configured = os.environ.get("TACEX_ASSETS_DATA_DIR")
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(
        vendor_root
        / "third_party"
        / "TacEx"
        / "source"
        / "tacex_assets"
        / "tacex_assets"
        / "data"
    )
    for parent in (REPO_ROOT, *REPO_ROOT.parents):
        mirror_root = parent / "assets_mirror" / "tacex-data"
        if mirror_root.is_dir():
            candidates.extend(sorted(path for path in mirror_root.iterdir() if path.is_dir()))

    for candidate in candidates:
        resolved = candidate.resolve()
        high_res_robot = resolved / TACEX_HIGH_RES_ROBOT_ASSET
        if (
            all((resolved / relative).is_file() for relative in TACEX_ASSET_SENTINELS)
            and not high_res_robot.is_symlink()
        ):
            return resolved
    checked = [str(candidate) for candidate in candidates]
    raise FileNotFoundError(
        "no complete TacEx data directory was found; checked " f"{checked}"
    )


def _prepare_vendored_tacex(vendor_root: Path, runtime_root: Path) -> Path:
    """Put checkout-owned TacEx code over a complete read-only data mirror."""

    source_root = vendor_root / "third_party" / "TacEx" / "source"
    extension_source = source_root / "tacex_assets"
    if not (extension_source / "config" / "extension.toml").is_file():
        raise FileNotFoundError(f"vendored TacEx extension is incomplete: {extension_source}")
    asset_root = _resolve_tacex_asset_root(vendor_root)

    overlay_root = runtime_root / "tacex_assets_extension"
    shutil.copytree(extension_source, overlay_root)
    overlay_data = overlay_root / "tacex_assets" / "data"
    if overlay_data.exists():
        raise RuntimeError(f"unexpected TacEx data already exists in overlay: {overlay_data}")
    overlay_data.symlink_to(asset_root, target_is_directory=True)

    python_roots = (
        vendor_root,
        source_root,
        source_root / "tacex",
        source_root / "tacex_uipc",
        source_root / "tacex_tasks",
        overlay_root,
    )
    for python_root in python_roots:
        if not python_root.is_dir():
            raise FileNotFoundError(f"vendored Python root does not exist: {python_root}")
    for python_root in python_roots:
        path = str(python_root)
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)

    for prefix in TACEX_PACKAGE_NAMES:
        for module_name in tuple(sys.modules):
            if module_name == prefix or module_name.startswith(f"{prefix}."):
                sys.modules.pop(module_name, None)
    for module_name in tuple(sys.modules):
        if module_name == "envs" or module_name.startswith("envs."):
            sys.modules.pop(module_name, None)
    envs_package = types.ModuleType("envs")
    envs_package.__path__ = [str(vendor_root / "envs")]
    envs_package.__package__ = "envs"
    sys.modules["envs"] = envs_package
    return asset_root


def _import_task_after_launcher(
    simulation_app: Any,
    vendor_root: Path,
    task_name: str,
    runtime_root: Path,
):
    if simulation_app is None:
        raise RuntimeError("AppLauncher must start before importing the UniVTAC task")
    asset_root = _prepare_vendored_tacex(vendor_root, runtime_root)
    task_module = importlib.import_module(f"envs.{task_name}")
    return task_module, asset_root


def _seed_result(seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "reset_ok": False,
        "observation_ok": False,
        "tactile_sensor_names": [],
        "tactile_shapes": {},
        "probe_ok": False,
        "step_count_changed": False,
        "action_count_changed": False,
        "tactile_changed": False,
        "tactile_changed_by_sensor": {},
        "native_check_success": None,
        "cleanup_ok": False,
        "error": None,
    }


def _write_summary(output_root: Path, summary: dict[str, Any]) -> None:
    write_json(output_root / "summary.json", summary)


def _configure_task(
    task_module: Any,
    config: dict[str, Any],
    native_root: Path,
    device: str | None,
    task_root: Path,
):
    vendor_config = _load_vendor_task_config(config["vendor_task_config"], task_root)
    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = str(native_root)
    env_cfg.worker_name = "direct_smoke"
    env_cfg.scene.num_envs = 1
    env_cfg.decimation = int(vendor_config.get("decimation", env_cfg.decimation))
    env_cfg.obs_data_type = dict(vendor_config.get("observations", {}))
    env_cfg.save_frequency = 0
    env_cfg.video_frequency = 0
    env_cfg.render_frequency = 0
    if config.get("runtime_render_frequency_override") is not None:
        env_cfg.render_frequency = int(config["runtime_render_frequency_override"])
    env_cfg.random_texture = False
    env_cfg.tactile_sensor_type = str(vendor_config.get("sensor_type", "gsmini"))
    if device:
        env_cfg.sim.device = device
    return env_cfg


def run_smoke(args: argparse.Namespace) -> int:
    config_path = Path(args.config).resolve()
    output_root = Path(args.output_root).resolve()
    task_root = Path(args.source_root).expanduser().resolve()
    asset_root_requested = (
        Path(args.asset_root).expanduser().resolve() if args.asset_root else None
    )
    config = _load_config(config_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "schema_version": "openeta.univtac.smoke_summary.v1",
        "task": config["task"],
        "status": "running",
        "seeds": {},
        "error": None,
    }
    run_manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.smoke_manifest.v1",
        "task": config["task"],
        "config_path": config_path.relative_to(REPO_ROOT).as_posix(),
        "output_root": output_root.relative_to(REPO_ROOT).as_posix(),
        "config": config,
        "task_source_root": str(task_root),
        "started_at": _utc_now(),
        "ended_at": None,
        "status": "starting",
        "app_launcher_initialized_before_task_import": False,
        "vendor_task_imported_after_app_launcher": False,
        "tacex_asset_root": None,
        "task_close_ok": None,
        "simulation_app_close_attempted": False,
        "observation_key_tree": None,
        "error": None,
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    _write_summary(output_root, summary)

    simulation_app = None
    task = None
    original_cwd = Path.cwd()
    native_runtime = tempfile.TemporaryDirectory(prefix="openeta-univtac-smoke-")
    overall_error = False
    cleanup_abort = False
    try:
        assert_task_not_imported(str(config["task"]))
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        if simulation_app is None:
            raise RuntimeError("AppLauncher returned no simulation_app")
        run_manifest["app_launcher_initialized_before_task_import"] = True
        run_manifest["status"] = "importing_vendor_task"
        write_json(output_root / "run_manifest.json", run_manifest)

        os.chdir(task_root)
        task_module, runtime_context = import_task_after_launcher(
            simulation_app=simulation_app,
            task_root=task_root,
            task_name=str(config["task"]),
            runtime_root=Path(native_runtime.name),
            asset_root=asset_root_requested,
        )
        run_manifest["vendor_task_imported_after_app_launcher"] = True
        run_manifest["tacex_asset_root"] = str(runtime_context.asset_root)
        run_manifest["status"] = "running"
        write_json(output_root / "run_manifest.json", run_manifest)

        env_cfg = _configure_task(
            task_module,
            config,
            Path(native_runtime.name),
            getattr(args, "device", None),
            task_root,
        )
        task = task_module.Task(env_cfg, mode="eval")
        displacement_mm = tuple(float(value) for value in config["probe_delta_mm"])
        displacement_m = tuple(value / 1000.0 for value in displacement_mm)
        seeds = [int(seed) for seed in config["seeds"]]

        for seed_index, seed in enumerate(seeds):
            result = _seed_result(seed)
            summary["seeds"][str(seed)] = result
            seed_dir = output_root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=False)
            action_id = f"insert-hole-seed-{seed}-probe-0001"
            reset_attempted = False
            try:
                reset_attempted = True
                task.reset(seed=seed)
                if task.plan_success is not True:
                    raise RuntimeError(
                        f"seed {seed} reset pre_move returned with plan_success=false"
                    )
                result["reset_ok"] = True
                task_instruction = str(task.instruction).strip() or str(config["task"]).replace(
                    "_", " "
                )

                pre_observation = task._get_observations()
                pre_capture = capture_snapshot(
                    pre_observation,
                    output_root=output_root,
                    seed_dir=seed_dir,
                    task_name=str(config["task"]),
                    seed=seed,
                    phase="pre_action",
                    action_id=action_id,
                    simulator_step=int(task.step_count),
                    take_action_count=int(task.take_action_cnt),
                    task_instruction=task_instruction,
                    task_metadata=task.metadata,
                    native_check_success=None,
                    save_host_only=bool(config["save_host_only"]),
                    strict_two_tactile_sensors=bool(
                        config["strict_two_tactile_sensors"]
                    ),
                    fail_on_missing_rgb_marker=bool(
                        config["fail_on_missing_rgb_marker"]
                    ),
                )
                write_snapshot(seed_dir / "snapshot_pre.json", pre_capture.snapshot)

                actions = task.atom.move_by_displacement(
                    x=displacement_m[0],
                    y=displacement_m[1],
                    z=displacement_m[2],
                    xyz_coord="world",
                )
                move_ok = task.move(
                    actions,
                    tag=action_id,
                    is_save=False,
                    delay=False,
                    time_dilation_factor=float(
                        config["probe_time_dilation_factor"]
                    ),
                )
                if move_ok is not True:
                    raise RuntimeError(f"official task.move probe failed for seed {seed}")
                result["probe_ok"] = True

                post_observation = task._get_observations()
                native_success = bool(task.check_success())
                post_capture = capture_snapshot(
                    post_observation,
                    output_root=output_root,
                    seed_dir=seed_dir,
                    task_name=str(config["task"]),
                    seed=seed,
                    phase="post_action",
                    action_id=action_id,
                    simulator_step=int(task.step_count),
                    take_action_count=int(task.take_action_cnt),
                    task_instruction=task_instruction,
                    task_metadata=task.metadata,
                    native_check_success=native_success,
                    save_host_only=bool(config["save_host_only"]),
                    strict_two_tactile_sensors=bool(
                        config["strict_two_tactile_sensors"]
                    ),
                    fail_on_missing_rgb_marker=bool(
                        config["fail_on_missing_rgb_marker"]
                    ),
                )
                write_snapshot(seed_dir / "snapshot_post.json", post_capture.snapshot)

                transition = build_transition(
                    output_root=output_root,
                    seed_dir=seed_dir,
                    pre=pre_capture.snapshot,
                    post=post_capture.snapshot,
                    pre_rgb_markers=pre_capture.rgb_markers,
                    post_rgb_markers=post_capture.rgb_markers,
                    requested_displacement_mm=displacement_mm,
                    native_check_success=native_success,
                )
                write_json(seed_dir / "transition.json", transition.to_dict())
                verify_artifacts(
                    output_root,
                    pre_capture.snapshot.to_dict(),
                    post_capture.snapshot.to_dict(),
                    transition.to_dict(),
                )
                for required_path in (
                    seed_dir / "snapshot_pre.json",
                    seed_dir / "snapshot_post.json",
                    seed_dir / "transition.json",
                ):
                    if not required_path.is_file():
                        raise RuntimeError(f"required trace file is missing: {required_path}")

                step_changed = (
                    post_capture.snapshot.simulator_step
                    != pre_capture.snapshot.simulator_step
                )
                action_changed = (
                    post_capture.snapshot.take_action_count
                    != pre_capture.snapshot.take_action_count
                )
                if not step_changed and not action_changed:
                    raise RuntimeError(
                        "world action changed neither simulator_step nor take_action_count"
                    )

                sensor_names = sorted(pre_capture.rgb_markers)
                tactile_shapes = {
                    name: {
                        "pre": list(pre_capture.rgb_markers[name].shape),
                        "post": list(post_capture.rgb_markers[name].shape),
                        "dtype": str(pre_capture.rgb_markers[name].dtype),
                        "range": [
                            float(pre_capture.rgb_markers[name].min()),
                            float(pre_capture.rgb_markers[name].max()),
                        ],
                    }
                    for name in sensor_names
                }
                changed_by_sensor = {
                    name: bool(
                        transition.tactile_differences[name][
                            "max_absolute_rgb_marker_difference"
                        ]
                        > 0
                    )
                    for name in sensor_names
                }
                result.update(
                    {
                        "observation_ok": True,
                        "tactile_sensor_names": sensor_names,
                        "tactile_shapes": tactile_shapes,
                        "step_count_changed": step_changed,
                        "action_count_changed": action_changed,
                        "tactile_changed": any(changed_by_sensor.values()),
                        "tactile_changed_by_sensor": changed_by_sensor,
                        "native_check_success": native_success,
                        "observation_key_tree": {
                            "pre": pre_capture.observation_key_tree,
                            "post": post_capture.observation_key_tree,
                        },
                        "tactile_difference_statistics": transition.tactile_differences,
                    }
                )
                if run_manifest["observation_key_tree"] is None:
                    run_manifest["observation_key_tree"] = pre_capture.observation_key_tree
            except Exception as exc:
                overall_error = True
                error_payload = write_exception(seed_dir / "exception.json", exc)
                result["error"] = {
                    "type": error_payload["error_type"],
                    "message": error_payload["error"],
                    "traceback_path": f"seed_{seed}/exception.json",
                }
            finally:
                if reset_attempted:
                    try:
                        task.clean_cache(result=None)
                        result["cleanup_ok"] = True
                    except Exception as cleanup_exc:
                        overall_error = True
                        cleanup_abort = True
                        cleanup_payload = write_exception(
                            seed_dir / "cleanup_exception.json", cleanup_exc
                        )
                        result["cleanup_ok"] = False
                        result["error"] = {
                            "type": cleanup_payload["error_type"],
                            "message": cleanup_payload["error"],
                            "traceback_path": f"seed_{seed}/cleanup_exception.json",
                        }
                _write_summary(output_root, summary)
                write_json(output_root / "run_manifest.json", run_manifest)

            if cleanup_abort:
                for skipped_seed in seeds[seed_index + 1 :]:
                    skipped = _seed_result(skipped_seed)
                    skipped["error"] = {
                        "type": "CleanupAbort",
                        "message": f"not run after cleanup failure for seed {seed}",
                    }
                    summary["seeds"][str(skipped_seed)] = skipped
                break
    except Exception as exc:
        overall_error = True
        fatal_payload = write_exception(output_root / "exception.json", exc)
        summary["error"] = {
            "type": fatal_payload["error_type"],
            "message": fatal_payload["error"],
            "traceback_path": "exception.json",
        }
        run_manifest["error"] = summary["error"]
    finally:
        close_errors: list[dict[str, str]] = []
        if task is not None:
            try:
                task.close()
                run_manifest["task_close_ok"] = True
            except Exception as exc:
                run_manifest["task_close_ok"] = False
                close_errors.append({"component": "task", "error": str(exc)})
        os.chdir(original_cwd)
        if close_errors:
            overall_error = True
            write_json(output_root / "close_errors.json", {"errors": close_errors})
            summary["error"] = {
                "type": "CleanupError",
                "message": "task or simulation_app cleanup failed",
                "traceback_path": "close_errors.json",
            }
        summary["status"] = "failed" if overall_error else "passed"
        run_manifest["status"] = summary["status"]
        run_manifest["ended_at"] = _utc_now()
        run_manifest["error"] = summary.get("error")
        _write_summary(output_root, summary)
        write_json(output_root / "run_manifest.json", run_manifest)

        if simulation_app is not None:
            run_manifest["simulation_app_close_attempted"] = True
            write_json(output_root / "run_manifest.json", run_manifest)
            try:
                simulation_app.close()
            except Exception as exc:
                overall_error = True
                close_errors.append({"component": "simulation_app", "error": str(exc)})
                write_json(output_root / "close_errors.json", {"errors": close_errors})
                summary["status"] = "failed"
                summary["error"] = {
                    "type": "CleanupError",
                    "message": "simulation_app cleanup failed",
                    "traceback_path": "close_errors.json",
                }
                run_manifest["status"] = "failed"
                run_manifest["error"] = summary["error"]
                _write_summary(output_root, summary)
                write_json(output_root / "run_manifest.json", run_manifest)
        native_runtime.cleanup()

    return 1 if overall_error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--source-root",
        default=str(REPO_ROOT / "third_party/ftp1-policy/UniVTAC"),
    )
    parser.add_argument("--asset-root", default=None)

    assert_task_not_imported("insert_hole")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return run_smoke(args)


if __name__ == "__main__":
    raise SystemExit(main())
