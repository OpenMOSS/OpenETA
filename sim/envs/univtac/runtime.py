"""Runtime and asset provenance helpers for direct UniVTAC diagnostics."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


TACEX_PACKAGE_NAMES = ("tacex_assets", "tacex_uipc", "tacex_tasks", "tacex")
HIGH_RES_ROBOT_RELATIVE_PATH = Path(
    "Robots/Franka/GelSight_Mini/Gripper/uipc_gelpads_high_res_wrist.usd"
)
HIGH_RES_GELPAD_RELATIVE_PATH = Path("Sensors/GelSight_Mini/Gelpad_high_res.usd")
CALIBRATION_RELATIVE_PATH = Path("Sensors/GelSight_Mini/calibs/640x480/dataPack.npz")
LOW_RES_ASSET_NAMES = frozenset({"uipc_gelpads.usd", "Gelpad_low_res.usd"})
RUNTIME_SCHEMA_VERSION = "openeta.univtac.runtime_manifest.v1"
ASSET_SCHEMA_VERSION = "openeta.univtac.asset_manifest.v1"


class UniVTACRuntimeError(RuntimeError):
    """Raised when the requested UniVTAC runtime cannot be assembled safely."""


@dataclass(frozen=True)
class RuntimeContext:
    """Resolved code and data roots used by one simulator process."""

    task_root: Path
    tacex_source_root: Path
    overlay_root: Path
    asset_root: Path


def _run_text(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def resolve_asset_root(task_root: Path, configured: str | Path | None = None) -> Path:
    """Resolve a complete high-resolution TacEx data root from explicit inputs."""

    candidates: list[Path] = []
    if configured is not None:
        candidates.append(Path(configured).expanduser())
    environment_path = os.environ.get("TACEX_ASSETS_DATA_DIR")
    if environment_path:
        candidates.append(Path(environment_path).expanduser())
    candidates.append(
        task_root
        / "third_party"
        / "TacEx"
        / "source"
        / "tacex_assets"
        / "tacex_assets"
        / "data"
    )

    failures: list[str] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        missing = [
            str(relative)
            for relative in (
                HIGH_RES_ROBOT_RELATIVE_PATH,
                HIGH_RES_GELPAD_RELATIVE_PATH,
                CALIBRATION_RELATIVE_PATH,
            )
            if not (candidate / relative).is_file()
        ]
        if missing:
            failures.append(f"{candidate}: missing {missing}")
            continue
        assert_high_res_assets(candidate)
        return candidate
    raise UniVTACRuntimeError(
        "no complete high-resolution TacEx asset root was found; " + "; ".join(failures)
    )


def _low_res_alias_details(path: Path) -> dict[str, Any] | None:
    if not path.is_symlink():
        return None
    target = os.readlink(path)
    realpath = path.resolve()
    target_name = Path(target).name
    if target_name in LOW_RES_ASSET_NAMES or realpath.name in LOW_RES_ASSET_NAMES:
        return {
            "path": str(path),
            "symlink_target": target,
            "realpath": str(realpath),
        }
    return None


def assert_high_res_assets(asset_root: Path) -> None:
    """Fail when a required high-resolution USD aliases a known low-res asset."""

    for relative in (HIGH_RES_ROBOT_RELATIVE_PATH, HIGH_RES_GELPAD_RELATIVE_PATH):
        path = asset_root / relative
        details = _low_res_alias_details(path)
        if details is not None:
            raise UniVTACRuntimeError(
                "high-resolution asset resolves to a low-resolution alias: "
                + json.dumps(details, sort_keys=True)
            )


def prepare_runtime(
    *,
    task_root: Path,
    runtime_root: Path,
    asset_root: str | Path | None,
) -> RuntimeContext:
    """Overlay checkout-owned TacEx code on an explicit read-only asset root."""

    task_root = task_root.expanduser().resolve()
    runtime_root = runtime_root.expanduser().resolve()
    source_root = task_root / "third_party" / "TacEx" / "source"
    extension_source = source_root / "tacex_assets"
    if not (task_root / "envs").is_dir():
        raise UniVTACRuntimeError(f"UniVTAC env package is missing: {task_root / 'envs'}")
    if not (extension_source / "config" / "extension.toml").is_file():
        raise UniVTACRuntimeError(f"vendored TacEx extension is incomplete: {extension_source}")

    resolved_assets = resolve_asset_root(task_root, asset_root)
    overlay_root = runtime_root / "tacex_assets_extension"
    shutil.copytree(extension_source, overlay_root)
    overlay_data = overlay_root / "tacex_assets" / "data"
    if overlay_data.is_symlink() or overlay_data.is_file():
        overlay_data.unlink()
    elif overlay_data.is_dir():
        shutil.rmtree(overlay_data)
    overlay_data.symlink_to(resolved_assets, target_is_directory=True)

    python_roots = (
        task_root,
        source_root,
        source_root / "tacex",
        source_root / "tacex_uipc",
        source_root / "tacex_tasks",
        overlay_root,
    )
    for python_root in python_roots:
        if not python_root.is_dir():
            raise UniVTACRuntimeError(f"runtime Python root does not exist: {python_root}")
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
    envs_package.__path__ = [str(task_root / "envs")]
    envs_package.__package__ = "envs"
    sys.modules["envs"] = envs_package

    os.environ["TACEX_ASSETS_DATA_DIR"] = str(resolved_assets)
    return RuntimeContext(
        task_root=task_root,
        tacex_source_root=source_root,
        overlay_root=overlay_root,
        asset_root=resolved_assets,
    )


def assert_task_not_imported(task_name: str) -> None:
    imported = [
        name
        for name in ("envs._base_task", f"envs.{task_name}")
        if name in sys.modules
    ]
    if imported:
        raise UniVTACRuntimeError(
            f"UniVTAC task modules were imported before AppLauncher: {imported}"
        )


def import_task_after_launcher(
    *,
    simulation_app: Any,
    task_root: Path,
    task_name: str,
    runtime_root: Path,
    asset_root: str | Path | None,
) -> tuple[Any, RuntimeContext]:
    if simulation_app is None:
        raise UniVTACRuntimeError("AppLauncher must start before UniVTAC task import")
    context = prepare_runtime(
        task_root=task_root,
        runtime_root=runtime_root,
        asset_root=asset_root,
    )
    task_module = importlib.import_module(f"envs.{task_name}")
    return task_module, context


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_asset(path: Path) -> dict[str, Any]:
    path = path.expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(f"required asset is missing: {path}")
    is_symlink = path.is_symlink()
    return {
        "path": str(path),
        "realpath": str(path.resolve()),
        "is_symlink": is_symlink,
        "symlink_target": os.readlink(path) if is_symlink else None,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def collect_asset_manifest(context: RuntimeContext) -> dict[str, Any]:
    assert_high_res_assets(context.asset_root)
    task_root = context.task_root
    asset_paths = {
        "high_res_franka_gelsight_robot_usd": (
            context.asset_root / HIGH_RES_ROBOT_RELATIVE_PATH
        ),
        "high_res_gelpad_usd": context.asset_root / HIGH_RES_GELPAD_RELATIVE_PATH,
        "gelsight_calibration_datapack": context.asset_root / CALIBRATION_RELATIVE_PATH,
        "insert_hole_slot_usd": task_root / "assets/objects/TestTubeHoleSlot.usd",
        "insert_hole_base_usd": task_root / "assets/objects/TestTubeBase.usd",
        "insert_hole_object_usd": task_root / "assets/objects/TestTube.usd",
        "franka_curobo_yaml": task_root / "assets/embodiments/franka/curobo.yml",
        "franka_curobo_urdf": task_root / "assets/embodiments/franka/panda.urdf",
        "franka_collision_spheres_yaml": (
            task_root / "assets/embodiments/franka/collision_franka.yml"
        ),
    }
    return {
        "schema_version": ASSET_SCHEMA_VERSION,
        "task_root": str(task_root),
        "asset_root": str(context.asset_root),
        "assets": {
            name: describe_asset(path) for name, path in sorted(asset_paths.items())
        },
    }


def _module_source_kind(
    path: Path | None,
    *,
    context: RuntimeContext,
    current_task_root: Path,
) -> str:
    if path is None:
        return "unavailable"
    resolved = path.resolve()
    if resolved.is_relative_to(context.overlay_root):
        return "current_runtime_overlay"
    if resolved.is_relative_to(current_task_root.resolve()):
        return "current_repository_vendored_source"
    if resolved.is_relative_to(context.task_root) or str(resolved).startswith("/tmp/"):
        return "external_checkout"
    if resolved.is_relative_to(Path(sys.prefix).resolve()):
        return "conda_site_packages"
    return "other"


def _module_record(
    module_name: str,
    *,
    context: RuntimeContext,
    current_task_root: Path,
) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        try:
            spec = importlib.util.find_spec(module_name)
        except Exception:
            spec = None
        origin = getattr(spec, "origin", None)
        module_file = Path(origin).absolute() if origin and origin != "namespace" else None
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "module_file": str(module_file) if module_file else None,
            "source_kind": _module_source_kind(
                module_file,
                context=context,
                current_task_root=current_task_root,
            ),
            "version": _package_version(module_name.split(".")[0]),
        }
    module_file_value = getattr(module, "__file__", None)
    module_file = Path(module_file_value).absolute() if module_file_value else None
    version = getattr(module, "__version__", None) or _package_version(
        module_name.split(".")[0]
    )
    return {
        "available": True,
        "module_file": str(module_file) if module_file else None,
        "source_kind": _module_source_kind(
            module_file,
            context=context,
            current_task_root=current_task_root,
        ),
        "version": str(version) if version is not None else None,
    }


def _read_sysctl(name: str) -> int | None:
    path = Path("/proc/sys") / Path(*name.split("."))
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _visible_isaac_process_count() -> int:
    count = 0
    uid = os.getuid()
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            if process_dir.stat().st_uid != uid:
                continue
            command = (process_dir / "cmdline").read_bytes().replace(b"\x00", b" ").lower()
        except OSError:
            continue
        if any(token in command for token in (b"isaac", b"kit", b"univtac")):
            count += 1
    return count


def _related_entries(values: Iterable[str]) -> list[str]:
    keywords = ("univtac", "tacex", "isaac", "curobo", "cuda")
    return sorted(
        value for value in values if any(keyword in value.lower() for keyword in keywords)
    )


def safe_runtime_path_snapshot(
    *,
    sys_path: Iterable[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Return only runtime-related paths, never arbitrary environment variables."""

    environment = os.environ if environment is None else environment
    ld_entries = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    return {
        "related_sys_path_entries": _related_entries(
            str(path) for path in (sys.path if sys_path is None else sys_path)
        ),
        "related_ld_library_path_entries": _related_entries(ld_entries),
    }


def collect_runtime_manifest(
    *,
    repo_root: Path,
    current_task_root: Path,
    context: RuntimeContext,
    task_name: str,
) -> dict[str, Any]:
    """Collect runtime provenance without copying credentials or arbitrary env values."""

    import torch

    driver_version = _run_text(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    modules = {
        name: _module_record(
            name,
            context=context,
            current_task_root=current_task_root,
        )
        for name in (
            "curobo",
            "isaaclab",
            "tacex",
            "tacex_assets",
            "tacex_tasks",
            "tacex_uipc",
            f"envs.{task_name}",
        )
    }
    path_snapshot = safe_runtime_path_snapshot()
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "git": {
            "head": _run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
            "branch": _run_text(["git", "branch", "--show-current"], cwd=repo_root),
        },
        "task_source_git": {
            "head": _run_text(["git", "rev-parse", "HEAD"], cwd=context.task_root),
            "branch": _run_text(
                ["git", "branch", "--show-current"], cwd=context.task_root
            ),
            "dirty": bool(
                _run_text(["git", "status", "--short"], cwd=context.task_root)
            ),
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "torch": {
            "version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        },
        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver_version": driver_version.splitlines()[0] if driver_version else None,
        },
        "dependencies": {
            "isaac_sim_version": _package_version("isaacsim"),
            "isaac_lab_version": modules["isaaclab"]["version"],
            "curobo_version": modules["curobo"]["version"],
        },
        "modules": modules,
        "paths": {
            "repo_root": str(repo_root.resolve()),
            "task_root": str(context.task_root),
            "tacex_source_root": str(context.tacex_source_root),
            "runtime_overlay": str(context.overlay_root),
            "tacex_asset_root": str(context.asset_root),
            "current_working_directory": str(Path.cwd()),
            **path_snapshot,
        },
        "system": {
            "inotify_max_user_instances": _read_sysctl("fs.inotify.max_user_instances"),
            "inotify_max_user_watches": _read_sysctl("fs.inotify.max_user_watches"),
            "visible_isaac_kit_process_count": _visible_isaac_process_count(),
        },
    }


def manifest_contains_secret_keys(value: Any) -> bool:
    """Return whether a manifest accidentally contains credential-shaped keys."""

    secret_tokens = ("token", "secret", "password", "authorization", "api_key", "apikey")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if any(token in str(key).lower() for token in secret_tokens):
                return True
            if manifest_contains_secret_keys(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(manifest_contains_secret_keys(child) for child in value)
    return False
