#!/usr/bin/env python3
"""Manage local OpenETA MCP services for OpenETA perception and manipulation."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.runtime.calibration_registry import (  # noqa: E402
    DEFAULT_GRASP_CALIBRATION_PROFILE,
    load_grasp_calibration_capabilities,
)
from agent.tools.anygrasp_capabilities import (  # noqa: E402
    ANYGRASP_WIDTH_TOLERANCE_M,
    check_anygrasp_compatibility,
    clear_anygrasp_capability_cache,
)


DEFAULT_STATE_DIR = Path("outputs/mcp_services")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_SAM3_PORT = 8773
DEFAULT_ANYGRASP_PORT = 8774
DEFAULT_ANYGRASP_GRIPPER_HEIGHT_M = 0.03
DEFAULT_ANYPLACE_PORT = 8775
DEFAULT_CONTACT_GRASPNET_PORT = 8776
DEFAULT_MOLMOPOINT_PORT = 8777
DEFAULT_GRASPGENX_PORT = 8778
DEFAULT_UNIDEPTH_V2_PORT = 8779
DEFAULT_UNIDEPTH_V2_MODEL_ID = "lpiccinelli/unidepth-v2-vitl14"
DEFAULT_MOLMOPOINT_MODEL_ID = "allenai/MolmoPoint-8B"
DEFAULT_MOLMOPOINT_MODEL_REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
STOP_TIMEOUT_S = 5.0
START_TIMEOUT_S = 8.0
ANYGRASP_CAPABILITY_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class ServiceConfig:
    name: str
    python: str
    host: str
    port: int
    state_dir: Path
    command: list[str]
    env: dict[str, str]
    health_server_name: str | None = None
    expected_max_gripper_width_m: float | None = None
    grasp_calibration_profile: Path | None = None
    capability_timeout_s: float = ANYGRASP_CAPABILITY_TIMEOUT_S

    @property
    def pid_file(self) -> Path:
        return self.state_dir / f"{self.name}.pid"

    @property
    def log_file(self) -> Path:
        return self.state_dir / f"{self.name}.log"

    @property
    def health_url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    @property
    def sse_url(self) -> str:
        return f"http://{self.host}:{self.port}/sse"

    @property
    def expected_health_server(self) -> str:
        return self.health_server_name or self.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage OpenETA MCP services.")
    parser.add_argument(
        "command",
        choices=("start", "stop", "restart", "status", "health", "smoke"),
    )
    parser.add_argument(
        "target",
        choices=(
            "sam3",
            "anygrasp",
            "anyplace",
            "contact_graspnet",
            "molmopoint",
            "graspgenx",
            "unidepth_v2",
            "all",
        ),
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--sam3-port", type=int, default=DEFAULT_SAM3_PORT)
    parser.add_argument("--anygrasp-port", type=int, default=DEFAULT_ANYGRASP_PORT)
    parser.add_argument("--anyplace-port", type=int, default=DEFAULT_ANYPLACE_PORT)
    parser.add_argument(
        "--contact-graspnet-port",
        type=int,
        default=DEFAULT_CONTACT_GRASPNET_PORT,
    )
    parser.add_argument("--molmopoint-port", type=int, default=DEFAULT_MOLMOPOINT_PORT)
    parser.add_argument("--graspgenx-port", type=int, default=DEFAULT_GRASPGENX_PORT)
    parser.add_argument("--unidepth-v2-port", type=int, default=DEFAULT_UNIDEPTH_V2_PORT)
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--sam3-python")
    parser.add_argument("--anygrasp-python")
    parser.add_argument("--anyplace-python")
    parser.add_argument("--contact-graspnet-python")
    parser.add_argument("--molmopoint-python")
    parser.add_argument("--graspgenx-python")
    parser.add_argument("--unidepth-v2-python")
    parser.add_argument("--anygrasp-sdk-root")
    parser.add_argument("--anygrasp-checkpoint-path")
    parser.add_argument(
        "--grasp-calibration-profile",
        type=Path,
        help=(
            "host calibration profile that owns the physical gripper width; "
            "defaults to OPENETA_GRASP_CALIBRATION_PROFILE or the repository "
            "LIBERO Panda profile"
        ),
    )
    parser.add_argument(
        "--anygrasp-max-gripper-width",
        type=float,
        default=None,
        help=(
            "optional deployment width override in metres; when supplied it must "
            "match --grasp-calibration-profile"
        ),
    )
    parser.add_argument(
        "--anygrasp-gripper-height",
        type=float,
        default=DEFAULT_ANYGRASP_GRIPPER_HEIGHT_M,
        help="finger height used by AnyGrasp collision geometry (metres)",
    )
    parser.add_argument(
        "--anygrasp-capability-timeout-s",
        type=float,
        default=ANYGRASP_CAPABILITY_TIMEOUT_S,
        help="timeout for pre/post-restart get_capabilities checks",
    )
    parser.add_argument("--anyplace-root")
    parser.add_argument("--anyplace-config-path")
    parser.add_argument("--contact-graspnet-root")
    parser.add_argument("--contact-graspnet-checkpoint-dir")
    parser.add_argument("--sam3-hf-home")
    parser.add_argument("--sam3-cache-dir")
    parser.add_argument("--molmopoint-hf-home")
    parser.add_argument("--molmopoint-model-id")
    parser.add_argument("--molmopoint-model-revision")
    parser.add_argument("--graspgenx-root")
    parser.add_argument("--graspgenx-checkpoint-root")
    parser.add_argument("--graspgenx-gripper-descriptions-root")
    parser.add_argument("--unidepth-v2-model-id")
    parser.add_argument("--unidepth-v2-device")
    parser.add_argument("--unidepth-v2-resolution-level", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        configs = _build_configs(args)
        if args.command in {"start", "restart"}:
            _validate_start_requirements(configs)
        if args.command == "start":
            results = {config.name: _start_service(config, dry_run=args.dry_run) for config in configs}
        elif args.command == "stop":
            results = {config.name: _stop_service(config, force=args.force) for config in configs}
        elif args.command == "restart":
            results = {}
            for config in configs:
                if args.dry_run:
                    results[config.name] = _restart_dry_run(config)
                else:
                    stop_result = _stop_service(config, force=args.force, missing_ok=True)
                    start_result = _start_service(config, dry_run=False)
                    results[config.name] = {
                        "stop": stop_result,
                        "start": start_result,
                        "ok": bool(stop_result.get("ok")) and bool(start_result.get("ok")),
                    }
        elif args.command == "status":
            results = {config.name: _status_service(config) for config in configs}
        elif args.command == "health":
            results = {config.name: _health_service(config) for config in configs}
        elif args.command == "smoke":
            results = {config.name: _smoke_service(config) for config in configs}
        else:  # pragma: no cover - argparse prevents this.
            parser.error(f"unknown command: {args.command}")
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    _print_results(results, json_output=args.json_output)
    if args.command == "status":
        return 0
    return 0 if _results_ok(results) else 1


def _build_configs(args: argparse.Namespace) -> list[ServiceConfig]:
    targets = (
        (
            "sam3",
            "anygrasp",
            "anyplace",
            "contact_graspnet",
            "molmopoint",
            "graspgenx",
            "unidepth_v2",
        )
        if args.target == "all"
        else (args.target,)
    )
    return [_build_config(target, args) for target in targets]


def _build_config(name: str, args: argparse.Namespace) -> ServiceConfig:
    state_dir = _resolve_state_dir(args.state_dir)
    env = os.environ.copy()
    if name == "sam3":
        python = args.sam3_python or os.environ.get("OPENETA_SAM3_PYTHON") or sys.executable
        if args.sam3_hf_home:
            env["HF_HOME"] = args.sam3_hf_home
        if args.sam3_cache_dir:
            env["HF_HUB_CACHE"] = args.sam3_cache_dir
        command = [
            python,
            str(REPO_ROOT / "tools" / "sam3_mcp_server.py"),
            "--transport",
            "sse",
            "--host",
            args.host,
            "--port",
            str(args.sam3_port),
        ]
        return ServiceConfig(
            name=name,
            python=python,
            host=args.host,
            port=args.sam3_port,
            state_dir=state_dir,
            command=command,
            env=env,
        )

    if name == "anygrasp":
        python = (
            args.anygrasp_python or os.environ.get("OPENETA_ANYGRASP_PYTHON") or sys.executable
        )
        sdk_root = args.anygrasp_sdk_root or os.environ.get("OPENETA_ANYGRASP_SDK_ROOT")
        checkpoint_path = args.anygrasp_checkpoint_path or os.environ.get(
            "OPENETA_ANYGRASP_CHECKPOINT_PATH"
        )
        (
            max_gripper_width,
            calibration_profile,
        ) = _resolve_anygrasp_deployment_width(args)
        command = [
            python,
            str(REPO_ROOT / "tools" / "anygrasp_mcp_server.py"),
            "--transport",
            "sse",
            "--host",
            args.host,
            "--port",
            str(args.anygrasp_port),
        ]
        if sdk_root:
            command.extend(["--sdk-root", sdk_root])
        if checkpoint_path:
            command.extend(["--checkpoint-path", checkpoint_path])
        command.extend(
            [
                "--max-gripper-width",
                str(max_gripper_width),
                "--gripper-height",
                str(args.anygrasp_gripper_height),
            ]
        )
        return ServiceConfig(
            name=name,
            python=python,
            host=args.host,
            port=args.anygrasp_port,
            state_dir=state_dir,
            command=command,
            env=env,
            expected_max_gripper_width_m=max_gripper_width,
            grasp_calibration_profile=calibration_profile,
            capability_timeout_s=float(args.anygrasp_capability_timeout_s),
        )

    if name == "anyplace":
        python = (
            args.anyplace_python
            or os.environ.get("OPENETA_ANYPLACE_PYTHON")
            or sys.executable
        )
        anyplace_root = args.anyplace_root or os.environ.get("OPENETA_ANYPLACE_ROOT")
        config_path = args.anyplace_config_path or os.environ.get(
            "OPENETA_ANYPLACE_CONFIG_PATH"
        )
        command = [
            python,
            str(REPO_ROOT / "tools" / "anyplace_mcp_server.py"),
            "--transport",
            "sse",
            "--host",
            args.host,
            "--port",
            str(args.anyplace_port),
        ]
        if anyplace_root:
            command.extend(["--anyplace-root", anyplace_root])
        if config_path:
            command.extend(["--config-path", config_path])
        return ServiceConfig(
            name=name,
            python=python,
            host=args.host,
            port=args.anyplace_port,
            state_dir=state_dir,
            command=command,
            env=env,
        )

    if name == "molmopoint":
        python = (
            args.molmopoint_python
            or os.environ.get("OPENETA_MOLMOPOINT_PYTHON")
            or sys.executable
        )
        hf_home = args.molmopoint_hf_home or os.environ.get("OPENETA_MOLMOPOINT_HF_HOME")
        model_id = (
            args.molmopoint_model_id
            or os.environ.get("OPENETA_MOLMOPOINT_MODEL_ID")
            or DEFAULT_MOLMOPOINT_MODEL_ID
        )
        model_revision = (
            args.molmopoint_model_revision
            or os.environ.get("OPENETA_MOLMOPOINT_MODEL_REVISION")
            or DEFAULT_MOLMOPOINT_MODEL_REVISION
        )
        command = [
            python,
            str(REPO_ROOT / "tools" / "molmopoint_mcp_server.py"),
            "--transport",
            "sse",
            "--host",
            args.host,
            "--port",
            str(args.molmopoint_port),
            "--model-id",
            model_id,
            "--model-revision",
            model_revision,
        ]
        if hf_home:
            command.extend(["--hf-home", hf_home])
            env["HF_HOME"] = hf_home
        return ServiceConfig(
            name=name,
            python=python,
            host=args.host,
            port=args.molmopoint_port,
            state_dir=state_dir,
            command=command,
            env=env,
        )

    if name == "graspgenx":
        python = (
            args.graspgenx_python
            or os.environ.get("OPENETA_GRASPGENX_PYTHON")
            or sys.executable
        )
        backend_root = args.graspgenx_root or os.environ.get(
            "OPENETA_GRASPGENX_ROOT"
        )
        checkpoint_root = args.graspgenx_checkpoint_root or os.environ.get(
            "OPENETA_GRASPGENX_CHECKPOINT_ROOT"
        )
        gripper_root = (
            args.graspgenx_gripper_descriptions_root
            or os.environ.get("OPENETA_GRASPGENX_GRIPPER_DESCRIPTIONS_ROOT")
        )
        command = [
            python,
            str(REPO_ROOT / "tools" / "graspgenx_mcp_server.py"),
            "--transport",
            "sse",
            "--host",
            args.host,
            "--port",
            str(args.graspgenx_port),
        ]
        if backend_root:
            command.extend(["--graspgenx-root", backend_root])
        if checkpoint_root:
            command.extend(["--checkpoint-root", checkpoint_root])
        if gripper_root:
            command.extend(["--gripper-descriptions-root", gripper_root])
        return ServiceConfig(
            name=name,
            python=python,
            host=args.host,
            port=args.graspgenx_port,
            state_dir=state_dir,
            command=command,
            env=env,
            health_server_name="openeta-graspgenx",
        )

    if name == "unidepth_v2":
        python = (
            args.unidepth_v2_python
            or os.environ.get("OPENETA_UNIDEPTH_V2_PYTHON")
            or sys.executable
        )
        model_id = (
            args.unidepth_v2_model_id
            or os.environ.get("OPENETA_UNIDEPTH_V2_MODEL_ID")
            or DEFAULT_UNIDEPTH_V2_MODEL_ID
        )
        device = (
            args.unidepth_v2_device
            or os.environ.get("OPENETA_UNIDEPTH_V2_DEVICE")
            or "auto"
        )
        resolution_level = (
            args.unidepth_v2_resolution_level
            if args.unidepth_v2_resolution_level is not None
            else int(os.environ.get("OPENETA_UNIDEPTH_V2_RESOLUTION_LEVEL", "4"))
        )
        command = [
            python,
            str(REPO_ROOT / "tools" / "unidepth_v2_mcp_server.py"),
            "--transport",
            "sse",
            "--host",
            args.host,
            "--port",
            str(args.unidepth_v2_port),
            "--model-id",
            model_id,
            "--device",
            device,
            "--resolution-level",
            str(resolution_level),
        ]
        return ServiceConfig(
            name=name,
            python=python,
            host=args.host,
            port=args.unidepth_v2_port,
            state_dir=state_dir,
            command=command,
            env=env,
            health_server_name="unidepth-v2",
        )

    python = (
        args.contact_graspnet_python
        or os.environ.get("OPENETA_CONTACT_GRASPNET_PYTHON")
        or sys.executable
    )
    backend_root = args.contact_graspnet_root or os.environ.get(
        "OPENETA_CONTACT_GRASPNET_ROOT"
    )
    checkpoint_dir = args.contact_graspnet_checkpoint_dir or os.environ.get(
        "OPENETA_CONTACT_GRASPNET_CHECKPOINT_DIR"
    )
    command = [
        python,
        str(REPO_ROOT / "tools" / "contact_graspnet_mcp_server.py"),
        "--transport",
        "sse",
        "--host",
        args.host,
        "--port",
        str(args.contact_graspnet_port),
    ]
    if backend_root:
        command.extend(["--contact-graspnet-root", backend_root])
    if checkpoint_dir:
        command.extend(["--checkpoint-dir", checkpoint_dir])
    return ServiceConfig(
        name=name,
        python=python,
        host=args.host,
        port=args.contact_graspnet_port,
        state_dir=state_dir,
        command=command,
        env=env,
    )


def _validate_start_requirements(configs: Iterable[ServiceConfig]) -> None:
    for config in configs:
        if config.name == "anygrasp":
            if "--sdk-root" not in config.command:
                raise ConfigError(
                    "AnyGrasp sdk root is required: pass --anygrasp-sdk-root or set "
                    "OPENETA_ANYGRASP_SDK_ROOT."
                )
            if "--checkpoint-path" not in config.command:
                raise ConfigError(
                    "AnyGrasp checkpoint path is required: pass --anygrasp-checkpoint-path or set "
                    "OPENETA_ANYGRASP_CHECKPOINT_PATH."
                )
        if config.name == "anyplace":
            if "--anyplace-root" not in config.command:
                raise ConfigError(
                    "AnyPlace root is required: pass --anyplace-root or set "
                    "OPENETA_ANYPLACE_ROOT."
                )
            if "--config-path" not in config.command:
                raise ConfigError(
                    "AnyPlace config path is required: pass --anyplace-config-path or set "
                    "OPENETA_ANYPLACE_CONFIG_PATH."
                )
        if config.name == "contact_graspnet":
            if "--contact-graspnet-root" not in config.command:
                raise ConfigError(
                    "Contact-GraspNet root is required: pass --contact-graspnet-root "
                    "or set OPENETA_CONTACT_GRASPNET_ROOT."
                )
            if "--checkpoint-dir" not in config.command:
                raise ConfigError(
                    "Contact-GraspNet checkpoint directory is required: pass "
                    "--contact-graspnet-checkpoint-dir or set "
                    "OPENETA_CONTACT_GRASPNET_CHECKPOINT_DIR."
                )
        if config.name == "graspgenx":
            if "--graspgenx-root" not in config.command:
                raise ConfigError(
                    "GraspGenX root is required: pass --graspgenx-root or set "
                    "OPENETA_GRASPGENX_ROOT."
                )
            if "--checkpoint-root" not in config.command:
                raise ConfigError(
                    "GraspGenX checkpoint root is required: pass "
                    "--graspgenx-checkpoint-root or set "
                    "OPENETA_GRASPGENX_CHECKPOINT_ROOT."
                )
            if "--gripper-descriptions-root" not in config.command:
                raise ConfigError(
                    "GraspGenX gripper descriptions root is required: pass "
                    "--graspgenx-gripper-descriptions-root or set "
                    "OPENETA_GRASPGENX_GRIPPER_DESCRIPTIONS_ROOT."
                )


def _resolve_anygrasp_deployment_width(
    args: argparse.Namespace,
) -> tuple[float, Path]:
    profile_value = (
        args.grasp_calibration_profile
        or os.environ.get("OPENETA_GRASP_CALIBRATION_PROFILE")
        or DEFAULT_GRASP_CALIBRATION_PROFILE
    )
    profile = Path(profile_value).expanduser()
    if not profile.is_absolute():
        profile = REPO_ROOT / profile
    try:
        capabilities = load_grasp_calibration_capabilities(profile)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(f"invalid grasp calibration profile {profile}: {exc}") from exc
    expected = float(capabilities["max_gripper_width_m"])
    explicit = args.anygrasp_max_gripper_width
    if explicit is None:
        environment_value = os.environ.get("OPENETA_ANYGRASP_MAX_GRIPPER_WIDTH_M")
        if environment_value:
            try:
                explicit = float(environment_value)
            except ValueError as exc:
                raise ConfigError(
                    "OPENETA_ANYGRASP_MAX_GRIPPER_WIDTH_M must be a number"
                ) from exc
    if explicit is not None:
        explicit = float(explicit)
        if not math.isfinite(explicit) or not 0.0 < explicit <= 0.2:
            raise ConfigError("AnyGrasp deployment width must be in (0, 0.2] m")
        if abs(explicit - expected) > ANYGRASP_WIDTH_TOLERANCE_M:
            raise ConfigError(
                "AnyGrasp deployment width "
                f"{explicit:.6f} m does not match calibration profile {profile} "
                f"width {expected:.6f} m"
            )
    timeout_s = float(args.anygrasp_capability_timeout_s)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ConfigError("AnyGrasp capability timeout must be positive")
    gripper_height = float(args.anygrasp_gripper_height)
    if not math.isfinite(gripper_height) or gripper_height <= 0:
        raise ConfigError("AnyGrasp gripper height must be positive")
    return expected, profile


def _start_service(config: ServiceConfig, *, dry_run: bool) -> dict[str, Any]:
    pid = _read_pid(config.pid_file)
    if pid is not None and _pid_alive(pid):
        if not _pid_matches_command(pid, config):
            return {
                "ok": False,
                "service": config.name,
                "action": "start",
                "reason": "pid_mismatch",
                "pid": pid,
                "url": config.sse_url,
                "log": str(config.log_file),
            }
        if config.name == "anygrasp" and not dry_run:
            return _reconcile_running_anygrasp(config, pid=pid)
        return {
            "ok": False,
            "service": config.name,
            "action": "start",
            "reason": "already_running",
            "pid": pid,
            "url": config.sse_url,
            "log": str(config.log_file),
        }

    if dry_run:
        result = {
            "ok": True,
            "service": config.name,
            "action": "start",
            "dry_run": True,
            "command": _shell_join(config.command),
            "url": config.sse_url,
            "log": str(config.log_file),
        }
        if config.expected_max_gripper_width_m is not None:
            result.update(
                {
                    "expected_max_gripper_width_m": (
                        config.expected_max_gripper_width_m
                    ),
                    "grasp_calibration_profile": str(
                        config.grasp_calibration_profile or ""
                    ),
                    "reconcile": "check_capability_and_restart_if_incompatible",
                }
            )
        return result

    config.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with config.log_file.open("ab") as log:
            process = subprocess.Popen(
                config.command,
                cwd=REPO_ROOT,
                env=config.env,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
    except OSError as exc:
        return {
            "ok": False,
            "service": config.name,
            "action": "start",
            "reason": "launch_failed",
            "error": str(exc),
            "url": config.sse_url,
            "log": str(config.log_file),
        }
    config.pid_file.write_text(f"{process.pid}\n", encoding="utf-8")
    ready = _wait_for_service_ready(config, process)
    if ready is not None:
        return ready
    result = {
        "ok": True,
        "service": config.name,
        "action": "start",
        "pid": process.pid,
        "url": config.sse_url,
        "log": str(config.log_file),
    }
    if config.name == "anygrasp":
        compatibility = _anygrasp_compatibility(config)
        result["capability"] = compatibility
        if compatibility.get("compatible") is not True:
            result.update(
                {
                    "ok": False,
                    "reason": "post_start_capability_check_failed",
                }
            )
    return result


def _reconcile_running_anygrasp(
    config: ServiceConfig,
    *,
    pid: int,
) -> dict[str, Any]:
    compatibility = _anygrasp_compatibility(config)
    if compatibility.get("compatible") is True:
        return {
            "ok": True,
            "service": config.name,
            "action": "start",
            "reason": "already_running_compatible",
            "pid": pid,
            "url": config.sse_url,
            "log": str(config.log_file),
            "capability": compatibility,
        }

    restart_reason = str(compatibility.get("reason") or "capability_incompatible")
    stop_result = _stop_service(config, force=False, missing_ok=False)
    if stop_result.get("ok") is not True:
        return {
            "ok": False,
            "service": config.name,
            "action": "reconcile",
            "reason": "automatic_restart_stop_failed",
            "restart_reason": restart_reason,
            "previous_capability": compatibility,
            "stop": stop_result,
            "url": config.sse_url,
            "log": str(config.log_file),
        }
    start_result = _start_service(config, dry_run=False)
    return {
        "ok": start_result.get("ok") is True,
        "service": config.name,
        "action": "reconcile",
        "reason": "automatic_restart",
        "restart_reason": restart_reason,
        "previous_capability": compatibility,
        "stop": stop_result,
        "start": start_result,
        "url": config.sse_url,
        "log": str(config.log_file),
    }


def _anygrasp_compatibility(config: ServiceConfig) -> dict[str, Any]:
    expected = config.expected_max_gripper_width_m
    if expected is None:
        return {
            "configured": True,
            "available": False,
            "compatible": False,
            "reason": "missing_expected_gripper_width",
            "message": "AnyGrasp launcher has no host calibration width.",
        }
    clear_anygrasp_capability_cache()
    return check_anygrasp_compatibility(
        url=config.sse_url,
        physical_max_gripper_width_m=expected,
        timeout_s=config.capability_timeout_s,
    )


def _restart_dry_run(config: ServiceConfig) -> dict[str, Any]:
    return {
        "ok": True,
        "service": config.name,
        "action": "restart",
        "dry_run": True,
        "stop": "would_stop_matching_service_if_running",
        "command": _shell_join(config.command),
        "url": config.sse_url,
        "log": str(config.log_file),
    }


def _stop_service(
    config: ServiceConfig,
    *,
    force: bool,
    missing_ok: bool = False,
) -> dict[str, Any]:
    pid = _read_pid(config.pid_file)
    if pid is None:
        return {
            "ok": missing_ok,
            "service": config.name,
            "action": "stop",
            "reason": "pid_missing",
        }
    if not _pid_alive(pid):
        _unlink_if_exists(config.pid_file)
        return {
            "ok": True,
            "service": config.name,
            "action": "stop",
            "reason": "not_running",
            "pid": pid,
        }
    if not _pid_matches_command(pid, config):
        return {
            "ok": False,
            "service": config.name,
            "action": "stop",
            "reason": "pid_mismatch",
            "pid": pid,
        }

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        _unlink_if_exists(config.pid_file)
        return {
            "ok": True,
            "service": config.name,
            "action": "stop",
            "reason": "not_running",
            "pid": pid,
        }
    except PermissionError:
        return {
            "ok": False,
            "service": config.name,
            "action": "stop",
            "reason": "permission_denied",
            "pid": pid,
        }
    if force:
        _unlink_if_exists(config.pid_file)
        return {
            "ok": True,
            "service": config.name,
            "action": "stop",
            "pid": pid,
            "signal": "SIGKILL",
        }

    deadline = time.monotonic() + STOP_TIMEOUT_S
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            _unlink_if_exists(config.pid_file)
            return {
                "ok": True,
                "service": config.name,
                "action": "stop",
                "pid": pid,
                "signal": "SIGTERM",
            }
        time.sleep(0.1)
    return {
        "ok": False,
        "service": config.name,
        "action": "stop",
        "pid": pid,
        "signal": "SIGTERM",
        "reason": "still_running",
    }


def _status_service(config: ServiceConfig) -> dict[str, Any]:
    pid = _read_pid(config.pid_file)
    running = bool(pid is not None and _pid_alive(pid) and _pid_matches_command(pid, config))
    health_ok = _http_health_check(
        config.health_url,
        expected_server=config.expected_health_server,
    )
    return {
        "ok": running and health_ok,
        "service": config.name,
        "running": running,
        "pid": pid if running else None,
        "health": "ok" if health_ok else "failed",
        "url": config.sse_url,
        "health_url": config.health_url,
        "log": str(config.log_file),
    }


def _health_service(config: ServiceConfig) -> dict[str, Any]:
    health_ok = _http_health_check(
        config.health_url,
        expected_server=config.expected_health_server,
    )
    return {
        "ok": health_ok,
        "service": config.name,
        "health": "ok" if health_ok else "failed",
        "health_url": config.health_url,
    }


def _smoke_service(config: ServiceConfig) -> dict[str, Any]:
    try:
        tools = _mcp_list_tools(config.sse_url)
    except Exception as exc:  # noqa: BLE001 - command output should show failure reason.
        return {
            "ok": False,
            "service": config.name,
            "smoke": "failed",
            "url": config.sse_url,
            "reason": str(exc),
        }
    expected_tools = {
        "anygrasp": ("detect_grasps", "get_capabilities"),
        "anyplace": "predict_placement",
        "contact_graspnet": "predict_grasps",
        "molmopoint": "point_image",
        "graspgenx": "predict_grasps",
        "unidepth_v2": "estimate_depth",
    }
    expected = expected_tools.get(config.name)
    required_tools = (expected,) if isinstance(expected, str) else tuple(expected or ())
    missing_tools = [tool for tool in required_tools if tool not in tools]
    if missing_tools:
        return {
            "ok": False,
            "service": config.name,
            "smoke": "failed",
            "url": config.sse_url,
            "reason": "missing_tool:" + ",".join(missing_tools),
            "tools": tools,
        }
    return {
        "ok": True,
        "service": config.name,
        "smoke": "ok",
        "url": config.sse_url,
        "tools": tools,
    }


def _mcp_list_tools(url: str) -> list[str]:
    return asyncio.run(_mcp_list_tools_async(url))


async def _mcp_list_tools_async(url: str) -> list[str]:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            result = await session.list_tools()
            return [tool.name for tool in result.tools]


def _wait_for_service_ready(
    config: ServiceConfig,
    process: subprocess.Popen[bytes],
) -> dict[str, Any] | None:
    deadline = time.monotonic() + START_TIMEOUT_S
    while time.monotonic() < deadline:
        exit_code = process.poll()
        if exit_code is not None:
            _unlink_if_exists(config.pid_file)
            return {
                "ok": False,
                "service": config.name,
                "action": "start",
                "reason": "exited_early",
                "exit_code": exit_code,
                "pid": process.pid,
                "url": config.sse_url,
                "log": str(config.log_file),
            }
        if _http_health_check(
            config.health_url,
            expected_server=config.expected_health_server,
        ):
            return None
        time.sleep(0.1)
    return {
        "ok": False,
        "service": config.name,
        "action": "start",
        "reason": "health_timeout",
        "pid": process.pid,
        "url": config.sse_url,
        "log": str(config.log_file),
    }


def _http_health_check(url: str, expected_server: str | None = None) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            if not 200 <= response.status < 300:
                return False
            if expected_server is None:
                return True
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("ok") is True and payload.get("server") == expected_server
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return False


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not text:
        return None
    try:
        pid = int(text)
    except ValueError:
        return None
    if pid <= 0:
        return None
    return pid


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pid_matches_command(pid: int, config: ServiceConfig) -> bool:
    if pid <= 0:
        return False
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        return False
    cmdline = raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    return str(config.command[1]) in cmdline


def _resolve_state_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _print_results(results: dict[str, Any], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    for name, result in results.items():
        if "command" in result:
            print(f"{name}: dry-run")
            print(f"  command: {result['command']}")
            print(f"  url: {result['url']}")
            continue
        fields = [f"{name}:"]
        if "action" in result:
            fields.append(str(result["action"]))
        fields.append("ok" if result.get("ok") else "failed")
        print(" ".join(fields))
        for key in ("pid", "running", "health", "smoke", "url", "health_url", "log", "reason"):
            if key in result:
                print(f"  {key}: {result[key]}")
        if "tools" in result:
            print(f"  tools: {', '.join(result['tools'])}")


def _results_ok(results: dict[str, Any]) -> bool:
    for result in results.values():
        if isinstance(result, dict) and "ok" in result:
            if not result["ok"]:
                return False
            continue
        if isinstance(result, dict):
            nested = [value for value in result.values() if isinstance(value, dict)]
            if nested and not all(value.get("ok") for value in nested):
                return False
    return True


def _shell_join(parts: Iterable[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in parts)


class ConfigError(Exception):
    """CLI configuration cannot satisfy a requested service action."""


if __name__ == "__main__":
    raise SystemExit(main())
