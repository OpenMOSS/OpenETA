"""CUDA device probes and gate decisions for clean UniVTAC diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

from .resource_sanitation import safe_device_environment, utc_now


CUDA_DIAGNOSTIC_SCHEMA_VERSION = "openeta.univtac.cuda_device_diagnostic.v1"
FIXED_FTP1_SEEDS = (1000000, 1000001, 1000002)


def _nvidia_identity(index: int = 0) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        f"--id={int(index)}",
        "--query-gpu=index,uuid,pci.bus_id,name",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    fields = [part.strip() for part in completed.stdout.strip().split(",")]
    if completed.returncode != 0 or len(fields) < 4:
        return {
            "available": False,
            "returncode": completed.returncode,
            "error": completed.stderr.strip(),
        }
    return {
        "available": True,
        "index": int(fields[0]),
        "uuid": fields[1],
        "pci_bus_id": fields[2],
        "name": ",".join(fields[3:]).strip(),
    }


def collect_cuda_diagnostic() -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "schema_version": CUDA_DIAGNOSTIC_SCHEMA_VERSION,
        "captured_at": utc_now(),
        "environment": safe_device_environment(),
        "torch": {
            "version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
        },
        "nvidia_smi_logical_zero": _nvidia_identity(0),
        "tests": {},
        "warp": {"available": False},
        "uipc": {"available": False},
        "tacex_uipc": {"available": False},
        "success": False,
        "error": None,
    }
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        payload["error"] = "cuda_unavailable"
        return payload
    try:
        current = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(0)
        payload["torch"].update(
            {
                "current_device": current,
                "device_name": torch.cuda.get_device_name(0),
                "compute_capability": [properties.major, properties.minor],
            }
        )
        tensor = torch.tensor([3.0, 1.0, 2.0], device="cuda:0")
        payload["tests"]["allocation"] = tensor.device.type == "cuda"
        payload["tests"]["sort"] = torch.sort(tensor).values.cpu().tolist() == [
            1.0,
            2.0,
            3.0,
        ]
        left = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda:0")
        right = torch.eye(2, device="cuda:0")
        payload["tests"]["matrix_multiply"] = torch.equal(left @ right, left)
        torch.cuda.synchronize()
        payload["tests"]["synchronize"] = True
        try:
            import warp as wp

            wp_device = wp.get_device("cuda:0")
            payload["warp"] = {
                "available": True,
                "device": str(wp_device),
                "ordinal": getattr(wp_device, "ordinal", None),
            }
        except Exception as exc:
            payload["warp"] = {
                "available": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        for module_name in ("uipc", "tacex_uipc"):
            try:
                module = __import__(module_name)
                payload[module_name] = {
                    "available": True,
                    "module_file": str(Path(module.__file__).resolve()),
                    "version": str(getattr(module, "__version__", "unknown")),
                }
            except Exception as exc:
                payload[module_name] = {
                    "available": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        payload["success"] = all(payload["tests"].values())
    except Exception as exc:
        payload["error"] = {"type": type(exc).__name__, "message": str(exc)}
    return payload


def compare_cuda_modes(d0: Mapping[str, Any], d1: Mapping[str, Any]) -> dict[str, Any]:
    d0_gpu = d0.get("nvidia_smi_logical_zero", {})
    d1_gpu = d1.get("nvidia_smi_logical_zero", {})
    same_uuid = bool(
        d0_gpu.get("uuid")
        and d0_gpu.get("uuid") == d1_gpu.get("uuid")
    )
    same_pci = bool(
        d0_gpu.get("pci_bus_id")
        and d0_gpu.get("pci_bus_id") == d1_gpu.get("pci_bus_id")
    )
    ordinal_mismatch = not same_uuid or not same_pci
    return {
        "d0_success": d0.get("success") is True,
        "d1_success": d1.get("success") is True,
        "same_physical_gpu_uuid": same_uuid,
        "same_pci_bus_id": same_pci,
        "classification": (
            "cuda_device_ordinal_mismatch"
            if ordinal_mismatch
            else "same_physical_device"
        ),
    }


def validate_fixed_seed(seed: int) -> int:
    seed = int(seed)
    if seed not in FIXED_FTP1_SEEDS:
        raise ValueError(f"seed must be one of {list(FIXED_FTP1_SEEDS)}")
    return seed


def gate_followups(seed_zero_record: Mapping[str, Any]) -> list[int]:
    validate_fixed_seed(int(seed_zero_record["seed"]))
    if int(seed_zero_record["seed"]) != FIXED_FTP1_SEEDS[0]:
        raise ValueError("gate must begin with seed 1000000")
    reached_planner = int(seed_zero_record.get("planning_call_count", 0)) > 0
    reset_returned = seed_zero_record.get("reset_returned") is True
    return list(FIXED_FTP1_SEEDS[1:]) if reached_planner or reset_returned else []


def should_continue_after_gate(
    record: Mapping[str, Any], *, require_reset_valid: bool
) -> bool:
    if record.get("failure_stage") in {"startup_before_reset", "reset_runtime_error"}:
        return False
    if require_reset_valid:
        return record.get("reset_valid") is True
    return bool(
        record.get("reset_returned") is True
        or int(record.get("planning_call_count", 0)) > 0
    )


def classify_uipc_runs(runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(runs)
    def passed(record: Mapping[str, Any]) -> bool:
        return bool(
            record.get("returncode") == 0
            and record.get("completed_step") is True
            and record.get("invalid_device") is False
            and record.get("cleanup_complete") is True
        )

    passing = [record for record in records if passed(record)]
    by_mode: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_mode.setdefault(str(record.get("mode")), []).append(record)
    stable_modes = sorted(
        mode
        for mode, values in by_mode.items()
        if any(passed(first) and passed(second) for first, second in zip(values, values[1:]))
    )
    all_invalid = bool(records) and all(record.get("invalid_device") for record in records)
    all_timed_out_cleanly = bool(records) and all(
        record.get("timed_out") is True
        and record.get("invalid_device") is False
        and record.get("cleanup_complete") is True
        and not record.get("final_process_group_members")
        and not record.get("new_gpu_pids_after")
        for record in records
    )
    all_clean_no_step = bool(records) and all(
        record.get("returncode") == 0
        and record.get("completed_step") is False
        and record.get("invalid_device") is False
        and record.get("cleanup_complete") is True
        and (
            record.get("native_result_missing") is True
            or record.get("native_result", {}).get("error") is None
        )
        and record.get("stage")
        in {"official_main_started", "official_main_returned", "pre_close_result_written"}
        for record in records
    )
    return {
        "runs": records,
        "passing_run_count": len(passing),
        "stable_modes": stable_modes,
        "consecutive_two_pass": bool(stable_modes),
        "classification": (
            "persistent_uipc_invalid_device_clean_state"
            if all_invalid
            else "uipc_sentinel_timeout_clean_state"
            if all_timed_out_cleanly
            else "uipc_no_step_clean_state"
            if all_clean_no_step
            else "uipc_sentinel_passed"
            if stable_modes
            else "uipc_sentinel_not_stable"
        ),
    }
