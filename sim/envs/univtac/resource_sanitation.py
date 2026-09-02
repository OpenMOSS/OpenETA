"""Resource inventory and narrowly scoped cleanup for UniVTAC diagnostics."""

from __future__ import annotations

import json
import os
import pwd
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RESOURCE_SCHEMA_VERSION = "openeta.univtac.resource_inventory.v1"
CUDA_ENV_ALLOWLIST = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "SLURM_LOCALID",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "PYTORCH_CUDA_ALLOC_CONF",
    "CUDA_MODULE_LOADING",
)
RELATED_TERMS = (
    "isaacsim",
    "isaac sim",
    "omni.kit",
    "univtac",
    "tacex",
    "uipc",
    "probe_ftp1_eval_seeds",
    "diagnose_reset",
    "smoke_insert_hole",
    "bench_worker.py",
)
RUNTIME_ENTRYPOINTS = (
    "probe_ftp1_eval_seeds.py",
    "diagnose_reset.py",
    "smoke_insert_hole.py",
    "rerun_clean_reset_gates.py",
    "probe_uipc_device.py",
    "sim/bench_worker.py",
    "isaacsim.simulation_app",
    "kit_",
)
EXCLUDED_TERMS = (
    "codex",
    "agentsdock",
    "code-server",
    "/code ",
    "/code-",
    "visual studio code",
    "jupyter",
    "sshd",
    " rosetta-mcp",
)
ALLOWED_TEMPORARY_SYSCTLS = {"fs.inotify.max_user_instances"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def read_sysctl(name: str) -> int | None:
    path = Path("/proc/sys") / Path(*name.split("."))
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def safe_device_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str | None]:
    environment = os.environ if environment is None else environment
    return {name: environment.get(name) for name in CUDA_ENV_ALLOWLIST}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_link(path: Path) -> str | None:
    try:
        return os.readlink(path)
    except OSError:
        return None


def _proc_stat(pid: int) -> dict[str, Any] | None:
    text = _read_text(Path("/proc") / str(pid) / "stat")
    if not text or ") " not in text:
        return None
    rest = text.rsplit(") ", 1)[1].split()
    if len(rest) < 20:
        return None
    try:
        return {
            "state": rest[0],
            "ppid": int(rest[1]),
            "process_group_id": int(rest[2]),
            "session_id": int(rest[3]),
            "tty_nr": int(rest[4]),
            "start_ticks": int(rest[19]),
        }
    except ValueError:
        return None


def _cmdline(pid: int) -> str:
    try:
        data = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return ""
    return data.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def ancestor_pids(pid: int | None = None) -> list[int]:
    current = os.getpid() if pid is None else int(pid)
    ancestors: list[int] = []
    seen: set[int] = set()
    while current > 1 and current not in seen:
        seen.add(current)
        ancestors.append(current)
        stat = _proc_stat(current)
        if stat is None:
            break
        current = int(stat["ppid"])
    if current == 1:
        ancestors.append(1)
    return ancestors


def _children_by_parent(pids: Iterable[int]) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for pid in pids:
        stat = _proc_stat(pid)
        if stat is not None:
            result.setdefault(int(stat["ppid"]), []).append(pid)
    return {key: sorted(value) for key, value in result.items()}


def _parent_chain(pid: int, stats: Mapping[int, Mapping[str, Any]]) -> list[int]:
    chain: list[int] = []
    seen: set[int] = set()
    parent = int(stats.get(pid, {}).get("ppid", 0))
    while parent > 0 and parent not in seen:
        chain.append(parent)
        seen.add(parent)
        parent = int(stats.get(parent, {}).get("ppid", 0))
    return chain


def _boot_time_seconds() -> float | None:
    try:
        uptime = float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return time.time() - uptime


def _start_time(start_ticks: int, boot_time: float | None) -> str | None:
    if boot_time is None:
        return None
    ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    return datetime.fromtimestamp(
        boot_time + start_ticks / ticks, timezone.utc
    ).isoformat()


def _inotify_for_pid(pid: int) -> tuple[int, int | None]:
    fd_root = Path("/proc") / str(pid) / "fd"
    instance_count = 0
    watch_count = 0
    reliable = True
    try:
        entries = list(fd_root.iterdir())
    except OSError:
        return 0, None
    for fd in entries:
        target = _read_link(fd)
        if target != "anon_inode:inotify":
            continue
        instance_count += 1
        info = _read_text(Path("/proc") / str(pid) / "fdinfo" / fd.name)
        if info is None:
            reliable = False
            continue
        watch_count += sum(1 for line in info.splitlines() if line.startswith("inotify wd:"))
    return instance_count, watch_count if reliable else None


def collect_inotify_inventory(uid: int | None = None) -> dict[str, Any]:
    uid = os.getuid() if uid is None else int(uid)
    owners: list[dict[str, Any]] = []
    instances = 0
    watches = 0
    watches_reliable = True
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            if process_dir.stat().st_uid != uid:
                continue
        except OSError:
            continue
        pid = int(process_dir.name)
        process_instances, process_watches = _inotify_for_pid(pid)
        if not process_instances:
            continue
        instances += process_instances
        if process_watches is None:
            watches_reliable = False
        else:
            watches += process_watches
        owners.append(
            {
                "pid": pid,
                "instances": process_instances,
                "watches": process_watches,
                "cmdline": _cmdline(pid),
            }
        )
    maximum = read_sysctl("fs.inotify.max_user_instances")
    maximum_watches = read_sysctl("fs.inotify.max_user_watches")
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "captured_at": utc_now(),
        "uid": uid,
        "max_user_instances": maximum,
        "max_user_watches": maximum_watches,
        "max_queued_events": read_sysctl("fs.inotify.max_queued_events"),
        "instance_count": instances,
        "instance_usage_ratio": (
            instances / maximum if maximum and maximum > 0 else None
        ),
        "watch_count": watches if watches_reliable else None,
        "watch_count_reliable": watches_reliable,
        "watch_usage_ratio": (
            watches / maximum_watches
            if watches_reliable and maximum_watches and maximum_watches > 0
            else None
        ),
        "owners": sorted(owners, key=lambda item: item["pid"]),
    }


def build_restore_ready(
    *,
    original_instances: int,
    original_watches: int,
    inotify_inventory: Mapping[str, Any],
    process_inventory: Mapping[str, Any],
    gpu_inventory: Mapping[str, Any],
    process_group_residual: Sequence[int] = (),
) -> dict[str, Any]:
    eligible = sorted(
        int(item["pid"])
        for item in process_inventory.get("processes", [])
        if item.get("eligible_for_cleanup") is True
    )
    runtime_pids = {
        int(item["pid"])
        for item in process_inventory.get("processes", [])
        if item.get("matched_reason", {}).get("runtime_entrypoints")
        and not item.get("cleanup_exclusion_reason")
    }
    gpu_pids = sorted(
        int(item["pid"])
        for item in gpu_inventory.get("compute_processes", [])
        if item.get("pid") is not None and int(item["pid"]) in runtime_pids
    )
    instance_count = inotify_inventory.get("instance_count")
    watch_count = inotify_inventory.get("watch_count")
    watch_reliable = inotify_inventory.get("watch_count_reliable") is True
    group_residual = sorted({int(pid) for pid in process_group_residual})
    reasons: list[str] = []
    if group_residual:
        reasons.append("simulator_process_group_remaining")
    if eligible:
        reasons.append("eligible_project_process_remaining")
    if gpu_pids:
        reasons.append("project_gpu_process_remaining")
    if instance_count is None or int(instance_count) >= int(original_instances):
        reasons.append("instance_count_not_below_original_limit")
    if not watch_reliable:
        reasons.append("watch_count_unreliable")
    elif watch_count is None or int(watch_count) >= int(original_watches):
        reasons.append("watch_count_not_below_original_limit")
    return {
        "schema_version": "openeta.univtac.restore_ready.v1",
        "original_instances": int(original_instances),
        "original_watches": int(original_watches),
        "current_instances_limit": inotify_inventory.get("max_user_instances"),
        "current_watches_limit": inotify_inventory.get("max_user_watches"),
        "current_instance_count": instance_count,
        "current_watch_count": watch_count,
        "watch_count_reliable": watch_reliable,
        "project_processes_remaining": eligible,
        "project_gpu_pids_remaining": gpu_pids,
        "process_group_members_remaining": group_residual,
        "safe_for_manual_restore": not reasons,
        "not_ready_reasons": reasons,
    }


def collect_gpu_inventory() -> dict[str, Any]:
    gpu_query = [
        "nvidia-smi",
        "--query-gpu=index,uuid,pci.bus_id,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    process_query = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]

    def run(command: list[str]) -> tuple[int, list[str], str | None]:
        try:
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True, timeout=15
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return 1, [], f"{type(exc).__name__}: {exc}"
        return (
            completed.returncode,
            [line.strip() for line in completed.stdout.splitlines() if line.strip()],
            completed.stderr.strip() or None,
        )

    gpu_code, gpu_lines, gpu_error = run(gpu_query)
    process_code, process_lines, process_error = run(process_query)
    gpus: list[dict[str, Any]] = []
    for line in gpu_lines:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 7:
            continue
        gpus.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "pci_bus_id": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": int(fields[4]),
                "memory_used_mib": int(fields[5]),
                "memory_free_mib": int(fields[6]),
            }
        )
    processes: list[dict[str, Any]] = []
    for line in process_lines:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3 or not fields[0].isdigit():
            continue
        pid = int(fields[0])
        process_dir = Path("/proc") / str(pid)
        try:
            uid = process_dir.stat().st_uid
            username = pwd.getpwuid(uid).pw_name
        except (OSError, KeyError):
            uid = None
            username = None
        processes.append(
            {
                "pid": pid,
                "gpu_uuid": fields[1],
                "gpu_memory_mib": int(fields[2]) if fields[2].isdigit() else None,
                "uid": uid,
                "username": username,
                "cmdline": _cmdline(pid),
            }
        )
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "captured_at": utc_now(),
        "gpu_query_returncode": gpu_code,
        "gpu_query_error": gpu_error,
        "process_query_returncode": process_code,
        "process_query_error": process_error,
        "gpus": gpus,
        "compute_processes": sorted(processes, key=lambda item: item["pid"]),
    }


def collect_process_inventory(
    *, related_roots: Sequence[Path], uid: int | None = None
) -> dict[str, Any]:
    uid = os.getuid() if uid is None else int(uid)
    current_pid = os.getpid()
    protected_ancestors = set(ancestor_pids(current_pid))
    all_pids: list[int] = []
    stats: dict[int, dict[str, Any]] = {}
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            if process_dir.stat().st_uid != uid:
                continue
        except OSError:
            continue
        pid = int(process_dir.name)
        stat = _proc_stat(pid)
        if stat is not None:
            all_pids.append(pid)
            stats[pid] = stat
    children = _children_by_parent(all_pids)
    roots = [str(path.expanduser().resolve()).lower() for path in related_roots]
    boot_time = _boot_time_seconds()
    records: list[dict[str, Any]] = []
    for pid in sorted(all_pids):
        process_dir = Path("/proc") / str(pid)
        stat = stats[pid]
        cmdline = _cmdline(pid)
        exe = _read_link(process_dir / "exe")
        cwd = _read_link(process_dir / "cwd")
        haystack = " ".join(item or "" for item in (cmdline, exe, cwd)).lower()
        matched = [term for term in RELATED_TERMS if term in haystack]
        scoped_roots = [root for root in roots if root in haystack]
        exclusion_terms = [term for term in EXCLUDED_TERMS if term in haystack]
        runtime_entries = [term for term in RUNTIME_ENTRYPOINTS if term in haystack]
        is_ancestor = pid in protected_ancestors
        open_tty = bool(stat["tty_nr"])
        reasons: list[str] = []
        eligible = False
        if is_ancestor:
            reasons.append("current_process_or_ancestor")
        if exclusion_terms:
            reasons.append("excluded_interactive_or_agent_process")
        if open_tty:
            reasons.append("has_active_tty")
        if not scoped_roots:
            reasons.append("no_project_root_evidence")
        if not runtime_entries:
            reasons.append("no_specific_runtime_entrypoint")
        if not reasons and matched:
            eligible = True
        related = bool(matched or scoped_roots or runtime_entries)
        if not related:
            continue
        instances, watches = _inotify_for_pid(pid)
        records.append(
            {
                "pid": pid,
                "ppid": int(stat["ppid"]),
                "process_group_id": int(stat["process_group_id"]),
                "session_id": int(stat["session_id"]),
                "uid": uid,
                "username": pwd.getpwuid(uid).pw_name,
                "start_ticks": int(stat["start_ticks"]),
                "start_time": _start_time(int(stat["start_ticks"]), boot_time),
                "elapsed_seconds": (
                    max(
                        0.0,
                        time.time()
                        - (
                            boot_time
                            + int(stat["start_ticks"])
                            / os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                        ),
                    )
                    if boot_time is not None
                    else None
                ),
                "state": stat["state"],
                "cmdline": cmdline,
                "exe": exe,
                "cwd": cwd,
                "open_tty": open_tty,
                "parent_chain": _parent_chain(pid, stats),
                "child_pids": children.get(pid, []),
                "is_ancestor_of_current_process": is_ancestor,
                "matched_reason": {
                    "keywords": matched,
                    "project_roots": scoped_roots,
                    "runtime_entrypoints": runtime_entries,
                },
                "inotify_instances": instances,
                "inotify_watches": watches,
                "eligible_for_cleanup": eligible,
                "cleanup_exclusion_reason": reasons,
            }
        )
    return {
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "captured_at": utc_now(),
        "uid": uid,
        "current_pid": current_pid,
        "protected_ancestor_pids": sorted(protected_ancestors),
        "related_roots": [str(path.expanduser().resolve()) for path in related_roots],
        "processes": records,
    }


def attach_gpu_usage(
    process_inventory: dict[str, Any], gpu_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    by_pid = {
        int(item["pid"]): item
        for item in gpu_inventory.get("compute_processes", [])
        if item.get("pid") is not None
    }
    for record in process_inventory.get("processes", []):
        usage = by_pid.get(int(record["pid"]))
        record["gpu_memory_mib"] = usage.get("gpu_memory_mib") if usage else None
        record["gpu_uuid"] = usage.get("gpu_uuid") if usage else None
    return process_inventory


def cleanup_action_for_record(
    record: Mapping[str, Any], *, current_uid: int | None = None
) -> str:
    current_uid = os.getuid() if current_uid is None else int(current_uid)
    if int(record.get("uid", -1)) != current_uid:
        return "keep"
    if record.get("is_ancestor_of_current_process") or record.get("open_tty"):
        return "keep"
    if record.get("cleanup_exclusion_reason"):
        return "keep"
    if record.get("eligible_for_cleanup") is True:
        return "terminate"
    return "uncertain"


def build_cleanup_plan(process_inventory: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for record in process_inventory.get("processes", []):
        action = cleanup_action_for_record(record)
        entries.append(
            {
                "pid": int(record["pid"]),
                "start_ticks": int(record["start_ticks"]),
                "action": action,
                "reason": (
                    "strict_cleanup_eligibility_satisfied"
                    if action == "terminate"
                    else record.get("cleanup_exclusion_reason", [])
                ),
                "identity": {
                    "uid": record.get("uid"),
                    "cmdline": record.get("cmdline"),
                    "cwd": record.get("cwd"),
                    "exe": record.get("exe"),
                    "process_group_id": record.get("process_group_id"),
                    "gpu_memory_mib": record.get("gpu_memory_mib"),
                },
            }
        )
    return {
        "schema_version": "openeta.univtac.cleanup_plan.v1",
        "created_at": utc_now(),
        "default_mode": "dry_run",
        "entries": entries,
        "counts": {
            action: sum(1 for entry in entries if entry["action"] == action)
            for action in ("terminate", "keep", "uncertain")
        },
    }


def _identity_matches(entry: Mapping[str, Any], current_uid: int) -> tuple[bool, str]:
    pid = int(entry["pid"])
    process_dir = Path("/proc") / str(pid)
    try:
        uid = process_dir.stat().st_uid
    except OSError:
        return False, "process_absent"
    if uid != current_uid:
        return False, "uid_changed"
    stat = _proc_stat(pid)
    if stat is None or int(stat["start_ticks"]) != int(entry["start_ticks"]):
        return False, "start_time_changed"
    identity = entry.get("identity", {})
    if _cmdline(pid) != identity.get("cmdline"):
        return False, "cmdline_changed"
    if _read_link(process_dir / "cwd") != identity.get("cwd"):
        return False, "cwd_changed"
    if _read_link(process_dir / "exe") != identity.get("exe"):
        return False, "exe_changed"
    return True, "matched"


def apply_cleanup_plan(
    plan: Mapping[str, Any], *, term_wait_seconds: float = 15.0
) -> list[dict[str, Any]]:
    current_uid = os.getuid()
    protected = set(ancestor_pids())
    selected = [entry for entry in plan.get("entries", []) if entry.get("action") == "terminate"]
    actions: list[dict[str, Any]] = []
    alive: list[tuple[Mapping[str, Any], int | None]] = []
    for entry in selected:
        pid = int(entry["pid"])
        if pid in protected:
            actions.append(
                {"at": utc_now(), "pid": pid, "signal": None, "result": "protected"}
            )
            continue
        matched, reason = _identity_matches(entry, current_uid)
        if not matched:
            actions.append(
                {"at": utc_now(), "pid": pid, "signal": None, "result": reason}
            )
            continue
        process_group_id = int(entry.get("identity", {}).get("process_group_id") or 0)
        group_members = process_group_members(process_group_id) if process_group_id else []
        group_is_scoped = bool(group_members)
        for member_pid in group_members:
            stat = _proc_stat(member_pid)
            try:
                member_uid = (Path("/proc") / str(member_pid)).stat().st_uid
            except OSError:
                group_is_scoped = False
                break
            if stat is None or member_uid != current_uid:
                group_is_scoped = False
                break
            if member_pid in protected:
                group_is_scoped = False
                break
            parent = member_pid
            seen: set[int] = set()
            while parent > 1 and parent not in seen and parent != pid:
                seen.add(parent)
                parent_stat = _proc_stat(parent)
                parent = int(parent_stat["ppid"]) if parent_stat else 0
            if member_pid != pid and parent != pid:
                group_is_scoped = False
                break
        signal_target = process_group_id if group_is_scoped else None
        try:
            if signal_target is not None:
                os.killpg(signal_target, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
            result = "sent"
            alive.append((entry, signal_target))
        except ProcessLookupError:
            result = "already_exited"
        except PermissionError:
            result = "permission_denied"
        actions.append(
            {
                "at": utc_now(),
                "pid": pid,
                "process_group_id": signal_target,
                "group_members": group_members if group_is_scoped else [],
                "signal": "SIGTERM",
                "result": result,
            }
        )
    deadline = time.monotonic() + max(0.0, term_wait_seconds)
    while alive and time.monotonic() < deadline:
        alive = [
            (entry, group_id)
            for entry, group_id in alive
            if (
                process_group_members(group_id)
                if group_id is not None
                else Path("/proc", str(entry["pid"])).exists()
            )
        ]
        if alive:
            time.sleep(0.1)
    for entry, group_id in alive:
        pid = int(entry["pid"])
        matched, reason = _identity_matches(entry, current_uid)
        if group_id is None and not matched:
            actions.append(
                {"at": utc_now(), "pid": pid, "signal": None, "result": reason}
            )
            continue
        try:
            if group_id is not None:
                os.killpg(group_id, signal.SIGKILL)
            else:
                os.kill(pid, signal.SIGKILL)
            result = "sent"
        except ProcessLookupError:
            result = "already_exited"
        except PermissionError:
            result = "permission_denied"
        actions.append(
            {
                "at": utc_now(),
                "pid": pid,
                "process_group_id": group_id,
                "signal": "SIGKILL",
                "result": result,
            }
        )
    return actions


def process_group_members(process_group_id: int, uid: int | None = None) -> list[int]:
    uid = os.getuid() if uid is None else int(uid)
    members: list[int] = []
    for process_dir in Path("/proc").iterdir():
        if not process_dir.name.isdigit():
            continue
        try:
            if process_dir.stat().st_uid != uid:
                continue
        except OSError:
            continue
        stat = _proc_stat(int(process_dir.name))
        if stat and int(stat["process_group_id"]) == int(process_group_id):
            members.append(int(process_dir.name))
    return sorted(members)


def run_managed_process(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
    cleanup_wait_seconds: float = 15.0,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    before_gpu = collect_gpu_inventory()
    before_inotify = collect_inotify_inventory()
    started_at = utc_now()
    started = time.monotonic()
    timed_out = False
    term_sent = False
    kill_sent = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            env=dict(environment) if environment is not None else None,
            start_new_session=True,
        )
        child_pid = process.pid
        process_group_id = os.getpgid(child_pid)
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            term_sent = True
            try:
                os.killpg(process_group_id, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                log.write(f"PROCESS GROUP SIGTERM DENIED: {exc}\n")
            try:
                returncode = process.wait(timeout=cleanup_wait_seconds)
            except subprocess.TimeoutExpired:
                kill_sent = True
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    log.write(f"PROCESS GROUP SIGKILL DENIED: {exc}\n")
                returncode = process.wait(timeout=cleanup_wait_seconds)
        lingering = process_group_members(process_group_id)
        if lingering:
            if not term_sent:
                term_sent = True
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    log.write(f"LINGERING PROCESS GROUP SIGTERM DENIED: {exc}\n")
                deadline = time.monotonic() + cleanup_wait_seconds
                while process_group_members(process_group_id) and time.monotonic() < deadline:
                    time.sleep(0.1)
            lingering = process_group_members(process_group_id)
            if lingering:
                kill_sent = True
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except PermissionError as exc:
                    log.write(f"LINGERING PROCESS GROUP SIGKILL DENIED: {exc}\n")
        final_members = process_group_members(process_group_id)
    after_gpu = collect_gpu_inventory()
    after_inotify = collect_inotify_inventory()
    before_gpu_pids = {item["pid"] for item in before_gpu["compute_processes"]}
    after_gpu_pids = {item["pid"] for item in after_gpu["compute_processes"]}
    return {
        "command": list(command),
        "cwd": str(cwd.resolve()),
        "started_at": started_at,
        "ended_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "child_root_pid": child_pid,
        "process_group_id": process_group_id,
        "returncode": returncode,
        "signal": (
            signal.Signals(-returncode).name
            if returncode < 0 and -returncode in signal.valid_signals()
            else None
        ),
        "timed_out": timed_out,
        "sigterm_sent": term_sent,
        "sigkill_sent": kill_sent,
        "final_process_group_members": final_members,
        "gpu_pids_before": sorted(before_gpu_pids),
        "gpu_pids_after": sorted(after_gpu_pids),
        "new_gpu_pids_after": sorted(after_gpu_pids - before_gpu_pids),
        "inotify_instances_before": before_inotify["instance_count"],
        "inotify_instances_after": after_inotify["instance_count"],
        "inotify_instance_delta": (
            after_inotify["instance_count"] - before_inotify["instance_count"]
        ),
        "cleanup_complete": not final_members and not (after_gpu_pids - before_gpu_pids),
        "log_path": str(log_path.resolve()),
    }


def decide_inotify_change(inventory: Mapping[str, Any]) -> dict[str, Any]:
    ratio = inventory.get("instance_usage_ratio")
    instances_required = ratio is not None and float(ratio) >= 0.75
    return {
        "instances_change_required": instances_required,
        "watches_change_required": False,
        "reason": (
            "instance_usage_at_or_above_75_percent"
            if instances_required
            else "instance_usage_below_75_percent"
        ),
    }


def run_noninteractive_sysctl(name: str, value: int) -> dict[str, Any]:
    if name not in ALLOWED_TEMPORARY_SYSCTLS:
        raise ValueError(f"sysctl is not approved for temporary change: {name}")
    command = ["sudo", "-n", "sysctl", "-w", f"{name}={int(value)}"]
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": command,
            "returncode": None,
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "success": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
