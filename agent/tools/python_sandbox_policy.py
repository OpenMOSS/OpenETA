"""Linux kernel policy for a disposable, single-threaded Python worker.

Landlock confines file contents and mutations; a seccomp allowlist excludes
networking, process creation, ptrace, and metadata-changing syscalls that older
Landlock ABIs do not mediate. Install before executing any generated code.
This module must never install restrictions in the host agent process.
"""

from __future__ import annotations

import ctypes
import errno
import fcntl
import os
from pathlib import Path
import platform
import resource
import sys
import sysconfig


class SandboxUnavailable(RuntimeError):
    """The required kernel boundary could not be established; never fall back."""


class _Ruleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathRule(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _ArgComparison(ctypes.Structure):
    _fields_ = [("arg", ctypes.c_uint), ("op", ctypes.c_uint),
                ("datum_a", ctypes.c_uint64), ("datum_b", ctypes.c_uint64)]


class _CapabilityHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int32)]


class _CapabilityData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32),
                ("inheritable", ctypes.c_uint32)]


def runtime_read_roots() -> list[str]:
    """Only interpreter/package data and system libraries, not cwd or HOME."""

    candidates = [
        sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib"),
        sysconfig.get_path("purelib"), sysconfig.get_path("platlib"),
        str(Path(sys.base_prefix) / "lib"),
        "/usr/lib", "/usr/lib64", "/lib", "/lib64", "/usr/share/fonts",
        "/usr/share/zoneinfo", "/etc/ld.so.cache", "/etc/localtime",
        "/dev/null", "/dev/urandom", "/dev/random",
    ]
    return sorted({str(Path(p).resolve()) for p in candidates if p and Path(p).exists()})


def install_policy(*, read_roots: list[str], write_root: str | None,
                   cpu_seconds: int, memory_bytes: int, file_bytes: int) -> dict:
    if sys.platform != "linux" or platform.machine() not in {"x86_64", "aarch64"}:
        raise SandboxUnavailable("Python sandbox requires Linux x86_64/aarch64 with Landlock >= 3")
    if os.geteuid() == 0:
        raise SandboxUnavailable("Python sandbox workers must run as an unprivileged user")
    # Load policy libraries before constraining dynamic loader reads.
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except OSError as exc:
        raise SandboxUnavailable("libseccomp.so.2 is required") from exc
    abi = libc.syscall(444, 0, 0, 1)  # landlock_create_ruleset VERSION
    if abi < 3:
        raise SandboxUnavailable(f"Landlock ABI >= 3 required; got {abi}, errno={ctypes.get_errno()}")
    if len(list(Path("/proc/self/task").iterdir())) != 1:
        raise SandboxUnavailable("Sandbox policy must be installed before any worker threads exist")
    # A non-root interpreter can still have file capabilities. Drop them before
    # fixing limits so generated code cannot raise its own hard resource caps.
    cap_header = _CapabilityHeader(0x20080522, 0)  # Linux capabilities ABI v3
    cap_data = (_CapabilityData * 2)()
    if libc.capset(ctypes.byref(cap_header), ctypes.byref(cap_data)):
        raise SandboxUnavailable("Could not drop worker capabilities")
    if libc.prctl(38, 1, 0, 0, 0):  # PR_SET_NO_NEW_PRIVS
        raise SandboxUnavailable("PR_SET_NO_NEW_PRIVS failed")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))

    # ABI 3 adds TRUNCATE; deny execute/devices/sockets/FIFOs even in workspace.
    handled = (1 << 15) - 1
    ruleset = _Ruleset(handled)
    ruleset_fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if ruleset_fd < 0:
        raise SandboxUnavailable(f"Landlock ruleset creation failed: errno={ctypes.get_errno()}")
    read = (1 << 2) | (1 << 3)
    write = read | (1 << 1) | (1 << 4) | (1 << 5) | (1 << 7) | (1 << 8) | (1 << 12) | (1 << 13) | (1 << 14)
    try:
        grants = [(root, read) for root in read_roots]
        if write_root:
            grants.append((write_root, write))
        for root, rights in grants:
            path = Path(root).resolve(strict=True)
            if not path.is_dir():
                rights &= (1 << 2) | (1 << 1) | (1 << 14)
            fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            try:
                rule = _PathRule(rights, fd)
                if libc.syscall(445, ruleset_fd, 1, ctypes.byref(rule), 0):
                    raise SandboxUnavailable(f"Landlock rule failed for {path}: errno={ctypes.get_errno()}")
            finally:
                os.close(fd)
        if libc.syscall(446, ruleset_fd, 0):
            raise SandboxUnavailable(f"Landlock enforcement failed: errno={ctypes.get_errno()}")
    finally:
        os.close(ruleset_fd)

    _install_seccomp(seccomp)
    return {"implementation": "linux_landlock_seccomp", "landlock_abi": abi,
            "network": "denied", "child_processes": "denied", "new_threads": "denied"}


def _install_seccomp(lib) -> None:
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                          ctypes.c_uint, ctypes.c_void_p]
    lib.seccomp_rule_add_array.restype = ctypes.c_int
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_load.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    # Unknown syscalls are denied, rather than accepted by an incomplete denylist.
    policy = lib.seccomp_init(0x00050000 | errno.EPERM)
    if not policy:
        raise SandboxUnavailable("seccomp_init failed")
    allowed = """
        read write readv writev pread64 pwrite64 preadv pwritev close close_range
        dup dup2 dup3 lseek open openat stat lstat fstat newfstatat statx
        statfs fstatfs access faccessat faccessat2 readlink readlinkat getdents getdents64
        mkdir mkdirat rmdir unlink unlinkat rename renameat renameat2
        link linkat symlink symlinkat truncate ftruncate fsync fdatasync
        mmap mprotect munmap mremap madvise brk
        rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack
        futex futex_time64 set_tid_address set_robust_list rseq
        clock_gettime clock_gettime64 clock_getres clock_nanosleep nanosleep gettimeofday time
        getpid getppid gettid getuid geteuid getgid getegid getgroups
        uname getcwd chdir fchdir getrandom getrusage sysinfo sched_yield sched_getaffinity
        getrlimit arch_prctl restart_syscall
        poll ppoll select pselect6 epoll_create1 epoll_ctl epoll_wait epoll_pwait
        pipe pipe2 exit exit_group
    """.split()
    try:
        for name in allowed:
            syscall = lib.seccomp_syscall_resolve_name(name.encode())
            if syscall == -1:
                continue  # Not present on this architecture.
            if lib.seccomp_rule_add_array(policy, 0x7FFF0000, syscall, 0, None):
                raise SandboxUnavailable(f"seccomp allow rule failed: {name}")
        # libc implements getrlimit via prlimit64. Never permit changing another
        # process's limits, taking locks on readonly host files, or selecting a
        # signal recipient through F_SETOWN.
        restricted = [("prlimit64", 0, 0)]
        restricted += [("fcntl", 1, command) for command in
                       (fcntl.F_DUPFD, fcntl.F_DUPFD_CLOEXEC, fcntl.F_GETFD, fcntl.F_SETFD,
                        fcntl.F_GETFL, fcntl.F_SETFL)]
        for name, arg, expected in restricted:
            comparison = _ArgComparison(arg, 4, expected, 0)  # SCMP_CMP_EQ
            syscall = lib.seccomp_syscall_resolve_name(name.encode())
            if syscall != -1 and lib.seccomp_rule_add_array(
                policy, 0x7FFF0000, syscall, 1, ctypes.byref(comparison)
            ):
                raise SandboxUnavailable(f"seccomp argument rule failed: {name}")
        if lib.seccomp_load(policy):
            raise SandboxUnavailable("seccomp_load failed")
    finally:
        lib.seccomp_release(policy)
