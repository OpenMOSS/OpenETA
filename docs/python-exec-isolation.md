# Default Python execution isolation

Status: implemented on the refactor branch, Linux x86_64 fixture-tested;
not a complete resolution of all harness cancellation or resource-budget issues.
This changes the default `python_exec` implementation, not the main closed-loop
tool-calling architecture or the separate optional `code_policy` path.

## Boundary

The host starts a fresh interpreter with `-I -B`, a private working directory
(or the configured sandbox), minimal environment, closed inherited descriptors,
and JSON-only inputs. Before compiling generated code, the trusted bootstrap
drops Linux capabilities and installs `no_new_privs`, resource limits, Landlock,
and a seccomp syscall allowlist. No policy failure falls back to host execution.

Generated Python can introspect its own interpreter and bypass convenience
import/path wrappers. Tests deliberately do so: the kernel must still deny
out-of-scope file content access, external mutations, sockets, process/thread
creation, execution, and signaling other processes. Readable roots are the
interpreter/package libraries and explicitly configured session/workspace;
only the workspace is writable. File metadata/existence is not confidential:
this policy is not a mount namespace or a complete filesystem-discovery barrier.

Landlock ABI >= 3 is required to mediate file truncation; seccomp additionally
denies metadata-changing syscalls that these older ABIs do not mediate. The
policy uses a default-deny syscall list, with argument restrictions for `fcntl`
and `prlimit64`. File locking is denied even for readable files, so a worker
cannot acquire a lock that obstructs host access. See the [Linux Landlock documentation](https://docs.kernel.org/userspace-api/landlock.html)
and [libseccomp rule API](https://github.com/seccomp/libseccomp/blob/main/doc/man/man3/seccomp_rule_add.3).

The parent drains stdout, stderr, and a separate JSON result pipe without
redirecting the host's process-global streams. It terminates/reaps the process
on timeout, cancellation, output overflow, and errors. The registry recognizes
a host-only cooperative-cancellation opt-in for the default sandbox path, so
it does not abandon that handler in a detached daemon thread. Episode idle
waits include supervisor cleanup. Legacy noncooperative handlers are unchanged.

Worker results, including their JSON fields and descriptive policy metadata,
remain untrusted data; they do not authorize environment actions. The host
does not accept a worker-supplied artifact manifest. It discovers changed
regular files under the configured workspace, excludes symlinks/escaped paths,
and hashes files incrementally after worker exit. Workspace files remain mutable
derived artifacts, not immutable host evidence.

## Configuration and compatibility

- Supported bootstrap: non-root Linux x86_64/aarch64, Landlock ABI >= 3,
  `libseccomp.so.2`; x86_64 / Linux 6.8 / ABI 4 is tested locally. aarch64 and
  other deployment environments are not yet verified.
- Policy must be installed before worker threads exist. Numerical libraries
  start with BLAS/OpenMP thread counts set to one; generated thread creation is
  denied. Packages requiring subprocesses or worker threads are unsupported.
- `PythonExecConfig.session_root` grants current-session reads;
  `workspace_root` grants reads/writes there. CLI assembly already supplies
  session-specific roots. `/`, HOME, `/tmp`, and the repository root are
  rejected as direct configured grants. Without roots, only computation and
  interpreter/package-data reads are available, not ambient temporary files.
- `extra_globals` is JSON data only. Live Python callbacks, events, host APIs,
  and Python object identity do not cross the boundary. Host monkeypatches do
  not automatically change the child interpreter.
- Default worker wall budget: 120 seconds (`default_timeout_s`), including
  launch/bootstrap; requested `timeout_s` can shorten but not extend this host
  ceiling. CPU hard limit rounds up to integer seconds.
- Default address-space limit: 2 GiB; individual file limit: 64,000,000 bytes;
  each stdout/stderr stream: 1,000,000 bytes; JSON request/result: 8,000,000
  bytes; descriptor limit: 128; core dumps disabled. These are host settings,
  not planner-supplied permission grants.
- Structured failures include `python_sandbox_unavailable`,
  `python_sandbox_invalid_config`, `python_exec_invalid_timeout`,
  `python_exec_invalid_worker_input`, `python_exec_timeout`,
  `python_exec_cancelled`, `python_exec_output_limit`, and
  `python_exec_worker_failed`. Never suggest bypassing a failed policy without
  explicit user approval of the separate outside-sandbox path.

No public tool name or request field is renamed. Changes to deployment
requirements, timeout semantics, diagnostic/output fields, and host integration
must be reviewed before shared-contract merge; shared RFC approval is not implied
by local implementation or tests. For example, a cancelled worker produces a
diagnostic `{"code": "python_exec_cancelled", "error_type": "SandboxExecutionError"}`
inside the existing result envelope; the outer registry may instead expose its
standard `execution_cancelled` result after discarding the cancelled tool result.

## Remaining work and validation scope

- Approved `outside_sandbox` keeps its existing host-subprocess permission and
  approval boundary. Its cancellation acknowledgment and streaming output limits
  are not fixed here; it still uses `communicate()` before truncating output.
- This is not a cgroup or filesystem quota: per-file size limits do not cap
  total bytes/inodes across many files. Host artifact scanning/materialization
  is not yet charged against the worker wall budget. Aggregate disk quotas and
  bounded host postprocessing remain open.
- Other daemon handlers, remote transports, session commit ownership beyond
  local memory, environment cleanup, and post-episode model calls need their
  own lifecycle work. No claim is made that all late writes are eliminated.
- No real robot, simulator canary, model request, or package installation is
  required by these tests. No kernel sandbox should be interpreted as proof
  against kernel vulnerabilities or every possible denial-of-service channel.

Run `pytest -q tests/test_python_sandbox_isolation.py` for raw NumPy/file/socket/
fork/symlink tests, environment scrubbing, output isolation, timeout/cancellation,
reaping, resource limits, fail-closed bootstrap, and episode idle ownership.
The native boundary tests were also run outside the coding agent's outer
filesystem/network sandbox; passing only under that outer sandbox would not
establish OpenETA's own isolation. Exact counts and broader regressions are in
the [refactor log](harness-refactor-progress-2026-09-05.md).
