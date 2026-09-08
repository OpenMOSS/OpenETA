from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ctypes
import errno
import os
from pathlib import Path
import threading
import time

import pytest

from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


# Deliberately bypass the convenience import allowlist. These tests establish
# the kernel boundary, not merely that a Python wrapper refused an operation.
RAW = "raw_import = json.__builtins__['__import__']\nos = raw_import('os')\nctypes = raw_import('ctypes')\n"


def run(runtime, code, *, parameters=None, cancel=None):
    return runtime.handler(ToolExecutionContext(
        name="python_exec", spec=build_default_tool_registry().get("python_exec"),
        parameters={"code": code, **(parameters or {})},
        metadata={"_cancel_event": cancel} if cancel is not None else {},
    ))


def configured(tmp_path, **options):
    session = tmp_path / "session"
    sandbox = session / "sandbox"
    sandbox.mkdir(parents=True)
    return PythonExecRuntime(PythonExecConfig(
        session_root=str(session), workspace_root=str(sandbox),
        image_output_root=str(session / "artifacts/images"),
        text_output_root=str(session / "artifacts/text"),
        response_output_root=str(session / "artifacts/responses"),
        structured_output_root=str(session / "artifacts"), **options,
    ))


def test_native_numpy_io_obeys_readonly_session_and_workspace_only_write(tmp_path):
    runtime = configured(tmp_path)
    session = Path(runtime.config.session_root)
    outside = tmp_path / "outside.txt"
    outside.write_text("123\n")
    owned = session / "input.txt"
    owned.write_text("7\n")
    result = run(runtime, "import numpy as np\nnp.savetxt('derived.txt', [np.loadtxt(parameters['path']) + 1])\nresult = float(np.loadtxt('derived.txt'))", parameters={"path": str(owned)})
    assert result.success, result.content
    assert result.details["outputs"]["result"] == 8
    assert result.details["outputs"]["policy"]["landlock_abi"] >= 3
    for path in (owned, outside):
        denied = run(runtime, "import numpy as np\nnp.savetxt(parameters['path'], [99])", parameters={"path": str(path)})
        assert not denied.success
        assert denied.details["diagnostics"][0]["error_type"] == "PermissionError"
    denied = run(runtime, "import numpy as np\nresult = float(np.loadtxt(parameters['path']))", parameters={"path": str(outside)})
    assert not denied.success
    assert owned.read_text() == "7\n"
    assert outside.read_text() == "123\n"


def test_raw_open_truncate_and_metadata_mutation_cannot_escape(tmp_path):
    runtime = configured(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve")
    mode = outside.stat().st_mode
    code = RAW + """
answers = []
for operation in ['open', 'truncate', 'chmod', 'unlink']:
    try:
        if operation == 'open':
            os.close(os.open(parameters['path'], os.O_WRONLY))
        elif operation == 'truncate':
            os.truncate(parameters['path'], 0)
        elif operation == 'chmod':
            os.chmod(parameters['path'], 0o600)
        else:
            os.unlink(parameters['path'])
        answers.append('ALLOWED')
    except Exception as error:
        answers.append(error.errno)
result = answers
"""
    result = run(runtime, code, parameters={"path": str(outside)})
    assert result.success, result.content
    assert all(value in (errno.EPERM, errno.EACCES) for value in result.details["outputs"]["result"])
    assert outside.read_text() == "preserve"
    assert outside.stat().st_mode == mode


def test_symlink_does_not_escape_or_become_a_host_materialized_artifact(tmp_path):
    runtime = configured(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.write_text("private-fixture")
    result = run(runtime, RAW + "os.symlink(parameters['path'], 'link.txt')\nresult = 'linked'", parameters={"path": str(outside)})
    assert result.success, result.content
    assert not any(a.get("path", "").endswith("link.txt") for a in result.details["artifacts"])
    denied = run(runtime, RAW + "fd = os.open('link.txt', os.O_RDONLY)\nresult = os.read(fd, 100)")
    assert not denied.success
    assert outside.read_text() == "private-fixture"


def test_default_worker_has_no_ambient_tmp_or_repository_read_access(tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_text("private-fixture")
    result = run(PythonExecRuntime(), RAW + "fd = os.open(parameters['path'], os.O_RDONLY)\nresult = os.read(fd, 100)", parameters={"path": str(outside)})
    assert not result.success
    assert result.details["diagnostics"][0]["error_type"] == "PermissionError"


def test_native_network_process_creation_and_foreign_process_access_are_denied(tmp_path):
    runtime = configured(tmp_path)
    # Signals use signal 0 and prlimit only queries: a defective policy must
    # fail this test without harming the parent test runner.
    code = RAW + """
libc = ctypes.CDLL(None, use_errno=True)
results = {}
for family in [1, 2, 10]:
    fd = libc.socket(family, 1, 0)
    results['socket_' + str(family)] = [fd, ctypes.get_errno()]
    if fd >= 0:
        os.close(fd)
results['signal'] = [libc.kill(parameters['parent'], 0), ctypes.get_errno()]
results['foreign_limit'] = [libc.prlimit64(parameters['parent'], 7, 0, 0), ctypes.get_errno()]
pid = libc.fork()
if pid == 0:
    os._exit(0)
results['fork'] = [pid, ctypes.get_errno()]
result = results
"""
    result = run(runtime, code, parameters={"parent": os.getpid()})
    assert result.success, result.content
    for name, (value, error) in result.details["outputs"]["result"].items():
        assert (value, error) == (-1, errno.EPERM), name


def test_worker_does_not_inherit_host_credentials_or_python_startup_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENETA_TEST_PRIVATE_TOKEN", "private-fixture")
    monkeypatch.setenv("HTTPS_PROXY", "http://private-fixture.invalid")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "untrusted-startup"))
    result = run(configured(tmp_path), RAW + "result = [os.environ.get(k) for k in ['OPENETA_TEST_PRIVATE_TOKEN', 'HTTPS_PROXY', 'PYTHONPATH']]")
    assert result.success, result.content
    assert result.details["outputs"]["result"] == [None, None, None]


def test_readonly_session_files_cannot_be_locked_to_block_the_host(tmp_path):
    runtime = configured(tmp_path)
    owned = Path(runtime.config.session_root) / "host.lock"
    owned.touch()
    result = run(runtime, RAW + """
fcntl = raw_import('fcntl')
fd = os.open(parameters['path'], os.O_RDONLY)
answers = []
for operation in ['flock', 'lockf']:
    try:
        if operation == 'flock':
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            fcntl.lockf(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        answers.append('ALLOWED')
    except Exception as error:
        answers.append(error.errno)
os.close(fd)
result = answers
""", parameters={"path": str(owned)})
    assert result.success, result.content
    assert result.details["outputs"]["result"] == [errno.EPERM, errno.EPERM]


def test_concurrent_workers_keep_print_and_native_stdout_separate(tmp_path):
    barrier = threading.Barrier(2)

    def execute(label):
        runtime = configured(tmp_path / label)
        barrier.wait(timeout=3)
        code = RAW + """
for i in range(40):
    print(parameters['label'])
    os.write(1, (parameters['label'] + '-native\\n').encode())
result = parameters['label']
"""
        return run(runtime, code, parameters={"label": label})

    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(execute, ["session-A", "session-B"]))
    assert a.success and b.success
    for result, own, other in ((a, "session-A", "session-B"), (b, "session-B", "session-A")):
        stdout = result.details["outputs"]["stdout"]
        assert stdout.count(own) == 80
        assert other not in stdout
    assert a.details["outputs"]["worker_pid"] != b.details["outputs"]["worker_pid"]


def assert_reaped(result):
    pid = result.details["outputs"]["worker_pid"]
    assert pid
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_wall_timeout_terminates_and_reaps_infinite_loop(tmp_path):
    runtime = configured(tmp_path, default_timeout_s=0.8)
    started = time.monotonic()
    result = run(runtime, "while True:\n    pass")
    assert not result.success
    assert result.details["diagnostics"][0]["code"] == "python_exec_timeout"
    assert time.monotonic() - started < 3
    assert_reaped(result)


def test_cancel_terminates_running_worker_before_handler_returns(tmp_path):
    runtime = configured(tmp_path, default_timeout_s=10)
    marker = Path(runtime.config.workspace_root) / "started.txt"
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(run, runtime, "Path('started.txt').write_text('ready')\nwhile True:\n    pass", cancel=cancel)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            if pending.done():
                pytest.fail(pending.result().content)
            cancel.wait(0.01)
        try:
            assert marker.exists()
        finally:
            cancel.set()
        result = pending.result(timeout=3)
    assert not result.success
    assert result.details["diagnostics"][0]["code"] == "python_exec_cancelled"
    assert_reaped(result)


def test_stdout_and_structured_result_limits_are_enforced(tmp_path):
    runtime = configured(tmp_path, max_sandbox_output_bytes=1024, max_sandbox_result_bytes=4096)
    output = run(runtime, "print('x' * 100000)")
    assert not output.success
    assert output.details["diagnostics"][0]["code"] == "python_exec_output_limit"
    assert_reaped(output)
    output = run(runtime, "result = 'x' * 100000")
    assert not output.success
    assert output.details["diagnostics"][0]["code"] == "python_exec_output_limit"
    assert_reaped(output)


def test_memory_and_file_limits_do_not_rely_on_python_wrappers(tmp_path):
    runtime = configured(tmp_path, max_sandbox_memory_bytes=256_000_000, max_sandbox_file_bytes=8192)
    result = run(runtime, "result = 'x' * 512000000")
    assert not result.success
    assert result.details["diagnostics"][0]["error_type"] == "MemoryError"
    result = run(runtime, RAW + "fd = os.open('large.bin', os.O_CREAT | os.O_WRONLY, 0o600)\nos.write(fd, b'x' * 100000)\nos.write(fd, b'x' * 100000)")
    assert not result.success
    assert (Path(runtime.config.workspace_root) / "large.bin").stat().st_size <= 8192


def test_non_json_host_objects_are_not_shared_with_worker(tmp_path):
    runtime = configured(tmp_path, extra_globals={"event": threading.Event()})
    result = run(runtime, "result = 1")
    assert not result.success
    assert result.details["diagnostics"][0]["code"] == "python_exec_invalid_worker_input"


def test_missing_kernel_capability_fails_closed_without_applying_policy_to_host(monkeypatch):
    from agent.tools import python_sandbox_policy as policy
    real_cdll = ctypes.CDLL

    def load(name, **kwargs):
        if name is None:
            class OldKernel:
                def syscall(self, *args):
                    return 2
            return OldKernel()
        return real_cdll(name, **kwargs)

    monkeypatch.setattr(policy.ctypes, "CDLL", load)
    with pytest.raises(policy.SandboxUnavailable, match="ABI >= 3"):
        policy.install_policy(read_roots=[], write_root=None, cpu_seconds=1,
                              memory_bytes=256000000, file_bytes=8192)


def test_episode_idle_waits_for_cooperative_python_handler_cleanup(tmp_path, monkeypatch):
    from agent.backends.planner import StaticPlannerBackend
    from agent.runtime.episode import DummyEpisodeEnvironment, EpisodeTimeoutError, OpenEtaEpisodeRunner
    from agent.runtime.planner import ToolCallingPlanner
    from agent.runtime.runtime import OpenEtaAgentRuntime
    from agent.tools import coding

    python = configured(tmp_path, default_timeout_s=30)
    tools = build_default_tool_registry()
    tools.bind_handler("python_exec", python.handler)
    runtime = OpenEtaAgentRuntime(planner=ToolCallingPlanner(StaticPlannerBackend({
        "kind": "tool_call", "name": "python_exec", "parameters": {"code": "while True: pass"},
    })), tools=tools)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    real_execute = coding.execute_sandbox
    reaped = threading.Event()
    release = threading.Event()
    executions = []

    def execute(*args, **kwargs):
        execution = real_execute(*args, **kwargs)
        executions.append(execution)
        reaped.set()
        assert release.wait(timeout=5)
        return execution

    monkeypatch.setattr(coding, "execute_sandbox", execute)
    runner.start(task="compute", max_turns=1, timeout_s=0.8)
    try:
        with pytest.raises(EpisodeTimeoutError):
            runner.step()
        assert reaped.wait(timeout=3)
        assert executions[0]["diagnostic"]["code"] == "python_exec_cancelled"
        with pytest.raises(ProcessLookupError):
            os.kill(executions[0]["worker_pid"], 0)
        assert runner.wait_for_idle(timeout_s=0.1) is False
    finally:
        release.set()
        assert runner.wait_for_idle(timeout_s=3)


def test_only_default_sandbox_path_opts_into_cooperative_cancellation(tmp_path):
    runtime = configured(tmp_path)
    predicate = runtime.handler._cooperative_cancellation_when
    spec = build_default_tool_registry().get("python_exec")
    assert predicate(ToolExecutionContext(name="python_exec", spec=spec))
    assert not predicate(ToolExecutionContext(name="python_exec", spec=spec,
                                              parameters={"sandbox": "outside_sandbox"}))


@pytest.mark.parametrize("requested,expected", [(0.1, 0.1), (600, 30)])
def test_request_timeout_can_shorten_but_not_extend_host_worker_budget(tmp_path, monkeypatch,
                                                                       requested, expected):
    from agent.tools import coding
    seen = []

    def execute(payload, **kwargs):
        seen.append(payload["timeout_s"])
        return {"success": True, "result": 1}

    monkeypatch.setattr(coding, "execute_sandbox", execute)
    assert run(configured(tmp_path, default_timeout_s=30), "result = 1",
               parameters={"timeout_s": requested}).success
    assert seen == [expected]


@pytest.mark.parametrize("timeout", [True, 0, -1, float("nan"), float("inf"), "1"])
def test_invalid_request_timeout_is_rejected_before_worker_launch(tmp_path, timeout):
    result = run(configured(tmp_path), "result = 1", parameters={"timeout_s": timeout})
    assert not result.success
    assert result.details["diagnostics"][0]["code"] == "python_exec_invalid_timeout"
