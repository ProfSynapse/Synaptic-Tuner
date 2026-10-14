from dataclasses import replace
import os
import signal
import select
import subprocess
import sys
import threading
import time

import pytest

from shared.environments.bounded_local_runtime import FilesystemLimits
from shared.environments.fixture_parser import EnvironmentFixture
from shared.environments.local_worker import WorkerEnvironmentRuntime, WorkerLifecycleError


@pytest.fixture
def limits():
    return FilesystemLimits(30, 4096, 2048, 4096, 8, 128, 10)


@pytest.fixture
def parent(tmp_path):
    path = tmp_path / 'private'
    path.mkdir(mode=0o700)
    return path


@pytest.fixture
def runtime(parent, limits):
    current = WorkerEnvironmentRuntime(parent, limits)
    current.setup(EnvironmentFixture(['a'], {'a/file': 'hello'}))
    yield current
    current.teardown()


def test_real_worker_interface(runtime, parent):
    assert runtime._process.pid != os.getpid()
    runtime.mkdir('folder')
    runtime.write_text('folder/x', 'updated')
    assert runtime.read_text('folder/x') == 'updated'
    assert runtime.list_dir('folder') == ['x']
    assert runtime.search('updated') == ['folder/x']
    runtime.copy('folder/x', 'copied')
    runtime.move('copied', 'moved')
    assert runtime.exists('moved')
    runtime.delete('folder', recursive=True)
    assert runtime.snapshot()['runtime'] == 'bounded_local'
    assert str(parent) not in str(runtime.snapshot())
    process = runtime._process
    runtime.teardown()
    assert process.returncode is not None and process.poll() is not None
    assert list(parent.iterdir()) == []
    with pytest.raises(RuntimeError, match='consumed'):
        runtime.setup(EnvironmentFixture())


def test_stopped_child_deadline_actual_exit_before_cleanup(parent, limits):
    runtime = WorkerEnvironmentRuntime(parent, limits)
    runtime.setup(EnvironmentFixture([], {'file': 'retained'}))
    process = runtime._process
    os.kill(runtime._worker_pid, signal.SIGSTOP)
    runtime._deadline = time.monotonic() + 0.1
    started = time.monotonic()
    with pytest.raises(WorkerLifecycleError, match='deadline'):
        runtime.read_text('file')
    assert time.monotonic() - started < 3
    assert process.returncode == 0
    assert list(parent.iterdir()) == []
    with pytest.raises(RuntimeError, match='unavailable'):
        runtime.read_text('file')


def test_boundary_cancellation_owning_thread_terminates(runtime, parent):
    owner = threading.get_ident()
    calls = []
    def cancel():
        calls.append(threading.get_ident())
        raise RuntimeError('host_cancelled')
    runtime._boundary = cancel
    process = runtime._process
    with pytest.raises(WorkerLifecycleError, match='boundary_interrupted'):
        runtime.write_text('after-cancel', 'no')
    assert calls == [owner]
    assert process.returncode is not None
    assert list(parent.iterdir()) == []


def test_prevalidation_no_effects(parent, limits):
    runtime = WorkerEnvironmentRuntime(parent, limits)
    with pytest.raises(ValueError):
        runtime.setup(EnvironmentFixture([], {'../escape': 'x'}))
    assert list(parent.iterdir()) == [] and runtime._process is None


def test_lost_reply_never_resends_or_respawns(runtime, parent, monkeypatch):
    process = runtime._process
    def lost():
        raise TimeoutError('lost_reply')
    monkeypatch.setattr(runtime, '_reply', lost)
    with pytest.raises(WorkerLifecycleError, match='protocol'):
        runtime.write_text('maybe-written', 'x')
    assert process.returncode is not None and list(parent.iterdir()) == []
    with pytest.raises(RuntimeError, match='unavailable'):
        runtime.write_text('maybe-written', 'x')
    assert runtime._process is process


def test_unknown_partial_spawn_retains_custody(parent, limits, monkeypatch):
    def unknown(*args, **kwargs):
        raise OSError('unknown_spawn')
    monkeypatch.setattr(subprocess, 'Popen', unknown)
    runtime = WorkerEnvironmentRuntime(parent, limits)
    with pytest.raises(RuntimeError, match='start_uncertain'):
        runtime.setup(EnvironmentFixture())
    assert runtime._storage._root_fd is not None
    assert list(parent.iterdir())
    with pytest.raises(RuntimeError, match='consumed'):
        runtime.setup(EnvironmentFixture())
    with pytest.raises(RuntimeError, match='start_uncertain'):
        runtime.teardown()
    # Test knows no child actually started; explicit host reconciliation only.
    runtime._spawn_uncertain = False
    runtime.teardown()


def test_root_substitution_child_exit_then_cleanup_uncertain(parent, limits):
    runtime = WorkerEnvironmentRuntime(parent, limits)
    runtime.setup(EnvironmentFixture([], {'file': 'owned'}))
    root = parent / runtime._storage._name
    retained = parent / 'retained'
    root.rename(retained)
    root.mkdir(mode=0o700)
    (root / 'foreign').write_text('preserved')
    with pytest.raises(WorkerLifecycleError, match='cleanup_uncertain'):
        runtime.teardown()
    assert runtime._process.returncode is not None
    assert (root / 'foreign').read_text() == 'preserved'
    assert (retained / 'file').read_text() == 'owned'
    (root / 'foreign').unlink()
    root.rmdir()
    retained.rename(root)
    runtime.teardown()


def test_parent_environment_never_forwarded(parent, limits, monkeypatch):
    monkeypatch.setenv('HF_TOKEN', 'not-a-real-secret')
    monkeypatch.setenv('MODAL_TOKEN_SECRET', 'not-a-real-secret')
    original = subprocess.Popen
    observed = {}
    def spawn(*args, **kwargs):
        observed.update(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    runtime = WorkerEnvironmentRuntime(parent, limits)
    runtime.setup(EnvironmentFixture())
    assert observed['env'] == {'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'}
    assert observed['shell'] is False and observed['close_fds'] is True
    runtime.teardown()


def test_original_expired_deadline_no_scratch(parent, limits):
    runtime = WorkerEnvironmentRuntime(parent, limits, deadline=time.monotonic() - 1)
    with pytest.raises(WorkerLifecycleError, match='deadline'):
        runtime.setup(EnvironmentFixture())
    assert list(parent.iterdir()) == []


def test_read_output_bound_definite_rejection(runtime):
    runtime.write_text('large', 'x' * 2049)
    with pytest.raises(ValueError, match='operation_rejected'):
        runtime.read_text('large')
    assert runtime.read_text('a/file') == 'hello'


def test_default_snapshot_small_bound_and_closed_error(runtime):
    assert runtime.snapshot()['truncated'] is False
    with pytest.raises(FileNotFoundError, match='operation_rejected') as error:
        runtime.read_text('missing')
    assert str(error.value) == 'worker_operation_rejected'


def test_exact_owned_fd_closed_after_exit(parent, limits):
    runtime = WorkerEnvironmentRuntime(parent, limits)
    runtime.setup(EnvironmentFixture())
    fd = runtime._storage._root_fd
    runtime.teardown()
    with pytest.raises(OSError):
        os.fstat(fd)


def test_escaped_unicode_control_text_wire_roundtrip(parent, limits):
    text = ('"\\\n\x00é😀' * 1500)
    size = len(text.encode('utf-8'))
    runtime = WorkerEnvironmentRuntime(parent, replace(limits, max_content_bytes=size, max_output_bytes=size))
    runtime.setup(EnvironmentFixture([], {'escaped': text}))
    assert runtime.read_text('escaped') == text
    runtime.teardown()


@pytest.mark.parametrize('operation,value', [('exists', 'true'), ('read_text', []), ('list_dir', [1]),
                                           ('mkdir', True), ('snapshot', {'root': '/private'})])
def test_malformed_reply_cannot_masquerade_as_success(runtime, operation, value, monkeypatch):
    args = {'exists': ['a/file'], 'read_text': ['a/file'], 'list_dir': ['a'],
            'mkdir': ['new'], 'snapshot': [200]}[operation]
    monkeypatch.setattr(runtime, '_reply', lambda: {'id': runtime._request_id, 'ok': True, 'value': value})
    with pytest.raises(WorkerLifecycleError, match='protocol'):
        runtime._request(operation, args)
    assert runtime._process.returncode is not None


def test_partial_reply_eof_is_uncertain_no_replay(runtime, monkeypatch):
    process = runtime._process
    original_reply = runtime._reply
    scanners = []
    # Keep the fixed child from resetting its inherited directory position
    # before the parent observes our deliberately truncated response.
    os.kill(runtime._worker_pid, signal.SIGSTOP)
    def partial_reply():
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b'\x00\x00')
        os.close(write_fd)
        process.stdout.close()
        process.stdout = os.fdopen(read_fd, 'rb', buffering=0)
        scanner = os.scandir(runtime._storage._root_fd)
        scanners.append(scanner)
        next(scanner)
        return original_reply()
    monkeypatch.setattr(runtime, '_reply', partial_reply)
    try:
        with pytest.raises(WorkerLifecycleError, match='protocol'):
            runtime.read_text('a/file')
        assert process.returncode == 0
        assert runtime._failed
        assert runtime._storage._root_fd is None
        with pytest.raises(WorkerLifecycleError, match='unavailable'):
            runtime.read_text('a/file')
    finally:
        for scanner in scanners:
            scanner.close()


def test_request_backpressure_cancellation_terminates(parent, limits):
    checks = 0
    def boundary():
        nonlocal checks
        checks += 1
        if checks > 15:
            raise RuntimeError('host_cancelled')
    runtime = WorkerEnvironmentRuntime(parent, replace(limits, max_content_bytes=1024 * 1024))
    runtime.setup(EnvironmentFixture())
    runtime._boundary = boundary
    process = runtime._process
    os.kill(runtime._worker_pid, signal.SIGSTOP)
    with pytest.raises(WorkerLifecycleError, match='boundary_interrupted'):
        runtime.write_text('large', 'x' * (1024 * 1024))
    assert process.returncode == 0
    assert list(parent.iterdir()) == []


def test_fixed_entry_ignores_cwd_and_pythonpath(parent, limits, tmp_path, monkeypatch):
    hostile = tmp_path / 'hostile'
    hostile.mkdir()
    (hostile / 'shared.py').write_text("raise AssertionError('ambient import')")
    monkeypatch.chdir(hostile)
    monkeypatch.setenv('PYTHONPATH', str(hostile))
    runtime = WorkerEnvironmentRuntime(parent, limits)
    runtime.setup(EnvironmentFixture([], {'file': 'trusted'}))
    assert runtime.read_text('file') == 'trusted'
    runtime.teardown()


def test_guardian_loss_preserves_owned_scratch(parent, limits):
    import shared.environments.local_worker as module
    from pathlib import Path
    program = r"""
import ctypes,os,signal,sys
sys.path.insert(0,sys.argv[1])
from shared.environments.local_worker import WorkerEnvironmentRuntime,WorkerLifecycleError
from shared.environments.bounded_local_runtime import FilesystemLimits
from shared.environments.fixture_parser import EnvironmentFixture
assert ctypes.CDLL(None).prctl(36,1,0,0,0)==0
runtime=WorkerEnvironmentRuntime(sys.argv[2],FilesystemLimits(30,4096,2048,4096,8,128,10))
runtime.setup(EnvironmentFixture([],{'file':'retained'}))
process=runtime._process
worker_pid=runtime._worker_pid
process.kill();process.wait(timeout=2)
try:
    try:
        runtime.teardown()
    except WorkerLifecycleError as error:
        assert error.code=='worker_termination_uncertain'
    else:
        raise AssertionError('guardian loss accepted')
    assert runtime._storage._root_fd is not None
    assert runtime._storage.read_text('file')=='retained'
finally:
    os.kill(worker_pid,signal.SIGKILL)
    assert os.waitpid(worker_pid,0)[0]==worker_pid
    # Independent test wait proved its owned child exit; production cannot adopt it.
    runtime._storage.teardown()
"""
    result = subprocess.run([sys.executable, '-I', '-c', program,
        str(Path(module.__file__).resolve().parents[2]), str(parent)],
        capture_output=True, timeout=8)
    assert result.returncode == 0, result.stderr.decode()
    assert list(parent.iterdir()) == []


def test_operation_whitelist_no_arbitrary_execution(runtime):
    with pytest.raises(ValueError, match='protocol'):
        runtime._request('exec', ['anything'])
    assert runtime.read_text('a/file') == 'hello'


def test_inherited_directory_cursor_does_not_hide_entries_or_cleanup(runtime, parent):
    fd = runtime._storage._root_fd
    with os.scandir(fd) as iterator:
        next(iterator)
        end = os.lseek(fd, 0, os.SEEK_CUR)
    os.lseek(fd, end, os.SEEK_SET)
    # Root enumeration must not consume or reset the inherited description.
    assert runtime._storage.list_dir() == ['a']
    assert os.lseek(fd, 0, os.SEEK_CUR) == end
    assert runtime._storage.exists('a/file')
    runtime.teardown()
    assert runtime._process.returncode is not None
    assert runtime._storage._root_fd is None
    assert list(parent.iterdir()) == []


def test_sigkill_during_inherited_scandir_cannot_hide_cleanup(runtime, parent):
    read_fd, write_fd = os.pipe()
    child = None
    try:
        child = subprocess.Popen(
            [sys.executable, '-I', '-B', '-c',
             "import os,signal,sys; scan=os.scandir(int(sys.argv[1])); "
             "next(scan); os.write(int(sys.argv[2]), b'x'); signal.pause()",
             str(runtime._storage._root_fd), str(write_fd)],
            pass_fds=(runtime._storage._root_fd, write_fd),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env={'PATH': os.environ.get('PATH', '')},
        )
        os.close(write_fd)
        write_fd = None
        assert select.select([read_fd], [], [], 10)[0], 'owned scanner did not become ready'
        assert os.read(read_fd, 1) == b'x'
        child.kill()
        assert child.wait(timeout=5) == -signal.SIGKILL
        # Cleanup is the first enumeration after abrupt iterator death.
        runtime.teardown()
        assert runtime._process.returncode is not None
        assert runtime._storage._root_fd is None
        assert list(parent.iterdir()) == []
    finally:
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=5)
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
