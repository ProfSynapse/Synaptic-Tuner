from dataclasses import replace
import os
import signal
import subprocess
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
    process.send_signal(signal.SIGSTOP)
    runtime._deadline = time.monotonic() + 0.1
    started = time.monotonic()
    with pytest.raises(WorkerLifecycleError, match='deadline'):
        runtime.read_text('file')
    assert time.monotonic() - started < 3
    assert process.returncode == -signal.SIGKILL
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
    def partial_reply():
        read_fd, write_fd = os.pipe()
        os.write(write_fd, b'\x00\x00')
        os.close(write_fd)
        process.stdout.close()
        process.stdout = os.fdopen(read_fd, 'rb', buffering=0)
        return original_reply()
    monkeypatch.setattr(runtime, '_reply', partial_reply)
    with pytest.raises(WorkerLifecycleError, match='protocol'):
        runtime.read_text('a/file')
    assert process.returncode is not None
    assert runtime._failed


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
    process.send_signal(signal.SIGSTOP)
    with pytest.raises(WorkerLifecycleError, match='boundary_interrupted'):
        runtime.write_text('large', 'x' * (1024 * 1024))
    assert process.returncode == -signal.SIGKILL
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


def test_exit_confirmation_failure_preserves_owned_scratch(runtime, parent, monkeypatch):
    process = runtime._process
    original_wait = process.wait
    def uncertain_wait(*args, **kwargs):
        raise subprocess.TimeoutExpired('fixed-child', kwargs.get('timeout', 0))
    monkeypatch.setattr(process, 'wait', uncertain_wait)
    with pytest.raises(WorkerLifecycleError, match='termination_uncertain'):
        runtime.teardown()
    assert runtime._storage._root_fd is not None and list(parent.iterdir())
    monkeypatch.setattr(process, 'wait', original_wait)
    original_wait(timeout=2)
    runtime.teardown()


def test_operation_whitelist_no_arbitrary_execution(runtime):
    with pytest.raises(ValueError, match='protocol'):
        runtime._request('exec', ['anything'])
    assert runtime.read_text('a/file') == 'hello'
