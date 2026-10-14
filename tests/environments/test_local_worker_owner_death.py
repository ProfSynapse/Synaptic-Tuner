"""Real owner SIGKILL; test-owned subreaper proves exact descendant exits."""
import subprocess
import sys
from pathlib import Path

import pytest
import shared.environments.local_worker as worker


@pytest.mark.parametrize('mode', ['stopped', 'partial_frame'])
def test_owner_death_reaps_worker_then_removes_original_root(tmp_path, mode):
    parent = tmp_path / 'private'
    parent.mkdir(mode=0o700)
    # Run the subreaper in an isolated test process, never alter pytest ancestry.
    program = r'''
import ctypes,json,os,signal,subprocess,sys,time
from pathlib import Path
module_root,parent,mode=sys.argv[1:]
sys.path.insert(0,module_root)
from shared.environments.local_worker import WorkerEnvironmentRuntime
from shared.environments.bounded_local_runtime import FilesystemLimits
from shared.environments.fixture_parser import EnvironmentFixture
if mode == 'owner':
    runtime=WorkerEnvironmentRuntime(parent,FilesystemLimits(30,4096,2048,4096,8,128,20))
    runtime.setup(EnvironmentFixture([],{'file':'owned'}))
    if os.environ.get('PARTIAL') == 'yes':
        os.write(runtime._process.stdin.fileno(), b'\x00\x00\x00\x64{')
    print(json.dumps([runtime._process.pid,runtime._worker_pid,runtime._storage._name]),flush=True)
    signal.pause()
else:
    assert ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)==0
    owner=subprocess.Popen([sys.executable,'-I',sys.argv[0],module_root,parent,'owner'],
        stdout=subprocess.PIPE,env={'PARTIAL':'yes' if mode=='partial_frame' else 'no'})
    descendants=[]
    try:
        import select
        assert select.select([owner.stdout],[],[],5)[0]
        guardian,child,name=json.loads(owner.stdout.readline())
        descendants=[guardian,child]
        if mode=='stopped':
            os.kill(child,signal.SIGSTOP)
            deadline=time.monotonic()+2
            while 'state:\tt' not in Path(f'/proc/{child}/status').read_text().lower():
                assert time.monotonic()<deadline
                time.sleep(.01)
        owner.kill(); assert owner.wait(timeout=3)==-signal.SIGKILL
        deadline=time.monotonic()+5
        remaining=set(descendants)
        while remaining:
            for pid in list(remaining):
                try:
                    result,status=os.waitpid(pid,os.WNOHANG)
                except ChildProcessError:
                    if pid==child and not Path(f'/proc/{pid}').exists():
                        remaining.remove(pid)
                    continue
                if result:
                    if pid==child:
                        assert os.WIFSIGNALED(status)
                    else:
                        assert os.WIFEXITED(status) and os.WEXITSTATUS(status)==0
                    remaining.remove(pid)
            assert time.monotonic()<deadline,remaining
            time.sleep(.01)
        assert not list(Path(parent).iterdir())
    finally:
        if owner.poll() is None: owner.kill()
        owner.wait(timeout=3)
        for pid in descendants:
            try: os.kill(pid,signal.SIGKILL)
            except ProcessLookupError: pass
'''
    script = tmp_path / 'harness.py'
    script.write_text(program)
    completed = subprocess.run(
        [sys.executable, '-I', str(script), str(Path(worker.__file__).resolve().parents[2]), str(parent), mode],
        capture_output=True, timeout=12,
    )
    assert completed.returncode == 0, completed.stderr.decode()


def test_guardian_unconfirmed_worker_exit_preserves_scratch(tmp_path, monkeypatch):
    import dataclasses
    import os
    import struct
    from shared.environments.bounded_local_runtime import BoundedLocalEnvironmentRuntime, FilesystemLimits
    from shared.environments.fixture_parser import EnvironmentFixture

    parent = tmp_path / 'private'
    parent.mkdir(mode=0o700)
    limits = FilesystemLimits(30,4096,2048,4096,8,128,20)
    storage = BoundedLocalEnvironmentRuntime(parent, limits)
    storage.setup(EnvironmentFixture([], {'file': 'owned'}))
    command_read, command_write = os.pipe()
    reply_read, reply_write = os.pipe()
    owner_read, owner_write = os.pipe()
    status_read, status_write = os.pipe()
    os.close(owner_write)
    calls = []

    class Unconfirmed:
        pid = 123
        def poll(self):
            return None
        def terminate(self):
            calls.append('term')
        def kill(self):
            calls.append('kill')
        def wait(self, *, timeout):
            calls.append(('wait', timeout))
            raise subprocess.TimeoutExpired('owned-child', timeout)

    initial = {'limits': dataclasses.asdict(limits), 'anchors': storage._anchors,
        'root_fd': storage._root_fd, 'name': storage._name, 'identity': storage._created_identity,
        'deadline': storage._deadline, 'command_read': command_read, 'reply_write': reply_write,
        'owner_read': owner_read, 'status_write': status_write}
    monkeypatch.setattr(worker, '_child_read', lambda maximum: initial)
    monkeypatch.setattr(worker.subprocess, 'Popen', lambda *args, **kwargs: Unconfirmed())
    try:
        worker._run_guardian()
        os.close(status_write)
        raw = os.read(status_read, 8192)
        first = struct.unpack('!I', raw[:4])[0] + 4
        status = worker._decode(raw[first + 4:])
        assert status == {'worker_exited': False, 'cleanup': False}
        assert calls == ['term', ('wait', .2), 'kill', ('wait', 2)]
        assert (parent / storage._name / 'file').read_text() == 'owned'
    finally:
        for fd in (command_write, reply_read, owner_read, status_read):
            os.close(fd)
        storage.teardown()


@pytest.mark.parametrize('failure', [1, 2, 3, 4])
def test_partial_pipe_allocation_closes_every_owned_descriptor(tmp_path, monkeypatch, failure):
    import os
    from shared.environments.bounded_local_runtime import FilesystemLimits
    from shared.environments.fixture_parser import EnvironmentFixture
    parent = tmp_path / 'private'
    parent.mkdir(mode=0o700)
    original = os.pipe
    calls = 0
    allocated = []
    def pipe():
        nonlocal calls
        calls += 1
        if calls == failure:
            raise OSError('injected pipe allocation failure')
        result = original()
        allocated.extend(result)
        return result
    monkeypatch.setattr(worker.os, 'pipe', pipe)
    runtime = worker.WorkerEnvironmentRuntime(parent, FilesystemLimits(30,4096,2048,4096,8,128,10))
    with pytest.raises(OSError):
        runtime.setup(EnvironmentFixture())
    assert not list(parent.iterdir()) and runtime._process is None
    for fd in allocated:
        with pytest.raises(OSError):
            os.fstat(fd)
